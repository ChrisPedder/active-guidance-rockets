#!/usr/bin/env python3
"""
PPO Training Script for Rocket Spin Control

Features:
- Configurable mass/thrust parameters
- Observation normalization
- Configurable reward function
- Curriculum learning support
- Fine-tuning from pre-trained models
- Logging and diagnostics

Usage:
    # Using config file
    uv run python train_improved.py --config configs/estes_c6.yaml

    # Quick test
    uv run python train_improved.py --config configs/debug.yaml

    # Override specific parameters
    uv run python train_improved.py --config configs/estes_c6.yaml --dry-mass 0.12 --timesteps 100000

    # Fine-tune from a pre-trained model (progressive difficulty training)
    uv run python train_improved.py --config configs/estes_c6_medium.yaml \
        --load-model models/rocket_estes_c6_easy_*/best_model.zip
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rocket_config import RocketTrainingConfig, load_config

from spin_stabilized_control_env import SpinStabilizedCameraRocket
from spin_stabilized_control_env import RocketConfig as CompositeRocketConfig
from realistic_spin_rocket import RealisticMotorRocket
from motor_loader import Motor
from rocket_env.sensors import IMUObservationWrapper, IMUConfig
from pid_controller import PIDController, PIDConfig


class ImprovedRewardWrapper(gym.Wrapper):
    """
    Wrapper that implements a configurable reward function.

    This allows experimenting with different reward structures without
    modifying the base environment.
    """

    def __init__(self, env: gym.Env, reward_config: Dict[str, float]):
        super().__init__(env)
        self.reward_config = reward_config
        self.prev_action = None
        self.prev_altitude = 0.0
        self.prev_spin_rate = 0.0
        self.step_count = 0
        self.settled = False  # Track if we've achieved stable low spin
        self.missed_deadline = False  # Track if we missed settling deadline

    def reset(self, **kwargs):
        self.prev_action = None
        self.prev_altitude = 0.0
        self.prev_spin_rate = 0.0
        self.step_count = 0
        self.settled = False
        self.missed_deadline = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, original_reward, terminated, truncated, info = self.env.step(action)

        # Calculate custom reward
        reward = self._calculate_reward(obs, action, info, terminated)

        # Store for next step
        self.prev_action = action.copy()
        self.prev_altitude = info.get("altitude_m", info.get("altitude", 0))
        self.prev_spin_rate = abs(info.get("roll_rate_deg_s", 0))

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, obs, action, info, terminated) -> float:
        """Calculate reward based on configuration"""
        rc = self.reward_config
        reward = 0.0

        # Get state info
        altitude = info.get("altitude_m", info.get("altitude", 0))
        spin_rate = abs(info.get("roll_rate_deg_s", 0))
        phase = info.get("phase", "boost")

        # 1. Altitude reward (progress toward goal)
        altitude_reward = altitude * rc.get("altitude_reward_scale", 0.01)
        reward += altitude_reward

        # 2. Spin penalty (quadratic - gentle on small errors, harsh on large)
        spin_penalty = (spin_rate**2) * rc.get("spin_penalty_scale", -0.002)
        reward += spin_penalty

        # 3. Low spin bonus (tiered bonus for maintaining low spin)
        # Gentle gradient - primary control comes from oscillation penalty
        threshold = rc.get("low_spin_threshold", 10.0)
        if spin_rate < threshold:
            # Smooth bonus that increases as spin decreases
            # At 0°/s: 1.2x bonus, at threshold: 1.0x bonus
            bonus_multiplier = 1.0 + 0.2 * (1.0 - spin_rate / threshold)
            reward += rc.get("low_spin_bonus", 1.0) * bonus_multiplier

        # 3b. Zero-spin bonus (extra reward for getting very close to zero)
        zero_threshold = rc.get("zero_spin_threshold", 1.0)
        zero_bonus = rc.get("zero_spin_bonus", 0.0)
        if zero_bonus > 0 and spin_rate < zero_threshold:
            # Quadratic scaling: max bonus at 0, decreasing to 0 at threshold
            # This creates strong gradient toward absolute zero
            zero_factor = (1.0 - spin_rate / zero_threshold) ** 2
            reward += zero_bonus * zero_factor

        # 4. Control effort penalty
        control_penalty = np.sum(np.abs(action)) * rc.get(
            "control_effort_penalty", -0.01
        )
        reward += control_penalty

        # 5. Control smoothness penalty
        if self.prev_action is not None:
            action_change = np.sum(np.abs(action - self.prev_action))
            smoothness_penalty = action_change * rc.get(
                "control_smoothness_penalty", -0.05
            )
            reward += smoothness_penalty

            # 5b. Control sign reversal penalty (penalize bang-bang control)
            for i in range(len(action)):
                if self.prev_action[i] * action[i] < 0:  # Sign changed
                    reward += rc.get("sign_reversal_penalty", -0.5)

        # 6. Spin oscillation penalty (penalize rapid changes in spin direction)
        if self.prev_spin_rate > 0:
            spin_change = abs(spin_rate - self.prev_spin_rate)
            oscillation_penalty = spin_change * rc.get(
                "spin_oscillation_penalty", -0.02
            )
            reward += oscillation_penalty

        # 7. Control saturation penalty (discourage hitting limits)
        saturation_threshold = 0.95
        for a in action:
            if abs(a) > saturation_threshold:
                reward += rc.get("saturation_penalty", -0.1)

        # 7b. Elastic net penalty on actions (L1 + L2 regularization)
        # L1: Encourages sparsity (zero actions when possible)
        # L2: Penalizes large actions quadratically
        l1_penalty = rc.get("action_l1_penalty", 0.0)
        l2_penalty = rc.get("action_l2_penalty", 0.0)
        if l1_penalty != 0.0 or l2_penalty != 0.0:
            action_l1 = np.sum(np.abs(action))  # Sum of |a_i|
            action_l2 = np.sum(action**2)  # Sum of a_i^2
            elastic_penalty = -(l1_penalty * action_l1 + l2_penalty * action_l2)
            reward += elastic_penalty

        # 8. Early settling bonus/penalty (reward quick stabilization, punish slow settling)
        time_s = info.get("time_s", self.step_count * 0.01)
        settling_threshold = rc.get("settling_spin_threshold", 5.0)
        settling_time_limit = rc.get("settling_time_limit", 0.5)

        if not self.settled and spin_rate < settling_threshold:
            self.settled = True
            if time_s < settling_time_limit:
                # Big bonus for settling within time limit
                reward += rc.get("early_settling_bonus", 50.0)
            elif time_s < settling_time_limit * 2:
                # Smaller bonus for settling within 2x time limit
                reward += rc.get("early_settling_bonus", 50.0) * 0.5

        # 9. Settling deadline penalty (one-time penalty for missing the deadline)
        if (
            not self.settled
            and not self.missed_deadline
            and time_s >= settling_time_limit
        ):
            self.missed_deadline = True
            # Apply penalty for missing the settling deadline
            reward += rc.get("settling_deadline_penalty", -50.0)

        # 10. Early-phase spin penalty multiplier
        # During the settling window, spin is penalized more heavily
        # This creates urgency to settle quickly
        if time_s < settling_time_limit and not self.settled:
            early_phase_multiplier = rc.get("early_phase_spin_multiplier", 2.0)
            # Apply additional spin penalty (on top of base spin penalty)
            extra_spin_penalty = (spin_rate**2) * rc.get("spin_penalty_scale", -0.002)
            reward += extra_spin_penalty * (early_phase_multiplier - 1.0)

        self.step_count += 1

        # 10. Terminal rewards
        if terminated:
            if altitude > rc.get("success_altitude", 100.0):
                reward += rc.get("success_bonus", 100.0)
            elif altitude < 1.0:  # Crash
                reward += rc.get("crash_penalty", -50.0)

        return reward


class ActionRateLimiter(gym.ActionWrapper):
    """
    Wrapper that limits how fast actions can change between timesteps.

    This physically prevents bang-bang control oscillation by constraining
    the slew rate of the actuators.

    DEPRECATED: Use ExponentialSmoothingWrapper for more realistic behavior.
    """

    def __init__(self, env: gym.Env, max_rate: float = 0.1):
        """
        Args:
            env: Environment to wrap
            max_rate: Maximum action change per timestep (in normalized [-1,1] space)
        """
        super().__init__(env)
        self.max_rate = max_rate
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def action(self, action):
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action)

        # Limit the rate of change
        delta = action - self.prev_action
        delta = np.clip(delta, -self.max_rate, self.max_rate)
        limited_action = self.prev_action + delta

        self.prev_action = limited_action.copy()
        return limited_action


class ExponentialSmoothingWrapper(gym.ActionWrapper):
    """
    Wrapper that applies exponential smoothing to actions.

    Models realistic servo/actuator dynamics where the actual position
    follows the commanded position with a time constant.

    Formula: action_applied = alpha * action_commanded + (1 - alpha) * prev_action

    This is more physically realistic than hard rate limiting because:
    - It models inertia and response time of real actuators
    - Smooth transitions even for large commanded changes
    - The time constant can be tuned to match real hardware
    """

    def __init__(self, env: gym.Env, alpha: float = 0.1):
        """
        Args:
            env: Environment to wrap
            alpha: Smoothing factor (0-1). Lower = smoother/slower response.
                   alpha=0.1 means ~10 timesteps to reach 63% of target
                   alpha=0.2 means ~5 timesteps to reach 63% of target
                   alpha=0.05 means ~20 timesteps to reach 63% of target
        """
        super().__init__(env)
        self.alpha = alpha
        self.smoothed_action = None

    def reset(self, **kwargs):
        self.smoothed_action = None
        return self.env.reset(**kwargs)

    def action(self, action):
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)

        # Exponential smoothing: new = alpha * commanded + (1-alpha) * previous
        self.smoothed_action = (
            self.alpha * action + (1 - self.alpha) * self.smoothed_action
        )

        return self.smoothed_action.copy()


class PreviousActionWrapper(gym.ObservationWrapper):
    """
    Wrapper that adds the previous APPLIED action to the observation space.

    This gives the agent information about its current actuator position,
    enabling it to learn incremental control strategies rather than
    absolute positioning.

    IMPORTANT: This wrapper looks for smoothed actions from ExponentialSmoothingWrapper
    to ensure the observation contains the actual applied action, not the commanded one.

    The observation space is extended by the action space dimensions.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.prev_applied_action = None

        # Extend observation space to include previous action
        obs_space = env.observation_space
        action_space = env.action_space

        # New observation space: [original_obs, prev_action]
        new_low = np.concatenate([obs_space.low, action_space.low])
        new_high = np.concatenate([obs_space.high, action_space.high])

        self.observation_space = spaces.Box(
            low=new_low.astype(np.float32),
            high=new_high.astype(np.float32),
            dtype=np.float32,
        )

        self._action_dim = action_space.shape[0]

    def reset(self, **kwargs):
        self.prev_applied_action = np.zeros(self._action_dim, dtype=np.float32)
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def observation(self, observation):
        """Append previous applied action to observation."""
        if self.prev_applied_action is None:
            self.prev_applied_action = np.zeros(self._action_dim, dtype=np.float32)
        return np.concatenate([observation, self.prev_applied_action]).astype(
            np.float32
        )

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Get the actual applied action (smoothed if smoothing wrapper exists)
        applied_action = self._get_applied_action(action)
        self.prev_applied_action = np.array(applied_action, dtype=np.float32)

        return self.observation(obs), reward, terminated, truncated, info

    def _get_applied_action(self, commanded_action):
        """Get the actual applied action, checking for smoothing wrappers."""
        # Walk through wrapped environments to find smoothing wrapper
        env = self.env
        while hasattr(env, "env"):
            if isinstance(env, ExponentialSmoothingWrapper):
                # Return the smoothed action
                return env.smoothed_action
            if isinstance(env, ActionRateLimiter):
                # Return the rate-limited action
                return env.prev_action
            env = env.env

        # No smoothing found, return commanded action
        return commanded_action


class NormalizedActionWrapper(gym.ActionWrapper):
    """
    Wrapper that normalizes action space to [-1, 1] and scales internally.

    PPO works better with normalized action spaces.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.original_low = env.action_space.low.copy()
        self.original_high = env.action_space.high.copy()

        # Create normalized action space
        self.action_space = spaces.Box(
            low=-np.ones_like(self.original_low),
            high=np.ones_like(self.original_high),
            dtype=np.float32,
        )

    def action(self, action):
        """Scale action from [-1, 1] to original range"""
        # Linear interpolation: [-1, 1] -> [low, high]
        scaled = self.original_low + (action + 1) * 0.5 * (
            self.original_high - self.original_low
        )
        return scaled


class DeltaActionWrapper(gym.ActionWrapper):
    """
    Wrapper that converts action space to incremental (delta) actions.

    Instead of commanding absolute positions, the agent commands changes:
        position_t = clip(position_{t-1} + delta * max_delta, -1, 1)

    This naturally limits the rate of change and encourages smoother control.
    The agent can still reach any position, but must do so gradually.

    Benefits:
    - Physical constraint on rate of change
    - Encourages smoother trajectories
    - Agent learns to "nudge" rather than "jump"
    """

    def __init__(self, env: gym.Env, max_delta: float = 0.1):
        """
        Args:
            env: Environment to wrap
            max_delta: Maximum change per timestep in normalized [-1,1] space.
                       0.1 means it takes 10 steps to go from -1 to 0.
        """
        super().__init__(env)
        self.max_delta = max_delta
        self.current_position = None

        # Action space remains [-1, 1] but now represents delta commands
        # The agent outputs desired change direction and magnitude

    def reset(self, **kwargs):
        self.current_position = None
        return self.env.reset(**kwargs)

    def action(self, delta_action):
        """Convert delta action to absolute position"""
        if self.current_position is None:
            # Start at neutral position
            self.current_position = np.zeros_like(delta_action)

        # Scale delta by max_delta and add to current position
        # delta_action is in [-1, 1], so actual delta is [-max_delta, max_delta]
        actual_delta = delta_action * self.max_delta
        self.current_position = self.current_position + actual_delta

        # Clip to valid range
        self.current_position = np.clip(self.current_position, -1.0, 1.0)

        return self.current_position.copy()


class ResidualPIDWrapper(gym.Wrapper):
    """
    Wrapper that combines a PID controller with RL residual corrections.

    The RL agent learns small corrections on top of PID output:
        final_action = PID_output + clip(RL_output * max_residual, -max_residual, max_residual)

    Benefits:
    - Inherits smooth behavior from PID baseline
    - RL only needs to learn small corrections
    - Robust: even if RL fails, PID provides reasonable control
    - Faster training: agent starts with good baseline behavior
    """

    def __init__(
        self,
        env: gym.Env,
        pid_config: PIDConfig = None,
        max_residual: float = 0.3,
        dt: float = 0.01,
    ):
        """
        Args:
            env: Environment to wrap
            pid_config: PID controller configuration
            max_residual: Maximum RL correction magnitude (e.g., 0.3 = ±30% adjustment)
            dt: Timestep for PID integration
        """
        super().__init__(env)
        self.pid = PIDController(pid_config or PIDConfig())
        self.max_residual = max_residual
        self.dt = dt
        self._last_info = {}
        self._last_obs = None

        # Track contributions for visualization/debugging
        self.last_pid_action = 0.0
        self.last_residual = 0.0
        self.last_combined_action = 0.0

    def reset(self, **kwargs):
        self.pid.reset()
        self._last_info = {}
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._last_info = info
        return obs, info

    def step(self, rl_action):
        """
        Combine PID output with RL residual and step the environment.

        Args:
            rl_action: RL agent's output in [-1, 1], interpreted as residual correction
        """
        # Get PID action based on current state
        pid_action = self.pid.step(self._last_obs, self._last_info, self.dt)

        # Scale RL action to residual range
        residual = np.clip(
            rl_action * self.max_residual, -self.max_residual, self.max_residual
        )

        # Combine PID + residual
        combined_action = pid_action + residual

        # Clip to valid action range
        combined_action = np.clip(combined_action, -1.0, 1.0)

        # Store for debugging/visualization
        self.last_pid_action = (
            float(pid_action[0]) if len(pid_action.shape) > 0 else float(pid_action)
        )
        self.last_residual = (
            float(residual[0]) if len(residual.shape) > 0 else float(residual)
        )
        self.last_combined_action = (
            float(combined_action[0])
            if len(combined_action.shape) > 0
            else float(combined_action)
        )

        # Step environment with combined action
        obs, reward, terminated, truncated, info = self.env.step(combined_action)

        # Store for next PID computation
        self._last_obs = obs
        self._last_info = info

        # Add PID/residual info for logging
        info["pid_action"] = self.last_pid_action
        info["rl_residual"] = self.last_residual
        info["combined_action"] = self.last_combined_action

        return obs, reward, terminated, truncated, info


class TrainingMetricsCallback(BaseCallback):
    """
    Callback for tracking and logging training metrics.
    """

    def __init__(self, config: RocketTrainingConfig, verbose: int = 0):
        super().__init__(verbose)
        self.config = config
        self.log_freq = config.logging.log_episode_freq

        self.episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_altitudes = []
        self.episode_mean_spin_rates = []
        self.episode_max_spin_rates = []
        self.episode_camera_scores = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get("dones", [])):
            if done and "infos" in self.locals:
                info = self.locals["infos"][i]
                if info:
                    self.episode_count += 1

                    # Track metrics
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"].get("r", 0))
                        self.episode_lengths.append(info["episode"].get("l", 0))

                    self.episode_altitudes.append(
                        info.get("altitude_m", info.get("altitude", 0))
                    )
                    # Track mean and max spin rates (fall back to final if not available)
                    self.episode_mean_spin_rates.append(
                        abs(
                            info.get(
                                "mean_spin_rate_deg_s", info.get("roll_rate_deg_s", 0)
                            )
                        )
                    )
                    self.episode_max_spin_rates.append(
                        abs(
                            info.get(
                                "max_spin_rate_deg_s", info.get("roll_rate_deg_s", 0)
                            )
                        )
                    )

                    # Camera quality
                    quality = info.get("horizontal_camera_quality", "")
                    if "Excellent" in quality:
                        self.episode_camera_scores.append(4)
                    elif "Good" in quality:
                        self.episode_camera_scores.append(3)
                    elif "Fair" in quality:
                        self.episode_camera_scores.append(2)
                    else:
                        self.episode_camera_scores.append(1)

                    # Log periodically
                    if self.episode_count % self.log_freq == 0:
                        self._log_metrics()

        return True

    def _log_metrics(self):
        n = min(self.log_freq, len(self.episode_rewards))
        if n == 0:
            return

        print(f"\n{'='*60}")
        print(f"Episode {self.episode_count} | Timestep {self.num_timesteps}")
        print(f"{'='*60}")

        # Recent performance
        recent_rewards = self.episode_rewards[-n:]
        recent_altitudes = self.episode_altitudes[-n:]
        recent_mean_spins = self.episode_mean_spin_rates[-n:]
        recent_max_spins = self.episode_max_spin_rates[-n:]

        print(
            f"Rewards (last {n}): {np.mean(recent_rewards):.1f} ± {np.std(recent_rewards):.1f}"
        )
        print(
            f"Max Altitude: {np.mean(recent_altitudes):.1f} ± {np.std(recent_altitudes):.1f} m"
        )
        print(
            f"Mean Spin Rate: {np.mean(recent_mean_spins):.1f} ± {np.std(recent_mean_spins):.1f} °/s"
        )
        print(
            f"Max Spin Rate: {np.mean(recent_max_spins):.1f} ± {np.std(recent_max_spins):.1f} °/s"
        )

        # Success metrics
        high_alt = sum(1 for a in recent_altitudes if a > 50) / n * 100
        low_mean_spin = sum(1 for s in recent_mean_spins if s < 10) / n * 100
        low_max_spin = sum(1 for s in recent_max_spins if s < 30) / n * 100
        print(f"High altitude (>50m): {high_alt:.0f}%")
        print(f"Low mean spin (<10°/s): {low_mean_spin:.0f}%")
        print(f"Low max spin (<30°/s): {low_max_spin:.0f}%")

        if self.episode_camera_scores:
            recent_cameras = self.episode_camera_scores[-n:]
            print(f"Camera quality: {np.mean(recent_cameras):.2f}/4.0")

        print(f"{'='*60}\n")


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning - gradually increasing difficulty.
    """

    def __init__(
        self, config: RocketTrainingConfig, env_factory: Callable, verbose: int = 0
    ):
        super().__init__(verbose)
        self.config = config
        self.env_factory = env_factory
        self.current_stage = 0
        self.stage_episode_rewards = []

    def _on_step(self) -> bool:
        if not self.config.curriculum.enabled:
            return True

        # Track episode rewards
        for i, done in enumerate(self.locals.get("dones", [])):
            if done and "infos" in self.locals:
                info = self.locals["infos"][i]
                if info and "episode" in info:
                    self.stage_episode_rewards.append(info["episode"].get("r", 0))

        # Check for stage advancement
        if (
            len(self.stage_episode_rewards)
            >= self.config.curriculum.episodes_to_evaluate
        ):
            self._check_advancement()

        return True

    def _check_advancement(self):
        stages = self.config.curriculum.stages
        if self.current_stage >= len(stages) - 1:
            return

        current = stages[self.current_stage]
        target = current.get("target_reward", 0)
        threshold = self.config.curriculum.advancement_threshold

        # Check if we've met the advancement criteria
        recent = self.stage_episode_rewards[
            -self.config.curriculum.episodes_to_evaluate :
        ]
        success_rate = sum(1 for r in recent if r >= target) / len(recent)

        if success_rate >= threshold:
            self.current_stage += 1
            self.stage_episode_rewards = []

            next_stage = stages[self.current_stage]
            print(f"\n🎓 CURRICULUM: Advancing to stage '{next_stage['name']}'!")
            print(f"   Success rate was {success_rate*100:.0f}%")

            # Update environment with new difficulty
            # Note: This requires recreating the environment
            # In practice, you might use a configurable environment instead


def create_environment(
    config: RocketTrainingConfig, seed: int = None, curriculum_stage: int = None
) -> gym.Env:
    """
    Create environment based on configuration.

    Args:
        config: Training configuration
        seed: Random seed
        curriculum_stage: If using curriculum, which stage to use
    """

    # Load airframe from config (handles both new-style and legacy configs)
    airframe = config.physics.resolve_airframe()

    # Build RocketConfig with simulation/physics tuning parameters only
    # (geometry comes from airframe)
    rocket_config = CompositeRocketConfig(
        max_tab_deflection=config.physics.max_tab_deflection,
        tab_chord_fraction=config.physics.tab_chord_fraction,
        tab_span_fraction=config.physics.tab_span_fraction,
        num_controlled_fins=getattr(config.physics, "num_controlled_fins", 2),
        disturbance_scale=getattr(config.physics, "disturbance_scale", 0.0001),
        initial_spin_std=getattr(config.physics, "initial_spin_std", 15.0),
        damping_scale=getattr(config.physics, "damping_scale", 2.0),
        max_roll_rate=getattr(config.physics, "max_roll_rate", 720.0),
        max_episode_time=getattr(config.physics, "max_episode_time", 15.0),
        dt=getattr(config.environment, "dt", 0.01),
    )

    # Build motor config dict from MotorConfig dataclass
    motor_config_dict = {
        "name": config.motor.name,
        "manufacturer": config.motor.manufacturer,
        "designation": config.motor.designation,
        "diameter_mm": config.motor.diameter_mm,
        "length_mm": config.motor.length_mm,
        "total_mass_g": config.motor.total_mass_g,
        "propellant_mass_g": config.motor.propellant_mass_g,
        "case_mass_g": config.motor.case_mass_g,
        "impulse_class": config.motor.impulse_class,
        "total_impulse_Ns": config.motor.total_impulse_Ns,
        "avg_thrust_N": config.motor.avg_thrust_N,
        "max_thrust_N": config.motor.max_thrust_N,
        "burn_time_s": config.motor.burn_time_s,
        "thrust_curve": config.motor.thrust_curve,
        "thrust_multiplier": config.motor.thrust_multiplier,
    }
    # Remove None values
    motor_config_dict = {k: v for k, v in motor_config_dict.items() if v is not None}

    # Create base environment with airframe and motor
    env = RealisticMotorRocket(
        airframe=airframe,
        motor_config=motor_config_dict,
        config=rocket_config,
    )

    print(f"Environment class: {type(env).__name__}")
    print(f"Disturbance scale in config: {rocket_config.disturbance_scale}")
    # Check if the env actually has/uses this parameter
    if hasattr(env, "config"):
        print(
            f"Env config disturbance_scale: {getattr(env.config, 'disturbance_scale', 'NOT FOUND')}"
        )

    # Apply wrappers
    # 1. IMU sensor noise wrapper (if enabled)
    if config.sensors.enabled:
        if config.sensors.imu_custom:
            imu_config = IMUConfig.from_dict(config.sensors.imu_custom)
        else:
            imu_config = IMUConfig.get_preset(config.sensors.imu_preset)

        env = IMUObservationWrapper(
            env,
            imu_config=imu_config,
            control_rate_hz=config.sensors.control_rate_hz,
            derive_acceleration=config.sensors.derive_acceleration,
        )
        print(f"IMU simulation: {imu_config.name} @ {config.sensors.control_rate_hz}Hz")

    # 2. Normalize actions to [-1, 1]
    env = NormalizedActionWrapper(env)

    # 2b. Residual PID (RL learns corrections on top of PID) - RECOMMENDED
    use_residual_pid = getattr(config.physics, "use_residual_pid", False)
    if use_residual_pid:
        max_residual = getattr(config.physics, "max_residual", 0.3)
        pid_config = PIDConfig(
            Cprop=getattr(config.physics, "pid_Kp", 0.02),
            Cint=getattr(config.physics, "pid_Ki", 0.005),
            Cderiv=getattr(config.physics, "pid_Kd", 0.05),
        )
        dt = getattr(config.environment, "dt", 0.01)
        env = ResidualPIDWrapper(
            env, pid_config=pid_config, max_residual=max_residual, dt=dt
        )
        print(
            f"Residual PID enabled: max_residual={max_residual}, PID gains=({pid_config.Cprop}, {pid_config.Cint}, {pid_config.Cderiv})"
        )

    # 2c. Delta actions (agent commands incremental changes) - alternative to residual PID
    use_delta_actions = getattr(config.physics, "use_delta_actions", False)
    max_delta_per_step = getattr(config.physics, "max_delta_per_step", 0.1)
    if use_delta_actions and not use_residual_pid:
        env = DeltaActionWrapper(env, max_delta=max_delta_per_step)
        print(f"Delta actions enabled: max_delta={max_delta_per_step} per step")

    # 3. Action smoothing (prevents bang-bang oscillation)
    # Check for new exponential smoothing parameter first, fall back to rate limiter
    action_smoothing_alpha = getattr(config.physics, "action_smoothing_alpha", None)
    action_rate_limit = getattr(config.physics, "action_rate_limit", 0.1)

    if action_smoothing_alpha is not None and action_smoothing_alpha > 0:
        # Use exponential smoothing (preferred - more realistic)
        env = ExponentialSmoothingWrapper(env, alpha=action_smoothing_alpha)
        print(f"Action smoothing: exponential alpha={action_smoothing_alpha}")
    elif action_rate_limit > 0:
        # Fall back to hard rate limiting (legacy)
        env = ActionRateLimiter(env, max_rate=action_rate_limit)
        print(f"Action smoothing: rate limit={action_rate_limit} per timestep")

    # 4. Add previous action to observations (helps agent learn incremental control)
    include_prev_action = getattr(config.physics, "include_previous_action", False)
    if include_prev_action:
        env = PreviousActionWrapper(env)
        print("Observation space extended with previous action")

    # 5. Custom reward function
    reward_dict = {
        "altitude_reward_scale": config.reward.altitude_reward_scale,
        "spin_penalty_scale": config.reward.spin_penalty_scale,
        "low_spin_bonus": config.reward.low_spin_bonus,
        "low_spin_threshold": config.reward.low_spin_threshold,
        "zero_spin_bonus": getattr(config.reward, "zero_spin_bonus", 0.0),
        "zero_spin_threshold": getattr(config.reward, "zero_spin_threshold", 1.0),
        "control_effort_penalty": config.reward.control_effort_penalty,
        "control_smoothness_penalty": config.reward.control_smoothness_penalty,
        "spin_oscillation_penalty": getattr(
            config.reward, "spin_oscillation_penalty", -0.02
        ),
        "sign_reversal_penalty": getattr(config.reward, "sign_reversal_penalty", -0.5),
        "saturation_penalty": getattr(config.reward, "saturation_penalty", -0.1),
        "early_settling_bonus": getattr(config.reward, "early_settling_bonus", 50.0),
        "settling_spin_threshold": getattr(
            config.reward, "settling_spin_threshold", 5.0
        ),
        "settling_time_limit": getattr(config.reward, "settling_time_limit", 0.5),
        "settling_deadline_penalty": getattr(
            config.reward, "settling_deadline_penalty", -50.0
        ),
        "early_phase_spin_multiplier": getattr(
            config.reward, "early_phase_spin_multiplier", 2.0
        ),
        "action_l1_penalty": getattr(config.reward, "action_l1_penalty", 0.0),
        "action_l2_penalty": getattr(config.reward, "action_l2_penalty", 0.0),
        "success_bonus": config.reward.success_bonus,
        "crash_penalty": config.reward.crash_penalty,
        "success_altitude": config.environment.max_altitude,
    }
    env = ImprovedRewardWrapper(env, reward_dict)

    # 6. Monitor wrapper (for logging)
    env = Monitor(env)

    return env


def train(
    config: RocketTrainingConfig,
    load_model_path: Optional[str] = None,
    early_stopping_patience: int = 0,
):
    """Main training function

    Args:
        config: Training configuration
        load_model_path: Optional path to pre-trained model for fine-tuning
        early_stopping_patience: Stop if no improvement for N evaluations (0=disabled)
    """

    # Validate configuration
    issues = config.validate()
    if issues:
        print("\n⚠️ Configuration Issues:")
        for issue in issues:
            print(f"  {issue}")

        critical = [i for i in issues if "CRITICAL" in i]
        if critical:
            print("\n❌ Cannot proceed with critical issues. Please fix configuration.")
            return None

        response = input("\nContinue anyway? (y/N): ")
        if response.lower() != "y":
            return None

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.logging.experiment_name}_{config.motor.name}_{timestamp}"

    log_dir = Path(config.logging.log_dir) / run_name
    save_dir = Path(config.logging.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config.save(save_dir / "config.yaml")

    print(f"\n{'='*70}")
    print("ROCKET SPIN CONTROL TRAINING")
    print(f"{'='*70}")
    print(f"Motor: {config.motor.name}")
    print(f"Dry mass: {config.physics.dry_mass*1000:.1f}g")
    print(f"Total timesteps: {config.ppo.total_timesteps:,}")
    print(f"Parallel envs: {config.ppo.n_envs}")
    print(f"Device: {config.ppo.device}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*70}\n")

    # Create vectorized environment
    def make_env():
        return create_environment(config)

    if config.ppo.n_envs > 1:
        train_env = make_vec_env(make_env, n_envs=config.ppo.n_envs)
    else:
        train_env = DummyVecEnv([make_env])

    # Observation normalization - handle differently for new vs loaded models
    vec_normalize_loaded = False
    if load_model_path and config.environment.normalize_observations:
        # Check for VecNormalize stats alongside the model
        load_path = Path(load_model_path)
        vec_normalize_path = load_path.parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            print(f"Loading VecNormalize stats from: {vec_normalize_path}")
            train_env = VecNormalize.load(str(vec_normalize_path), train_env)
            train_env.training = True  # Enable stats updates during training
            vec_normalize_loaded = True

    if config.environment.normalize_observations and not vec_normalize_loaded:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,  # We handle reward in wrapper
            clip_obs=config.environment.obs_clip_value,
        )

    # Create eval environment
    eval_env = DummyVecEnv([make_env])
    if config.environment.normalize_observations:
        if vec_normalize_loaded:
            # Copy stats from training env
            eval_env = VecNormalize.load(
                str(Path(load_model_path).parent / "vec_normalize.pkl"), eval_env
            )
        else:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,
            )
        eval_env.training = False  # Don't update stats during eval

    # Setup policy network
    activation_fn = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
    }.get(config.ppo.activation, torch.nn.Tanh)

    policy_kwargs = dict(
        net_arch=dict(pi=config.ppo.policy_net_arch, vf=config.ppo.value_net_arch),
        activation_fn=activation_fn,
    )

    # Create or load PPO model
    if load_model_path:
        # Load pre-trained model for fine-tuning
        load_path = Path(load_model_path)
        print(f"Loading pre-trained model from: {load_path}")

        model = PPO.load(
            str(load_path),
            env=train_env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            clip_range_vf=config.ppo.clip_range_vf,
            normalize_advantage=config.ppo.normalize_advantage,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            verbose=1,
            device=config.ppo.device,
            tensorboard_log=str(log_dir) if config.logging.tensorboard_log else None,
        )
        print("Model loaded successfully for fine-tuning")
    else:
        # Create new PPO model
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.ppo.learning_rate,
            n_steps=config.ppo.n_steps,
            batch_size=config.ppo.batch_size,
            n_epochs=config.ppo.n_epochs,
            gamma=config.ppo.gamma,
            gae_lambda=config.ppo.gae_lambda,
            clip_range=config.ppo.clip_range,
            clip_range_vf=config.ppo.clip_range_vf,
            normalize_advantage=config.ppo.normalize_advantage,
            ent_coef=config.ppo.ent_coef,
            vf_coef=config.ppo.vf_coef,
            max_grad_norm=config.ppo.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=config.ppo.device,
            tensorboard_log=str(log_dir) if config.logging.tensorboard_log else None,
        )

    # Setup callbacks
    callbacks = []

    # Metrics callback
    metrics_callback = TrainingMetricsCallback(config, verbose=1)
    callbacks.append(metrics_callback)

    # Early stopping callback (if enabled)
    stop_callback = None
    if early_stopping_patience > 0:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=early_stopping_patience,
            min_evals=early_stopping_patience,  # Wait at least this many evals before stopping
            verbose=1,
        )
        print(
            f"Early stopping enabled: will stop after {early_stopping_patience} evaluations without improvement"
        )

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_dir),
        log_path=str(save_dir / "eval"),
        eval_freq=config.logging.eval_freq,
        n_eval_episodes=config.logging.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=stop_callback,
    )
    callbacks.append(eval_callback)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=config.logging.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="rocket_ppo",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Train!
    print("Starting training...")
    model.learn(
        total_timesteps=config.ppo.total_timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )

    # Save final model
    final_path = save_dir / "final_model"
    model.save(final_path)

    if config.environment.normalize_observations:
        train_env.save(save_dir / "vec_normalize.pkl")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Final model: {final_path}.zip")
    print(f"Best model: {save_dir}/best_model.zip")
    print(f"\nTo evaluate:")
    print(
        f"  uv run python visualizations/visualize_spin_agent.py {save_dir}/best_model.zip --config {save_dir}/config.yaml"
    )
    print(f"\nTo view tensorboard:")
    print(f" uv run tensorboard --logdir {log_dir}")
    print(f"{'='*70}\n")

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train rocket spin control agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  uv run python train_improved.py --config configs/estes_c6.yaml

  # Quick debug run
  uv run python train_improved.py --config configs/debug.yaml

  # Override parameters
  uv run python train_improved.py --config configs/estes_c6.yaml --dry-mass 0.12 --timesteps 100000

  # Fine-tune from a pre-trained model
  uv run python train_improved.py --config configs/estes_c6_medium.yaml --load-model models/rocket_estes_c6_easy_*/best_model.zip

  # Create default config files
  uv run python train_improved.py --create-configs
        """,
    )

    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument(
        "--create-configs",
        action="store_true",
        help="Create default config files and exit",
    )

    # Config overrides
    parser.add_argument("--dry-mass", type=float, help="Override dry mass (kg)")
    parser.add_argument("--motor", type=str, help="Override motor name")
    parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--n-envs", type=int, help="Override number of parallel envs")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda/auto)")
    parser.add_argument(
        "--load-model",
        type=str,
        help="Path to pre-trained model (.zip) to fine-tune from",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        metavar="N",
        help="Stop training if no improvement for N evaluations (0=disabled)",
    )

    args = parser.parse_args()

    if args.create_configs:
        from rocket_config import create_default_configs

        create_default_configs()
        return

    if not args.config:
        parser.print_help()
        print("\n❌ Error: --config is required (or use --create-configs)")
        return

    # Load config
    config = load_config(args.config)

    # Apply overrides
    if args.dry_mass is not None:
        config.physics.dry_mass = args.dry_mass
    if args.motor is not None:
        config.motor.name = args.motor
    if args.timesteps is not None:
        config.ppo.total_timesteps = args.timesteps
    if args.lr is not None:
        config.ppo.learning_rate = args.lr
    if args.n_envs is not None:
        config.ppo.n_envs = args.n_envs
    if args.device is not None:
        config.ppo.device = args.device

    # Train (with optional model loading for fine-tuning)
    train(
        config,
        load_model_path=args.load_model,
        early_stopping_patience=args.early_stopping,
    )


if __name__ == "__main__":
    main()
