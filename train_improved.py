#!/usr/bin/env python3
"""
PPO Training Script for Rocket Spin Control

Features:
- Configurable mass/thrust parameters
- Observation normalization
- Configurable reward function
- Curriculum learning support
- Logging and diagnostics

Usage:
    # Using config file
    python train_improved.py --config configs/estes_c6.yaml

    # Quick test
    python train_improved.py --config configs/debug.yaml

    # Override specific parameters
    python train_improved.py --config configs/estes_c6.yaml --dry-mass 0.12 --timesteps 100000
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
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from rocket_config import RocketTrainingConfig, load_config

from spin_stabilized_control_env import SpinStabilizedCameraRocket
from spin_stabilized_control_env import RocketConfig as CompositeRocketConfig
from realistic_spin_rocket import RealisticMotorRocket
from motor_loader import Motor


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

    def reset(self, **kwargs):
        self.prev_action = None
        self.prev_altitude = 0.0
        self.prev_spin_rate = 0.0
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

        # 2. Spin penalty (penalize high spin rates)
        spin_penalty = spin_rate * rc.get("spin_penalty_scale", -0.1)
        reward += spin_penalty

        # 3. Low spin bonus (bonus for maintaining low spin)
        threshold = rc.get("low_spin_threshold", 10.0)
        if spin_rate < threshold:
            reward += rc.get("low_spin_bonus", 1.0)

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

        # 6. Terminal rewards
        if terminated:
            if altitude > rc.get("success_altitude", 100.0):
                reward += rc.get("success_bonus", 100.0)
            elif altitude < 1.0:  # Crash
                reward += rc.get("crash_penalty", -50.0)

        return reward


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
        self.episode_spin_rates = []
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
                    self.episode_spin_rates.append(abs(info.get("roll_rate_deg_s", 0)))

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
        recent_spins = self.episode_spin_rates[-n:]

        print(
            f"Rewards (last {n}): {np.mean(recent_rewards):.1f} ± {np.std(recent_rewards):.1f}"
        )
        print(
            f"Max Altitude: {np.mean(recent_altitudes):.1f} ± {np.std(recent_altitudes):.1f} m"
        )
        print(
            f"Final Spin Rate: {np.mean(recent_spins):.1f} ± {np.std(recent_spins):.1f} °/s"
        )

        # Success metrics
        high_alt = sum(1 for a in recent_altitudes if a > 50) / n * 100
        low_spin = sum(1 for s in recent_spins if s < 30) / n * 100
        print(f"High altitude (>50m): {high_alt:.0f}%")
        print(f"Low spin (<30°/s): {low_spin:.0f}%")

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

    # Build rocket config from training config
    # Include patched-specific parameters if using patched environment
    config_kwargs = dict(
        dry_mass=config.physics.dry_mass,
        propellant_mass=config.physics.propellant_mass,
        diameter=config.physics.diameter,
        length=config.physics.length,
        num_fins=config.physics.num_fins,
        fin_span=config.physics.fin_span,
        max_tab_deflection=config.physics.max_tab_deflection,
        tab_chord_fraction=config.physics.tab_chord_fraction,
        tab_span_fraction=config.physics.tab_span_fraction,
        # === PHYSICS FIX PARAMETERS (already present) ===
        disturbance_scale=getattr(config.physics, "disturbance_scale", 0.0001),
        initial_spin_std=getattr(config.physics, "initial_spin_std", 15.0),
        damping_scale=getattr(config.physics, "damping_scale", 2.0),
        # === NEW: EPISODE TERMINATION PARAMETERS ===
        max_roll_rate=getattr(config.physics, "max_roll_rate", 720.0),
        max_episode_time=getattr(config.environment, "max_episode_time", 15.0),
        dt=getattr(config.environment, "dt", 0.01),
    )

    rocket_config = CompositeRocketConfig(**config_kwargs)

    # Get motor from config using motor_loader
    # Convert MotorConfig to Motor object
    motor = config.motor.to_motor()

    # Create base environment
    env = RealisticMotorRocket(motor, rocket_config)

    print(f"Environment class: {type(env).__name__}")
    print(f"Disturbance scale in config: {rocket_config.disturbance_scale}")
    # Check if the env actually has/uses this parameter
    if hasattr(env, "config"):
        print(
            f"Env config disturbance_scale: {getattr(env.config, 'disturbance_scale', 'NOT FOUND')}"
        )

    # Apply wrappers
    # 1. Normalize actions to [-1, 1]
    env = NormalizedActionWrapper(env)

    # 2. Custom reward function
    reward_dict = {
        "altitude_reward_scale": config.reward.altitude_reward_scale,
        "spin_penalty_scale": config.reward.spin_penalty_scale,
        "low_spin_bonus": config.reward.low_spin_bonus,
        "low_spin_threshold": config.reward.low_spin_threshold,
        "control_effort_penalty": config.reward.control_effort_penalty,
        "control_smoothness_penalty": config.reward.control_smoothness_penalty,
        "success_bonus": config.reward.success_bonus,
        "crash_penalty": config.reward.crash_penalty,
        "success_altitude": config.environment.max_altitude,
    }
    env = ImprovedRewardWrapper(env, reward_dict)

    # 3. Monitor wrapper (for logging)
    env = Monitor(env)

    return env


def train(config: RocketTrainingConfig):
    """Main training function"""

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

    # Observation normalization
    if config.environment.normalize_observations:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,  # We handle reward in wrapper
            clip_obs=config.environment.obs_clip_value,
        )

    # Create eval environment
    eval_env = DummyVecEnv([make_env])
    if config.environment.normalize_observations:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False,  # Don't update stats during eval
        )

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

    # Create PPO model
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
        f"  python evaluate_rocket.py --model {save_dir}/best_model.zip --config {save_dir}/config.yaml"
    )
    print(f"\nTo view tensorboard:")
    print(f"  tensorboard --logdir {log_dir}")
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
  python train_improved.py --config configs/estes_c6.yaml

  # Quick debug run
  python train_improved.py --config configs/debug.yaml

  # Override parameters
  python train_improved.py --config configs/estes_c6.yaml --dry-mass 0.12 --timesteps 100000

  # Create default config files
  python train_improved.py --create-configs
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

    # Train
    train(config)


if __name__ == "__main__":
    main()
