#!/usr/bin/env python3
"""
Visualization Script for Spin-Stabilized Rocket Control Agent

This script evaluates a trained agent and creates comprehensive visualizations
of flight performance, spin control, and camera quality.

Usage:
    uv run python visualizations/visualize_spin_agent.py models/best_model.zip --config configs/estes_c6_easy.yaml
    uv run python visualizations/visualize_spin_agent.py models/best_model.zip --n-episodes 50 --save-dir results/
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import argparse
import yaml
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

# Import environment components
from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig
from realistic_spin_rocket import RealisticMotorRocket
from rocket_config import load_config


@dataclass
class SpinEpisodeData:
    """Data structure for spin control episode results"""

    episode_id: int
    time: np.ndarray
    altitude: np.ndarray
    velocity: np.ndarray
    roll_rate: np.ndarray  # deg/s
    roll_angle: np.ndarray  # rad
    actions: np.ndarray  # normalized [-1, 1]
    tab_deflection: np.ndarray  # degrees
    rewards: np.ndarray
    thrust: np.ndarray
    dynamic_pressure: np.ndarray

    # Summary metrics
    max_altitude: float
    final_spin_rate: float
    mean_spin_rate: float
    total_reward: float
    episode_length: int
    camera_quality: str

    # Success criteria
    reached_target_altitude: bool
    maintained_low_spin: bool
    success: bool

    # Optional fields (must come after required fields)
    disturbance_torque: np.ndarray = None  # Atmospheric disturbance (Nm)


class NormalizedActionWrapper(gym.ActionWrapper):
    """Wrapper to normalize actions to [-1, 1]"""

    def __init__(self, env):
        super().__init__(env)
        self.original_low = env.action_space.low.copy()
        self.original_high = env.action_space.high.copy()
        self.action_space = spaces.Box(
            low=-np.ones_like(self.original_low),
            high=np.ones_like(self.original_high),
            dtype=np.float32,
        )

    def action(self, action):
        return self.original_low + (action + 1) * 0.5 * (
            self.original_high - self.original_low
        )


class ActionRateLimiter(gym.ActionWrapper):
    """Limits how fast actions can change between timesteps (legacy)."""

    def __init__(self, env, max_rate: float = 0.1):
        super().__init__(env)
        self.max_rate = max_rate
        self.prev_action = None

    def reset(self, **kwargs):
        self.prev_action = None
        return self.env.reset(**kwargs)

    def action(self, action):
        if self.prev_action is None:
            self.prev_action = np.zeros_like(action)
        delta = action - self.prev_action
        delta = np.clip(delta, -self.max_rate, self.max_rate)
        limited_action = self.prev_action + delta
        self.prev_action = limited_action.copy()
        return limited_action


class ExponentialSmoothingWrapper(gym.ActionWrapper):
    """Applies exponential smoothing to actions for realistic actuator dynamics."""

    def __init__(self, env, alpha: float = 0.1):
        super().__init__(env)
        self.alpha = alpha
        self.smoothed_action = None

    def reset(self, **kwargs):
        self.smoothed_action = None
        return self.env.reset(**kwargs)

    def action(self, action):
        if self.smoothed_action is None:
            self.smoothed_action = np.zeros_like(action)
        self.smoothed_action = (
            self.alpha * action + (1 - self.alpha) * self.smoothed_action
        )
        return self.smoothed_action.copy()


class DeltaActionWrapper(gym.ActionWrapper):
    """
    Wrapper that converts action space to incremental (delta) actions.

    Instead of commanding absolute positions, the agent commands changes:
        position_t = clip(position_{t-1} + delta * max_delta, -1, 1)
    """

    def __init__(self, env, max_delta: float = 0.1):
        super().__init__(env)
        self.max_delta = max_delta
        self.current_position = None

    def reset(self, **kwargs):
        self.current_position = None
        return self.env.reset(**kwargs)

    def action(self, delta_action):
        if self.current_position is None:
            self.current_position = np.zeros_like(delta_action)
        actual_delta = delta_action * self.max_delta
        self.current_position = self.current_position + actual_delta
        self.current_position = np.clip(self.current_position, -1.0, 1.0)
        return self.current_position.copy()


class PreviousActionWrapper(gym.ObservationWrapper):
    """Adds previous APPLIED action to observation space for incremental control learning."""

    def __init__(self, env):
        super().__init__(env)
        self.prev_applied_action = None
        obs_space = env.observation_space
        action_space = env.action_space
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
        env = self.env
        while hasattr(env, "env"):
            if isinstance(env, ExponentialSmoothingWrapper):
                return env.smoothed_action
            if isinstance(env, DeltaActionWrapper):
                return env.current_position
            if isinstance(env, ActionRateLimiter):
                return env.prev_action
            env = env.env
        return commanded_action


class SpinAgentEvaluator:
    """Evaluates trained spin control agent"""

    def __init__(
        self, model_path: str, config_path: str = None, vec_normalize_path: str = None
    ):
        """
        Initialize evaluator.

        Args:
            model_path: Path to trained PPO model
            config_path: Path to YAML config file
            vec_normalize_path: Path to VecNormalize stats (optional, auto-detected if not provided)
        """
        self.model = PPO.load(model_path)
        self.config_path = config_path

        # Auto-detect vec_normalize.pkl if not provided
        if vec_normalize_path is None:
            model_dir = Path(model_path).parent
            auto_path = model_dir / "vec_normalize.pkl"
            if auto_path.exists():
                vec_normalize_path = str(auto_path)
                print(f"Auto-detected VecNormalize stats: {vec_normalize_path}")

        self.vec_normalize_path = vec_normalize_path

        # Load config if provided (use structured loader)
        self.config = None
        if config_path and Path(config_path).exists():
            self.config = load_config(config_path)

        print(f"Loaded model from: {model_path}")
        if self.config:
            print(f"Using config: {config_path}")
        if self.vec_normalize_path:
            print(f"Using VecNormalize stats: {self.vec_normalize_path}")

    def create_env(self) -> gym.Env:
        """Create environment matching training config"""
        if self.config:
            # Load airframe from config (handles both new-style and legacy configs)
            airframe = self.config.physics.resolve_airframe()

            # Build RocketConfig with simulation/physics tuning parameters only
            # (geometry comes from airframe)
            rocket_config = RocketConfig(
                max_tab_deflection=self.config.physics.max_tab_deflection,
                tab_chord_fraction=self.config.physics.tab_chord_fraction,
                tab_span_fraction=self.config.physics.tab_span_fraction,
                num_controlled_fins=getattr(
                    self.config.physics, "num_controlled_fins", 2
                ),
                disturbance_scale=getattr(
                    self.config.physics, "disturbance_scale", 0.0001
                ),
                initial_spin_std=getattr(self.config.physics, "initial_spin_std", 15.0),
                damping_scale=getattr(self.config.physics, "damping_scale", 2.0),
                max_roll_rate=getattr(self.config.physics, "max_roll_rate", 720.0),
                max_episode_time=getattr(self.config.physics, "max_episode_time", 15.0),
                dt=getattr(self.config.environment, "dt", 0.01),
            )

            # Build motor config dict from MotorConfig dataclass
            motor_config_dict = {
                "name": self.config.motor.name,
                "manufacturer": self.config.motor.manufacturer,
                "designation": self.config.motor.designation,
                "diameter_mm": self.config.motor.diameter_mm,
                "length_mm": self.config.motor.length_mm,
                "total_mass_g": self.config.motor.total_mass_g,
                "propellant_mass_g": self.config.motor.propellant_mass_g,
                "case_mass_g": self.config.motor.case_mass_g,
                "impulse_class": self.config.motor.impulse_class,
                "total_impulse_Ns": self.config.motor.total_impulse_Ns,
                "avg_thrust_N": self.config.motor.avg_thrust_N,
                "max_thrust_N": self.config.motor.max_thrust_N,
                "burn_time_s": self.config.motor.burn_time_s,
                "thrust_curve": self.config.motor.thrust_curve,
                "thrust_multiplier": self.config.motor.thrust_multiplier,
            }
            # Remove None values
            motor_config_dict = {
                k: v for k, v in motor_config_dict.items() if v is not None
            }

            # Create environment with airframe and motor
            env = RealisticMotorRocket(
                airframe=airframe,
                motor_config=motor_config_dict,
                config=rocket_config,
            )
        else:
            raise ValueError(
                "Config file required for visualization. "
                "Use --config to specify a config YAML file."
            )

        env = NormalizedActionWrapper(env)

        # Apply delta actions if configured
        use_delta_actions = getattr(self.config.physics, "use_delta_actions", False)
        max_delta_per_step = getattr(self.config.physics, "max_delta_per_step", 0.1)
        if use_delta_actions:
            env = DeltaActionWrapper(env, max_delta=max_delta_per_step)

        # Apply action smoothing (exponential or rate limiting)
        action_smoothing_alpha = getattr(
            self.config.physics, "action_smoothing_alpha", None
        )
        action_rate_limit = getattr(self.config.physics, "action_rate_limit", 0.1)

        if action_smoothing_alpha is not None and action_smoothing_alpha > 0:
            env = ExponentialSmoothingWrapper(env, alpha=action_smoothing_alpha)
        elif action_rate_limit > 0:
            env = ActionRateLimiter(env, max_rate=action_rate_limit)

        # Add previous action to observations if configured
        include_prev_action = getattr(
            self.config.physics, "include_previous_action", False
        )
        if include_prev_action:
            env = PreviousActionWrapper(env)

        return env

    def run_episode(
        self, episode_id: int, deterministic: bool = True
    ) -> SpinEpisodeData:
        """Run single evaluation episode"""
        base_env = self.create_env()

        # Wrap in DummyVecEnv for VecNormalize compatibility
        vec_env = DummyVecEnv([lambda: base_env])

        # Apply VecNormalize if stats are available
        if self.vec_normalize_path:
            vec_env = VecNormalize.load(self.vec_normalize_path, vec_env)
            vec_env.training = False  # Don't update stats during evaluation
            vec_env.norm_reward = False

        # Storage
        times, altitudes, velocities = [], [], []
        roll_rates, roll_angles = [], []
        actions, tab_deflections = [], []
        rewards, thrusts, dynamic_pressures = [], [], []
        disturbance_torques = []  # Atmospheric disturbance

        obs = vec_env.reset()
        # Get info from underlying env
        info = {}
        if hasattr(vec_env, "get_attr"):
            try:
                infos = vec_env.get_attr("_last_info")
                info = infos[0] if infos else {}
            except AttributeError:
                pass

        total_reward = 0.0
        step = 0

        while True:
            # Get action from agent
            action, _ = self.model.predict(obs, deterministic=deterministic)

            # Store pre-step data
            times.append(info.get("time_s", step * 0.01))
            altitudes.append(info.get("altitude_m", 0))
            velocities.append(info.get("vertical_velocity_ms", 0))
            roll_rates.append(info.get("roll_rate_deg_s", 0))
            roll_angles.append(
                obs[0][2] if len(obs[0]) > 2 else 0
            )  # roll angle from obs
            actions.append(action[0][0] if len(action.shape) > 1 else action[0])
            tab_deflections.append(info.get("tab_deflection_deg", 0))
            thrusts.append(info.get("current_thrust_N", 0))

            # Calculate dynamic pressure
            v = max(info.get("vertical_velocity_ms", 0), 0)
            alt = info.get("altitude_m", 0)
            rho = 1.225 * np.exp(-alt / 8000)
            dynamic_pressures.append(0.5 * rho * v**2)

            # Track atmospheric disturbance
            disturbance_torques.append(info.get("disturbance_torque_Nm", 0))

            # Take step
            obs, reward_arr, done_arr, infos = vec_env.step(action)
            reward = reward_arr[0]
            done = done_arr[0]
            info = infos[0] if infos else {}

            rewards.append(reward)
            total_reward += reward
            step += 1

            if done:
                break

        vec_env.close()

        # Convert to arrays
        times = np.array(times)
        altitudes = np.array(altitudes)
        roll_rates = np.array(roll_rates)

        # Calculate metrics
        max_altitude = info.get("max_altitude_m", np.max(altitudes))
        final_spin = abs(
            info.get("roll_rate_deg_s", roll_rates[-1] if len(roll_rates) > 0 else 0)
        )
        mean_spin = np.mean(np.abs(roll_rates))

        # Camera quality assessment
        if mean_spin < 10:
            camera_quality = "Excellent"
        elif mean_spin < 30:
            camera_quality = "Good"
        elif mean_spin < 60:
            camera_quality = "Fair"
        else:
            camera_quality = "Poor"

        # Success criteria
        target_altitude = self.config.environment.max_altitude if self.config else 100
        reached_altitude = max_altitude > target_altitude * 0.5
        low_spin = mean_spin < 30
        success = reached_altitude and low_spin

        return SpinEpisodeData(
            episode_id=episode_id,
            time=times,
            altitude=altitudes,
            velocity=np.array(velocities),
            roll_rate=roll_rates,
            roll_angle=np.array(roll_angles),
            actions=np.array(actions),
            tab_deflection=np.array(tab_deflections),
            rewards=np.array(rewards),
            thrust=np.array(thrusts),
            dynamic_pressure=np.array(dynamic_pressures),
            disturbance_torque=np.array(disturbance_torques),
            max_altitude=max_altitude,
            final_spin_rate=final_spin,
            mean_spin_rate=mean_spin,
            total_reward=total_reward,
            episode_length=step,
            camera_quality=camera_quality,
            reached_target_altitude=reached_altitude,
            maintained_low_spin=low_spin,
            success=success,
        )

    def evaluate(
        self, n_episodes: int = 100, deterministic: bool = True
    ) -> List[SpinEpisodeData]:
        """Run multiple evaluation episodes"""
        print(f"Evaluating agent over {n_episodes} episodes...")

        episodes = []
        success_count = 0

        for i in range(n_episodes):
            episode = self.run_episode(i, deterministic=deterministic)
            episodes.append(episode)

            if episode.success:
                success_count += 1

            if (i + 1) % 10 == 0:
                print(
                    f"  Completed {i+1}/{n_episodes} episodes. "
                    f"Success rate: {success_count/(i+1)*100:.1f}%"
                )

        print(f"\nEvaluation complete!")
        print(f"Success rate: {success_count/n_episodes*100:.1f}%")

        return episodes


class SpinVisualizationPlotter:
    """Creates visualizations for spin control agent"""

    def __init__(self, episodes: List[SpinEpisodeData], config: dict = None):
        self.episodes = episodes
        self.config = config
        self.n_episodes = len(episodes)

        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("husl")

    def plot_performance_overview(self, save_path: str = None, show: bool = True):
        """Create overview of agent performance"""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Spin Control Agent Performance ({self.n_episodes} Episodes)",
            fontsize=14,
            fontweight="bold",
        )

        # Extract metrics
        altitudes = [ep.max_altitude for ep in self.episodes]
        mean_spins = [ep.mean_spin_rate for ep in self.episodes]
        final_spins = [ep.final_spin_rate for ep in self.episodes]
        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.episode_length for ep in self.episodes]
        successes = [ep.success for ep in self.episodes]

        # 1. Altitude distribution
        ax = axes[0, 0]
        ax.hist(altitudes, bins=25, alpha=0.7, edgecolor="black", color="steelblue")
        ax.axvline(
            np.mean(altitudes),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(altitudes):.1f}m",
        )
        ax.set_xlabel("Max Altitude (m)")
        ax.set_ylabel("Frequency")
        ax.set_title("Altitude Distribution")
        ax.legend()

        # 2. Spin rate distribution
        ax = axes[0, 1]
        ax.hist(mean_spins, bins=25, alpha=0.7, edgecolor="black", color="coral")
        ax.axvline(30, color="green", linestyle="--", label="Good threshold (30°/s)")
        ax.axvline(
            np.mean(mean_spins),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(mean_spins):.1f}°/s",
        )
        ax.set_xlabel("Mean Spin Rate (°/s)")
        ax.set_ylabel("Frequency")
        ax.set_title("Spin Rate Distribution")
        ax.legend()

        # 3. Camera quality pie chart
        ax = axes[0, 2]
        qualities = [ep.camera_quality for ep in self.episodes]
        quality_counts = {
            q: qualities.count(q) for q in ["Excellent", "Good", "Fair", "Poor"]
        }
        colors = {
            "Excellent": "green",
            "Good": "lightgreen",
            "Fair": "orange",
            "Poor": "red",
        }

        labels = [k for k, v in quality_counts.items() if v > 0]
        sizes = [quality_counts[k] for k in labels]
        pie_colors = [colors[k] for k in labels]

        ax.pie(
            sizes, labels=labels, colors=pie_colors, autopct="%1.1f%%", startangle=90
        )
        ax.set_title("Camera Quality Distribution")

        # 4. Rewards distribution
        ax = axes[1, 0]
        ax.hist(rewards, bins=25, alpha=0.7, edgecolor="black", color="mediumpurple")
        ax.axvline(
            np.mean(rewards),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(rewards):.1f}",
        )
        ax.set_xlabel("Total Episode Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Reward Distribution")
        ax.legend()

        # 5. Success rate
        ax = axes[1, 1]
        success_count = sum(successes)
        fail_count = len(successes) - success_count
        ax.bar(
            ["Successful", "Failed"],
            [success_count, fail_count],
            color=["green", "red"],
            alpha=0.7,
            edgecolor="black",
        )
        ax.set_ylabel("Count")
        ax.set_title(
            f"Mission Outcomes (Success Rate: {success_count/len(successes)*100:.1f}%)"
        )

        # 6. Altitude vs Spin scatter
        ax = axes[1, 2]
        colors = ["green" if s else "red" for s in successes]
        ax.scatter(altitudes, mean_spins, c=colors, alpha=0.6, s=30)
        ax.axhline(30, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Max Altitude (m)")
        ax.set_ylabel("Mean Spin Rate (°/s)")
        ax.set_title("Altitude vs Spin Rate")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def plot_best_trajectory(self, save_path: str = None, show: bool = True):
        """Plot the best performing trajectory in detail"""
        # Find best episode (highest altitude with lowest spin)
        scores = [ep.max_altitude / (ep.mean_spin_rate + 1) for ep in self.episodes]
        best_idx = np.argmax(scores)
        best = self.episodes[best_idx]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle(
            f"Best Flight Trajectory (Episode {best.episode_id})\n"
            f"Altitude: {best.max_altitude:.1f}m | Mean Spin: {best.mean_spin_rate:.1f}°/s | "
            f"Camera: {best.camera_quality}",
            fontsize=12,
            fontweight="bold",
        )

        t = best.time

        # 1. Altitude profile
        ax = axes[0, 0]
        ax.plot(t, best.altitude, "b-", linewidth=2)
        ax.fill_between(t, best.altitude, alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Altitude (m)")
        ax.set_title("Altitude Profile")
        ax.grid(True, alpha=0.3)

        # Mark motor burnout
        if self.config:
            burn_time = 1.85  # Default, could get from motor
            ax.axvline(
                burn_time,
                color="orange",
                linestyle="--",
                alpha=0.7,
                label="Motor burnout",
            )
            ax.legend()

        # 2. Velocity profile
        ax = axes[0, 1]
        ax.plot(t, best.velocity, "g-", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vertical Velocity (m/s)")
        ax.set_title("Velocity Profile")
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)

        # 3. Roll rate
        ax = axes[0, 2]
        ax.plot(t, best.roll_rate, "r-", linewidth=1.5)
        ax.fill_between(t, best.roll_rate, alpha=0.3, color="red")
        ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
        ax.axhline(30, color="green", linestyle="--", alpha=0.5, label="Good threshold")
        ax.axhline(-30, color="green", linestyle="--", alpha=0.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Roll Rate (°/s)")
        ax.set_title("Spin Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Control actions
        ax = axes[1, 0]
        ax.plot(t, best.actions, "purple", linewidth=1.5, label="Normalized action")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control Input")
        ax.set_title("Control Actions")
        ax.set_ylim(-1.1, 1.1)
        ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
        ax.grid(True, alpha=0.3)

        # 5. Thrust and dynamic pressure
        ax = axes[1, 1]
        ax2 = ax.twinx()

        (l1,) = ax.plot(t, best.thrust, "orange", linewidth=2, label="Thrust")
        (l2,) = ax2.plot(
            t, best.dynamic_pressure, "blue", linewidth=2, label="Dynamic Pressure"
        )

        # Add disturbance torque if available
        if best.disturbance_torque is not None and len(best.disturbance_torque) > 0:
            # Scale disturbance for visibility (multiply by 1000 to show mN·m)
            disturbance_scaled = np.array(best.disturbance_torque) * 1000
            (l3,) = ax2.plot(
                t[: len(disturbance_scaled)],
                disturbance_scaled,
                "red",
                linewidth=1,
                alpha=0.7,
                label="Disturbance (mN·m)",
            )
            ax.legend(handles=[l1, l2, l3], loc="upper right")
        else:
            ax.legend(handles=[l1, l2], loc="upper right")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thrust (N)", color="orange")
        ax2.set_ylabel("Dynamic Pressure (Pa) / Disturbance (mN·m)", color="blue")
        ax.set_title("Propulsion & Aerodynamics")
        ax.grid(True, alpha=0.3)

        # 6. Cumulative reward
        ax = axes[1, 2]
        cumulative_reward = np.cumsum(best.rewards)
        ax.plot(t, cumulative_reward, "green", linewidth=2)
        ax.fill_between(t, cumulative_reward, alpha=0.3, color="green")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(f"Reward Accumulation (Total: {best.total_reward:.1f})")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig, best

    def plot_trajectory_comparison(
        self, n_trajectories: int = 10, save_path: str = None, show: bool = True
    ):
        """Compare multiple trajectories (best vs worst)"""
        # Sort by success score
        scored = [
            (i, ep.max_altitude / (ep.mean_spin_rate + 1))
            for i, ep in enumerate(self.episodes)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_indices = [s[0] for s in scored[: n_trajectories // 2]]
        worst_indices = [s[0] for s in scored[-n_trajectories // 2 :]]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Trajectory Comparison (Best {n_trajectories//2} vs Worst {n_trajectories//2})",
            fontsize=14,
            fontweight="bold",
        )

        # Plot best trajectories
        for idx in best_indices:
            ep = self.episodes[idx]
            axes[0, 0].plot(ep.time, ep.altitude, "g-", alpha=0.6, linewidth=1)
            axes[0, 1].plot(ep.time, ep.roll_rate, "g-", alpha=0.6, linewidth=1)

        # Plot worst trajectories
        for idx in worst_indices:
            ep = self.episodes[idx]
            axes[0, 0].plot(ep.time, ep.altitude, "r-", alpha=0.6, linewidth=1)
            axes[0, 1].plot(ep.time, ep.roll_rate, "r-", alpha=0.6, linewidth=1)

        # Format
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Altitude (m)")
        axes[0, 0].set_title("Altitude Trajectories")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Roll Rate (°/s)")
        axes[0, 1].set_title("Spin Rate Trajectories")
        axes[0, 1].axhline(0, color="gray", linestyle="-", alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)

        # Control comparison
        for idx in best_indices:
            ep = self.episodes[idx]
            axes[1, 0].plot(ep.time, ep.actions, "g-", alpha=0.6, linewidth=1)
        for idx in worst_indices:
            ep = self.episodes[idx]
            axes[1, 0].plot(ep.time, ep.actions, "r-", alpha=0.6, linewidth=1)

        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Control Action")
        axes[1, 0].set_title("Control Strategies")
        axes[1, 0].grid(True, alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="green", label="Best performing"),
            Line2D([0], [0], color="red", label="Worst performing"),
        ]
        axes[1, 1].legend(handles=legend_elements, loc="center", fontsize=12)
        axes[1, 1].axis("off")

        # Add statistics text
        best_eps = [self.episodes[i] for i in best_indices]
        worst_eps = [self.episodes[i] for i in worst_indices]

        stats_text = f"""
        BEST TRAJECTORIES
        ─────────────────────
        Mean Altitude: {np.mean([e.max_altitude for e in best_eps]):.1f} m
        Mean Spin Rate: {np.mean([e.mean_spin_rate for e in best_eps]):.1f} °/s
        Mean Reward: {np.mean([e.total_reward for e in best_eps]):.1f}

        WORST TRAJECTORIES
        ─────────────────────
        Mean Altitude: {np.mean([e.max_altitude for e in worst_eps]):.1f} m
        Mean Spin Rate: {np.mean([e.mean_spin_rate for e in worst_eps]):.1f} °/s
        Mean Reward: {np.mean([e.total_reward for e in worst_eps]):.1f}
        """

        axes[1, 1].text(
            0.5,
            0.3,
            stats_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        if show:
            plt.show()

        return fig

    def generate_report(self, save_dir: str):
        """Generate comprehensive evaluation report"""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\nGenerating evaluation report in: {save_dir}")

        # Generate all plots
        self.plot_performance_overview(
            save_path=os.path.join(save_dir, f"performance_overview_{timestamp}.png"),
            show=False,
        )

        self.plot_best_trajectory(
            save_path=os.path.join(save_dir, f"best_trajectory_{timestamp}.png"),
            show=False,
        )

        self.plot_trajectory_comparison(
            save_path=os.path.join(save_dir, f"trajectory_comparison_{timestamp}.png"),
            show=False,
        )

        # Generate text report
        self._save_text_report(save_dir, timestamp)

        print(f"Report complete! Files saved to: {save_dir}")

    def _save_text_report(self, save_dir: str, timestamp: str):
        """Save text summary report"""
        altitudes = [ep.max_altitude for ep in self.episodes]
        spins = [ep.mean_spin_rate for ep in self.episodes]
        rewards = [ep.total_reward for ep in self.episodes]
        successes = [ep.success for ep in self.episodes]

        report = f"""
SPIN CONTROL AGENT EVALUATION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Episodes Evaluated: {self.n_episodes}

════════════════════════════════════════════════════════════════

MISSION SUCCESS
────────────────────────────────────────────────────────────────
Success Rate:           {sum(successes)/len(successes)*100:.1f}%
Successful Missions:    {sum(successes)}/{len(successes)}

ALTITUDE PERFORMANCE
────────────────────────────────────────────────────────────────
Mean Max Altitude:      {np.mean(altitudes):.1f} ± {np.std(altitudes):.1f} m
Maximum Achieved:       {np.max(altitudes):.1f} m
Minimum Achieved:       {np.min(altitudes):.1f} m

SPIN CONTROL PERFORMANCE
────────────────────────────────────────────────────────────────
Mean Spin Rate:         {np.mean(spins):.1f} ± {np.std(spins):.1f} °/s
Best (lowest) Spin:     {np.min(spins):.1f} °/s
Worst (highest) Spin:   {np.max(spins):.1f} °/s

Episodes with <10°/s:   {sum(s < 10 for s in spins)/len(spins)*100:.1f}%
Episodes with <30°/s:   {sum(s < 30 for s in spins)/len(spins)*100:.1f}%
Episodes with <60°/s:   {sum(s < 60 for s in spins)/len(spins)*100:.1f}%

CAMERA QUALITY
────────────────────────────────────────────────────────────────
Excellent (<10°/s):     {sum(ep.camera_quality == 'Excellent' for ep in self.episodes)/len(self.episodes)*100:.1f}%
Good (<30°/s):          {sum(ep.camera_quality == 'Good' for ep in self.episodes)/len(self.episodes)*100:.1f}%
Fair (<60°/s):          {sum(ep.camera_quality == 'Fair' for ep in self.episodes)/len(self.episodes)*100:.1f}%
Poor (≥60°/s):          {sum(ep.camera_quality == 'Poor' for ep in self.episodes)/len(self.episodes)*100:.1f}%

REWARD PERFORMANCE
────────────────────────────────────────────────────────────────
Mean Episode Reward:    {np.mean(rewards):.1f} ± {np.std(rewards):.1f}
Best Episode Reward:    {np.max(rewards):.1f}
Worst Episode Reward:   {np.min(rewards):.1f}

════════════════════════════════════════════════════════════════
"""

        with open(
            os.path.join(save_dir, f"evaluation_report_{timestamp}.txt"), "w"
        ) as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize spin control agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic evaluation
    uv run python visualize_spin_agent.py models/best_model.zip

    # With config file
    uv run python visualize_spin_agent.py models/best_model.zip --config configs/estes_c6_easy.yaml

    # Full evaluation with saved report
    uv run python visualize_spin_agent.py models/best_model.zip \\
        --config configs/estes_c6_easy.yaml \\
        --n-episodes 100 \\
        --save-dir evaluation_results/
        """,
    )

    parser.add_argument("model_path", type=str, help="Path to trained PPO model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--n-episodes", type=int, default=50, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--save-dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plots")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = SpinAgentEvaluator(args.model_path, args.config)
    episodes = evaluator.evaluate(args.n_episodes, deterministic=args.deterministic)

    # Load config for plotter
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    # Create visualizations
    plotter = SpinVisualizationPlotter(episodes, config)

    if args.save_dir:
        plotter.generate_report(args.save_dir)
    else:
        plotter.plot_performance_overview(show=not args.no_show)
        plotter.plot_best_trajectory(show=not args.no_show)


if __name__ == "__main__":
    main()
