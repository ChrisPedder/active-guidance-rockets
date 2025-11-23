#!/usr/bin/env python3
# train_ppo.py
"""
Unified PPO Training Script for Rocket Control Environments

Supports both:
- Original 6DOF rocket boost control
- Spin-stabilized camera rocket control
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
import argparse
from datetime import datetime
from dataclasses import dataclass

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

from rocket_boost_control_env import RocketBoostControlEnv
from spin_stabilized_control_env import CompositeRocketConfig, SpinStabilizedCameraRocket
from thrustcurve_motor_data import (
    ThrustCurveParser
)
from realistic_spin_rocket import RealisticMotorRocket, CommonMotors


class UnifiedMetricsCallback(BaseCallback):
    """
    Unified callback that tracks appropriate metrics for each environment type.
    """

    def __init__(self, env_type: str = "original", log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.env_type = env_type
        self.log_freq = log_freq
        self.episode_count = 0

        # Common metrics
        self.episode_rewards = []
        self.episode_lengths = []

        # Environment-specific metrics
        if env_type == "original":
            self.episode_altitudes = []
            self.episode_attitude_devs = []
        elif env_type == "spin":
            self.episode_roll_rates = []
            self.episode_camera_qualities = []
            self.episode_altitudes = []

    def _on_step(self) -> bool:
        """Track metrics at each step"""
        for i, done in enumerate(self.locals.get("dones", [])):
            if done and "infos" in self.locals and i < len(self.locals["infos"]):
                info = self.locals["infos"][i]
                if info:
                    self.episode_count += 1

                    # Common metrics
                    if "episode" in info:
                        ep_info = info["episode"]
                        self.episode_rewards.append(ep_info.get("r", 0))
                        self.episode_lengths.append(ep_info.get("l", 0))

                    # Environment-specific metrics
                    if self.env_type == "original":
                        self.episode_altitudes.append(info.get("altitude", 0))
                        if "attitude_degrees" in info:
                            max_attitude = np.max(np.abs(info["attitude_degrees"]))
                            self.episode_attitude_devs.append(max_attitude)

                    elif self.env_type == "spin":
                        self.episode_altitudes.append(info.get("altitude_m", 0))
                        self.episode_roll_rates.append(abs(info.get("roll_rate_deg_s", 0)))

                        # Extract camera quality
                        h_quality = info.get("horizontal_camera_quality", "")
                        if "Excellent" in h_quality:
                            quality_score = 4
                        elif "Good" in h_quality:
                            quality_score = 3
                        elif "Fair" in h_quality:
                            quality_score = 2
                        else:
                            quality_score = 1
                        self.episode_camera_qualities.append(quality_score)

                    # Log progress
                    if self.episode_count % self.log_freq == 0:
                        self._print_progress()

        return True

    def _print_progress(self):
        """Print training progress based on environment type"""
        print(f"\n{'='*60}")
        print(f"Training Progress - Episode {self.episode_count}")
        print(f"Environment: {self.env_type.upper()}")
        print(f"{'='*60}")

        # Recent episodes for statistics
        recent_rewards = self.episode_rewards[-self.log_freq:] if self.episode_rewards else [0]

        print(f"Reward Stats (last {self.log_freq} episodes):")
        print(f"  Mean: {np.mean(recent_rewards):.1f}")
        print(f"  Max:  {np.max(recent_rewards):.1f}")
        print(f"  Min:  {np.min(recent_rewards):.1f}")

        if self.env_type == "original":
            if self.episode_altitudes:
                recent_altitudes = self.episode_altitudes[-self.log_freq:]
                print(f"\nAltitude Stats:")
                print(f"  Mean: {np.mean(recent_altitudes):.1f} m")
                print(f"  Max:  {np.max(recent_altitudes):.1f} m")
                print(f"  Success (>2500m): {sum(a > 2500 for a in recent_altitudes)}/{len(recent_altitudes)}")

            if self.episode_attitude_devs:
                recent_attitudes = self.episode_attitude_devs[-self.log_freq:]
                print(f"\nAttitude Control:")
                print(f"  Mean Max Deviation: {np.mean(recent_attitudes):.1f}°")
                print(f"  Good Control (<30°): {sum(a < 30 for a in recent_attitudes)}/{len(recent_attitudes)}")

        elif self.env_type == "spin":
            if self.episode_altitudes:
                recent_altitudes = self.episode_altitudes[-self.log_freq:]
                print(f"\nAltitude Stats:")
                print(f"  Mean: {np.mean(recent_altitudes):.1f} m")
                print(f"  Max:  {np.max(recent_altitudes):.1f} m")

            if self.episode_roll_rates:
                recent_roll_rates = self.episode_roll_rates[-self.log_freq:]
                print(f"\nRoll Control:")
                print(f"  Mean Rate: {np.mean(recent_roll_rates):.1f}°/s")
                print(f"  Stable (<30°/s): {sum(r < 30 for r in recent_roll_rates)}/{len(recent_roll_rates)}")
                print(f"  Excellent (<10°/s): {sum(r < 10 for r in recent_roll_rates)}/{len(recent_roll_rates)}")

            if self.episode_camera_qualities:
                recent_qualities = self.episode_camera_qualities[-self.log_freq:]
                avg_quality = np.mean(recent_qualities)
                print(f"\nCamera Quality:")
                print(f"  Average Score: {avg_quality:.2f}/4.0")
                print(f"  Excellent: {recent_qualities.count(4)}/{len(recent_qualities)}")
                print(f"  Good: {recent_qualities.count(3)}/{len(recent_qualities)}")

        print(f"{'='*60}\n")


def create_environment(env_type: str = "original", config: Dict[str, Any] = None,
                       motor_name: str = None) -> gym.Env:
    """
    Create the specified environment type

    Args:
        env_type: "original" or "spin"
        config: Environment configuration
        motor_name: Motor name for spin environment (e.g., "estes_c6")
    """

    if env_type == "original":
        if RocketBoostControlEnv is None:
            raise ImportError("RocketBoostControlEnv not available")

        if config is None:
            config = {
                'max_episode_steps': 1000,
                'dt': 0.02,
                'max_wind_speed': 20.0,
                'max_gust_speed': 10.0,
                'max_flap_angle': 30.0,
                'target_altitude': 3000.0,
            }
        return RocketBoostControlEnv(config)

    elif env_type == "spin":
        if SpinStabilizedCameraRocket is None:
            raise ImportError("SpinStabilizedCameraRocket not available")

        # Create rocket configuration
        rocket_config = CompositeRocketConfig(
            dry_mass=1.5,           # 1.5 kg rocket
            propellant_mass=0.2,    # Will be overridden by motor
            diameter=0.054,         # 54mm diameter
            length=0.75,            # 75cm length
            num_fins=4,
            fin_span=0.06,
            max_tab_deflection=15.0,
            tab_chord_fraction=0.25,
            tab_span_fraction=0.5,
        )

        # Select motor
        if motor_name and RealisticMotorRocket is not None:
            # Use motor-based environment
            if motor_name == "estes_c6":
                motor = CommonMotors.estes_c6()
            elif motor_name == "aerotech_f40":
                motor = CommonMotors.aerotech_f40()
            elif motor_name == "cesaroni_g79":
                motor = CommonMotors.cesaroni_g79()
            else:
                # Try to parse as file path
                try:
                    if motor_name.endswith('.eng'):
                        motor = ThrustCurveParser.parse_eng_file(motor_name)
                    else:
                        # Assume it's a motor ID for download
                        motor = ThrustCurveParser.download_from_thrustcurve(motor_name, "eng")
                except Exception as e:
                    print(f"Warning: Could not load motor {motor_name}: {e}")
                    print("Falling back to default Estes C6")
                    motor = CommonMotors.estes_c6()

            return RealisticMotorRocket(motor, rocket_config)
        else:
            # Use simplified constant thrust environment
            return SpinStabilizedCameraRocket(rocket_config)

    else:
        raise ValueError(f"Unknown environment type: {env_type}")


def train_unified_agent(
    env_type: str = "original",
    motor_name: str = None,
    total_timesteps: int = 500000,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    device: str = "auto",
) -> PPO:
    """
    Train PPO agent on specified rocket control task
    """

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp and environment-specific naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    env_name = f"{env_type}_rocket"
    if motor_name:
        env_name += f"_{motor_name.replace('/', '_')}"

    run_log_dir = os.path.join(log_dir, f"{env_name}_{timestamp}")

    print(f"\n{'='*60}")
    print(f"UNIFIED PPO ROCKET CONTROL TRAINING")
    print(f"{'='*60}")
    print(f"Environment Type: {env_type.upper()}")
    if motor_name:
        print(f"Motor Configuration: {motor_name}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {device}")
    print(f"Parallel environments: {n_envs}")
    print(f"Model save path: {save_dir}/{env_name}_{timestamp}")
    print(f"Tensorboard logs: {run_log_dir}")
    print(f"{'='*60}\n")

    # Create vectorized training environment
    def make_env():
        env = create_environment(env_type, motor_name=motor_name)
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)

    # Create evaluation environment
    eval_env = Monitor(create_environment(env_type, motor_name=motor_name))

    # Configure PPO model - adjust hyperparameters based on environment
    if env_type == "spin":
        # Spin control needs more precise control
        policy_kwargs = dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Smaller networks
            activation_fn=torch.nn.Tanh,  # Smoother activation for control
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,  # Lower learning rate for stability
            n_steps=512,  # Shorter rollouts for faster feedback
            batch_size=32,
            n_epochs=50,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.1,  # Smaller clip range for finer control
            normalize_advantage=True,
            ent_coef=0.001,  # Less exploration needed
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=run_log_dir
        )
    else:
        # Original environment - more complex dynamics
        policy_kwargs = dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=torch.nn.ReLU,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=device,
            tensorboard_log=run_log_dir
        )

    # Setup callbacks
    callbacks = []

    # Add metrics tracking callback
    metrics_callback = UnifiedMetricsCallback(env_type=env_type, log_freq=100, verbose=1)
    callbacks.append(metrics_callback)

    # Set appropriate reward threshold based on environment
    if env_type == "spin":
        reward_threshold = 500.0  # Lower threshold for spin control
    else:
        reward_threshold = 1000.0  # Higher threshold for full control

    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=reward_threshold,
        verbose=1
    )

    # Add evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"{env_name}_{timestamp}"),
        log_path=os.path.join(save_dir, f"eval_{env_name}_{timestamp}"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=reward_threshold_callback
    )
    callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks)

    # Train the model
    print(f"Starting training for {env_type} environment...")
    print("(Progress will be shown every 100 episodes)\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=f"{env_name}_training"
    )

    # Save final model
    final_model_path = os.path.join(save_dir, f"{env_name}_{timestamp}_final.zip")
    model.save(final_model_path)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED!")
    print(f"{'='*60}")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {os.path.join(save_dir, f'{env_name}_{timestamp}')}/best_model.zip")

    # Final evaluation
    print(f"\nRunning final evaluation...")
    evaluate_agent(model, env_type, motor_name, n_episodes=20)

    # Close environments
    train_env.close()
    eval_env.close()

    return model


def evaluate_agent(model: PPO, env_type: str, motor_name: str = None,
                  n_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluate trained agent performance
    """
    env = create_environment(env_type, motor_name=motor_name)

    episode_rewards = []

    if env_type == "original":
        final_altitudes = []
        attitude_deviations = []
        successful_flights = 0

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            max_attitude_dev = 0

            for step in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if 'attitude_degrees' in info:
                    attitude_dev = np.max(np.abs(info['attitude_degrees']))
                    max_attitude_dev = max(max_attitude_dev, attitude_dev)

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            final_altitudes.append(info['altitude'])
            attitude_deviations.append(max_attitude_dev)

            if info['altitude'] > 2500 and max_attitude_dev < 30:
                successful_flights += 1

        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - ORIGINAL ENVIRONMENT")
        print(f"{'='*60}")
        print(f"Mean Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"Mean Altitude: {np.mean(final_altitudes):.1f} ± {np.std(final_altitudes):.1f} m")
        print(f"Mean Attitude Deviation: {np.mean(attitude_deviations):.1f}°")
        print(f"Success Rate: {successful_flights/n_episodes*100:.1f}%")

    elif env_type == "spin":
        final_altitudes = []
        avg_roll_rates = []
        camera_scores = []

        for episode in range(n_episodes):
            obs, info = env.reset()
            episode_reward = 0
            roll_rates = []

            for step in range(1000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if 'roll_rate_deg_s' in info:
                    roll_rates.append(abs(info['roll_rate_deg_s']))

                if terminated or truncated:
                    break

            episode_rewards.append(episode_reward)
            final_altitudes.append(info.get('altitude_m', 0))
            avg_roll_rates.append(np.mean(roll_rates) if roll_rates else 0)

            # Score camera quality
            h_quality = info.get('horizontal_camera_quality', "")
            if "Excellent" in h_quality:
                camera_scores.append(4)
            elif "Good" in h_quality:
                camera_scores.append(3)
            elif "Fair" in h_quality:
                camera_scores.append(2)
            else:
                camera_scores.append(1)

        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS - SPIN CONTROL ENVIRONMENT")
        if motor_name:
            print(f"Motor: {motor_name}")
        print(f"{'='*60}")
        print(f"Mean Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
        print(f"Mean Altitude: {np.mean(final_altitudes):.1f} ± {np.std(final_altitudes):.1f} m")
        print(f"Mean Roll Rate: {np.mean(avg_roll_rates):.1f} ± {np.std(avg_roll_rates):.1f}°/s")
        print(f"Camera Quality Score: {np.mean(camera_scores):.2f}/4.0")
        print(f"Excellent Footage: {camera_scores.count(4)/len(camera_scores)*100:.1f}%")
        print(f"Good or Better: {(camera_scores.count(4) + camera_scores.count(3))/len(camera_scores)*100:.1f}%")

    print(f"{'='*60}\n")

    env.close()

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards)
    }


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for rocket control")

    # Environment selection
    parser.add_argument("--env", type=str, default="original",
                       choices=["original", "spin"],
                       help="Environment type: 'original' (6DOF) or 'spin' (camera stabilization)")

    # Motor selection for spin environment
    parser.add_argument("--motor", type=str, default=None,
                       help="Motor for spin env: 'estes_c6', 'aerotech_f40', 'cesaroni_g79', or path to .eng file")

    # Training parameters
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="models",
                       help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs",
                       help="Directory for logs")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n-envs", type=int, default=4,
                       help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device for training")

    # Evaluation mode
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate existing model")
    parser.add_argument("--model-path", type=str,
                       help="Path to model for evaluation")

    args = parser.parse_args()

    if args.eval_only:
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            return

        print(f"Loading model from: {args.model_path}")

        if not args.model_path.endswith('.zip'):
            if os.path.exists(args.model_path + '.zip'):
                args.model_path = args.model_path + '.zip'

        model = PPO.load(args.model_path)

        print(f"Evaluating agent on {args.env} environment...")
        evaluate_agent(model, args.env, args.motor, n_episodes=20)

    else:
        # Train new model
        model = train_unified_agent(
            env_type=args.env,
            motor_name=args.motor,
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            device=args.device
        )

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE - NEXT STEPS:")
        print(f"{'='*60}")
        print(f"1. View training progress in TensorBoard:")
        print(f"   tensorboard --logdir {args.log_dir}")
        print(f"\n2. Evaluate your trained model:")
        print(f"   python train_unified.py --eval-only --model-path models/[your_model].zip --env {args.env}")
        if args.motor:
            print(f"   (Use --motor {args.motor} for correct environment)")
        print(f"\n3. The best model was saved based on evaluation performance")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
