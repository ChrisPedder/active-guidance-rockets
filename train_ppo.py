#!/usr/bin/env python3
"""
PPO Training Script for Rocket Boost Control Environment

This script trains a PPO agent to control rocket directional stability
during the boost phase using Stable-Baselines3.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
import argparse
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import torch

# Import your environment
from rocket_boost_control_env import RocketBoostControlEnv


class TrainingCallback:
    """Custom callback to track training progress"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.rewards = []
        self.episode_lengths = []
        self.altitudes = []
        self.attitude_deviations = []

    def __call__(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]) -> bool:
        # This is called during training
        if 'infos' in locals_dict:
            for info in locals_dict['infos']:
                if info:
                    self.altitudes.append(info.get('altitude', 0))
                    if 'attitude_degrees' in info:
                        attitude_dev = np.max(np.abs(info['attitude_degrees']))
                        self.attitude_deviations.append(attitude_dev)
        return True


def create_environment(config: Dict[str, Any] = None) -> gym.Env:
    """Create and configure the rocket environment"""
    if config is None:
        config = {
            'max_episode_steps': 1000,
            'dt': 0.02,
            'max_wind_speed': 20.0,
            'max_gust_speed': 10.0,
            'max_flap_angle': 30.0,
            'target_altitude': 3000.0,
        }

    env = RocketBoostControlEnv(config)
    return env


def train_ppo_agent(
    total_timesteps: int = 500000,
    save_dir: str = "models",
    log_dir: str = "logs",
    eval_freq: int = 10000,
    n_eval_episodes: int = 10,
    learning_rate: float = 3e-4,
    n_envs: int = 4,
    device: str = "auto"
) -> PPO:
    """
    Train PPO agent on rocket control task

    Args:
        total_timesteps: Total training timesteps
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation
        n_eval_episodes: Number of episodes for evaluation
        learning_rate: PPO learning rate
        n_envs: Number of parallel environments
        device: Device for training ('cpu', 'cuda', or 'auto')

    Returns:
        Trained PPO model
    """

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, f"ppo_rocket_{timestamp}")

    print(f"Training PPO agent for {total_timesteps:,} timesteps")
    print(f"Using device: {device}")
    print(f"Parallel environments: {n_envs}")
    print(f"Logs will be saved to: {run_log_dir}")

    # Create vectorized training environment
    def make_env():
        env = create_environment()
        env = Monitor(env)
        return env

    train_env = make_vec_env(make_env, n_envs=n_envs)

    # Create evaluation environment
    eval_env = Monitor(create_environment())

    # Configure PPO model
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
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
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, f"best_model_{timestamp}"),
        log_path=os.path.join(save_dir, f"eval_logs_{timestamp}"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    # Stop training if reward threshold is reached
    reward_threshold_callback = StopTrainingOnRewardThreshold(
        reward_threshold=1000.0,  # Adjust based on your reward scale
        verbose=1
    )

    callback_list = CallbackList([eval_callback, reward_threshold_callback])

    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="ppo_rocket_training"
    )

    # Save final model
    final_model_path = os.path.join(save_dir, f"final_model_{timestamp}")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    # Close environments
    train_env.close()
    eval_env.close()

    return model


def evaluate_agent(model: PPO, n_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluate trained agent performance

    Args:
        model: Trained PPO model
        n_episodes: Number of evaluation episodes

    Returns:
        Dictionary with evaluation metrics
    """
    env = create_environment()

    episode_rewards = []
    final_altitudes = []
    attitude_deviations = []
    successful_flights = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        max_attitude_dev = 0

        for step in range(1000):  # Max steps per episode
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Track maximum attitude deviation
            if 'attitude_degrees' in info:
                attitude_dev = np.max(np.abs(info['attitude_degrees']))
                max_attitude_dev = max(max_attitude_dev, attitude_dev)

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        final_altitudes.append(info['altitude'])
        attitude_deviations.append(max_attitude_dev)

        # Consider successful if reached high altitude with good control
        if info['altitude'] > 2500 and max_attitude_dev < 30:
            successful_flights += 1

        print(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
              f"Altitude={info['altitude']:.1f}m, "
              f"Max Attitude Dev={max_attitude_dev:.1f}°")

    env.close()

    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_altitude': np.mean(final_altitudes),
        'std_altitude': np.std(final_altitudes),
        'mean_attitude_deviation': np.mean(attitude_deviations),
        'success_rate': successful_flights / n_episodes
    }

    return metrics


def plot_training_progress(log_dir: str):
    """Plot training progress from tensorboard logs"""
    # This would require parsing tensorboard logs
    # For now, we'll create a placeholder
    print("To view training progress, run:")
    print(f"tensorboard --logdir {log_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent for rocket control")
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
        model = PPO.load(args.model_path)

        print("Evaluating agent...")
        metrics = evaluate_agent(model, n_episodes=20)

        print("\n=== Evaluation Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")

    else:
        # Train new model
        model = train_ppo_agent(
            total_timesteps=args.timesteps,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            learning_rate=args.learning_rate,
            n_envs=args.n_envs,
            device=args.device
        )

        print("\nTraining completed! Evaluating final model...")
        metrics = evaluate_agent(model, n_episodes=10)

        print("\n=== Final Evaluation Results ===")
        for key, value in metrics.items():
            print(f"{key}: {value:.3f}")

        print(f"\nTo view training progress:")
        print(f"tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    main()
