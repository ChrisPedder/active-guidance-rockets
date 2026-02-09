#!/usr/bin/env python3
"""
Residual SAC Training Script for Rocket Spin Control with Wind Disturbances

Architecture: PID handles base control (using ground truth from info dict),
SAC learns small wind-rejection corrections on top. Inherits PID's strong
zero-wind performance and focuses SAC on asymmetric wind disturbance rejection.

Wrapper chain (via create_environment):
    RealisticMotorRocket
      -> IMUObservationWrapper
      -> NormalizedActionWrapper
      -> ExponentialSmoothingWrapper(alpha from config, smooths RL output only)
      -> ResidualPIDWrapper(max_residual=0.2, PID gains from config)
      -> ImprovedRewardWrapper
      -> Monitor

Wind: Uniform random sampling from [0, base_speed] per episode.
    SAC learns both when to intervene (high wind) and when to stay quiet (low/no wind)
    throughout training.

Usage:
    # Train with uniform random wind
    uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml

    # Quick smoke test
    uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml --timesteps 10000

    # Override SAC hyperparameters
    uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml \\
        --timesteps 2000000 --early-stopping 30 \\
        --lr 0.0003 --buffer-size 300000 --batch-size 256
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
)

from rocket_config import RocketTrainingConfig, load_config
from training.train_improved import create_environment, TrainingMetricsCallback


class MovingAverageEarlyStoppingCallback(BaseCallback):
    """
    Early stopping based on moving average of eval rewards.

    More robust than SB3's StopTrainingOnNoModelImprovement which compares
    each eval against a single best value.  With high-variance evals (common
    in wind-disturbed rocket control), a single lucky eval sets an unreachable
    bar and triggers premature stopping.

    This callback tracks a rolling window of eval rewards and stops only when
    the moving average hasn't improved for ``max_no_improvement_evals`` evals.

    Used as ``callback_after_eval`` in EvalCallback -- accesses
    ``self.parent.last_mean_reward`` after each evaluation round.
    """

    def __init__(
        self,
        window_size: int = 20,
        max_no_improvement_evals: int = 40,
        min_evals: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.window_size = window_size
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals

        # State (reset on curriculum stage transitions via .reset())
        self.eval_rewards: list = []
        self.best_moving_avg: float = -np.inf
        self.no_improvement_evals: int = 0
        self.n_evals: int = 0

        # EMA tracking (reacts faster to improvements than windowed MA)
        self.ema_reward: float | None = None
        self.ema_alpha: float = 0.15  # ~7 eval half-life

    def _on_step(self) -> bool:
        """Called by EvalCallback after each evaluation round."""
        last_reward = self.parent.last_mean_reward
        self.eval_rewards.append(last_reward)
        self.n_evals += 1

        # Need a full window before tracking improvement
        if self.n_evals < self.min_evals:
            if self.verbose > 0:
                print(
                    f"  Moving avg early stop: warming up "
                    f"({self.n_evals}/{self.min_evals} evals)"
                )
            return True

        # Exponential moving average (reacts faster to improvements)
        if self.ema_reward is None:
            self.ema_reward = last_reward
        else:
            self.ema_reward = (
                self.ema_alpha * last_reward + (1 - self.ema_alpha) * self.ema_reward
            )
        current_avg = self.ema_reward

        if current_avg > self.best_moving_avg:
            self.best_moving_avg = current_avg
            self.no_improvement_evals = 0
        else:
            self.no_improvement_evals += 1

        if self.verbose > 0:
            print(
                f"  EMA reward: {current_avg:.1f} "
                f"(best: {self.best_moving_avg:.1f}, "
                f"no improvement: {self.no_improvement_evals}"
                f"/{self.max_no_improvement_evals})"
            )

        if self.no_improvement_evals >= self.max_no_improvement_evals:
            if self.verbose > 0:
                print(
                    f"Stopping training: EMA hasn't improved "
                    f"for {self.no_improvement_evals} evals "
                    f"(best EMA: {self.best_moving_avg:.1f})"
                )
            return False

        return True

    def reset(self):
        """Reset state for curriculum stage transitions."""
        self.eval_rewards.clear()
        self.best_moving_avg = -np.inf
        self.no_improvement_evals = 0
        self.n_evals = 0
        self.ema_reward = None


def train_residual_sac(
    config: RocketTrainingConfig,
    early_stopping_patience: int = 0,
    load_model_path: Optional[str] = None,
):
    """Main residual SAC training function.

    Args:
        config: Training configuration (must have sac section and use_residual_pid: true)
        early_stopping_patience: Stop if no improvement for N evals (0=disabled)
        load_model_path: Optional path to pre-trained model for fine-tuning
    """
    sac_cfg = config.sac
    if sac_cfg is None:
        print("ERROR: No 'sac' section in config. Add SAC hyperparameters.")
        return None

    # Verify residual PID is enabled
    if not getattr(config.physics, "use_residual_pid", False):
        print("WARNING: use_residual_pid is not enabled in config.")
        print("  This script is designed for residual SAC (PID + RL corrections).")
        print("  Set use_residual_pid: true in your config.")
        return None

    # Validate
    issues = config.validate()
    if issues:
        print("\nConfiguration Issues:")
        for issue in issues:
            print(f"  {issue}")
        critical = [i for i in issues if "CRITICAL" in i]
        if critical:
            print("\nCannot proceed with critical issues.")
            return None

    # Setup directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.logging.experiment_name}_{config.motor.name}_{timestamp}"
    log_dir = Path(config.logging.log_dir) / run_name
    save_dir = Path(config.logging.save_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    config.save(save_dir / "config.yaml")

    max_residual = getattr(config.physics, "max_residual", 0.2)
    pid_Kp = getattr(config.physics, "pid_Kp", 0.005208)
    pid_Ki = getattr(config.physics, "pid_Ki", 0.000324)
    pid_Kd = getattr(config.physics, "pid_Kd", 0.016524)

    print(f"\n{'='*70}")
    print("RESIDUAL SAC ROCKET SPIN CONTROL TRAINING")
    print(f"{'='*70}")
    print(f"Motor: {config.motor.name}")
    print(f"Algorithm: Residual SAC (PID base + RL corrections)")
    print(f"PID gains: Kp={pid_Kp}, Ki={pid_Ki}, Kd={pid_Kd}")
    print(f"Max residual: {max_residual}")
    print(f"Entropy coef: {sac_cfg.ent_coef}")
    print(f"Network: {sac_cfg.net_arch}")
    print(f"Wind: {'enabled' if config.physics.enable_wind else 'disabled'}")
    if config.physics.enable_wind:
        print(f"  Base wind: {config.physics.base_wind_speed:.1f} m/s")
        print(f"  Max gust: {config.physics.max_gust_speed:.1f} m/s")
    print(f"Total timesteps: {sac_cfg.total_timesteps:,}")
    print(f"Device: {sac_cfg.device}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*70}\n")

    # Create training environment (single env for SAC)
    def make_env():
        return create_environment(config)

    train_env = DummyVecEnv([make_env])

    # Observation normalization
    vec_normalize_loaded = False
    if load_model_path and config.environment.normalize_observations:
        load_path = Path(load_model_path)
        vec_normalize_path = load_path.parent / "vec_normalize.pkl"
        if vec_normalize_path.exists():
            print(f"Loading VecNormalize stats from: {vec_normalize_path}")
            train_env = VecNormalize.load(str(vec_normalize_path), train_env)
            train_env.training = True
            vec_normalize_loaded = True

    if config.environment.normalize_observations and not vec_normalize_loaded:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=config.environment.obs_clip_value,
        )

    # Create eval environment
    eval_env = DummyVecEnv([make_env])
    if config.environment.normalize_observations:
        if vec_normalize_loaded:
            eval_env = VecNormalize.load(
                str(Path(load_model_path).parent / "vec_normalize.pkl"), eval_env
            )
        else:
            eval_env = VecNormalize(
                eval_env,
                norm_obs=True,
                norm_reward=False,
            )
        eval_env.training = False

    # Create or load SAC model
    policy_kwargs = dict(
        net_arch=sac_cfg.net_arch,
    )

    if load_model_path:
        print(f"Loading pre-trained model: {load_model_path}")
        model = SAC.load(
            load_model_path,
            env=train_env,
            learning_rate=sac_cfg.learning_rate,
            buffer_size=sac_cfg.buffer_size,
            batch_size=sac_cfg.batch_size,
            tau=sac_cfg.tau,
            gamma=sac_cfg.gamma,
            ent_coef=sac_cfg.ent_coef,
            train_freq=sac_cfg.train_freq,
            gradient_steps=sac_cfg.gradient_steps,
            verbose=1,
            device=sac_cfg.device,
            tensorboard_log=str(log_dir) if config.logging.tensorboard_log else None,
        )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=sac_cfg.learning_rate,
            buffer_size=sac_cfg.buffer_size,
            batch_size=sac_cfg.batch_size,
            tau=sac_cfg.tau,
            gamma=sac_cfg.gamma,
            ent_coef=sac_cfg.ent_coef,
            train_freq=sac_cfg.train_freq,
            gradient_steps=sac_cfg.gradient_steps,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=sac_cfg.device,
            tensorboard_log=str(log_dir) if config.logging.tensorboard_log else None,
        )

    # Callbacks
    callbacks = []

    # Metrics
    metrics_callback = TrainingMetricsCallback(config, verbose=1)
    callbacks.append(metrics_callback)

    # Uniform random wind sampling (no curriculum)
    if config.physics.enable_wind:
        print(
            f"Uniform random wind enabled: [0, {config.physics.base_wind_speed:.1f}] m/s "
            f"(sampled per episode)"
        )

    # Early stopping (moving average -- robust to high-variance evals)
    stop_callback = None
    if early_stopping_patience > 0:
        stop_callback = MovingAverageEarlyStoppingCallback(
            window_size=20,
            max_no_improvement_evals=early_stopping_patience,
            min_evals=20,
            verbose=1,
        )
        print(
            f"Early stopping: EMA (alpha=0.15), "
            f"patience={early_stopping_patience} evals without improvement"
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

    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=config.logging.save_freq,
        save_path=str(save_dir / "checkpoints"),
        name_prefix="rocket_residual_sac",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Train
    print("Starting residual SAC training...")
    model.learn(
        total_timesteps=sac_cfg.total_timesteps,
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
        f"  uv run python visualizations/visualize_spin_agent.py "
        f"{save_dir}/best_model.zip --config {save_dir}/config.yaml"
    )
    print(f"\nTo compare controllers:")
    print(
        f"  uv run python compare_controllers.py "
        f"--residual-sac {save_dir}/best_model.zip "
        f"--config {save_dir}/config.yaml"
    )
    print(f"{'='*70}\n")

    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train residual SAC agent (PID + RL corrections) for rocket spin control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with wind curriculum
  uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml

  # Quick smoke test
  uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml --timesteps 10000

  # Override hyperparameters
  uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml \\
      --timesteps 2000000 --early-stopping 30 \\
      --lr 0.0003 --buffer-size 300000 --batch-size 256

  # Fine-tune from existing model
  uv run python train_residual_sac.py --config configs/estes_c6_residual_sac_wind.yaml \\
      --load-model models/rocket_residual_sac_wind_*/best_model.zip
        """,
    )

    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--timesteps", type=int, help="Override total timesteps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--buffer-size", type=int, help="Override replay buffer size")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda/auto)")
    parser.add_argument(
        "--load-model", type=str, help="Path to pre-trained model for fine-tuning"
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N evals without improvement (0=disabled)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Ensure SAC config exists
    if config.sac is None:
        from rocket_config import SACConfig

        config.sac = SACConfig()

    # Apply overrides
    if args.timesteps is not None:
        config.sac.total_timesteps = args.timesteps
    if args.lr is not None:
        config.sac.learning_rate = args.lr
    if args.buffer_size is not None:
        config.sac.buffer_size = args.buffer_size
    if args.batch_size is not None:
        config.sac.batch_size = args.batch_size
    if args.device is not None:
        config.sac.device = args.device

    train_residual_sac(
        config,
        early_stopping_patience=args.early_stopping,
        load_model_path=args.load_model,
    )


if __name__ == "__main__":
    main()
