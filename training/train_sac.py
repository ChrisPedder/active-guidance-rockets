#!/usr/bin/env python3
"""
SAC Training Script for Rocket Spin Control with Wind Disturbances

Architecture: SAC directly controls tabs (no PID in loop).
SAC's entropy regularization + action smoothing wrapper naturally
produce smooth policies. Wind curriculum progressively increases
difficulty during training.

Usage:
    # Train with wind curriculum
    uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml

    # Quick smoke test
    uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml --timesteps 10000

    # Override SAC hyperparameters
    uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml \
        --timesteps 2000000 --early-stopping 30 \
        --lr 0.0003 --buffer-size 300000 --batch-size 256
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from rocket_config import RocketTrainingConfig, load_config
from training.train_improved import (
    create_environment,
    TrainingMetricsCallback,
    ImprovedRewardWrapper,
    NormalizedActionWrapper,
    ExponentialSmoothingWrapper,
    PreviousActionWrapper,
)


class WindCurriculumCallback(BaseCallback):
    """
    Callback that progressively increases wind during training.

    Stages:
        1. 0-300K steps: No wind (learn basic control)
        2. 300K-800K: Light wind (1 m/s base)
        3. 800K-1.5M: Moderate wind (3 m/s base)
        4. 1.5M+: Full wind from config
    """

    def __init__(
        self,
        config: RocketTrainingConfig,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.config = config
        self.current_stage = 0

        # Wind curriculum stages: (step_threshold, base_wind_speed, max_gust_speed)
        final_base = getattr(config.physics, "base_wind_speed", 3.0)
        final_gust = getattr(config.physics, "max_gust_speed", 2.0)

        self.stages = [
            (0, 0.0, 0.0),  # Stage 1: No wind
            (300_000, 1.0, 0.5),  # Stage 2: Light wind
            (800_000, min(3.0, final_base), min(1.5, final_gust)),  # Stage 3: Moderate
            (1_500_000, final_base, final_gust),  # Stage 4: Full wind
        ]

    def _on_step(self) -> bool:
        # Check if we should advance to next stage
        new_stage = self.current_stage
        for i, (threshold, _, _) in enumerate(self.stages):
            if self.num_timesteps >= threshold:
                new_stage = i

        if new_stage != self.current_stage:
            self.current_stage = new_stage
            _, base_speed, gust_speed = self.stages[new_stage]

            # Update wind in all training environments
            self._update_wind(base_speed, gust_speed)

            if self.verbose > 0:
                print(
                    f"\n>>> WIND CURRICULUM: Stage {new_stage + 1}/4 "
                    f"at step {self.num_timesteps:,} - "
                    f"wind={base_speed:.1f} m/s, gusts={gust_speed:.1f} m/s"
                )

        return True

    def _update_wind(self, base_speed: float, gust_speed: float):
        """Update wind parameters in all wrapped environments."""
        # Access underlying environments through VecEnv
        vec_env = self.model.get_env()
        for i in range(vec_env.num_envs):
            env = vec_env.envs[i]
            # Walk through wrappers to find the base environment
            base_env = env
            while hasattr(base_env, "env"):
                base_env = base_env.env

            # Update wind model config
            if hasattr(base_env, "wind_model") and base_env.wind_model is not None:
                base_env.wind_model.config.base_speed = base_speed
                base_env.wind_model.config.max_gust_speed = gust_speed
            elif hasattr(base_env, "config"):
                # Wind was disabled, need to enable it
                from wind_model import WindModel, WindConfig

                base_env.config.enable_wind = base_speed > 0
                base_env.config.base_wind_speed = base_speed
                base_env.config.max_gust_speed = gust_speed
                if base_speed > 0 and base_env.wind_model is None:
                    wind_cfg = WindConfig(
                        enable=True,
                        base_speed=base_speed,
                        max_gust_speed=gust_speed,
                        variability=getattr(base_env.config, "wind_variability", 0.3),
                    )
                    base_env.wind_model = WindModel(wind_cfg)


def create_sac_environment(
    config: RocketTrainingConfig,
    wind_override: Optional[dict] = None,
) -> Monitor:
    """
    Create environment for SAC training.

    Uses the same wrapper chain as create_environment from train_improved.py
    but ensures wind config is properly passed through.

    Args:
        config: Training configuration
        wind_override: Optional dict to override wind settings
            e.g. {"base_wind_speed": 0.0, "max_gust_speed": 0.0}
    """
    # Apply wind overrides if provided
    if wind_override:
        for key, val in wind_override.items():
            if hasattr(config.physics, key):
                setattr(config.physics, key, val)

    return create_environment(config)


def train_sac(
    config: RocketTrainingConfig,
    early_stopping_patience: int = 0,
    load_model_path: Optional[str] = None,
):
    """Main SAC training function.

    Args:
        config: Training configuration (must have sac section)
        early_stopping_patience: Stop if no improvement for N evals (0=disabled)
        load_model_path: Optional path to pre-trained model for fine-tuning
    """
    sac_cfg = config.sac
    if sac_cfg is None:
        print("ERROR: No 'sac' section in config. Add SAC hyperparameters.")
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

    print(f"\n{'='*70}")
    print("SAC ROCKET SPIN CONTROL TRAINING")
    print(f"{'='*70}")
    print(f"Motor: {config.motor.name}")
    print(f"Algorithm: SAC (direct control, no PID)")
    print(f"Wind: {'enabled' if config.physics.enable_wind else 'disabled'}")
    if config.physics.enable_wind:
        print(f"  Base wind: {config.physics.base_wind_speed:.1f} m/s")
        print(f"  Max gust: {config.physics.max_gust_speed:.1f} m/s")
    print(f"Total timesteps: {sac_cfg.total_timesteps:,}")
    print(f"Device: {sac_cfg.device}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*70}\n")

    # Create training environment (single env for SAC - off-policy doesn't
    # benefit much from parallel envs during collection)
    def make_env():
        return create_sac_environment(config)

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

    # Wind curriculum
    if config.physics.enable_wind:
        wind_callback = WindCurriculumCallback(config, verbose=1)
        callbacks.append(wind_callback)
        print("Wind curriculum enabled (4 stages)")

    # Early stopping
    stop_callback = None
    if early_stopping_patience > 0:
        stop_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=early_stopping_patience,
            min_evals=early_stopping_patience,
            verbose=1,
        )
        print(f"Early stopping: {early_stopping_patience} evals without improvement")

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
        name_prefix="rocket_sac",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Train
    print("Starting SAC training...")
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
        f"--sac {save_dir}/best_model.zip --config {save_dir}/config.yaml"
    )
    print(f"{'='*70}\n")

    train_env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train SAC agent for rocket spin control with wind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with wind curriculum
  uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml

  # Quick smoke test
  uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml --timesteps 10000

  # Override hyperparameters
  uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml \\
      --timesteps 2000000 --early-stopping 30 \\
      --lr 0.0003 --buffer-size 300000 --batch-size 256

  # Fine-tune from existing model
  uv run python train_sac.py --config configs/estes_c6_sac_wind.yaml \\
      --load-model models/rocket_sac_wind_*/best_model.zip
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

    train_sac(
        config,
        early_stopping_patience=args.early_stopping,
        load_model_path=args.load_model,
    )


if __name__ == "__main__":
    main()
