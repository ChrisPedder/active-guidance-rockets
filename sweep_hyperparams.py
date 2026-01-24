#!/usr/bin/env python3
"""
Hyperparameter Sweep Script for Rocket Spin Control

This script runs multiple training experiments with different parameter
combinations to find the best configuration.

Usage:
    # Run pre-defined sweep
    python sweep_hyperparams.py --sweep physics
    python sweep_hyperparams.py --sweep reward
    python sweep_hyperparams.py --sweep ppo

    # Custom sweep from YAML
    python sweep_hyperparams.py --sweep-file my_sweep.yaml

    # Dry run (show configurations without training)
    python sweep_hyperparams.py --sweep physics --dry-run
"""

import os
import sys
import argparse
import itertools
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import json

from rocket_config import RocketTrainingConfig, MotorConfig


def generate_sweep_configs(
    sweep_type: str, base_config: RocketTrainingConfig
) -> List[Dict[str, Any]]:
    """Generate configurations for different sweep types"""

    sweeps = []

    if sweep_type == "physics":
        # Sweep mass configurations to find optimal TWR
        motor_specs = MotorConfig.get_motor_specs(base_config.motor.name)
        avg_thrust = motor_specs["average_thrust"]
        g = 9.81

        # Generate masses for TWR from 2.0 to 6.0
        for target_twr in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
            dry_mass = (avg_thrust / (target_twr * g)) - motor_specs["propellant_mass"]
            if dry_mass > 0.02:  # Minimum realistic mass
                sweeps.append(
                    {
                        "name": f"twr_{target_twr:.1f}",
                        "physics.dry_mass": dry_mass,
                        "description": f"TWR={target_twr:.1f}, mass={dry_mass*1000:.1f}g",
                    }
                )

        # Also sweep control authority
        for tab_deflection in [10.0, 15.0, 20.0, 25.0]:
            for tab_fraction in [0.2, 0.3, 0.4]:
                sweeps.append(
                    {
                        "name": f"tab_{tab_deflection:.0f}deg_{tab_fraction:.1f}frac",
                        "physics.max_tab_deflection": tab_deflection,
                        "physics.tab_chord_fraction": tab_fraction,
                        "description": f"Tab deflection {tab_deflection}°, chord fraction {tab_fraction}",
                    }
                )

    elif sweep_type == "reward":
        # Sweep reward function weights
        spin_scales = [-0.05, -0.1, -0.2, -0.5]
        altitude_scales = [0.005, 0.01, 0.02, 0.05]

        for spin_scale in spin_scales:
            for alt_scale in altitude_scales:
                sweeps.append(
                    {
                        "name": f"spin{abs(spin_scale):.2f}_alt{alt_scale:.3f}",
                        "reward.spin_penalty_scale": spin_scale,
                        "reward.altitude_reward_scale": alt_scale,
                        "description": f"Spin penalty {spin_scale}, altitude reward {alt_scale}",
                    }
                )

        # Also sweep low-spin bonus
        for bonus in [0.0, 0.5, 1.0, 2.0]:
            for threshold in [5.0, 10.0, 20.0]:
                sweeps.append(
                    {
                        "name": f"bonus{bonus:.1f}_thresh{threshold:.0f}",
                        "reward.low_spin_bonus": bonus,
                        "reward.low_spin_threshold": threshold,
                        "description": f"Low spin bonus {bonus} at threshold {threshold}°/s",
                    }
                )

    elif sweep_type == "ppo":
        # Sweep PPO hyperparameters
        learning_rates = [1e-4, 3e-4, 1e-3]
        clip_ranges = [0.1, 0.2, 0.3]
        batch_sizes = [32, 64, 128]

        for lr in learning_rates:
            for clip in clip_ranges:
                for batch in batch_sizes:
                    sweeps.append(
                        {
                            "name": f"lr{lr:.0e}_clip{clip:.1f}_batch{batch}",
                            "ppo.learning_rate": lr,
                            "ppo.clip_range": clip,
                            "ppo.batch_size": batch,
                            "description": f"LR={lr:.0e}, clip={clip}, batch={batch}",
                        }
                    )

        # Also sweep network architecture
        architectures = [
            ([64, 64], "small"),
            ([128, 128], "medium"),
            ([256, 256], "large"),
            ([256, 128, 64], "tapered"),
        ]
        for arch, name in architectures:
            sweeps.append(
                {
                    "name": f"arch_{name}",
                    "ppo.policy_net_arch": arch,
                    "ppo.value_net_arch": arch,
                    "description": f"Network architecture: {arch}",
                }
            )

    elif sweep_type == "motors":
        # Sweep across different motors
        motors_and_masses = [
            ("estes_c6", 0.100),  # 100g for C motor
            ("aerotech_f40", 0.400),  # 400g for F motor
            ("cesaroni_g79", 0.800),  # 800g for G motor
        ]
        for motor, mass in motors_and_masses:
            sweeps.append(
                {
                    "name": f"motor_{motor}",
                    "motor.name": motor,
                    "physics.dry_mass": mass,
                    "description": f"Motor: {motor}, mass: {mass*1000:.0f}g",
                }
            )

    elif sweep_type == "quick":
        # Quick sweep with fewer configs for testing
        sweeps = [
            {"name": "baseline", "description": "Default configuration"},
            {
                "name": "high_lr",
                "ppo.learning_rate": 1e-3,
                "description": "Higher learning rate",
            },
            {
                "name": "low_lr",
                "ppo.learning_rate": 1e-4,
                "description": "Lower learning rate",
            },
            {
                "name": "more_spin_penalty",
                "reward.spin_penalty_scale": -0.2,
                "description": "More spin penalty",
            },
            {
                "name": "less_spin_penalty",
                "reward.spin_penalty_scale": -0.05,
                "description": "Less spin penalty",
            },
        ]

    else:
        raise ValueError(f"Unknown sweep type: {sweep_type}")

    return sweeps


def apply_config_overrides(
    base_config: RocketTrainingConfig, overrides: Dict[str, Any]
) -> RocketTrainingConfig:
    """Apply a dictionary of overrides to a configuration"""
    import copy

    config = copy.deepcopy(base_config)

    for key, value in overrides.items():
        if key in ("name", "description"):
            continue

        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return config


def run_sweep(
    sweep_configs: List[Dict[str, Any]],
    base_config: RocketTrainingConfig,
    output_dir: str,
    dry_run: bool = False,
    parallel: int = 1,
    timesteps_override: Optional[int] = None,
):
    """Run sweep experiments"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save sweep configuration
    sweep_info = {
        "start_time": datetime.now().isoformat(),
        "num_configs": len(sweep_configs),
        "configs": sweep_configs,
    }

    with open(output_path / "sweep_info.json", "w") as f:
        json.dump(sweep_info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SWEEP")
    print(f"{'='*70}")
    print(f"Number of configurations: {len(sweep_configs)}")
    print(f"Output directory: {output_path}")
    print(f"{'='*70}\n")

    results = []

    for i, sweep in enumerate(sweep_configs):
        name = sweep.get("name", f"config_{i}")
        description = sweep.get("description", "")

        print(f"\n[{i+1}/{len(sweep_configs)}] {name}")
        print(f"  {description}")

        # Create config
        config = apply_config_overrides(base_config, sweep)

        # Override timesteps if specified (for quick testing)
        if timesteps_override:
            config.ppo.total_timesteps = timesteps_override

        # Save this config
        config_path = output_path / f"{name}_config.yaml"
        config.save(config_path)

        if dry_run:
            print(f"  [DRY RUN] Would train with config: {config_path}")
            continue

        # Run training
        try:
            # Import and run training
            from train_improved import train

            config.logging.experiment_name = f"sweep_{name}"
            config.logging.log_dir = str(output_path / "logs")
            config.logging.save_dir = str(output_path / "models")

            model = train(config)

            results.append(
                {
                    "name": name,
                    "config": str(config_path),
                    "status": "success",
                }
            )

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append(
                {
                    "name": name,
                    "config": str(config_path),
                    "status": "error",
                    "error": str(e),
                }
            )

    # Save results
    results_file = output_path / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"SWEEP COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")

    if not dry_run:
        # Print summary
        successful = sum(1 for r in results if r["status"] == "success")
        print(f"Successful: {successful}/{len(results)}")

    return results


def load_sweep_from_yaml(path: str) -> List[Dict[str, Any]]:
    """Load sweep configuration from YAML file"""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("sweeps", [])


def main():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for rocket training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sweep Types:
  physics   - Sweep mass (TWR) and control authority parameters
  reward    - Sweep reward function weights
  ppo       - Sweep PPO hyperparameters (LR, clip, batch size, architecture)
  motors    - Sweep across different motor types
  quick     - Quick test sweep with few configurations

Examples:
  # Run physics sweep
  python sweep_hyperparams.py --sweep physics --base-config configs/estes_c6.yaml

  # Dry run to preview configurations
  python sweep_hyperparams.py --sweep reward --dry-run

  # Quick sweep with reduced timesteps
  python sweep_hyperparams.py --sweep quick --timesteps 50000

  # Custom sweep from file
  python sweep_hyperparams.py --sweep-file my_sweep.yaml
        """,
    )

    parser.add_argument(
        "--sweep",
        type=str,
        choices=["physics", "reward", "ppo", "motors", "quick"],
        help="Type of sweep to run",
    )
    parser.add_argument(
        "--sweep-file", type=str, help="Custom sweep configuration YAML file"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default="configs/estes_c6.yaml",
        help="Base configuration to modify",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: sweeps/<timestamp>)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show configurations without running training",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override timesteps for all runs (for testing)",
    )
    parser.add_argument(
        "--create-example-sweep",
        action="store_true",
        help="Create an example sweep YAML file",
    )

    args = parser.parse_args()

    if args.create_example_sweep:
        example = {
            "description": "Example custom sweep configuration",
            "sweeps": [
                {
                    "name": "baseline",
                    "description": "Default configuration",
                },
                {
                    "name": "light_rocket",
                    "physics.dry_mass": 0.08,
                    "description": "80g rocket",
                },
                {
                    "name": "heavy_rocket",
                    "physics.dry_mass": 0.15,
                    "description": "150g rocket",
                },
                {
                    "name": "aggressive_spin_control",
                    "reward.spin_penalty_scale": -0.3,
                    "reward.low_spin_bonus": 2.0,
                    "description": "Emphasize spin control over altitude",
                },
            ],
        }

        Path("configs").mkdir(exist_ok=True)
        with open("configs/example_sweep.yaml", "w") as f:
            yaml.dump(example, f, default_flow_style=False)
        print("Created configs/example_sweep.yaml")
        return

    if not args.sweep and not args.sweep_file:
        parser.print_help()
        print("\n❌ Error: --sweep or --sweep-file is required")
        return

    # Load base configuration
    if not Path(args.base_config).exists():
        # Create default config if it doesn't exist
        from rocket_config import create_default_configs

        create_default_configs()

    base_config = RocketTrainingConfig.load(args.base_config)

    # Generate or load sweep configurations
    if args.sweep_file:
        sweep_configs = load_sweep_from_yaml(args.sweep_file)
    else:
        sweep_configs = generate_sweep_configs(args.sweep, base_config)

    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sweep_name = args.sweep or Path(args.sweep_file).stem
        output_dir = f"sweeps/{sweep_name}_{timestamp}"

    # Run sweep
    run_sweep(
        sweep_configs=sweep_configs,
        base_config=base_config,
        output_dir=output_dir,
        dry_run=args.dry_run,
        timesteps_override=args.timesteps,
    )


if __name__ == "__main__":
    main()
