#!/usr/bin/env python3
"""
Batch Export CLI for Raspberry Pi Deployment

Creates a self-contained deployment bundle with all files needed to run
a controller on Raspberry Pi. No torch or SB3 needed at runtime.

Usage:
    # PID-only (gains JSON, no ONNX)
    python deployment/export_all.py pid \
        --Kp 0.0203 --Ki 0.0002 --Kd 0.0118 --output deploy_pid/

    # Standalone SAC (ONNX + normalization)
    python deployment/export_all.py sac \
        --model models/.../best_model.zip --output deploy_sac/

    # Residual SAC (ONNX + normalization + PID gains)
    python deployment/export_all.py residual-sac \
        --model models/.../best_model.zip \
        --Kp 0.0203 --Ki 0.0002 --Kd 0.0118 --max-residual 0.2 \
        --output deploy_residual/

Bundle contents:
    deploy_bundle/
    ├── controller_config.json   # Type, gains, max_residual, obs_size, dt
    ├── model.onnx               # (SAC/residual-SAC only)
    └── normalize.json           # (SAC/residual-SAC only) obs mean/var arrays
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _export_normalize_json(vec_normalize_path: str, output_path: Path) -> None:
    """Export VecNormalize stats as JSON (no SB3 pickle dependency on Pi)."""
    with open(vec_normalize_path, "rb") as f:
        vec_normalize = pickle.load(f)

    if hasattr(vec_normalize, "obs_rms"):
        mean = vec_normalize.obs_rms.mean.tolist()
        var = vec_normalize.obs_rms.var.tolist()
    elif hasattr(vec_normalize, "running_mean"):
        mean = vec_normalize.running_mean.tolist()
        var = vec_normalize.running_var.tolist()
    else:
        raise ValueError("Could not extract normalization stats from VecNormalize file")

    stats = {"obs_mean": mean, "obs_var": var}
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Normalization stats: {output_path}")


def _find_vec_normalize(model_path: str) -> str:
    """Find vec_normalize.pkl in the same directory as the model."""
    model_dir = Path(model_path).parent
    for name in ["vec_normalize.pkl", "vecnormalize.pkl"]:
        candidate = model_dir / name
        if candidate.exists():
            return str(candidate)
    return None


def export_pid(args) -> None:
    """Export PID-only deployment bundle."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "controller_type": "pid",
        "Kp": args.Kp,
        "Ki": args.Ki,
        "Kd": args.Kd,
        "q_ref": args.q_ref,
        "max_deflection": args.max_deflection,
        "dt": args.dt,
    }

    config_path = output_dir / "controller_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"PID deployment bundle created: {output_dir}")
    print(f"  Config: {config_path}")


def export_sac(args) -> None:
    """Export standalone SAC deployment bundle."""
    from deployment.export_onnx import export_sb3_to_onnx

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX model
    onnx_path = str(output_dir / "model.onnx")
    export_sb3_to_onnx(
        model_path=args.model,
        output_path=onnx_path,
        obs_size=args.obs_size,
        verbose=True,
    )

    # Export normalization stats as JSON
    vec_norm_path = args.normalize or _find_vec_normalize(args.model)
    if vec_norm_path:
        _export_normalize_json(vec_norm_path, output_dir / "normalize.json")
    else:
        print("  Warning: No vec_normalize.pkl found, skipping normalization export")

    config = {
        "controller_type": "sac",
        "obs_size": args.obs_size,
        "dt": args.dt,
    }

    config_path = output_dir / "controller_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSAC deployment bundle created: {output_dir}")
    print(f"  Config: {config_path}")


def export_residual_sac(args) -> None:
    """Export residual SAC deployment bundle (PID + SAC ONNX)."""
    from deployment.export_onnx import export_sb3_to_onnx

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX model
    onnx_path = str(output_dir / "model.onnx")
    export_sb3_to_onnx(
        model_path=args.model,
        output_path=onnx_path,
        obs_size=args.obs_size,
        verbose=True,
    )

    # Export normalization stats as JSON
    vec_norm_path = args.normalize or _find_vec_normalize(args.model)
    if vec_norm_path:
        _export_normalize_json(vec_norm_path, output_dir / "normalize.json")
    else:
        print("  Warning: No vec_normalize.pkl found, skipping normalization export")

    config = {
        "controller_type": "residual-sac",
        "Kp": args.Kp,
        "Ki": args.Ki,
        "Kd": args.Kd,
        "q_ref": args.q_ref,
        "max_deflection": args.max_deflection,
        "max_residual": args.max_residual,
        "obs_size": args.obs_size,
        "dt": args.dt,
    }

    config_path = output_dir / "controller_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nResidual SAC deployment bundle created: {output_dir}")
    print(f"  Config: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create deployment bundles for Raspberry Pi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # PID-only bundle
    python deployment/export_all.py pid --Kp 0.0203 --Ki 0.0002 --Kd 0.0118 --output deploy_pid/

    # SAC bundle
    python deployment/export_all.py sac --model models/.../best_model.zip --output deploy_sac/

    # Residual SAC bundle
    python deployment/export_all.py residual-sac \\
        --model models/.../best_model.zip \\
        --Kp 0.0203 --Ki 0.0002 --Kd 0.0118 --max-residual 0.2 \\
        --output deploy_residual/

Deploy on Raspberry Pi:
    pip install numpy onnxruntime  # ~50MB total, no torch needed
    from rocket_env.inference import PIDDeployController, ResidualSACController
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common_args(sub):
        sub.add_argument(
            "--output", type=str, required=True, help="Output directory for bundle"
        )
        sub.add_argument("--dt", type=float, default=0.01, help="Control timestep (s)")

    def add_pid_args(sub):
        sub.add_argument("--Kp", type=float, default=0.0203, help="Proportional gain")
        sub.add_argument("--Ki", type=float, default=0.0002, help="Integral gain")
        sub.add_argument("--Kd", type=float, default=0.0118, help="Derivative gain")
        sub.add_argument(
            "--q-ref",
            type=float,
            default=500.0,
            help="Reference dynamic pressure for gain scheduling (Pa)",
        )
        sub.add_argument(
            "--max-deflection",
            type=float,
            default=30.0,
            help="Max tab deflection (degrees)",
        )

    def add_model_args(sub):
        sub.add_argument(
            "--model", type=str, required=True, help="Path to SB3 model (.zip)"
        )
        sub.add_argument(
            "--normalize",
            type=str,
            default=None,
            help="Path to vec_normalize.pkl (auto-detected if not set)",
        )
        sub.add_argument(
            "--obs-size", type=int, default=10, help="Observation vector size"
        )

    # PID subcommand
    pid_parser = subparsers.add_parser("pid", help="Export PID-only bundle")
    add_common_args(pid_parser)
    add_pid_args(pid_parser)
    pid_parser.set_defaults(func=export_pid)

    # SAC subcommand
    sac_parser = subparsers.add_parser("sac", help="Export standalone SAC bundle")
    add_common_args(sac_parser)
    add_model_args(sac_parser)
    sac_parser.set_defaults(func=export_sac)

    # Residual SAC subcommand
    res_parser = subparsers.add_parser(
        "residual-sac", help="Export residual SAC bundle"
    )
    add_common_args(res_parser)
    add_pid_args(res_parser)
    add_model_args(res_parser)
    res_parser.add_argument(
        "--max-residual",
        type=float,
        default=0.2,
        help="Max RL residual magnitude",
    )
    res_parser.set_defaults(func=export_residual_sac)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
