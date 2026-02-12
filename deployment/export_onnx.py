#!/usr/bin/env python3
"""
Export SB3 PPO/SAC Model to ONNX

Converts a trained Stable-Baselines3 PPO or SAC model to ONNX format for
deployment on Raspberry Pi and other embedded systems.

Usage:
    python deployment/export_onnx.py --model models/rocket_ppo/best_model.zip
    python deployment/export_onnx.py --model models/rocket_sac/best_model.zip
    python deployment/export_onnx.py --model models/rocket_sac/best_model.zip --verify --benchmark

The exported ONNX model can be loaded with:
    from rocket_env.inference import RocketController
    controller = RocketController("model.onnx", "vec_normalize.pkl")
"""

import argparse
from pathlib import Path
from typing import Optional
import numpy as np


def _load_sb3_model(model_path: str):
    """
    Auto-detect and load an SB3 model (SAC or PPO).

    Returns:
        (model, model_type) where model_type is "SAC" or "PPO".
    """
    try:
        from stable_baselines3 import SAC

        model = SAC.load(str(model_path))
        return model, "SAC"
    except Exception:
        pass

    try:
        from stable_baselines3 import PPO

        model = PPO.load(str(model_path))
        return model, "PPO"
    except Exception:
        pass

    raise ValueError(
        f"Could not load model as SAC or PPO: {model_path}. "
        "Ensure the file is a valid SB3 model (.zip)."
    )


def export_sb3_to_onnx(
    model_path: str,
    output_path: Optional[str] = None,
    obs_size: int = 10,
    opset_version: int = 11,
    verbose: bool = True,
) -> str:
    """
    Export SB3 PPO or SAC model to ONNX format.

    Args:
        model_path: Path to SB3 model (.zip file)
        output_path: Output ONNX file path (default: same dir as model)
        obs_size: Size of observation vector
        opset_version: ONNX opset version (11 is widely compatible)
        verbose: Print export information

    Returns:
        Path to exported ONNX file
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch and Stable-Baselines3 are required for export.\n"
            "Install with: pip install torch stable-baselines3"
        )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Default output path
    if output_path is None:
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    else:
        output_path = Path(output_path)

    if verbose:
        print(f"Loading SB3 model: {model_path}")

    # Auto-detect model type
    model, model_type = _load_sb3_model(str(model_path))

    if verbose:
        print(f"Model type: {model_type}")
        print(f"Policy type: {type(model.policy).__name__}")
        print(f"Observation space: {model.observation_space}")
        print(f"Action space: {model.action_space}")

    # Create dummy input
    dummy_input = torch.randn(1, obs_size, dtype=torch.float32)

    if model_type == "SAC":
        deterministic_policy = _build_sac_wrapper(model)
    else:
        deterministic_policy = _build_ppo_wrapper(model)

    deterministic_policy.eval()

    if verbose:
        print(f"Exporting to ONNX: {output_path}")

    # Export to ONNX
    torch.onnx.export(
        deterministic_policy,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )

    if verbose:
        print(f"Export complete: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")

    return str(output_path)


def _build_sac_wrapper(model):
    """Build a deterministic SAC policy wrapper for ONNX export."""
    import torch

    class DeterministicSACPolicy(torch.nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.features_extractor = actor.features_extractor
            self.latent_pi = actor.latent_pi
            self.mu = actor.mu

        def forward(self, obs):
            features = self.features_extractor(obs)
            latent_pi = self.latent_pi(features)
            return torch.tanh(self.mu(latent_pi))

    return DeterministicSACPolicy(model.policy.actor)


def _build_ppo_wrapper(model):
    """Build a deterministic PPO policy wrapper for ONNX export."""
    import torch

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            features = self.policy.extract_features(obs)
            if hasattr(self.policy, "mlp_extractor"):
                latent_pi, _ = self.policy.mlp_extractor(features)
            else:
                latent_pi = features
            mean_actions = self.policy.action_net(latent_pi)
            return mean_actions

    return DeterministicPolicy(model.policy)


def verify_onnx_model(
    onnx_path: str,
    sb3_model_path: str,
    obs_size: int = 10,
    n_tests: int = 100,
    tolerance: float = 1e-5,
) -> bool:
    """
    Verify ONNX model matches SB3 model outputs.

    Args:
        onnx_path: Path to ONNX model
        sb3_model_path: Path to original SB3 model
        obs_size: Size of observation vector
        n_tests: Number of random test cases
        tolerance: Maximum allowed difference

    Returns:
        True if verification passes
    """
    try:
        import torch
        import onnxruntime as ort
    except ImportError:
        print("Skipping verification (requires torch, sb3, onnxruntime)")
        return True

    print("Verifying ONNX model against SB3 model...")

    # Load models
    model, model_type = _load_sb3_model(sb3_model_path)
    ort_session = ort.InferenceSession(onnx_path)

    # Test random observations
    max_diff = 0.0
    for i in range(n_tests):
        obs = np.random.randn(1, obs_size).astype(np.float32)

        # SB3 prediction
        with torch.no_grad():
            sb3_action, _ = model.predict(obs, deterministic=True)

        # ONNX prediction
        onnx_action = ort_session.run(None, {"observation": obs})[0]

        diff = np.abs(sb3_action - onnx_action).max()
        max_diff = max(max_diff, diff)

    print(f"Maximum difference: {max_diff:.2e}")

    if max_diff > tolerance:
        print(f"WARNING: Difference exceeds tolerance ({tolerance})")
        return False

    print("Verification passed!")
    return True


def benchmark_onnx_model(onnx_path: str, obs_size: int = 10) -> None:
    """Run inference benchmark on ONNX model."""
    from rocket_env.inference import ONNXRunner, benchmark_inference

    print(f"\nBenchmarking ONNX model: {onnx_path}")
    results = benchmark_inference(onnx_path, obs_size=obs_size)

    print(f"\nResults ({results['iterations']} iterations):")
    print(f"  Mean:       {results['mean_ms']:.3f} ms")
    print(f"  Std:        {results['std_ms']:.3f} ms")
    print(f"  Min:        {results['min_ms']:.3f} ms")
    print(f"  Max:        {results['max_ms']:.3f} ms")
    print(f"  P99:        {results['p99_ms']:.3f} ms")
    print(f"  Throughput: {results['throughput_hz']:.0f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="Export SB3 PPO/SAC model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic export (auto-detects PPO or SAC)
    python deployment/export_onnx.py --model models/rocket_ppo/best_model.zip

    # Export with custom output path
    python deployment/export_onnx.py --model models/rocket_sac/best_model.zip --output model.onnx

    # Export, verify, and benchmark
    python deployment/export_onnx.py --model models/rocket_sac/best_model.zip --verify --benchmark
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to SB3 model (.zip file, PPO or SAC)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output ONNX file path (default: same directory as model)",
    )
    parser.add_argument(
        "--obs-size",
        type=int,
        default=10,
        help="Observation vector size (default: 10)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify ONNX output matches SB3 model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run inference benchmark after export",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Export model
    onnx_path = export_sb3_to_onnx(
        model_path=args.model,
        output_path=args.output,
        obs_size=args.obs_size,
        verbose=not args.quiet,
    )

    # Verify if requested
    if args.verify:
        verify_onnx_model(
            onnx_path=onnx_path,
            sb3_model_path=args.model,
            obs_size=args.obs_size,
        )

    # Benchmark if requested
    if args.benchmark:
        benchmark_onnx_model(onnx_path, obs_size=args.obs_size)

    print(f"\nExport complete!")
    print(f"ONNX model: {onnx_path}")
    print(f"\nTo deploy on Raspberry Pi:")
    print(f"  from rocket_env.inference import RocketController")
    print(f"  controller = RocketController('{onnx_path}')")


if __name__ == "__main__":
    main()
