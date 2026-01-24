"""
ONNX Runtime Inference

Lightweight ONNX model inference without PyTorch dependency.
Suitable for deployment on Raspberry Pi and other embedded systems.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path


class ONNXRunner:
    """
    Lightweight ONNX model runner for inference.

    This class provides a simple interface for running ONNX models
    without requiring PyTorch. It's optimized for deployment on
    resource-constrained devices like Raspberry Pi.

    Example:
        >>> runner = ONNXRunner("model.onnx")
        >>> obs = np.array([0.0, 5.0, 0.1, 0.5, ...], dtype=np.float32)
        >>> action = runner.predict(obs)
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
    ):
        """
        Initialize ONNX runner.

        Args:
            model_path: Path to ONNX model file
            providers: ONNX Runtime execution providers (default: CPU)
                Options: ["CPUExecutionProvider"], ["CUDAExecutionProvider"]
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime is required for ONNX inference.\n"
                "Install with: pip install onnxruntime"
            )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        # Default to CPU execution
        if providers is None:
            providers = ["CPUExecutionProvider"]

        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Enable threading for better performance on multi-core devices
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 1

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        # Get input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Determine if this is a deterministic or stochastic model
        self._is_deterministic = len(self.output_names) == 1

    def predict(
        self, observation: np.ndarray, deterministic: bool = True
    ) -> np.ndarray:
        """
        Run inference on observation.

        Args:
            observation: Input observation array
            deterministic: If True, return mean action (for stochastic policies)

        Returns:
            Action array
        """
        # Ensure correct dtype and shape
        obs = np.asarray(observation, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: obs})

        # Handle different output formats
        if self._is_deterministic:
            # Simple model: single output is the action
            action = outputs[0]
        else:
            # Stochastic policy: first output is action mean
            # (SB3 exports mean, log_std for stochastic policies)
            action = outputs[0]

        # Remove batch dimension
        return action.squeeze(0)

    def get_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_path": str(self.model_path),
            "input_name": self.input_name,
            "input_shape": self.input_shape,
            "output_names": self.output_names,
            "is_deterministic": self._is_deterministic,
        }


def benchmark_inference(
    model_path: str,
    obs_size: int = 10,
    n_iterations: int = 1000,
) -> dict:
    """
    Benchmark ONNX model inference speed.

    Args:
        model_path: Path to ONNX model
        obs_size: Size of observation vector
        n_iterations: Number of inference iterations

    Returns:
        Dictionary with timing statistics
    """
    import time

    runner = ONNXRunner(model_path)

    # Create random observations
    observations = np.random.randn(n_iterations, obs_size).astype(np.float32)

    # Warmup
    for i in range(10):
        runner.predict(observations[i])

    # Timed run
    times = []
    for i in range(n_iterations):
        start = time.perf_counter()
        runner.predict(observations[i])
        times.append(time.perf_counter() - start)

    times = np.array(times) * 1000  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p99_ms": np.percentile(times, 99),
        "iterations": n_iterations,
        "throughput_hz": 1000 / np.mean(times),
    }
