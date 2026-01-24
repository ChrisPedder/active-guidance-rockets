"""
Rocket Controller for Deployment

High-level controller interface that wraps ONNX inference with
observation normalization. Designed for deployment on Raspberry Pi.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union
import pickle

from .onnx_runner import ONNXRunner


class RocketController:
    """
    Lightweight rocket controller for embedded deployment.

    This class provides a complete inference pipeline:
    1. Observation normalization (using saved VecNormalize stats)
    2. ONNX model inference
    3. Action denormalization

    No PyTorch dependency required - uses only numpy and onnxruntime.

    Example:
        >>> controller = RocketController(
        ...     model_path="model.onnx",
        ...     normalize_path="vec_normalize.pkl"
        ... )
        >>> # Raw observation from sensors
        >>> obs = np.array([altitude, velocity, roll_angle, roll_rate, ...])
        >>> # Get control action
        >>> action = controller.get_action(obs)
        >>> # Apply action to actuator
        >>> set_tab_deflection(action[0] * max_deflection)
    """

    def __init__(
        self,
        model_path: str,
        normalize_path: Optional[str] = None,
        action_low: float = -1.0,
        action_high: float = 1.0,
        clip_obs: float = 10.0,
    ):
        """
        Initialize rocket controller.

        Args:
            model_path: Path to ONNX model file
            normalize_path: Path to VecNormalize pickle file (optional)
            action_low: Minimum action value
            action_high: Maximum action value
            clip_obs: Observation clipping value after normalization
        """
        self.model_path = Path(model_path)
        self.action_low = action_low
        self.action_high = action_high
        self.clip_obs = clip_obs

        # Load ONNX model
        self.runner = ONNXRunner(str(model_path))

        # Load normalization stats if provided
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_var: Optional[np.ndarray] = None
        self.epsilon: float = 1e-8

        if normalize_path:
            self._load_normalize_stats(normalize_path)

    def _load_normalize_stats(self, normalize_path: str) -> None:
        """
        Load observation normalization statistics from VecNormalize.

        The stats are extracted from the pickle file saved by SB3's
        VecNormalize wrapper.
        """
        normalize_path = Path(normalize_path)
        if not normalize_path.exists():
            raise FileNotFoundError(f"Normalization file not found: {normalize_path}")

        with open(normalize_path, "rb") as f:
            vec_normalize = pickle.load(f)

        # Extract running mean and variance
        if hasattr(vec_normalize, "obs_rms"):
            self.obs_mean = vec_normalize.obs_rms.mean.astype(np.float32)
            self.obs_var = vec_normalize.obs_rms.var.astype(np.float32)
        elif hasattr(vec_normalize, "running_mean"):
            # Alternative format
            self.obs_mean = vec_normalize.running_mean.astype(np.float32)
            self.obs_var = vec_normalize.running_var.astype(np.float32)
        else:
            print("Warning: Could not extract normalization stats from file")

    def normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Normalize observation using saved statistics.

        Args:
            observation: Raw observation vector

        Returns:
            Normalized and clipped observation
        """
        obs = np.asarray(observation, dtype=np.float32)

        if self.obs_mean is not None and self.obs_var is not None:
            # Normalize: (obs - mean) / sqrt(var + eps)
            obs = (obs - self.obs_mean) / np.sqrt(self.obs_var + self.epsilon)

            # Clip to prevent extreme values
            obs = np.clip(obs, -self.clip_obs, self.clip_obs)

        return obs

    def get_action(
        self,
        observation: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Get control action from observation.

        Args:
            observation: Raw or pre-normalized observation
            normalize: Whether to apply normalization (default: True)

        Returns:
            Action array clipped to valid range
        """
        obs = np.asarray(observation, dtype=np.float32)

        # Normalize if requested and stats are available
        if normalize:
            obs = self.normalize_observation(obs)

        # Run inference
        action = self.runner.predict(obs, deterministic=True)

        # Clip to valid action range
        action = np.clip(action, self.action_low, self.action_high)

        return action

    def get_tab_deflection(
        self,
        observation: np.ndarray,
        max_deflection_deg: float = 15.0,
    ) -> float:
        """
        Get tab deflection in degrees.

        Convenience method that converts normalized action to actual
        deflection angle.

        Args:
            observation: Raw observation vector
            max_deflection_deg: Maximum tab deflection in degrees

        Returns:
            Tab deflection in degrees
        """
        action = self.get_action(observation)
        return float(action[0] * max_deflection_deg)

    def get_info(self) -> dict:
        """Get controller metadata."""
        info = {
            "model_path": str(self.model_path),
            "has_normalization": self.obs_mean is not None,
            "action_range": (self.action_low, self.action_high),
        }
        info.update(self.runner.get_info())
        return info


def load_controller_from_training(
    model_dir: str,
    model_name: str = "best_model.onnx",
    normalize_name: str = "vec_normalize.pkl",
) -> RocketController:
    """
    Load controller from training output directory.

    Args:
        model_dir: Directory containing model and normalization files
        model_name: Name of ONNX model file
        normalize_name: Name of normalization pickle file

    Returns:
        Configured RocketController instance

    Example:
        >>> controller = load_controller_from_training("models/rocket_ppo_20240115/")
    """
    model_dir = Path(model_dir)
    model_path = model_dir / model_name
    normalize_path = model_dir / normalize_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    normalize_path = normalize_path if normalize_path.exists() else None

    return RocketController(
        model_path=str(model_path),
        normalize_path=str(normalize_path) if normalize_path else None,
    )
