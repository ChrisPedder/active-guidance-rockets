"""
Rocket Controller for Deployment

High-level controller interface that wraps ONNX inference with
observation normalization. Designed for deployment on Raspberry Pi.

Controllers:
    RocketController - Standalone SAC/PPO via ONNX inference
    PIDDeployController - Lightweight gain-scheduled PID (pure numpy)
    ResidualSACController - PID + SAC residual corrections
"""

import json
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


def load_normalize_json(json_path: str) -> tuple:
    """
    Load observation normalization stats from a JSON file.

    Args:
        json_path: Path to normalize.json (exported by deployment/export_all.py)

    Returns:
        (obs_mean, obs_var) as float32 numpy arrays
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    obs_mean = np.array(data["obs_mean"], dtype=np.float32)
    obs_var = np.array(data["obs_var"], dtype=np.float32)
    return obs_mean, obs_var


class PIDDeployController:
    """
    Lightweight gain-scheduled PID for deployment. Pure numpy, no torch/SB3.

    Takes physical sensor values (deg, deg/s, Pa) rather than the 10-element
    obs vector -- more natural for embedded code reading from IMU/barometer.

    Mirrors GainScheduledPIDController from controllers/pid_controller.py.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        Kp: float = 0.0203,
        Ki: float = 0.0002,
        Kd: float = 0.0118,
        q_ref: float = 500.0,
        max_deflection: float = 30.0,
        dt: float = 0.01,
        launch_accel_threshold: float = 20.0,
        max_roll_rate: float = 100.0,
    ):
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            Kp = cfg.get("Kp", Kp)
            Ki = cfg.get("Ki", Ki)
            Kd = cfg.get("Kd", Kd)
            q_ref = cfg.get("q_ref", q_ref)
            max_deflection = cfg.get("max_deflection", max_deflection)
            dt = cfg.get("dt", dt)

        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.q_ref = q_ref
        self.max_deflection = max_deflection
        self.dt = dt
        self.launch_accel_threshold = launch_accel_threshold
        self.max_roll_rate = max_roll_rate

        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)
        self.reset()

    def reset(self):
        """Reset controller state."""
        self.launch_detected = False
        self.launch_orient = 0.0
        self.integ_error = 0.0
        self.target_orient = 0.0

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure."""
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def get_action(
        self,
        roll_angle_deg: float,
        roll_rate_deg_s: float,
        dynamic_pressure_pa: float,
        vertical_accel_ms2: Optional[float] = None,
    ) -> float:
        """
        Compute control action from physical sensor values.

        Args:
            roll_angle_deg: Roll angle in degrees
            roll_rate_deg_s: Roll rate in deg/s
            dynamic_pressure_pa: Dynamic pressure in Pa
            vertical_accel_ms2: Vertical acceleration in m/s^2 (for launch detection).
                If None, launch is assumed on first call.

        Returns:
            Action in [-1, 1]
        """
        if not self.launch_detected:
            if vertical_accel_ms2 is not None:
                if vertical_accel_ms2 > self.launch_accel_threshold:
                    self.launch_detected = True
                    self.launch_orient = roll_angle_deg
                    self.target_orient = self.launch_orient
                else:
                    return 0.0
            else:
                self.launch_detected = True
                self.launch_orient = roll_angle_deg
                self.target_orient = self.launch_orient

        scale = self._gain_scale(dynamic_pressure_pa)

        roll_rate_clamped = np.clip(
            roll_rate_deg_s, -self.max_roll_rate, self.max_roll_rate
        )

        prop_error = roll_angle_deg - self.target_orient
        while prop_error > 180:
            prop_error -= 360
        while prop_error < -180:
            prop_error += 360

        integ_error_new = prop_error * self.dt
        self.integ_error += integ_error_new
        max_integ = self.max_deflection / (self.Ki + 1e-6)
        self.integ_error = np.clip(self.integ_error, -max_integ, max_integ)

        cmd_p = prop_error * self.Kp * scale
        cmd_i = self.integ_error * self.Ki
        cmd_d = roll_rate_clamped * self.Kd * scale

        servo_cmd = cmd_p + cmd_i + cmd_d
        servo_cmd = np.clip(servo_cmd, -self.max_deflection, self.max_deflection)

        action = -servo_cmd / self.max_deflection
        return float(np.clip(action, -1.0, 1.0))

    def get_tab_deflection(
        self,
        roll_angle_deg: float,
        roll_rate_deg_s: float,
        dynamic_pressure_pa: float,
        max_deflection_deg: float = 30.0,
        vertical_accel_ms2: Optional[float] = None,
    ) -> float:
        """Get tab deflection in degrees."""
        action = self.get_action(
            roll_angle_deg,
            roll_rate_deg_s,
            dynamic_pressure_pa,
            vertical_accel_ms2,
        )
        return action * max_deflection_deg


class ResidualSACController:
    """
    PID + SAC residual for deployment.

    Combines a lightweight PID running natively with an ONNX SAC model.
    Output = clip(pid_action + clip(sac_residual * max_residual), -1, 1)
    """

    def __init__(
        self,
        model_path: str,
        normalize_path: Optional[str] = None,
        config_path: Optional[str] = None,
        max_residual: float = 0.2,
        **pid_kwargs,
    ):
        if config_path is not None:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            max_residual = cfg.get("max_residual", max_residual)
            for key in ["Kp", "Ki", "Kd", "q_ref", "max_deflection", "dt"]:
                if key in cfg and key not in pid_kwargs:
                    pid_kwargs[key] = cfg[key]

        self.pid = PIDDeployController(config_path=None, **pid_kwargs)
        self.sac = RocketController(model_path, normalize_path)
        self.max_residual = max_residual

    def reset(self):
        """Reset both PID and SAC state."""
        self.pid.reset()

    def get_action(
        self,
        observation: np.ndarray,
        roll_angle_deg: float,
        roll_rate_deg_s: float,
        dynamic_pressure_pa: float,
        vertical_accel_ms2: Optional[float] = None,
    ) -> float:
        """
        Compute combined PID + SAC residual action.

        Args:
            observation: 10-element obs vector for SAC model
            roll_angle_deg: Roll angle in degrees (for PID)
            roll_rate_deg_s: Roll rate in deg/s (for PID)
            dynamic_pressure_pa: Dynamic pressure in Pa (for PID)
            vertical_accel_ms2: Vertical acceleration for launch detection

        Returns:
            Combined action in [-1, 1]
        """
        pid_action = self.pid.get_action(
            roll_angle_deg,
            roll_rate_deg_s,
            dynamic_pressure_pa,
            vertical_accel_ms2,
        )

        sac_raw = self.sac.get_action(observation)
        sac_residual = float(sac_raw[0])

        residual = np.clip(
            sac_residual * self.max_residual,
            -self.max_residual,
            self.max_residual,
        )

        return float(np.clip(pid_action + residual, -1.0, 1.0))
