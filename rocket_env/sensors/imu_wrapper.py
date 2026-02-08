"""
IMU Observation Wrapper

Gymnasium ObservationWrapper that applies realistic IMU sensor noise
to the observation vector. This allows training controllers that are
robust to real sensor characteristics.

The wrapper intercepts observations and applies noise to the roll rate
measurement (obs[3]), which represents the gyroscope output.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Optional, Tuple, Dict

from .imu_config import IMUConfig
from .gyro_model import GyroModel


class IMUObservationWrapper(gym.ObservationWrapper):
    """
    Wrapper that applies realistic IMU noise to observations.

    This wrapper intercepts the environment's observations and applies
    sensor noise models to simulate real IMU behavior. The primary target
    is the roll rate measurement (obs[3]), which simulates a gyroscope.

    The wrapper is composable with other wrappers and doesn't modify
    the underlying environment's physics.

    Example:
        >>> from rocket_env.sensors import IMUObservationWrapper, IMUConfig
        >>> base_env = RealisticMotorRocket(airframe=airframe, motor_config=motor)
        >>> imu_config = IMUConfig.icm_20948()
        >>> env = IMUObservationWrapper(base_env, imu_config=imu_config)
        >>> obs, info = env.reset()
        >>> # obs[3] now contains noisy roll rate measurement

    Attributes:
        imu_config: IMU configuration with noise parameters
        gyro_model: Gyroscope noise model instance
        roll_rate_index: Index of roll rate in observation vector
        derive_acceleration: Whether to derive noisy acceleration
    """

    # Default observation indices for rocket environment
    DEFAULT_ROLL_RATE_INDEX = 3
    DEFAULT_ROLL_ACCEL_INDEX = 4

    def __init__(
        self,
        env: gym.Env,
        imu_config: Optional[IMUConfig] = None,
        control_rate_hz: float = 100.0,
        roll_rate_index: int = DEFAULT_ROLL_RATE_INDEX,
        roll_accel_index: int = DEFAULT_ROLL_ACCEL_INDEX,
        derive_acceleration: bool = True,
        seed: Optional[int] = None,
    ):
        """
        Initialize IMU observation wrapper.

        Args:
            env: Base gymnasium environment to wrap
            imu_config: IMU configuration (default: ICM-20948 preset)
            control_rate_hz: Control loop frequency in Hz
            roll_rate_index: Index of roll rate in observation vector
            roll_accel_index: Index of roll acceleration in observation
            derive_acceleration: If True, derive noisy acceleration from rate
            seed: Random seed for reproducibility
        """
        super().__init__(env)

        # Use default ICM-20948 if no config provided
        self.imu_config = imu_config or IMUConfig.icm_20948()

        # Store configuration
        self.control_rate_hz = control_rate_hz
        self.roll_rate_index = roll_rate_index
        self.roll_accel_index = roll_accel_index
        self.derive_acceleration = derive_acceleration

        # Calculate timestep
        dt = 1.0 / control_rate_hz

        # Create gyroscope model
        self.gyro_model = GyroModel(
            config=self.imu_config.gyro,
            dt=dt,
            seed=seed,
        )

        # State for acceleration derivation
        self._prev_noisy_rate: Optional[float] = None
        self._dt = dt

        # Update observation space bounds if needed
        # The noisy observations should still fit within the original bounds
        # since we apply saturation in the gyro model

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment and sensor models.

        Args:
            seed: Random seed
            options: Additional options passed to base env

        Returns:
            Tuple of (observation, info) with noisy sensor readings
        """
        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Reset gyroscope model with new random bias
        self.gyro_model.reset(seed=seed)

        # Reset acceleration derivation state
        self._prev_noisy_rate = None

        # Apply observation noise
        noisy_obs = self.observation(obs)

        # Add sensor info to info dict
        info["imu"] = {
            "preset": self.imu_config.name,
            "gyro_bias": self.gyro_model.current_bias,
            "gyro_scale_factor": self.gyro_model.current_scale_factor,
        }

        return noisy_obs, info

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Apply sensor noise to observation.

        This method is called for every observation, both from reset()
        and step(). It modifies the roll rate measurement to include
        realistic gyroscope noise.

        Args:
            observation: Clean observation from base environment

        Returns:
            Observation with noisy sensor readings
        """
        # Make a copy to avoid modifying the original
        noisy_obs = observation.copy()

        # Get true roll rate (in rad/s from environment)
        true_rate_rad = observation[self.roll_rate_index]
        true_rate_deg = np.rad2deg(true_rate_rad)

        # Apply gyroscope noise model
        noisy_rate_deg = self.gyro_model.measure(true_rate_deg)
        noisy_rate_rad = np.deg2rad(noisy_rate_deg)

        # Update observation with noisy rate
        noisy_obs[self.roll_rate_index] = noisy_rate_rad

        # Optionally derive noisy acceleration from noisy rate
        if self.derive_acceleration and self.roll_accel_index is not None:
            if self._prev_noisy_rate is not None:
                # Finite difference approximation of angular acceleration
                noisy_accel = (noisy_rate_rad - self._prev_noisy_rate) / self._dt
                noisy_obs[self.roll_accel_index] = noisy_accel
            self._prev_noisy_rate = noisy_rate_rad

        return noisy_obs

    def step(self, action):
        """Step environment and apply IMU noise to both obs and info.

        The obs vector may be subject to sensor_delay_steps, but the info dict
        always contains real-time state. We apply gyro noise to info so that
        classical controllers reading from info get noisy-but-current values,
        matching what a real gyro reads.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = self.observation(obs)

        # Apply gyro noise to info dict roll_rate using the same noisy
        # measurement that was applied to obs (avoids double-advancing
        # the gyro model's bias state). The last measure() call in
        # observation() produced a noisy rate — retrieve it.
        if "roll_rate_deg_s" in info:
            # Use the noisy rate from the most recent observation() call
            # (stored via _prev_noisy_rate) to get the noise delta,
            # and apply an equivalent noise to the info rate.
            true_info_rate = info["roll_rate_deg_s"]
            # The gyro noise is predominantly white noise + bias.
            # The bias is the same for obs and info readings. For the
            # white noise component, we add a fresh sample.
            noise_std = self.gyro_model._white_noise_std
            bias = self.gyro_model._bias
            scale = self.gyro_model._scale_factor
            white_noise = self.gyro_model.rng.normal(0.0, noise_std)
            noisy_info_rate = true_info_rate * scale + bias + white_noise
            info["roll_rate_deg_s"] = float(noisy_info_rate)

        return noisy_obs, reward, terminated, truncated, info

    def get_sensor_state(self) -> Dict[str, Any]:
        """
        Get current sensor state for debugging/logging.

        Returns:
            Dictionary with sensor state information
        """
        return {
            "imu_preset": self.imu_config.name,
            "gyro": self.gyro_model.get_state(),
        }

    @property
    def uses_noisy_observations(self) -> bool:
        """Flag indicating this wrapper applies sensor noise."""
        return True


def create_imu_wrapped_env(
    base_env: gym.Env,
    imu_preset: str = "icm_20948",
    control_rate_hz: float = 100.0,
    derive_acceleration: bool = True,
    seed: Optional[int] = None,
) -> IMUObservationWrapper:
    """
    Convenience function to create an IMU-wrapped environment.

    Args:
        base_env: Base gymnasium environment
        imu_preset: Name of IMU preset ("icm_20948", "mpu_6050", "bmi088", "ideal")
        control_rate_hz: Control loop frequency
        derive_acceleration: Whether to derive noisy acceleration
        seed: Random seed

    Returns:
        Environment with IMU noise wrapper applied

    Example:
        >>> env = RealisticMotorRocket(airframe=airframe, motor_config=motor)
        >>> env = create_imu_wrapped_env(env, imu_preset="mpu_6050")
    """
    imu_config = IMUConfig.get_preset(imu_preset)
    return IMUObservationWrapper(
        env=base_env,
        imu_config=imu_config,
        control_rate_hz=control_rate_hz,
        derive_acceleration=derive_acceleration,
        seed=seed,
    )
