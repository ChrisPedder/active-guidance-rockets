"""
Gyroscope Noise Model

Implements realistic gyroscope sensor noise including:
- White noise (angular random walk)
- Bias offset and drift
- Scale factor error
- Saturation
- Quantization

Based on IEEE Std 952-1997 (Standard Specification Format Guide and Test
Procedure for Single-Axis Interferometric Fiber Optic Gyros).
"""

import numpy as np
from typing import Optional

from .imu_config import GyroConfig


class GyroModel:
    """
    Realistic gyroscope sensor model.

    Simulates measurement noise and errors typical of MEMS gyroscopes.
    Call reset() at the start of each episode to initialize random bias.

    Example:
        >>> config = GyroConfig(noise_density=0.015, bias_instability=5.0)
        >>> gyro = GyroModel(config, dt=0.02)
        >>> gyro.reset()
        >>> noisy_rate = gyro.measure(true_rate=45.0)  # deg/s in, deg/s out
    """

    def __init__(
        self,
        config: GyroConfig,
        dt: float = 0.02,
        seed: Optional[int] = None,
    ):
        """
        Initialize gyroscope model.

        Args:
            config: Gyroscope configuration with noise parameters
            dt: Control loop timestep in seconds
            seed: Random seed for reproducibility (None for random)
        """
        self.config = config
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # State variables (initialized in reset)
        self._bias: float = 0.0
        self._scale_factor: float = 1.0
        self._last_sample: float = 0.0
        self._sample_timer: float = 0.0

        # Pre-calculate noise parameters
        self._white_noise_std = self._calculate_white_noise_std()
        self._bias_drift_std = self._calculate_bias_drift_std()

    def _calculate_white_noise_std(self) -> float:
        """
        Calculate white noise standard deviation.

        Noise density (ARW) in deg/s/sqrt(Hz) converts to deg/s when
        multiplied by sqrt(bandwidth). For discrete sampling:
        sigma = noise_density * sqrt(sample_rate)
        """
        if self.config.noise_density <= 0:
            return 0.0
        return self.config.noise_density * np.sqrt(self.config.sample_rate)

    def _calculate_bias_drift_std(self) -> float:
        """
        Calculate bias random walk standard deviation per timestep.

        Bias instability in deg/hr represents the 1-sigma drift over 1 hour.
        For random walk, drift per step is:
        sigma_step = (bias_instability / 3600) * sqrt(dt)

        This models a random walk where variance grows linearly with time.
        """
        if self.config.bias_instability <= 0:
            return 0.0
        # Convert deg/hr to deg/s, then scale by sqrt(dt)
        bias_rate_std = self.config.bias_instability / 3600.0
        return bias_rate_std * np.sqrt(self.dt)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset sensor state for new episode.

        Initializes random bias offset and scale factor error.
        Must be called at the start of each episode.

        Args:
            seed: Optional seed for this episode (None to continue sequence)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Random initial bias within specified range
        self._bias = self.rng.uniform(
            -self.config.bias_initial_range, self.config.bias_initial_range
        )

        # Random scale factor error
        self._scale_factor = 1.0 + self.rng.uniform(
            -self.config.scale_factor_error, self.config.scale_factor_error
        )

        # Reset sample-and-hold state
        self._last_sample = 0.0
        self._sample_timer = 0.0

    def measure(self, true_rate: float) -> float:
        """
        Simulate gyroscope measurement of true angular rate.

        Applies all configured noise sources in sequence:
        1. Scale factor error (multiplicative)
        2. Bias offset (additive)
        3. Bias drift (random walk)
        4. White noise (Gaussian)
        5. Saturation (clipping)
        6. Quantization (rounding to LSB)
        7. Sample-and-hold (if control rate > sensor rate)

        Args:
            true_rate: True angular rate in deg/s

        Returns:
            Noisy measurement in deg/s
        """
        # Update sample timer
        self._sample_timer += self.dt

        # Sample-and-hold: only update measurement at sensor sample rate
        sample_period = 1.0 / self.config.sample_rate
        if self._sample_timer >= sample_period:
            self._sample_timer = 0.0
            self._last_sample = self._compute_sample(true_rate)

        return self._last_sample

    def _compute_sample(self, true_rate: float) -> float:
        """Compute a new sensor sample with all noise effects."""
        measurement = true_rate

        # 1. Scale factor error
        measurement *= self._scale_factor

        # 2. Bias offset
        measurement += self._bias

        # 3. Bias drift (random walk)
        if self._bias_drift_std > 0:
            self._bias += self.rng.normal(0, self._bias_drift_std)

        # 4. White noise
        if self._white_noise_std > 0:
            measurement += self.rng.normal(0, self._white_noise_std)

        # 5. Saturation
        measurement = np.clip(
            measurement, -self.config.saturation, self.config.saturation
        )

        # 6. Quantization
        if self.config.quantization > 0:
            measurement = (
                np.round(measurement / self.config.quantization)
                * self.config.quantization
            )

        return measurement

    @property
    def current_bias(self) -> float:
        """Current bias value (for debugging/logging)."""
        return self._bias

    @property
    def current_scale_factor(self) -> float:
        """Current scale factor (for debugging/logging)."""
        return self._scale_factor

    def get_state(self) -> dict:
        """Get current sensor state for logging."""
        return {
            "bias": self._bias,
            "scale_factor": self._scale_factor,
            "sample_timer": self._sample_timer,
            "last_sample": self._last_sample,
        }
