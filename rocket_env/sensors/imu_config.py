"""
IMU Configuration Dataclasses

Defines configuration for realistic IMU sensor simulation with
preset values based on common hobby-grade sensors.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class GyroConfig:
    """
    Configuration for a gyroscope sensor model.

    Parameters are based on typical MEMS gyroscope specifications.
    Values should match sensor datasheets for realistic simulation.

    Attributes:
        noise_density: Angular random walk in deg/s/sqrt(Hz).
            Typical values: 0.005-0.1 for MEMS gyros.
        bias_instability: Long-term bias drift in deg/hr.
            Typical values: 1-50 for MEMS gyros.
        bias_initial_range: Initial bias offset range in deg/s.
            This represents manufacturing variation.
        scale_factor_error: Multiplicative gain error as fraction.
            E.g., 0.01 means +/- 1% scale factor error.
        saturation: Maximum measurable rate in deg/s.
            Readings outside this range are clipped.
        quantization: Least significant bit in deg/s.
            Set to 0 to disable quantization effects.
        sample_rate: Sensor sample rate in Hz.
            Used for sample-and-hold when control rate differs.
    """

    noise_density: float = 0.015  # deg/s/sqrt(Hz)
    bias_instability: float = 5.0  # deg/hr
    bias_initial_range: float = 0.5  # deg/s
    scale_factor_error: float = 0.005  # fraction (+/- 0.5%)
    saturation: float = 2000.0  # deg/s
    quantization: float = 0.0  # deg/s (0 = disabled)
    sample_rate: float = 200.0  # Hz


@dataclass
class IMUConfig:
    """
    Complete IMU configuration with preset factory methods.

    This class bundles gyroscope configuration with metadata and
    provides factory methods for common sensor presets.

    Attributes:
        name: Human-readable identifier for the IMU preset.
        gyro: Gyroscope configuration.
    """

    name: str = "default"
    gyro: GyroConfig = field(default_factory=GyroConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "gyro": {
                "noise_density": self.gyro.noise_density,
                "bias_instability": self.gyro.bias_instability,
                "bias_initial_range": self.gyro.bias_initial_range,
                "scale_factor_error": self.gyro.scale_factor_error,
                "saturation": self.gyro.saturation,
                "quantization": self.gyro.quantization,
                "sample_rate": self.gyro.sample_rate,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IMUConfig":
        """Create from dictionary."""
        gyro_data = data.get("gyro", {})
        return cls(
            name=data.get("name", "custom"),
            gyro=GyroConfig(**gyro_data),
        )

    @classmethod
    def icm_20948(cls) -> "IMUConfig":
        """
        InvenSense ICM-20948 9-axis IMU (recommended default).

        This is a high-quality, affordable IMU commonly used in
        hobby drones and robotics. Good balance of performance and cost.

        Datasheet specifications:
        - Gyro noise density: 0.015 deg/s/sqrt(Hz)
        - Bias instability: ~5 deg/hr (typical)
        - Full-scale range: +/- 2000 deg/s
        - Sample rate: up to 1.125 kHz, typical 200 Hz
        """
        return cls(
            name="icm_20948",
            gyro=GyroConfig(
                noise_density=0.015,  # deg/s/sqrt(Hz)
                bias_instability=5.0,  # deg/hr
                bias_initial_range=0.3,  # deg/s
                scale_factor_error=0.005,  # +/- 0.5%
                saturation=2000.0,  # deg/s
                quantization=0.0076,  # 16-bit ADC at 2000 dps
                sample_rate=200.0,  # Hz
            ),
        )

    @classmethod
    def mpu_6050(cls) -> "IMUConfig":
        """
        InvenSense MPU-6050 6-axis IMU (budget option).

        Very popular, low-cost sensor used in many Arduino projects.
        Higher noise than ICM-20948 but widely available.

        Datasheet specifications:
        - Gyro noise density: 0.05 deg/s/sqrt(Hz)
        - Bias instability: ~20 deg/hr (typical)
        - Full-scale range: +/- 2000 deg/s
        - Sample rate: up to 1 kHz
        """
        return cls(
            name="mpu_6050",
            gyro=GyroConfig(
                noise_density=0.05,  # deg/s/sqrt(Hz)
                bias_instability=20.0,  # deg/hr
                bias_initial_range=1.0,  # deg/s
                scale_factor_error=0.03,  # +/- 3%
                saturation=2000.0,  # deg/s
                quantization=0.0076,  # 16-bit ADC at 2000 dps
                sample_rate=1000.0,  # Hz
            ),
        )

    @classmethod
    def bmi088(cls) -> "IMUConfig":
        """
        Bosch BMI088 6-axis IMU (high performance).

        Industrial-grade sensor with excellent specs, used in
        professional drones and robotics applications.

        Datasheet specifications:
        - Gyro noise density: 0.014 deg/s/sqrt(Hz)
        - Bias instability: ~3 deg/hr
        - Full-scale range: +/- 2000 deg/s
        """
        return cls(
            name="bmi088",
            gyro=GyroConfig(
                noise_density=0.014,  # deg/s/sqrt(Hz)
                bias_instability=3.0,  # deg/hr
                bias_initial_range=0.2,  # deg/s
                scale_factor_error=0.003,  # +/- 0.3%
                saturation=2000.0,  # deg/s
                quantization=0.004,  # 16-bit ADC at 2000 dps
                sample_rate=2000.0,  # Hz
            ),
        )

    @classmethod
    def ideal(cls) -> "IMUConfig":
        """
        Ideal/perfect sensor with no noise (for comparison/debugging).

        Use this to verify controller behavior without sensor noise,
        or as a baseline for comparing performance with realistic sensors.
        """
        return cls(
            name="ideal",
            gyro=GyroConfig(
                noise_density=0.0,
                bias_instability=0.0,
                bias_initial_range=0.0,
                scale_factor_error=0.0,
                saturation=10000.0,  # Effectively unlimited
                quantization=0.0,
                sample_rate=10000.0,  # Effectively continuous
            ),
        )

    @classmethod
    def get_preset(cls, name: str) -> "IMUConfig":
        """
        Get IMU configuration by preset name.

        Args:
            name: Preset name ("icm_20948", "mpu_6050", "bmi088", "ideal")

        Returns:
            IMUConfig for the specified preset

        Raises:
            ValueError: If preset name is not recognized
        """
        presets = {
            "icm_20948": cls.icm_20948,
            "mpu_6050": cls.mpu_6050,
            "bmi088": cls.bmi088,
            "ideal": cls.ideal,
        }

        if name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown IMU preset '{name}'. Available: {available}")

        return presets[name]()

    @classmethod
    def custom(
        cls,
        noise_density: float,
        bias_instability: float,
        bias_initial_range: float = 0.5,
        scale_factor_error: float = 0.01,
        saturation: float = 2000.0,
        quantization: float = 0.0,
        sample_rate: float = 200.0,
    ) -> "IMUConfig":
        """
        Create custom IMU configuration.

        Args:
            noise_density: Angular random walk (deg/s/sqrt(Hz))
            bias_instability: Bias drift rate (deg/hr)
            bias_initial_range: Initial bias offset range (deg/s)
            scale_factor_error: Scale factor error as fraction
            saturation: Maximum measurable rate (deg/s)
            quantization: LSB quantization (deg/s)
            sample_rate: Sample rate (Hz)

        Returns:
            Custom IMUConfig instance
        """
        return cls(
            name="custom",
            gyro=GyroConfig(
                noise_density=noise_density,
                bias_instability=bias_instability,
                bias_initial_range=bias_initial_range,
                scale_factor_error=scale_factor_error,
                saturation=saturation,
                quantization=quantization,
                sample_rate=sample_rate,
            ),
        )
