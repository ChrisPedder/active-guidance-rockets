"""
Tests for IMU configuration dataclasses.
"""

import pytest
from rocket_env.sensors import IMUConfig, GyroConfig


class TestGyroConfig:
    """Tests for GyroConfig dataclass."""

    def test_default_values(self):
        """Test default gyro configuration values."""
        config = GyroConfig()
        assert config.noise_density == 0.015
        assert config.bias_instability == 5.0
        assert config.saturation == 2000.0
        assert config.sample_rate == 200.0

    def test_custom_values(self):
        """Test custom gyro configuration."""
        config = GyroConfig(
            noise_density=0.1,
            bias_instability=50.0,
            saturation=500.0,
        )
        assert config.noise_density == 0.1
        assert config.bias_instability == 50.0
        assert config.saturation == 500.0


class TestIMUConfig:
    """Tests for IMUConfig dataclass."""

    def test_default_config(self):
        """Test default IMU configuration."""
        config = IMUConfig()
        assert config.name == "default"
        assert isinstance(config.gyro, GyroConfig)

    def test_icm_20948_preset(self):
        """Test ICM-20948 preset values."""
        config = IMUConfig.icm_20948()
        assert config.name == "icm_20948"
        assert config.gyro.noise_density == 0.015
        assert config.gyro.bias_instability == 5.0
        assert config.gyro.saturation == 2000.0

    def test_mpu_6050_preset(self):
        """Test MPU-6050 preset values."""
        config = IMUConfig.mpu_6050()
        assert config.name == "mpu_6050"
        assert config.gyro.noise_density == 0.05
        assert config.gyro.bias_instability == 20.0

    def test_bmi088_preset(self):
        """Test BMI088 preset values."""
        config = IMUConfig.bmi088()
        assert config.name == "bmi088"
        assert config.gyro.noise_density == 0.014
        assert config.gyro.bias_instability == 3.0

    def test_ideal_preset(self):
        """Test ideal (no noise) preset."""
        config = IMUConfig.ideal()
        assert config.name == "ideal"
        assert config.gyro.noise_density == 0.0
        assert config.gyro.bias_instability == 0.0
        assert config.gyro.scale_factor_error == 0.0

    def test_get_preset(self):
        """Test preset lookup by name."""
        config = IMUConfig.get_preset("icm_20948")
        assert config.name == "icm_20948"

        config = IMUConfig.get_preset("mpu_6050")
        assert config.name == "mpu_6050"

    def test_get_preset_invalid(self):
        """Test invalid preset name raises error."""
        with pytest.raises(ValueError, match="Unknown IMU preset"):
            IMUConfig.get_preset("invalid_preset")

    def test_custom_factory(self):
        """Test custom IMU configuration factory."""
        config = IMUConfig.custom(
            noise_density=0.02,
            bias_instability=10.0,
            saturation=1000.0,
        )
        assert config.name == "custom"
        assert config.gyro.noise_density == 0.02
        assert config.gyro.bias_instability == 10.0
        assert config.gyro.saturation == 1000.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = IMUConfig.icm_20948()
        d = config.to_dict()

        assert d["name"] == "icm_20948"
        assert "gyro" in d
        assert d["gyro"]["noise_density"] == 0.015

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {
            "name": "test",
            "gyro": {
                "noise_density": 0.03,
                "bias_instability": 15.0,
            },
        }
        config = IMUConfig.from_dict(d)

        assert config.name == "test"
        assert config.gyro.noise_density == 0.03
        assert config.gyro.bias_instability == 15.0

    def test_roundtrip_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        original = IMUConfig.icm_20948()
        d = original.to_dict()
        restored = IMUConfig.from_dict(d)

        assert restored.name == original.name
        assert restored.gyro.noise_density == original.gyro.noise_density
        assert restored.gyro.bias_instability == original.gyro.bias_instability
