"""
Tests for gyroscope noise model.
"""

import pytest
import numpy as np
from rocket_env.sensors import GyroConfig, GyroModel


class TestGyroModel:
    """Tests for GyroModel class."""

    def test_initialization(self):
        """Test gyro model initialization."""
        config = GyroConfig()
        model = GyroModel(config, dt=0.01)

        assert model.config == config
        assert model.dt == 0.01

    def test_reset(self):
        """Test reset initializes bias and scale factor."""
        config = GyroConfig(bias_initial_range=1.0, scale_factor_error=0.1)
        model = GyroModel(config, dt=0.01)

        # Initial values should be default
        assert model.current_bias == 0.0
        assert model.current_scale_factor == 1.0

        # After reset, should have random bias and scale factor
        model.reset(seed=42)
        assert model.current_bias != 0.0
        assert model.current_scale_factor != 1.0

        # Bias should be within range
        assert abs(model.current_bias) <= 1.0

        # Scale factor should be within range
        assert 0.9 <= model.current_scale_factor <= 1.1

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        config = GyroConfig()
        model1 = GyroModel(config, dt=0.01)
        model2 = GyroModel(config, dt=0.01)

        model1.reset(seed=42)
        model2.reset(seed=42)

        # Same seed should give same bias
        assert model1.current_bias == model2.current_bias
        assert model1.current_scale_factor == model2.current_scale_factor

        # Same measurements
        true_rate = 100.0
        m1 = model1.measure(true_rate)
        m2 = model2.measure(true_rate)
        assert m1 == m2

    def test_ideal_sensor_no_noise(self):
        """Test ideal sensor passes through true value."""
        config = GyroConfig(
            noise_density=0.0,
            bias_instability=0.0,
            bias_initial_range=0.0,
            scale_factor_error=0.0,
            saturation=10000.0,
            quantization=0.0,
        )
        model = GyroModel(config, dt=0.01)
        model.reset()

        # Ideal sensor should return exactly the true rate
        true_rate = 45.678
        measured = model.measure(true_rate)
        assert np.isclose(measured, true_rate, rtol=1e-6)

    def test_saturation(self):
        """Test saturation clipping."""
        config = GyroConfig(
            noise_density=0.0,
            bias_instability=0.0,
            bias_initial_range=0.0,
            scale_factor_error=0.0,
            saturation=100.0,
            quantization=0.0,
        )
        model = GyroModel(config, dt=0.01)
        model.reset()

        # Rate above saturation should be clipped
        measured = model.measure(200.0)
        assert measured == 100.0

        measured = model.measure(-200.0)
        assert measured == -100.0

    def test_quantization(self):
        """Test quantization effects."""
        config = GyroConfig(
            noise_density=0.0,
            bias_instability=0.0,
            bias_initial_range=0.0,
            scale_factor_error=0.0,
            saturation=10000.0,
            quantization=1.0,  # 1 deg/s LSB
        )
        model = GyroModel(config, dt=0.01)
        model.reset()

        # Values should be quantized to nearest integer
        measured = model.measure(45.3)
        assert measured == 45.0

        measured = model.measure(45.7)
        assert measured == 46.0

    def test_noise_adds_variance(self):
        """Test that noise increases measurement variance."""
        config_noisy = GyroConfig(noise_density=0.05, bias_initial_range=0.0)
        config_quiet = GyroConfig(noise_density=0.0, bias_initial_range=0.0)

        model_noisy = GyroModel(config_noisy, dt=0.01)
        model_quiet = GyroModel(config_quiet, dt=0.01)

        model_noisy.reset(seed=42)
        model_quiet.reset(seed=42)

        # Collect many measurements
        true_rate = 50.0
        noisy_measurements = []
        quiet_measurements = []

        for _ in range(1000):
            noisy_measurements.append(model_noisy.measure(true_rate))
            quiet_measurements.append(model_quiet.measure(true_rate))

        # Noisy should have more variance
        noisy_std = np.std(noisy_measurements)
        quiet_std = np.std(quiet_measurements)

        assert noisy_std > quiet_std
        assert noisy_std > 0.1  # Should have measurable noise

    def test_bias_affects_mean(self):
        """Test that bias shifts the mean measurement."""
        config = GyroConfig(
            noise_density=0.0,
            bias_initial_range=2.0,
            bias_instability=0.0,
            scale_factor_error=0.0,
        )
        model = GyroModel(config, dt=0.01)
        model.reset(seed=42)

        bias = model.current_bias

        # Measurement should be offset by bias
        true_rate = 50.0
        measured = model.measure(true_rate)
        assert np.isclose(measured, true_rate + bias, rtol=1e-6)

    def test_scale_factor_affects_slope(self):
        """Test that scale factor affects measurement slope."""
        config = GyroConfig(
            noise_density=0.0,
            bias_initial_range=0.0,
            bias_instability=0.0,
            scale_factor_error=0.1,
        )
        model = GyroModel(config, dt=0.01)
        model.reset(seed=42)

        scale = model.current_scale_factor

        # Measurement should be scaled
        true_rate = 100.0
        measured = model.measure(true_rate)
        assert np.isclose(measured, true_rate * scale, rtol=1e-6)

    def test_get_state(self):
        """Test get_state returns sensor state."""
        config = GyroConfig()
        model = GyroModel(config, dt=0.01)
        model.reset(seed=42)
        model.measure(50.0)

        state = model.get_state()

        assert "bias" in state
        assert "scale_factor" in state
        assert "sample_timer" in state
        assert "last_sample" in state


class TestGyroModelRealisticBehavior:
    """Tests for realistic gyro behavior."""

    def test_bias_drift_over_time(self):
        """Test that bias drifts over time with bias instability."""
        config = GyroConfig(
            noise_density=0.0,
            bias_instability=100.0,  # High drift for testing
            bias_initial_range=0.0,
            scale_factor_error=0.0,
        )
        model = GyroModel(config, dt=0.01)
        model.reset(seed=42)

        initial_bias = model.current_bias

        # Simulate many steps
        for _ in range(10000):
            model.measure(0.0)

        final_bias = model.current_bias

        # Bias should have drifted
        assert abs(final_bias - initial_bias) > 0.01

    def test_noise_statistics(self):
        """Test that noise statistics match expected values."""
        config = GyroConfig(
            noise_density=0.05,  # deg/s/sqrt(Hz)
            sample_rate=200.0,  # Hz
            bias_initial_range=0.0,
            bias_instability=0.0,
            scale_factor_error=0.0,
        )
        model = GyroModel(config, dt=0.005)  # 200 Hz
        model.reset(seed=42)

        # Expected std: noise_density * sqrt(sample_rate) = 0.05 * sqrt(200) ≈ 0.707
        expected_std = 0.05 * np.sqrt(200.0)

        # Collect measurements
        measurements = []
        true_rate = 0.0
        for _ in range(10000):
            measurements.append(model.measure(true_rate))

        measured_std = np.std(measurements)

        # Should be within 10% of expected
        assert np.isclose(measured_std, expected_std, rtol=0.1)
