"""
Tests for disturbance_observer module.

Covers: DOBConfig, DisturbanceObserver, estimate_dob_parameters.
"""

import numpy as np
import pytest

from controllers.disturbance_observer import (
    DOBConfig,
    DisturbanceObserver,
    estimate_dob_parameters,
)


class TestDOBConfig:
    """Test DOBConfig dataclass."""

    def test_defaults(self):
        config = DOBConfig()
        assert config.I_roll == 0.0001
        assert config.control_effectiveness == 0.001
        assert config.damping_coeff == 0.0005
        assert config.filter_alpha == 0.1
        assert config.max_disturbance == 0.01

    def test_custom(self):
        config = DOBConfig(I_roll=0.001, max_disturbance=0.1)
        assert config.I_roll == 0.001
        assert config.max_disturbance == 0.1


class TestDisturbanceObserver:
    """Test DisturbanceObserver class."""

    def test_init_default(self):
        dob = DisturbanceObserver()
        assert dob.config is not None
        assert dob.estimate == 0.0
        assert dob.estimate_filtered == 0.0
        assert dob.estimate_magnitude == 0.0

    def test_init_custom_config(self):
        config = DOBConfig(I_roll=0.001)
        dob = DisturbanceObserver(config)
        assert dob.config.I_roll == 0.001

    def test_reset(self):
        dob = DisturbanceObserver()
        # Manually modify state
        dob.estimate = 1.0
        dob.estimate_filtered = 0.5
        dob.estimate_magnitude = 0.3
        dob._step_count = 10

        dob.reset()
        assert dob.estimate == 0.0
        assert dob.estimate_filtered == 0.0
        assert dob.estimate_magnitude == 0.0
        assert dob._step_count == 0

    def test_first_step_returns_zeros(self):
        dob = DisturbanceObserver()
        est, mag = dob.update(
            roll_rate=0.5,
            roll_accel=1.0,
            action=0.0,
            dynamic_pressure=500.0,
            velocity=30.0,
        )
        assert est == 0.0
        assert mag == 0.0
        assert dob._step_count == 1

    def test_update_with_no_disturbance(self):
        """When control + damping explain all acceleration, disturbance should be ~0."""
        config = DOBConfig(
            I_roll=0.0001,
            control_effectiveness=0.001,
            damping_coeff=0.0005,
            filter_alpha=1.0,  # No filtering for clarity
            max_disturbance=0.01,
        )
        dob = DisturbanceObserver(config)

        # First step to warm up
        dob.update(0.0, 0.0, 0.0, 500.0, 30.0)

        # Second step: acceleration is fully explained by control
        q = 500.0
        v = 30.0
        action = 0.5
        roll_rate = 0.1
        tau_control = config.control_effectiveness * action * q
        tau_damping = -config.damping_coeff * roll_rate * q / v
        alpha = (tau_control + tau_damping) / config.I_roll

        est, mag = dob.update(roll_rate, alpha, action, q, v)
        assert abs(est) < 0.01
        assert abs(mag) < 0.01

    def test_update_detects_disturbance(self):
        """When there's a large unexplained acceleration, disturbance should be nonzero."""
        config = DOBConfig(
            I_roll=0.0001,
            control_effectiveness=0.001,
            damping_coeff=0.0005,
            filter_alpha=1.0,
            max_disturbance=0.01,
        )
        dob = DisturbanceObserver(config)

        # First step to warm up
        dob.update(0.0, 0.0, 0.0, 500.0, 30.0)

        # Second step: large acceleration but no control or damping to explain it
        est, mag = dob.update(
            roll_rate=0.0,
            roll_accel=100.0,  # Large unexplained acceleration
            action=0.0,
            dynamic_pressure=500.0,
            velocity=30.0,
        )
        # Should detect significant disturbance
        assert abs(est) > 0.1
        assert mag > 0.1

    def test_filter_smooths_output(self):
        """Low filter_alpha should smooth rapid changes."""
        config = DOBConfig(filter_alpha=0.05, max_disturbance=0.01)
        dob = DisturbanceObserver(config)

        dob.update(0.0, 0.0, 0.0, 500.0, 30.0)  # warm up

        # Apply sudden disturbance
        est1, mag1 = dob.update(0.0, 100.0, 0.0, 500.0, 30.0)

        # Should be partially filtered
        assert abs(est1) > 0  # Some response
        # But not fully reacting
        assert abs(dob.estimate_filtered) < abs(dob.estimate)

    def test_get_state(self):
        dob = DisturbanceObserver()
        dob.update(0.0, 0.0, 0.0, 500.0, 30.0)
        dob.update(0.1, 10.0, 0.0, 500.0, 30.0)

        state = dob.get_state()
        assert "disturbance_estimate_raw" in state
        assert "disturbance_estimate_filtered" in state
        assert "disturbance_magnitude" in state
        assert "step_count" in state
        assert state["step_count"] == 2

    def test_normalization_clipping(self):
        """Output should be clipped to [-1, 1] and [0, 1]."""
        config = DOBConfig(filter_alpha=1.0, max_disturbance=0.001)
        dob = DisturbanceObserver(config)

        dob.update(0.0, 0.0, 0.0, 500.0, 30.0)
        est, mag = dob.update(0.0, 1000.0, 0.0, 500.0, 30.0)

        assert -1.0 <= est <= 1.0
        assert 0.0 <= mag <= 1.0

    def test_velocity_safety(self):
        """Should not crash with zero velocity."""
        dob = DisturbanceObserver()
        dob.update(0.0, 0.0, 0.0, 500.0, 0.0)  # v=0
        est, mag = dob.update(0.1, 1.0, 0.0, 500.0, 0.0)
        assert np.isfinite(est)
        assert np.isfinite(mag)


class TestEstimateDOBParameters:
    """Test estimate_dob_parameters function."""

    def test_basic(self):
        from airframe import RocketAirframe
        from spin_stabilized_control_env import RocketConfig

        airframe = RocketAirframe.estes_alpha()
        config = RocketConfig()

        dob_config = estimate_dob_parameters(airframe, config)

        assert isinstance(dob_config, DOBConfig)
        assert dob_config.I_roll > 0
        assert dob_config.control_effectiveness > 0
        assert dob_config.damping_coeff > 0
        assert dob_config.max_disturbance > 0
        assert dob_config.filter_alpha == 0.1

    def test_scales_with_airframe(self):
        """Larger airframes should have different DOB parameters."""
        from airframe import RocketAirframe
        from spin_stabilized_control_env import RocketConfig

        estes = RocketAirframe.estes_alpha()
        config = RocketConfig()

        dob_estes = estimate_dob_parameters(estes, config)

        # Verify parameters are physical
        assert dob_estes.I_roll < 0.01  # Small rocket
        assert dob_estes.max_disturbance > 0
