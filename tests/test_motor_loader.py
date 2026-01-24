"""
Tests for motor_loader.py - motor data loading and calculations.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import yaml


class TestMotor:
    """Tests for Motor class."""

    def test_construction(self, sample_motor_config):
        """Test Motor construction from config."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        assert motor.name == "test_motor"
        assert motor.manufacturer == "Test"
        assert motor.designation == "T100"

    def test_si_unit_conversion(self, sample_motor_config):
        """Test that units are converted to SI."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # Diameter: mm to m
        assert motor.diameter == 0.018
        # Length: mm to m
        assert motor.length == 0.070
        # Mass: g to kg
        assert motor.total_mass == 0.024
        assert motor.propellant_mass == 0.012
        assert motor.case_mass == 0.012

    def test_get_thrust_during_burn(self, sample_motor_config):
        """Test thrust retrieval during burn."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # At t=0, thrust should be 0 (start of curve)
        thrust_0 = motor.get_thrust(0.0)
        assert thrust_0 >= 0

        # At t=0.1 (just after ignition), should have thrust
        thrust_01 = motor.get_thrust(0.1)
        assert thrust_01 > 0

        # At t=1.0 (mid-burn), should have thrust
        thrust_1 = motor.get_thrust(1.0)
        assert thrust_1 > 0

    def test_get_thrust_after_burnout(self, sample_motor_config):
        """Test that thrust is zero after burnout."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # After burn time, thrust should be 0
        thrust = motor.get_thrust(motor.burn_time + 1.0)
        assert thrust == 0.0

    def test_get_thrust_before_ignition(self, sample_motor_config):
        """Test that thrust is zero before ignition."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        thrust = motor.get_thrust(-0.1)
        assert thrust == 0.0

    def test_get_mass_at_start(self, sample_motor_config):
        """Test mass at start of burn."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        mass = motor.get_mass(0.0)
        assert mass == motor.total_mass

    def test_get_mass_at_end(self, sample_motor_config):
        """Test mass at end of burn (empty)."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        mass = motor.get_mass(motor.burn_time + 1.0)
        assert mass == motor.case_mass

    def test_get_mass_linear_burn(self, sample_motor_config):
        """Test that mass decreases linearly during burn."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # At 50% burn time, should have ~50% propellant
        mass_mid = motor.get_mass(motor.burn_time / 2)
        expected = motor.case_mass + motor.propellant_mass * 0.5
        assert mass_mid == pytest.approx(expected, rel=0.01)

    def test_thrust_multiplier(self, sample_motor_config):
        """Test that thrust multiplier works."""
        from motor_loader import Motor

        # Without multiplier
        motor_normal = Motor(sample_motor_config)
        thrust_normal = motor_normal.get_thrust(0.5)

        # With 2x multiplier
        config_2x = dict(sample_motor_config)
        config_2x["thrust_multiplier"] = 2.0
        motor_2x = Motor(config_2x)
        thrust_2x = motor_2x.get_thrust(0.5)

        assert thrust_2x == pytest.approx(2 * thrust_normal, rel=0.01)

    def test_repr(self, sample_motor_config):
        """Test string representation."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)
        repr_str = repr(motor)

        assert "Test" in repr_str
        assert "T100" in repr_str
        assert "C-class" in repr_str

    def test_mass_flow_rate(self, sample_motor_config):
        """Test mass flow rate calculation."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # During burn
        mdot = motor.get_mass_flow_rate(0.5)
        assert mdot > 0

        # After burnout
        mdot_end = motor.get_mass_flow_rate(motor.burn_time + 1)
        assert mdot_end == 0.0


class TestLoadMotorFromConfig:
    """Tests for loading motor from config file."""

    def test_load_from_yaml(self, tmp_path, sample_motor_config):
        """Test loading motor from YAML config file."""
        from motor_loader import load_motor_from_config

        config = {"motor": sample_motor_config}
        config_path = tmp_path / "config.yaml"

        with open(config_path, "w") as f:
            yaml.dump(config, f)

        motor = load_motor_from_config(str(config_path))

        assert motor.name == "test_motor"
        assert motor.burn_time == 2.0


class TestLoadMotorFromDict:
    """Tests for loading motor from dictionary."""

    def test_load_from_dict(self, sample_motor_config):
        """Test loading motor from config dict."""
        from motor_loader import load_motor_from_dict

        config = {"motor": sample_motor_config}
        motor = load_motor_from_dict(config)

        assert motor.name == "test_motor"


class TestMotorDefaults:
    """Tests for motor with missing/default values."""

    def test_minimal_config(self):
        """Test motor with minimal configuration."""
        from motor_loader import Motor

        minimal_config = {
            "name": "minimal",
            "thrust_curve": {"time_s": [0, 1], "thrust_N": [0, 10]},
        }

        motor = Motor(minimal_config)

        assert motor.name == "minimal"
        assert motor.manufacturer == "Unknown"

    def test_missing_thrust_curve(self):
        """Test motor with missing thrust curve."""
        from motor_loader import Motor

        config = {"name": "no_curve"}
        motor = Motor(config)

        # Should return 0 thrust
        assert motor.get_thrust(0.5) == 0.0


class TestMotorPhysics:
    """Tests for motor physics calculations."""

    def test_specific_impulse_consistency(self, sample_motor_config):
        """Test that specific impulse is consistent."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # Isp = total_impulse / (propellant_mass * g0)
        # Or equivalently, exhaust velocity = Isp * g0 = total_impulse / propellant_mass
        if motor.propellant_mass > 0:
            exhaust_velocity = motor.total_impulse / motor.propellant_mass
            assert exhaust_velocity > 0

    def test_thrust_curve_bounds(self, sample_motor_config):
        """Test that thrust stays within bounds."""
        from motor_loader import Motor

        motor = Motor(sample_motor_config)

        # Sample thrust at various times
        for t in np.linspace(0, motor.burn_time, 20):
            thrust = motor.get_thrust(t)
            assert thrust >= 0
            assert (
                thrust <= motor.max_thrust * motor.thrust_multiplier * 1.1
            )  # Allow 10% margin


class TestRealMotorConfigs:
    """Tests using actual motor configuration files from the project."""

    def test_load_estes_c6_if_exists(self):
        """Test loading Estes C6 config if it exists."""
        from motor_loader import load_motor_from_config
        from pathlib import Path

        config_path = Path("configs/estes_c6_easy.yaml")
        if not config_path.exists():
            pytest.skip("Config file not found")

        motor = load_motor_from_config(str(config_path))

        # Estes C6 specs
        assert motor.impulse_class == "C"
        assert 5.0 < motor.total_impulse < 15.0  # C class is 5-10 Ns
        assert motor.burn_time > 0
        assert motor.burn_time < 5.0  # Reasonable upper bound
