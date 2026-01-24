"""
Tests for thrustcurve_motor_data module - motor data classes and thrust curve parsing.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


class TestMotorData:
    """Tests for MotorData dataclass."""

    def test_motor_data_creation(self):
        """Test basic MotorData creation."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Estes",
            designation="C6",
            diameter=18.0,  # mm
            length=70.0,  # mm
            total_mass=24.0,  # g
            propellant_mass=12.3,  # g
            case_mass=11.7,  # g
            total_impulse=10.0,  # N·s
            burn_time=1.85,  # s
            average_thrust=5.4,  # N
            max_thrust=14.0,  # N
            time_points=np.array([0.0, 0.1, 1.0, 1.85]),
            thrust_points=np.array([0.0, 14.0, 5.0, 0.0]),
        )

        # Values should be converted to SI units
        assert motor.diameter == pytest.approx(0.018, rel=0.01)  # m
        assert motor.length == pytest.approx(0.070, rel=0.01)  # m
        assert motor.total_mass == pytest.approx(0.024, rel=0.01)  # kg
        assert motor.propellant_mass == pytest.approx(0.0123, rel=0.01)  # kg
        assert motor.case_mass == pytest.approx(0.0117, rel=0.01)  # kg

    def test_thrust_interpolation(self):
        """Test thrust interpolation at various times."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=14.0,
            time_points=np.array([0.0, 0.1, 1.0, 2.0]),
            thrust_points=np.array([0.0, 14.0, 5.0, 0.0]),
        )

        # At t=0, thrust should be 0
        assert motor.get_thrust(0.0) == pytest.approx(0.0, abs=0.1)

        # At t=0.1, thrust should be 14
        assert motor.get_thrust(0.1) == pytest.approx(14.0, abs=0.1)

        # At t=2.0 (end), thrust should be 0
        assert motor.get_thrust(2.0) == pytest.approx(0.0, abs=0.1)

        # After burn time, thrust should be 0
        assert motor.get_thrust(3.0) == pytest.approx(0.0, abs=0.1)

        # Before t=0, thrust should be 0 (bounds handling)
        assert motor.get_thrust(-1.0) == pytest.approx(0.0, abs=0.1)

    def test_mass_calculation(self):
        """Test mass calculation during burn."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,  # g
            propellant_mass=12.0,  # g
            case_mass=12.0,  # g
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
            thrust_points=np.array([0.0, 10.0, 5.0, 5.0, 0.0]),
        )

        # At t=0, mass should be total
        assert motor.get_mass(0.0) == pytest.approx(motor.total_mass, rel=0.01)

        # At t=burn_time, mass should be case mass only
        assert motor.get_mass(2.0) == pytest.approx(motor.case_mass, rel=0.01)

        # After burn, mass should be case mass
        assert motor.get_mass(5.0) == pytest.approx(motor.case_mass, rel=0.01)

        # During burn, mass should be between total and case
        mid_mass = motor.get_mass(1.0)
        assert mid_mass > motor.case_mass
        assert mid_mass < motor.total_mass

    def test_cg_shift(self):
        """Test center of gravity shift calculation."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # At t=0, CG shift should be minimal
        cg_start = motor.get_cg_shift(0.0)

        # At end of burn, CG shift should be positive
        cg_end = motor.get_cg_shift(2.0)

        # CG should shift forward as propellant burns
        assert cg_end >= cg_start

    def test_computed_impulse_verification(self):
        """Test that computed impulse is verified against stated impulse."""
        from thrustcurve_motor_data import MotorData

        # Create motor where computed impulse matches stated impulse
        time_points = np.array([0.0, 1.0, 2.0])
        thrust_points = np.array([0.0, 10.0, 0.0])
        # Trapezoidal area = 10 N·s

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,  # Should match computed
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=time_points,
            thrust_points=thrust_points,
        )

        # Computed impulse should exist
        assert hasattr(motor, "computed_impulse")
        assert motor.computed_impulse > 0

    def test_exhaust_velocity_calculation(self):
        """Test exhaust velocity calculation."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # Exhaust velocity = total_impulse / propellant_mass
        expected_ve = 10.0 / 0.012  # ~833 m/s
        assert motor.exhaust_velocity == pytest.approx(expected_ve, rel=0.01)

    def test_zero_propellant_mass(self):
        """Test handling of zero propellant mass."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=12.0,
            propellant_mass=0.0,  # No propellant
            case_mass=12.0,
            total_impulse=0.0,
            burn_time=0.0,
            average_thrust=0.0,
            max_thrust=0.0,
            time_points=np.array([0.0]),
            thrust_points=np.array([0.0]),
        )

        # Should handle gracefully
        assert motor.exhaust_velocity == 0
        assert motor.get_cg_shift(0.0) == 0.0


class TestThrustCurveParser:
    """Tests for ThrustCurveParser class."""

    def test_parse_eng_file(self, tmp_path):
        """Test parsing .eng file format."""
        from thrustcurve_motor_data import ThrustCurveParser

        # Create a sample .eng file
        eng_content = """; Test motor
C6 18.0 70.0 3-5-7 12.3 24.0 Estes
0.0 0.0
0.05 14.0
0.1 12.0
0.5 6.0
1.0 5.0
1.5 4.0
1.85 0.0
"""
        eng_file = tmp_path / "test_motor.eng"
        eng_file.write_text(eng_content)

        motor = ThrustCurveParser.parse_eng_file(str(eng_file))

        assert motor.manufacturer == "Estes"
        assert motor.designation == "C6"
        assert motor.diameter == pytest.approx(0.018, rel=0.01)  # Converted to m
        assert motor.length == pytest.approx(0.070, rel=0.01)
        assert len(motor.time_points) == 7
        assert len(motor.thrust_points) == 7
        assert motor.delays == [3.0, 5.0, 7.0]
        assert motor.plugged is False

    def test_parse_eng_file_plugged(self, tmp_path):
        """Test parsing .eng file with plugged motor."""
        from thrustcurve_motor_data import ThrustCurveParser

        eng_content = """; Plugged test motor
D12 24.0 70.0 P 21.0 44.0 Estes
0.0 0.0
0.5 30.0
1.0 15.0
1.6 0.0
"""
        eng_file = tmp_path / "plugged_motor.eng"
        eng_file.write_text(eng_content)

        motor = ThrustCurveParser.parse_eng_file(str(eng_file))

        assert motor.designation == "D12"
        assert motor.plugged is True
        assert motor.delays == []

    def test_parse_rse_file(self, tmp_path):
        """Test parsing RockSim .rse file format."""
        from thrustcurve_motor_data import ThrustCurveParser

        rse_content = """<?xml version="1.0"?>
<engine-database>
  <engine-list>
    <engine>
      <manufacturer>AeroTech</manufacturer>
      <designation>F40</designation>
      <diameter>29.0</diameter>
      <length>124.0</length>
      <total-mass>90.0</total-mass>
      <prop-mass>39.0</prop-mass>
      <data>
        <sample><time>0.0</time><thrust>0.0</thrust></sample>
        <sample><time>0.05</time><thrust>65.0</thrust></sample>
        <sample><time>0.5</time><thrust>45.0</thrust></sample>
        <sample><time>1.0</time><thrust>40.0</thrust></sample>
        <sample><time>2.0</time><thrust>0.0</thrust></sample>
      </data>
    </engine>
  </engine-list>
</engine-database>
"""
        rse_file = tmp_path / "test_motor.rse"
        rse_file.write_text(rse_content)

        motor = ThrustCurveParser.parse_rse_file(str(rse_file))

        assert motor.manufacturer == "AeroTech"
        assert motor.designation == "F40"
        assert motor.diameter == pytest.approx(0.029, rel=0.01)
        assert len(motor.time_points) == 5
        assert motor.max_thrust == pytest.approx(65.0, abs=0.1)

    def test_parse_rse_missing_engine(self, tmp_path):
        """Test parsing .rse file with no engine element."""
        from thrustcurve_motor_data import ThrustCurveParser

        rse_content = """<?xml version="1.0"?>
<engine-database>
  <engine-list>
  </engine-list>
</engine-database>
"""
        rse_file = tmp_path / "empty.rse"
        rse_file.write_text(rse_content)

        with pytest.raises(ValueError, match="No engine data found"):
            ThrustCurveParser.parse_rse_file(str(rse_file))

    def test_eng_file_with_comments(self, tmp_path):
        """Test that comments are properly skipped in .eng files."""
        from thrustcurve_motor_data import ThrustCurveParser

        eng_content = """; First comment line
; Second comment
B6 18.0 70.0 3-5 5.6 19.5 Estes
; Comment in middle (should be ignored in data section)
0.0 0.0
0.1 12.0
0.8 0.0
"""
        eng_file = tmp_path / "commented.eng"
        eng_file.write_text(eng_content)

        motor = ThrustCurveParser.parse_eng_file(str(eng_file))

        assert motor.designation == "B6"
        # The comment line ";comment in middle" should be skipped
        # so we should have 3 data points
        assert len(motor.time_points) == 3


class TestMotorDataEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_data_point(self):
        """Test motor with single data point."""
        from thrustcurve_motor_data import MotorData

        # Single point thrust curve shouldn't compute impulse well
        # but should still work
        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=10.0,
            time_points=np.array([0.5]),
            thrust_points=np.array([10.0]),
        )

        # Should still be able to get thrust at the single point
        assert motor.get_thrust(0.5) == pytest.approx(10.0, abs=0.1)

    def test_motor_with_delays_list(self):
        """Test that delays list is preserved."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0]),
            thrust_points=np.array([10.0, 0.0]),
            delays=[3.0, 5.0, 7.0],
        )

        assert motor.delays == [3.0, 5.0, 7.0]

    def test_sparky_motor(self):
        """Test sparky motor flag."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0]),
            thrust_points=np.array([10.0, 0.0]),
            sparky=True,
        )

        assert motor.sparky is True

    def test_negative_time_thrust_lookup(self):
        """Test thrust lookup at negative time."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 0.5, 1.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # Negative time should return 0 (fill_value)
        assert motor.get_thrust(-0.5) == pytest.approx(0.0, abs=0.01)

    def test_mass_negative_time(self):
        """Test mass calculation at negative time."""
        from thrustcurve_motor_data import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0]),
            thrust_points=np.array([10.0, 0.0]),
        )

        # Negative time should return total mass
        assert motor.get_mass(-1.0) == motor.total_mass
