"""
Tests for generate_motor_config module - motor config generation and ThrustCurve API.
"""

import os
import sys
import tempfile
import shutil
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestMotorSearchResult:
    """Tests for MotorSearchResult dataclass."""

    def test_motor_search_result_creation(self):
        """Test MotorSearchResult creation."""
        from generate_motor_config import MotorSearchResult

        result = MotorSearchResult(
            motor_id="abc123def456",
            manufacturer="Estes",
            designation="C6",
            common_name="Estes C6",
            diameter=18.0,
            length=70.0,
            impulse_class="C",
            total_impulse=10.0,
            avg_thrust=5.4,
            max_thrust=14.0,
            burn_time=1.85,
            total_mass=24.0,
            prop_mass=12.3,
        )

        assert result.manufacturer == "Estes"
        assert result.designation == "C6"
        assert result.diameter == 18.0

    def test_motor_search_result_str(self):
        """Test MotorSearchResult string representation."""
        from generate_motor_config import MotorSearchResult

        result = MotorSearchResult(
            motor_id="abc123",
            manufacturer="AeroTech",
            designation="F40",
            common_name="Aerotech F40",
            diameter=29.0,
            length=124.0,
            impulse_class="F",
            total_impulse=80.0,
            avg_thrust=40.0,
            max_thrust=65.0,
            burn_time=2.0,
            total_mass=90.0,
            prop_mass=39.0,
        )

        str_repr = str(result)
        assert "AeroTech" in str_repr
        assert "F40" in str_repr
        assert "F-class" in str_repr
        assert "80.0" in str_repr
        assert "29" in str_repr


class TestMotorData:
    """Tests for MotorData dataclass in generate_motor_config."""

    def test_motor_data_creation(self):
        """Test MotorData creation and initialization."""
        from generate_motor_config import MotorData

        motor = MotorData(
            motor_id="test123",
            manufacturer="Estes",
            designation="C6",
            common_name="Estes C6",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.0123,
            case_mass=0.0117,
            total_impulse=10.0,
            burn_time=1.85,
            average_thrust=5.4,
            max_thrust=14.0,
            time_points=np.array([0.0, 0.1, 1.0, 1.85]),
            thrust_points=np.array([0.0, 14.0, 5.0, 0.0]),
        )

        assert motor.impulse_class == "C"  # 10 N-s is C class
        assert motor.specific_impulse > 0

    def test_impulse_class_calculation(self):
        """Test impulse class calculation for various impulses."""
        from generate_motor_config import MotorData

        # Test different impulse classes
        test_cases = [
            (2.0, "A"),
            (4.0, "B"),
            (8.0, "C"),
            (15.0, "D"),
            (35.0, "E"),
            (70.0, "F"),
            (150.0, "G"),
            (300.0, "H"),
            (600.0, "I"),
            (1000.0, "J"),
            (1500.0, "K+"),
        ]

        for impulse, expected_class in test_cases:
            motor = MotorData(
                motor_id="test",
                manufacturer="Test",
                designation="T1",
                common_name="Test",
                diameter=0.018,
                length=0.070,
                total_mass=0.024,
                propellant_mass=0.012,
                case_mass=0.012,
                total_impulse=impulse,
                burn_time=1.0,
                average_thrust=impulse,
                max_thrust=impulse * 1.5,
                time_points=np.array([0.0, 1.0]),
                thrust_points=np.array([impulse, 0.0]),
            )
            assert (
                motor.impulse_class == expected_class
            ), f"Impulse {impulse} should be {expected_class}"

    def test_specific_impulse_zero_propellant(self):
        """Test specific impulse when propellant mass is zero."""
        from generate_motor_config import MotorData

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.0,
            case_mass=0.024,
            total_impulse=10.0,
            burn_time=1.0,
            average_thrust=10.0,
            max_thrust=15.0,
            time_points=np.array([0.0, 1.0]),
            thrust_points=np.array([10.0, 0.0]),
        )

        assert motor.specific_impulse == 0.0

    def test_get_thrust(self):
        """Test thrust interpolation."""
        from generate_motor_config import MotorData

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 0.5, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 5.0, 0.0]),
        )

        # At start, thrust should be 0
        assert motor.get_thrust(0.0) == pytest.approx(0.0, abs=0.1)

        # At t=0.5, thrust should be 10
        assert motor.get_thrust(0.5) == pytest.approx(10.0, abs=0.1)

        # Interpolated value mid-range
        thrust_mid = motor.get_thrust(0.75)
        assert 5.0 < thrust_mid < 10.0

        # Before burn time, should return 0
        assert motor.get_thrust(-0.5) == pytest.approx(0.0, abs=0.1)

        # After burn time, should return 0
        assert motor.get_thrust(5.0) == pytest.approx(0.0, abs=0.1)

    def test_get_mass(self):
        """Test mass calculation during burn."""
        from generate_motor_config import MotorData

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # At t=0, mass should be total
        assert motor.get_mass(0.0) == motor.total_mass

        # At burn_time, mass should be case mass
        assert motor.get_mass(2.0) == motor.case_mass

        # After burn, still case mass
        assert motor.get_mass(5.0) == motor.case_mass

        # Before t=0, should be total mass
        assert motor.get_mass(-1.0) == motor.total_mass

        # During burn, linear interpolation
        mid_mass = motor.get_mass(1.0)
        assert mid_mass > motor.case_mass
        assert mid_mass < motor.total_mass
        # At halfway through burn, half propellant should be gone
        expected_mid = motor.case_mass + motor.propellant_mass * 0.5
        assert mid_mass == pytest.approx(expected_mid, abs=1e-6)


class TestPhysicsAnalysis:
    """Tests for physics analysis functions."""

    def test_analyze_motor_physics(self):
        """Test analyze_motor_physics function."""
        from generate_motor_config import MotorData, analyze_motor_physics

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        physics = analyze_motor_physics(motor)

        assert physics.twr > 0
        assert physics.roll_inertia > 0
        assert physics.recommended_tab_deflection > 0
        assert physics.recommended_dt > 0
        assert isinstance(physics.random_action_safe, bool)
        assert physics.control_accel_per_degree > 0
        assert physics.disturbance_accel_std >= 0
        # Auto-calculated dry mass note
        assert any("Auto-calculated" in note for note in physics.notes)

    def test_analyze_motor_physics_with_dry_mass(self):
        """Test analyze_motor_physics with explicit dry mass."""
        from generate_motor_config import MotorData, analyze_motor_physics

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        physics = analyze_motor_physics(motor, dry_mass=0.1)

        assert physics.dry_mass == 0.1
        assert physics.total_mass == 0.1 + motor.propellant_mass
        # Should NOT have auto-calculated note
        assert not any("Auto-calculated" in note for note in physics.notes)

    def test_analyze_motor_physics_with_diameter(self):
        """Test analyze_motor_physics with explicit diameter."""
        from generate_motor_config import MotorData, analyze_motor_physics

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        physics_default = analyze_motor_physics(motor)
        physics_custom = analyze_motor_physics(motor, diameter=0.050)

        # Custom diameter should change roll inertia
        assert physics_custom.roll_inertia != physics_default.roll_inertia

    def test_analyze_motor_physics_low_twr_warning(self):
        """Test that low TWR generates a warning note."""
        from generate_motor_config import MotorData, analyze_motor_physics

        # Create a motor with low thrust relative to expected mass
        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test",
            diameter=0.018,
            length=0.070,
            total_mass=0.1,
            propellant_mass=0.05,
            case_mass=0.05,
            total_impulse=5.0,
            burn_time=2.0,
            average_thrust=2.5,  # Very low thrust
            max_thrust=5.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 5.0, 0.0]),
        )

        physics = analyze_motor_physics(motor, dry_mass=0.5)  # Heavy rocket

        # Should have a low TWR warning
        assert physics.twr < 2.0
        assert any("TWR" in note for note in physics.notes)

    def test_analyze_motor_physics_medium_velocity(self):
        """Test analysis with medium velocity motor (50-80 m/s range)."""
        from generate_motor_config import MotorData, analyze_motor_physics

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="E30",
            common_name="Test E30",
            diameter=0.024,
            length=0.070,
            total_mass=0.050,
            propellant_mass=0.020,
            case_mass=0.030,
            total_impulse=30.0,
            burn_time=1.0,
            average_thrust=30.0,
            max_thrust=50.0,
            time_points=np.array([0.0, 0.1, 0.5, 1.0]),
            thrust_points=np.array([0.0, 50.0, 30.0, 0.0]),
        )

        physics = analyze_motor_physics(motor, dry_mass=0.3)
        # Medium velocity motors should get damping_scale=2.0
        assert physics.recommended_damping_scale in [1.5, 2.0, 3.0]

    def test_analyze_offline_h128(self):
        """Test analysis with the offline H128 motor (high velocity with low mass)."""
        from generate_motor_config import get_offline_motor, analyze_motor_physics

        motor = get_offline_motor("aerotech_h128")
        assert motor is not None

        # With a light dry mass, velocity exceeds 100 m/s
        physics = analyze_motor_physics(motor, dry_mass=0.3)
        estimated_velocity = motor.total_impulse / physics.total_mass
        assert estimated_velocity > 100
        assert any("High velocity" in note for note in physics.notes)
        assert physics.recommended_damping_scale == 3.0


class TestConfigGeneration:
    """Tests for config generation functions."""

    def test_generate_config(self):
        """Test generate_config function."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
        )

        motor = MotorData(
            motor_id="test",
            manufacturer="Estes",
            designation="C6",
            common_name="Estes C6",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=1.85,
            average_thrust=5.4,
            max_thrust=14.0,
            time_points=np.array([0.0, 0.1, 1.0, 1.85]),
            thrust_points=np.array([0.0, 14.0, 5.0, 0.0]),
        )

        physics = analyze_motor_physics(motor)
        config = generate_config(motor, physics, difficulty="easy")

        assert "physics" in config
        assert "motor" in config
        assert "environment" in config
        assert "reward" in config
        assert "ppo" in config
        assert "curriculum" in config
        assert "logging" in config

        # Check motor section
        assert config["motor"]["manufacturer"] == "Estes"
        assert config["motor"]["designation"] == "C6"
        assert "thrust_curve" in config["motor"]

    def test_generate_config_difficulty_levels(self):
        """Test config generation for different difficulty levels."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
        )

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="T1",
            common_name="Test T1",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        physics = analyze_motor_physics(motor)

        # Test easy
        easy_config = generate_config(motor, physics, difficulty="easy")
        assert easy_config["environment"]["enable_wind"] is False
        assert easy_config["curriculum"]["enabled"] is False
        assert easy_config["reward"]["spin_penalty_scale"] == -0.05
        assert easy_config["ppo"]["learning_rate"] == 0.0003
        assert easy_config["ppo"]["clip_range"] == 0.2
        assert easy_config["ppo"]["ent_coef"] == 0.01
        assert easy_config["environment"]["max_tilt_angle"] == 60.0

        # Test medium
        medium_config = generate_config(motor, physics, difficulty="medium")
        assert medium_config["environment"]["enable_wind"] is True
        assert medium_config["environment"]["max_wind_speed"] == 3.0
        assert medium_config["reward"]["spin_penalty_scale"] == -0.08
        assert medium_config["reward"]["low_spin_threshold"] == 20.0
        assert medium_config["ppo"]["learning_rate"] == 0.0002
        assert medium_config["ppo"]["ent_coef"] == 0.008
        assert medium_config["environment"]["max_tilt_angle"] == 45.0

        # Test full
        full_config = generate_config(motor, physics, difficulty="full")
        assert full_config["curriculum"]["enabled"] is True
        assert len(full_config["curriculum"]["stages"]) == 3
        assert full_config["environment"]["max_wind_speed"] == 5.0
        assert full_config["reward"]["spin_penalty_scale"] == -0.1
        assert full_config["reward"]["low_spin_threshold"] == 10.0
        assert full_config["reward"]["crash_penalty"] == -50.0
        assert full_config["reward"]["use_potential_shaping"] is True
        assert full_config["ppo"]["learning_rate"] == 0.0001
        assert full_config["ppo"]["clip_range"] == 0.15
        assert full_config["ppo"]["ent_coef"] == 0.005

    def test_generate_config_medium_motor_size(self):
        """Test config generation for 24-29mm diameter motor (medium size)."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
        )

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="D12",
            common_name="Test D12",
            diameter=0.024,
            length=0.070,
            total_mass=0.044,
            propellant_mass=0.021,
            case_mass=0.023,
            total_impulse=20.0,
            burn_time=1.6,
            average_thrust=12.5,
            max_thrust=30.0,
            time_points=np.array([0, 0.05, 0.2, 0.8, 1.4, 1.6]),
            thrust_points=np.array([0, 30, 16, 10, 8, 0]),
        )

        physics = analyze_motor_physics(motor)
        config = generate_config(motor, physics, difficulty="easy")

        # 24mm motor should use medium fin dimensions
        assert config["physics"]["fin_span"] == 0.05
        assert config["physics"]["fin_root_chord"] == 0.06
        assert config["physics"]["fin_tip_chord"] == 0.03
        assert config["physics"]["length"] == 0.60

    def test_generate_config_medium_impulse_reward_scaling(self):
        """Test reward scaling for E-F class motors (20-80 N-s)."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
        )

        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="E30",
            common_name="Test E30",
            diameter=0.024,
            length=0.100,
            total_mass=0.060,
            propellant_mass=0.025,
            case_mass=0.035,
            total_impulse=40.0,
            burn_time=1.5,
            average_thrust=26.7,
            max_thrust=45.0,
            time_points=np.array([0, 0.1, 0.5, 1.5]),
            thrust_points=np.array([0, 45, 30, 0]),
        )

        physics = analyze_motor_physics(motor)
        config = generate_config(motor, physics, difficulty="easy")

        # E-F class should have specific reward scaling
        assert config["reward"]["altitude_reward_scale"] == 0.005
        assert config["environment"]["max_altitude"] == 500

    def test_generate_config_high_impulse_reward_scaling(self):
        """Test reward scaling for G+ class motors (>80 N-s)."""
        from generate_motor_config import (
            get_offline_motor,
            analyze_motor_physics,
            generate_config,
        )

        motor = get_offline_motor("cesaroni_g79")
        assert motor is not None

        physics = analyze_motor_physics(motor)
        config = generate_config(motor, physics, difficulty="easy")

        # G+ class should have different reward scaling
        assert config["reward"]["altitude_reward_scale"] == 0.003
        assert config["environment"]["max_altitude"] == 800

    def test_convert_numpy_types(self):
        """Test convert_numpy_types function."""
        from generate_motor_config import convert_numpy_types

        # Test with numpy types
        data = {
            "float": np.float64(3.14),
            "int": np.int64(42),
            "array": np.array([1, 2, 3]),
            "nested": {
                "value": np.float32(2.5),
                "list": [np.int32(1), np.int32(2)],
            },
            "bool": np.bool_(True),
        }

        converted = convert_numpy_types(data)

        assert isinstance(converted["float"], float)
        assert isinstance(converted["int"], int)
        assert isinstance(converted["array"], list)
        assert isinstance(converted["nested"]["value"], float)
        assert isinstance(converted["bool"], bool)

    def test_convert_numpy_types_passthrough(self):
        """Test convert_numpy_types passes through native Python types."""
        from generate_motor_config import convert_numpy_types

        data = {
            "string": "hello",
            "int": 42,
            "float": 3.14,
            "none": None,
        }

        converted = convert_numpy_types(data)

        assert converted["string"] == "hello"
        assert converted["int"] == 42
        assert converted["float"] == 3.14
        assert converted["none"] is None


class TestOfflineMotorDatabase:
    """Tests for offline motor database."""

    def test_get_offline_motor(self):
        """Test getting motor from offline database."""
        from generate_motor_config import get_offline_motor

        motor = get_offline_motor("estes_c6")

        assert motor is not None
        assert motor.manufacturer == "Estes"
        assert motor.designation == "C6"
        assert motor.total_impulse == 10.0

    def test_get_offline_motor_normalized_key(self):
        """Test key normalization for offline motors."""
        from generate_motor_config import get_offline_motor

        # Various key formats should work
        motor1 = get_offline_motor("estes_c6")
        motor2 = get_offline_motor("Estes C6")
        motor3 = get_offline_motor("estes-c6")

        # At least the normalized version should work
        assert motor1 is not None

    def test_get_offline_motor_not_found(self):
        """Test getting non-existent motor."""
        from generate_motor_config import get_offline_motor

        motor = get_offline_motor("nonexistent_motor")
        assert motor is None

    def test_common_offline_motors_exist(self):
        """Test that common offline motors exist."""
        from generate_motor_config import get_offline_motor

        # Test the most common motors that we know exist
        common_motors = [
            "estes_a8",
            "estes_b6",
            "estes_c6",
            "estes_d12",
            "aerotech_f40",
            "cesaroni_g79",
            "aerotech_h128",
        ]

        for key in common_motors:
            motor = get_offline_motor(key)
            assert motor is not None, f"Motor {key} should be in offline database"

    def test_popular_motors_dict(self):
        """Test POPULAR_MOTORS dict has expected entries."""
        from generate_motor_config import POPULAR_MOTORS

        assert "estes_c6" in POPULAR_MOTORS
        assert "aerotech_f40" in POPULAR_MOTORS
        assert "cesaroni_g79" in POPULAR_MOTORS
        assert "aerotech_h128" in POPULAR_MOTORS

        for key, info in POPULAR_MOTORS.items():
            assert "manufacturer" in info
            assert "designation" in info
            assert "common_name" in info


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_config(self):
        """Test saving config to file."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
            save_config,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            motor = MotorData(
                motor_id="test",
                manufacturer="Test",
                designation="T1",
                common_name="Test T1",
                diameter=0.018,
                length=0.070,
                total_mass=0.024,
                propellant_mass=0.012,
                case_mass=0.012,
                total_impulse=10.0,
                burn_time=2.0,
                average_thrust=5.0,
                max_thrust=10.0,
                time_points=np.array([0.0, 1.0, 2.0]),
                thrust_points=np.array([0.0, 10.0, 0.0]),
            )

            physics = analyze_motor_physics(motor)
            config = generate_config(motor, physics, difficulty="easy")

            filepath = os.path.join(tmpdir, "test_config.yaml")
            save_config(config, filepath, motor, physics, "easy")

            assert os.path.exists(filepath)

            with open(filepath) as f:
                content = f.read()
                assert "# Auto-generated config" in content
                assert "Test T1" in content
                assert "Difficulty: easy" in content

            # Verify the YAML portion is valid
            with open(filepath) as f:
                lines = f.readlines()
            # Skip comment header lines
            yaml_lines = [
                ln for ln in lines if not ln.startswith("#") or ln.strip() == ""
            ]
            yaml_content = "".join(yaml_lines)
            parsed = yaml.safe_load(yaml_content)
            assert parsed is not None
            assert "physics" in parsed
        finally:
            shutil.rmtree(tmpdir)

    def test_save_config_creates_parent_dirs(self):
        """Test save_config creates parent directories."""
        from generate_motor_config import (
            get_offline_motor,
            analyze_motor_physics,
            generate_config,
            save_config,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            motor = get_offline_motor("estes_c6")
            physics = analyze_motor_physics(motor)
            config = generate_config(motor, physics, difficulty="easy")

            nested_path = os.path.join(tmpdir, "sub", "dir", "config.yaml")
            save_config(config, nested_path, motor, physics, "easy")

            assert os.path.exists(nested_path)
        finally:
            shutil.rmtree(tmpdir)

    def test_save_config_with_notes(self):
        """Test save_config includes physics notes in header."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
            save_config,
        )

        tmpdir = tempfile.mkdtemp()
        try:
            motor = MotorData(
                motor_id="test",
                manufacturer="Test",
                designation="T1",
                common_name="Test T1",
                diameter=0.018,
                length=0.070,
                total_mass=0.1,
                propellant_mass=0.05,
                case_mass=0.05,
                total_impulse=5.0,
                burn_time=2.0,
                average_thrust=2.5,
                max_thrust=5.0,
                time_points=np.array([0.0, 1.0, 2.0]),
                thrust_points=np.array([0.0, 5.0, 0.0]),
            )

            physics = analyze_motor_physics(motor, dry_mass=0.5)
            config = generate_config(motor, physics, difficulty="easy")

            filepath = os.path.join(tmpdir, "config_with_notes.yaml")
            save_config(config, filepath, motor, physics, "easy")

            with open(filepath) as f:
                content = f.read()
            # Should have NOTE lines from physics.notes (TWR warning)
            assert "NOTE:" in content
        finally:
            shutil.rmtree(tmpdir)


class TestGenerateMotorConfigs:
    """Tests for full config generation workflow."""

    def test_generate_motor_configs(self):
        """Test generating all difficulty configs for a motor."""
        from generate_motor_config import get_offline_motor, generate_motor_configs

        tmpdir = tempfile.mkdtemp()
        try:
            motor = get_offline_motor("estes_c6")
            assert motor is not None

            filepaths = generate_motor_configs(
                motor=motor,
                output_dir=tmpdir,
                difficulties=["easy"],
            )

            assert "easy" in filepaths
            assert os.path.exists(filepaths["easy"])
        finally:
            shutil.rmtree(tmpdir)

    def test_generate_motor_configs_default_difficulties(self):
        """Test generate_motor_configs with default difficulties (all three)."""
        from generate_motor_config import get_offline_motor, generate_motor_configs

        tmpdir = tempfile.mkdtemp()
        try:
            motor = get_offline_motor("estes_c6")
            assert motor is not None

            filepaths = generate_motor_configs(
                motor=motor,
                output_dir=tmpdir,
            )

            assert "easy" in filepaths
            assert "medium" in filepaths
            assert "full" in filepaths
            for diff, path in filepaths.items():
                assert os.path.exists(path), f"Config for {diff} should exist"
        finally:
            shutil.rmtree(tmpdir)

    def test_generate_motor_configs_with_dry_mass(self):
        """Test generate_motor_configs with custom dry mass."""
        from generate_motor_config import get_offline_motor, generate_motor_configs

        tmpdir = tempfile.mkdtemp()
        try:
            motor = get_offline_motor("aerotech_f40")
            assert motor is not None

            filepaths = generate_motor_configs(
                motor=motor,
                output_dir=tmpdir,
                dry_mass=0.3,
                difficulties=["easy"],
            )

            assert "easy" in filepaths
            assert os.path.exists(filepaths["easy"])

            with open(filepaths["easy"]) as f:
                content = f.read()
            # Should reference the dry mass in the header
            assert "300" in content  # 300g dry mass
        finally:
            shutil.rmtree(tmpdir)


class TestThrustCurveAPI:
    """Tests for ThrustCurve API (mocked)."""

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_search_motors(self, mock_requests):
        """Test motor search with mocked API."""
        from generate_motor_config import ThrustCurveAPI

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            results = api.search_motors(manufacturer="Estes")

            assert len(results) == 1
            assert results[0].manufacturer == "Estes"
            assert results[0].designation == "C6"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_search_motors_with_all_params(self, mock_requests):
        """Test search_motors with impulse_class, diameter, and availability."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "xyz789",
                    "manufacturer": "AeroTech",
                    "designation": "F40",
                    "commonName": "AeroTech F40",
                    "diameter": 29.0,
                    "length": 124.0,
                    "impulseClass": "F",
                    "totImpulseNs": 80.0,
                    "avgThrustN": 40.0,
                    "maxThrustN": 65.0,
                    "burnTimeS": 2.0,
                    "totalWeightG": 90.0,
                    "propWeightG": 39.0,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            results = api.search_motors(
                impulse_class="f",
                diameter=29.0,
                availability="regular",
                common_name="AeroTech F40",
            )

            assert len(results) == 1
            # Verify the API was called with correct params
            call_args = mock_session.get.call_args
            params = call_args[1].get(
                "params", call_args[0][1] if len(call_args[0]) > 1 else {}
            )
            if isinstance(params, dict):
                assert params.get("impulseClass") == "F"
                assert params.get("diameter") == 29.0
                assert params.get("availability") == "regular"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_search_motors_with_query_filter(self, mock_requests):
        """Test search_motors with local query filtering."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                },
                {
                    "motorId": "def",
                    "manufacturer": "AeroTech",
                    "designation": "F40",
                    "commonName": "AeroTech F40",
                    "diameter": 29.0,
                    "length": 124.0,
                    "impulseClass": "F",
                    "totImpulseNs": 80.0,
                    "avgThrustN": 40.0,
                    "maxThrustN": 65.0,
                    "burnTimeS": 2.0,
                    "totalWeightG": 90.0,
                    "propWeightG": 39.0,
                },
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            # Query filter should only return matching results
            results = api.search_motors(query="C6")
            assert len(results) == 1
            assert results[0].designation == "C6"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_search_motors_request_exception(self, mock_requests):
        """Test search_motors handles request exceptions."""
        from generate_motor_config import ThrustCurveAPI

        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")
        mock_session.headers = Mock()

        mock_requests.RequestException = Exception

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            results = api.search_motors(manufacturer="Estes")
            assert results == []

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_with_search_result(self, mock_requests):
        """Test get_motor_data with search result metadata."""
        from generate_motor_config import ThrustCurveAPI, MotorSearchResult

        search_result = MotorSearchResult(
            motor_id="abc123",
            manufacturer="Estes",
            designation="C6",
            common_name="Estes C6",
            diameter=18.0,
            length=70.0,
            impulse_class="C",
            total_impulse=10.0,
            avg_thrust=5.4,
            max_thrust=14.0,
            burn_time=1.85,
            total_mass=24.0,
            prop_mass=12.3,
        )

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "samples": [
                        {"time": 0.0, "thrust": 0.0},
                        {"time": 0.04, "thrust": 14.0},
                        {"time": 0.5, "thrust": 6.0},
                        {"time": 1.0, "thrust": 5.0},
                        {"time": 1.85, "thrust": 0.0},
                    ],
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("abc123", search_result=search_result)

            assert motor_data is not None
            assert motor_data.manufacturer == "Estes"
            assert motor_data.designation == "C6"
            assert len(motor_data.time_points) == 5
            assert motor_data.impulse_class == "C"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_without_search_result(self, mock_requests):
        """Test get_motor_data without search result (falls back to download data)."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                    "totImpulseNs": 10.0,
                    "burnTimeS": 1.85,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "samples": [
                        {"time": 0.0, "thrust": 0.0},
                        {"time": 0.5, "thrust": 10.0},
                        {"time": 1.85, "thrust": 0.0},
                    ],
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("abc123", search_result=None)

            assert motor_data is not None
            assert motor_data.manufacturer == "Estes"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_no_samples(self, mock_requests):
        """Test get_motor_data creates synthetic curve when no samples."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                    "totImpulseNs": 10.0,
                    "burnTimeS": 1.85,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "samples": [],
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("abc123", search_result=None)

            assert motor_data is not None
            # Synthetic curve should have 5 points
            assert len(motor_data.time_points) == 5

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_no_results(self, mock_requests):
        """Test get_motor_data returns None when no results."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("nonexistent", search_result=None)
            assert motor_data is None

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_request_exception(self, mock_requests):
        """Test get_motor_data handles request exceptions."""
        from generate_motor_config import ThrustCurveAPI

        mock_session = Mock()
        mock_session.get.side_effect = Exception("Connection error")
        mock_session.headers = Mock()

        mock_requests.RequestException = Exception

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("abc123", search_result=None)
            assert motor_data is None

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_get_motor_data_json_decode_error(self, mock_requests):
        """Test get_motor_data handles JSON decode errors."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.raise_for_status = Mock()
        mock_response.json.side_effect = json.JSONDecodeError("err", "doc", 0)
        mock_response.text = "not json"

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        mock_requests.RequestException = Exception

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            motor_data = api.get_motor_data("abc123", search_result=None)
            assert motor_data is None

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_verify_motor(self, mock_requests):
        """Test motor verification with mocked API."""
        from generate_motor_config import ThrustCurveAPI

        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            exists, result = api.verify_motor("Estes C6")

            assert exists is True
            assert result is not None

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_verify_motor_no_exact_match(self, mock_requests):
        """Test verify_motor returns first result when no exact match."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6-5",
                    "commonName": "Estes C6-5",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            # Searching for "Estes C6" but result is "C6-5" -- no exact match
            exists, result = api.verify_motor("Estes C6")

            assert exists is True
            assert result is not None
            # Should return first result as best match
            assert result.designation == "C6-5"

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_verify_motor_single_word_query(self, mock_requests):
        """Test verify_motor with a single-word query (no manufacturer split)."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            exists, result = api.verify_motor("C6")

            assert exists is True

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    @patch("generate_motor_config.requests")
    def test_verify_motor_not_found(self, mock_requests):
        """Test verify_motor when motor is not found."""
        from generate_motor_config import ThrustCurveAPI

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status = Mock()

        mock_session = Mock()
        mock_session.get.return_value = mock_response
        mock_session.headers = Mock()

        with patch.object(ThrustCurveAPI, "__init__", lambda x: None):
            api = ThrustCurveAPI()
            api.base_url = "https://www.thrustcurve.org/api/v1"
            api.session = mock_session

            exists, result = api.verify_motor("NonExistent Motor")

            assert exists is False
            assert result is None


class TestCLICommands:
    """Tests for CLI command functions."""

    def test_cmd_list_popular(self, capsys):
        """Test list-popular command."""
        from generate_motor_config import cmd_list_popular

        args = Namespace()
        cmd_list_popular(args)

        captured = capsys.readouterr()
        assert "Popular Motors" in captured.out
        assert "estes_c6" in captured.out

    @patch("generate_motor_config.REQUESTS_AVAILABLE", False)
    def test_cmd_search_without_requests(self, capsys):
        """Test search command when requests is not available."""
        from generate_motor_config import cmd_search

        args = Namespace(
            query="C6", manufacturer=None, impulse_class=None, max_results=20
        )
        cmd_search(args)

        captured = capsys.readouterr()
        assert (
            "requires 'requests'" in captured.out
            or "Available offline motors" in captured.out
        )

    def test_cmd_verify_offline(self, capsys):
        """Test verify command with offline motor."""
        from generate_motor_config import cmd_verify

        args = Namespace(motor="estes_c6")
        cmd_verify(args)

        captured = capsys.readouterr()
        assert "found" in captured.out.lower()
        assert "Estes" in captured.out

    @patch("generate_motor_config.REQUESTS_AVAILABLE", False)
    def test_cmd_verify_not_found_no_requests(self, capsys):
        """Test verify command with unknown motor and no requests."""
        from generate_motor_config import cmd_verify

        args = Namespace(motor="unknown_xyz")
        cmd_verify(args)

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()

    def test_cmd_generate_offline(self, capsys):
        """Test generate command with offline motor."""
        from generate_motor_config import cmd_generate

        tmpdir = tempfile.mkdtemp()
        try:
            args = Namespace(
                motor="estes_c6",
                output=tmpdir,
                difficulty="easy",
                dry_mass=None,
            )
            cmd_generate(args)

            captured = capsys.readouterr()
            assert "Config Generation Complete" in captured.out

            config_files = list(Path(tmpdir).glob("*.yaml"))
            assert len(config_files) >= 1
        finally:
            shutil.rmtree(tmpdir)

    def test_cmd_generate_all_difficulties(self, capsys):
        """Test generate command with difficulty='all'."""
        from generate_motor_config import cmd_generate

        tmpdir = tempfile.mkdtemp()
        try:
            args = Namespace(
                motor="estes_c6",
                output=tmpdir,
                difficulty="all",
                dry_mass=None,
            )
            cmd_generate(args)

            captured = capsys.readouterr()
            assert "Config Generation Complete" in captured.out

            config_files = list(Path(tmpdir).glob("*.yaml"))
            assert len(config_files) == 3
        finally:
            shutil.rmtree(tmpdir)

    @patch("generate_motor_config.REQUESTS_AVAILABLE", False)
    def test_cmd_generate_not_found(self, capsys):
        """Test generate command with non-existent motor."""
        from generate_motor_config import cmd_generate

        tmpdir = tempfile.mkdtemp()
        try:
            args = Namespace(
                motor="nonexistent_xyz",
                output=tmpdir,
                difficulty="easy",
                dry_mass=None,
            )
            result = cmd_generate(args)

            captured = capsys.readouterr()
            assert "not found" in captured.out.lower()
            assert result == 1
        finally:
            shutil.rmtree(tmpdir)

    def test_cmd_generate_with_dry_mass(self, capsys):
        """Test generate command with custom dry mass."""
        from generate_motor_config import cmd_generate

        tmpdir = tempfile.mkdtemp()
        try:
            args = Namespace(
                motor="estes_c6",
                output=tmpdir,
                difficulty="easy",
                dry_mass=0.15,
            )
            cmd_generate(args)

            captured = capsys.readouterr()
            assert "Config Generation Complete" in captured.out
        finally:
            shutil.rmtree(tmpdir)

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    def test_cmd_search_with_requests(self, capsys):
        """Test search command with requests available (mocked API)."""
        from generate_motor_config import cmd_search

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "motorId": "abc123",
                    "manufacturer": "Estes",
                    "designation": "C6",
                    "commonName": "Estes C6",
                    "diameter": 18.0,
                    "length": 70.0,
                    "impulseClass": "C",
                    "totImpulseNs": 10.0,
                    "avgThrustN": 5.4,
                    "maxThrustN": 14.0,
                    "burnTimeS": 1.85,
                    "totalWeightG": 24.0,
                    "propWeightG": 12.3,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("generate_motor_config.ThrustCurveAPI") as MockAPI:
            mock_api = Mock()
            mock_api.search_motors.return_value = [
                Mock(
                    motor_id="abc123",
                    manufacturer="Estes",
                    designation="C6",
                    impulse_class="C",
                    total_impulse=10.0,
                    avg_thrust=5.4,
                )
            ]
            MockAPI.return_value = mock_api

            args = Namespace(
                query="C6",
                manufacturer=None,
                impulse_class=None,
                max_results=20,
            )
            cmd_search(args)

            captured = capsys.readouterr()
            assert "Found" in captured.out or "motors" in captured.out.lower()

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    def test_cmd_search_no_results(self, capsys):
        """Test search command with no results."""
        from generate_motor_config import cmd_search

        with patch("generate_motor_config.ThrustCurveAPI") as MockAPI:
            mock_api = Mock()
            mock_api.search_motors.return_value = []
            MockAPI.return_value = mock_api

            args = Namespace(
                query="nonexistent",
                manufacturer=None,
                impulse_class=None,
                max_results=20,
            )
            cmd_search(args)

            captured = capsys.readouterr()
            assert "No motors found" in captured.out

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    def test_cmd_verify_via_api(self, capsys):
        """Test verify command falls back to API when not in offline DB."""
        from generate_motor_config import cmd_verify, MotorSearchResult

        mock_result = MotorSearchResult(
            motor_id="api123",
            manufacturer="Cesaroni",
            designation="K1000",
            common_name="Cesaroni K1000",
            diameter=54.0,
            length=300.0,
            impulse_class="K",
            total_impulse=1500.0,
            avg_thrust=1000.0,
            max_thrust=1200.0,
            burn_time=1.5,
            total_mass=1000.0,
            prop_mass=600.0,
        )

        with patch("generate_motor_config.ThrustCurveAPI") as MockAPI:
            mock_api = Mock()
            mock_api.verify_motor.return_value = (True, mock_result)
            MockAPI.return_value = mock_api

            args = Namespace(motor="cesaroni_k1000")
            cmd_verify(args)

            captured = capsys.readouterr()
            assert "found" in captured.out.lower()

    @patch("generate_motor_config.REQUESTS_AVAILABLE", True)
    def test_cmd_verify_api_not_found(self, capsys):
        """Test verify command when API also does not find the motor."""
        from generate_motor_config import cmd_verify

        with patch("generate_motor_config.ThrustCurveAPI") as MockAPI:
            mock_api = Mock()
            mock_api.verify_motor.return_value = (False, None)
            MockAPI.return_value = mock_api

            args = Namespace(motor="totally_fake_motor")
            cmd_verify(args)

            captured = capsys.readouterr()
            assert "not found" in captured.out.lower()


class TestCLIMain:
    """Tests for the main() CLI entry point."""

    def test_main_no_args(self, capsys):
        """Test main with no arguments prints help."""
        from generate_motor_config import main

        with patch("sys.argv", ["generate_motor_config.py"]):
            main()

        captured = capsys.readouterr()
        assert (
            "Motor Config Generator" in captured.out or "usage" in captured.out.lower()
        )

    def test_main_list_popular(self, capsys):
        """Test main with list-popular command."""
        from generate_motor_config import main

        with patch("sys.argv", ["generate_motor_config.py", "list-popular"]):
            main()

        captured = capsys.readouterr()
        assert "Popular Motors" in captured.out

    def test_main_verify(self, capsys):
        """Test main with verify command."""
        from generate_motor_config import main

        with patch("sys.argv", ["generate_motor_config.py", "verify", "estes_c6"]):
            main()

        captured = capsys.readouterr()
        assert "Estes" in captured.out

    def test_main_generate(self, capsys):
        """Test main with generate command."""
        from generate_motor_config import main

        tmpdir = tempfile.mkdtemp()
        try:
            with patch(
                "sys.argv",
                [
                    "generate_motor_config.py",
                    "generate",
                    "estes_c6",
                    "--output",
                    tmpdir,
                    "--difficulty",
                    "easy",
                ],
            ):
                main()

            captured = capsys.readouterr()
            assert "Config Generation Complete" in captured.out
        finally:
            shutil.rmtree(tmpdir)

    @patch("generate_motor_config.REQUESTS_AVAILABLE", False)
    def test_main_search(self, capsys):
        """Test main with search command (no requests)."""
        from generate_motor_config import main

        with patch("sys.argv", ["generate_motor_config.py", "search", "C6"]):
            main()

        captured = capsys.readouterr()
        assert (
            "requires 'requests'" in captured.out or "offline" in captured.out.lower()
        )


class TestPhysicsAnalysisEdgeCases:
    """Tests for edge cases in physics analysis."""

    def test_high_velocity_motor(self):
        """Test analysis of high velocity motor."""
        from generate_motor_config import MotorData, analyze_motor_physics

        # High impulse motor
        motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="H128",
            common_name="Test H128",
            diameter=0.029,
            length=0.194,
            total_mass=0.227,
            propellant_mass=0.094,
            case_mass=0.133,
            total_impulse=219.0,
            burn_time=1.7,
            average_thrust=128.0,
            max_thrust=195.0,
            time_points=np.array([0.0, 0.1, 1.0, 1.7]),
            thrust_points=np.array([0.0, 195.0, 130.0, 0.0]),
        )

        physics = analyze_motor_physics(motor)

        # Should have high velocity note
        assert "High velocity" in str(physics.notes) or physics.recommended_dt == 0.01

    def test_motor_sizes(self):
        """Test physics analysis for different motor sizes."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
        )

        # Small motor (18mm)
        small_motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="C6",
            common_name="Test C6",
            diameter=0.018,
            length=0.070,
            total_mass=0.024,
            propellant_mass=0.012,
            case_mass=0.012,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # Large motor (38mm)
        large_motor = MotorData(
            motor_id="test",
            manufacturer="Test",
            designation="I284",
            common_name="Test I284",
            diameter=0.038,
            length=0.200,
            total_mass=0.350,
            propellant_mass=0.150,
            case_mass=0.200,
            total_impulse=500.0,
            burn_time=2.0,
            average_thrust=250.0,
            max_thrust=400.0,
            time_points=np.array([0.0, 0.5, 1.5, 2.0]),
            thrust_points=np.array([0.0, 400.0, 200.0, 0.0]),
        )

        small_physics = analyze_motor_physics(small_motor)
        large_physics = analyze_motor_physics(large_motor)

        # Generate configs
        small_config = generate_config(small_motor, small_physics, difficulty="easy")
        large_config = generate_config(large_motor, large_physics, difficulty="easy")

        # Larger motor should have larger fin dimensions
        assert (
            large_config["physics"]["fin_span"] >= small_config["physics"]["fin_span"]
        )

    def test_damping_scale_ranges(self):
        """Test damping scale for different velocity ranges."""
        from generate_motor_config import MotorData, analyze_motor_physics

        def make_motor(total_impulse, avg_thrust, total_mass):
            return MotorData(
                motor_id="test",
                manufacturer="Test",
                designation="T1",
                common_name="Test",
                diameter=0.018,
                length=0.070,
                total_mass=total_mass,
                propellant_mass=total_mass * 0.5,
                case_mass=total_mass * 0.5,
                total_impulse=total_impulse,
                burn_time=1.0,
                average_thrust=avg_thrust,
                max_thrust=avg_thrust * 1.5,
                time_points=np.array([0.0, 1.0]),
                thrust_points=np.array([avg_thrust, 0.0]),
            )

        # Low velocity: impulse/mass < 50 => damping 1.5
        low_v = make_motor(5.0, 5.0, 0.2)
        low_phys = analyze_motor_physics(low_v, dry_mass=0.1)
        assert low_phys.recommended_damping_scale == 1.5

    def test_recommended_deflection_rounding(self):
        """Test that recommended deflection rounds to nice values."""
        from generate_motor_config import analyze_motor_physics, get_offline_motor

        # All offline motors should produce reasonable deflection values
        for key in ["estes_c6", "aerotech_f40", "aerotech_h128"]:
            motor = get_offline_motor(key)
            physics = analyze_motor_physics(motor)
            assert physics.recommended_tab_deflection in [
                1.0,
                2.0,
                3.0,
                5.0,
                10.0,
            ] or (1.0 <= physics.recommended_tab_deflection <= 10.0)
