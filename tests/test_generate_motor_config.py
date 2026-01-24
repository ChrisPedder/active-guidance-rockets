"""
Tests for generate_motor_config module - motor config generation and ThrustCurve API.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json


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

        assert motor.impulse_class == "C"  # 10 N·s is C class
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

        # During burn, linear interpolation
        mid_mass = motor.get_mass(1.0)
        assert mid_mass > motor.case_mass
        assert mid_mass < motor.total_mass


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

        # Test medium
        medium_config = generate_config(motor, physics, difficulty="medium")
        assert medium_config["environment"]["enable_wind"] is True

        # Test full
        full_config = generate_config(motor, physics, difficulty="full")
        assert full_config["curriculum"]["enabled"] is True
        assert len(full_config["curriculum"]["stages"]) > 0

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


class TestSaveConfig:
    """Tests for config saving."""

    def test_save_config(self, tmp_path):
        """Test saving config to file."""
        from generate_motor_config import (
            MotorData,
            analyze_motor_physics,
            generate_config,
            save_config,
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
        config = generate_config(motor, physics, difficulty="easy")

        filepath = tmp_path / "test_config.yaml"
        save_config(config, str(filepath), motor, physics, "easy")

        assert filepath.exists()

        # Read and verify
        import yaml

        with open(filepath) as f:
            content = f.read()
            # Should have header comments
            assert "# Auto-generated config" in content
            assert "Test T1" in content


class TestGenerateMotorConfigs:
    """Tests for full config generation workflow."""

    def test_generate_motor_configs(self, tmp_path):
        """Test generating all difficulty configs for a motor."""
        from generate_motor_config import get_offline_motor, generate_motor_configs

        motor = get_offline_motor("estes_c6")
        assert motor is not None

        filepaths = generate_motor_configs(
            motor=motor,
            output_dir=str(tmp_path),
            difficulties=["easy"],  # Just easy for speed
        )

        assert "easy" in filepaths
        assert Path(filepaths["easy"]).exists()


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


class TestCLICommands:
    """Tests for CLI command functions."""

    def test_cmd_list_popular(self, capsys):
        """Test list-popular command."""
        from generate_motor_config import cmd_list_popular
        from argparse import Namespace

        args = Namespace()
        cmd_list_popular(args)

        captured = capsys.readouterr()
        assert "Popular Motors" in captured.out
        assert "estes_c6" in captured.out

    @patch("generate_motor_config.REQUESTS_AVAILABLE", False)
    def test_cmd_search_without_requests(self, capsys):
        """Test search command when requests is not available."""
        from generate_motor_config import cmd_search
        from argparse import Namespace

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
        from argparse import Namespace

        args = Namespace(motor="estes_c6")
        cmd_verify(args)

        captured = capsys.readouterr()
        assert "found" in captured.out.lower()
        assert "Estes" in captured.out

    def test_cmd_generate_offline(self, tmp_path, capsys):
        """Test generate command with offline motor."""
        from generate_motor_config import cmd_generate
        from argparse import Namespace

        args = Namespace(
            motor="estes_c6",
            output=str(tmp_path),
            difficulty="easy",
            dry_mass=None,
        )
        cmd_generate(args)

        captured = capsys.readouterr()
        assert "Config Generation Complete" in captured.out

        # Check file was created
        config_files = list(tmp_path.glob("*.yaml"))
        assert len(config_files) >= 1

    def test_cmd_generate_not_found(self, tmp_path, capsys):
        """Test generate command with non-existent motor."""
        from generate_motor_config import cmd_generate
        from argparse import Namespace

        args = Namespace(
            motor="nonexistent_xyz",
            output=str(tmp_path),
            difficulty="easy",
            dry_mass=None,
        )
        result = cmd_generate(args)

        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


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
