"""
Tests targeting remaining coverage gaps in:
- realistic_spin_rocket.py (63% -> 80%+)
- rocket_env/inference/controller.py (78% -> 80%+)
- visualizations/wind_field_visualization.py (72% -> 80%+)

These tests exercise uncovered branches and methods not reached by
existing test files.
"""

import os
import sys
import pickle
import tempfile

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers for mock objects that can be pickled
# ---------------------------------------------------------------------------


class _MockObsRms:
    """Picklable mock for VecNormalize.obs_rms."""

    def __init__(self, size=10):
        self.mean = np.zeros(size, dtype=np.float64)
        self.var = np.ones(size, dtype=np.float64)


class _MockVecNormalize:
    """Picklable mock for SB3 VecNormalize with obs_rms attribute."""

    def __init__(self, size=10):
        self.obs_rms = _MockObsRms(size)


class _MockVecNormalizeAlt:
    """Picklable mock for alternative VecNormalize format (running_mean/var)."""

    def __init__(self, size=10):
        self.running_mean = np.zeros(size, dtype=np.float64)
        self.running_var = np.ones(size, dtype=np.float64)


class _MockVecNormalizeEmpty:
    """Picklable mock that has neither obs_rms nor running_mean."""

    pass


# ===========================================================================
# realistic_spin_rocket.py — additional coverage
# ===========================================================================


class TestRealisticMotorRocketInitBranches:
    """Cover __init__ branches not exercised by existing tests."""

    @pytest.fixture
    def airframe(self):
        from airframe import RocketAirframe

        return RocketAirframe.estes_alpha()

    @pytest.fixture
    def motor_config(self):
        return {
            "name": "estes_c6",
            "manufacturer": "Estes",
            "designation": "C6",
            "total_impulse_Ns": 10.0,
            "avg_thrust_N": 5.4,
            "max_thrust_N": 14.0,
            "burn_time_s": 1.85,
            "propellant_mass_g": 12.3,
            "case_mass_g": 12.7,
            "thrust_curve": {
                "time_s": [0.0, 0.1, 0.5, 1.0, 1.5, 1.85],
                "thrust_N": [0.0, 14.0, 6.0, 5.0, 4.0, 0.0],
            },
        }

    def test_init_with_motor_config_default_rocketconfig(self, airframe, motor_config):
        """__init__ with motor_config and config=None (default RocketConfig)."""
        from realistic_spin_rocket import RealisticMotorRocket

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=None,
        )
        assert env is not None
        assert env.motor is not None
        env.close()

    def test_init_raises_when_airframe_none(self, motor_config):
        """__init__ raises ValueError when airframe is None."""
        from realistic_spin_rocket import RealisticMotorRocket

        with pytest.raises(ValueError, match="RocketAirframe is required"):
            RealisticMotorRocket(airframe=None, motor_config=motor_config)

    def test_init_raises_when_no_motor_data_or_config(self, airframe):
        """__init__ raises ValueError when neither motor_data nor motor_config."""
        from realistic_spin_rocket import RealisticMotorRocket

        with pytest.raises(ValueError, match="Must provide either"):
            RealisticMotorRocket(airframe=airframe, motor_data=None, motor_config=None)

    def test_init_with_motor_data_object(self, airframe):
        """__init__ with a MotorData object instead of motor_config dict."""
        from realistic_spin_rocket import RealisticMotorRocket, MotorData

        motor_data = MotorData(
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
            time_points=np.array([0.0, 0.5, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 5.0, 0.0]),
        )

        env = RealisticMotorRocket(
            airframe=airframe, motor_data=motor_data, config=None
        )
        assert env.motor is motor_data
        env.close()

    def test_low_twr_warning(self, airframe, capsys):
        """__init__ prints TWR < 1.0 warning for very low thrust motors."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        # Extremely low thrust -> TWR < 1.0
        motor_config = {
            "name": "weak",
            "manufacturer": "Test",
            "designation": "W1",
            "total_impulse_Ns": 0.1,
            "avg_thrust_N": 0.05,
            "max_thrust_N": 0.1,
            "burn_time_s": 2.0,
            "propellant_mass_g": 1.0,
            "case_mass_g": 1.0,
            "thrust_curve": {
                "time_s": [0.0, 2.0],
                "thrust_N": [0.1, 0.0],
            },
        }

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=RocketConfig(),
        )
        captured = capsys.readouterr()
        assert "cannot lift off" in captured.out.lower() or "TWR" in captured.out
        env.close()

    def test_marginal_twr_warning(self, airframe, capsys):
        """__init__ prints TWR < 2.0 warning for marginal thrust motors."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        # Estes alpha dry mass ~ 0.022 kg, propellant = 0.001 kg => ~0.023 kg
        # Need TWR = avg_thrust / (0.023 * 9.81) between 1.0 and 2.0
        # TWR = 1.5 => avg_thrust = 1.5 * 0.023 * 9.81 = 0.338 N
        motor_config = {
            "name": "marginal",
            "manufacturer": "Test",
            "designation": "M1",
            "total_impulse_Ns": 0.7,
            "avg_thrust_N": 0.34,
            "max_thrust_N": 0.5,
            "burn_time_s": 2.0,
            "propellant_mass_g": 1.0,
            "case_mass_g": 1.0,
            "thrust_curve": {
                "time_s": [0.0, 2.0],
                "thrust_N": [0.5, 0.2],
            },
        }

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=RocketConfig(),
        )
        captured = capsys.readouterr()
        assert "marginal" in captured.out.lower()
        env.close()


class TestUpdatePropulsionDetailed:
    """Detailed tests for _update_propulsion during and after burn."""

    @pytest.fixture
    def env(self):
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        motor_config = {
            "name": "test",
            "manufacturer": "Test",
            "designation": "T1",
            "total_impulse_Ns": 10.0,
            "avg_thrust_N": 5.0,
            "max_thrust_N": 10.0,
            "burn_time_s": 2.0,
            "propellant_mass_g": 12.0,
            "case_mass_g": 12.0,
            "thrust_curve": {
                "time_s": [0.0, 0.5, 1.5, 2.0],
                "thrust_N": [0.0, 10.0, 5.0, 0.0],
            },
        }
        config = RocketConfig(dt=0.01)
        env = RealisticMotorRocket(
            airframe=airframe, motor_config=motor_config, config=config
        )
        env.reset(seed=42)
        return env

    def test_during_burn_returns_positive_thrust_and_mass(self, env):
        """_update_propulsion during burn returns (thrust > 0, mass > case_mass)."""
        env.time = 0.5
        thrust, mass = env._update_propulsion()
        assert thrust > 0.0
        assert mass > env.airframe.dry_mass + env.motor_case_mass
        assert env.propellant_remaining > 0.0

    def test_after_burn_returns_zero_thrust(self, env):
        """_update_propulsion after burn returns (0.0, dry_mass + case_mass)."""
        env.time = 10.0
        thrust, mass = env._update_propulsion()
        assert thrust == 0.0
        expected_mass = env.airframe.dry_mass + env.motor_case_mass
        assert mass == pytest.approx(expected_mass, rel=1e-3)
        assert env.propellant_remaining == 0.0

    def test_at_burn_boundary(self, env):
        """_update_propulsion at exactly burn_time uses after-burn path."""
        env.time = env.motor.burn_time
        thrust, mass = env._update_propulsion()
        assert thrust == 0.0

    def test_propellant_decreases_over_time(self, env):
        """propellant_remaining decreases as time advances during burn."""
        env.time = 0.1
        env._update_propulsion()
        prop_early = env.propellant_remaining

        env.time = 1.5
        env._update_propulsion()
        prop_late = env.propellant_remaining

        assert prop_late < prop_early


class TestGetInfoMotorFields:
    """Test _get_info returns motor-specific fields."""

    @pytest.fixture
    def env(self):
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        motor_config = {
            "name": "test",
            "manufacturer": "TestMfg",
            "designation": "TX",
            "total_impulse_Ns": 10.0,
            "avg_thrust_N": 5.0,
            "max_thrust_N": 10.0,
            "burn_time_s": 2.0,
            "propellant_mass_g": 12.0,
            "case_mass_g": 12.0,
            "thrust_curve": {
                "time_s": [0.0, 1.0, 2.0],
                "thrust_N": [0.0, 10.0, 0.0],
            },
        }
        config = RocketConfig(dt=0.01)
        env = RealisticMotorRocket(
            airframe=airframe, motor_config=motor_config, config=config
        )
        env.reset(seed=42)
        return env

    def test_get_info_during_burn(self, env):
        """_get_info during burn includes motor name and positive thrust."""
        env.time = 0.5
        env._update_propulsion()
        info = env._get_info()
        assert "motor" in info
        assert "TestMfg" in info["motor"]
        assert "TX" in info["motor"]
        assert info["current_thrust_N"] > 0
        assert info["propellant_remaining_g"] > 0

    def test_get_info_after_burn(self, env):
        """_get_info after burn shows zero thrust."""
        env.time = 10.0
        env._update_propulsion()
        info = env._get_info()
        assert info["current_thrust_N"] == 0.0
        assert info["propellant_remaining_g"] == pytest.approx(0.0, abs=0.01)


class TestCreateEnvironmentFromConfigAdditional:
    """Extra coverage for create_environment_from_config."""

    def test_missing_airframe_file_key(self, tmp_path):
        """Error when physics section lacks airframe_file."""
        import yaml
        from realistic_spin_rocket import create_environment_from_config

        config = {
            "physics": {"max_tab_deflection": 30.0},
            "motor": {
                "name": "t",
                "manufacturer": "T",
                "designation": "T1",
                "total_impulse_Ns": 10.0,
                "avg_thrust_N": 5.0,
                "max_thrust_N": 10.0,
                "burn_time_s": 2.0,
                "propellant_mass_g": 12.0,
                "case_mass_g": 12.0,
                "thrust_curve": {"time_s": [0.0, 2.0], "thrust_N": [10.0, 0.0]},
            },
        }
        config_file = tmp_path / "no_airframe.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="airframe_file"):
            create_environment_from_config(str(config_file))

    def test_create_with_rank_gt_zero(self, tmp_path):
        """create_environment_from_config with rank > 0 seeds env."""
        import yaml
        from realistic_spin_rocket import create_environment_from_config

        airframe_content = (
            "name: Test Airframe\n"
            "components:\n"
            "- type: NoseCone\n"
            "  name: Nose\n"
            "  position: 0.0\n"
            "  length: 0.07\n"
            "  base_diameter: 0.024\n"
            "  shape: ogive\n"
            "  thickness: 0.002\n"
            "  material: ABS Plastic\n"
            "- type: BodyTube\n"
            "  name: Body\n"
            "  position: 0.07\n"
            "  length: 0.24\n"
            "  outer_diameter: 0.024\n"
            "  inner_diameter: 0.022\n"
            "  material: Cardboard\n"
            "- type: TrapezoidFinSet\n"
            "  name: Fins\n"
            "  position: 0.24\n"
            "  num_fins: 4\n"
            "  root_chord: 0.05\n"
            "  tip_chord: 0.025\n"
            "  span: 0.04\n"
            "  sweep_length: 0.0\n"
            "  thickness: 0.002\n"
            "  material: Balsa\n"
        )
        airframe_file = tmp_path / "airframe.yaml"
        airframe_file.write_text(airframe_content)

        config = {
            "physics": {
                "airframe_file": str(airframe_file),
                "max_tab_deflection": 30.0,
                "disturbance_scale": 0.0001,
            },
            "motor": {
                "name": "test",
                "manufacturer": "T",
                "designation": "T1",
                "total_impulse_Ns": 10.0,
                "avg_thrust_N": 5.0,
                "max_thrust_N": 10.0,
                "burn_time_s": 2.0,
                "propellant_mass_g": 12.0,
                "case_mass_g": 12.0,
                "thrust_curve": {
                    "time_s": [0.0, 1.0, 2.0],
                    "thrust_N": [0.0, 10.0, 0.0],
                },
            },
            "environment": {"dt": 0.01},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        env = create_environment_from_config(str(config_file), rank=3)
        assert env is not None
        # If rank > 0, env.reset(seed=rank) was called inside the function
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_create_with_relative_airframe_path(self, tmp_path):
        """create_environment_from_config resolves relative airframe_file paths."""
        import yaml
        from realistic_spin_rocket import create_environment_from_config

        # Create airframe in a subdirectory of tmp_path
        airframes_dir = tmp_path / "airframes"
        airframes_dir.mkdir()
        airframe_content = (
            "name: Test Airframe\n"
            "components:\n"
            "- type: NoseCone\n"
            "  name: Nose\n"
            "  position: 0.0\n"
            "  length: 0.07\n"
            "  base_diameter: 0.024\n"
            "  shape: ogive\n"
            "  thickness: 0.002\n"
            "  material: ABS Plastic\n"
            "- type: BodyTube\n"
            "  name: Body\n"
            "  position: 0.07\n"
            "  length: 0.24\n"
            "  outer_diameter: 0.024\n"
            "  inner_diameter: 0.022\n"
            "  material: Cardboard\n"
            "- type: TrapezoidFinSet\n"
            "  name: Fins\n"
            "  position: 0.24\n"
            "  num_fins: 4\n"
            "  root_chord: 0.05\n"
            "  tip_chord: 0.025\n"
            "  span: 0.04\n"
            "  sweep_length: 0.0\n"
            "  thickness: 0.002\n"
            "  material: Balsa\n"
        )
        (airframes_dir / "airframe.yaml").write_text(airframe_content)

        # Use a RELATIVE path for airframe_file (relative to config file dir)
        config = {
            "physics": {
                "airframe_file": "airframes/airframe.yaml",
                "max_tab_deflection": 30.0,
                "disturbance_scale": 0.0001,
            },
            "motor": {
                "name": "test",
                "manufacturer": "T",
                "designation": "T1",
                "total_impulse_Ns": 10.0,
                "avg_thrust_N": 5.0,
                "max_thrust_N": 10.0,
                "burn_time_s": 2.0,
                "propellant_mass_g": 12.0,
                "case_mass_g": 12.0,
                "thrust_curve": {
                    "time_s": [0.0, 1.0, 2.0],
                    "thrust_N": [0.0, 10.0, 0.0],
                },
            },
            "environment": {"dt": 0.01},
        }
        config_file = tmp_path / "config_rel.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        env = create_environment_from_config(str(config_file))
        assert env is not None
        obs, info = env.reset()
        assert obs is not None
        env.close()


# ===========================================================================
# rocket_env/inference/controller.py — additional coverage
# ===========================================================================


class TestRocketControllerFullInit:
    """Test RocketController __init__ via mocked ONNXRunner."""

    def test_init_without_normalize(self):
        """__init__ without normalize_path leaves obs_mean/obs_var as None."""
        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.predict.return_value = np.array([0.5])
            mock_runner.get_info.return_value = {"model": "test"}
            MockRunner.return_value = mock_runner

            from rocket_env.inference.controller import RocketController

            ctrl = RocketController(model_path="fake_model.onnx")

            assert ctrl.obs_mean is None
            assert ctrl.obs_var is None
            assert ctrl.runner is mock_runner
            assert ctrl.action_low == -1.0
            assert ctrl.action_high == 1.0

    def test_init_with_normalize(self, tmp_path):
        """__init__ with normalize_path loads stats."""
        pkl_path = tmp_path / "vec_normalize.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(_MockVecNormalize(size=5), f)

        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            mock_runner = MagicMock()
            MockRunner.return_value = mock_runner

            from rocket_env.inference.controller import RocketController

            ctrl = RocketController(
                model_path="fake_model.onnx",
                normalize_path=str(pkl_path),
            )

            assert ctrl.obs_mean is not None
            assert ctrl.obs_var is not None
            assert len(ctrl.obs_mean) == 5

    def test_init_custom_action_range(self):
        """__init__ with custom action_low, action_high, clip_obs."""
        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            MockRunner.return_value = MagicMock()

            from rocket_env.inference.controller import RocketController

            ctrl = RocketController(
                model_path="fake.onnx",
                action_low=-0.5,
                action_high=0.5,
                clip_obs=5.0,
            )
            assert ctrl.action_low == -0.5
            assert ctrl.action_high == 0.5
            assert ctrl.clip_obs == 5.0


class TestRocketControllerNormalization:
    """Test normalize_observation with and without loaded stats."""

    def _make_controller(self, obs_mean=None, obs_var=None):
        """Create a controller with mocked runner."""
        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.predict.return_value = np.array([0.0])
            mock_runner.get_info.return_value = {}
            MockRunner.return_value = mock_runner

            from rocket_env.inference.controller import RocketController

            ctrl = RocketController(model_path="fake.onnx")
            ctrl.obs_mean = obs_mean
            ctrl.obs_var = obs_var
            return ctrl

    def test_normalize_without_stats_passes_through(self):
        """Without stats, normalize_observation returns original."""
        ctrl = self._make_controller(obs_mean=None, obs_var=None)
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ctrl.normalize_observation(obs)
        assert np.allclose(result, obs)

    def test_normalize_with_stats_applies_formula(self):
        """With stats, normalize_observation applies (obs - mean) / sqrt(var + eps)."""
        mean = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        var = np.array([4.0, 4.0, 4.0], dtype=np.float32)
        ctrl = self._make_controller(obs_mean=mean, obs_var=var)

        obs = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        result = ctrl.normalize_observation(obs)

        expected = (obs - mean) / np.sqrt(var + 1e-8)
        assert np.allclose(result, expected, atol=1e-5)

    def test_normalize_clips_extreme_values(self):
        """Normalization clips to [-clip_obs, clip_obs]."""
        mean = np.array([0.0], dtype=np.float32)
        var = np.array([0.001], dtype=np.float32)  # tiny variance
        ctrl = self._make_controller(obs_mean=mean, obs_var=var)
        ctrl.clip_obs = 5.0

        obs = np.array([1000.0], dtype=np.float32)
        result = ctrl.normalize_observation(obs)
        assert result[0] == pytest.approx(5.0)


class TestRocketControllerGetAction:
    """Test get_action and get_tab_deflection via mocked runner."""

    def _make_controller(self, predict_return=np.array([0.5])):
        """Create a controller with mocked runner that returns predict_return."""
        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            mock_runner = MagicMock()
            mock_runner.predict.return_value = predict_return
            mock_runner.get_info.return_value = {"model": "test"}
            MockRunner.return_value = mock_runner

            from rocket_env.inference.controller import RocketController

            ctrl = RocketController(model_path="fake.onnx")
            return ctrl

    def test_get_action_normalize_true_no_stats(self):
        """get_action with normalize=True but no stats still works."""
        ctrl = self._make_controller(np.array([0.3]))
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = ctrl.get_action(obs, normalize=True)
        assert action[0] == pytest.approx(0.3)

    def test_get_action_normalize_false(self):
        """get_action with normalize=False skips normalization."""
        ctrl = self._make_controller(np.array([0.7]))
        obs = np.array([1.0, 2.0], dtype=np.float32)
        action = ctrl.get_action(obs, normalize=False)
        assert action[0] == pytest.approx(0.7)

    def test_get_action_clips_high(self):
        """get_action clips action above action_high."""
        ctrl = self._make_controller(np.array([5.0]))
        obs = np.array([1.0], dtype=np.float32)
        action = ctrl.get_action(obs, normalize=False)
        assert action[0] == pytest.approx(1.0)

    def test_get_action_clips_low(self):
        """get_action clips action below action_low."""
        ctrl = self._make_controller(np.array([-5.0]))
        obs = np.array([1.0], dtype=np.float32)
        action = ctrl.get_action(obs, normalize=False)
        assert action[0] == pytest.approx(-1.0)

    def test_get_tab_deflection_returns_float(self):
        """get_tab_deflection returns a Python float."""
        ctrl = self._make_controller(np.array([0.5]))
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        deflection = ctrl.get_tab_deflection(obs, max_deflection_deg=20.0)
        assert isinstance(deflection, float)
        assert deflection == pytest.approx(10.0)

    def test_get_tab_deflection_default_max(self):
        """get_tab_deflection uses default 15 deg max."""
        ctrl = self._make_controller(np.array([1.0]))
        obs = np.array([0.0], dtype=np.float32)
        deflection = ctrl.get_tab_deflection(obs)
        assert deflection == pytest.approx(15.0)

    def test_get_info_returns_dict(self):
        """get_info returns dict with expected keys."""
        ctrl = self._make_controller()
        info = ctrl.get_info()
        assert isinstance(info, dict)
        assert "model_path" in info
        assert "has_normalization" in info
        assert "action_range" in info
        assert info["has_normalization"] is False
        assert info["action_range"] == (-1.0, 1.0)


class TestLoadNormalizeStatsFormats:
    """Test _load_normalize_stats with different pickle formats."""

    def _make_bare_controller(self):
        """Create a controller without calling __init__."""
        from rocket_env.inference.controller import RocketController

        ctrl = RocketController.__new__(RocketController)
        ctrl.obs_mean = None
        ctrl.obs_var = None
        return ctrl

    def test_obs_rms_format(self, tmp_path):
        """Load stats from pickle with obs_rms attribute."""
        pkl_path = tmp_path / "stats.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(_MockVecNormalize(size=8), f)

        ctrl = self._make_bare_controller()
        ctrl._load_normalize_stats(str(pkl_path))

        assert ctrl.obs_mean is not None
        assert ctrl.obs_var is not None
        assert len(ctrl.obs_mean) == 8
        assert ctrl.obs_mean.dtype == np.float32

    def test_running_mean_format(self, tmp_path):
        """Load stats from pickle with running_mean/running_var attributes."""
        pkl_path = tmp_path / "stats_alt.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(_MockVecNormalizeAlt(size=6), f)

        ctrl = self._make_bare_controller()
        ctrl._load_normalize_stats(str(pkl_path))

        assert ctrl.obs_mean is not None
        assert ctrl.obs_var is not None
        assert len(ctrl.obs_mean) == 6

    def test_unknown_format_warns(self, tmp_path, capsys):
        """Load stats from pickle with unknown format prints warning."""
        pkl_path = tmp_path / "stats_unk.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(_MockVecNormalizeEmpty(), f)

        ctrl = self._make_bare_controller()
        ctrl._load_normalize_stats(str(pkl_path))

        assert ctrl.obs_mean is None
        assert ctrl.obs_var is None
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Could not extract" in captured.out

    def test_file_not_found_raises(self, tmp_path):
        """_load_normalize_stats raises FileNotFoundError for missing file."""
        ctrl = self._make_bare_controller()
        with pytest.raises(FileNotFoundError):
            ctrl._load_normalize_stats(str(tmp_path / "nonexistent.pkl"))


class TestLoadControllerFromTraining:
    """Test the load_controller_from_training convenience function."""

    def test_raises_file_not_found(self, tmp_path):
        """Raises FileNotFoundError when model file is missing."""
        from rocket_env.inference.controller import load_controller_from_training

        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_controller_from_training(str(tmp_path))

    def test_loads_without_normalize_file(self, tmp_path):
        """Loads successfully when normalize file is missing."""
        model_path = tmp_path / "best_model.onnx"
        model_path.touch()

        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            MockRunner.return_value = MagicMock()

            from rocket_env.inference.controller import load_controller_from_training

            ctrl = load_controller_from_training(str(tmp_path))
            assert ctrl is not None
            assert ctrl.obs_mean is None

    def test_loads_with_normalize_file(self, tmp_path):
        """Loads successfully when both model and normalize file exist."""
        model_path = tmp_path / "best_model.onnx"
        model_path.touch()

        pkl_path = tmp_path / "vec_normalize.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(_MockVecNormalize(size=10), f)

        with patch("rocket_env.inference.controller.ONNXRunner") as MockRunner:
            MockRunner.return_value = MagicMock()

            from rocket_env.inference.controller import load_controller_from_training

            ctrl = load_controller_from_training(str(tmp_path))
            assert ctrl is not None
            assert ctrl.obs_mean is not None
            assert len(ctrl.obs_mean) == 10


# ===========================================================================
# visualizations/wind_field_visualization.py — additional coverage
# ===========================================================================


class TestGetFlightParams:
    """Test get_flight_params for different rocket types."""

    def test_estes_alpha_params(self):
        from visualizations.wind_field_visualization import get_flight_params

        max_alt, flight_dur, vel = get_flight_params("estes_alpha")
        assert max_alt == pytest.approx(200.0)
        assert flight_dur == pytest.approx(8.0)
        assert vel == pytest.approx(30.0)

    def test_j800_params(self):
        from visualizations.wind_field_visualization import get_flight_params

        max_alt, flight_dur, vel = get_flight_params("j800")
        assert max_alt == pytest.approx(2000.0)
        assert flight_dur == pytest.approx(25.0)
        assert vel == pytest.approx(200.0)

    def test_unknown_rocket_defaults_to_estes(self):
        """Unknown rocket name falls through to else branch (Estes defaults)."""
        from visualizations.wind_field_visualization import get_flight_params

        max_alt, flight_dur, vel = get_flight_params("some_unknown")
        assert max_alt == pytest.approx(200.0)
        assert flight_dur == pytest.approx(8.0)
        assert vel == pytest.approx(30.0)


class TestSampleWindField:
    """Test sample_wind_field returns correct shapes and values."""

    @pytest.fixture
    def wind_model(self):
        from wind_model import WindModel, WindConfig

        config = WindConfig(
            enable=True,
            base_speed=2.0,
            max_gust_speed=1.0,
            variability=0.3,
        )
        model = WindModel(config)
        model.reset(seed=42)
        return model

    def test_output_shapes(self, wind_model):
        from visualizations.wind_field_visualization import sample_wind_field

        altitudes = np.linspace(1.0, 200.0, 20)
        times = np.linspace(0.0, 5.0, 10)

        speeds, directions = sample_wind_field(
            wind_model, altitudes, times, rocket_velocity=30.0
        )

        assert speeds.shape == (10, 20)
        assert directions.shape == (10, 20)

    def test_speeds_nonnegative(self, wind_model):
        from visualizations.wind_field_visualization import sample_wind_field

        altitudes = np.linspace(1.0, 100.0, 10)
        times = np.linspace(0.0, 3.0, 5)

        speeds, directions = sample_wind_field(
            wind_model, altitudes, times, rocket_velocity=30.0
        )

        assert np.all(speeds >= 0)

    def test_directions_in_range(self, wind_model):
        from visualizations.wind_field_visualization import sample_wind_field

        altitudes = np.linspace(1.0, 100.0, 10)
        times = np.linspace(0.0, 3.0, 5)

        speeds, directions = sample_wind_field(
            wind_model, altitudes, times, rocket_velocity=30.0
        )

        assert np.all(directions >= 0)
        assert np.all(directions < 360)

    def test_single_point(self, wind_model):
        from visualizations.wind_field_visualization import sample_wind_field

        altitudes = np.array([50.0])
        times = np.array([1.0])

        speeds, directions = sample_wind_field(
            wind_model, altitudes, times, rocket_velocity=30.0
        )

        assert speeds.shape == (1, 1)
        assert directions.shape == (1, 1)


class TestSampleTimeseries:
    """Test sample_timeseries returns correct shapes and values."""

    @pytest.fixture
    def wind_model(self):
        from wind_model import WindModel, WindConfig

        config = WindConfig(
            enable=True,
            base_speed=3.0,
            max_gust_speed=1.5,
            variability=0.3,
        )
        model = WindModel(config)
        model.reset(seed=123)
        return model

    def test_output_shapes(self, wind_model):
        from visualizations.wind_field_visualization import sample_timeseries

        times = np.linspace(0.0, 8.0, 100)

        ts_speeds, ts_dirs = sample_timeseries(
            wind_model, fixed_altitude=50.0, times=times, rocket_velocity=30.0
        )

        assert ts_speeds.shape == (100,)
        assert ts_dirs.shape == (100,)

    def test_speeds_nonnegative(self, wind_model):
        from visualizations.wind_field_visualization import sample_timeseries

        times = np.linspace(0.0, 5.0, 50)

        ts_speeds, ts_dirs = sample_timeseries(
            wind_model, fixed_altitude=100.0, times=times, rocket_velocity=30.0
        )

        assert np.all(ts_speeds >= 0)

    def test_directions_in_range(self, wind_model):
        from visualizations.wind_field_visualization import sample_timeseries

        times = np.linspace(0.0, 5.0, 50)

        ts_speeds, ts_dirs = sample_timeseries(
            wind_model, fixed_altitude=100.0, times=times, rocket_velocity=30.0
        )

        assert np.all(ts_dirs >= 0)
        assert np.all(ts_dirs < 360)

    def test_single_time_point(self, wind_model):
        from visualizations.wind_field_visualization import sample_timeseries

        times = np.array([2.0])

        ts_speeds, ts_dirs = sample_timeseries(
            wind_model, fixed_altitude=50.0, times=times, rocket_velocity=30.0
        )

        assert ts_speeds.shape == (1,)
        assert ts_dirs.shape == (1,)

    def test_different_altitudes_give_different_results(self):
        """Wind at different altitudes should differ when altitude gradient is set."""
        from wind_model import WindModel, WindConfig
        from visualizations.wind_field_visualization import sample_timeseries

        config = WindConfig(
            enable=True,
            base_speed=3.0,
            max_gust_speed=1.5,
            variability=0.3,
            altitude_gradient=0.5,  # speed increases with altitude
        )
        model = WindModel(config)
        model.reset(seed=123)

        times = np.linspace(0.0, 5.0, 50)

        speeds_low, _ = sample_timeseries(
            model, fixed_altitude=10.0, times=times, rocket_velocity=30.0
        )
        model.reset(seed=123)
        speeds_high, _ = sample_timeseries(
            model, fixed_altitude=500.0, times=times, rocket_velocity=30.0
        )

        # With altitude_gradient, higher altitude should have higher speeds
        assert np.mean(speeds_high) >= np.mean(speeds_low)


class TestSampleWindFieldWithDryden:
    """Test sampling with Dryden turbulence model enabled."""

    def test_dryden_model_sampling(self):
        from wind_model import WindModel, WindConfig
        from visualizations.wind_field_visualization import (
            sample_wind_field,
            sample_timeseries,
        )

        config = WindConfig(
            enable=True,
            base_speed=3.0,
            use_dryden=True,
            turbulence_severity="moderate",
            altitude_profile_alpha=0.14,
            reference_altitude=10.0,
        )
        model = WindModel(config)
        model.reset(seed=99)

        altitudes = np.linspace(1.0, 200.0, 15)
        times = np.linspace(0.0, 4.0, 20)

        speeds, directions = sample_wind_field(
            model, altitudes, times, rocket_velocity=30.0
        )
        assert speeds.shape == (20, 15)
        assert np.all(speeds >= 0)

        model.reset(seed=99)
        ts_speeds, ts_dirs = sample_timeseries(
            model, fixed_altitude=50.0, times=times, rocket_velocity=30.0
        )
        assert ts_speeds.shape == (20,)
        assert np.all(ts_speeds >= 0)


class TestWindFieldMain:
    """Test main() function of wind_field_visualization with mocked args."""

    def test_main_default_args(self):
        """main() with default args (no save) calls create_animation."""
        with (
            patch(
                "sys.argv",
                ["wind_field_visualization.py"],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()
            call_args = mock_anim.call_args
            # save_path should be None (no --save flag)
            assert call_args[0][7] is None or call_args.kwargs.get("save_path") is None

    def test_main_with_save_flag(self, tmp_path):
        """main() with --save flag passes a save_path to create_animation."""
        with (
            patch(
                "sys.argv",
                ["wind_field_visualization.py", "--save", "--rocket", "estes_alpha"],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()
            call_args = mock_anim.call_args
            save_path = call_args[0][7]
            assert save_path is not None
            assert "sinusoidal" in save_path
            assert "estes_alpha" in save_path

    def test_main_j800_rocket(self):
        """main() with --rocket j800."""
        with (
            patch(
                "sys.argv",
                ["wind_field_visualization.py", "--rocket", "j800"],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()

    def test_main_with_fixed_altitude(self):
        """main() with --fixed-altitude flag uses provided altitude."""
        with (
            patch(
                "sys.argv",
                ["wind_field_visualization.py", "--fixed-altitude", "75.0"],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()
            call_args = mock_anim.call_args
            fixed_alt = call_args[0][6]
            assert fixed_alt == pytest.approx(75.0)

    def test_main_with_dryden_flag(self):
        """main() with --dryden flag uses Dryden turbulence model."""
        with (
            patch(
                "sys.argv",
                ["wind_field_visualization.py", "--dryden", "--severity", "severe"],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()

    def test_main_save_with_mp4_format(self):
        """main() with --save --format mp4."""
        with (
            patch(
                "sys.argv",
                [
                    "wind_field_visualization.py",
                    "--save",
                    "--format",
                    "mp4",
                    "--rocket",
                    "j800",
                ],
            ),
            patch(
                "visualizations.wind_field_visualization.create_animation"
            ) as mock_anim,
        ):
            from visualizations.wind_field_visualization import main

            main()

            mock_anim.assert_called_once()
            call_args = mock_anim.call_args
            save_path = call_args[0][7]
            assert save_path.endswith(".mp4")
            assert "j800" in save_path
            assert "dryden" not in save_path


# ---------------------------------------------------------------------------
# Roll Rate Monte Carlo — main() coverage
# ---------------------------------------------------------------------------


class TestRollRateMCMain:
    """Tests for roll_rate_montecarlo.py main() to boost coverage above 80%."""

    def test_main_default_pid(self):
        """Test main() with default PID controller."""
        with (
            patch("sys.argv", ["roll_rate_montecarlo.py"]),
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation") as mock_anim,
        ):
            mock_collect.return_value = {
                1: [(np.array([0, 1]), np.array([5, 3]))],
                2: [(np.array([0, 1]), np.array([8, 6]))],
                3: [(np.array([0, 1]), np.array([12, 10]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            mock_collect.assert_called_once()
            mock_anim.assert_called_once()
            # Default: no save_path
            call_args = mock_anim.call_args
            assert call_args[0][6] is None  # save_path

    def test_main_save_gif(self):
        """Test main() with --save flag."""
        with (
            patch("sys.argv", ["roll_rate_montecarlo.py", "--save"]),
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation") as mock_anim,
        ):
            mock_collect.return_value = {
                1: [(np.array([0, 1]), np.array([5, 3]))],
                2: [(np.array([0, 1]), np.array([8, 6]))],
                3: [(np.array([0, 1]), np.array([12, 10]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            call_args = mock_anim.call_args
            save_path = call_args[0][6]
            assert save_path is not None
            assert save_path.endswith(".gif")

    def test_main_save_mp4_j800(self):
        """Test main() with --save --format mp4 --rocket j800."""
        with (
            patch(
                "sys.argv",
                [
                    "roll_rate_montecarlo.py",
                    "--rocket",
                    "j800",
                    "--save",
                    "--format",
                    "mp4",
                ],
            ),
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation") as mock_anim,
        ):
            mock_collect.return_value = {
                1: [(np.array([0, 1]), np.array([5, 3]))],
                2: [(np.array([0, 1]), np.array([8, 6]))],
                3: [(np.array([0, 1]), np.array([12, 10]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            call_args = mock_anim.call_args
            save_path = call_args[0][6]
            assert save_path.endswith(".mp4")
            assert "j800" in save_path

    def test_main_custom_wind_levels(self):
        """Test main() with custom --wind-levels and --n-runs."""
        with (
            patch(
                "sys.argv",
                [
                    "roll_rate_montecarlo.py",
                    "--wind-levels",
                    "2",
                    "5",
                    "--n-runs",
                    "3",
                ],
            ),
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation") as mock_anim,
        ):
            mock_collect.return_value = {
                2: [(np.array([0, 1]), np.array([5, 3]))],
                5: [(np.array([0, 1]), np.array([8, 6]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            call_args = mock_collect.call_args
            assert call_args[0][1] == [2.0, 5.0]  # wind_levels
            assert call_args[0][2] == 3  # n_runs

    def test_main_sac_model(self):
        """Test main() with --sac flag."""
        with (
            patch(
                "sys.argv",
                [
                    "roll_rate_montecarlo.py",
                    "--sac",
                    "/fake/model.zip",
                ],
            ),
            patch("visualizations.roll_rate_montecarlo.load_rl_model") as mock_load,
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation"),
        ):
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, None)
            mock_collect.return_value = {
                1: [(np.array([0, 1]), np.array([5, 3]))],
                2: [(np.array([0, 1]), np.array([8, 6]))],
                3: [(np.array([0, 1]), np.array([12, 10]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            mock_load.assert_called_once()
            # controller_name should be "sac"
            call_args = mock_collect.call_args
            assert call_args[0][3] == "sac"

    def test_main_ppo_model(self):
        """Test main() with --ppo flag."""
        with (
            patch(
                "sys.argv",
                [
                    "roll_rate_montecarlo.py",
                    "--ppo",
                    "/fake/model.zip",
                ],
            ),
            patch("visualizations.roll_rate_montecarlo.load_rl_model") as mock_load,
            patch("visualizations.roll_rate_montecarlo.collect_data") as mock_collect,
            patch("visualizations.roll_rate_montecarlo.create_animation"),
        ):
            mock_model = MagicMock()
            mock_load.return_value = (mock_model, None)
            mock_collect.return_value = {
                1: [(np.array([0, 1]), np.array([5, 3]))],
                2: [(np.array([0, 1]), np.array([8, 6]))],
                3: [(np.array([0, 1]), np.array([12, 10]))],
            }
            from visualizations.roll_rate_montecarlo import main

            main()

            call_args = mock_collect.call_args
            assert call_args[0][3] == "ppo"

    def test_main_residual_sac_model(self):
        """Test main() with --residual-sac flag (no config.yaml in dir)."""
        with (tempfile.TemporaryDirectory() as tmpdir,):
            model_path = os.path.join(tmpdir, "best_model.zip")
            with open(model_path, "w") as f:
                f.write("fake")
            with (
                patch(
                    "sys.argv",
                    [
                        "roll_rate_montecarlo.py",
                        "--residual-sac",
                        model_path,
                    ],
                ),
                patch("visualizations.roll_rate_montecarlo.load_rl_model") as mock_load,
                patch(
                    "visualizations.roll_rate_montecarlo.collect_data"
                ) as mock_collect,
                patch("visualizations.roll_rate_montecarlo.create_animation"),
            ):
                mock_model = MagicMock()
                mock_load.return_value = (mock_model, None)
                mock_collect.return_value = {
                    1: [(np.array([0, 1]), np.array([5, 3]))],
                    2: [(np.array([0, 1]), np.array([8, 6]))],
                    3: [(np.array([0, 1]), np.array([12, 10]))],
                }
                from visualizations.roll_rate_montecarlo import main

                main()

                call_args = mock_collect.call_args
                assert call_args[0][3] == "residual-sac"


# ---------------------------------------------------------------------------
# Trajectory Monte Carlo — main() coverage
# ---------------------------------------------------------------------------


def _make_traj():
    """Create a minimal trajectory dict for mocking."""
    return {
        "time": np.array([0, 0.5, 1.0]),
        "altitude": np.array([0, 50, 100]),
        "x": np.array([0, 0.1, 0.3]),
        "y": np.array([0, 0.05, 0.15]),
        "velocity": np.array([0, 30, 60]),
    }


class TestTrajectoryMCMain:
    """Tests for trajectory_montecarlo.py main() to boost coverage above 80%."""

    def test_main_default_3d(self):
        """Test main() with default args (3D mode)."""
        with (
            patch("sys.argv", ["trajectory_montecarlo.py"]),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch(
                "visualizations.trajectory_montecarlo.create_3d_animation"
            ) as mock_3d,
            patch(
                "visualizations.trajectory_montecarlo.create_2d_animation"
            ) as mock_2d,
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            mock_3d.assert_called_once()
            mock_2d.assert_not_called()

    def test_main_2d_mode(self):
        """Test main() with --mode 2d."""
        with (
            patch("sys.argv", ["trajectory_montecarlo.py", "--mode", "2d"]),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch(
                "visualizations.trajectory_montecarlo.create_3d_animation"
            ) as mock_3d,
            patch(
                "visualizations.trajectory_montecarlo.create_2d_animation"
            ) as mock_2d,
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            mock_3d.assert_not_called()
            mock_2d.assert_called_once()

    def test_main_both_mode(self):
        """Test main() with --mode both."""
        with (
            patch("sys.argv", ["trajectory_montecarlo.py", "--mode", "both"]),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch(
                "visualizations.trajectory_montecarlo.create_3d_animation"
            ) as mock_3d,
            patch(
                "visualizations.trajectory_montecarlo.create_2d_animation"
            ) as mock_2d,
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            mock_3d.assert_called_once()
            mock_2d.assert_called_once()

    def test_main_save_3d(self):
        """Test main() with --save (3D mode)."""
        with (
            patch("sys.argv", ["trajectory_montecarlo.py", "--save"]),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch(
                "visualizations.trajectory_montecarlo.create_3d_animation"
            ) as mock_3d,
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            call_args = mock_3d.call_args
            save_path = call_args[0][6]
            assert save_path is not None
            assert save_path.endswith(".gif")

    def test_main_save_both_mp4(self):
        """Test main() with --save --mode both --format mp4."""
        with (
            patch(
                "sys.argv",
                [
                    "trajectory_montecarlo.py",
                    "--mode",
                    "both",
                    "--save",
                    "--format",
                    "mp4",
                ],
            ),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch(
                "visualizations.trajectory_montecarlo.create_3d_animation"
            ) as mock_3d,
            patch(
                "visualizations.trajectory_montecarlo.create_2d_animation"
            ) as mock_2d,
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            save_3d = mock_3d.call_args[0][6]
            save_2d = mock_2d.call_args[0][6]
            assert save_3d.endswith(".mp4")
            assert save_2d.endswith(".mp4")

    def test_main_j800_rocket(self):
        """Test main() with --rocket j800."""
        with (
            patch("sys.argv", ["trajectory_montecarlo.py", "--rocket", "j800"]),
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch("visualizations.trajectory_montecarlo.create_3d_animation"),
        ):
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            # j800 gains
            call_args = mock_collect.call_args
            assert call_args[0][3] == "gs-pid"

    def test_main_sac_model(self):
        """Test main() with --sac model."""
        with (
            patch(
                "sys.argv",
                ["trajectory_montecarlo.py", "--sac", "/fake/model.zip"],
            ),
            patch("visualizations.trajectory_montecarlo.load_rl_model") as mock_load,
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch("visualizations.trajectory_montecarlo.create_3d_animation"),
        ):
            mock_load.return_value = (MagicMock(), None)
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            call_args = mock_collect.call_args
            assert call_args[0][3] == "sac"

    def test_main_ppo_model(self):
        """Test main() with --ppo model."""
        with (
            patch(
                "sys.argv",
                ["trajectory_montecarlo.py", "--ppo", "/fake/model.zip"],
            ),
            patch("visualizations.trajectory_montecarlo.load_rl_model") as mock_load,
            patch("visualizations.trajectory_montecarlo.collect_data") as mock_collect,
            patch("visualizations.trajectory_montecarlo.create_3d_animation"),
        ):
            mock_load.return_value = (MagicMock(), None)
            mock_collect.return_value = {
                1: [_make_traj()],
                2: [_make_traj()],
                3: [_make_traj()],
            }
            from visualizations.trajectory_montecarlo import main

            main()

            call_args = mock_collect.call_args
            assert call_args[0][3] == "ppo"

    def test_main_residual_sac_model(self):
        """Test main() with --residual-sac (no config.yaml in dir)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "best_model.zip")
            with open(model_path, "w") as f:
                f.write("fake")
            with (
                patch(
                    "sys.argv",
                    [
                        "trajectory_montecarlo.py",
                        "--residual-sac",
                        model_path,
                    ],
                ),
                patch(
                    "visualizations.trajectory_montecarlo.load_rl_model"
                ) as mock_load,
                patch(
                    "visualizations.trajectory_montecarlo.collect_data"
                ) as mock_collect,
                patch("visualizations.trajectory_montecarlo.create_3d_animation"),
            ):
                mock_load.return_value = (MagicMock(), None)
                mock_collect.return_value = {
                    1: [_make_traj()],
                    2: [_make_traj()],
                    3: [_make_traj()],
                }
                from visualizations.trajectory_montecarlo import main

                main()

                call_args = mock_collect.call_args
                assert call_args[0][3] == "residual-sac"
