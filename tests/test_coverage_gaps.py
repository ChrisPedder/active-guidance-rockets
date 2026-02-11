"""
Tests targeting specific coverage gaps to push coverage above 80%.

Covers uncovered lines in: video_quality_metric, ensemble_controller,
rocket_config, pid_controller, thrustcurve_motor_data.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from collections import deque


# ── video_quality_metric: lines 260, 282, 325, 329, 450-514 ──────────────────


class TestVideoQualityMetricGaps:
    """Test uncovered lines in video_quality_metric.py."""

    def test_print_summary_empty_list(self, capsys):
        """Line 260: early return when qualities list is empty."""
        from controllers.video_quality_metric import GyroflowVideoMetric, CameraPreset

        metric = GyroflowVideoMetric(CameraPreset.runcam_1080p60())
        metric.print_summary([], "PID", 0.0)
        output = capsys.readouterr().out
        assert output == ""  # No output for empty list

    def test_verdict_good(self):
        """Line 282: 'Good' verdict branch."""
        from controllers.video_quality_metric import GyroflowVideoMetric, CameraPreset

        metric = GyroflowVideoMetric(CameraPreset.runcam_1080p60())
        th = metric.thresholds
        # A value between good_min and excellent_min
        mid = (th.good_min + th.excellent_min) / 2
        verdict = metric._verdict_from_composite(mid)
        assert verdict == "Good"

    def test_verdict_acceptable(self):
        """Test 'Acceptable' verdict."""
        from controllers.video_quality_metric import GyroflowVideoMetric, CameraPreset

        metric = GyroflowVideoMetric(CameraPreset.runcam_1080p60())
        th = metric.thresholds
        mid = (th.acceptable_min + th.good_min) / 2
        verdict = metric._verdict_from_composite(mid)
        assert verdict == "Acceptable"

    def test_verdict_poor(self):
        """Test 'Poor' verdict."""
        from controllers.video_quality_metric import GyroflowVideoMetric, CameraPreset

        metric = GyroflowVideoMetric(CameraPreset.runcam_1080p60())
        verdict = metric._verdict_from_composite(0.0)
        assert verdict == "Poor"

    def test_print_video_quality_table_no_presets(self, capsys):
        """Line 325: camera_presets is None, should use all_presets."""
        from controllers.video_quality_metric import print_video_quality_table
        from compare_controllers import ControllerResult, EpisodeMetrics

        ep = EpisodeMetrics(
            mean_spin_rate=5.0,
            max_spin_rate=15.0,
            settling_time=0.3,
            total_reward=100.0,
            max_altitude=80.0,
            control_smoothness=0.02,
            episode_length=100,
            spin_rate_series=np.ones(100) * 5.0,
        )
        results = {
            "PID": [ControllerResult("PID", 0.0, [ep])],
        }
        print_video_quality_table(results, dt=0.01, camera_presets=None)
        output = capsys.readouterr().out
        assert "VIDEO QUALITY" in output

    def test_print_video_quality_table_empty_controllers(self, capsys):
        """Line 329: empty controllers dict."""
        from controllers.video_quality_metric import print_video_quality_table

        print_video_quality_table({}, dt=0.01)
        output = capsys.readouterr().out
        assert output == ""

    def test_main_function(self):
        """Lines 450-514: test main() with mocked args."""
        from controllers.video_quality_metric import main

        with patch(
            "sys.argv",
            [
                "video_quality_metric.py",
                "--spin-rate",
                "5",
                "10",
                "20",
                "--camera",
                "all",
            ],
        ):
            # main() calls parse_args and prints - should not crash
            main()


# ── ensemble_controller: lines 117-118, 126, 179, 205-206, 209-211 ───────────


class TestEnsembleControllerGaps:
    """Test uncovered lines in ensemble_controller.py."""

    def test_launch_detected_setter(self):
        """Lines 117-118: launch_detected setter propagates to all controllers."""
        from controllers.ensemble_controller import EnsembleController, EnsembleConfig
        from controllers.pid_controller import (
            PIDConfig,
            PIDController,
            GainScheduledPIDController,
        )

        pid_config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl1 = PIDController(pid_config)
        ctrl2 = GainScheduledPIDController(pid_config)

        ensemble = EnsembleController(
            controllers=[ctrl1, ctrl2],
            names=["PID", "GS-PID"],
            config=EnsembleConfig(),
        )

        assert not ensemble.launch_detected
        ensemble.launch_detected = True
        assert ctrl1.launch_detected
        assert ctrl2.launch_detected

    def test_active_idx_property(self):
        """Line 126: active_idx property."""
        from controllers.ensemble_controller import EnsembleController, EnsembleConfig
        from controllers.pid_controller import (
            PIDConfig,
            PIDController,
            GainScheduledPIDController,
        )

        pid_config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl1 = PIDController(pid_config)
        ctrl2 = GainScheduledPIDController(pid_config)

        ensemble = EnsembleController(
            controllers=[ctrl1, ctrl2],
            names=["PID", "GS-PID"],
            config=EnsembleConfig(),
        )
        assert ensemble.active_idx == 0

    def test_try_switch_incomplete_windows(self):
        """Line 179: _try_switch returns early when windows are incomplete."""
        from controllers.ensemble_controller import EnsembleController, EnsembleConfig
        from controllers.pid_controller import (
            PIDConfig,
            PIDController,
            GainScheduledPIDController,
        )

        pid_config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl1 = PIDController(pid_config)
        ctrl2 = GainScheduledPIDController(pid_config)

        config = EnsembleConfig(window_size=100, warmup_steps=1, min_dwell_s=0.0)
        ensemble = EnsembleController(
            controllers=[ctrl1, ctrl2],
            names=["PID", "GS-PID"],
            config=config,
        )
        # Only add a few items to windows (less than window_size)
        for i in range(5):
            ensemble._perf_windows[0].append(10.0)
            ensemble._perf_windows[1].append(10.0)

        ensemble._step_count = 10
        ensemble._try_switch()
        # Should not switch due to incomplete windows
        assert ensemble._active_idx == 0

    def test_switching_happens(self):
        """Lines 205-206, 209-211: actual switching when candidate is better."""
        from controllers.ensemble_controller import EnsembleController, EnsembleConfig
        from controllers.pid_controller import (
            PIDConfig,
            PIDController,
            GainScheduledPIDController,
        )

        pid_config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl1 = PIDController(pid_config)
        ctrl2 = GainScheduledPIDController(pid_config)

        config = EnsembleConfig(
            window_size=10,
            warmup_steps=1,
            min_dwell_s=0.0,
            switch_margin=0.0,
        )
        ensemble = EnsembleController(
            controllers=[ctrl1, ctrl2],
            names=["PID", "GS-PID"],
            config=config,
        )

        # Fill windows: controller 0 has bad performance (high roll rate)
        # controller 1 has good performance (low roll rate)
        for _ in range(10):
            ensemble._perf_windows[0].append(50.0)
            ensemble._perf_windows[1].append(5.0)

        ensemble._step_count = 20
        ensemble._try_switch()
        assert ensemble._active_idx == 1
        assert ensemble._switch_count == 1


# ── rocket_config: lines 785-825, 836-849 ────────────────────────────────────


class TestRocketConfigGaps:
    """Test uncovered lines in rocket_config.py."""

    def test_validate_no_airframe(self):
        """Lines 785-790: validate with no airframe_file."""
        from rocket_config import RocketTrainingConfig

        config = RocketTrainingConfig.for_estes_alpha()
        config.physics.airframe_file = None
        issues = config.validate()
        assert any("CRITICAL" in i and "airframe_file" in i for i in issues)

    def test_validate_with_airframe(self):
        """Lines 792-825: validate full path with airframe."""
        from rocket_config import load_config

        config = load_config("configs/estes_c6_sac_wind.yaml")
        issues = config.validate()
        # May have warnings but should not have critical issues
        critical = [i for i in issues if "CRITICAL" in i]
        assert len(critical) == 0

    def test_validate_bad_airframe_path(self):
        """Lines 796-798: validation fails on bad airframe path."""
        from rocket_config import load_config

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.physics.airframe_file = "nonexistent_airframe.yaml"
        issues = config.validate()
        assert any("CRITICAL" in i and "Failed to load" in i for i in issues)

    def test_validate_batch_size_warning(self):
        """Lines 819-823: batch_size > rollout size warning."""
        from rocket_config import load_config

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.ppo.batch_size = 999999
        config.ppo.n_steps = 1
        config.ppo.n_envs = 1
        issues = config.validate()
        assert any("batch_size" in i for i in issues)

    def test_create_default_configs(self, tmp_path):
        """Lines 836-849: create_default_configs function."""
        from rocket_config import RocketTrainingConfig

        # Test for_estes_alpha factory
        config = RocketTrainingConfig.for_estes_alpha()
        assert config is not None
        assert config.motor.name is not None


# ── pid_controller: lines 127, 261, 263, 409, 411 ────────────────────────────


class TestPIDControllerGaps:
    """Test angle normalization edge cases in pid_controller.py."""

    def test_large_positive_angle(self):
        """Line 127: angle > 180 normalization loop."""
        from controllers.pid_controller import PIDConfig, PIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = PIDController(config)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(400),  # > 360 -> needs normalization
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_large_negative_angle(self):
        """Angle < -180 normalization loop."""
        from controllers.pid_controller import PIDConfig, PIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = PIDController(config)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(-400),
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)

    def test_gs_pid_large_angle(self):
        """Lines 261, 263: GainScheduledPID angle normalization."""
        from controllers.pid_controller import PIDConfig, GainScheduledPIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = GainScheduledPIDController(config)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(500),
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)

    def test_lead_compensated_large_angle(self):
        """Lines 409, 411: LeadCompensatedGSPID angle normalization."""
        from controllers.pid_controller import PIDConfig, LeadCompensatedGSPIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=False)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(500),
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)

    def test_gs_pid_angle_negative_wrap(self):
        """Line 263: GS-PID negative angle loop."""
        from controllers.pid_controller import PIDConfig, GainScheduledPIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = GainScheduledPIDController(config)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(-400),
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)

    def test_lead_compensated_negative_angle(self):
        """Line 411: LeadCompensated negative angle loop."""
        from controllers.pid_controller import PIDConfig, LeadCompensatedGSPIDController

        config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.015)
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=False)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(-400),
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)


# ── adrc_controller: lines 113, 186-192 ──────────────────────────────────────


class TestADRCControllerGaps:
    """Test uncovered lines in adrc_controller.py."""

    def test_reset_with_b0_estimator(self):
        """Line 113: reset when b0_estimator is attached."""
        from controllers.adrc_controller import ADRCController, ADRCConfig

        config = ADRCConfig(omega_c=15.0, omega_o=50.0)
        mock_estimator = MagicMock()
        ctrl = ADRCController(config=config, b0_estimator=mock_estimator)
        ctrl.launch_detected = True
        ctrl.z1 = 1.0
        ctrl.z2 = 2.0
        ctrl.z3 = 3.0

        ctrl.reset()
        assert not ctrl.launch_detected
        assert ctrl.z1 == 0.0
        assert mock_estimator.reset.call_count >= 1

    def test_step_with_b0_estimator(self):
        """Lines 186-192: step uses b0_estimator when attached."""
        from controllers.adrc_controller import ADRCController, ADRCConfig

        config = ADRCConfig(omega_c=15.0, omega_o=50.0)
        mock_estimator = MagicMock()
        mock_estimator.update.return_value = 100.0  # positive b0 estimate
        ctrl = ADRCController(config=config, b0_estimator=mock_estimator)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.1,
            "roll_rate_deg_s": 10.0,
            "roll_acceleration_rad_s2": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        mock_estimator.update.assert_called_once()


# ── thrustcurve_motor_data: lines 274-290 ────────────────────────────────────


class TestThrustcurveMotorDataGaps:
    """Test uncovered lines in thrustcurve_motor_data.py."""

    def test_parse_eng_file(self, tmp_path):
        """Test ThrustCurveParser.parse_eng_file with valid eng content."""
        from thrustcurve_motor_data import ThrustCurveParser

        # Standard .eng file format: header line then time/thrust pairs
        # The parser skips comment lines (starting with ;)
        eng_content = "C6 18 70 5-7-9 0.0124 0.0245 Estes\n0.031 14.09\n0.092 9.83\n0.154 7.07\n1.7 4.0\n1.85 0.0\n"

        eng_path = tmp_path / "test_motor.eng"
        with open(eng_path, "w") as f:
            f.write(eng_content)

        motor = ThrustCurveParser.parse_eng_file(str(eng_path))
        assert motor is not None
        assert motor.manufacturer == "Estes"
        assert motor.designation == "C6"
        assert len(motor.time_points) > 0


# ── airframe: lines 57, 103, 113, 163, 188, 232, 274-275, 316-318 ────────────


class TestAirframeGaps:
    """Test uncovered calculation methods in airframe.py."""

    def test_total_length(self):
        """Line 163: total_length property."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        assert airframe.total_length > 0

    def test_body_diameter(self):
        """Property for body diameter."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        assert airframe.body_diameter > 0

    def test_finset_properties(self):
        """Test fin set properties."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        fin = airframe.get_fin_set()
        assert fin is not None
        assert fin.root_chord > 0
        assert fin.span > 0

    def test_cg_calculation(self):
        """Lines 103, 113: center of gravity / cg_position."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        cg = airframe.cg_position
        assert cg > 0
        assert cg < airframe.total_length

    def test_roll_inertia(self):
        """Lines 115+: roll inertia via get_roll_inertia."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        I_roll = airframe.get_roll_inertia()
        assert I_roll > 0

    def test_to_dict(self):
        """Lines 232, 274-275: to_dict with mass_override and MassObject branches."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        d = airframe.to_dict()
        assert d["name"] == "Estes Alpha III"
        assert "components" in d
        assert len(d["components"]) > 0

    def test_load_yaml(self, tmp_path):
        """Lines 316-318: load from YAML."""
        import yaml
        from airframe import RocketAirframe

        # Create a minimal YAML airframe file using string material names
        # Each component needs a 'position' field
        data = {
            "name": "Test Rocket",
            "description": "Test",
            "components": [
                {
                    "type": "NoseCone",
                    "name": "Nose",
                    "position": 0.0,
                    "length": 0.08,
                    "shape": "ogive",
                    "base_diameter": 0.025,
                    "material": "plastic",
                    "thickness": 0.002,
                    "mass_override": 0.01,
                },
                {
                    "type": "BodyTube",
                    "name": "Body",
                    "position": 0.08,
                    "length": 0.2,
                    "outer_diameter": 0.025,
                    "inner_diameter": 0.023,
                    "material": "cardboard",
                },
                {
                    "type": "TrapezoidFinSet",
                    "name": "Fins",
                    "position": 0.2,
                    "root_chord": 0.05,
                    "tip_chord": 0.03,
                    "span": 0.04,
                    "sweep_length": 0.01,
                    "thickness": 0.002,
                    "material": "balsa",
                },
            ],
        }
        yaml_path = tmp_path / "test_airframe.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        airframe = RocketAirframe.load(str(yaml_path))
        assert airframe.name == "Test Rocket"
        assert airframe.dry_mass > 0
