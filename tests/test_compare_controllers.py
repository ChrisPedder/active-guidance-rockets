"""
Tests for compare_controllers module.

Covers: EpisodeMetrics, ControllerResult properties, create_env, create_env_with_imu,
run_pid_episode, run_controller_episode, run_rl_episode, load_rl_model,
evaluate_controller, create_wrapped_env, print_comparison_table, plot_comparison,
and main() with various flag combinations.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

matplotlib.use("Agg")

from compare_controllers import (
    EpisodeMetrics,
    ControllerResult,
    create_env,
    create_env_with_imu,
    create_wrapped_env,
    run_controller_episode,
    run_pid_episode,
    run_rl_episode,
    load_rl_model,
    evaluate_controller,
    print_comparison_table,
    plot_comparison,
    main,
)
from controllers.pid_controller import (
    PIDConfig,
    PIDController,
    GainScheduledPIDController,
)
from rocket_config import load_config


@pytest.fixture
def estes_config():
    return load_config("configs/estes_c6_sac_wind.yaml")


@pytest.fixture
def sample_episodes():
    """Create sample EpisodeMetrics for testing."""
    return [
        EpisodeMetrics(
            mean_spin_rate=5.0,
            max_spin_rate=15.0,
            settling_time=0.3,
            total_reward=100.0,
            max_altitude=80.0,
            control_smoothness=0.02,
            episode_length=300,
        ),
        EpisodeMetrics(
            mean_spin_rate=8.0,
            max_spin_rate=25.0,
            settling_time=0.5,
            total_reward=80.0,
            max_altitude=75.0,
            control_smoothness=0.03,
            episode_length=280,
        ),
        EpisodeMetrics(
            mean_spin_rate=12.0,
            max_spin_rate=35.0,
            settling_time=float("inf"),
            total_reward=60.0,
            max_altitude=70.0,
            control_smoothness=0.04,
            episode_length=250,
        ),
    ]


# ---------------------------------------------------------------------------
# EpisodeMetrics dataclass
# ---------------------------------------------------------------------------


class TestEpisodeMetrics:
    """Test EpisodeMetrics creation and defaults."""

    def test_creation_basic(self):
        m = EpisodeMetrics(
            mean_spin_rate=5.0,
            max_spin_rate=10.0,
            settling_time=0.3,
            total_reward=50.0,
            max_altitude=80.0,
            control_smoothness=0.01,
            episode_length=200,
        )
        assert m.mean_spin_rate == 5.0
        assert m.max_spin_rate == 10.0
        assert m.settling_time == 0.3
        assert m.total_reward == 50.0
        assert m.max_altitude == 80.0
        assert m.control_smoothness == 0.01
        assert m.episode_length == 200
        assert m.spin_rate_series is None

    def test_creation_with_spin_series(self):
        series = np.array([1.0, 2.0, 3.0])
        m = EpisodeMetrics(
            mean_spin_rate=2.0,
            max_spin_rate=3.0,
            settling_time=0.1,
            total_reward=90.0,
            max_altitude=100.0,
            control_smoothness=0.005,
            episode_length=3,
            spin_rate_series=series,
        )
        np.testing.assert_array_equal(m.spin_rate_series, series)


# ---------------------------------------------------------------------------
# ControllerResult properties
# ---------------------------------------------------------------------------


class TestControllerResult:
    """Test ControllerResult properties."""

    def test_mean_spin(self, sample_episodes):
        cr = ControllerResult("PID", 1.0, sample_episodes)
        assert cr.mean_spin == pytest.approx(np.mean([5.0, 8.0, 12.0]))

    def test_std_spin(self, sample_episodes):
        cr = ControllerResult("PID", 1.0, sample_episodes)
        assert cr.std_spin == pytest.approx(np.std([5.0, 8.0, 12.0]))

    def test_mean_settling_ignores_inf(self, sample_episodes):
        cr = ControllerResult("PID", 1.0, sample_episodes)
        # Only finite values: 0.3, 0.5
        assert cr.mean_settling == pytest.approx(np.mean([0.3, 0.5]))

    def test_mean_settling_all_inf(self):
        eps = [
            EpisodeMetrics(5.0, 15.0, float("inf"), 0.0, 80.0, 0.02, 100),
        ]
        cr = ControllerResult("PID", 1.0, eps)
        assert cr.mean_settling == float("inf")

    def test_success_rate(self, sample_episodes):
        cr = ControllerResult("PID", 1.0, sample_episodes)
        # All < 30 deg/s
        assert cr.success_rate == pytest.approx(1.0)

    def test_success_rate_partial(self):
        eps = [
            EpisodeMetrics(5.0, 15.0, 0.3, 100.0, 80.0, 0.02, 300),
            EpisodeMetrics(35.0, 50.0, float("inf"), 20.0, 80.0, 0.1, 300),
        ]
        cr = ControllerResult("PID", 1.0, eps)
        assert cr.success_rate == pytest.approx(0.5)

    def test_mean_smoothness(self, sample_episodes):
        cr = ControllerResult("PID", 1.0, sample_episodes)
        expected = np.mean([0.02, 0.03, 0.04])
        assert cr.mean_smoothness == pytest.approx(expected)


# ---------------------------------------------------------------------------
# create_env / create_env_with_imu
# ---------------------------------------------------------------------------


class TestCreateEnv:
    """Test environment creation functions."""

    def test_create_env_no_wind(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 10
        env.close()

    def test_create_env_with_wind(self, estes_config):
        env = create_env(estes_config, wind_speed=2.0)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_create_env_with_imu(self, estes_config):
        env = create_env_with_imu(estes_config, wind_speed=1.0)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_create_env_with_imu_no_wind(self, estes_config):
        env = create_env_with_imu(estes_config, wind_speed=0.0)
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_create_env_with_imu_custom(self, estes_config):
        """Cover line 184: imu_custom branch in create_env_with_imu."""
        import copy

        cfg = copy.deepcopy(estes_config)
        cfg.sensors.imu_custom = {
            "name": "custom_test",
            "gyro": {
                "noise_density": 0.01,
                "bias_instability": 5.0,
                "bias_initial_range": 0.5,
            },
        }
        env = create_env_with_imu(cfg, wind_speed=0.0)
        obs, info = env.reset()
        assert obs is not None
        env.close()


# ---------------------------------------------------------------------------
# run_controller_episode / run_pid_episode
# ---------------------------------------------------------------------------


class TestRunEpisode:
    """Test episode running functions."""

    def test_run_controller_episode(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        controller = PIDController(pid_config)
        metrics = run_controller_episode(env, controller, dt=0.01)
        assert metrics.mean_spin_rate >= 0
        assert metrics.max_spin_rate >= 0
        assert metrics.episode_length > 0
        assert metrics.max_altitude > 0
        env.close()

    def test_run_controller_episode_with_spin_series(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        controller = PIDController(pid_config)
        metrics = run_controller_episode(
            env, controller, dt=0.01, collect_spin_series=True
        )
        assert metrics.spin_rate_series is not None
        assert len(metrics.spin_rate_series) == metrics.episode_length
        env.close()

    def test_run_pid_episode(self, estes_config):
        env = create_env(estes_config, wind_speed=1.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        metrics = run_pid_episode(env, pid_config, dt=0.01)
        assert metrics.mean_spin_rate >= 0
        assert metrics.episode_length > 0
        env.close()

    def test_run_pid_episode_with_imu(self, estes_config):
        env = create_env_with_imu(estes_config, wind_speed=1.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        metrics = run_pid_episode(env, pid_config, dt=0.01, use_observations=True)
        assert metrics.mean_spin_rate >= 0
        env.close()

    def test_run_gs_pid_episode(self, estes_config):
        env = create_env(estes_config, wind_speed=2.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        controller = GainScheduledPIDController(pid_config)
        metrics = run_controller_episode(env, controller, dt=0.01)
        assert metrics.mean_spin_rate >= 0
        env.close()


# ---------------------------------------------------------------------------
# run_rl_episode (mocked)
# ---------------------------------------------------------------------------


class TestRunRlEpisode:
    """Test run_rl_episode with mocked RL models."""

    def test_run_rl_episode_no_vec_normalize(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        metrics = run_rl_episode(env, mock_model, vec_normalize=None, dt=0.01)
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.mean_spin_rate >= 0
        assert metrics.episode_length > 0
        assert metrics.max_altitude > 0
        env.close()

    def test_run_rl_episode_with_vec_normalize(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        mock_vec = MagicMock()
        mock_vec.normalize_obs.side_effect = lambda x: x

        metrics = run_rl_episode(env, mock_model, vec_normalize=mock_vec, dt=0.01)
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.episode_length > 0
        assert mock_vec.normalize_obs.call_count > 0
        env.close()

    def test_run_rl_episode_with_spin_series(self, estes_config):
        env = create_env(estes_config, wind_speed=0.0)
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        metrics = run_rl_episode(
            env, mock_model, vec_normalize=None, dt=0.01, collect_spin_series=True
        )
        assert metrics.spin_rate_series is not None
        assert len(metrics.spin_rate_series) == metrics.episode_length
        env.close()


# ---------------------------------------------------------------------------
# load_rl_model
# ---------------------------------------------------------------------------


class TestLoadRlModel:
    """Test load_rl_model with mocked SAC/PPO loaders."""

    def test_load_sac_model_no_vec_normalize(self, estes_config, tmp_path):
        """Lines 322-340: load SAC model, no vec_normalize.pkl present."""
        fake_model = tmp_path / "best_model.zip"
        fake_model.touch()

        mock_sac_instance = MagicMock()
        with patch(
            "compare_controllers.SAC.load", return_value=mock_sac_instance
        ) as mock_load:
            model, vec_normalize = load_rl_model(str(fake_model), "sac", estes_config)
            mock_load.assert_called_once_with(str(fake_model))
            assert model is mock_sac_instance
            assert vec_normalize is None

    def test_load_ppo_model_no_vec_normalize(self, estes_config, tmp_path):
        """Load PPO model, no vec_normalize.pkl present."""
        fake_model = tmp_path / "best_model.zip"
        fake_model.touch()

        mock_ppo_instance = MagicMock()
        with patch(
            "compare_controllers.PPO.load", return_value=mock_ppo_instance
        ) as mock_load:
            model, vec_normalize = load_rl_model(str(fake_model), "ppo", estes_config)
            mock_load.assert_called_once_with(str(fake_model))
            assert model is mock_ppo_instance
            assert vec_normalize is None

    def test_load_sac_model_with_vec_normalize(self, estes_config, tmp_path):
        """Lines 333-338: load SAC model with vec_normalize.pkl present."""
        fake_model = tmp_path / "best_model.zip"
        fake_model.touch()
        vec_norm_file = tmp_path / "vec_normalize.pkl"
        vec_norm_file.touch()

        mock_sac_instance = MagicMock()
        mock_vec_normalize = MagicMock()
        mock_vec_normalize.training = True

        with (
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
            patch(
                "compare_controllers.VecNormalize.load",
                return_value=mock_vec_normalize,
            ) as mock_vn_load,
        ):
            model, vec_norm = load_rl_model(str(fake_model), "sac", estes_config)
            assert model is mock_sac_instance
            assert vec_norm is mock_vec_normalize
            assert mock_vec_normalize.training is False
            mock_vn_load.assert_called_once()


# ---------------------------------------------------------------------------
# create_wrapped_env
# ---------------------------------------------------------------------------


class TestCreateWrappedEnv:
    """Test create_wrapped_env function."""

    def test_create_wrapped_env_no_wind(self, estes_config):
        env = create_wrapped_env(estes_config, wind_speed=0.0)
        assert env is not None
        obs, info = env.reset()
        assert obs is not None
        env.close()

    def test_create_wrapped_env_with_wind(self, estes_config):
        env = create_wrapped_env(estes_config, wind_speed=2.0)
        assert env is not None
        obs, info = env.reset()
        assert obs is not None
        env.close()


# ---------------------------------------------------------------------------
# evaluate_controller
# ---------------------------------------------------------------------------


class TestEvaluateController:
    """Test evaluate_controller function."""

    def test_evaluate_with_pid_config(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        results = evaluate_controller(
            estes_config,
            "PID",
            wind_levels=[0.0],
            n_episodes=1,
            pid_config=pid_config,
        )
        assert len(results) == 1
        assert results[0].controller_name == "PID"
        assert results[0].wind_speed == 0.0
        assert len(results[0].episodes) == 1
        assert results[0].mean_spin >= 0

    def test_evaluate_with_controller_object(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        ctrl = GainScheduledPIDController(pid_config)
        results = evaluate_controller(
            estes_config,
            "GS-PID",
            wind_levels=[0.0],
            n_episodes=1,
            controller=ctrl,
        )
        assert len(results) == 1
        assert results[0].controller_name == "GS-PID"
        assert results[0].mean_spin >= 0

    def test_evaluate_with_rl_model(self, estes_config):
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)
        results = evaluate_controller(
            estes_config,
            "SAC",
            wind_levels=[0.0],
            n_episodes=1,
            model=mock_model,
        )
        assert len(results) == 1
        assert results[0].controller_name == "SAC"
        assert results[0].mean_spin >= 0


# ---------------------------------------------------------------------------
# print_comparison_table
# ---------------------------------------------------------------------------


class TestPrintComparisonTable:
    """Test print_comparison_table output."""

    def test_print_comparison_table(self, sample_episodes, capsys):
        results = {
            "PID": [
                ControllerResult("PID", 0.0, sample_episodes),
                ControllerResult("PID", 2.0, sample_episodes),
            ],
            "GS-PID": [
                ControllerResult("GS-PID", 0.0, sample_episodes),
                ControllerResult("GS-PID", 2.0, sample_episodes),
            ],
        }
        print_comparison_table(results)
        output = capsys.readouterr().out
        assert "CONTROLLER COMPARISON" in output
        assert "PID" in output
        assert "GS-PID" in output
        assert "Mean Spin Rate" in output
        assert "Success Rate" in output
        assert "Settling Time" in output
        assert "Control Smoothness" in output

    def test_print_comparison_table_with_inf_settling(self, capsys):
        """Ensure N/A is printed when all settling times are infinite."""
        eps = [
            EpisodeMetrics(5.0, 15.0, float("inf"), 0.0, 80.0, 0.02, 100),
        ]
        results = {
            "PID": [ControllerResult("PID", 0.0, eps)],
        }
        print_comparison_table(results)
        output = capsys.readouterr().out
        assert "N/A" in output


# ---------------------------------------------------------------------------
# plot_comparison
# ---------------------------------------------------------------------------


class TestPlotComparison:
    """Test plot_comparison."""

    def test_plot_comparison_saves(self, sample_episodes, tmp_path):
        results = {
            "PID": [
                ControllerResult("PID", 0.0, sample_episodes),
                ControllerResult("PID", 2.0, sample_episodes),
            ],
        }
        save_path = str(tmp_path / "comparison.png")
        plot_comparison(results, save_path=save_path)
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0

    def test_plot_comparison_no_save(self, sample_episodes):
        """Plot without saving (save_path=None)."""
        results = {
            "PID": [
                ControllerResult("PID", 0.0, sample_episodes),
            ],
        }
        # Should not raise
        plot_comparison(results, save_path=None)

    def test_plot_comparison_multiple_controllers(self, sample_episodes):
        """Plot with multiple controllers exercises color lookup."""
        results = {
            "PID": [
                ControllerResult("PID", 0.0, sample_episodes),
            ],
            "GS-PID": [
                ControllerResult("GS-PID", 0.0, sample_episodes),
            ],
            "Custom": [
                ControllerResult("Custom", 0.0, sample_episodes),
            ],
        }
        # "Custom" falls back to gray color
        plot_comparison(results, save_path=None)

    def test_plot_comparison_all_inf_settling(self):
        """Plot where all settling times are inf - exercises valid_settling filter."""
        eps = [
            EpisodeMetrics(5.0, 15.0, float("inf"), 0.0, 80.0, 0.02, 100),
        ]
        results = {
            "PID": [ControllerResult("PID", 0.0, eps)],
        }
        plot_comparison(results, save_path=None)


# ---------------------------------------------------------------------------
# main() — PID-only baseline
# ---------------------------------------------------------------------------


class TestMainPidOnly:
    """Test main() with --pid-only to cover lines 598-727, 1020-1038."""

    @patch("compare_controllers.plt.show")
    def test_main_pid_only(self, mock_show, tmp_path):
        """Covers main arg parsing, PID baseline, print_comparison_table, plot."""
        save_plot = str(tmp_path / "test_plot.png")
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--pid-only",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
            "--save-plot",
            save_plot,
        ]
        with patch("sys.argv", test_args):
            main()
        assert os.path.exists(save_plot)

    @patch("compare_controllers.plt.show")
    def test_main_pid_with_imu_flag(self, mock_show):
        """Covers lines 702-704: --imu flag and imu_suffix logic."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--pid-only",
            "--imu",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()

    @patch("compare_controllers.plt.show")
    def test_main_pid_imu_deprecated(self, mock_show):
        """Covers backward-compatible --pid-imu flag."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--pid-only",
            "--pid-imu",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()


# ---------------------------------------------------------------------------
# main() — Gain-scheduled PID
# ---------------------------------------------------------------------------


class TestMainGainScheduled:
    """Test main() with --gain-scheduled."""

    @patch("compare_controllers.plt.show")
    def test_main_gain_scheduled(self, mock_show):
        """Covers lines 730-749: gain-scheduled PID path in main."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--gain-scheduled",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()


# ---------------------------------------------------------------------------
# main() — Lead-compensated GS-PID
# ---------------------------------------------------------------------------


class TestMainLead:
    """Test main() with --lead."""

    @patch("compare_controllers.plt.show")
    def test_main_lead(self, mock_show):
        """Covers lines 752-773: lead-compensated GS-PID path in main."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--lead",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()


# ---------------------------------------------------------------------------
# main() — ADRC
# ---------------------------------------------------------------------------


class TestMainAdrc:
    """Test main() with --adrc."""

    @patch("compare_controllers.plt.show")
    def test_main_adrc(self, mock_show):
        """Covers lines 900-928: ADRC controller path in main."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--adrc",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()


# ---------------------------------------------------------------------------
# main() — Ensemble
# ---------------------------------------------------------------------------


class TestMainEnsemble:
    """Test main() with --ensemble."""

    @patch("compare_controllers.plt.show")
    def test_main_ensemble(self, mock_show):
        """Covers lines 776-815: ensemble controller path in main."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--ensemble",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()


# ---------------------------------------------------------------------------
# main() — SAC model (mocked)
# ---------------------------------------------------------------------------


class TestMainSac:
    """Test main() with --sac flag using a mocked SAC model."""

    @patch("compare_controllers.plt.show")
    def test_main_sac(self, mock_show, tmp_path):
        """Covers lines 946-958: SAC model loading and evaluation in main."""
        fake_model = tmp_path / "best_model.zip"
        fake_model.touch()

        mock_sac_instance = MagicMock()
        mock_sac_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--sac",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
        ):
            main()


# ---------------------------------------------------------------------------
# main() — PPO model (mocked)
# ---------------------------------------------------------------------------


class TestMainPpo:
    """Test main() with --ppo flag using a mocked PPO model."""

    @patch("compare_controllers.plt.show")
    def test_main_ppo(self, mock_show, tmp_path):
        """Covers lines 931-943: PPO model loading and evaluation in main."""
        fake_model = tmp_path / "best_model.zip"
        fake_model.touch()

        mock_ppo_instance = MagicMock()
        mock_ppo_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--ppo",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.PPO.load", return_value=mock_ppo_instance),
        ):
            main()


# ---------------------------------------------------------------------------
# main() — Residual SAC (mocked, no config.yaml in model dir)
# ---------------------------------------------------------------------------


class TestMainResidualSac:
    """Test main() with --residual-sac using mocked models."""

    @patch("compare_controllers.plt.show")
    def test_main_residual_sac_no_config(self, mock_show, tmp_path):
        """Covers lines 961-987: residual SAC fallback path (no config.yaml)."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        fake_model = model_dir / "best_model.zip"
        fake_model.touch()

        mock_sac_instance = MagicMock()
        mock_sac_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--residual-sac",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
        ):
            main()

    @patch("compare_controllers.plt.show")
    def test_main_residual_sac_with_config(self, mock_show, tmp_path):
        """Covers lines 967-969: residual SAC with config.yaml in model dir."""
        import shutil

        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        fake_model = model_dir / "best_model.zip"
        fake_model.touch()
        # Copy a real config so load_config works
        shutil.copy("configs/estes_c6_sac_wind.yaml", model_dir / "config.yaml")

        mock_sac_instance = MagicMock()
        mock_sac_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--residual-sac",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
        ):
            main()


# ---------------------------------------------------------------------------
# main() — DOB SAC (mocked)
# ---------------------------------------------------------------------------


class TestMainDobSac:
    """Test main() with --dob-sac using mocked models."""

    @patch("compare_controllers.plt.show")
    def test_main_dob_sac_no_config(self, mock_show, tmp_path):
        """Covers lines 990-1018: DOB SAC fallback path (no config.yaml)."""
        model_dir = tmp_path / "dob_dir"
        model_dir.mkdir()
        fake_model = model_dir / "best_model.zip"
        fake_model.touch()

        mock_sac_instance = MagicMock()
        mock_sac_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--dob-sac",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
        ):
            main()

    @patch("compare_controllers.plt.show")
    def test_main_dob_sac_with_config(self, mock_show, tmp_path):
        """Covers lines 996-1000: DOB SAC with config.yaml in model dir."""
        import shutil

        model_dir = tmp_path / "dob_dir"
        model_dir.mkdir()
        fake_model = model_dir / "best_model.zip"
        fake_model.touch()
        shutil.copy("configs/estes_c6_sac_wind.yaml", model_dir / "config.yaml")

        mock_sac_instance = MagicMock()
        mock_sac_instance.predict.return_value = (np.array([0.0]), None)

        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--dob-sac",
            str(fake_model),
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with (
            patch("sys.argv", test_args),
            patch("compare_controllers.SAC.load", return_value=mock_sac_instance),
        ):
            main()


# ---------------------------------------------------------------------------
# main() — pid-only suppresses other flags
# ---------------------------------------------------------------------------


class TestMainPidOnlySuppression:
    """Verify --pid-only suppresses --gain-scheduled, --adrc, --ensemble, etc."""

    @patch("compare_controllers.plt.show")
    def test_pid_only_suppresses_gain_scheduled(self, mock_show, capsys):
        """With --pid-only, --gain-scheduled is ignored."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--pid-only",
            "--gain-scheduled",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()
        output = capsys.readouterr().out
        # Only PID should appear, not GS-PID
        assert "GS-PID" not in output.split("CONTROLLER COMPARISON")[1]


# ---------------------------------------------------------------------------
# main() — custom PID gains
# ---------------------------------------------------------------------------


class TestMainCustomGains:
    """Test main() with custom PID gain arguments."""

    @patch("compare_controllers.plt.show")
    def test_main_custom_pid_gains(self, mock_show):
        """Covers lines 660-674, 707-711: custom PID gains via CLI args."""
        test_args = [
            "compare_controllers.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--pid-only",
            "--pid-Kp",
            "0.02",
            "--pid-Ki",
            "0.001",
            "--pid-Kd",
            "0.03",
            "--wind-levels",
            "0",
            "--n-episodes",
            "1",
        ]
        with patch("sys.argv", test_args):
            main()
