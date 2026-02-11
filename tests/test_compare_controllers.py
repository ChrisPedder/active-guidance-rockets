"""
Tests for compare_controllers module.

Covers: ControllerResult properties, create_env, create_env_with_imu,
run_pid_episode, run_controller_episode, print_comparison_table, plot_comparison.
"""

import numpy as np
import pytest
from unittest.mock import patch
from io import StringIO

from compare_controllers import (
    EpisodeMetrics,
    ControllerResult,
    create_env,
    create_env_with_imu,
    run_controller_episode,
    run_pid_episode,
    print_comparison_table,
    plot_comparison,
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


class TestPlotComparison:
    """Test plot_comparison."""

    def test_plot_comparison_saves(self, sample_episodes, tmp_path):
        import matplotlib

        matplotlib.use("Agg")

        results = {
            "PID": [
                ControllerResult("PID", 0.0, sample_episodes),
                ControllerResult("PID", 2.0, sample_episodes),
            ],
        }
        save_path = str(tmp_path / "comparison.png")
        plot_comparison(results, save_path=save_path)
        import os

        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
