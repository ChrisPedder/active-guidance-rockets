"""
Tests for compare_controllers.py evaluate_controller and related functions.

Covers: evaluate_controller, run_rl_episode (mocked), create_wrapped_env.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from compare_controllers import (
    EpisodeMetrics,
    ControllerResult,
    create_env,
    create_wrapped_env,
    evaluate_controller,
    run_controller_episode,
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


class TestEvaluateController:
    """Test evaluate_controller function."""

    def test_evaluate_with_pid_config(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        results = evaluate_controller(
            estes_config,
            "PID",
            wind_levels=[0.0],
            n_episodes=2,
            pid_config=pid_config,
        )
        assert len(results) == 1
        assert results[0].controller_name == "PID"
        assert results[0].wind_speed == 0.0
        assert len(results[0].episodes) == 2
        assert results[0].mean_spin >= 0

    def test_evaluate_with_controller_object(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        ctrl = GainScheduledPIDController(pid_config)
        results = evaluate_controller(
            estes_config,
            "GS-PID",
            wind_levels=[0.0],
            n_episodes=2,
            controller=ctrl,
        )
        assert len(results) == 1
        assert results[0].controller_name == "GS-PID"
        assert results[0].mean_spin >= 0

    def test_evaluate_multiple_wind_levels(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        results = evaluate_controller(
            estes_config,
            "PID",
            wind_levels=[0.0, 1.0],
            n_episodes=2,
            pid_config=pid_config,
        )
        assert len(results) == 2
        assert results[0].wind_speed == 0.0
        assert results[1].wind_speed == 1.0

    def test_evaluate_with_observations(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        results = evaluate_controller(
            estes_config,
            "PID (IMU)",
            wind_levels=[0.0],
            n_episodes=2,
            pid_config=pid_config,
            use_observations=True,
        )
        assert len(results) == 1
        assert results[0].controller_name == "PID (IMU)"

    def test_evaluate_with_spin_series(self, estes_config):
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)
        results = evaluate_controller(
            estes_config,
            "PID",
            wind_levels=[0.0],
            n_episodes=2,
            pid_config=pid_config,
            collect_spin_series=True,
        )
        for ep in results[0].episodes:
            assert ep.spin_rate_series is not None
            assert len(ep.spin_rate_series) == ep.episode_length


class TestCreateWrappedEnv:
    """Test create_wrapped_env function."""

    def test_create_wrapped_env_no_wind(self, estes_config):
        """Lines 315-319: create_wrapped_env with no wind."""
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


class TestEvaluateControllerWithModel:
    """Test evaluate_controller with RL model (mocked)."""

    def test_evaluate_with_rl_model(self, estes_config):
        """Lines 371-386: evaluate_controller with model argument."""
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        results = evaluate_controller(
            estes_config,
            "SAC",
            wind_levels=[0.0],
            n_episodes=2,
            model=mock_model,
        )
        assert len(results) == 1
        assert results[0].controller_name == "SAC"
        assert results[0].mean_spin >= 0


class TestRunRlEpisodeMocked:
    """Test run_rl_episode with a mocked RL model."""

    def test_run_rl_episode_mock(self, estes_config):
        from compare_controllers import run_rl_episode

        env = create_env(estes_config, wind_speed=0.0)

        # Create mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        metrics = run_rl_episode(env, mock_model, vec_normalize=None, dt=0.01)
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.mean_spin_rate >= 0
        assert metrics.episode_length > 0
        assert metrics.max_altitude > 0
        env.close()

    def test_run_rl_episode_with_vec_normalize_mock(self, estes_config):
        from compare_controllers import run_rl_episode

        env = create_env(estes_config, wind_speed=0.0)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        mock_vec_normalize = MagicMock()
        mock_vec_normalize.normalize_obs.side_effect = lambda x: x

        metrics = run_rl_episode(
            env, mock_model, vec_normalize=mock_vec_normalize, dt=0.01
        )
        assert isinstance(metrics, EpisodeMetrics)
        assert metrics.episode_length > 0
        # normalize_obs should have been called
        assert mock_vec_normalize.normalize_obs.call_count > 0
        env.close()

    def test_run_rl_episode_with_spin_series(self, estes_config):
        from compare_controllers import run_rl_episode

        env = create_env(estes_config, wind_speed=0.0)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        metrics = run_rl_episode(
            env, mock_model, vec_normalize=None, dt=0.01, collect_spin_series=True
        )
        assert metrics.spin_rate_series is not None
        assert len(metrics.spin_rate_series) == metrics.episode_length
        env.close()
