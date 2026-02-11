"""
Tests for pid_controller.py evaluation functions.

Covers: run_episode, evaluate_pid, print_summary, EpisodeResult.
"""

import numpy as np
import pytest
from io import StringIO
from unittest.mock import patch

from controllers.pid_controller import (
    PIDConfig,
    PIDController,
    GainScheduledPIDController,
    EpisodeResult,
    run_episode,
    evaluate_pid,
    print_summary,
)
from compare_controllers import create_env
from rocket_config import load_config


@pytest.fixture
def estes_config():
    return load_config("configs/estes_c6_sac_wind.yaml")


@pytest.fixture
def pid_config():
    return PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118)


class TestEpisodeResult:
    """Test EpisodeResult dataclass."""

    def test_create(self):
        result = EpisodeResult(
            max_altitude=80.0,
            mean_spin_rate=5.0,
            final_spin_rate=2.0,
            total_reward=100.0,
            episode_length=300,
            times=np.array([0.0, 0.01, 0.02]),
            altitudes=np.array([0.0, 1.0, 2.0]),
            roll_rates=np.array([10.0, 5.0, 2.0]),
            actions=np.array([0.0, 0.1, -0.1]),
        )
        assert result.max_altitude == 80.0
        assert result.mean_spin_rate == 5.0
        assert result.episode_length == 300
        assert len(result.times) == 3


class TestRunEpisode:
    """Test run_episode function."""

    def test_run_episode_basic(self, estes_config, pid_config):
        env = create_env(estes_config, wind_speed=0.0)
        controller = PIDController(pid_config)
        result = run_episode(env, controller, dt=0.01)
        assert isinstance(result, EpisodeResult)
        assert result.max_altitude > 0
        assert result.mean_spin_rate >= 0
        assert result.episode_length > 0
        assert len(result.times) == result.episode_length
        assert len(result.altitudes) == result.episode_length
        assert len(result.roll_rates) == result.episode_length
        assert len(result.actions) == result.episode_length
        env.close()

    def test_run_episode_with_gs_pid(self, estes_config, pid_config):
        env = create_env(estes_config, wind_speed=0.0)
        controller = GainScheduledPIDController(pid_config)
        result = run_episode(env, controller, dt=0.01)
        assert isinstance(result, EpisodeResult)
        assert result.episode_length > 0
        env.close()


class TestEvaluatePid:
    """Test evaluate_pid function."""

    def test_evaluate_pid_basic(self, pid_config):
        results = evaluate_pid(
            "configs/estes_c6_sac_wind.yaml", pid_config, n_episodes=2
        )
        assert len(results) == 2
        for r in results:
            assert isinstance(r, EpisodeResult)
            assert r.max_altitude > 0
            assert r.mean_spin_rate >= 0

    def test_evaluate_pid_with_progress_print(self, pid_config):
        """Lines 546-549: trigger periodic progress print (every 10 episodes)."""
        results = evaluate_pid(
            "configs/estes_c6_sac_wind.yaml", pid_config, n_episodes=10
        )
        assert len(results) == 10


class TestPrintSummary:
    """Test print_summary function."""

    def test_print_summary(self, capsys):
        results = [
            EpisodeResult(
                max_altitude=80.0,
                mean_spin_rate=5.0,
                final_spin_rate=2.0,
                total_reward=100.0,
                episode_length=300,
                times=np.arange(300) * 0.01,
                altitudes=np.linspace(0, 80, 300),
                roll_rates=np.random.randn(300) * 5,
                actions=np.random.randn(300) * 0.1,
            ),
            EpisodeResult(
                max_altitude=75.0,
                mean_spin_rate=8.0,
                final_spin_rate=3.0,
                total_reward=90.0,
                episode_length=280,
                times=np.arange(280) * 0.01,
                altitudes=np.linspace(0, 75, 280),
                roll_rates=np.random.randn(280) * 8,
                actions=np.random.randn(280) * 0.1,
            ),
        ]
        print_summary(results, "PID")
        output = capsys.readouterr().out
        assert "PID Controller Results" in output
        assert "Max Altitude" in output
        assert "Mean Spin Rate" in output
        assert "Total Reward" in output
        assert "Success" in output
