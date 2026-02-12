"""
Tests for optimize_pid module.

Covers: evaluate_gains, objective, search space bounds, run_lhs_phase.
Uses minimal episodes and wind levels for speed.
"""

import numpy as np
import pytest

from optimization.optimize_pid import (
    KP_BOUNDS,
    KI_BOUNDS,
    KD_BOUNDS,
    QREF_BOUNDS,
    evaluate_gains,
    objective,
    run_lhs_phase,
    run_nelder_mead_phase,
)
from rocket_config import load_config


@pytest.fixture
def estes_config():
    return load_config("configs/estes_c6_sac_wind.yaml")


class TestSearchBounds:
    """Test search space bounds are reasonable."""

    def test_kp_bounds(self):
        assert KP_BOUNDS[0] > 0
        assert KP_BOUNDS[1] > KP_BOUNDS[0]

    def test_ki_bounds(self):
        assert KI_BOUNDS[0] > 0
        assert KI_BOUNDS[1] > KI_BOUNDS[0]

    def test_kd_bounds(self):
        assert KD_BOUNDS[0] > 0
        assert KD_BOUNDS[1] > KD_BOUNDS[0]

    def test_qref_bounds(self):
        assert QREF_BOUNDS[0] > 0
        assert QREF_BOUNDS[1] > QREF_BOUNDS[0]

    def test_optimized_gains_within_bounds(self):
        """Check that the documented optimized gains are within bounds."""
        assert KP_BOUNDS[0] <= 0.0203 <= KP_BOUNDS[1]
        assert KI_BOUNDS[0] <= 0.0002 <= KI_BOUNDS[1]
        assert KD_BOUNDS[0] <= 0.0118 <= KD_BOUNDS[1]


class TestEvaluateGains:
    """Test evaluate_gains function."""

    def test_evaluate_gains_basic(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=2,
        )
        assert "kp" in result
        assert "ki" in result
        assert "kd" in result
        assert "score" in result
        assert "weighted_spin" in result
        assert "by_wind" in result
        assert 0 in result["by_wind"]
        assert result["by_wind"][0]["mean_spin"] >= 0

    def test_evaluate_gains_with_wind(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0, 2],
            n_episodes=2,
        )
        assert 0 in result["by_wind"]
        assert 2 in result["by_wind"]
        # Higher wind should generally give higher spin
        # (not guaranteed with 2 episodes but structure should exist)
        assert result["by_wind"][2]["mean_spin"] >= 0

    def test_evaluate_gains_gain_scheduled(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=2,
            gain_scheduled=True,
            q_ref=500.0,
        )
        assert "q_ref" in result
        assert result["q_ref"] == 500.0

    def test_evaluate_gains_with_imu(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=2,
            imu=True,
        )
        assert result["score"] >= 0

    def test_evaluate_gains_custom_weights(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0, 2],
            n_episodes=2,
            wind_weights={0: 0.5, 2: 0.5},
        )
        assert result["score"] >= 0

    def test_evaluate_gains_success_penalty(self, estes_config):
        """Terrible gains should produce success penalty."""
        result = evaluate_gains(
            kp=0.001,
            ki=0.001,
            kd=0.001,
            config=estes_config,
            wind_levels=[0],
            n_episodes=2,
        )
        # Score should exist, may or may not have penalty
        assert "success_penalty" in result
        assert result["success_penalty"] >= 0


class TestObjective:
    """Test objective wrapper function."""

    def test_objective_returns_float(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118],
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            wind_weights={0: 1.0},
            base_seed=42,
        )
        assert isinstance(score, float)
        assert score >= 0

    def test_objective_gain_scheduled(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118],
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            wind_weights={0: 1.0},
            base_seed=42,
            gain_scheduled=True,
        )
        assert isinstance(score, float)

    def test_objective_with_qref(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118, 5000.0],
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            wind_weights={0: 1.0},
            base_seed=42,
            gain_scheduled=True,
            optimize_qref=True,
        )
        assert isinstance(score, float)


class TestRunLHSPhase:
    """Test Latin Hypercube Sampling phase."""

    def test_lhs_returns_results(self, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            n_samples=3,
            wind_weights={0: 1.0},
        )
        assert len(results) == 3
        for r in results:
            assert "score" in r
            assert "kp" in r

    def test_lhs_gain_scheduled(self, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            n_samples=2,
            wind_weights={0: 1.0},
            gain_scheduled=True,
        )
        assert len(results) == 2

    def test_lhs_with_qref(self, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            n_samples=2,
            wind_weights={0: 1.0},
            gain_scheduled=True,
            optimize_qref=True,
        )
        assert len(results) == 2
        for r in results:
            assert "q_ref" in r


class TestRunNelderMeadPhase:
    """Test Nelder-Mead refinement phase."""

    def test_nelder_mead_basic(self, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            wind_weights={0: 1.0},
            start_point=(0.0203, 0.0002, 0.0118),
            maxiter=3,
        )
        assert "score" in result
        assert "kp" in result
        assert "ki" in result
        assert "kd" in result
        assert result["score"] >= 0

    def test_nelder_mead_gain_scheduled(self, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=2,
            wind_weights={0: 1.0},
            start_point=(0.0203, 0.0002, 0.0118),
            gain_scheduled=True,
            maxiter=3,
        )
        assert "score" in result
        assert result["score"] >= 0
