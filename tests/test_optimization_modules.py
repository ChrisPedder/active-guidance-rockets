"""
Tests for optimization modules targeting coverage gaps above 80%.

Covers uncovered code in:
- optimization/bayesian_optimize.py (evaluate_adrc, evaluate_ensemble,
  compute_objective edge cases, ParamLookupTable, get_baseline_params,
  bounds constants, EVALUATORS dict, optimize_one_wind_level, main)
- optimization/optimize_pid.py (evaluate_gains with gain_scheduled/imu,
  objective with optimize_qref, bounds constants, run_lhs_phase,
  run_nelder_mead_phase, main)

Uses mocks for simulation-heavy functions to keep runtime fast.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from optimization.bayesian_optimize import (
    ADRC_BOUNDS,
    ENSEMBLE_BOUNDS,
    EVALUATORS,
    GS_PID_BOUNDS,
    OptimizationResult,
    ParamLookupTable,
    compute_objective,
    evaluate_adrc,
    evaluate_ensemble,
    evaluate_gs_pid,
    get_baseline_params,
    optimize_one_wind_level,
)
from optimization.optimize_pid import (
    KD_BOUNDS,
    KI_BOUNDS,
    KP_BOUNDS,
    QREF_BOUNDS,
    evaluate_gains,
    objective,
    run_lhs_phase,
    run_nelder_mead_phase,
)
from rocket_config import load_config


@pytest.fixture(scope="module")
def estes_config():
    """Load the Estes C6 config once for all tests in this module."""
    return load_config("configs/estes_c6_sac_wind.yaml")


# =====================================================================
# bayesian_optimize.py — OptimizationResult dataclass
# =====================================================================


class TestOptimizationResult:
    """Test OptimizationResult dataclass creation and field access."""

    def test_create_optimization_result(self):
        result = OptimizationResult(
            wind_level=2.0,
            controller="gs-pid",
            params={"Kp": 0.01, "Ki": 0.001, "Kd": 0.02, "q_ref": 500.0},
            mean_spin=8.5,
            std_spin=2.3,
            objective=13.1,
            success_rate=0.95,
            n_evaluations=100,
        )
        assert result.wind_level == 2.0
        assert result.controller == "gs-pid"
        assert result.params["Kp"] == 0.01
        assert result.mean_spin == 8.5
        assert result.std_spin == 2.3
        assert result.objective == 13.1
        assert result.success_rate == 0.95
        assert result.n_evaluations == 100

    def test_optimization_result_equality(self):
        kwargs = dict(
            wind_level=1.0,
            controller="adrc",
            params={"omega_c": 15.0, "omega_o": 50.0},
            mean_spin=10.0,
            std_spin=3.0,
            objective=16.0,
            success_rate=0.9,
            n_evaluations=50,
        )
        r1 = OptimizationResult(**kwargs)
        r2 = OptimizationResult(**kwargs)
        assert r1 == r2

    def test_optimization_result_different_values(self):
        r1 = OptimizationResult(
            wind_level=0.0,
            controller="gs-pid",
            params={"Kp": 0.01},
            mean_spin=5.0,
            std_spin=1.0,
            objective=7.0,
            success_rate=1.0,
            n_evaluations=10,
        )
        r2 = OptimizationResult(
            wind_level=0.0,
            controller="gs-pid",
            params={"Kp": 0.02},
            mean_spin=6.0,
            std_spin=1.5,
            objective=9.0,
            success_rate=0.85,
            n_evaluations=20,
        )
        assert r1 != r2


# =====================================================================
# bayesian_optimize.py — ParamLookupTable
# =====================================================================


class TestParamLookupTableAdvanced:
    """Additional ParamLookupTable tests for coverage gaps."""

    def _make_result(self, wind, controller="gs-pid", spin=10.0):
        return OptimizationResult(
            wind_level=wind,
            controller=controller,
            params={"Kp": 0.01 * (1 + wind), "Ki": 0.001, "Kd": 0.02, "q_ref": 500.0},
            mean_spin=spin,
            std_spin=2.0,
            objective=spin + 4.0,
            success_rate=0.9,
            n_evaluations=50,
        )

    def test_get_params_exact_match(self):
        table = ParamLookupTable(controller="gs-pid")
        table.results["2.0"] = self._make_result(2.0)
        params = table.get_params(2.0)
        assert params["Kp"] == pytest.approx(0.01 * 3.0)

    def test_get_params_closest_wind_level(self):
        table = ParamLookupTable(controller="gs-pid")
        table.results["0.0"] = self._make_result(0.0)
        table.results["5.0"] = self._make_result(5.0)
        # 1.5 is closer to 0.0
        params = table.get_params(1.5)
        assert params["Kp"] == pytest.approx(0.01 * 1.0)
        # 4.0 is closer to 5.0
        params = table.get_params(4.0)
        assert params["Kp"] == pytest.approx(0.01 * 6.0)

    def test_get_params_empty_results_returns_baseline(self):
        baseline = {"Kp": 0.005, "Ki": 0.0003, "Kd": 0.016, "q_ref": 500.0}
        table = ParamLookupTable(
            controller="gs-pid",
            baseline_params=baseline,
        )
        params = table.get_params(3.0)
        assert params == baseline

    def test_to_dict_and_from_dict_roundtrip(self):
        table = ParamLookupTable(
            controller="ensemble",
            baseline_params={
                "Kp": 0.005,
                "Ki": 0.0003,
                "Kd": 0.016,
                "q_ref": 500.0,
                "omega_c": 15.0,
                "omega_o": 50.0,
            },
        )
        table.results["0.0"] = self._make_result(0.0, controller="ensemble", spin=5.0)
        table.results["3.0"] = self._make_result(3.0, controller="ensemble", spin=12.0)

        data = table.to_dict()
        json_str = json.dumps(data)
        loaded = ParamLookupTable.from_dict(json.loads(json_str))

        assert loaded.controller == "ensemble"
        assert loaded.baseline_params["omega_c"] == 15.0
        assert "0.0" in loaded.results
        assert "3.0" in loaded.results
        assert loaded.results["0.0"].mean_spin == 5.0
        assert loaded.results["3.0"].mean_spin == 12.0
        # Verify get_params still works after roundtrip
        params = loaded.get_params(0.0)
        assert params["Kp"] == pytest.approx(0.01 * 1.0)

    def test_from_dict_with_empty_results(self):
        data = {
            "controller": "adrc",
            "baseline_params": {"omega_c": 15.0, "omega_o": 50.0},
            "results": {},
        }
        table = ParamLookupTable.from_dict(data)
        assert table.controller == "adrc"
        assert len(table.results) == 0
        params = table.get_params(1.0)
        assert params["omega_c"] == 15.0

    def test_from_dict_missing_baseline(self):
        data = {
            "controller": "gs-pid",
            "results": {},
        }
        table = ParamLookupTable.from_dict(data)
        assert table.baseline_params == {}


# =====================================================================
# bayesian_optimize.py — compute_objective
# =====================================================================


class TestComputeObjectiveAdvanced:
    """Additional tests for compute_objective edge cases."""

    def test_no_success_penalty_at_threshold(self):
        obj = compute_objective(mean_spin=10.0, std_spin=2.0, success_rate=0.8)
        expected = 10.0 + 2.0 * 2.0  # no penalty
        assert obj == pytest.approx(expected)

    def test_no_success_penalty_above_threshold(self):
        obj = compute_objective(mean_spin=10.0, std_spin=2.0, success_rate=1.0)
        expected = 10.0 + 2.0 * 2.0
        assert obj == pytest.approx(expected)

    def test_success_penalty_below_threshold(self):
        obj = compute_objective(mean_spin=10.0, std_spin=2.0, success_rate=0.5)
        # penalty = 50 * max(0, 0.8 - 0.5) = 50 * 0.3 = 15
        expected = 10.0 + 2.0 * 2.0 + 15.0
        assert obj == pytest.approx(expected)

    def test_success_penalty_zero_success(self):
        obj = compute_objective(mean_spin=10.0, std_spin=2.0, success_rate=0.0)
        # penalty = 50 * 0.8 = 40
        expected = 10.0 + 2.0 * 2.0 + 40.0
        assert obj == pytest.approx(expected)

    def test_zero_everything(self):
        obj = compute_objective(mean_spin=0.0, std_spin=0.0, success_rate=1.0)
        assert obj == pytest.approx(0.0)


# =====================================================================
# bayesian_optimize.py — get_baseline_params
# =====================================================================


class TestGetBaselineParamsAll:
    """Test get_baseline_params for every controller type."""

    def test_gs_pid_baseline(self):
        params = get_baseline_params("gs-pid")
        assert "Kp" in params
        assert "Ki" in params
        assert "Kd" in params
        assert "q_ref" in params
        assert all(isinstance(v, float) for v in params.values())

    def test_adrc_baseline(self):
        params = get_baseline_params("adrc")
        assert "omega_c" in params
        assert "omega_o" in params
        assert params["omega_c"] == 15.0
        assert params["omega_o"] == 50.0

    def test_ensemble_baseline(self):
        params = get_baseline_params("ensemble")
        # Should have both GS-PID and ADRC params
        assert "Kp" in params
        assert "Ki" in params
        assert "Kd" in params
        assert "q_ref" in params
        assert "omega_c" in params
        assert "omega_o" in params
        assert len(params) == 6

    def test_unknown_controller_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown controller"):
            get_baseline_params("nonexistent")


# =====================================================================
# bayesian_optimize.py — bounds and EVALUATORS
# =====================================================================


class TestBoundsAndEvaluators:
    """Test bounds constants and EVALUATORS dict."""

    def test_gs_pid_bounds_keys(self):
        assert set(GS_PID_BOUNDS.keys()) == {"Kp", "Ki", "Kd", "q_ref"}

    def test_adrc_bounds_keys(self):
        assert set(ADRC_BOUNDS.keys()) == {"omega_c", "omega_o"}

    def test_ensemble_bounds_is_superset(self):
        for key in GS_PID_BOUNDS:
            assert key in ENSEMBLE_BOUNDS
        for key in ADRC_BOUNDS:
            assert key in ENSEMBLE_BOUNDS
        assert len(ENSEMBLE_BOUNDS) == len(GS_PID_BOUNDS) + len(ADRC_BOUNDS)

    def test_all_bounds_have_valid_ranges(self):
        for bounds_dict in [GS_PID_BOUNDS, ADRC_BOUNDS, ENSEMBLE_BOUNDS]:
            for name, (lo, hi) in bounds_dict.items():
                assert lo < hi, f"Bound {name}: {lo} >= {hi}"
                assert lo > 0, f"Bound {name} lower must be positive"

    def test_evaluators_has_correct_keys(self):
        assert set(EVALUATORS.keys()) == {"gs-pid", "adrc", "ensemble"}

    def test_evaluators_values_are_callable_and_dict(self):
        for name, (fn, bounds) in EVALUATORS.items():
            assert callable(fn), f"Evaluator {name} function not callable"
            assert isinstance(bounds, dict), f"Evaluator {name} bounds not dict"

    def test_evaluators_functions_match(self):
        assert EVALUATORS["gs-pid"][0] is evaluate_gs_pid
        assert EVALUATORS["adrc"][0] is evaluate_adrc
        assert EVALUATORS["ensemble"][0] is evaluate_ensemble

    def test_evaluators_bounds_match(self):
        assert EVALUATORS["gs-pid"][1] is GS_PID_BOUNDS
        assert EVALUATORS["adrc"][1] is ADRC_BOUNDS
        assert EVALUATORS["ensemble"][1] is ENSEMBLE_BOUNDS


# =====================================================================
# bayesian_optimize.py — evaluate_gs_pid (sim test)
# =====================================================================


class TestEvaluateGsPid:
    """Test evaluate_gs_pid with actual simulation (1 episode, 1 wind level)."""

    def test_evaluate_gs_pid_returns_valid_tuple(self, estes_config):
        params = get_baseline_params("gs-pid")
        mean_spin, std_spin, success_rate = evaluate_gs_pid(
            params, estes_config, wind_level=0.0, n_episodes=1
        )
        assert isinstance(mean_spin, float)
        assert isinstance(std_spin, float)
        assert isinstance(success_rate, float)
        assert mean_spin >= 0.0
        assert std_spin >= 0.0
        assert 0.0 <= success_rate <= 1.0


# =====================================================================
# bayesian_optimize.py — evaluate_adrc (sim test, coverage gap)
# =====================================================================


class TestEvaluateAdrc:
    """Test evaluate_adrc with actual simulation (1 episode, 1 wind level)."""

    def test_evaluate_adrc_returns_valid_tuple(self, estes_config):
        params = get_baseline_params("adrc")
        mean_spin, std_spin, success_rate = evaluate_adrc(
            params, estes_config, wind_level=0.0, n_episodes=1
        )
        assert isinstance(mean_spin, float)
        assert isinstance(std_spin, float)
        assert isinstance(success_rate, float)
        assert mean_spin >= 0.0
        assert std_spin >= 0.0
        assert 0.0 <= success_rate <= 1.0

    def test_evaluate_adrc_with_wind(self, estes_config):
        params = get_baseline_params("adrc")
        mean_spin, std_spin, success_rate = evaluate_adrc(
            params, estes_config, wind_level=1.0, n_episodes=1
        )
        assert mean_spin >= 0.0
        assert 0.0 <= success_rate <= 1.0


# =====================================================================
# bayesian_optimize.py — evaluate_ensemble (sim test, coverage gap)
# =====================================================================


class TestEvaluateEnsemble:
    """Test evaluate_ensemble with actual simulation (1 episode, 1 wind level)."""

    def test_evaluate_ensemble_returns_valid_tuple(self, estes_config):
        params = get_baseline_params("ensemble")
        mean_spin, std_spin, success_rate = evaluate_ensemble(
            params, estes_config, wind_level=0.0, n_episodes=1
        )
        assert isinstance(mean_spin, float)
        assert isinstance(std_spin, float)
        assert isinstance(success_rate, float)
        assert mean_spin >= 0.0
        assert std_spin >= 0.0
        assert 0.0 <= success_rate <= 1.0

    def test_evaluate_ensemble_with_wind(self, estes_config):
        params = get_baseline_params("ensemble")
        mean_spin, std_spin, success_rate = evaluate_ensemble(
            params, estes_config, wind_level=1.0, n_episodes=1
        )
        assert mean_spin >= 0.0
        assert 0.0 <= success_rate <= 1.0


# =====================================================================
# optimize_pid.py — bounds constants
# =====================================================================


class TestOptimizePidBounds:
    """Test bounds constants in optimize_pid.py."""

    def test_kp_bounds_exist_and_valid(self):
        assert isinstance(KP_BOUNDS, tuple)
        assert len(KP_BOUNDS) == 2
        assert KP_BOUNDS[0] > 0
        assert KP_BOUNDS[1] > KP_BOUNDS[0]

    def test_ki_bounds_exist_and_valid(self):
        assert isinstance(KI_BOUNDS, tuple)
        assert len(KI_BOUNDS) == 2
        assert KI_BOUNDS[0] > 0
        assert KI_BOUNDS[1] > KI_BOUNDS[0]

    def test_kd_bounds_exist_and_valid(self):
        assert isinstance(KD_BOUNDS, tuple)
        assert len(KD_BOUNDS) == 2
        assert KD_BOUNDS[0] > 0
        assert KD_BOUNDS[1] > KD_BOUNDS[0]

    def test_qref_bounds_exist_and_valid(self):
        assert isinstance(QREF_BOUNDS, tuple)
        assert len(QREF_BOUNDS) == 2
        assert QREF_BOUNDS[0] > 0
        assert QREF_BOUNDS[1] > QREF_BOUNDS[0]

    def test_bounds_values_match_documented(self):
        assert KP_BOUNDS == (0.001, 0.024)
        assert KI_BOUNDS == (0.00012, 0.006)
        assert KD_BOUNDS == (0.0024, 0.036)
        assert QREF_BOUNDS == (100.0, 20000.0)


# =====================================================================
# optimize_pid.py — evaluate_gains (sim tests)
# =====================================================================


class TestEvaluateGainsAdvanced:
    """Test evaluate_gains function with various modes."""

    def test_evaluate_gains_basic(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
        )
        assert "kp" in result
        assert "ki" in result
        assert "kd" in result
        assert "score" in result
        assert "weighted_spin" in result
        assert "success_penalty" in result
        assert "by_wind" in result
        assert 0 in result["by_wind"]
        assert result["score"] >= 0

    def test_evaluate_gains_gain_scheduled(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
            gain_scheduled=True,
            q_ref=500.0,
        )
        assert "q_ref" in result
        assert result["q_ref"] == 500.0
        assert result["score"] >= 0

    def test_evaluate_gains_with_imu(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
            imu=True,
        )
        assert result["score"] >= 0
        assert 0 in result["by_wind"]

    def test_evaluate_gains_gain_scheduled_and_imu(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
            gain_scheduled=True,
            q_ref=500.0,
            imu=True,
        )
        assert "q_ref" in result
        assert result["score"] >= 0

    def test_evaluate_gains_default_wind_weights(self, estes_config):
        """When wind_weights is None, defaults are used."""
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights=None,
        )
        assert result["score"] >= 0

    def test_evaluate_gains_custom_wind_weights(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
        )
        assert result["score"] >= 0

    def test_evaluate_gains_result_structure(self, estes_config):
        result = evaluate_gains(
            kp=0.0203,
            ki=0.0002,
            kd=0.0118,
            config=estes_config,
            wind_levels=[0],
            n_episodes=1,
        )
        wind_data = result["by_wind"][0]
        assert "mean_spin" in wind_data
        assert "std_spin" in wind_data
        assert "success_rate" in wind_data
        assert wind_data["mean_spin"] >= 0
        assert wind_data["std_spin"] >= 0
        assert 0.0 <= wind_data["success_rate"] <= 1.0


# =====================================================================
# optimize_pid.py — objective function
# =====================================================================


class TestObjectiveFunction:
    """Test the objective wrapper function."""

    def test_objective_returns_float(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118],
            estes_config,
            wind_levels=[0],
            n_episodes=1,
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
            n_episodes=1,
            wind_weights={0: 1.0},
            base_seed=42,
            gain_scheduled=True,
        )
        assert isinstance(score, float)
        assert score >= 0

    def test_objective_with_qref_optimization(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118, 5000.0],
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            base_seed=42,
            gain_scheduled=True,
            optimize_qref=True,
        )
        assert isinstance(score, float)
        assert score >= 0

    def test_objective_with_imu(self, estes_config):
        score = objective(
            [0.0203, 0.0002, 0.0118],
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            base_seed=42,
            imu=True,
        )
        assert isinstance(score, float)
        assert score >= 0


# =====================================================================
# optimize_pid.py — run_lhs_phase (mocked)
# =====================================================================


def _mock_evaluate_gains(
    kp,
    ki,
    kd,
    config,
    wind_levels,
    n_episodes,
    wind_weights=None,
    base_seed=42,
    gain_scheduled=False,
    q_ref=500.0,
    imu=False,
):
    """Fast mock for evaluate_gains that returns deterministic results."""
    score = kp * 100 + ki * 1000 + kd * 10
    result = {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "score": float(score),
        "weighted_spin": float(score * 0.9),
        "success_penalty": 0.0,
        "by_wind": {},
    }
    for w in wind_levels:
        result["by_wind"][w] = {
            "mean_spin": float(score),
            "std_spin": 1.0,
            "success_rate": 1.0,
        }
    if gain_scheduled:
        result["q_ref"] = float(q_ref)
    return result


class TestRunLhsPhase:
    """Test run_lhs_phase with mocked evaluate_gains."""

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_lhs_returns_list_of_results(self, mock_eval, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            n_samples=5,
            wind_weights={0: 1.0},
            base_seed=42,
        )
        assert isinstance(results, list)
        assert len(results) == 5
        for r in results:
            assert "kp" in r
            assert "score" in r

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_lhs_gain_scheduled(self, mock_eval, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            n_samples=3,
            wind_weights={0: 1.0},
            gain_scheduled=True,
        )
        assert len(results) == 3

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_lhs_optimize_qref(self, mock_eval, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            n_samples=3,
            wind_weights={0: 1.0},
            gain_scheduled=True,
            optimize_qref=True,
        )
        assert len(results) == 3

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_lhs_with_imu(self, mock_eval, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            n_samples=3,
            wind_weights={0: 1.0},
            imu=True,
        )
        assert len(results) == 3

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_lhs_best_result_has_lowest_score(self, mock_eval, estes_config):
        results = run_lhs_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            n_samples=10,
            wind_weights={0: 1.0},
        )
        best = min(results, key=lambda r: r["score"])
        assert best["score"] <= results[0]["score"]


# =====================================================================
# optimize_pid.py — run_nelder_mead_phase (mocked)
# =====================================================================


class TestRunNelderMeadPhase:
    """Test run_nelder_mead_phase with mocked evaluate_gains."""

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_nelder_mead_returns_result(self, mock_eval, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            start_point=(0.01, 0.001, 0.01),
        )
        assert "kp" in result
        assert "score" in result
        assert result["score"] >= 0

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_nelder_mead_gain_scheduled(self, mock_eval, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            start_point=(0.01, 0.001, 0.01),
            gain_scheduled=True,
        )
        assert "kp" in result

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_nelder_mead_optimize_qref(self, mock_eval, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            start_point=(0.01, 0.001, 0.01, 500.0),
            gain_scheduled=True,
            optimize_qref=True,
        )
        assert "kp" in result

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_nelder_mead_with_imu(self, mock_eval, estes_config):
        result = run_nelder_mead_phase(
            estes_config,
            wind_levels=[0],
            n_episodes=1,
            wind_weights={0: 1.0},
            start_point=(0.01, 0.001, 0.01),
            imu=True,
        )
        assert "kp" in result


# =====================================================================
# optimize_pid.py — main() (mocked)
# =====================================================================


class TestOptimizePidMain:
    """Test optimize_pid.py main() function with mocked simulation."""

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_default_args(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--n-lhs-samples",
                    "3",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert "baseline" in data
            assert "optimized" in data

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_skip_lhs(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--skip-lhs",
                    "--start-kp",
                    "0.01",
                    "--start-ki",
                    "0.001",
                    "--start-kd",
                    "0.01",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_gain_scheduled(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--n-lhs-samples",
                    "3",
                    "--gain-scheduled",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_gain_scheduled_optimize_qref(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--n-lhs-samples",
                    "3",
                    "--gain-scheduled",
                    "--optimize-qref",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_with_imu(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--n-lhs-samples",
                    "3",
                    "--imu",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_skip_lhs_with_qref(self, mock_eval):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "--n-episodes",
                    "1",
                    "--skip-lhs",
                    "--start-kp",
                    "0.01",
                    "--start-ki",
                    "0.001",
                    "--start-kd",
                    "0.01",
                    "--start-qref",
                    "500",
                    "--gain-scheduled",
                    "--optimize-qref",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    @patch("optimization.optimize_pid.evaluate_gains", side_effect=_mock_evaluate_gains)
    def test_main_multiple_wind_levels(self, mock_eval):
        """Cover wind weight branches for w <= 2 and w > 2."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.json")
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--wind-levels",
                    "0",
                    "2",
                    "5",
                    "--n-episodes",
                    "1",
                    "--n-lhs-samples",
                    "3",
                    "--output",
                    output_path,
                ],
            ):
                from optimization.optimize_pid import main

                main()
            assert os.path.exists(output_path)

    def test_main_optimize_qref_without_gain_scheduled_errors(self):
        with pytest.raises(SystemExit):
            with patch(
                "sys.argv",
                [
                    "optimize_pid.py",
                    "--config",
                    "configs/estes_c6_sac_wind.yaml",
                    "--optimize-qref",
                ],
            ):
                from optimization.optimize_pid import main

                main()


# =====================================================================
# bayesian_optimize.py — optimize_one_wind_level (mocked)
# =====================================================================


def _mock_evaluate_gs_pid(params, config, wind_level, n_episodes):
    """Fast mock for evaluate_gs_pid."""
    mean_spin = params.get("Kp", 0.01) * 500 + wind_level
    return float(mean_spin), 1.0, 1.0


def _mock_evaluate_adrc(params, config, wind_level, n_episodes):
    """Fast mock for evaluate_adrc."""
    mean_spin = params.get("omega_c", 15.0) * 0.5 + wind_level
    return float(mean_spin), 1.0, 1.0


def _mock_evaluate_ensemble(params, config, wind_level, n_episodes):
    """Fast mock for evaluate_ensemble."""
    mean_spin = params.get("Kp", 0.01) * 500 + params.get("omega_c", 15.0) * 0.1
    return float(mean_spin), 1.0, 1.0


class TestOptimizeOneWindLevel:
    """Test optimize_one_wind_level with mocked evaluators."""

    @patch(
        "optimization.bayesian_optimize.evaluate_gs_pid",
        side_effect=_mock_evaluate_gs_pid,
    )
    def test_optimize_gs_pid(self, mock_eval, estes_config):
        # Patch EVALUATORS to use the mock
        with patch.dict(
            EVALUATORS,
            {"gs-pid": (_mock_evaluate_gs_pid, GS_PID_BOUNDS)},
        ):
            result = optimize_one_wind_level(
                "gs-pid",
                estes_config,
                wind_level=0.0,
                n_episodes=1,
                max_evaluations=20,
                seed=42,
            )
        assert isinstance(result, OptimizationResult)
        assert result.wind_level == 0.0
        assert result.controller == "gs-pid"
        assert "Kp" in result.params
        assert result.n_evaluations > 0

    @patch(
        "optimization.bayesian_optimize.evaluate_adrc",
        side_effect=_mock_evaluate_adrc,
    )
    def test_optimize_adrc(self, mock_eval, estes_config):
        with patch.dict(
            EVALUATORS,
            {"adrc": (_mock_evaluate_adrc, ADRC_BOUNDS)},
        ):
            result = optimize_one_wind_level(
                "adrc",
                estes_config,
                wind_level=1.0,
                n_episodes=1,
                max_evaluations=20,
                seed=42,
            )
        assert isinstance(result, OptimizationResult)
        assert result.controller == "adrc"
        assert "omega_c" in result.params

    @patch(
        "optimization.bayesian_optimize.evaluate_ensemble",
        side_effect=_mock_evaluate_ensemble,
    )
    def test_optimize_ensemble(self, mock_eval, estes_config):
        with patch.dict(
            EVALUATORS,
            {"ensemble": (_mock_evaluate_ensemble, ENSEMBLE_BOUNDS)},
        ):
            result = optimize_one_wind_level(
                "ensemble",
                estes_config,
                wind_level=0.0,
                n_episodes=1,
                max_evaluations=20,
                seed=42,
            )
        assert isinstance(result, OptimizationResult)
        assert result.controller == "ensemble"


# =====================================================================
# bayesian_optimize.py — main() (mocked)
# =====================================================================


class TestBayesianOptimizeMain:
    """Test bayesian_optimize.py main() function with mocked evaluators."""

    def test_main_gs_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "bo_results.json")
            with (
                patch.dict(
                    EVALUATORS,
                    {"gs-pid": (_mock_evaluate_gs_pid, GS_PID_BOUNDS)},
                ),
                patch(
                    "sys.argv",
                    [
                        "bayesian_optimize.py",
                        "--config",
                        "configs/estes_c6_sac_wind.yaml",
                        "--controller",
                        "gs-pid",
                        "--wind-level",
                        "0",
                        "--n-episodes",
                        "1",
                        "--n-trials",
                        "20",
                        "--output",
                        output_path,
                    ],
                ),
            ):
                from optimization.bayesian_optimize import main

                main()
            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert data["controller"] == "gs-pid"
            assert "0.0" in data["results"] or "0" in data["results"]

    def test_main_adrc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "bo_results.json")
            with (
                patch.dict(
                    EVALUATORS,
                    {"adrc": (_mock_evaluate_adrc, ADRC_BOUNDS)},
                ),
                patch(
                    "sys.argv",
                    [
                        "bayesian_optimize.py",
                        "--config",
                        "configs/estes_c6_sac_wind.yaml",
                        "--controller",
                        "adrc",
                        "--wind-level",
                        "0",
                        "--n-episodes",
                        "1",
                        "--n-trials",
                        "20",
                        "--output",
                        output_path,
                    ],
                ),
            ):
                from optimization.bayesian_optimize import main

                main()
            assert os.path.exists(output_path)

    def test_main_multiple_wind_levels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "bo_results.json")
            with (
                patch.dict(
                    EVALUATORS,
                    {"gs-pid": (_mock_evaluate_gs_pid, GS_PID_BOUNDS)},
                ),
                patch(
                    "sys.argv",
                    [
                        "bayesian_optimize.py",
                        "--config",
                        "configs/estes_c6_sac_wind.yaml",
                        "--controller",
                        "gs-pid",
                        "--wind-level",
                        "0",
                        "2",
                        "--n-episodes",
                        "1",
                        "--n-trials",
                        "20",
                        "--output",
                        output_path,
                    ],
                ),
            ):
                from optimization.bayesian_optimize import main

                main()
            assert os.path.exists(output_path)
            with open(output_path) as f:
                data = json.load(f)
            assert len(data["results"]) == 2
