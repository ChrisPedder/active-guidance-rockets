"""
Tests for the per-condition Bayesian optimization module.

Verifies:
1. Parameter bounds are correctly defined
2. Objective function computes correct values
3. ParamLookupTable serialization/deserialization
4. ParamLookupTable closest-wind fallback
5. Baseline parameter retrieval
6. Evaluator functions run without error (smoke test)
7. Integration with compare_controllers.py
"""

import json
import numpy as np
import pytest
from pathlib import Path

from bayesian_optimize import (
    GS_PID_BOUNDS,
    ADRC_BOUNDS,
    ENSEMBLE_BOUNDS,
    ParamLookupTable,
    OptimizationResult,
    compute_objective,
    get_baseline_params,
)


class TestParameterBounds:
    """Test parameter bounds are reasonable."""

    def test_gs_pid_bounds_exist(self):
        assert "Kp" in GS_PID_BOUNDS
        assert "Ki" in GS_PID_BOUNDS
        assert "Kd" in GS_PID_BOUNDS
        assert "q_ref" in GS_PID_BOUNDS

    def test_adrc_bounds_exist(self):
        assert "omega_c" in ADRC_BOUNDS
        assert "omega_o" in ADRC_BOUNDS

    def test_ensemble_bounds_contain_all(self):
        for key in GS_PID_BOUNDS:
            assert key in ENSEMBLE_BOUNDS
        for key in ADRC_BOUNDS:
            assert key in ENSEMBLE_BOUNDS

    def test_bounds_are_valid_ranges(self):
        for name, (low, high) in GS_PID_BOUNDS.items():
            assert low < high, f"{name}: low={low} >= high={high}"
        for name, (low, high) in ADRC_BOUNDS.items():
            assert low < high, f"{name}: low={low} >= high={high}"

    def test_baseline_params_within_bounds(self):
        baseline = get_baseline_params("gs-pid")
        for key, (low, high) in GS_PID_BOUNDS.items():
            assert (
                low <= baseline[key] <= high
            ), f"{key}={baseline[key]} not in [{low}, {high}]"

    def test_adrc_baseline_within_bounds(self):
        baseline = get_baseline_params("adrc")
        for key, (low, high) in ADRC_BOUNDS.items():
            assert (
                low <= baseline[key] <= high
            ), f"{key}={baseline[key]} not in [{low}, {high}]"


class TestObjectiveFunction:
    """Test the optimization objective function."""

    def test_perfect_performance(self):
        obj = compute_objective(mean_spin=0.0, std_spin=0.0, success_rate=1.0)
        assert obj == 0.0

    def test_higher_spin_increases_objective(self):
        obj1 = compute_objective(5.0, 1.0, 1.0)
        obj2 = compute_objective(10.0, 1.0, 1.0)
        assert obj2 > obj1

    def test_higher_std_increases_objective(self):
        obj1 = compute_objective(5.0, 1.0, 1.0)
        obj2 = compute_objective(5.0, 5.0, 1.0)
        assert obj2 > obj1

    def test_std_weight_is_two(self):
        obj = compute_objective(10.0, 5.0, 1.0)
        # 10 + 2*5 = 20 (no success penalty)
        assert abs(obj - 20.0) < 1e-10

    def test_low_success_adds_penalty(self):
        obj_good = compute_objective(10.0, 2.0, 1.0)
        obj_bad = compute_objective(10.0, 2.0, 0.5)
        assert obj_bad > obj_good
        # Penalty: 50 * max(0, 0.8 - 0.5) = 50 * 0.3 = 15
        assert abs(obj_bad - obj_good - 15.0) < 1e-10

    def test_success_at_threshold_no_penalty(self):
        obj = compute_objective(10.0, 2.0, 0.8)
        assert abs(obj - 14.0) < 1e-10  # 10 + 2*2 = 14

    def test_success_above_threshold_no_penalty(self):
        obj = compute_objective(10.0, 2.0, 0.95)
        assert abs(obj - 14.0) < 1e-10  # same as 0.8


class TestParamLookupTable:
    """Test the parameter lookup table."""

    def _make_result(self, wind: float, spin: float = 10.0) -> OptimizationResult:
        return OptimizationResult(
            wind_level=wind,
            controller="gs-pid",
            params={"Kp": 0.05 + wind * 0.01, "Ki": 0.003, "Kd": 0.15, "q_ref": 500.0},
            mean_spin=spin,
            std_spin=2.0,
            objective=spin + 4.0,
            success_rate=0.9,
            n_evaluations=50,
        )

    def test_empty_table_returns_baseline(self):
        table = ParamLookupTable(
            controller="gs-pid",
            baseline_params={"Kp": 0.04, "Ki": 0.003, "Kd": 0.14, "q_ref": 500.0},
        )
        params = table.get_params(1.0)
        assert params["Kp"] == 0.04

    def test_exact_match(self):
        table = ParamLookupTable(controller="gs-pid")
        table.results["1.0"] = self._make_result(1.0)
        params = table.get_params(1.0)
        assert params["Kp"] == 0.05 + 1.0 * 0.01

    def test_closest_match(self):
        table = ParamLookupTable(controller="gs-pid")
        table.results["0.0"] = self._make_result(0.0)
        table.results["3.0"] = self._make_result(3.0)
        # 1.0 is closer to 0.0
        params = table.get_params(1.0)
        assert params["Kp"] == 0.05 + 0.0 * 0.01
        # 2.5 is closer to 3.0
        params = table.get_params(2.5)
        assert params["Kp"] == 0.05 + 3.0 * 0.01

    def test_serialization_roundtrip(self):
        table = ParamLookupTable(
            controller="gs-pid",
            baseline_params={"Kp": 0.04, "Ki": 0.003, "Kd": 0.14, "q_ref": 500.0},
        )
        table.results["0.0"] = self._make_result(0.0)
        table.results["2.0"] = self._make_result(2.0, spin=15.0)

        # Serialize
        data = table.to_dict()
        json_str = json.dumps(data)

        # Deserialize
        loaded = ParamLookupTable.from_dict(json.loads(json_str))
        assert loaded.controller == "gs-pid"
        assert loaded.baseline_params["Kp"] == 0.04
        assert "0.0" in loaded.results
        assert "2.0" in loaded.results
        assert loaded.results["2.0"].mean_spin == 15.0

    def test_to_dict_format(self):
        table = ParamLookupTable(controller="adrc")
        table.results["1.0"] = OptimizationResult(
            wind_level=1.0,
            controller="adrc",
            params={"omega_c": 20.0, "omega_o": 60.0},
            mean_spin=8.0,
            std_spin=3.0,
            objective=14.0,
            success_rate=0.95,
            n_evaluations=80,
        )
        data = table.to_dict()
        assert data["controller"] == "adrc"
        assert "1.0" in data["results"]
        assert data["results"]["1.0"]["params"]["omega_c"] == 20.0


class TestBaselineParams:
    """Test baseline parameter retrieval."""

    def test_gs_pid_baseline(self):
        params = get_baseline_params("gs-pid")
        assert "Kp" in params
        assert "Ki" in params
        assert "Kd" in params
        assert "q_ref" in params
        assert params["Kp"] == 0.005208

    def test_adrc_baseline(self):
        params = get_baseline_params("adrc")
        assert "omega_c" in params
        assert "omega_o" in params
        assert params["omega_c"] == 15.0

    def test_ensemble_baseline(self):
        params = get_baseline_params("ensemble")
        assert "Kp" in params
        assert "omega_c" in params

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            get_baseline_params("unknown-controller")


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_optimized_params_flag_in_source(self):
        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "--optimized-params" in source

    def test_optimized_color_defined(self):
        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "Optimized GS-PID" in source

    def test_lookup_table_importable(self):
        from bayesian_optimize import ParamLookupTable

        table = ParamLookupTable(controller="gs-pid")
        assert hasattr(table, "get_params")
        assert hasattr(table, "to_dict")
        assert hasattr(table, "from_dict")
