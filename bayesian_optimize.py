#!/usr/bin/env python3
"""
Per-Condition Controller Parameter Optimization

Optimizes controller parameters SEPARATELY for each wind level using
scipy's differential_evolution (global optimizer), then exports a
parameter lookup table. At runtime, the wind level proxy (e.g., ADRC z3
magnitude or simply the configured wind level) selects the best
parameter set.

Current PID gains were found with optimize_pid.py, which finds ONE
compromise set of gains across all wind levels. Wind-specific tuning
can find, e.g., higher Ki for 0 m/s (no sinusoidal disturbance) and
lower Ki for 2 m/s (sinusoidal lag problem).

Supported controllers:
    - GS-PID: optimizes Kp, Ki, Kd, q_ref
    - ADRC: optimizes omega_c, omega_o (b0 is physics-derived)
    - Ensemble: optimizes both GS-PID and ADRC parameters

Algorithm:
    Differential evolution (scipy.optimize.differential_evolution) with
    Latin Hypercube initialization. DE is gradient-free, handles bounds
    naturally, and avoids local minima better than Nelder-Mead.

Objective:
    minimize  mean_spin + 2 * std_spin
    subject to  success_rate >= 0.8 (soft penalty)

Usage:
    # Optimize GS-PID for wind=1 m/s
    uv run python bayesian_optimize.py --config configs/estes_c6_sac_wind.yaml \\
        --controller gs-pid --wind-level 1 --n-trials 100

    # Optimize for all wind levels
    uv run python bayesian_optimize.py --config configs/estes_c6_sac_wind.yaml \\
        --controller gs-pid --wind-level 0 1 2 3 5 --n-trials 80

    # Evaluate optimized parameters
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \\
        --optimized-params optimization_results/bo_params.json --imu \\
        --wind-levels 0 1 2 3 5 --n-episodes 50
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from compare_controllers import (
    create_env_with_imu,
    run_controller_episode,
    EpisodeMetrics,
)
from pid_controller import PIDConfig, GainScheduledPIDController
from adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config
from ensemble_controller import EnsembleController, EnsembleConfig
from rocket_config import load_config


@dataclass
class OptimizationResult:
    """Result of optimizing one controller at one wind level."""

    wind_level: float
    controller: str
    params: Dict[str, float]
    mean_spin: float
    std_spin: float
    objective: float
    success_rate: float
    n_evaluations: int


@dataclass
class ParamLookupTable:
    """Lookup table mapping wind levels to optimal parameters."""

    controller: str
    results: Dict[str, OptimizationResult] = field(default_factory=dict)
    baseline_params: Dict[str, float] = field(default_factory=dict)

    def get_params(self, wind_level: float) -> Dict[str, float]:
        """Get optimal parameters for a wind level.

        Returns exact match if available, else the closest wind level.
        """
        key = str(wind_level)
        if key in self.results:
            return self.results[key].params

        # Find closest wind level
        available = [float(k) for k in self.results.keys()]
        if not available:
            return self.baseline_params
        closest = min(available, key=lambda w: abs(w - wind_level))
        return self.results[str(closest)].params

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "controller": self.controller,
            "baseline_params": self.baseline_params,
            "results": {k: asdict(v) for k, v in self.results.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParamLookupTable":
        """Deserialize from JSON dict."""
        table = cls(
            controller=data["controller"],
            baseline_params=data.get("baseline_params", {}),
        )
        for k, v in data.get("results", {}).items():
            table.results[k] = OptimizationResult(**v)
        return table


# --- Parameter bounds for each controller ---

GS_PID_BOUNDS = {
    "Kp": (0.001, 0.024),
    "Ki": (0.00006, 0.006),
    "Kd": (0.0024, 0.048),
    "q_ref": (200.0, 1000.0),
}

ADRC_BOUNDS = {
    "omega_c": (5.0, 40.0),
    "omega_o": (20.0, 150.0),
}

ENSEMBLE_BOUNDS = {
    **GS_PID_BOUNDS,
    **ADRC_BOUNDS,
}


def evaluate_gs_pid(
    params: Dict[str, float],
    config,
    wind_level: float,
    n_episodes: int,
) -> Tuple[float, float, float]:
    """Evaluate GS-PID with given parameters at one wind level.

    Returns: (mean_spin, std_spin, success_rate)
    """
    pid_config = PIDConfig(
        Cprop=params["Kp"],
        Cint=params["Ki"],
        Cderiv=params["Kd"],
        q_ref=params["q_ref"],
    )
    ctrl = GainScheduledPIDController(pid_config, use_observations=True)
    dt = getattr(config.environment, "dt", 0.01)

    env = create_env_with_imu(config, wind_level)
    spins = []
    for _ in range(n_episodes):
        metrics = run_controller_episode(env, ctrl, dt)
        spins.append(metrics.mean_spin_rate)
    env.close()

    mean_spin = np.mean(spins)
    std_spin = np.std(spins)
    success_rate = np.mean([s < 30.0 for s in spins])
    return float(mean_spin), float(std_spin), float(success_rate)


def evaluate_adrc(
    params: Dict[str, float],
    config,
    wind_level: float,
    n_episodes: int,
) -> Tuple[float, float, float]:
    """Evaluate ADRC with given parameters at one wind level."""
    airframe = config.physics.resolve_airframe()
    adrc_config = estimate_adrc_config(
        airframe,
        config.physics,
        omega_c=params["omega_c"],
        omega_o=params["omega_o"],
    )
    adrc_config.use_observations = True
    ctrl = ADRCController(adrc_config)
    dt = getattr(config.environment, "dt", 0.01)

    env = create_env_with_imu(config, wind_level)
    spins = []
    for _ in range(n_episodes):
        metrics = run_controller_episode(env, ctrl, dt)
        spins.append(metrics.mean_spin_rate)
    env.close()

    mean_spin = np.mean(spins)
    std_spin = np.std(spins)
    success_rate = np.mean([s < 30.0 for s in spins])
    return float(mean_spin), float(std_spin), float(success_rate)


def evaluate_ensemble(
    params: Dict[str, float],
    config,
    wind_level: float,
    n_episodes: int,
) -> Tuple[float, float, float]:
    """Evaluate ensemble (GS-PID + ADRC) with given parameters."""
    pid_config = PIDConfig(
        Cprop=params["Kp"],
        Cint=params["Ki"],
        Cderiv=params["Kd"],
        q_ref=params["q_ref"],
    )
    gs_pid = GainScheduledPIDController(pid_config, use_observations=True)

    airframe = config.physics.resolve_airframe()
    adrc_config = estimate_adrc_config(
        airframe,
        config.physics,
        omega_c=params["omega_c"],
        omega_o=params["omega_o"],
    )
    adrc_config.use_observations = True
    adrc = ADRCController(adrc_config)

    ctrl = EnsembleController(
        controllers=[gs_pid, adrc],
        names=["GS-PID", "ADRC"],
        config=EnsembleConfig(),
    )
    dt = getattr(config.environment, "dt", 0.01)

    env = create_env_with_imu(config, wind_level)
    spins = []
    for _ in range(n_episodes):
        metrics = run_controller_episode(env, ctrl, dt)
        spins.append(metrics.mean_spin_rate)
    env.close()

    mean_spin = np.mean(spins)
    std_spin = np.std(spins)
    success_rate = np.mean([s < 30.0 for s in spins])
    return float(mean_spin), float(std_spin), float(success_rate)


EVALUATORS = {
    "gs-pid": (evaluate_gs_pid, GS_PID_BOUNDS),
    "adrc": (evaluate_adrc, ADRC_BOUNDS),
    "ensemble": (evaluate_ensemble, ENSEMBLE_BOUNDS),
}


def compute_objective(mean_spin: float, std_spin: float, success_rate: float) -> float:
    """Compute the optimization objective.

    Objective: mean_spin + 2 * std_spin + success_penalty
    Lower is better.
    """
    # Soft penalty below 80% success rate
    success_penalty = 50.0 * max(0.0, 0.8 - success_rate)
    return mean_spin + 2.0 * std_spin + success_penalty


def optimize_one_wind_level(
    controller_name: str,
    config,
    wind_level: float,
    n_episodes: int,
    max_evaluations: int,
    seed: int = 42,
) -> OptimizationResult:
    """Optimize controller parameters for one wind level.

    Uses differential evolution for global optimization.
    """
    evaluate_fn, bounds_dict = EVALUATORS[controller_name]
    param_names = list(bounds_dict.keys())
    bounds_list = [bounds_dict[name] for name in param_names]

    eval_count = [0]
    best_obj = [float("inf")]

    def objective_wrapper(x):
        params = dict(zip(param_names, x))
        mean_spin, std_spin, success_rate = evaluate_fn(
            params,
            config,
            wind_level,
            n_episodes,
        )
        obj = compute_objective(mean_spin, std_spin, success_rate)
        eval_count[0] += 1

        if obj < best_obj[0]:
            best_obj[0] = obj
            param_str = ", ".join(f"{k}={v:.4f}" for k, v in params.items())
            print(
                f"    [eval {eval_count[0]:3d}] obj={obj:.2f} "
                f"(spin={mean_spin:.1f}+/-{std_spin:.1f}, "
                f"success={success_rate*100:.0f}%) {param_str}"
            )

        return obj

    # Determine DE parameters based on budget
    n_params = len(param_names)
    # popsize * n_params = total population; maxiter controls generations
    popsize = max(5, min(15, max_evaluations // (n_params * 5)))
    maxiter = max(3, max_evaluations // (popsize * n_params) - 1)

    result = differential_evolution(
        objective_wrapper,
        bounds=bounds_list,
        seed=seed,
        maxiter=maxiter,
        popsize=popsize,
        tol=0.5,
        init="latinhypercube",
        mutation=(0.5, 1.5),
        recombination=0.7,
    )

    # Final evaluation with the optimal parameters
    optimal_params = dict(zip(param_names, result.x))
    mean_spin, std_spin, success_rate = evaluate_fn(
        optimal_params,
        config,
        wind_level,
        n_episodes,
    )

    return OptimizationResult(
        wind_level=wind_level,
        controller=controller_name,
        params=optimal_params,
        mean_spin=mean_spin,
        std_spin=std_spin,
        objective=compute_objective(mean_spin, std_spin, success_rate),
        success_rate=success_rate,
        n_evaluations=eval_count[0],
    )


def get_baseline_params(controller_name: str) -> Dict[str, float]:
    """Get the baseline (default) parameters for a controller."""
    if controller_name == "gs-pid":
        return {"Kp": 0.005208, "Ki": 0.000324, "Kd": 0.016524, "q_ref": 500.0}
    elif controller_name == "adrc":
        return {"omega_c": 15.0, "omega_o": 50.0}
    elif controller_name == "ensemble":
        return {
            "Kp": 0.005208,
            "Ki": 0.000324,
            "Kd": 0.016524,
            "q_ref": 500.0,
            "omega_c": 15.0,
            "omega_o": 50.0,
        }
    else:
        raise ValueError(f"Unknown controller: {controller_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-condition controller parameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--controller",
        type=str,
        default="gs-pid",
        choices=["gs-pid", "adrc", "ensemble"],
        help="Controller to optimize",
    )
    parser.add_argument(
        "--wind-level",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3, 5],
        help="Wind levels to optimize for (separately)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=15,
        help="Episodes per evaluation (default: 15)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=80,
        help="Maximum number of objective evaluations per wind level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results/bo_params.json",
        help="Output JSON path for parameter lookup table",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    print(f"Controller: {args.controller}")
    print(f"Wind levels: {args.wind_level}")
    print(f"Episodes/eval: {args.n_episodes}")
    print(f"Max trials/wind: {args.n_trials}")

    baseline_params = get_baseline_params(args.controller)
    print(f"\nBaseline parameters: {baseline_params}")

    # Evaluate baseline at each wind level
    evaluate_fn, _ = EVALUATORS[args.controller]
    print("\nBaseline performance:")
    for wind in args.wind_level:
        mean_spin, std_spin, success_rate = evaluate_fn(
            baseline_params,
            config,
            wind,
            args.n_episodes,
        )
        obj = compute_objective(mean_spin, std_spin, success_rate)
        print(
            f"  Wind {wind:.0f} m/s: spin={mean_spin:.1f}+/-{std_spin:.1f}, "
            f"success={success_rate*100:.0f}%, obj={obj:.2f}"
        )

    # Optimize each wind level
    lookup = ParamLookupTable(
        controller=args.controller,
        baseline_params=baseline_params,
    )

    for wind in args.wind_level:
        print(f"\n{'='*60}")
        print(f"Optimizing {args.controller} for wind={wind:.0f} m/s")
        print(f"{'='*60}")

        t0 = time.time()
        result = optimize_one_wind_level(
            args.controller,
            config,
            wind,
            n_episodes=args.n_episodes,
            max_evaluations=args.n_trials,
            seed=args.seed + int(wind * 100),
        )
        elapsed = time.time() - t0

        lookup.results[str(wind)] = result

        print(
            f"\n  Optimal for wind={wind:.0f} m/s ({elapsed:.0f}s, {result.n_evaluations} evals):"
        )
        print(f"    Params: {result.params}")
        print(f"    Spin: {result.mean_spin:.1f}+/-{result.std_spin:.1f}")
        print(f"    Success: {result.success_rate*100:.0f}%")
        print(f"    Objective: {result.objective:.2f}")

    # Summary table
    print(f"\n{'='*60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(
        f"{'Wind':>6} | {'Baseline obj':>12} | {'Optimized obj':>13} | {'Improvement':>11}"
    )
    print("-" * 52)

    for wind in args.wind_level:
        mean_b, std_b, sr_b = evaluate_fn(
            baseline_params,
            config,
            wind,
            args.n_episodes,
        )
        obj_b = compute_objective(mean_b, std_b, sr_b)
        result = lookup.results[str(wind)]
        improvement = obj_b - result.objective
        print(
            f"{wind:>6.0f} | {obj_b:>12.2f} | {result.objective:>13.2f} | "
            f"{improvement:>+11.2f}"
        )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(lookup.to_dict(), f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print evaluation command
    print(f"\nTo evaluate optimized parameters:")
    print(
        f"  uv run python compare_controllers.py --config {args.config} "
        f"--optimized-params {args.output} --imu "
        f"--wind-levels {' '.join(str(int(w)) for w in args.wind_level)} "
        f"--n-episodes 50"
    )


if __name__ == "__main__":
    main()
