#!/usr/bin/env python3
"""
PID Gain Optimization for Rocket Roll Stabilization

Systematic search over (Kp, Ki, Kd) to find optimal PID gains.
Uses Latin Hypercube Sampling for initial exploration, then
Nelder-Mead refinement from the best point.

Usage:
    uv run python optimize_pid.py --config configs/estes_c6_residual_sac_wind.yaml \
        --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80

    # Quick test
    uv run python optimize_pid.py --config configs/estes_c6_residual_sac_wind.yaml \
        --wind-levels 0 2 5 --n-episodes 5 --n-lhs-samples 10
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

from compare_controllers import (
    create_env,
    create_env_with_imu,
    run_pid_episode,
    run_controller_episode,
    EpisodeMetrics,
)
from pid_controller import PIDConfig, PIDController, GainScheduledPIDController
from rocket_config import load_config


# Search space bounds
KP_BOUNDS = (0.001, 0.024)
KI_BOUNDS = (0.00012, 0.006)
KD_BOUNDS = (0.0024, 0.036)
QREF_BOUNDS = (100.0, 20000.0)


def evaluate_gains(
    kp: float,
    ki: float,
    kd: float,
    config,
    wind_levels: list,
    n_episodes: int,
    wind_weights: dict = None,
    base_seed: int = 42,
    gain_scheduled: bool = False,
    q_ref: float = 500.0,
    imu: bool = False,
) -> dict:
    """Evaluate a set of PID gains across wind levels.

    Args:
        gain_scheduled: If True, evaluate using GainScheduledPIDController
        q_ref: Reference dynamic pressure for gain scheduling
        imu: If True, wrap env with IMU noise and use observation-based control

    Returns dict with per-wind and aggregate metrics.
    """
    if wind_weights is None:
        wind_weights = {0: 0.2, 2: 0.35, 5: 0.45}

    pid_config = PIDConfig(Cprop=kp, Cint=ki, Cderiv=kd, q_ref=q_ref)
    dt = getattr(config.environment, "dt", 0.01)

    results_by_wind = {}
    weighted_spin = 0.0
    total_success_penalty = 0.0

    for wind_speed in wind_levels:
        if imu:
            env = create_env_with_imu(config, wind_speed)
        else:
            env = create_env(config, wind_speed)
        episodes = []

        for ep in range(n_episodes):
            # Deterministic seeding per candidate + episode + wind
            seed = base_seed + int(wind_speed * 1000) + ep
            env.reset(seed=seed)
            if gain_scheduled:
                controller = GainScheduledPIDController(
                    pid_config, use_observations=imu
                )
                metrics = run_controller_episode(env, controller, dt)
            else:
                controller = PIDController(pid_config, use_observations=imu)
                metrics = run_controller_episode(env, controller, dt)
            episodes.append(metrics)

        env.close()

        mean_spin = np.mean([e.mean_spin_rate for e in episodes])
        success_rate = np.mean([e.mean_spin_rate < 30.0 for e in episodes])

        results_by_wind[wind_speed] = {
            "mean_spin": float(mean_spin),
            "std_spin": float(np.std([e.mean_spin_rate for e in episodes])),
            "success_rate": float(success_rate),
        }

        weight = wind_weights.get(wind_speed, 1.0 / len(wind_levels))
        weighted_spin += weight * mean_spin

        # Soft penalty below 80% success
        total_success_penalty += 50.0 * max(0.0, 0.8 - success_rate)

    score = weighted_spin + total_success_penalty

    result = {
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "score": float(score),
        "weighted_spin": float(weighted_spin),
        "success_penalty": float(total_success_penalty),
        "by_wind": results_by_wind,
    }
    if gain_scheduled:
        result["q_ref"] = float(q_ref)
    return result


def objective(
    params,
    config,
    wind_levels,
    n_episodes,
    wind_weights,
    base_seed,
    gain_scheduled=False,
    optimize_qref=False,
    imu=False,
):
    """Objective function for scipy.optimize (minimise score)."""
    if optimize_qref:
        kp, ki, kd, q_ref = params
    else:
        kp, ki, kd = params
        q_ref = 500.0
    result = evaluate_gains(
        kp,
        ki,
        kd,
        config,
        wind_levels,
        n_episodes,
        wind_weights,
        base_seed,
        gain_scheduled=gain_scheduled,
        q_ref=q_ref,
        imu=imu,
    )
    return result["score"]


def run_lhs_phase(
    config,
    wind_levels: list,
    n_episodes: int,
    n_samples: int,
    wind_weights: dict,
    base_seed: int = 42,
    gain_scheduled: bool = False,
    optimize_qref: bool = False,
    imu: bool = False,
) -> list:
    """Phase 1: Latin Hypercube Sampling exploration."""
    print(f"\n{'='*60}")
    print(f"PHASE 1: Latin Hypercube Sampling ({n_samples} samples)")
    print(f"{'='*60}")
    ctrl_type = "GS-PID" if gain_scheduled else "PID"
    if imu:
        ctrl_type += " (IMU)"
    if optimize_qref:
        print(
            f"Search space ({ctrl_type}): Kp={KP_BOUNDS}, Ki={KI_BOUNDS}, Kd={KD_BOUNDS}, q_ref={QREF_BOUNDS}"
        )
    else:
        print(
            f"Search space ({ctrl_type}): Kp={KP_BOUNDS}, Ki={KI_BOUNDS}, Kd={KD_BOUNDS}"
        )
    print(f"Wind levels: {wind_levels}, Episodes/wind: {n_episodes}")

    ndim = 4 if optimize_qref else 3
    sampler = LatinHypercube(d=ndim, seed=base_seed)
    samples = sampler.random(n=n_samples)

    # Scale to bounds
    kp_vals = KP_BOUNDS[0] + samples[:, 0] * (KP_BOUNDS[1] - KP_BOUNDS[0])
    ki_vals = KI_BOUNDS[0] + samples[:, 1] * (KI_BOUNDS[1] - KI_BOUNDS[0])
    kd_vals = KD_BOUNDS[0] + samples[:, 2] * (KD_BOUNDS[1] - KD_BOUNDS[0])
    if optimize_qref:
        qref_vals = QREF_BOUNDS[0] + samples[:, 3] * (QREF_BOUNDS[1] - QREF_BOUNDS[0])
    else:
        qref_vals = np.full(n_samples, 500.0)

    results = []
    best_score = float("inf")
    best_idx = -1

    for i in range(n_samples):
        t0 = time.time()
        result = evaluate_gains(
            kp_vals[i],
            ki_vals[i],
            kd_vals[i],
            config,
            wind_levels,
            n_episodes,
            wind_weights,
            base_seed,
            gain_scheduled=gain_scheduled,
            q_ref=qref_vals[i],
            imu=imu,
        )
        elapsed = time.time() - t0
        results.append(result)

        marker = ""
        if result["score"] < best_score:
            best_score = result["score"]
            best_idx = i
            marker = " <-- NEW BEST"

        qref_str = f" q_ref={qref_vals[i]:.0f}" if optimize_qref else ""
        print(
            f"  [{i+1:3d}/{n_samples}] "
            f"Kp={kp_vals[i]:.4f} Ki={ki_vals[i]:.4f} Kd={kd_vals[i]:.4f}{qref_str} "
            f"score={result['score']:.2f} "
            f"(spin={result['weighted_spin']:.2f}, pen={result['success_penalty']:.1f}) "
            f"[{elapsed:.1f}s]{marker}"
        )

    print(f"\nBest LHS result (#{best_idx+1}):")
    best = results[best_idx]
    qref_str = f", q_ref={best.get('q_ref', 500.0):.0f}" if optimize_qref else ""
    print(f"  Kp={best['kp']:.4f}, Ki={best['ki']:.4f}, Kd={best['kd']:.4f}{qref_str}")
    print(f"  Score={best['score']:.2f}")
    for w, data in best["by_wind"].items():
        print(
            f"  Wind {w} m/s: spin={data['mean_spin']:.1f}+/-{data['std_spin']:.1f}, "
            f"success={data['success_rate']*100:.0f}%"
        )

    return results


def run_nelder_mead_phase(
    config,
    wind_levels: list,
    n_episodes: int,
    wind_weights: dict,
    start_point: tuple,
    base_seed: int = 42,
    gain_scheduled: bool = False,
    optimize_qref: bool = False,
    imu: bool = False,
) -> dict:
    """Phase 2: Nelder-Mead refinement from best LHS point."""
    print(f"\n{'='*60}")
    print("PHASE 2: Nelder-Mead Refinement")
    print(f"{'='*60}")
    ctrl_type = "GS-PID" if gain_scheduled else "PID"
    if imu:
        ctrl_type += " (IMU)"
    if optimize_qref:
        print(
            f"Starting from ({ctrl_type}): Kp={start_point[0]:.4f}, Ki={start_point[1]:.4f}, Kd={start_point[2]:.4f}, q_ref={start_point[3]:.0f}"
        )
    else:
        print(
            f"Starting from ({ctrl_type}): Kp={start_point[0]:.4f}, Ki={start_point[1]:.4f}, Kd={start_point[2]:.4f}"
        )
    print(f"Episodes/wind: {n_episodes}")

    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1
        if optimize_qref:
            q_ref = xk[3]
            qref_str = f" q_ref={q_ref:.0f}"
        else:
            q_ref = 500.0
            qref_str = ""
        result = evaluate_gains(
            xk[0],
            xk[1],
            xk[2],
            config,
            wind_levels,
            n_episodes,
            wind_weights,
            base_seed,
            gain_scheduled=gain_scheduled,
            q_ref=q_ref,
            imu=imu,
        )
        print(
            f"  NM iter {iteration_count[0]:3d}: "
            f"Kp={xk[0]:.4f} Ki={xk[1]:.4f} Kd={xk[2]:.4f}{qref_str} "
            f"score={result['score']:.2f}"
        )

    opt_result = minimize(
        objective,
        x0=np.array(start_point),
        args=(
            config,
            wind_levels,
            n_episodes,
            wind_weights,
            base_seed,
            gain_scheduled,
            optimize_qref,
            imu,
        ),
        method="Nelder-Mead",
        callback=callback,
        options={
            "maxiter": 200,
            "xatol": 0.001,
            "fatol": 0.1,
            "adaptive": True,
        },
    )

    # Clip to bounds
    kp = np.clip(opt_result.x[0], *KP_BOUNDS)
    ki = np.clip(opt_result.x[1], *KI_BOUNDS)
    kd = np.clip(opt_result.x[2], *KD_BOUNDS)
    if optimize_qref:
        q_ref = np.clip(opt_result.x[3], *QREF_BOUNDS)
    else:
        q_ref = 500.0

    # Final evaluation with more episodes
    final_result = evaluate_gains(
        kp,
        ki,
        kd,
        config,
        wind_levels,
        n_episodes,
        wind_weights,
        base_seed,
        gain_scheduled=gain_scheduled,
        q_ref=q_ref,
        imu=imu,
    )

    print(f"\nNelder-Mead converged after {opt_result.nfev} function evaluations")
    qref_str = f", q_ref={q_ref:.0f}" if optimize_qref else ""
    print(f"  Kp={kp:.4f}, Ki={ki:.4f}, Kd={kd:.4f}{qref_str}")
    print(f"  Score={final_result['score']:.2f}")

    return final_result


def main():
    parser = argparse.ArgumentParser(
        description="Optimize PID gains for rocket roll stabilization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument(
        "--wind-levels",
        type=float,
        nargs="+",
        default=[0, 2, 5],
        help="Wind speeds to test (m/s)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=20,
        help="Episodes per wind level per candidate (default: 20, LHS uses half)",
    )
    parser.add_argument(
        "--n-lhs-samples",
        type=int,
        default=80,
        help="Number of Latin Hypercube samples (default: 80)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="optimization_results/pid_optimization.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip-lhs",
        action="store_true",
        help="Skip LHS phase, start Nelder-Mead from default gains",
    )
    parser.add_argument(
        "--start-kp",
        type=float,
        default=0.05,
        help="Starting Kp for Nelder-Mead (if --skip-lhs)",
    )
    parser.add_argument(
        "--start-ki",
        type=float,
        default=0.01,
        help="Starting Ki for Nelder-Mead (if --skip-lhs)",
    )
    parser.add_argument(
        "--start-kd",
        type=float,
        default=0.08,
        help="Starting Kd for Nelder-Mead (if --skip-lhs)",
    )
    parser.add_argument(
        "--gain-scheduled",
        action="store_true",
        help="Optimize GainScheduledPIDController instead of plain PID",
    )
    parser.add_argument(
        "--optimize-qref",
        action="store_true",
        help="Include q_ref as a 4th optimization parameter (requires --gain-scheduled)",
    )
    parser.add_argument(
        "--start-qref",
        type=float,
        default=500.0,
        help="Starting q_ref for Nelder-Mead (if --skip-lhs --optimize-qref)",
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Evaluate with IMU noise wrapper and sensor delay (observation-based control)",
    )

    args = parser.parse_args()
    if args.optimize_qref and not args.gain_scheduled:
        parser.error("--optimize-qref requires --gain-scheduled")
    config = load_config(args.config)

    # Wind weights: harder conditions weighted more
    wind_weights = {}
    for w in args.wind_levels:
        if w == 0:
            wind_weights[w] = 0.2
        elif w <= 2:
            wind_weights[w] = 0.35
        else:
            wind_weights[w] = 0.45
    # Normalize
    total = sum(wind_weights.values())
    wind_weights = {k: v / total for k, v in wind_weights.items()}

    ctrl_type = "GS-PID" if args.gain_scheduled else "PID"
    if args.imu:
        ctrl_type += " (IMU)"
    print(f"Controller type: {ctrl_type}")
    print(f"Wind weights: {wind_weights}")

    # Current baseline
    current_gains = {
        "kp": getattr(config.physics, "pid_Kp", 0.005208),
        "ki": getattr(config.physics, "pid_Ki", 0.000324),
        "kd": getattr(config.physics, "pid_Kd", 0.016524),
    }
    print(
        f"\nCurrent {ctrl_type} gains: Kp={current_gains['kp']}, Ki={current_gains['ki']}, Kd={current_gains['kd']}"
    )

    # Evaluate current baseline
    print("\nEvaluating current baseline...")
    baseline_result = evaluate_gains(
        current_gains["kp"],
        current_gains["ki"],
        current_gains["kd"],
        config,
        args.wind_levels,
        args.n_episodes,
        wind_weights,
        args.seed,
        gain_scheduled=args.gain_scheduled,
        imu=args.imu,
    )
    print(f"Baseline score: {baseline_result['score']:.2f}")
    for w, data in baseline_result["by_wind"].items():
        print(
            f"  Wind {w} m/s: spin={data['mean_spin']:.1f}+/-{data['std_spin']:.1f}, "
            f"success={data['success_rate']*100:.0f}%"
        )

    # Phase 1: LHS exploration
    if not args.skip_lhs:
        lhs_n_episodes = max(5, args.n_episodes // 2)
        lhs_results = run_lhs_phase(
            config,
            args.wind_levels,
            lhs_n_episodes,
            args.n_lhs_samples,
            wind_weights,
            args.seed,
            gain_scheduled=args.gain_scheduled,
            optimize_qref=args.optimize_qref,
            imu=args.imu,
        )
        best_lhs = min(lhs_results, key=lambda r: r["score"])
        if args.optimize_qref:
            start_point = (
                best_lhs["kp"],
                best_lhs["ki"],
                best_lhs["kd"],
                best_lhs.get("q_ref", 500.0),
            )
        else:
            start_point = (best_lhs["kp"], best_lhs["ki"], best_lhs["kd"])
    else:
        lhs_results = []
        if args.optimize_qref:
            start_point = (args.start_kp, args.start_ki, args.start_kd, args.start_qref)
        else:
            start_point = (args.start_kp, args.start_ki, args.start_kd)

    # Phase 2: Nelder-Mead refinement
    optimized_result = run_nelder_mead_phase(
        config,
        args.wind_levels,
        args.n_episodes,
        wind_weights,
        start_point,
        args.seed,
        gain_scheduled=args.gain_scheduled,
        optimize_qref=args.optimize_qref,
        imu=args.imu,
    )

    # Comparison table
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION RESULTS ({ctrl_type})")
    print(f"{'='*60}")
    print(f"{'':>20} {'Baseline':>15} {'Optimized':>15} {'Change':>12}")
    print(f"{'-'*62}")
    print(
        f"{'Kp':>20} {baseline_result['kp']:>15.4f} {optimized_result['kp']:>15.4f} "
        f"{optimized_result['kp'] - baseline_result['kp']:>+12.4f}"
    )
    print(
        f"{'Ki':>20} {baseline_result['ki']:>15.4f} {optimized_result['ki']:>15.4f} "
        f"{optimized_result['ki'] - baseline_result['ki']:>+12.4f}"
    )
    print(
        f"{'Kd':>20} {baseline_result['kd']:>15.4f} {optimized_result['kd']:>15.4f} "
        f"{optimized_result['kd'] - baseline_result['kd']:>+12.4f}"
    )
    if args.optimize_qref:
        base_qref = baseline_result.get("q_ref", 500.0)
        opt_qref = optimized_result.get("q_ref", 500.0)
        print(
            f"{'q_ref':>20} {base_qref:>15.0f} {opt_qref:>15.0f} "
            f"{opt_qref - base_qref:>+12.0f}"
        )
    print(
        f"{'Score':>20} {baseline_result['score']:>15.2f} {optimized_result['score']:>15.2f} "
        f"{optimized_result['score'] - baseline_result['score']:>+12.2f}"
    )

    for w in args.wind_levels:
        w_key = w
        if w_key in baseline_result["by_wind"] and w_key in optimized_result["by_wind"]:
            b = baseline_result["by_wind"][w_key]
            o = optimized_result["by_wind"][w_key]
            print(
                f"{'Spin @' + str(int(w)) + 'm/s':>20} "
                f"{b['mean_spin']:>15.1f} {o['mean_spin']:>15.1f} "
                f"{o['mean_spin'] - b['mean_spin']:>+12.1f}"
            )

    improvement = baseline_result["score"] - optimized_result["score"]
    pct = (
        improvement / baseline_result["score"] * 100
        if baseline_result["score"] > 0
        else 0
    )
    print(f"\nScore improvement: {improvement:.2f} ({pct:.1f}%)")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "baseline": baseline_result,
        "optimized": optimized_result,
        "config_path": args.config,
        "controller_type": ctrl_type,
        "imu": args.imu,
        "wind_levels": args.wind_levels,
        "wind_weights": {str(k): v for k, v in wind_weights.items()},
        "n_episodes": args.n_episodes,
        "n_lhs_samples": args.n_lhs_samples,
        "lhs_top10": (
            sorted(lhs_results, key=lambda r: r["score"])[:10] if lhs_results else []
        ),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    gs_flag = " --gain-scheduled" if args.gain_scheduled else ""
    imu_flag = " --imu" if args.imu else ""
    qref_flag = (
        f" --pid-qref {optimized_result.get('q_ref', 500.0):.0f}"
        if args.optimize_qref
        else ""
    )
    print(f"\nTo use optimized gains:")
    print(
        f"  uv run python compare_controllers.py --config {args.config}{gs_flag}{imu_flag} "
        f"--pid-Kp {optimized_result['kp']:.4f} "
        f"--pid-Ki {optimized_result['ki']:.4f} "
        f"--pid-Kd {optimized_result['kd']:.4f}{qref_flag} "
        f"--wind-levels {' '.join(str(int(w)) for w in args.wind_levels)} "
        f"--n-episodes 30"
    )


if __name__ == "__main__":
    main()
