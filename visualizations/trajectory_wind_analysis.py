#!/usr/bin/env python3
"""
Trajectory Wind Analysis — Compare Residual SAC vs PID across wind speeds.

Uses the physics-based LateralTracker for x-y displacement (wind drag,
thrust-vector tilt, gyroscopic suppression).  Also runs the old linear
drift estimate for before/after comparison.

Usage:
    uv run python visualizations/trajectory_wind_analysis.py \
        --rocket estes_alpha \
        --residual-sac models/rocket_residual_sac_wind_estes_c6_20260209_205718/best_model.zip \
        --wind-levels 0 1 3 5\
        --n-runs 5 --save
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_config import load_config
from compare_controllers import create_env, create_wrapped_env, load_rl_model
from controllers.pid_controller import (
    GainScheduledPIDController,
    PIDConfig,
)
from lateral_dynamics import LateralTracker
from airframe import RocketAirframe


def get_config_and_gains(rocket: str):
    """Return (config_path, PIDConfig) for the given rocket name."""
    if rocket == "j800":
        return "configs/aerotech_j800_wind.yaml", PIDConfig(
            Cprop=0.0213, Cint=0.0050, Cderiv=0.0271, q_ref=13268
        )
    else:
        return "configs/estes_c6_sac_wind.yaml", PIDConfig(
            Cprop=0.0203, Cint=0.0002, Cderiv=0.0118
        )


# ---------------------------------------------------------------------------
# Episode runners — return trajectories with BOTH drift-estimate AND
# physics-based x-y positions.
# ---------------------------------------------------------------------------


def run_episode_pid(config, wind_speed, controller, seed):
    """Run a single PID episode.  Return trajectory with both drift & physics x-y."""
    env = create_env(config, wind_speed)
    airframe = env.airframe
    tracker = LateralTracker(airframe)
    tracker.reset()

    controller.reset()
    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = [0.0]
    altitudes = [0.0]
    drift_x = [0.0]
    drift_y = [0.0]
    velocities = [0.0]
    spin_rates = [0.0]
    cur_dx, cur_dy = 0.0, 0.0

    while True:
        action = controller.step(obs, info, dt)
        obs, reward, terminated, truncated, info = env.step(action)

        t = info.get("time_s", 0.0)
        alt = info.get("altitude_m", 0.0)
        ws = info.get("wind_speed_ms", 0.0)
        wd = info.get("wind_direction_rad", 0.0)
        v = info.get("vertical_velocity_ms", 0.0)
        sr = info.get("roll_rate_deg_s", 0.0)
        mass = info.get("mass_kg", 0.1)

        # Infer thrust from environment state
        thrust = info.get("current_thrust_N", 0.0)
        if thrust == 0.0 and t < getattr(config.motor, "burn_time_s", 2.0):
            # Fallback for envs that don't provide current_thrust_N
            thrust = getattr(config.motor, "avg_thrust_N", 5.4)
        if t >= getattr(config.motor, "burn_time_s", 2.0):
            thrust = 0.0

        # Physics-based lateral tracking
        tracker.update(info, thrust, mass, dt)

        # Legacy drift estimate
        drift_factor = 0.3
        cur_dx += ws * np.cos(wd) * drift_factor * dt
        cur_dy += ws * np.sin(wd) * drift_factor * dt

        times.append(t)
        altitudes.append(alt)
        drift_x.append(cur_dx)
        drift_y.append(cur_dy)
        velocities.append(v)
        spin_rates.append(sr)

        if terminated or truncated:
            break

    env.close()
    return {
        "time": np.array(times),
        "altitude": np.array(altitudes),
        "x": np.array(tracker.x_history),
        "y": np.array(tracker.y_history),
        "drift_x": np.array(drift_x),
        "drift_y": np.array(drift_y),
        "velocity": np.array(velocities),
        "spin_rate": np.array(spin_rates),
        "tilt": np.array(tracker.tilt_history),
    }


def run_episode_rl(config, wind_speed, model, vec_normalize, seed):
    """Run a single RL episode.  Return trajectory with both drift & physics x-y."""
    env = create_wrapped_env(config, wind_speed)

    # Resolve airframe from config for the lateral tracker
    airframe = config.physics.resolve_airframe()
    tracker = LateralTracker(airframe)
    tracker.reset()

    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = [0.0]
    altitudes = [0.0]
    drift_x = [0.0]
    drift_y = [0.0]
    velocities = [0.0]
    spin_rates = [0.0]
    cur_dx, cur_dy = 0.0, 0.0

    while True:
        if vec_normalize is not None:
            obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
        else:
            obs_normalized = obs

        action, _ = model.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        t = info.get("time_s", 0.0)
        alt = info.get("altitude_m", 0.0)
        ws = info.get("wind_speed_ms", 0.0)
        wd = info.get("wind_direction_rad", 0.0)
        v = info.get("vertical_velocity_ms", 0.0)
        sr = info.get("roll_rate_deg_s", 0.0)
        mass = info.get("mass_kg", 0.1)

        thrust = info.get("current_thrust_N", 0.0)
        if thrust == 0.0 and t < getattr(config.motor, "burn_time_s", 2.0):
            thrust = getattr(config.motor, "avg_thrust_N", 5.4)
        if t >= getattr(config.motor, "burn_time_s", 2.0):
            thrust = 0.0

        tracker.update(info, thrust, mass, dt)

        drift_factor = 0.3
        cur_dx += ws * np.cos(wd) * drift_factor * dt
        cur_dy += ws * np.sin(wd) * drift_factor * dt

        times.append(t)
        altitudes.append(alt)
        drift_x.append(cur_dx)
        drift_y.append(cur_dy)
        velocities.append(v)
        spin_rates.append(sr)

        if terminated or truncated:
            break

    env.close()
    return {
        "time": np.array(times),
        "altitude": np.array(altitudes),
        "x": np.array(tracker.x_history),
        "y": np.array(tracker.y_history),
        "drift_x": np.array(drift_x),
        "drift_y": np.array(drift_y),
        "velocity": np.array(velocities),
        "spin_rate": np.array(spin_rates),
        "tilt": np.array(tracker.tilt_history),
    }


# ---------------------------------------------------------------------------
# Data collection & statistics
# ---------------------------------------------------------------------------


def collect_data(
    config,
    wind_levels,
    n_runs,
    controller=None,
    model=None,
    vec_normalize=None,
    label="",
):
    """Collect trajectory data for all wind levels and runs."""
    data = {}
    for ws in wind_levels:
        runs = []
        for i in range(n_runs):
            seed = 2000 * int(ws) + i
            if model is not None:
                traj = run_episode_rl(config, ws, model, vec_normalize, seed)
            else:
                traj = run_episode_pid(config, ws, controller, seed)
            runs.append(traj)
            phys_xy = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2).max()
            old_xy = np.sqrt(traj["drift_x"] ** 2 + traj["drift_y"] ** 2).max()
            mean_sr = np.mean(np.abs(traj["spin_rate"]))
            print(
                f"  [{label}] Wind {ws:5.1f} m/s, run {i+1}/{n_runs}: "
                f"phys_xy={phys_xy:.1f}m, old_xy={old_xy:.1f}m, "
                f"spin={mean_sr:.1f}deg/s"
            )
        data[ws] = runs
    return data


def _disp_stats(data, x_key="x", y_key="y"):
    """Max radial displacement statistics per wind level."""
    stats = {}
    for ws, runs in data.items():
        vals = [np.sqrt(t[x_key] ** 2 + t[y_key] ** 2).max() for t in runs]
        stats[ws] = {
            "mean": np.mean(vals),
            "std": np.std(vals),
            "max": np.max(vals),
            "min": np.min(vals),
            "all": vals,
        }
    return stats


def _spin_stats(data):
    stats = {}
    for ws, runs in data.items():
        vals = [np.mean(np.abs(t["spin_rate"])) for t in runs]
        stats[ws] = {"mean": np.mean(vals), "std": np.std(vals)}
    return stats


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_xy_comparison(pid_data, sac_data, wind_levels, rocket, save_dir=None):
    """Side-by-side x-y ground tracks (physics-based)."""
    n_winds = len(wind_levels)
    cols = min(n_winds, 4)
    rows = (n_winds + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(5 * cols * 2, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.set_facecolor("#fafafa")

    for idx, ws in enumerate(wind_levels):
        row = idx // cols
        col_base = (idx % cols) * 2
        for panel_offset, (data, lbl) in enumerate(
            [(pid_data, "GS-PID"), (sac_data, "R-SAC")]
        ):
            ax = axes[row, col_base + panel_offset]
            ax.set_facecolor("#fafafa")
            ax.grid(True, alpha=0.3)
            for i, traj in enumerate(data[ws]):
                ax.plot(
                    traj["x"],
                    traj["y"],
                    color=COLORS[i % len(COLORS)],
                    alpha=0.6,
                    linewidth=1.0,
                )
            ax.plot(0, 0, "k^", markersize=8, zorder=5)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.set_title(f"{lbl} — {ws} m/s")
            ax.set_aspect("equal", adjustable="datalim")

    for idx in range(n_winds, rows * cols):
        row = idx // cols
        col_base = (idx % cols) * 2
        axes[row, col_base].set_visible(False)
        axes[row, col_base + 1].set_visible(False)

    fig.suptitle(
        f"X-Y Ground Track (Physics) — {rocket.upper()}\n"
        f"GS-PID (left) vs Residual SAC (right)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, f"xy_physics_{rocket}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def plot_displacement_comparison(
    pid_phys, sac_phys, pid_drift, sac_drift, wind_levels, rocket, save_dir=None
):
    """Physics vs old drift: max displacement vs wind speed."""
    wl = sorted(wind_levels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("#fafafa")

    for ax, pid_s, sac_s, title in [
        (ax1, pid_phys, sac_phys, "Physics-based displacement"),
        (ax2, pid_drift, sac_drift, "Old drift-estimate displacement"),
    ]:
        ax.set_facecolor("#fafafa")
        ax.errorbar(
            wl,
            [pid_s[w]["mean"] for w in wl],
            yerr=[pid_s[w]["std"] for w in wl],
            marker="o",
            linewidth=2,
            capsize=4,
            label="GS-PID",
            color="#1f77b4",
        )
        ax.errorbar(
            wl,
            [sac_s[w]["mean"] for w in wl],
            yerr=[sac_s[w]["std"] for w in wl],
            marker="s",
            linewidth=2,
            capsize=4,
            label="Residual SAC",
            color="#ff7f0e",
        )
        ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
        ax.set_ylabel("Max X-Y Displacement (m)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, f"displacement_before_after_{rocket}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


def plot_spin_and_tilt(pid_data, sac_data, wind_levels, rocket, save_dir=None):
    """Spin rate + mean tilt angle vs wind speed."""
    wl = sorted(wind_levels)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor("#fafafa")

    # Spin rate
    for data, lbl, clr, mk in [
        (pid_data, "GS-PID", "#1f77b4", "o"),
        (sac_data, "R-SAC", "#ff7f0e", "s"),
    ]:
        means = [
            np.mean([np.mean(np.abs(t["spin_rate"])) for t in data[w]]) for w in wl
        ]
        stds = [np.std([np.mean(np.abs(t["spin_rate"])) for t in data[w]]) for w in wl]
        ax1.errorbar(
            wl,
            means,
            yerr=stds,
            marker=mk,
            linewidth=2,
            capsize=4,
            label=lbl,
            color=clr,
        )
    ax1.set_xlabel("Wind Speed (m/s)")
    ax1.set_ylabel("Mean |Spin Rate| (deg/s)")
    ax1.set_title("Mean Spin Rate vs Wind Speed", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean tilt angle
    for data, lbl, clr, mk in [
        (pid_data, "GS-PID", "#1f77b4", "o"),
        (sac_data, "R-SAC", "#ff7f0e", "s"),
    ]:
        means = [np.mean([np.mean(np.abs(t["tilt"])) for t in data[w]]) for w in wl]
        ax2.plot(wl, means, f"{mk}-", linewidth=2, label=lbl, color=clr)
    ax2.set_xlabel("Wind Speed (m/s)")
    ax2.set_ylabel("Mean |Tilt Angle| (deg)")
    ax2.set_title("Mean Pitch Tilt vs Wind Speed", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_dir:
        p = os.path.join(save_dir, f"spin_tilt_vs_wind_{rocket}.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_summary(
    pid_phys, sac_phys, pid_drift, sac_drift, pid_spin, sac_spin, wind_levels
):
    wl = sorted(wind_levels)
    hdr = (
        f"{'Wind':>5s}  |  {'PID phys (m)':>14s}  {'SAC phys (m)':>14s}  |  "
        f"{'PID drift (m)':>14s}  {'SAC drift (m)':>14s}  |  "
        f"{'PID spin':>11s}  {'SAC spin':>11s}"
    )
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for w in wl:
        pp = pid_phys[w]
        sp = sac_phys[w]
        pd = pid_drift[w]
        sd = sac_drift[w]
        ps = pid_spin[w]
        ss = sac_spin[w]
        print(
            f"{w:5.1f}  |  {pp['mean']:6.1f}+/-{pp['std']:4.1f}   "
            f"{sp['mean']:6.1f}+/-{sp['std']:4.1f}   |  "
            f"{pd['mean']:6.1f}+/-{pd['std']:4.1f}   "
            f"{sd['mean']:6.1f}+/-{sd['std']:4.1f}   |  "
            f"{ps['mean']:5.1f}+/-{ps['std']:3.1f}  "
            f"{ss['mean']:5.1f}+/-{ss['std']:3.1f}"
        )
    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Trajectory wind analysis with physics-based lateral dynamics"
    )
    parser.add_argument(
        "--rocket", type=str, default="estes_alpha", choices=["estes_alpha", "j800"]
    )
    parser.add_argument(
        "--residual-sac", type=str, default=None, help="Path to residual SAC model .zip"
    )
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument(
        "--wind-levels", type=float, nargs="+", default=[0, 1, 3, 5, 10, 15, 20, 25]
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    config_path, pid_config = get_config_and_gains(args.rocket)
    config = load_config(config_path)
    save_dir = "visualizations/outputs" if args.save else None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Load SAC model
    sac_model, sac_vn, sac_config = None, None, config
    if args.residual_sac:
        mp = Path(args.residual_sac)
        cp = mp.parent / "config.yaml"
        if cp.exists():
            sac_config = load_config(str(cp))
            print(f"  SAC config: {cp}")
        sac_model, sac_vn = load_rl_model(args.residual_sac, "sac", sac_config)
        print(f"  Loaded model: {args.residual_sac}")

    # --- PID ---
    print(f"\n{'='*60}\nRunning GS-PID: {args.rocket}\n{'='*60}")
    pid_ctrl = GainScheduledPIDController(pid_config, use_observations=False)
    pid_data = collect_data(
        config, args.wind_levels, args.n_runs, controller=pid_ctrl, label="GS-PID"
    )

    # --- SAC ---
    sac_data = None
    if sac_model is not None:
        print(f"\n{'='*60}\nRunning Residual SAC: {args.rocket}\n{'='*60}")
        sac_data = collect_data(
            sac_config,
            args.wind_levels,
            args.n_runs,
            model=sac_model,
            vec_normalize=sac_vn,
            label="R-SAC",
        )

    # --- Stats ---
    pid_phys = _disp_stats(pid_data, "x", "y")
    pid_drift = _disp_stats(pid_data, "drift_x", "drift_y")
    pid_spin = _spin_stats(pid_data)

    if sac_data is not None:
        sac_phys = _disp_stats(sac_data, "x", "y")
        sac_drift = _disp_stats(sac_data, "drift_x", "drift_y")
        sac_spin = _spin_stats(sac_data)
    else:
        z = {
            w: {"mean": 0, "std": 0, "max": 0, "min": 0, "all": []}
            for w in args.wind_levels
        }
        sac_phys = sac_drift = z
        sac_spin = {w: {"mean": 0, "std": 0} for w in args.wind_levels}

    # --- Output ---
    print_summary(
        pid_phys, sac_phys, pid_drift, sac_drift, pid_spin, sac_spin, args.wind_levels
    )

    if sac_data is not None:
        print("\nGenerating plots...")
        plot_xy_comparison(
            pid_data, sac_data, sorted(args.wind_levels), args.rocket, save_dir
        )
        plot_displacement_comparison(
            pid_phys,
            sac_phys,
            pid_drift,
            sac_drift,
            args.wind_levels,
            args.rocket,
            save_dir,
        )
        plot_spin_and_tilt(pid_data, sac_data, args.wind_levels, args.rocket, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
