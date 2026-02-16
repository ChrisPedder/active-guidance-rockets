#!/usr/bin/env python3
"""
Trajectory Monte Carlo Visualization

Shows 3D flight paths for multiple simulation runs under varying wind
conditions. Also provides a 2D panel view (x-z, y-z, x-y) which is
often easier to read.

Lateral displacement is computed using the physics-based LateralTracker
(wind drag, thrust-vector tilt, gyroscopic suppression).  Pass --legacy
to fall back to the old linear drift estimate for comparison.

Usage:
    # Display 3D plot (Estes, GS-PID)
    uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid

    # Save 2D panel view
    uv run python visualizations/trajectory_montecarlo.py --rocket j800 --controller pid \
        --mode 2d --save

    # Both modes
    uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid \
        --mode both --save

    # Old drift estimate for comparison
    uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid --legacy
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_config import load_config
from compare_controllers import create_env, create_wrapped_env, load_rl_model
from controllers.pid_controller import (
    PIDController,
    GainScheduledPIDController,
    PIDConfig,
)
from lateral_dynamics import LateralTracker

# Consistent color scheme
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
BG_COLOR = "#fafafa"
GRID_ALPHA = 0.3


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


def make_controller(controller_name: str, pid_config: PIDConfig):
    """Create a controller instance from name."""
    if controller_name == "gs-pid":
        return GainScheduledPIDController(pid_config, use_observations=False)
    else:
        return PIDController(pid_config, use_observations=False)


def _get_thrust(info, config):
    """Extract thrust from info dict with fallback."""
    t = info.get("time_s", 0.0)
    thrust = info.get("current_thrust_N", 0.0)
    burn_time = getattr(config.motor, "burn_time_s", None) or 2.0
    if thrust == 0.0 and t < burn_time:
        thrust = getattr(config.motor, "avg_thrust_N", None) or 5.4
    if t >= burn_time:
        thrust = 0.0
    return thrust


def run_episode_trajectory(
    config, wind_speed, controller_name, pid_config, seed, legacy=False
):
    """Run a single episode and return trajectory data.

    Returns dict with keys: time, altitude, x, y, velocity.
    Uses physics-based LateralTracker unless legacy=True.
    """
    env = create_env(config, wind_speed)
    airframe = env.airframe
    tracker = LateralTracker(airframe)
    tracker.reset()

    controller = make_controller(controller_name, pid_config)
    controller.reset()
    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = [0.0]
    altitudes = [0.0]
    velocities = [0.0]
    cur_dx, cur_dy = 0.0, 0.0
    drift_x = [0.0]
    drift_y = [0.0]

    while True:
        action = controller.step(obs, info, dt)
        obs, reward, terminated, truncated, info = env.step(action)

        t = info.get("time_s", 0.0)
        alt = info.get("altitude_m", 0.0)
        ws = info.get("wind_speed_ms", 0.0)
        wd = info.get("wind_direction_rad", 0.0)
        v = info.get("vertical_velocity_ms", 0.0)
        mass = info.get("mass_kg", 0.1)
        thrust = _get_thrust(info, config)

        # Physics-based lateral tracking
        tracker.update(info, thrust, mass, dt)

        # Legacy drift estimate (kept for --legacy flag)
        cur_dx += ws * np.cos(wd) * 0.3 * dt
        cur_dy += ws * np.sin(wd) * 0.3 * dt
        drift_x.append(cur_dx)
        drift_y.append(cur_dy)

        times.append(t)
        altitudes.append(alt)
        velocities.append(v)

        if terminated or truncated:
            break

    env.close()

    if legacy:
        x_arr, y_arr = np.array(drift_x), np.array(drift_y)
    else:
        x_arr, y_arr = np.array(tracker.x_history), np.array(tracker.y_history)

    return {
        "time": np.array(times),
        "altitude": np.array(altitudes),
        "x": x_arr,
        "y": y_arr,
        "velocity": np.array(velocities),
    }


def run_episode_trajectory_rl(
    config, wind_speed, model, vec_normalize, seed, legacy=False
):
    """Run a single episode with an RL model and return trajectory data.

    Uses physics-based LateralTracker unless legacy=True.
    """
    env = create_wrapped_env(config, wind_speed)
    airframe = config.physics.resolve_airframe()
    tracker = LateralTracker(airframe)
    tracker.reset()

    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = [0.0]
    altitudes = [0.0]
    velocities = [0.0]
    cur_dx, cur_dy = 0.0, 0.0
    drift_x = [0.0]
    drift_y = [0.0]

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
        mass = info.get("mass_kg", 0.1)
        thrust = _get_thrust(info, config)

        tracker.update(info, thrust, mass, dt)

        cur_dx += ws * np.cos(wd) * 0.3 * dt
        cur_dy += ws * np.sin(wd) * 0.3 * dt
        drift_x.append(cur_dx)
        drift_y.append(cur_dy)

        times.append(t)
        altitudes.append(alt)
        velocities.append(v)

        if terminated or truncated:
            break

    env.close()

    if legacy:
        x_arr, y_arr = np.array(drift_x), np.array(drift_y)
    else:
        x_arr, y_arr = np.array(tracker.x_history), np.array(tracker.y_history)

    return {
        "time": np.array(times),
        "altitude": np.array(altitudes),
        "x": x_arr,
        "y": y_arr,
        "velocity": np.array(velocities),
    }


def collect_data(
    config,
    wind_levels,
    n_runs,
    controller_name,
    pid_config,
    model=None,
    vec_normalize=None,
    legacy=False,
):
    """Collect trajectory data for all wind levels and runs."""
    data = {}
    for ws in wind_levels:
        runs = []
        for i in range(n_runs):
            seed = 2000 * int(ws) + i
            if model is not None:
                traj = run_episode_trajectory_rl(
                    config, ws, model, vec_normalize, seed, legacy=legacy
                )
            else:
                traj = run_episode_trajectory(
                    config, ws, controller_name, pid_config, seed, legacy=legacy
                )
            runs.append(traj)
            max_alt = traj["altitude"].max()
            max_xy = np.sqrt(traj["x"] ** 2 + traj["y"] ** 2).max()
            print(
                f"  Wind {ws} m/s, run {i+1}/{n_runs}: "
                f"max_alt={max_alt:.1f}m, max_xy={max_xy:.1f}m"
            )
        data[ws] = runs
    return data


def compute_axis_ranges(data):
    """Compute fixed axis ranges across all wind conditions."""
    x_min, x_max = 0.0, 0.0
    y_min, y_max = 0.0, 0.0
    z_max = 0.0

    for runs in data.values():
        for traj in runs:
            x_min = min(x_min, traj["x"].min())
            x_max = max(x_max, traj["x"].max())
            y_min = min(y_min, traj["y"].min())
            y_max = max(y_max, traj["y"].max())
            z_max = max(z_max, traj["altitude"].max())

    # Add margin
    x_range = max(x_max - x_min, 1.0) * 1.2
    y_range = max(y_max - y_min, 1.0) * 1.2
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return {
        "x_lim": (x_center - x_range / 2, x_center + x_range / 2),
        "y_lim": (y_center - y_range / 2, y_center + y_range / 2),
        "z_lim": (0, z_max * 1.1),
    }


def create_3d_animation(
    data, wind_levels, ranges, rocket, controller_name, n_runs, save_path=None
):
    """Create animated 3D trajectory plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.set_facecolor(BG_COLOR)

    pause_frames = 20
    frames_per_wind = n_runs + pause_frames
    total_frames = frames_per_wind * len(wind_levels)

    lines_3d = []

    def animate(frame):
        nonlocal lines_3d

        wind_idx = frame // frames_per_wind
        local_frame = frame % frames_per_wind

        if wind_idx >= len(wind_levels):
            return []

        ws = wind_levels[wind_idx]
        runs = data[ws]

        if local_frame == 0:
            ax.clear()
            lines_3d = []
            ax.set_xlabel("X drift (m)")
            ax.set_ylabel("Y drift (m)")
            ax.set_zlabel("Altitude (m)")
            ax.set_title(
                f"3D Trajectory — {rocket.upper()} / {controller_name.upper()}"
                f" — Wind: {ws} m/s",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlim(ranges["x_lim"])
            ax.set_ylim(ranges["y_lim"])
            ax.set_zlim(ranges["z_lim"])

        if local_frame < n_runs:
            traj = runs[local_frame]
            color = COLORS[local_frame % len(COLORS)]
            (line,) = ax.plot(
                traj["x"],
                traj["y"],
                traj["altitude"],
                color=color,
                alpha=0.6,
                linewidth=1.0,
            )
            lines_3d.append(line)

        return lines_3d

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=100,
        blit=False,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=10, bitrate=2000)
        else:
            writer = animation.PillowWriter(fps=10)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved 3D animation to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def create_2d_animation(
    data, wind_levels, ranges, rocket, controller_name, n_runs, save_path=None
):
    """Create animated 2D panel trajectory plot (x-z, y-z, x-y)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    ax_xz, ax_yz, ax_xy = axes

    pause_frames = 20
    frames_per_wind = n_runs + pause_frames
    total_frames = frames_per_wind * len(wind_levels)

    all_lines = []

    def animate(frame):
        nonlocal all_lines

        wind_idx = frame // frames_per_wind
        local_frame = frame % frames_per_wind

        if wind_idx >= len(wind_levels):
            return []

        ws = wind_levels[wind_idx]
        runs = data[ws]

        if local_frame == 0:
            for ax in axes:
                ax.clear()
                ax.set_facecolor(BG_COLOR)
                ax.grid(True, alpha=GRID_ALPHA)
            all_lines = []

            ax_xz.set_xlabel("X drift (m)")
            ax_xz.set_ylabel("Altitude (m)")
            ax_xz.set_title("X vs Altitude")
            ax_xz.set_xlim(ranges["x_lim"])
            ax_xz.set_ylim(ranges["z_lim"])

            ax_yz.set_xlabel("Y drift (m)")
            ax_yz.set_ylabel("Altitude (m)")
            ax_yz.set_title("Y vs Altitude")
            ax_yz.set_xlim(ranges["y_lim"])
            ax_yz.set_ylim(ranges["z_lim"])

            ax_xy.set_xlabel("X drift (m)")
            ax_xy.set_ylabel("Y drift (m)")
            ax_xy.set_title("Ground Track")
            ax_xy.set_xlim(ranges["x_lim"])
            ax_xy.set_ylim(ranges["y_lim"])
            ax_xy.set_aspect("equal", adjustable="datalim")

            fig.suptitle(
                f"2D Trajectory — {rocket.upper()} / "
                f"{controller_name.upper()} — Wind: {ws} m/s",
                fontsize=14,
                fontweight="bold",
            )

        if local_frame < n_runs:
            traj = runs[local_frame]
            color = COLORS[local_frame % len(COLORS)]
            (l1,) = ax_xz.plot(
                traj["x"], traj["altitude"], color=color, alpha=0.6, linewidth=1.0
            )
            (l2,) = ax_yz.plot(
                traj["y"], traj["altitude"], color=color, alpha=0.6, linewidth=1.0
            )
            (l3,) = ax_xy.plot(
                traj["x"], traj["y"], color=color, alpha=0.6, linewidth=1.0
            )
            # Mark launch point
            ax_xy.plot(0, 0, "k^", markersize=6, zorder=5)
            all_lines.extend([l1, l2, l3])

        return all_lines

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total_frames,
        interval=100,
        blit=False,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=10, bitrate=2000)
        else:
            writer = animation.PillowWriter(fps=10)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved 2D animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animated trajectory Monte Carlo visualization",
    )
    parser.add_argument(
        "--rocket",
        type=str,
        default="estes_alpha",
        choices=["estes_alpha", "j800"],
        help="Rocket configuration (default: estes_alpha)",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="gs-pid",
        help="Controller type (default: gs-pid). Ignored when an RL model is provided.",
    )
    parser.add_argument(
        "--sac",
        type=str,
        default=None,
        help="Path to SAC model .zip (overrides --controller)",
    )
    parser.add_argument(
        "--residual-sac",
        type=str,
        default=None,
        help="Path to residual SAC model .zip (overrides --controller)",
    )
    parser.add_argument(
        "--ppo",
        type=str,
        default=None,
        help="Path to PPO model .zip (overrides --controller)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=10,
        help="Number of Monte Carlo runs per wind level (default: 10)",
    )
    parser.add_argument(
        "--wind-levels",
        type=float,
        nargs="+",
        default=[1, 2, 3],
        help="Wind speeds to test (default: 1 2 3)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="3d",
        choices=["3d", "2d", "both"],
        help="Plot mode: 3d, 2d panels, or both (default: 3d)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save animation to visualizations/outputs/",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gif",
        choices=["gif", "mp4"],
        help="Output format when saving (default: gif)",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Use old linear drift estimate instead of physics-based LateralTracker",
    )
    args = parser.parse_args()

    config_path, pid_config = get_config_and_gains(args.rocket)
    config = load_config(config_path)

    # Determine controller name and load RL model if provided
    model = None
    vec_normalize = None
    controller_name = args.controller

    if args.residual_sac:
        model_path = Path(args.residual_sac)
        model_config_path = model_path.parent / "config.yaml"
        if model_config_path.exists():
            config = load_config(str(model_config_path))
            print(f"  Using config from: {model_config_path}")
        model, vec_normalize = load_rl_model(args.residual_sac, "sac", config)
        controller_name = "residual-sac"
    elif args.sac:
        model, vec_normalize = load_rl_model(args.sac, "sac", config)
        controller_name = "sac"
    elif args.ppo:
        model, vec_normalize = load_rl_model(args.ppo, "ppo", config)
        controller_name = "ppo"

    print(f"Running trajectory Monte Carlo: {args.rocket} / {controller_name}")
    print(f"  Wind levels: {args.wind_levels}")
    print(f"  Runs per level: {args.n_runs}")

    mode_label = "legacy drift" if args.legacy else "physics-based"
    print(f"  Lateral model: {mode_label}")

    data = collect_data(
        config,
        args.wind_levels,
        args.n_runs,
        controller_name,
        pid_config,
        model=model,
        vec_normalize=vec_normalize,
        legacy=args.legacy,
    )

    ranges = compute_axis_ranges(data)

    if args.mode in ("3d", "both"):
        save_path_3d = None
        if args.save:
            save_path_3d = (
                f"visualizations/outputs/trajectory_3d_"
                f"{args.rocket}_{controller_name}.{args.format}"
            )
        create_3d_animation(
            data,
            args.wind_levels,
            ranges,
            args.rocket,
            controller_name,
            args.n_runs,
            save_path_3d,
        )

    if args.mode in ("2d", "both"):
        save_path_2d = None
        if args.save:
            save_path_2d = (
                f"visualizations/outputs/trajectory_2d_"
                f"{args.rocket}_{controller_name}.{args.format}"
            )
        create_2d_animation(
            data,
            args.wind_levels,
            ranges,
            args.rocket,
            controller_name,
            args.n_runs,
            save_path_2d,
        )


if __name__ == "__main__":
    main()
