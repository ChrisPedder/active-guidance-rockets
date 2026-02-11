#!/usr/bin/env python3
"""
Trajectory Monte Carlo Visualization

Shows 3D flight paths for multiple simulation runs under varying wind
conditions. Also provides a 2D panel view (x-z, y-z, x-y) which is
often easier to read.

Lateral drift is estimated from wind speed and direction at each timestep
(the simulation only tracks altitude; this gives a first-order wind-drift
estimate).

Usage:
    # Display 3D plot (Estes, GS-PID)
    uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid

    # Save 2D panel view
    uv run python visualizations/trajectory_montecarlo.py --rocket j800 --controller pid \
        --mode 2d --save

    # Both modes
    uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid \
        --mode both --save
"""

import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_config import load_config
from compare_controllers import create_env
from controllers.pid_controller import (
    PIDController,
    GainScheduledPIDController,
    PIDConfig,
)

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


def run_episode_trajectory(config, wind_speed, controller_name, pid_config, seed):
    """Run a single episode and return trajectory data.

    Returns dict with keys: time, altitude, x, y, velocity
    Lateral position is estimated by integrating wind drift.
    """
    env = create_env(config, wind_speed)
    controller = make_controller(controller_name, pid_config)
    controller.reset()
    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = [0.0]
    altitudes = [0.0]
    x_pos = [0.0]
    y_pos = [0.0]
    velocities = [0.0]

    # Integrate lateral drift from wind
    cur_x, cur_y = 0.0, 0.0

    while True:
        action = controller.step(obs, info, dt)
        obs, reward, terminated, truncated, info = env.step(action)

        t = info.get("time_s", 0.0)
        alt = info.get("altitude_m", 0.0)
        ws = info.get("wind_speed_ms", 0.0)
        wd = info.get("wind_direction_rad", 0.0)
        v = info.get("vertical_velocity_ms", 0.0)

        # Estimate lateral drift from wind
        # Wind pushes rocket sideways; drift ~ wind_speed * dt
        # Use a fraction of wind speed (rocket has some inertia)
        drift_factor = 0.3  # empirical scaling for lateral drift
        cur_x += ws * np.cos(wd) * drift_factor * dt
        cur_y += ws * np.sin(wd) * drift_factor * dt

        times.append(t)
        altitudes.append(alt)
        x_pos.append(cur_x)
        y_pos.append(cur_y)
        velocities.append(v)

        if terminated or truncated:
            break

    env.close()
    return {
        "time": np.array(times),
        "altitude": np.array(altitudes),
        "x": np.array(x_pos),
        "y": np.array(y_pos),
        "velocity": np.array(velocities),
    }


def collect_data(config, wind_levels, n_runs, controller_name, pid_config):
    """Collect trajectory data for all wind levels and runs."""
    data = {}
    for ws in wind_levels:
        runs = []
        for i in range(n_runs):
            seed = 2000 * int(ws) + i
            traj = run_episode_trajectory(config, ws, controller_name, pid_config, seed)
            runs.append(traj)
            max_alt = traj["altitude"].max()
            print(f"  Wind {ws} m/s, run {i+1}/{n_runs}: " f"max_alt={max_alt:.1f}m")
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
        choices=["pid", "gs-pid"],
        help="Controller type (default: gs-pid)",
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
    args = parser.parse_args()

    config_path, pid_config = get_config_and_gains(args.rocket)
    config = load_config(config_path)

    print(f"Running trajectory Monte Carlo: {args.rocket} / {args.controller}")
    print(f"  Wind levels: {args.wind_levels}")
    print(f"  Runs per level: {args.n_runs}")

    data = collect_data(
        config,
        args.wind_levels,
        args.n_runs,
        args.controller,
        pid_config,
    )

    ranges = compute_axis_ranges(data)

    if args.mode in ("3d", "both"):
        save_path_3d = None
        if args.save:
            save_path_3d = (
                f"visualizations/outputs/trajectory_3d_"
                f"{args.rocket}_{args.controller}.{args.format}"
            )
        create_3d_animation(
            data,
            args.wind_levels,
            ranges,
            args.rocket,
            args.controller,
            args.n_runs,
            save_path_3d,
        )

    if args.mode in ("2d", "both"):
        save_path_2d = None
        if args.save:
            save_path_2d = (
                f"visualizations/outputs/trajectory_2d_"
                f"{args.rocket}_{args.controller}.{args.format}"
            )
        create_2d_animation(
            data,
            args.wind_levels,
            ranges,
            args.rocket,
            args.controller,
            args.n_runs,
            save_path_2d,
        )


if __name__ == "__main__":
    main()
