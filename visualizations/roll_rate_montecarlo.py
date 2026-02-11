#!/usr/bin/env python3
"""
Roll Rate Monte Carlo Visualization

Animated visualization of roll rate vs time across multiple simulation runs
under varying wind conditions. Traces are drawn one at a time with brief
pauses between each, then the display clears and repeats for the next wind
level. This makes the stochastic spread and wind-degradation visible.

Usage:
    # Display on screen (Estes, GS-PID)
    uv run python visualizations/roll_rate_montecarlo.py --rocket estes_alpha --controller gs-pid

    # Save to file
    uv run python visualizations/roll_rate_montecarlo.py --rocket j800 --controller pid --save

    # Custom settings
    uv run python visualizations/roll_rate_montecarlo.py --rocket estes_alpha --controller pid \
        --n-runs 10 --wind-levels 1 2 3
"""

import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rocket_config import load_config
from compare_controllers import create_env
from controllers.pid_controller import (
    PIDController,
    GainScheduledPIDController,
    PIDConfig,
)

# Consistent color scheme across all visualizations
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
WIND_COLORS = {"1": "#2ca02c", "2": "#ff7f0e", "3": "#d62728"}
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


def run_episode(config, wind_speed, controller_name, pid_config, seed):
    """Run a single episode and return (times, roll_rates_deg_s)."""
    env = create_env(config, wind_speed)
    controller = make_controller(controller_name, pid_config)
    controller.reset()
    obs, info = env.reset(seed=seed)

    dt = getattr(config.environment, "dt", 0.01)
    times = []
    roll_rates = []

    while True:
        action = controller.step(obs, info, dt)
        obs, reward, terminated, truncated, info = env.step(action)
        times.append(info.get("time_s", 0.0))
        roll_rates.append(abs(info.get("roll_rate_deg_s", 0.0)))
        if terminated or truncated:
            break

    env.close()
    return np.array(times), np.array(roll_rates)


def collect_data(config, wind_levels, n_runs, controller_name, pid_config):
    """Collect roll rate traces for all wind levels and runs."""
    data = {}
    for ws in wind_levels:
        traces = []
        for i in range(n_runs):
            seed = 1000 * int(ws) + i
            t, rr = run_episode(config, ws, controller_name, pid_config, seed)
            traces.append((t, rr))
            print(
                f"  Wind {ws} m/s, run {i+1}/{n_runs}: " f"mean={rr.mean():.1f} deg/s"
            )
        data[ws] = traces
    return data


def compute_y_max(data):
    """Compute a fixed y-axis max across all wind conditions."""
    global_max = 0.0
    for traces in data.values():
        for _, rr in traces:
            global_max = max(global_max, rr.max())
    return max(global_max * 1.1, 10.0)


def create_animation(
    data, wind_levels, y_max, rocket, controller_name, n_runs, save_path=None
):
    """Create the animated plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Frames: for each wind level, n_runs frames (one per trace) + pause frames
    pause_frames = 20  # ~2s at 10fps
    frames_per_wind = n_runs + pause_frames
    total_frames = frames_per_wind * len(wind_levels)

    lines = []
    target_line = None

    def init():
        ax.clear()
        ax.set_facecolor(BG_COLOR)
        return []

    def animate(frame):
        nonlocal lines, target_line

        wind_idx = frame // frames_per_wind
        local_frame = frame % frames_per_wind

        if wind_idx >= len(wind_levels):
            return []

        ws = wind_levels[wind_idx]
        traces = data[ws]

        # On first frame of a new wind level, clear and set up axes
        if local_frame == 0:
            ax.clear()
            ax.set_facecolor(BG_COLOR)
            lines = []

            ax.set_xlabel("Time (s)", fontsize=12)
            ax.set_ylabel("|Roll Rate| (deg/s)", fontsize=12)
            ax.set_title(
                f"Roll Rate Monte Carlo — {rocket.upper()} / "
                f"{controller_name.upper()} — Wind: {ws} m/s",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylim(0, y_max)
            ax.grid(True, alpha=GRID_ALPHA)

            # Target line
            targets = {1: 10, 2: 15, 3: 20}
            if ws in targets:
                target_val = targets[ws]
                target_line = ax.axhline(
                    y=target_val,
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label=f"Target < {target_val} deg/s",
                )
                ax.legend(loc="upper right", fontsize=10)

        # Draw traces one at a time during non-pause frames
        if local_frame < n_runs:
            trace_idx = local_frame
            t, rr = traces[trace_idx]
            color = COLORS[trace_idx % len(COLORS)]
            (line,) = ax.plot(t, rr, color=color, alpha=0.6, linewidth=1.0)
            lines.append(line)

        return lines

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
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
        print(f"Saved animation to {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animated roll rate Monte Carlo visualization",
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
        "--save",
        action="store_true",
        help="Save animation to visualizations/outputs/ instead of displaying",
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

    print(f"Running roll rate Monte Carlo: {args.rocket} / {args.controller}")
    print(f"  Wind levels: {args.wind_levels}")
    print(f"  Runs per level: {args.n_runs}")

    data = collect_data(
        config,
        args.wind_levels,
        args.n_runs,
        args.controller,
        pid_config,
    )

    y_max = compute_y_max(data)

    save_path = None
    if args.save:
        save_path = (
            f"visualizations/outputs/roll_rate_montecarlo_"
            f"{args.rocket}_{args.controller}.{args.format}"
        )

    create_animation(
        data,
        args.wind_levels,
        y_max,
        args.rocket,
        args.controller,
        args.n_runs,
        save_path,
    )


if __name__ == "__main__":
    main()
