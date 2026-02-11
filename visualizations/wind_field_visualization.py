#!/usr/bin/env python3
"""
Wind Field Visualization

Visualizes the wind model (sinusoidal or Dryden) to give intuition about
what disturbances the rocket faces during flight. Shows:
  1. Wind speed magnitude vs altitude at multiple time snapshots
  2. Wind direction vs altitude at multiple time snapshots
  3. Time-series of wind speed and direction at a fixed altitude

The plots animate over a typical flight duration so the viewer sees the
wind field evolving in real time.

Usage:
    # Display on screen
    uv run python visualizations/wind_field_visualization.py --rocket estes_alpha

    # Save to file
    uv run python visualizations/wind_field_visualization.py --rocket j800 --save

    # Custom wind speed
    uv run python visualizations/wind_field_visualization.py --rocket estes_alpha \
        --wind-speed 5.0 --dryden
"""

import argparse
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wind_model import WindModel, WindConfig

# Consistent styling
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


def get_flight_params(rocket: str):
    """Return (max_altitude, flight_duration, typical_velocity) for rocket."""
    if rocket == "j800":
        return 2000.0, 25.0, 200.0
    else:
        return 200.0, 8.0, 30.0


def sample_wind_field(wind_model, altitudes, times, rocket_velocity):
    """Sample wind speed and direction over a grid of altitudes and times.

    Returns:
        speeds: array of shape (len(times), len(altitudes))
        directions: array of shape (len(times), len(altitudes))
    """
    speeds = np.zeros((len(times), len(altitudes)))
    directions = np.zeros((len(times), len(altitudes)))

    for i, t in enumerate(times):
        for j, alt in enumerate(altitudes):
            s, d = wind_model.get_wind(t, alt, rocket_velocity)
            speeds[i, j] = s
            directions[i, j] = np.rad2deg(d) % 360
    return speeds, directions


def sample_timeseries(wind_model, fixed_altitude, times, rocket_velocity):
    """Sample wind at a fixed altitude over time.

    Returns:
        ts_speeds: array of shape (len(times),)
        ts_directions: array of shape (len(times),)
    """
    ts_speeds = np.zeros(len(times))
    ts_directions = np.zeros(len(times))
    for i, t in enumerate(times):
        s, d = wind_model.get_wind(t, fixed_altitude, rocket_velocity)
        ts_speeds[i] = s
        ts_directions[i] = np.rad2deg(d) % 360
    return ts_speeds, ts_directions


def create_animation(
    wind_model,
    rocket,
    wind_speed,
    max_alt,
    flight_dur,
    rocket_vel,
    fixed_alt,
    save_path=None,
):
    """Create the animated 3-panel wind field visualization."""
    altitudes = np.linspace(1.0, max_alt, 100)
    dt_sample = 0.05  # 20 Hz sampling
    times = np.arange(0, flight_dur, dt_sample)

    # Pre-compute full time series at fixed altitude
    ts_speeds, ts_dirs = sample_timeseries(wind_model, fixed_alt, times, rocket_vel)

    # Snapshot times for profile plots
    n_snapshots = 8
    snapshot_interval = max(1, len(times) // (n_snapshots * 4))

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.set_facecolor(BG_COLOR)
    for ax in axes:
        ax.set_facecolor(BG_COLOR)

    fig.suptitle(
        f"Wind Field — {rocket.upper()} — " f"Base Speed: {wind_speed} m/s",
        fontsize=14,
        fontweight="bold",
    )

    # Axis setup
    ax_speed, ax_dir, ax_ts = axes

    ax_speed.set_xlabel("Wind Speed (m/s)", fontsize=11)
    ax_speed.set_ylabel("Altitude (m)", fontsize=11)
    ax_speed.set_title("Wind Speed vs Altitude", fontsize=12)
    ax_speed.set_xlim(0, max(wind_speed * 3, 1.0))
    ax_speed.set_ylim(0, max_alt)
    ax_speed.grid(True, alpha=GRID_ALPHA)

    ax_dir.set_xlabel("Wind Direction (deg)", fontsize=11)
    ax_dir.set_ylabel("Altitude (m)", fontsize=11)
    ax_dir.set_title("Wind Direction vs Altitude", fontsize=12)
    ax_dir.set_xlim(0, 360)
    ax_dir.set_ylim(0, max_alt)
    ax_dir.grid(True, alpha=GRID_ALPHA)

    ax_ts.set_xlabel("Time (s)", fontsize=11)
    ax_ts.set_title(f"Wind at {fixed_alt:.0f}m Altitude", fontsize=12)
    ax_ts.set_xlim(0, flight_dur)
    ax_ts.grid(True, alpha=GRID_ALPHA)

    # Dual y-axis for time series
    ax_ts_dir = ax_ts.twinx()
    ax_ts.set_ylabel("Speed (m/s)", fontsize=11, color=COLORS[0])
    ax_ts_dir.set_ylabel("Direction (deg)", fontsize=11, color=COLORS[1])
    ax_ts.set_ylim(0, max(ts_speeds.max() * 1.3, 1.0))
    ax_ts_dir.set_ylim(0, 360)

    # Animation state
    profile_lines_speed = []
    profile_lines_dir = []
    (speed_line,) = ax_ts.plot([], [], color=COLORS[0], linewidth=1.5, label="Speed")
    (dir_line,) = ax_ts_dir.plot(
        [], [], color=COLORS[1], linewidth=1.5, alpha=0.7, label="Direction"
    )
    ax_ts.legend(loc="upper left", fontsize=9)
    ax_ts_dir.legend(loc="upper right", fontsize=9)

    # Animate: step through time
    # Target ~200 frames, but ensure at least 2 frames for valid animation
    step_size = max(1, len(times) // 200)
    n_frames = max(2, len(times) // step_size)

    def animate(frame_idx):
        nonlocal profile_lines_speed, profile_lines_dir

        t_idx = frame_idx * step_size
        if t_idx >= len(times):
            t_idx = len(times) - 1

        current_time = times[t_idx]

        # Update time series (show data up to current time)
        speed_line.set_data(times[: t_idx + 1], ts_speeds[: t_idx + 1])
        dir_line.set_data(times[: t_idx + 1], ts_dirs[: t_idx + 1])

        # Add altitude profile snapshot at regular intervals
        if frame_idx % snapshot_interval == 0 and frame_idx > 0:
            # Fade old profiles
            for line in profile_lines_speed:
                line.set_alpha(max(line.get_alpha() - 0.1, 0.1))
            for line in profile_lines_dir:
                line.set_alpha(max(line.get_alpha() - 0.1, 0.1))

            # Compute current profile
            snap_speeds = np.zeros(len(altitudes))
            snap_dirs = np.zeros(len(altitudes))
            for j, alt in enumerate(altitudes):
                s, d = wind_model.get_wind(current_time, alt, rocket_vel)
                snap_speeds[j] = s
                snap_dirs[j] = np.rad2deg(d) % 360

            color_idx = len(profile_lines_speed) % len(COLORS)
            label = f"t={current_time:.1f}s"
            (ln_s,) = ax_speed.plot(
                snap_speeds,
                altitudes,
                color=COLORS[color_idx],
                alpha=0.7,
                linewidth=1.2,
                label=label,
            )
            (ln_d,) = ax_dir.plot(
                snap_dirs,
                altitudes,
                color=COLORS[color_idx],
                alpha=0.7,
                linewidth=1.2,
            )
            profile_lines_speed.append(ln_s)
            profile_lines_dir.append(ln_d)

            # Update legend (only show last few)
            if len(profile_lines_speed) <= 8:
                ax_speed.legend(fontsize=8, loc="upper right")

        return [speed_line, dir_line] + profile_lines_speed + profile_lines_dir

    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=n_frames,
        interval=50,
        blit=False,
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=20, bitrate=2000)
        else:
            writer = animation.PillowWriter(fps=20)
        anim.save(save_path, writer=writer, dpi=120)
        print(f"Saved animation to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Animated wind field visualization",
    )
    parser.add_argument(
        "--rocket",
        type=str,
        default="estes_alpha",
        choices=["estes_alpha", "j800"],
        help="Rocket configuration (default: estes_alpha)",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=2.0,
        help="Mean wind speed in m/s (default: 2.0)",
    )
    parser.add_argument(
        "--dryden",
        action="store_true",
        help="Use Dryden turbulence model instead of sinusoidal",
    )
    parser.add_argument(
        "--severity",
        type=str,
        default="moderate",
        choices=["light", "moderate", "severe"],
        help="Dryden turbulence severity (default: moderate)",
    )
    parser.add_argument(
        "--fixed-altitude",
        type=float,
        default=None,
        help="Altitude for time-series plot (default: auto)",
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

    max_alt, flight_dur, rocket_vel = get_flight_params(args.rocket)

    if args.fixed_altitude is not None:
        fixed_alt = args.fixed_altitude
    else:
        fixed_alt = min(200.0, max_alt * 0.5)

    wind_config = WindConfig(
        enable=True,
        base_speed=args.wind_speed,
        max_gust_speed=args.wind_speed * 0.5,
        variability=0.3,
        use_dryden=args.dryden,
        turbulence_severity=args.severity,
        altitude_profile_alpha=0.14,
        reference_altitude=10.0,
    )
    wind_model = WindModel(wind_config)
    wind_model.reset(seed=42)

    model_name = "Dryden" if args.dryden else "Sinusoidal"
    print(f"Wind field visualization: {args.rocket}")
    print(f"  Model: {model_name}")
    print(f"  Base speed: {args.wind_speed} m/s")
    print(f"  Max altitude: {max_alt} m")
    print(f"  Flight duration: {flight_dur} s")
    print(f"  Fixed altitude for timeseries: {fixed_alt} m")

    save_path = None
    if args.save:
        model_tag = "dryden" if args.dryden else "sinusoidal"
        save_path = (
            f"visualizations/outputs/wind_field_"
            f"{args.rocket}_{model_tag}_{args.wind_speed}ms.{args.format}"
        )

    create_animation(
        wind_model,
        args.rocket,
        args.wind_speed,
        max_alt,
        flight_dur,
        rocket_vel,
        fixed_alt,
        save_path,
    )


if __name__ == "__main__":
    main()
