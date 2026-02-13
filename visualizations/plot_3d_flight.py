#!/usr/bin/env python3
"""
3D flight trajectory plot from altimeter telemetry.

The CSV has 1D altitude data only (no GPS), so lateral (X/Y) position is
synthesised from a simple wind-drift model: the rocket ascends nearly
vertically during boost, then drifts downwind under parachute.  This gives
a physically plausible 3D shape while faithfully reproducing the measured
altitude and speed profiles.

Usage:
    uv run python visualizations/plot_3d_flight.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "flight_data"
    / "2024-10-12-serial-8617-flight-0001.csv"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "slides"
    / "images"
    / "frenzy_3d_flight_plot.png"
)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, skipinitialspace=True)
df.columns = df.columns.str.strip().str.lstrip("#")

# Drop duplicate t=0 rows and rows before launch
df = df.drop_duplicates(subset="time", keep="first")
df = df.sort_values("time").reset_index(drop=True)

time = df["time"].values
height = df["height"].values  # metres AGL
speed = df["speed"].values  # m/s  (magnitude)
state = df["state_name"].str.strip()

# ── synthesise lateral drift ─────────────────────────────────────────────
# Wind at ~2 m/s from SW pushes the rocket NE during coast/descent.
# During boost the rocket is fast & vertical so lateral drift is tiny.
WIND_EAST = 4.4  # m/s  (component)
WIND_NORTH = 4.4  # m/s
dt = np.diff(time, prepend=time[0])

# Lateral speed is proportional to wind and inversely proportional to
# vertical speed (fast rocket ≈ vertical, slow descent ≈ wind-dominated).
vert_speed = np.abs(np.gradient(height, time))
drift_frac = np.clip(1.0 - vert_speed / 50.0, 0.0, 1.0)

x = np.cumsum(WIND_EAST * drift_frac * dt)
y = np.cumsum(WIND_NORTH * drift_frac * dt)

# Centre so launch pad is at origin
x -= x[0]
y -= y[0]

# ── find parachute deployment indices ────────────────────────────────────
state_arr = state.values

drogue_idx = None
for i in range(1, len(state_arr)):
    if state_arr[i] == "drogue" and state_arr[i - 1] != "drogue":
        drogue_idx = i
        break

main_idx = None
for i in range(1, len(state_arr)):
    if state_arr[i] == "main" and state_arr[i - 1] != "main":
        main_idx = i
        break

# ── build colour-mapped 3D line collection ───────────────────────────────
points = np.column_stack([x, y, height]).reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(vmin=speed.min(), vmax=speed.max())
cmap = plt.cm.plasma

# ── plot ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 9), facecolor="white")
ax = fig.add_subplot(111, projection="3d", facecolor="white")

lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=2.5)
lc.set_array(speed[:-1])
ax.add_collection3d(lc)

# Parachute markers
marker_kw = dict(s=120, zorder=5, edgecolors="black", linewidths=1.2)
if drogue_idx is not None:
    ax.scatter(
        x[drogue_idx],
        y[drogue_idx],
        height[drogue_idx],
        color="#2ecc71",
        marker="v",
        label="Drogue deploy",
        **marker_kw,
    )
if main_idx is not None:
    ax.scatter(
        x[main_idx],
        y[main_idx],
        height[main_idx],
        color="#e74c3c",
        marker="v",
        label="Main deploy",
        **marker_kw,
    )

# Launch point
ax.scatter(
    x[0],
    y[0],
    height[0],
    color="white",
    edgecolors="black",
    marker="o",
    s=90,
    linewidths=1.2,
    zorder=5,
    label="Launch",
)

# ── shadow / ground projection ───────────────────────────────────────────
ground_z = height.min() - 5
ground_pts = np.column_stack([x, y, np.full_like(height, ground_z)])
ground_segs = np.column_stack(
    [ground_pts[:-1].reshape(-1, 1, 3), ground_pts[1:].reshape(-1, 1, 3)]
)
ground_segs = np.concatenate(
    [ground_pts[:-1].reshape(-1, 1, 3), ground_pts[1:].reshape(-1, 1, 3)], axis=1
)
ground_lc = Line3DCollection(
    ground_segs, colors=(0.55, 0.55, 0.55, 0.4), linewidths=1.0
)
ax.add_collection3d(ground_lc)

# ── axes & labels ────────────────────────────────────────────────────────
pad = 5
ax.set_xlim(x.min() - pad, x.max() + pad)
ax.set_ylim(y.min() - pad, y.max() + pad)
ax.set_zlim(ground_z, height.max() * 1.05)

ax.set_xlabel("East (m)", fontsize=11, labelpad=10)
ax.set_ylabel("North (m)", fontsize=11, labelpad=10)
ax.set_zlabel("Height AGL (m)", fontsize=11, labelpad=10)
ax.set_title("Frenzy Flight Trajectory", fontsize=15, fontweight="bold", pad=18)

ax.view_init(elev=25, azim=-50)
ax.tick_params(labelsize=9)

# Colour bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.55, aspect=20, pad=0.10)
cbar.set_label("Speed (m/s)", fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
plt.close(fig)
