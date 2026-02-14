#!/usr/bin/env python3
"""Dynamic pressure and 3D trajectory during boost + coast from real flight data.

Computes q = 0.5 * rho * v^2 using measured barometric pressure, temperature,
and speed from the altimeter CSV.  Air density is derived via the ideal gas law.

Left panel:  3D trajectory of the ascent (boost + coast), coloured by q.
Right panel: Dynamic pressure vs time with boost/coast phases shaded.

Usage:
    uv run python visualizations/plot_flight_dynamic_pressure.py
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
OUT_PATH = Path(__file__).resolve().parent / "outputs" / "flight_dynamic_pressure.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────
R_AIR = 287.05  # J/(kg K) — specific gas constant for dry air

# ── load data ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, skipinitialspace=True)
df.columns = df.columns.str.strip().str.lstrip("#")

df = df.drop_duplicates(subset="time", keep="first")
df = df.sort_values("time").reset_index(drop=True)

time = df["time"].values
height = df["height"].values  # m AGL
speed = df["speed"].values  # m/s
pressure = df["pressure"].values  # Pa
temperature = df["temperature"].values  # deg C
state = df["state_name"].str.strip().values

# ── filter to boost + coast only ─────────────────────────────────────────
up_mask = np.isin(state, ["boost", "coast"])
t = time[up_mask]
h = height[up_mask]
v = speed[up_mask]
p = pressure[up_mask]
T_kelvin = temperature[up_mask] + 273.15
st = state[up_mask]

# ── compute dynamic pressure ─────────────────────────────────────────────
rho = p / (R_AIR * T_kelvin)  # ideal gas law
q = 0.5 * rho * v**2  # Pa

# ── synthesise lateral drift (same wind model as plot_3d_flight.py) ──────
WIND_EAST = 4.4  # m/s
WIND_NORTH = 4.4
dt = np.diff(t, prepend=t[0])
vert_speed = np.abs(np.gradient(h, t))
drift_frac = np.clip(1.0 - vert_speed / 50.0, 0.0, 1.0)

x = np.cumsum(WIND_EAST * drift_frac * dt)
y = np.cumsum(WIND_NORTH * drift_frac * dt)
x -= x[0]
y -= y[0]

# ── phase boundary ───────────────────────────────────────────────────────
boost_mask = st == "boost"
coast_mask = st == "coast"

# Time of motor burnout (last boost sample)
t_burnout = t[boost_mask][-1] if boost_mask.any() else None

# ── plot ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 6.5), facecolor="white")

# --- Left: 3D trajectory coloured by q ---
ax3d = fig.add_subplot(121, projection="3d", facecolor="white")

points = np.column_stack([x, y, h]).reshape(-1, 1, 3)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

norm = plt.Normalize(vmin=q.min(), vmax=q.max())
cmap = plt.cm.inferno

lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidths=2.5)
lc.set_array(q[:-1])
ax3d.add_collection3d(lc)

# Launch marker
ax3d.scatter(
    x[0],
    y[0],
    h[0],
    color="white",
    edgecolors="black",
    marker="o",
    s=90,
    linewidths=1.2,
    zorder=5,
    label="Launch",
)

# Burnout marker
if t_burnout is not None:
    bo_idx = np.where(boost_mask)[0][-1]
    ax3d.scatter(
        x[bo_idx],
        y[bo_idx],
        h[bo_idx],
        color="#2ecc71",
        edgecolors="black",
        marker="^",
        s=100,
        linewidths=1.2,
        zorder=5,
        label="Burnout",
    )

# Apogee marker
apogee_idx = np.argmax(h)
ax3d.scatter(
    x[apogee_idx],
    y[apogee_idx],
    h[apogee_idx],
    color="#e74c3c",
    edgecolors="black",
    marker="v",
    s=100,
    linewidths=1.2,
    zorder=5,
    label="Apogee",
)

pad = 5
ax3d.set_xlim(x.min() - pad, x.max() + pad)
ax3d.set_ylim(y.min() - pad, y.max() + pad)
ax3d.set_zlim(0, h.max() * 1.05)

ax3d.set_xlabel("East (m)", fontsize=10, labelpad=8)
ax3d.set_ylabel("North (m)", fontsize=10, labelpad=8)
ax3d.set_zlabel("Height AGL (m)", fontsize=10, labelpad=8)
ax3d.set_title("Ascent Trajectory", fontsize=13, fontweight="bold", pad=12)
ax3d.view_init(elev=25, azim=-50)
ax3d.tick_params(labelsize=8)
ax3d.legend(loc="upper left", fontsize=9, framealpha=0.9)

# Colour bar for 3D panel
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar3d = fig.colorbar(sm, ax=ax3d, shrink=0.55, aspect=20, pad=0.10)
cbar3d.set_label("Dynamic pressure q (Pa)", fontsize=10)
cbar3d.ax.tick_params(labelsize=8)

# --- Right: q vs time ---
ax_q = fig.add_subplot(122, facecolor="white")

# Phase shading
if boost_mask.any():
    ax_q.axvspan(
        t[boost_mask][0], t[boost_mask][-1], alpha=0.12, color="#3b82f6", label="Boost"
    )
if coast_mask.any():
    ax_q.axvspan(
        t[coast_mask][0], t[coast_mask][-1], alpha=0.12, color="#22c55e", label="Coast"
    )

# Burnout line
if t_burnout is not None:
    ax_q.axvline(t_burnout, color="#6b7280", ls="--", lw=1.0, alpha=0.7)
    ax_q.text(
        t_burnout + 0.15,
        q.max() * 0.95,
        "Burnout",
        fontsize=8,
        color="#6b7280",
        va="top",
    )

# q curve coloured by phase
ax_q.plot(t[boost_mask], q[boost_mask], color="#3b82f6", lw=2.0)
ax_q.plot(t[coast_mask], q[coast_mask], color="#22c55e", lw=2.0)

# Peak q marker
q_max_idx = np.argmax(q)
ax_q.plot(
    t[q_max_idx],
    q[q_max_idx],
    "o",
    color="#e74c3c",
    ms=8,
    zorder=5,
    markeredgecolor="black",
    mew=1.0,
)
ax_q.annotate(
    f"Max q = {q[q_max_idx]:.0f} Pa\n(t = {t[q_max_idx]:.1f} s)",
    xy=(t[q_max_idx], q[q_max_idx]),
    xytext=(t[q_max_idx] + 1.5, q[q_max_idx] * 0.85),
    fontsize=9,
    color="#e74c3c",
    arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2),
)

ax_q.set_xlabel("Time (s)", fontsize=11)
ax_q.set_ylabel("Dynamic pressure q (Pa)", fontsize=11)
ax_q.set_title("Dynamic Pressure During Ascent", fontsize=13, fontweight="bold")
ax_q.legend(loc="upper right", fontsize=9, framealpha=0.9)
ax_q.grid(True, alpha=0.3)
ax_q.tick_params(labelsize=9)
ax_q.set_xlim(t[0] - 0.5, t[-1] + 0.5)
ax_q.set_ylim(0, q.max() * 1.1)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight")
print(f"Saved \u2192 {OUT_PATH}")
plt.close(fig)
