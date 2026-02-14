#!/usr/bin/env python3
"""Side-view rocket stability animation: CG forward of CP → restoring torque.

Produces a looping GIF of a rocket oscillating after a wind disturbance,
demonstrating the weathercock / restoring torque mechanism.  The rocket
pivots about its CG while the aerodynamic normal force acts at the CP
(aft of CG), creating a restoring moment.

Usage:
    uv run python visualizations/animate_stability.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyArrowPatch, Arc
from pathlib import Path

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_PATH = Path(__file__).resolve().parent / "outputs" / "rocket_stability.gif"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physics parameters (sinusoidal loop)
# ---------------------------------------------------------------------------
THETA_MAX_DEG = 15.0  # peak tilt angle (degrees)
T_PERIOD = 6.0  # oscillation period (s)

# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------
FPS = 25
DPI = 72
N_FRAMES = 150  # 6 s x 25 fps — exact loop
FIGSIZE = (7, 8)

# ---------------------------------------------------------------------------
# Geometry (in body diameters; body_diam = 1.0)
# ---------------------------------------------------------------------------
BODY_DIAM = 1.0
BODY_R = BODY_DIAM / 2  # 0.5

# Rocket polygons in local coords (CG at origin, +y = nose)
# Nose cone: triangle
NOSE_PTS = np.array(
    [
        [0.0, 5.0],  # tip
        [BODY_R, 3.0],  # right shoulder
        [-BODY_R, 3.0],  # left shoulder
    ]
)

# Body tube: rectangle
BODY_PTS = np.array(
    [
        [BODY_R, 3.0],  # top right
        [BODY_R, -4.5],  # bottom right
        [-BODY_R, -4.5],  # bottom left
        [-BODY_R, 3.0],  # top left
    ]
)

# Right fin
FIN_R_PTS = np.array(
    [
        [BODY_R, -3.0],
        [1.3, -4.0],
        [BODY_R, -4.5],
    ]
)

# Left fin
FIN_L_PTS = np.array(
    [
        [-BODY_R, -3.0],
        [-1.3, -4.0],
        [-BODY_R, -4.5],
    ]
)

# CG at origin, CP aft
CG_POS = np.array([0.0, 0.0])
CP_LOCAL = np.array([0.0, -1.5])  # 1.5 body diameters aft of CG

# ---------------------------------------------------------------------------
# Colours (dark theme matching animate_roll_control.py)
# ---------------------------------------------------------------------------
BG_COLOR = "#0f0f1a"
ROCKET_FILL = "#2a2a3a"
ROCKET_EDGE = "#c0c0c0"
CG_COLOR = "#22c55e"  # green
CP_COLOR = "#f59e0b"  # amber
FORCE_COLOR = "#ef4444"  # red
WIND_COLOR = "#60a5fa"  # light blue
TORQUE_COLOR = "#a78bfa"  # purple
TEXT_COLOR = "#e2e8f0"

# ---------------------------------------------------------------------------
# Screen layout
# ---------------------------------------------------------------------------
SCREEN_CG_Y = 0.5  # CG placed at y=0.5 in data coords
WIND_ARROW_X = -5.0  # left edge for wind arrows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rotate(pts, theta):
    """Rotate Nx2 array of points by *theta* radians about the origin."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T


def _theta(t):
    """Tilt angle in radians at time t."""
    return np.deg2rad(THETA_MAX_DEG) * np.sin(2 * np.pi * t / T_PERIOD)


def _draw_frame(ax, t):
    """Clear and draw a single animation frame at time *t*."""
    ax.clear()
    ax.set_xlim(-5.5, 5.5)
    ax.set_ylim(-6.5, 7.0)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    theta = _theta(t)
    theta_deg = np.rad2deg(theta)

    # --- Dashed vertical reference line ---
    ax.axvline(
        0.0,
        color=TEXT_COLOR,
        ls="--",
        lw=0.8,
        alpha=0.3,
        zorder=1,
    )

    # --- Wind arrows (always horizontal, left side) ---
    arrow_ys = np.linspace(-4.0, 4.5, 6)
    for wy in arrow_ys:
        ax.annotate(
            "",
            xy=(WIND_ARROW_X + 2.0, wy),
            xytext=(WIND_ARROW_X, wy),
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.2",
                color=WIND_COLOR,
                lw=1.5,
                alpha=0.6,
            ),
            zorder=2,
        )
    ax.text(
        WIND_ARROW_X + 1.0,
        arrow_ys[-1] + 0.7,
        "Wind",
        color=WIND_COLOR,
        fontsize=9,
        ha="center",
        alpha=0.8,
    )

    # --- Rocket polygons (rotated about origin = CG) ---
    for pts, zorder in [
        (BODY_PTS, 10),
        (NOSE_PTS, 10),
        (FIN_R_PTS, 9),
        (FIN_L_PTS, 9),
    ]:
        rotated = _rotate(pts, theta)
        patch = plt.Polygon(
            rotated,
            fc=ROCKET_FILL,
            ec=ROCKET_EDGE,
            lw=1.2,
            zorder=zorder,
        )
        ax.add_patch(patch)

    # --- CG marker (fixed at origin) ---
    ax.plot(
        0.0,
        0.0,
        "o",
        color=CG_COLOR,
        ms=10,
        zorder=20,
        markeredgecolor="white",
        mew=0.8,
    )
    ax.text(
        0.45,
        0.35,
        "CG",
        color=CG_COLOR,
        fontsize=11,
        fontweight="bold",
        zorder=20,
    )

    # --- CP marker (rotates with rocket) ---
    cp_screen = _rotate(CP_LOCAL.reshape(1, 2), theta).flatten()
    ax.plot(
        cp_screen[0],
        cp_screen[1],
        "o",
        mfc="none",
        mec=CP_COLOR,
        ms=10,
        mew=2.0,
        zorder=20,
    )
    # Label offset: place text to the right of the CP marker, accounting for rotation
    cp_label_offset = _rotate(np.array([[0.45, -0.3]]), theta).flatten()
    ax.text(
        cp_screen[0] + cp_label_offset[0],
        cp_screen[1] + cp_label_offset[1],
        "CP",
        color=CP_COLOR,
        fontsize=11,
        fontweight="bold",
        zorder=20,
    )

    # --- Moment arm (dotted line CG to CP) ---
    ax.plot(
        [0.0, cp_screen[0]],
        [0.0, cp_screen[1]],
        color=TEXT_COLOR,
        ls=":",
        lw=1.2,
        alpha=0.5,
        zorder=15,
    )

    # --- Aero force vector at CP (perpendicular to rocket axis) ---
    if abs(theta_deg) > 0.5:
        # Rocket axis unit vector (nose direction)
        axis_hat = np.array([-np.sin(theta), np.cos(theta)])
        # Perpendicular to axis (points "right" when upright)
        perp_hat = np.array([np.cos(theta), np.sin(theta)])

        # Force pushes the tail in the direction the nose has tilted.
        # When theta > 0 (nose tilted right), force on the tail pushes left
        # relative to ground = restoring.  The perpendicular component at CP
        # points in the -perp direction when theta > 0.
        force_sign = -np.sign(theta)
        force_mag = abs(np.sin(theta)) * 2.5  # scale for visibility

        f_end = cp_screen + force_sign * perp_hat * force_mag

        ax.annotate(
            "",
            xy=(f_end[0], f_end[1]),
            xytext=(cp_screen[0], cp_screen[1]),
            arrowprops=dict(
                arrowstyle="->,head_width=0.3,head_length=0.25",
                color=FORCE_COLOR,
                lw=2.5,
            ),
            zorder=18,
        )
        # Label
        f_mid = cp_screen + force_sign * perp_hat * (force_mag + 0.5)
        ax.text(
            f_mid[0],
            f_mid[1],
            "F",
            color=FORCE_COLOR,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=18,
        )

    # --- Restoring torque arc at CG ---
    if abs(theta_deg) > 0.5:
        # Torque opposes tilt: CCW when theta > 0, CW when theta < 0
        arc_radius = 1.2
        arc_extent = min(abs(theta_deg) / THETA_MAX_DEG * 60, 60)

        if theta > 0:
            # CCW arc (restoring toward theta=0)
            arc_start = -10
            arc_angles = np.linspace(
                np.deg2rad(arc_start),
                np.deg2rad(arc_start + arc_extent),
                40,
            )
        else:
            # CW arc
            arc_start = 10
            arc_angles = np.linspace(
                np.deg2rad(arc_start),
                np.deg2rad(arc_start - arc_extent),
                40,
            )

        arc_x = arc_radius * np.cos(arc_angles)
        arc_y = arc_radius * np.sin(arc_angles)

        alpha = min(abs(theta_deg) / THETA_MAX_DEG * 0.8 + 0.2, 1.0)
        lw = 1.5 + abs(theta_deg) / THETA_MAX_DEG * 1.5
        ax.plot(arc_x, arc_y, color=TORQUE_COLOR, lw=lw, alpha=alpha, zorder=17)

        # Arrowhead
        dx = arc_x[-1] - arc_x[-2]
        dy = arc_y[-1] - arc_y[-2]
        ax.annotate(
            "",
            xy=(arc_x[-1], arc_y[-1]),
            xytext=(arc_x[-1] - dx * 2, arc_y[-1] - dy * 2),
            arrowprops=dict(
                arrowstyle="->,head_width=0.3,head_length=0.2",
                color=TORQUE_COLOR,
                lw=lw,
                alpha=alpha,
            ),
            zorder=17,
        )
        # Torque label
        label_angle = arc_angles[len(arc_angles) // 2]
        lx = (arc_radius + 0.6) * np.cos(label_angle)
        ly = (arc_radius + 0.6) * np.sin(label_angle)
        ax.text(
            lx,
            ly,
            "\u03c4",
            color=TORQUE_COLOR,
            fontsize=13,
            fontweight="bold",
            ha="center",
            va="center",
            alpha=alpha,
            zorder=17,
        )

    # --- Angle readout ---
    sign = "+" if theta_deg >= 0 else "\u2212"
    ax.text(
        0.0,
        -5.8,
        f"\u03b8 = {sign}{abs(theta_deg):4.1f}\u00b0",
        color=TEXT_COLOR,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        fontfamily="monospace",
    )

    # --- Subtitle ---
    ax.text(
        0.0,
        -6.3,
        "Force at CP (aft of CG) \u2192 restoring torque about CG",
        color=TEXT_COLOR,
        fontsize=9,
        ha="center",
        va="center",
        alpha=0.7,
    )

    # --- Title ---
    ax.text(
        0.0,
        6.5,
        "Rocket Stability: CG forward of CP",
        color=TEXT_COLOR,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)

    times = np.linspace(0, T_PERIOD, N_FRAMES, endpoint=False)

    def animate(frame_idx):
        _draw_frame(ax, times[frame_idx])

    anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS)
    anim.save(str(OUT_PATH), writer=PillowWriter(fps=FPS), dpi=DPI)

    print(f"Saved \u2192 {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
