#!/usr/bin/env python3
"""Nose-on roll control animation showing underdamped PID-like tab response.

Produces a looping GIF of a rocket cross-section (looking down the axis from
nose to tail) with 4 fins, 2 of which have actively deflecting tabs that
oppose the current roll rate.  The roll dynamics follow a simple sinusoidal
model that gives a clean loop: spin CW → tabs correct → overshoot CCW →
tabs reverse → back to start.

Usage:
    uv run python visualizations/animate_roll_control.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUT_PATH = Path(__file__).resolve().parent / "outputs" / "roll_control.gif"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Physics parameters (simple sinusoidal loop)
# ---------------------------------------------------------------------------
OMEGA_MAX_DEG = 30.0  # peak roll rate (deg/s)
T_PERIOD = 6.0  # oscillation period (s)
DELTA_MAX_DEG = 30.0  # peak tab deflection (deg)

# ---------------------------------------------------------------------------
# Animation parameters
# ---------------------------------------------------------------------------
FPS = 25
DPI = 85
N_FRAMES = 150  # 6 s × 25 fps — exact loop
FIGSIZE = (5, 5.5)

# ---------------------------------------------------------------------------
# Visual layout (normalised to body_radius = 1)
# ---------------------------------------------------------------------------
BODY_RADIUS = 0.22
FIN_INNER = BODY_RADIUS  # fin starts at body surface
FIN_OUTER = 0.72  # fin tip distance from centre
HINGE_FRAC = 0.50  # fraction along fin where tab region starts (from body)
TAB_LEN = FIN_OUTER - (FIN_INNER + HINGE_FRAC * (FIN_OUTER - FIN_INNER))

# Colours
BG_COLOR = "#0f0f1a"
BODY_COLOR = "#c0c0c0"
FIN_ACTIVE_COLOR = "#3b82f6"  # blue
TAB_COLOR = "#ef4444"  # red
FIN_PASSIVE_COLOR = "#6b7280"  # grey
ARROW_COLOR = "#f59e0b"  # orange
TEXT_COLOR = "#e2e8f0"

# Fin base angles (before rotation) — 0°/90°/180°/270°
FIN_ANGLES_DEG = np.array([0.0, 90.0, 180.0, 270.0])
CONTROLLED_FINS = [0, 2]  # opposite pair
PASSIVE_FINS = [1, 3]

# Line widths
FIN_LW = 4.0

# Tab rectangle geometry
TAB_WIDTH = 0.09  # thickness of tab rectangle (perpendicular to fin)
TAB_MAX_OFFSET = 0.14  # max perpendicular displacement at full deflection


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _omega(t):
    """Roll rate (deg/s) at time t."""
    return OMEGA_MAX_DEG * np.cos(2 * np.pi * t / T_PERIOD)


def theta_rad(t):
    """Roll angle in radians at time t."""
    omega_rad = np.deg2rad(OMEGA_MAX_DEG)
    return (omega_rad * T_PERIOD / (2 * np.pi)) * np.sin(2 * np.pi * t / T_PERIOD)


def delta_deg(t):
    """Tab deflection in degrees — opposes roll rate."""
    return -DELTA_MAX_DEG * _omega(t) / OMEGA_MAX_DEG


def _omega_color(omega_abs):
    """Colour for the omega readout based on magnitude."""
    if omega_abs < 5:
        return "#22c55e"  # green
    elif omega_abs < 15:
        return "#f59e0b"  # orange
    return "#ef4444"  # red


def _draw_arc_arrow(ax, theta_rot, omega_val, radius=0.85):
    """Draw a curved arrow showing spin direction, sized by |omega|."""
    omega_abs = abs(omega_val)
    if omega_abs < 0.5:
        return  # too small to show

    # Arc extent proportional to |omega|, max ~270°
    arc_extent = min(omega_abs / OMEGA_MAX_DEG * 80, 80)
    half = np.deg2rad(arc_extent / 2)
    n_pts = max(int(arc_extent / 2), 20)

    # Direction: positive omega = CCW in our convention
    if omega_val > 0:
        angles = np.linspace(-half, half, n_pts) + theta_rot
    else:
        angles = np.linspace(half, -half, n_pts) + theta_rot

    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    alpha = min(omega_abs / OMEGA_MAX_DEG * 0.9 + 0.1, 1.0)
    lw = 1.5 + omega_abs / OMEGA_MAX_DEG * 1.5

    ax.plot(xs, ys, color=ARROW_COLOR, lw=lw, alpha=alpha, solid_capstyle="round")

    # Arrowhead at the end of the arc
    dx = xs[-1] - xs[-2]
    dy = ys[-1] - ys[-2]
    ax.annotate(
        "",
        xy=(xs[-1], ys[-1]),
        xytext=(xs[-1] - dx * 2, ys[-1] - dy * 2),
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.25",
            color=ARROW_COLOR,
            lw=lw,
            alpha=alpha,
        ),
    )


def _draw_frame(ax, t):
    """Draw a single animation frame at time *t*."""
    ax.clear()
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.25, 1.15)
    ax.set_aspect("equal")
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    theta = theta_rad(t)
    omega_val = _omega(t)
    delta_val = delta_deg(t)

    # --- Body tube (filled circle) ---
    body = plt.Circle((0, 0), BODY_RADIUS, fc=BODY_COLOR, ec="white", lw=1.2, zorder=5)
    ax.add_patch(body)

    # --- Fins ---
    for i, base_deg in enumerate(FIN_ANGLES_DEG):
        angle_rad = np.deg2rad(base_deg) + theta  # rotate with body

        # Fin root (body surface) and tip
        x0 = FIN_INNER * np.cos(angle_rad)
        y0 = FIN_INNER * np.sin(angle_rad)

        if i in CONTROLLED_FINS:
            # Full fin line: body surface → tip
            x1 = FIN_OUTER * np.cos(angle_rad)
            y1 = FIN_OUTER * np.sin(angle_rad)
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=FIN_ACTIVE_COLOR,
                lw=FIN_LW,
                solid_capstyle="round",
                zorder=4,
            )

            # Tab: filled region from the fin centerline out to the
            # deflected tab edge, spanning the outer portion of the fin.
            hinge_r = FIN_INNER + HINGE_FRAC * (FIN_OUTER - FIN_INNER)
            tab_half_len = TAB_LEN / 2
            tab_center_r = hinge_r + tab_half_len
            tab_perp = (delta_val / DELTA_MAX_DEG) * TAB_MAX_OFFSET

            # Unit vectors: radial (along fin) and perpendicular
            r_hat = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            p_hat = np.array([-np.sin(angle_rad), np.cos(angle_rad)])

            # Far edge of the tab (away from fin centerline)
            if abs(tab_perp) < 1e-6:
                far_perp = TAB_WIDTH / 2
            elif tab_perp > 0:
                far_perp = tab_perp + TAB_WIDTH / 2
            else:
                far_perp = tab_perp - TAB_WIDTH / 2

            # Polygon from fin centerline (perp=0) to far tab edge
            corners = []
            for dr, dp in [
                (-tab_half_len, 0.0),
                (tab_half_len, 0.0),
                (tab_half_len, far_perp),
                (-tab_half_len, far_perp),
            ]:
                pt = (tab_center_r + dr) * r_hat + dp * p_hat
                corners.append(pt)

            tab_patch = plt.Polygon(
                corners,
                fc=TAB_COLOR,
                ec=TAB_COLOR,
                lw=0.5,
                zorder=6,
            )
            ax.add_patch(tab_patch)
        else:
            # Passive fin — straight line, no tab
            x1 = FIN_OUTER * np.cos(angle_rad)
            y1 = FIN_OUTER * np.sin(angle_rad)
            ax.plot(
                [x0, x1],
                [y0, y1],
                color=FIN_PASSIVE_COLOR,
                lw=FIN_LW,
                solid_capstyle="round",
                zorder=4,
            )

    # --- Rotation arrow ---
    _draw_arc_arrow(ax, theta, omega_val)

    # --- Text readouts ---
    omega_abs = abs(omega_val)
    omega_col = _omega_color(omega_abs)
    sign = "+" if omega_val >= 0 else "\u2212"
    ax.text(
        0.0,
        -1.05,
        f"\u03c9 = {sign}{omega_abs:4.1f} \u00b0/s",
        color=omega_col,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
        fontfamily="monospace",
        transform=ax.transData,
    )
    ax.text(
        0.0,
        -1.18,
        f"\u03b4 = {delta_val:+5.1f}\u00b0",
        color=TEXT_COLOR,
        fontsize=10,
        ha="center",
        va="center",
        fontfamily="monospace",
        transform=ax.transData,
    )

    # --- Legend ---
    legend_x, legend_y = 0.02, 0.98
    legend_items = [
        (FIN_ACTIVE_COLOR, "Fin (active)"),
        (TAB_COLOR, "Tab"),
        (FIN_PASSIVE_COLOR, "Fin (passive)"),
        (ARROW_COLOR, "Spin direction"),
    ]
    for idx, (col, label) in enumerate(legend_items):
        ax.plot(
            legend_x,
            legend_y - idx * 0.08,
            "s",
            color=col,
            ms=6,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.text(
            legend_x + 0.05,
            legend_y - idx * 0.08,
            label,
            color=TEXT_COLOR,
            fontsize=7,
            va="center",
            ha="left",
            transform=ax.transAxes,
            clip_on=False,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    fig, ax = plt.subplots(figsize=FIGSIZE, facecolor=BG_COLOR)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)

    times = np.linspace(0, T_PERIOD, N_FRAMES, endpoint=False)

    def animate(frame_idx):
        _draw_frame(ax, times[frame_idx])

    anim = FuncAnimation(fig, animate, frames=N_FRAMES, interval=1000 // FPS)
    anim.save(str(OUT_PATH), writer=PillowWriter(fps=FPS), dpi=DPI)

    print(f"Saved \u2192 {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
