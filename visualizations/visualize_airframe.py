#!/usr/bin/env python3
"""
Static side-profile visualization of a rocket airframe.

Loads airframe geometry from YAML and draws a to-scale side profile
using matplotlib, with each component rendered from its actual
position, length, and diameter fields.

Usage:
    uv run python visualizations/visualize_airframe.py --rocket j800
    uv run python visualizations/visualize_airframe.py --rocket estes_alpha --save viz.png
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patheffects as patheffects

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airframe.airframe import RocketAirframe
from airframe.components import (
    NoseCone,
    BodyTube,
    TrapezoidFinSet,
    MotorMount,
    MassObject,
)

# --- Dark theme colors (matching animate_stability.py) ---
BG_COLOR = "#0f0f1a"
BODY_COLOR = "#2a2a3a"
BODY_EDGE = "#c0c0c0"
CG_COLOR = "#22c55e"
NOSE_COLOR = "#3a3a4a"
FIN_COLOR = "#4a4a5a"
FIN_EDGE = "#c0c0c0"
MOTOR_MOUNT_COLOR = "#1a1a2a"
MOTOR_MOUNT_EDGE = "#808080"
TAB_COLOR = "#ef4444"
TAB_EDGE = "#ff6b6b"
MASS_OBJ_COLOR = "#60a5fa"
DIM_COLOR = "#a78bfa"
TEXT_COLOR = "#e2e8f0"
TITLE_COLOR = "#e2e8f0"


def get_airframe_config(rocket: str):
    """Return airframe YAML path and config path for a given rocket."""
    if rocket == "j800":
        return (
            "configs/airframes/j800_75mm.yaml",
            "configs/aerotech_j800_wind.yaml",
        )
    else:
        return (
            "configs/airframes/estes_alpha.yaml",
            "configs/estes_c6_sac_wind.yaml",
        )


def ogive_profile(length, base_radius, n_points=80):
    """
    Generate ogive nose cone profile points.

    Returns arrays of (x, y) where x=0 is the tip and x=length is the base.
    y values are the radius at each x position.
    """
    rho = (base_radius**2 + length**2) / (2 * base_radius)
    x = np.linspace(0, length, n_points)
    y = np.sqrt(rho**2 - (length - x) ** 2) - (rho - base_radius)
    # Clamp any tiny numerical negatives at the tip
    y = np.maximum(y, 0.0)
    return x, y


def draw_nose_cone(ax, comp, body_radius):
    """Draw an ogive nose cone as a filled polygon."""
    x_profile, y_profile = ogive_profile(comp.length, comp.base_diameter / 2)
    x_offset = comp.position

    # Upper and lower profile (symmetric about centerline)
    x_pts = np.concatenate([x_profile + x_offset, (x_profile + x_offset)[::-1]])
    y_pts = np.concatenate([y_profile, -y_profile[::-1]])

    poly = Polygon(
        list(zip(x_pts, y_pts)),
        closed=True,
        facecolor=NOSE_COLOR,
        edgecolor=BODY_EDGE,
        linewidth=1.2,
        zorder=3,
    )
    ax.add_patch(poly)


def draw_body_tube(ax, comp):
    """Draw a body tube as a rectangle."""
    r = comp.outer_diameter / 2
    rect = Rectangle(
        (comp.position, -r),
        comp.length,
        comp.outer_diameter,
        facecolor=BODY_COLOR,
        edgecolor=BODY_EDGE,
        linewidth=1.2,
        zorder=2,
    )
    ax.add_patch(rect)


def draw_motor_mount(ax, comp):
    """Draw motor mount inside the body as a smaller rectangle."""
    r = comp.outer_diameter / 2
    rect = Rectangle(
        (comp.position, -r),
        comp.length,
        comp.outer_diameter,
        facecolor=MOTOR_MOUNT_COLOR,
        edgecolor=MOTOR_MOUNT_EDGE,
        linewidth=0.8,
        linestyle="--",
        zorder=4,
    )
    ax.add_patch(rect)


def draw_fin(ax, comp, body_radius, side=1, tab_chord_frac=0.25, tab_span_frac=0.5):
    """
    Draw a single trapezoidal fin on one side.

    side: +1 for upper, -1 for lower.
    Also draws the tab region if tab_chord_frac > 0.
    """
    root = comp.root_chord
    tip = comp.tip_chord
    span = comp.span
    sweep = comp.sweep_length
    x0 = comp.position

    # Fin corner points (from root leading edge, clockwise for upper fin)
    # Root leading edge, root trailing edge, tip trailing edge, tip leading edge
    pts = [
        (x0, side * body_radius),
        (x0 + root, side * body_radius),
        (x0 + sweep + tip, side * (body_radius + span)),
        (x0 + sweep, side * (body_radius + span)),
    ]

    poly = Polygon(
        pts,
        closed=True,
        facecolor=FIN_COLOR,
        edgecolor=FIN_EDGE,
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(poly)

    # Draw tab region (trailing edge portion of fin)
    if tab_chord_frac > 0:
        tab_root_chord = root * tab_chord_frac
        tab_tip_chord = tip * tab_chord_frac
        tab_span = span * tab_span_frac

        # Tab starts at trailing edge and goes inward
        # Root side of tab
        tab_root_le_x = x0 + root - tab_root_chord
        tab_root_te_x = x0 + root
        # Tip side: interpolate at tab_span_frac of the way up
        # The fin leading edge at tab_span height
        fin_le_at_tab = x0 + sweep * tab_span_frac
        fin_te_at_tab = fin_le_at_tab + root + (tip - root) * tab_span_frac
        # Actually, the trailing edge of the fin at height h is:
        # TE(h) = x0 + root + (sweep + tip - root) * (h/span)
        # Wait, let me recalculate properly.
        # At height h from root: LE(h) = x0 + sweep*(h/span), TE(h) = x0 + root + (sweep+tip-root)*(h/span)
        # Simplify: TE(h) = x0 + root + (sweep + tip - root) * (h/span)
        h_tab = span * tab_span_frac
        te_at_tab = x0 + root + (sweep + tip - root) * (h_tab / span)
        chord_at_tab = root + (tip - root) * (h_tab / span)
        tab_le_at_tab = te_at_tab - chord_at_tab * tab_chord_frac

        tab_pts = [
            (tab_root_le_x, side * body_radius),
            (tab_root_te_x, side * body_radius),
            (te_at_tab, side * (body_radius + h_tab)),
            (tab_le_at_tab, side * (body_radius + h_tab)),
        ]

        tab_poly = Polygon(
            tab_pts,
            closed=True,
            facecolor=TAB_COLOR,
            edgecolor=TAB_EDGE,
            linewidth=0.8,
            alpha=0.5,
            zorder=6,
        )
        ax.add_patch(tab_poly)


def draw_mass_object(ax, comp, body_radius):
    """Draw a mass object as a labeled bracket region inside the body."""
    x_start = comp.position
    x_end = comp.position + comp.length
    # Draw bracket lines
    bracket_y = body_radius * 0.6
    bracket_lw = 1.0

    ax.plot(
        [x_start, x_start],
        [-bracket_y, bracket_y],
        color=MASS_OBJ_COLOR,
        linewidth=bracket_lw,
        alpha=0.7,
        zorder=7,
    )
    ax.plot(
        [x_end, x_end],
        [-bracket_y, bracket_y],
        color=MASS_OBJ_COLOR,
        linewidth=bracket_lw,
        alpha=0.7,
        zorder=7,
    )
    ax.plot(
        [x_start, x_end],
        [bracket_y, bracket_y],
        color=MASS_OBJ_COLOR,
        linewidth=bracket_lw,
        alpha=0.7,
        linestyle=":",
        zorder=7,
    )
    ax.plot(
        [x_start, x_end],
        [-bracket_y, -bracket_y],
        color=MASS_OBJ_COLOR,
        linewidth=bracket_lw,
        alpha=0.7,
        linestyle=":",
        zorder=7,
    )

    # Label
    mid_x = (x_start + x_end) / 2
    ax.text(
        mid_x,
        0,
        comp.name,
        ha="center",
        va="center",
        fontsize=6,
        color=MASS_OBJ_COLOR,
        fontweight="bold",
        zorder=8,
        path_effects=[
            patheffects.withStroke(linewidth=2, foreground=BG_COLOR),
        ],
    )


def draw_cg_marker(ax, cg_pos, body_radius):
    """Draw CG position marker."""
    marker_y = body_radius * 1.8
    ax.plot(
        cg_pos,
        marker_y,
        marker="v",
        markersize=10,
        color=CG_COLOR,
        zorder=10,
    )
    ax.text(
        cg_pos,
        marker_y + body_radius * 0.5,
        "CG",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
        color=CG_COLOR,
        zorder=10,
    )


def draw_dimension(ax, x1, x2, y, label, offset=0.0):
    """Draw a dimension annotation line with label."""
    arrow_style = "<->"
    ax.annotate(
        "",
        xy=(x2, y + offset),
        xytext=(x1, y + offset),
        arrowprops=dict(
            arrowstyle=arrow_style,
            color=DIM_COLOR,
            lw=0.8,
        ),
        zorder=9,
    )
    mid_x = (x1 + x2) / 2
    ax.text(
        mid_x,
        y + offset + 0.003,
        label,
        ha="center",
        va="bottom",
        fontsize=7,
        color=DIM_COLOR,
        fontweight="bold",
        zorder=9,
    )


def visualize_airframe(airframe, config_path=None, annotate=True):
    """
    Create a side-profile visualization of the rocket airframe.

    Args:
        airframe: RocketAirframe instance
        config_path: Path to training config (for tab parameters)
        annotate: Whether to add dimension annotations

    Returns:
        (fig, ax) matplotlib figure and axes
    """
    # Load tab parameters from config if available
    tab_chord_frac = 0.25
    tab_span_frac = 0.5
    if config_path:
        from rocket_config import load_config

        config = load_config(config_path)
        tab_chord_frac = config.physics.tab_chord_fraction
        tab_span_frac = config.physics.tab_span_fraction

    body_radius = airframe.body_radius
    total_length = airframe.total_length

    # Figure sizing: scale to rocket proportions
    aspect = total_length / (body_radius * 2)
    fig_width = max(10, min(16, aspect * 2))
    fig_height = max(4, fig_width / aspect * 2.5)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Draw components in order
    for comp in airframe.components:
        if isinstance(comp, NoseCone):
            draw_nose_cone(ax, comp, body_radius)
        elif isinstance(comp, MotorMount):
            draw_motor_mount(ax, comp)
        elif isinstance(comp, BodyTube):
            draw_body_tube(ax, comp)
        elif isinstance(comp, TrapezoidFinSet):
            # Draw fins on both visible sides
            draw_fin(
                ax,
                comp,
                body_radius,
                side=1,
                tab_chord_frac=tab_chord_frac,
                tab_span_frac=tab_span_frac,
            )
            draw_fin(
                ax,
                comp,
                body_radius,
                side=-1,
                tab_chord_frac=tab_chord_frac,
                tab_span_frac=tab_span_frac,
            )
        elif isinstance(comp, MassObject):
            draw_mass_object(ax, comp, body_radius)

    # CG marker
    draw_cg_marker(ax, airframe.cg_position, body_radius)

    # Annotations
    if annotate:
        dim_y_total = -(body_radius * 2.5)
        dim_y_diam = total_length + body_radius * 0.5

        # Total length dimension
        draw_dimension(
            ax,
            0,
            total_length,
            dim_y_total,
            f"Total length: {total_length*1000:.0f} mm",
        )

        # Vertical diameter dimension line (right side of rocket)
        ax.annotate(
            "",
            xy=(dim_y_diam, body_radius),
            xytext=(dim_y_diam, -body_radius),
            arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=0.8),
            zorder=9,
        )
        ax.text(
            dim_y_diam + 0.01,
            0,
            f"{airframe.body_diameter*1000:.0f} mm dia.",
            ha="left",
            va="center",
            fontsize=7,
            color=DIM_COLOR,
            fontweight="bold",
            rotation=90,
            zorder=9,
        )

        # Tab region label
        fin_set = airframe.get_fin_set()
        if fin_set:
            tab_label_x = fin_set.position + fin_set.root_chord * 0.85
            tab_label_y = body_radius + fin_set.span * 0.2
            ax.text(
                tab_label_x,
                tab_label_y,
                f"Tab\n({tab_chord_frac*100:.0f}% chord\n {tab_span_frac*100:.0f}% span)",
                ha="center",
                va="bottom",
                fontsize=6,
                color=TAB_COLOR,
                fontweight="bold",
                zorder=10,
                path_effects=[
                    patheffects.withStroke(linewidth=2, foreground=BG_COLOR),
                ],
            )

    # Title and info
    info_lines = [
        f"Dry mass: {airframe.dry_mass*1000:.0f} g",
        f"CG: {airframe.cg_position*1000:.0f} mm from nose",
    ]
    ax.set_title(
        f"{airframe.name}",
        fontsize=14,
        fontweight="bold",
        color=TITLE_COLOR,
        pad=12,
    )
    ax.text(
        0.99,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=TEXT_COLOR,
        fontfamily="monospace",
        zorder=10,
        path_effects=[
            patheffects.withStroke(linewidth=2, foreground=BG_COLOR),
        ],
    )

    # Legend for tab color
    ax.plot([], [], color=TAB_COLOR, linewidth=4, alpha=0.5, label="Control tab region")
    ax.plot(
        [],
        [],
        color=CG_COLOR,
        marker="v",
        linestyle="None",
        markersize=8,
        label="Center of gravity",
    )
    ax.plot(
        [],
        [],
        color=MASS_OBJ_COLOR,
        linewidth=1.5,
        linestyle=":",
        label="Internal mass object",
    )
    ax.legend(
        loc="lower right",
        fontsize=7,
        facecolor=BG_COLOR,
        edgecolor="#404060",
        labelcolor=TEXT_COLOR,
        framealpha=0.9,
    )

    # Axis settings
    margin_x = total_length * 0.05
    margin_y = (
        body_radius + (fin_set.span if (fin_set := airframe.get_fin_set()) else 0)
    ) * 1.5
    ax.set_xlim(-margin_x, total_length + margin_x)
    ax.set_ylim(-margin_y, margin_y)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig, ax


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rocket airframe side profile"
    )
    parser.add_argument(
        "--rocket",
        type=str,
        default="j800",
        choices=["estes_alpha", "j800"],
        help="Rocket to visualize (default: j800)",
    )
    parser.add_argument(
        "--save",
        type=str,
        nargs="?",
        const=None,
        default="__no_save__",
        help="Save to file (default path if no argument given)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying the plot window",
    )
    parser.add_argument(
        "--no-annotate",
        action="store_true",
        help="Disable dimension annotations",
    )
    args = parser.parse_args()

    airframe_path, config_path = get_airframe_config(args.rocket)
    print(f"Loading airframe: {airframe_path}")

    airframe = RocketAirframe.load_yaml(airframe_path)
    print(airframe.summary())

    fig, ax = visualize_airframe(
        airframe,
        config_path=config_path,
        annotate=not args.no_annotate,
    )

    # Handle save
    if args.save != "__no_save__":
        if args.save is None:
            # Default path
            out_dir = Path("visualizations/outputs")
            out_dir.mkdir(parents=True, exist_ok=True)
            save_path = out_dir / f"{args.rocket}_airframe.png"
        else:
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            save_path,
            dpi=200,
            facecolor=BG_COLOR,
            bbox_inches="tight",
            pad_inches=0.2,
        )
        print(f"Saved to {save_path}")

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
