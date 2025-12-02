#!/usr/bin/env python3
"""
Motor Visualization Script

Plots thrust curves, mass profiles, and performance characteristics
for rocket motors used in training.

Usage:
    python visualize_motor.py --motor estes_c6
    python visualize_motor.py --motor aerotech_f40 --save motor_profile.png
    python visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Try to import the full motor data module
try:
    from thrustcurve_motor_data import MotorData, ThrustCurveParser
    FULL_MOTOR_DATA = True
except ImportError:
    FULL_MOTOR_DATA = False

# Import common motors from our environment
from realistic_spin_rocket import CommonMotors


def get_motor(motor_name: str):
    """Get motor data by name"""
    motors = {
        'estes_c6': CommonMotors.estes_c6,
        'aerotech_f40': CommonMotors.aerotech_f40,
        'cesaroni_g79': CommonMotors.cesaroni_g79,
    }
    
    if motor_name.lower() not in motors:
        raise ValueError(f"Unknown motor: {motor_name}. Available: {list(motors.keys())}")
    
    return motors[motor_name.lower()]()


def plot_motor_profile(motor, save_path: str = None, show: bool = True):
    """
    Create comprehensive motor profile visualization.
    
    Args:
        motor: MotorData object
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{motor.manufacturer} {motor.designation} Motor Profile', 
                 fontsize=16, fontweight='bold')
    
    # Time array for plotting
    t = np.linspace(0, motor.burn_time * 1.1, 500)
    
    # 1. Thrust Curve
    ax1 = axes[0, 0]
    thrust = [motor.get_thrust(ti) for ti in t]
    ax1.plot(t, thrust, 'b-', linewidth=2, label='Thrust')
    ax1.fill_between(t, thrust, alpha=0.3)
    ax1.axhline(motor.average_thrust, color='r', linestyle='--', 
                label=f'Average: {motor.average_thrust:.1f} N')
    ax1.axhline(motor.max_thrust, color='orange', linestyle=':', 
                label=f'Peak: {motor.max_thrust:.1f} N')
    ax1.axvline(motor.burn_time, color='gray', linestyle='--', alpha=0.5,
                label=f'Burnout: {motor.burn_time:.2f} s')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title('Thrust Curve')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, t[-1])
    ax1.set_ylim(0, motor.max_thrust * 1.15)
    
    # 2. Mass Profile
    ax2 = axes[0, 1]
    mass = [motor.get_mass(ti) for ti in t]
    
    # Convert to grams for display
    mass_g = [m * 1000 for m in mass]
    total_mass_g = motor.total_mass * 1000
    case_mass_g = motor.case_mass * 1000
    prop_mass_g = motor.propellant_mass * 1000
    
    ax2.plot(t, mass_g, 'g-', linewidth=2, label='Total Mass')
    ax2.axhline(total_mass_g, color='blue', linestyle=':', alpha=0.5,
                label=f'Initial: {total_mass_g:.1f} g')
    ax2.axhline(case_mass_g, color='red', linestyle=':', alpha=0.5,
                label=f'Case: {case_mass_g:.1f} g')
    ax2.fill_between(t, case_mass_g, mass_g, alpha=0.3, color='orange',
                     label=f'Propellant: {prop_mass_g:.1f} g')
    ax2.axvline(motor.burn_time, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Mass (g)')
    ax2.set_title('Mass Profile')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, t[-1])
    
    # 3. Impulse Accumulation
    ax3 = axes[1, 0]
    dt = t[1] - t[0]
    impulse = np.cumsum(thrust) * dt
    ax3.plot(t, impulse, 'purple', linewidth=2)
    ax3.axhline(motor.total_impulse, color='r', linestyle='--',
                label=f'Total: {motor.total_impulse:.1f} N·s')
    ax3.axvline(motor.burn_time, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Impulse (N·s)')
    ax3.set_title('Impulse Accumulation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, t[-1])
    
    # 4. Motor Classification & Stats
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Determine motor class
    impulse = motor.total_impulse
    if impulse <= 2.5:
        motor_class = 'A'
    elif impulse <= 5:
        motor_class = 'B'
    elif impulse <= 10:
        motor_class = 'C'
    elif impulse <= 20:
        motor_class = 'D'
    elif impulse <= 40:
        motor_class = 'E'
    elif impulse <= 80:
        motor_class = 'F'
    elif impulse <= 160:
        motor_class = 'G'
    elif impulse <= 320:
        motor_class = 'H'
    else:
        motor_class = 'I+'
    
    # Calculate specific impulse
    if motor.propellant_mass > 0:
        isp = motor.total_impulse / (motor.propellant_mass * 9.81)
    else:
        isp = 0
    
    stats_text = f"""
    MOTOR SPECIFICATIONS
    ════════════════════════════════════
    
    Manufacturer:     {motor.manufacturer}
    Designation:      {motor.designation}
    Motor Class:      {motor_class}
    
    ─────────────────────────────────────
    PHYSICAL PROPERTIES
    ─────────────────────────────────────
    Diameter:         {motor.diameter * 1000:.1f} mm
    Length:           {motor.length * 1000:.1f} mm
    Total Mass:       {motor.total_mass * 1000:.1f} g
    Propellant Mass:  {motor.propellant_mass * 1000:.1f} g
    Case Mass:        {motor.case_mass * 1000:.1f} g
    
    ─────────────────────────────────────
    PERFORMANCE
    ─────────────────────────────────────
    Total Impulse:    {motor.total_impulse:.1f} N·s
    Burn Time:        {motor.burn_time:.2f} s
    Average Thrust:   {motor.average_thrust:.1f} N
    Peak Thrust:      {motor.max_thrust:.1f} N
    Specific Impulse: {isp:.0f} s
    
    ─────────────────────────────────────
    RECOMMENDED ROCKET MASS
    ─────────────────────────────────────
    For TWR = 5.0:    {motor.average_thrust / (5.0 * 9.81) * 1000:.0f} g
    For TWR = 3.0:    {motor.average_thrust / (3.0 * 9.81) * 1000:.0f} g
    """
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved motor profile to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def plot_motor_comparison(motor_names: list, save_path: str = None, show: bool = True):
    """
    Compare multiple motors side by side.
    
    Args:
        motor_names: List of motor names to compare
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    motors = [get_motor(name) for name in motor_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Motor Comparison', fontsize=16, fontweight='bold')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(motors)))
    
    # Find max burn time for consistent x-axis
    max_burn = max(m.burn_time for m in motors) * 1.1
    
    for i, (motor, color) in enumerate(zip(motors, colors)):
        label = f"{motor.manufacturer} {motor.designation}"
        t = np.linspace(0, motor.burn_time, 200)
        
        # 1. Thrust curves
        thrust = [motor.get_thrust(ti) for ti in t]
        axes[0, 0].plot(t, thrust, color=color, linewidth=2, label=label)
        
        # 2. Mass profiles
        mass = [motor.get_mass(ti) * 1000 for ti in t]
        axes[0, 1].plot(t, mass, color=color, linewidth=2, label=label)
        
        # 3. Impulse accumulation
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        impulse = np.cumsum(thrust) * dt
        axes[1, 0].plot(t, impulse, color=color, linewidth=2, label=label)
    
    # Format thrust plot
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Thrust (N)')
    axes[0, 0].set_title('Thrust Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim(0, max_burn)
    
    # Format mass plot
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Mass (g)')
    axes[0, 1].set_title('Mass Profiles')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, max_burn)
    
    # Format impulse plot
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Cumulative Impulse (N·s)')
    axes[1, 0].set_title('Impulse Accumulation')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(0, max_burn)
    
    # 4. Bar chart comparison
    ax4 = axes[1, 1]
    x = np.arange(len(motors))
    width = 0.2
    
    impulses = [m.total_impulse for m in motors]
    avg_thrusts = [m.average_thrust for m in motors]
    burn_times = [m.burn_time * 10 for m in motors]  # Scale for visibility
    
    ax4.bar(x - width, impulses, width, label='Total Impulse (N·s)', color='blue', alpha=0.7)
    ax4.bar(x, avg_thrusts, width, label='Avg Thrust (N)', color='green', alpha=0.7)
    ax4.bar(x + width, burn_times, width, label='Burn Time (×10 s)', color='orange', alpha=0.7)
    
    ax4.set_xlabel('Motor')
    ax4.set_ylabel('Value')
    ax4.set_title('Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{m.manufacturer}\n{m.designation}" for m in motors])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved motor comparison to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize rocket motor characteristics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Plot single motor profile
    python visualize_motor.py --motor estes_c6
    
    # Save to file
    python visualize_motor.py --motor aerotech_f40 --save motor_f40.png
    
    # Compare multiple motors
    python visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79
        """
    )
    
    parser.add_argument('--motor', type=str, 
                        help='Motor to visualize (estes_c6, aerotech_f40, cesaroni_g79)')
    parser.add_argument('--compare', nargs='+',
                        help='Compare multiple motors')
    parser.add_argument('--save', type=str,
                        help='Save figure to file')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plot (useful for scripting)')
    
    args = parser.parse_args()
    
    if args.compare:
        plot_motor_comparison(args.compare, save_path=args.save, show=not args.no_show)
    elif args.motor:
        motor = get_motor(args.motor)
        plot_motor_profile(motor, save_path=args.save, show=not args.no_show)
    else:
        # Default: show all motors comparison
        print("No motor specified. Showing comparison of all available motors...")
        plot_motor_comparison(['estes_c6', 'aerotech_f40', 'cesaroni_g79'],
                             save_path=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
