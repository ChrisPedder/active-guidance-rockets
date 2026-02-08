#!/usr/bin/env python3
"""
PID Controller for Rocket Roll Stabilization

A simple PID controller that mimics the Arduino implementation for baseline comparison
with the RL agent.

Usage:
    # Evaluate PID controller
    uv run python pid_controller.py --config configs/estes_c6_easy.yaml --n-episodes 50

    # Compare with RL agent
    uv run python pid_controller.py --config configs/estes_c6_easy.yaml --compare models/best_model.zip
"""

import numpy as np
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

from rocket_config import load_config
from spin_stabilized_control_env import RocketConfig
from realistic_spin_rocket import RealisticMotorRocket


@dataclass
class PIDConfig:
    """PID controller configuration"""

    Cprop: float = 0.005208  # Proportional gain (optimized for 30 deg max deflection)
    Cint: float = 0.000324  # Integral gain (optimized for 30 deg max deflection)
    Cderiv: float = 0.016524  # Derivative gain (optimized for 30 deg max deflection)
    max_roll_rate: float = 100.0  # Roll rate clamp (deg/s)
    max_deflection: float = 30.0  # Max servo deflection from neutral (deg)
    launch_accel_threshold: float = 20.0  # Launch detection threshold (m/s²)
    q_ref: float = 500.0  # Reference dynamic pressure for gain scheduling (Pa)


class PIDController:
    """
    PID controller for rocket roll stabilization.

    Mimics the Arduino implementation:
    - Launch detection based on acceleration
    - Stores launch orientation as target
    - P: Proportional to orientation error
    - I: Integral of orientation error
    - D: Based on roll rate (derivative of orientation)
    """

    def __init__(self, config: PIDConfig = None, use_observations: bool = False):
        self.config = config or PIDConfig()
        self.use_observations = use_observations
        self.reset()

    def reset(self):
        """Reset controller state"""
        self.launch_detected = False
        self.launch_orient = 0.0
        self.integ_error = 0.0
        self.target_orient = 0.0

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """
        Compute control action based on observation.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        if self.use_observations:
            # IMU mode: roll angle from obs (not affected by gyro noise),
            # roll rate from info dict (noisy but current — the IMU wrapper
            # applies gyro noise to info['roll_rate_deg_s'], avoiding the
            # obs vector's sensor_delay_steps which is an RL-specific feature).
            roll_angle_rad = obs[2] if len(obs) > 2 else 0.0
            roll_rate = info.get(
                "roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0
            )
            roll_angle_deg = np.degrees(roll_angle_rad)

            # In obs-based mode, assume launched immediately (motor fires at t=0)
            if not self.launch_detected:
                self.launch_detected = True
                self.launch_orient = roll_angle_deg
                self.target_orient = self.launch_orient
        else:
            # Ground-truth mode: PID reads exact state from info dict
            roll_angle_rad = info.get("roll_angle_rad", 0.0)
            roll_rate = info.get("roll_rate_deg_s", 0.0)
            accel = info.get("vertical_acceleration_ms2", 0.0)

            # Convert roll angle to degrees for consistency with Arduino code
            roll_angle_deg = np.degrees(roll_angle_rad)

            # Launch detection
            if not self.launch_detected:
                if accel > self.config.launch_accel_threshold:
                    self.launch_detected = True
                    self.launch_orient = roll_angle_deg
                    self.target_orient = self.launch_orient
                else:
                    # Before launch, output zero
                    return np.array([0.0], dtype=np.float32)

        # Clamp roll rate input
        roll_rate_clamped = np.clip(
            roll_rate, -self.config.max_roll_rate, self.config.max_roll_rate
        )

        # Calculate errors
        prop_error = roll_angle_deg - self.target_orient

        # Normalize angle error to [-180, 180]
        while prop_error > 180:
            prop_error -= 360
        while prop_error < -180:
            prop_error += 360

        # Integral error accumulation
        integ_error_new = prop_error * dt
        self.integ_error += integ_error_new

        # Anti-windup: clamp integral
        max_integ = self.config.max_deflection / (self.config.Cint + 1e-6)
        self.integ_error = np.clip(self.integ_error, -max_integ, max_integ)

        # PID terms
        cmd_p = prop_error * self.config.Cprop
        cmd_i = self.integ_error * self.config.Cint
        cmd_d = roll_rate_clamped * self.config.Cderiv

        # Total command (in degrees of deflection)
        servo_cmd = cmd_p + cmd_i + cmd_d

        # Clamp to max deflection
        servo_cmd = np.clip(
            servo_cmd, -self.config.max_deflection, self.config.max_deflection
        )

        # Normalize to [-1, 1] for environment
        # Negate: PID computes deflection in roll-error direction,
        # but environment action creates torque in the same direction,
        # so we need to oppose the error.
        action = -servo_cmd / self.config.max_deflection

        return np.array([action], dtype=np.float32)


class GainScheduledPIDController:
    """
    Gain-scheduled PID controller for rocket roll stabilization.

    Scales Kp and Kd gains with dynamic pressure to maintain consistent
    loop gain as control effectiveness varies during flight. Control
    effectiveness is proportional to q * tanh(q/200), so we divide by
    this to keep effective gains constant.

    Ki is NOT scaled — integral action should remain consistent regardless
    of flight phase.
    """

    def __init__(self, config: PIDConfig = None, use_observations: bool = False):
        self.config = config or PIDConfig()
        self.use_observations = use_observations
        # Pre-compute reference effectiveness for gain scheduling
        q_ref = self.config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)
        self.reset()

    def reset(self):
        """Reset controller state."""
        self.launch_detected = False
        self.launch_orient = 0.0
        self.integ_error = 0.0
        self.target_orient = 0.0

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure.

        Returns q_ref_effectiveness / current_effectiveness, so that
        the product (gain * effectiveness) stays constant across flight.

        Clamped to [0.5, 5.0] to prevent extreme scaling at very low
        or very high dynamic pressure.
        """
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            # Below ~0.1 Pa there's no aerodynamic control at all;
            # return max scale so the controller tries its hardest,
            # but this is physically a no-control region.
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """
        Compute control action with gain scheduling.

        Args:
            obs: Observation from environment (obs[5] = dynamic pressure)
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        # Always use info dict for gain scheduling q — dynamic pressure
        # is not a gyro measurement and should not be subject to sensor delay.
        q = info.get("dynamic_pressure_Pa", obs[5] if len(obs) > 5 else 0.0)

        if self.use_observations:
            # IMU mode: roll angle from obs, roll rate from info (noisy
            # but current — bypasses sensor_delay_steps).
            roll_angle_rad = obs[2] if len(obs) > 2 else 0.0
            roll_rate = info.get(
                "roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0
            )
            roll_angle_deg = np.degrees(roll_angle_rad)

            if not self.launch_detected:
                self.launch_detected = True
                self.launch_orient = roll_angle_deg
                self.target_orient = self.launch_orient
        else:
            roll_angle_rad = info.get("roll_angle_rad", 0.0)
            roll_rate = info.get("roll_rate_deg_s", 0.0)
            accel = info.get("vertical_acceleration_ms2", 0.0)

            roll_angle_deg = np.degrees(roll_angle_rad)

            if not self.launch_detected:
                if accel > self.config.launch_accel_threshold:
                    self.launch_detected = True
                    self.launch_orient = roll_angle_deg
                    self.target_orient = self.launch_orient
                else:
                    return np.array([0.0], dtype=np.float32)

        # Gain scheduling: scale Kp and Kd to compensate for varying
        # control effectiveness (proportional to q * tanh(q/200))
        scale = self._gain_scale(q)

        # Clamp roll rate input
        roll_rate_clamped = np.clip(
            roll_rate, -self.config.max_roll_rate, self.config.max_roll_rate
        )

        # Calculate errors
        prop_error = roll_angle_deg - self.target_orient
        while prop_error > 180:
            prop_error -= 360
        while prop_error < -180:
            prop_error += 360

        # Integral error accumulation (NOT scaled by gain schedule)
        integ_error_new = prop_error * dt
        self.integ_error += integ_error_new
        max_integ = self.config.max_deflection / (self.config.Cint + 1e-6)
        self.integ_error = np.clip(self.integ_error, -max_integ, max_integ)

        # PID terms — Kp and Kd are scaled, Ki is not
        cmd_p = prop_error * self.config.Cprop * scale
        cmd_i = self.integ_error * self.config.Cint
        cmd_d = roll_rate_clamped * self.config.Cderiv * scale

        servo_cmd = cmd_p + cmd_i + cmd_d
        servo_cmd = np.clip(
            servo_cmd, -self.config.max_deflection, self.config.max_deflection
        )

        action = -servo_cmd / self.config.max_deflection
        return np.array([action], dtype=np.float32)


class LeadCompensatedGSPIDController:
    """
    Gain-scheduled PID with a lead compensator on the derivative channel.

    Adds phase lead at the spin frequency band (6-30 rad/s) to counteract
    the 90-degree phase lag of PID integral action against sinusoidal wind
    disturbances.

    The lead compensator transfer function is:
        H(s) = (s + z) / (s + p)   with z < p
    which adds phase lead between z and p. Maximum phase lead occurs at
    sqrt(z * p) rad/s.

    Default design: z=5, p=50 gives ~45 deg lead at ~16 rad/s (typical
    spin frequency), with 10x gain boost at high frequencies.
    """

    def __init__(
        self,
        config: PIDConfig = None,
        use_observations: bool = False,
        lead_zero: float = 5.0,
        lead_pole: float = 50.0,
    ):
        self.config = config or PIDConfig()
        self.use_observations = use_observations

        # Pre-compute reference effectiveness for gain scheduling
        q_ref = self.config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)

        # Discretize the lead compensator via Tustin (bilinear) transform.
        # Continuous: H(s) = (s + z) / (s + p)
        # Tustin substitution: s = (2/T) * (1 - z^-1) / (1 + z^-1)
        # where T = dt = 0.01 (100 Hz).
        #
        # H(z) = [(2/T + z) + (z - 2/T)*z^-1] / [(2/T + p) + (p - 2/T)*z^-1]
        #
        # Normalized: H(z) = (b0 + b1*z^-1) / (1 + a1*z^-1)
        T = 0.01  # 100 Hz control rate
        self._lead_b0 = (2.0 / T + lead_zero) / (2.0 / T + lead_pole)
        self._lead_b1 = (lead_zero - 2.0 / T) / (2.0 / T + lead_pole)
        self._lead_a1 = (lead_pole - 2.0 / T) / (2.0 / T + lead_pole)

        # Compensate the DC gain of the lead filter so that the D channel
        # gain at DC remains unchanged. DC gain of H(z) = (b0+b1)/(1+a1) = z/p.
        self._lead_dc_gain = lead_zero / lead_pole
        self._lead_dc_inv = 1.0 / self._lead_dc_gain if self._lead_dc_gain > 0 else 1.0

        self.reset()

    def reset(self):
        """Reset controller state."""
        self.launch_detected = False
        self.launch_orient = 0.0
        self.integ_error = 0.0
        self.target_orient = 0.0
        # Lead filter state (one delay element for first-order IIR)
        self._lead_x_prev = 0.0  # Previous input to lead filter
        self._lead_y_prev = 0.0  # Previous output of lead filter

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure."""
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def _lead_filter(self, x: float) -> float:
        """Apply the discrete lead compensator to input x.

        Returns filtered output with DC gain normalized to 1.0.
        """
        y = (
            self._lead_b0 * x
            + self._lead_b1 * self._lead_x_prev
            - self._lead_a1 * self._lead_y_prev
        )
        self._lead_x_prev = x
        self._lead_y_prev = y
        # Normalize so DC gain = 1 (the phase lead is what we want, not gain change)
        return y * self._lead_dc_inv

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action with gain scheduling + lead compensation."""
        q = info.get("dynamic_pressure_Pa", obs[5] if len(obs) > 5 else 0.0)

        if self.use_observations:
            # IMU mode: roll angle from obs, roll rate from info (noisy
            # but current — bypasses sensor_delay_steps).
            roll_angle_rad = obs[2] if len(obs) > 2 else 0.0
            roll_rate = info.get(
                "roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0
            )
            roll_angle_deg = np.degrees(roll_angle_rad)

            if not self.launch_detected:
                self.launch_detected = True
                self.launch_orient = roll_angle_deg
                self.target_orient = self.launch_orient
        else:
            roll_angle_rad = info.get("roll_angle_rad", 0.0)
            roll_rate = info.get("roll_rate_deg_s", 0.0)
            accel = info.get("vertical_acceleration_ms2", 0.0)

            roll_angle_deg = np.degrees(roll_angle_rad)

            if not self.launch_detected:
                if accel > self.config.launch_accel_threshold:
                    self.launch_detected = True
                    self.launch_orient = roll_angle_deg
                    self.target_orient = self.launch_orient
                else:
                    return np.array([0.0], dtype=np.float32)

        scale = self._gain_scale(q)

        roll_rate_clamped = np.clip(
            roll_rate, -self.config.max_roll_rate, self.config.max_roll_rate
        )

        prop_error = roll_angle_deg - self.target_orient
        while prop_error > 180:
            prop_error -= 360
        while prop_error < -180:
            prop_error += 360

        # Integral (NOT scaled)
        integ_error_new = prop_error * dt
        self.integ_error += integ_error_new
        max_integ = self.config.max_deflection / (self.config.Cint + 1e-6)
        self.integ_error = np.clip(self.integ_error, -max_integ, max_integ)

        # PID terms — D channel goes through lead compensator
        cmd_p = prop_error * self.config.Cprop * scale
        cmd_i = self.integ_error * self.config.Cint
        # Apply lead filter to roll rate before multiplying by Kd
        rate_filtered = self._lead_filter(roll_rate_clamped)
        cmd_d = rate_filtered * self.config.Cderiv * scale

        servo_cmd = cmd_p + cmd_i + cmd_d
        servo_cmd = np.clip(
            servo_cmd, -self.config.max_deflection, self.config.max_deflection
        )

        action = -servo_cmd / self.config.max_deflection
        return np.array([action], dtype=np.float32)


def create_env(config):
    """Create environment from config"""
    airframe = config.physics.resolve_airframe()

    rocket_config = RocketConfig(
        max_tab_deflection=config.physics.max_tab_deflection,
        tab_chord_fraction=config.physics.tab_chord_fraction,
        tab_span_fraction=config.physics.tab_span_fraction,
        num_controlled_fins=getattr(config.physics, "num_controlled_fins", 2),
        disturbance_scale=getattr(config.physics, "disturbance_scale", 0.0001),
        initial_spin_std=getattr(config.physics, "initial_spin_std", 15.0),
        damping_scale=getattr(config.physics, "damping_scale", 2.0),
        max_roll_rate=getattr(config.physics, "max_roll_rate", 720.0),
        max_episode_time=getattr(config.physics, "max_episode_time", 15.0),
        dt=getattr(config.environment, "dt", 0.01),
    )

    motor_config_dict = {
        "name": config.motor.name,
        "manufacturer": config.motor.manufacturer,
        "designation": config.motor.designation,
        "total_impulse_Ns": config.motor.total_impulse_Ns,
        "avg_thrust_N": config.motor.avg_thrust_N,
        "max_thrust_N": config.motor.max_thrust_N,
        "burn_time_s": config.motor.burn_time_s,
        "propellant_mass_g": config.motor.propellant_mass_g,
        "case_mass_g": config.motor.case_mass_g,
        "thrust_curve": config.motor.thrust_curve,
    }
    motor_config_dict = {k: v for k, v in motor_config_dict.items() if v is not None}

    env = RealisticMotorRocket(
        airframe=airframe,
        motor_config=motor_config_dict,
        config=rocket_config,
    )

    return env


@dataclass
class EpisodeResult:
    """Results from one episode"""

    max_altitude: float
    mean_spin_rate: float
    final_spin_rate: float
    total_reward: float
    episode_length: int
    times: np.ndarray
    altitudes: np.ndarray
    roll_rates: np.ndarray
    actions: np.ndarray


def run_episode(env, controller: PIDController, dt: float = 0.01) -> EpisodeResult:
    """Run one episode with PID controller"""
    controller.reset()
    obs, info = env.reset()

    times, altitudes, roll_rates, actions = [], [], [], []
    total_reward = 0.0
    step = 0

    while True:
        # Get PID action
        action = controller.step(obs, info, dt)

        # Store data
        times.append(info.get("time_s", step * dt))
        altitudes.append(info.get("altitude_m", 0))
        roll_rates.append(info.get("roll_rate_deg_s", 0))
        actions.append(action[0])

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if terminated or truncated:
            break

    return EpisodeResult(
        max_altitude=max(altitudes),
        mean_spin_rate=np.mean(np.abs(roll_rates)),
        final_spin_rate=abs(roll_rates[-1]) if roll_rates else 0,
        total_reward=total_reward,
        episode_length=step,
        times=np.array(times),
        altitudes=np.array(altitudes),
        roll_rates=np.array(roll_rates),
        actions=np.array(actions),
    )


def evaluate_pid(
    config_path: str, pid_config: PIDConfig, n_episodes: int = 50
) -> List[EpisodeResult]:
    """Evaluate PID controller over multiple episodes"""
    config = load_config(config_path)
    env = create_env(config)
    controller = PIDController(pid_config)
    dt = getattr(config.environment, "dt", 0.01)

    results = []
    print(f"Evaluating PID controller over {n_episodes} episodes...")

    for i in range(n_episodes):
        result = run_episode(env, controller, dt)
        results.append(result)

        if (i + 1) % 10 == 0:
            recent = results[-10:]
            success_rate = sum(1 for r in recent if r.mean_spin_rate < 30) / len(recent)
            print(
                f"  Episode {i+1}/{n_episodes} - Success rate: {success_rate*100:.0f}%"
            )

    env.close()
    return results


def print_summary(results: List[EpisodeResult], label: str = "PID"):
    """Print summary statistics"""
    altitudes = [r.max_altitude for r in results]
    mean_spins = [r.mean_spin_rate for r in results]
    rewards = [r.total_reward for r in results]

    print(f"\n{'='*60}")
    print(f"{label} Controller Results ({len(results)} episodes)")
    print(f"{'='*60}")
    print(f"Max Altitude:    {np.mean(altitudes):.1f} ± {np.std(altitudes):.1f} m")
    print(f"Mean Spin Rate:  {np.mean(mean_spins):.1f} ± {np.std(mean_spins):.1f} °/s")
    print(f"Total Reward:    {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(
        f"Success (<30°/s): {sum(1 for s in mean_spins if s < 30)/len(mean_spins)*100:.1f}%"
    )
    print(f"{'='*60}\n")


def plot_comparison(
    pid_results: List[EpisodeResult],
    rl_results: List[EpisodeResult] = None,
    save_path: str = None,
):
    """Plot comparison between PID and RL controllers"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Find best episodes
    pid_best_idx = np.argmin([r.mean_spin_rate for r in pid_results])
    pid_best = pid_results[pid_best_idx]

    # Plot PID best trajectory
    axes[0, 0].plot(
        pid_best.times, pid_best.roll_rates, "b-", label="PID", linewidth=1.5
    )
    axes[0, 0].axhline(0, color="gray", linestyle="-", alpha=0.3)
    axes[0, 0].axhline(
        30, color="green", linestyle="--", alpha=0.5, label="Good threshold"
    )
    axes[0, 0].axhline(-30, color="green", linestyle="--", alpha=0.5)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Roll Rate (°/s)")
    axes[0, 0].set_title("Best PID Trajectory - Roll Rate")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot PID control actions
    axes[0, 1].plot(pid_best.times, pid_best.actions, "b-", linewidth=1.5)
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Control Input")
    axes[0, 1].set_title("Best PID Trajectory - Control")
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].grid(True, alpha=0.3)

    if rl_results:
        rl_best_idx = np.argmin([r.mean_spin_rate for r in rl_results])
        rl_best = rl_results[rl_best_idx]

        # Overlay RL on roll rate plot
        axes[0, 0].plot(
            rl_best.times,
            rl_best.roll_rates,
            "r-",
            label="RL",
            linewidth=1.5,
            alpha=0.7,
        )
        axes[0, 0].legend()

        # Overlay RL on control plot
        axes[0, 1].plot(
            rl_best.times, rl_best.actions, "r-", linewidth=1.5, alpha=0.7, label="RL"
        )
        axes[0, 1].legend()

    # Distribution comparison
    pid_spins = [r.mean_spin_rate for r in pid_results]
    axes[1, 0].hist(pid_spins, bins=20, alpha=0.7, label="PID", color="blue")
    if rl_results:
        rl_spins = [r.mean_spin_rate for r in rl_results]
        axes[1, 0].hist(rl_spins, bins=20, alpha=0.7, label="RL", color="red")
    axes[1, 0].axvline(30, color="green", linestyle="--", label="Good threshold")
    axes[1, 0].set_xlabel("Mean Spin Rate (°/s)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Spin Rate Distribution")
    axes[1, 0].legend()

    # Summary stats
    ax = axes[1, 1]
    ax.axis("off")

    pid_success = sum(1 for s in pid_spins if s < 30) / len(pid_spins) * 100
    stats_text = f"""
    PID Controller
    ─────────────────────
    Mean Spin Rate: {np.mean(pid_spins):.1f} ± {np.std(pid_spins):.1f} °/s
    Success Rate: {pid_success:.1f}%
    """

    if rl_results:
        rl_success = sum(1 for s in rl_spins if s < 30) / len(rl_spins) * 100
        stats_text += f"""
    RL Controller
    ─────────────────────
    Mean Spin Rate: {np.mean(rl_spins):.1f} ± {np.std(rl_spins):.1f} °/s
    Success Rate: {rl_success:.1f}%
    """

    ax.text(
        0.5,
        0.5,
        stats_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="PID controller for rocket roll stabilization"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--n-episodes", type=int, default=50, help="Number of episodes")
    parser.add_argument("--compare", type=str, help="Path to RL model for comparison")
    parser.add_argument("--save-plot", type=str, help="Save comparison plot")

    # PID gains
    parser.add_argument(
        "--Cprop", type=float, default=0.005208, help="Proportional gain"
    )
    parser.add_argument("--Cint", type=float, default=0.000324, help="Integral gain")
    parser.add_argument(
        "--Cderiv", type=float, default=0.016524, help="Derivative gain"
    )

    args = parser.parse_args()

    # Create PID config
    pid_config = PIDConfig(
        Cprop=args.Cprop,
        Cint=args.Cint,
        Cderiv=args.Cderiv,
    )

    print(
        f"PID Gains: Cprop={pid_config.Cprop}, Cint={pid_config.Cint}, Cderiv={pid_config.Cderiv}"
    )

    # Evaluate PID
    pid_results = evaluate_pid(args.config, pid_config, args.n_episodes)
    print_summary(pid_results, "PID")

    # Compare with RL if requested
    rl_results = None
    if args.compare:
        print(f"\nEvaluating RL model: {args.compare}")
        # Import here to avoid circular imports
        from visualizations.visualize_spin_agent import SpinAgentEvaluator

        evaluator = SpinAgentEvaluator(args.compare, args.config)
        rl_episodes = evaluator.evaluate(args.n_episodes)

        # Convert to our format
        rl_results = [
            EpisodeResult(
                max_altitude=ep.max_altitude,
                mean_spin_rate=ep.mean_spin_rate,
                final_spin_rate=ep.final_spin_rate,
                total_reward=ep.total_reward,
                episode_length=ep.episode_length,
                times=ep.time,
                altitudes=ep.altitude,
                roll_rates=ep.roll_rate,
                actions=ep.actions,
            )
            for ep in rl_episodes
        ]
        print_summary(rl_results, "RL")

    # Plot
    plot_comparison(pid_results, rl_results, args.save_plot)


if __name__ == "__main__":
    main()
