#!/usr/bin/env python3
"""
Compare PID, PPO, SAC, Residual SAC, and DOB SAC controllers under varying wind conditions.

Tests each controller at multiple wind levels and reports metrics:
- Mean spin rate
- Settling time
- Success rate (spin < threshold)
- Control smoothness

Usage:
    # PID only baseline
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml --pid-only

    # Compare SAC with PID
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
        --sac models/rocket_sac_wind_*/best_model.zip

    # Compare all three
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
        --sac models/sac_model/best_model.zip \
        --ppo models/ppo_model/best_model.zip

    # Compare residual SAC (uses its own config with use_residual_pid: true)
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
        --sac models/rocket_sac_wind_*/best_model.zip \
        --residual-sac models/rocket_residual_sac_wind_*/best_model.zip

    # Compare DOB SAC (uses config with use_disturbance_observer: true)
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
        --dob-sac models/rocket_dob_sac_*/best_model.zip

    # Custom wind levels
    uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
        --sac models/best_model.zip --wind-levels 0 1 3 5 10
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import copy

from rocket_config import RocketTrainingConfig, load_config
from spin_stabilized_control_env import RocketConfig
from realistic_spin_rocket import RealisticMotorRocket
from pid_controller import (
    PIDController,
    GainScheduledPIDController,
    LeadCompensatedGSPIDController,
    PIDConfig,
)
from adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config
from wind_feedforward import WindFeedforwardADRC, WindFeedforwardConfig
from wind_estimator import NNFeedforwardADRC, load_estimator
from online_identification import B0Estimator, B0EstimatorConfig
from repetitive_controller import RepetitiveGSPIDController, RepetitiveConfig
from ensemble_controller import EnsembleController, EnsembleConfig
from fourier_adaptive import FourierAdaptiveADRC, FourierAdaptiveConfig
from gp_disturbance import GPFeedforwardController, GPDisturbanceConfig
from sta_smc_controller import STASMCController, STASMCConfig
from cascade_dob import CascadeDOBController, CascadeDOBConfig
from fll_controller import FLLController, FLLConfig
from hinf_controller import HinfController, HinfConfig
from wind_model import WindModel, WindConfig
from train_improved import create_environment
from rocket_env.sensors import IMUObservationWrapper, IMUConfig


@dataclass
class EpisodeMetrics:
    """Metrics from a single episode."""

    mean_spin_rate: float  # deg/s
    max_spin_rate: float  # deg/s
    settling_time: float  # seconds to reach < 10 deg/s (inf if never)
    total_reward: float
    max_altitude: float  # meters
    control_smoothness: float  # mean |action_change|
    episode_length: int
    spin_rate_series: Optional[np.ndarray] = None  # full spin rate time series (deg/s)


@dataclass
class ControllerResult:
    """Aggregated results for one controller at one wind level."""

    controller_name: str
    wind_speed: float
    episodes: List[EpisodeMetrics]

    @property
    def mean_spin(self) -> float:
        return np.mean([e.mean_spin_rate for e in self.episodes])

    @property
    def std_spin(self) -> float:
        return np.std([e.mean_spin_rate for e in self.episodes])

    @property
    def mean_settling(self) -> float:
        settling = [
            e.settling_time for e in self.episodes if np.isfinite(e.settling_time)
        ]
        return np.mean(settling) if settling else float("inf")

    @property
    def success_rate(self) -> float:
        return np.mean([e.mean_spin_rate < 30.0 for e in self.episodes])

    @property
    def mean_smoothness(self) -> float:
        return np.mean([e.control_smoothness for e in self.episodes])


def create_env(config: RocketTrainingConfig, wind_speed: float = 0.0):
    """Create bare environment (no wrappers) with specified wind speed."""
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
        enable_wind=wind_speed > 0,
        base_wind_speed=wind_speed,
        max_gust_speed=wind_speed * 0.5,
        wind_variability=getattr(config.physics, "wind_variability", 0.3),
        use_dryden=getattr(config.physics, "use_dryden", False),
        turbulence_severity=getattr(config.physics, "turbulence_severity", "light"),
        altitude_profile_alpha=getattr(config.physics, "altitude_profile_alpha", 0.14),
        reference_altitude=getattr(config.physics, "reference_altitude", 10.0),
        body_shadow_factor=getattr(config.physics, "body_shadow_factor", 0.90),
        # Mach-dependent aerodynamics
        use_mach_aero=getattr(config.physics, "use_mach_aero", False),
        use_isa_full=getattr(config.physics, "use_isa_full", False),
        cd_mach_table=getattr(config.physics, "cd_mach_table", None),
        # Servo dynamics
        servo_time_constant=getattr(config.physics, "servo_time_constant", 0.0),
        servo_rate_limit=getattr(config.physics, "servo_rate_limit", 0.0),
        servo_deadband=getattr(config.physics, "servo_deadband", 0.0),
        # Sensor latency
        sensor_delay_steps=getattr(config.physics, "sensor_delay_steps", 0),
        # Observation space bounds
        max_velocity=getattr(config.physics, "max_velocity", 100.0),
        max_dynamic_pressure=getattr(config.physics, "max_dynamic_pressure", 3000.0),
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


def create_env_with_imu(config: RocketTrainingConfig, wind_speed: float = 0.0):
    """Create bare environment wrapped with IMU noise (for IMU-based PID eval).

    This gives the PID controller the same noisy observations the RL agent sees,
    without any of the training wrappers (reward shaping, action smoothing, etc.).
    """
    env = create_env(config, wind_speed)

    # Apply IMU wrapper with same config as training
    if config.sensors.imu_custom:
        imu_config = IMUConfig.from_dict(config.sensors.imu_custom)
    else:
        imu_config = IMUConfig.get_preset(config.sensors.imu_preset)

    env = IMUObservationWrapper(
        env,
        imu_config=imu_config,
        control_rate_hz=config.sensors.control_rate_hz,
        derive_acceleration=config.sensors.derive_acceleration,
    )
    return env


def run_controller_episode(
    env, controller, dt: float, collect_spin_series: bool = False
) -> EpisodeMetrics:
    """Run one episode with any controller that has reset()/step(obs, info, dt) interface."""
    controller.reset()
    obs, info = env.reset()

    spin_rates = []
    actions = []
    total_reward = 0.0
    settling_time = float("inf")
    step = 0

    while True:
        action = controller.step(obs, info, dt)
        obs, reward, terminated, truncated, info = env.step(action)

        spin_rate = abs(info.get("roll_rate_deg_s", 0.0))
        spin_rates.append(spin_rate)
        actions.append(float(action[0]))
        total_reward += reward
        step += 1

        # Check settling
        if settling_time == float("inf") and spin_rate < 10.0:
            settling_time = info.get("time_s", step * dt)

        if terminated or truncated:
            break

    action_changes = np.abs(np.diff(actions)) if len(actions) > 1 else [0.0]

    return EpisodeMetrics(
        mean_spin_rate=np.mean(spin_rates),
        max_spin_rate=np.max(spin_rates),
        settling_time=settling_time,
        total_reward=total_reward,
        max_altitude=info.get("max_altitude_m", 0.0),
        control_smoothness=np.mean(action_changes),
        episode_length=step,
        spin_rate_series=np.array(spin_rates) if collect_spin_series else None,
    )


def run_pid_episode(
    env,
    pid_config: PIDConfig,
    dt: float,
    use_observations: bool = False,
    collect_spin_series: bool = False,
) -> EpisodeMetrics:
    """Run one episode with PID controller."""
    controller = PIDController(pid_config, use_observations=use_observations)
    return run_controller_episode(
        env, controller, dt, collect_spin_series=collect_spin_series
    )


def run_rl_episode(
    env,
    model,
    vec_normalize: Optional[VecNormalize],
    dt: float,
    collect_spin_series: bool = False,
) -> EpisodeMetrics:
    """Run one episode with an RL model (PPO or SAC)."""
    obs, info = env.reset()

    spin_rates = []
    actions = []
    total_reward = 0.0
    settling_time = float("inf")
    step = 0

    while True:
        # Normalize obs if needed
        if vec_normalize is not None:
            obs_normalized = vec_normalize.normalize_obs(obs.reshape(1, -1))[0]
        else:
            obs_normalized = obs

        action, _ = model.predict(obs_normalized, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        spin_rate = abs(info.get("roll_rate_deg_s", 0.0))
        spin_rates.append(spin_rate)
        # Use combined_action if available (residual SAC), else raw action
        actual_action = info.get("combined_action", float(action[0]))
        actions.append(float(actual_action))
        total_reward += reward
        step += 1

        if settling_time == float("inf") and spin_rate < 10.0:
            settling_time = info.get("time_s", step * dt)

        if terminated or truncated:
            break

    action_changes = np.abs(np.diff(actions)) if len(actions) > 1 else [0.0]

    return EpisodeMetrics(
        mean_spin_rate=np.mean(spin_rates),
        max_spin_rate=np.max(spin_rates),
        settling_time=settling_time,
        total_reward=total_reward,
        max_altitude=info.get("max_altitude_m", 0.0),
        control_smoothness=np.mean(action_changes),
        episode_length=step,
        spin_rate_series=np.array(spin_rates) if collect_spin_series else None,
    )


def create_wrapped_env(config: RocketTrainingConfig, wind_speed: float = 0.0):
    """Create a fully-wrapped environment (matching training wrappers) with a specific wind speed.

    This ensures RL models get the same observation space they were trained with
    (e.g. PreviousActionWrapper adds an extra dimension).
    """
    cfg = copy.deepcopy(config)
    cfg.physics.enable_wind = wind_speed > 0
    cfg.physics.base_wind_speed = wind_speed
    cfg.physics.max_gust_speed = wind_speed * 0.5
    return create_environment(cfg)


def load_rl_model(model_path: str, algo: str, config: RocketTrainingConfig):
    """Load an RL model and its VecNormalize stats if available."""
    model_path = Path(model_path)
    if algo == "sac":
        model = SAC.load(str(model_path))
    else:
        model = PPO.load(str(model_path))

    # Check for VecNormalize stats
    vec_normalize = None
    vec_norm_path = model_path.parent / "vec_normalize.pkl"
    if vec_norm_path.exists():
        # Create a dummy env with the full wrapper chain so obs shape matches
        dummy_env = DummyVecEnv([lambda: create_wrapped_env(config, 0.0)])
        vec_normalize = VecNormalize.load(str(vec_norm_path), dummy_env)
        vec_normalize.training = False
        print(f"  Loaded VecNormalize from {vec_norm_path}")

    return model, vec_normalize


def evaluate_controller(
    config: RocketTrainingConfig,
    controller_name: str,
    wind_levels: List[float],
    n_episodes: int,
    model=None,
    vec_normalize=None,
    pid_config: PIDConfig = None,
    use_observations: bool = False,
    controller=None,
    collect_spin_series: bool = False,
) -> List[ControllerResult]:
    """Evaluate a controller across wind levels.

    Args:
        controller: Any object with reset()/step(obs, info, dt) interface
                    (e.g. ADRCController). Takes precedence over pid_config.
        collect_spin_series: If True, store per-step spin rate arrays in each
                             EpisodeMetrics (for video quality analysis).
    """
    dt = getattr(config.environment, "dt", 0.01)
    results = []

    for wind_speed in wind_levels:
        episodes = []
        # Use wrapped env for RL (matches training obs space),
        # IMU-wrapped env for observation-mode controllers,
        # bare env for ground-truth controllers
        if model is not None:
            env = create_wrapped_env(config, wind_speed)
        elif use_observations:
            env = create_env_with_imu(config, wind_speed)
        else:
            env = create_env(config, wind_speed)

        for ep in range(n_episodes):
            if model is not None:
                metrics = run_rl_episode(
                    env,
                    model,
                    vec_normalize,
                    dt,
                    collect_spin_series=collect_spin_series,
                )
            elif controller is not None:
                metrics = run_controller_episode(
                    env, controller, dt, collect_spin_series=collect_spin_series
                )
            else:
                metrics = run_pid_episode(
                    env,
                    pid_config,
                    dt,
                    use_observations=use_observations,
                    collect_spin_series=collect_spin_series,
                )
            episodes.append(metrics)

        env.close()

        result = ControllerResult(
            controller_name=controller_name,
            wind_speed=wind_speed,
            episodes=episodes,
        )
        results.append(result)

        print(
            f"  {controller_name} @ {wind_speed:.0f} m/s: "
            f"spin={result.mean_spin:.1f}+/-{result.std_spin:.1f} deg/s, "
            f"success={result.success_rate*100:.0f}%, "
            f"settling={result.mean_settling:.2f}s"
        )

    return results


def print_comparison_table(all_results: Dict[str, List[ControllerResult]]):
    """Print a formatted comparison table."""
    print(f"\n{'='*80}")
    print("CONTROLLER COMPARISON")
    print(f"{'='*80}")

    # Header
    controllers = list(all_results.keys())
    wind_levels = [r.wind_speed for r in list(all_results.values())[0]]

    header = f"{'Wind (m/s)':>12}"
    for c in controllers:
        header += f" | {c:>20}"
    print(header)
    print("-" * len(header))

    # Mean spin rate rows
    print("\nMean Spin Rate (deg/s):")
    for i, wind in enumerate(wind_levels):
        row = f"{wind:>12.0f}"
        for c in controllers:
            r = all_results[c][i]
            row += f" | {r.mean_spin:>8.1f} +/- {r.std_spin:>5.1f}"
        print(row)

    # Success rate rows
    print("\nSuccess Rate (spin < 30 deg/s):")
    for i, wind in enumerate(wind_levels):
        row = f"{wind:>12.0f}"
        for c in controllers:
            r = all_results[c][i]
            row += f" | {r.success_rate*100:>20.0f}%"
        print(row)

    # Settling time rows
    print("\nMean Settling Time (s):")
    for i, wind in enumerate(wind_levels):
        row = f"{wind:>12.0f}"
        for c in controllers:
            r = all_results[c][i]
            settling = (
                f"{r.mean_settling:.2f}" if np.isfinite(r.mean_settling) else "N/A"
            )
            row += f" | {settling:>20}"
        print(row)

    # Smoothness rows
    print("\nControl Smoothness (mean |delta_action|):")
    for i, wind in enumerate(wind_levels):
        row = f"{wind:>12.0f}"
        for c in controllers:
            r = all_results[c][i]
            row += f" | {r.mean_smoothness:>20.4f}"
        print(row)

    print(f"\n{'='*80}")


def plot_comparison(
    all_results: Dict[str, List[ControllerResult]], save_path: str = None
):
    """Plot comparison charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "PID": "blue",
        "PID (IMU)": "cyan",
        "GS-PID": "dodgerblue",
        "GS-PID (IMU)": "lightskyblue",
        "PPO": "green",
        "SAC": "red",
        "Residual SAC": "purple",
        "DOB SAC": "orange",
        "ADRC": "darkgreen",
        "ADRC (IMU)": "limegreen",
        "ADRC+FF": "forestgreen",
        "ADRC+FF (IMU)": "springgreen",
        "Lead GS-PID": "darkorange",
        "Lead GS-PID (IMU)": "orange",
        "ADRC+RLS": "teal",
        "ADRC+RLS (IMU)": "lightseagreen",
        "ADRC+FF+RLS": "darkslategray",
        "ADRC+FF+RLS (IMU)": "cadetblue",
        "ADRC+NN": "darkviolet",
        "ADRC+NN (IMU)": "mediumpurple",
        "Rep GS-PID": "crimson",
        "Rep GS-PID (IMU)": "lightcoral",
        "Ensemble": "goldenrod",
        "Ensemble (IMU)": "gold",
        "Optimized GS-PID": "darkred",
        "Optimized GS-PID (IMU)": "indianred",
        "Optimized ADRC": "darkblue",
        "Optimized ADRC (IMU)": "royalblue",
        "Optimized Ensemble": "darkgoldenrod",
        "Optimized Ensemble (IMU)": "khaki",
        "Fourier ADRC": "darkcyan",
        "Fourier ADRC (IMU)": "cyan",
        "GP GS-PID": "sienna",
        "GP GS-PID (IMU)": "sandybrown",
        "STA-SMC": "maroon",
        "STA-SMC (IMU)": "salmon",
        "CDO GS-PID": "olive",
        "CDO GS-PID (IMU)": "yellowgreen",
        "FLL GS-PID": "navy",
        "FLL GS-PID (IMU)": "cornflowerblue",
        "H-inf": "darkgoldenrod",
        "H-inf (IMU)": "goldenrod",
    }

    for name, results in all_results.items():
        wind_levels = [r.wind_speed for r in results]
        color = colors.get(name, "gray")

        # Mean spin rate
        means = [r.mean_spin for r in results]
        stds = [r.std_spin for r in results]
        axes[0, 0].errorbar(
            wind_levels,
            means,
            yerr=stds,
            label=name,
            color=color,
            marker="o",
            linewidth=2,
            capsize=5,
        )

        # Success rate
        success = [r.success_rate * 100 for r in results]
        axes[0, 1].plot(
            wind_levels,
            success,
            label=name,
            color=color,
            marker="o",
            linewidth=2,
        )

        # Settling time
        settling = [
            r.mean_settling if np.isfinite(r.mean_settling) else None for r in results
        ]
        valid_wind = [w for w, s in zip(wind_levels, settling) if s is not None]
        valid_settling = [s for s in settling if s is not None]
        if valid_settling:
            axes[1, 0].plot(
                valid_wind,
                valid_settling,
                label=name,
                color=color,
                marker="o",
                linewidth=2,
            )

        # Smoothness
        smoothness = [r.mean_smoothness for r in results]
        axes[1, 1].plot(
            wind_levels,
            smoothness,
            label=name,
            color=color,
            marker="o",
            linewidth=2,
        )

    axes[0, 0].set_xlabel("Wind Speed (m/s)")
    axes[0, 0].set_ylabel("Mean Spin Rate (deg/s)")
    axes[0, 0].set_title("Spin Control Performance")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Wind Speed (m/s)")
    axes[0, 1].set_ylabel("Success Rate (%)")
    axes[0, 1].set_title("Success Rate (spin < 30 deg/s)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-5, 105)

    axes[1, 0].set_xlabel("Wind Speed (m/s)")
    axes[1, 0].set_ylabel("Settling Time (s)")
    axes[1, 0].set_title("Time to Settle (< 10 deg/s)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Wind Speed (m/s)")
    axes[1, 1].set_ylabel("Mean |delta_action|")
    axes[1, 1].set_title("Control Smoothness")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Controller Comparison Under Wind", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\nPlot saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare PID, PPO, and SAC under varying wind",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--sac", type=str, help="Path to SAC model (.zip)")
    parser.add_argument("--ppo", type=str, help="Path to PPO model (.zip)")
    parser.add_argument(
        "--residual-sac",
        type=str,
        help="Path to residual SAC model (.zip). Uses config from model dir "
        "(with use_residual_pid: true) for correct wrapper chain.",
    )
    parser.add_argument(
        "--dob-sac",
        type=str,
        help="Path to DOB SAC model (.zip). Uses config from model dir "
        "(with use_disturbance_observer: true) for correct wrapper chain.",
    )
    parser.add_argument(
        "--gain-scheduled",
        action="store_true",
        help="Include gain-scheduled PID controller",
    )
    parser.add_argument(
        "--lead", action="store_true", help="Include lead-compensated GS-PID controller"
    )
    parser.add_argument("--adrc", action="store_true", help="Include ADRC controller")
    parser.add_argument(
        "--adrc-ff",
        action="store_true",
        help="Include ADRC + wind feedforward controller",
    )
    parser.add_argument(
        "--adrc-nn",
        type=str,
        help="Include ADRC + NN wind estimator (path to .pt model)",
    )
    parser.add_argument(
        "--rls-b0",
        action="store_true",
        help="Use online RLS b0 identification for ADRC controllers",
    )
    parser.add_argument(
        "--repetitive",
        action="store_true",
        help="Include repetitive (resonant) GS-PID controller",
    )
    parser.add_argument(
        "--repetitive-krc",
        type=float,
        default=0.5,
        help="Repetitive controller gain K_rc",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Include multi-controller ensemble with switching",
    )
    parser.add_argument(
        "--fourier-ff",
        action="store_true",
        help="Include Fourier-domain adaptive ADRC controller",
    )
    parser.add_argument(
        "--gp-ff",
        action="store_true",
        help="Include GP-based uncertainty-gated GS-PID controller",
    )
    parser.add_argument(
        "--sta-smc",
        action="store_true",
        help="Include Super-Twisting Sliding Mode controller",
    )
    parser.add_argument(
        "--cascade-dob",
        action="store_true",
        help="Include Cascade Disturbance Observer controller",
    )
    parser.add_argument(
        "--fll", action="store_true", help="Include Frequency-Locked Loop controller"
    )
    parser.add_argument(
        "--hinf",
        action="store_true",
        help="Include H-infinity (LQG/LTR) robust controller",
    )
    parser.add_argument(
        "--optimized-params",
        type=str,
        help="Path to optimized parameter lookup table (JSON from bayesian_optimize.py)",
    )
    parser.add_argument(
        "--adrc-omega-c", type=float, default=15.0, help="ADRC controller bandwidth"
    )
    parser.add_argument(
        "--adrc-omega-o", type=float, default=50.0, help="ADRC observer bandwidth"
    )
    parser.add_argument(
        "--ff-gain", type=float, default=0.5, help="Feedforward gain K_ff (0-1)"
    )
    parser.add_argument(
        "--pid-only", action="store_true", help="Only test PID baseline"
    )
    parser.add_argument(
        "--wind-levels",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Wind speeds to test (m/s)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20, help="Episodes per condition"
    )
    parser.add_argument("--save-plot", type=str, help="Save comparison plot path")

    # PID gains
    parser.add_argument(
        "--pid-Kp", type=float, default=0.005208, help="PID proportional gain"
    )
    parser.add_argument(
        "--pid-Ki", type=float, default=0.000324, help="PID integral gain"
    )
    parser.add_argument(
        "--pid-Kd", type=float, default=0.016524, help="PID derivative gain"
    )
    parser.add_argument(
        "--pid-qref",
        type=float,
        default=500.0,
        help="GS-PID reference dynamic pressure (Pa)",
    )
    parser.add_argument(
        "--pid-imu",
        action="store_true",
        help="(Deprecated: use --imu instead) Run PID with noisy IMU observations",
    )
    parser.add_argument(
        "--imu",
        action="store_true",
        help="Run ALL classical controllers (PID, GS-PID, ADRC, ADRC+FF) using "
        "noisy IMU observations instead of ground-truth state",
    )
    parser.add_argument(
        "--video-quality",
        action="store_true",
        help="Compute and print Gyroflow post-stabilization video quality analysis",
    )
    parser.add_argument(
        "--camera-preset",
        type=str,
        default="all",
        choices=["1080p60", "4k30", "1080p120", "all"],
        help="Camera preset for video quality analysis (default: all)",
    )

    args = parser.parse_args()
    config = load_config(args.config)

    # --imu applies to all controllers; --pid-imu is backward-compatible alias
    use_imu = args.imu or args.pid_imu
    imu_suffix = " (IMU)" if use_imu else ""
    collect_spin = args.video_quality

    pid_config = PIDConfig(
        Cprop=args.pid_Kp,
        Cint=args.pid_Ki,
        Cderiv=args.pid_Kd,
    )

    all_results = {}

    # PID baseline
    pid_label = f"PID{imu_suffix}"
    print(f"\nEvaluating {pid_label} controller...")
    pid_results = evaluate_controller(
        config,
        pid_label,
        args.wind_levels,
        args.n_episodes,
        pid_config=pid_config,
        use_observations=use_imu,
        collect_spin_series=collect_spin,
    )
    all_results[pid_label] = pid_results

    # Gain-scheduled PID (if requested)
    if args.gain_scheduled and not args.pid_only:
        gs_label = f"GS-PID{imu_suffix}"
        print(f"\nEvaluating {gs_label} controller...")
        gs_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        gs_ctrl = GainScheduledPIDController(gs_pid_config, use_observations=use_imu)
        gs_results = evaluate_controller(
            config,
            gs_label,
            args.wind_levels,
            args.n_episodes,
            controller=gs_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[gs_label] = gs_results

    # Lead-compensated GS-PID (if requested)
    if args.lead and not args.pid_only:
        lead_label = f"Lead GS-PID{imu_suffix}"
        print(f"\nEvaluating {lead_label} controller...")
        lead_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        lead_ctrl = LeadCompensatedGSPIDController(
            lead_pid_config, use_observations=use_imu
        )
        lead_results = evaluate_controller(
            config,
            lead_label,
            args.wind_levels,
            args.n_episodes,
            controller=lead_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[lead_label] = lead_results

    # Repetitive GS-PID (if requested)
    if args.repetitive and not args.pid_only:
        rep_label = f"Rep GS-PID{imu_suffix}"
        print(f"\nEvaluating {rep_label} controller (K_rc={args.repetitive_krc})...")
        rep_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        rep_config = RepetitiveConfig(K_rc=args.repetitive_krc)
        rep_ctrl = RepetitiveGSPIDController(
            rep_pid_config,
            rep_config,
            use_observations=use_imu,
        )
        rep_results = evaluate_controller(
            config,
            rep_label,
            args.wind_levels,
            args.n_episodes,
            controller=rep_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[rep_label] = rep_results

    # Ensemble controller (if requested)
    if args.ensemble and not args.pid_only:
        ens_label = f"Ensemble{imu_suffix}"
        print(f"\nEvaluating {ens_label} controller...")
        # Build controller bank: GS-PID + ADRC
        ens_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        ens_gs_pid = GainScheduledPIDController(
            ens_pid_config, use_observations=use_imu
        )

        airframe_ens = config.physics.resolve_airframe()
        adrc_ens_config = estimate_adrc_config(
            airframe_ens,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        adrc_ens_config.use_observations = use_imu
        ens_adrc = ADRCController(adrc_ens_config)

        ens_config = EnsembleConfig()
        ens_ctrl = EnsembleController(
            controllers=[ens_gs_pid, ens_adrc],
            names=["GS-PID", "ADRC"],
            config=ens_config,
        )
        ens_results = evaluate_controller(
            config,
            ens_label,
            args.wind_levels,
            args.n_episodes,
            controller=ens_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[ens_label] = ens_results

    # Fourier-domain adaptive ADRC (if requested)
    if args.fourier_ff and not args.pid_only:
        fourier_label = f"Fourier ADRC{imu_suffix}"
        print(f"\nEvaluating {fourier_label} controller...")
        airframe_fourier = config.physics.resolve_airframe()
        adrc_fourier_config = estimate_adrc_config(
            airframe_fourier,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        adrc_fourier_config.use_observations = use_imu
        fourier_config = FourierAdaptiveConfig(K_ff=args.ff_gain)
        print(
            f"  Estimated b0={adrc_fourier_config.b0:.1f} (b0_per_pa={adrc_fourier_config.b0_per_pa:.4f})"
        )
        fourier_ctrl = FourierAdaptiveADRC(adrc_fourier_config, fourier_config)
        fourier_results = evaluate_controller(
            config,
            fourier_label,
            args.wind_levels,
            args.n_episodes,
            controller=fourier_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[fourier_label] = fourier_results

    # GP-based uncertainty-gated GS-PID (if requested)
    if args.gp_ff and not args.pid_only:
        gp_label = f"GP GS-PID{imu_suffix}"
        print(f"\nEvaluating {gp_label} controller...")
        gp_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        gp_config = GPDisturbanceConfig(K_ff=args.ff_gain)
        gp_ctrl = GPFeedforwardController(
            gp_pid_config, gp_config, use_observations=use_imu
        )
        gp_results = evaluate_controller(
            config,
            gp_label,
            args.wind_levels,
            args.n_episodes,
            controller=gp_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[gp_label] = gp_results

    # STA-SMC (if requested)
    if args.sta_smc and not args.pid_only:
        smc_label = f"STA-SMC{imu_suffix}"
        print(f"\nEvaluating {smc_label} controller...")
        airframe_smc = config.physics.resolve_airframe()
        adrc_smc_config = estimate_adrc_config(
            airframe_smc,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        smc_config = STASMCConfig(
            b0=adrc_smc_config.b0,
            b0_per_pa=adrc_smc_config.b0_per_pa,
            use_observations=use_imu,
        )
        smc_ctrl = STASMCController(smc_config)
        smc_results = evaluate_controller(
            config,
            smc_label,
            args.wind_levels,
            args.n_episodes,
            controller=smc_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[smc_label] = smc_results

    # Cascade DOB (if requested)
    if args.cascade_dob and not args.pid_only:
        cdo_label = f"CDO GS-PID{imu_suffix}"
        print(f"\nEvaluating {cdo_label} controller...")
        airframe_cdo = config.physics.resolve_airframe()
        adrc_cdo_config = estimate_adrc_config(
            airframe_cdo,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        cdo_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        cdo_config = CascadeDOBConfig(
            K_ff=args.ff_gain,
            b0=adrc_cdo_config.b0,
            b0_per_pa=adrc_cdo_config.b0_per_pa,
        )
        cdo_ctrl = CascadeDOBController(
            cdo_pid_config,
            cdo_config,
            use_observations=use_imu,
        )
        cdo_results = evaluate_controller(
            config,
            cdo_label,
            args.wind_levels,
            args.n_episodes,
            controller=cdo_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[cdo_label] = cdo_results

    # FLL (if requested)
    if args.fll and not args.pid_only:
        fll_label = f"FLL GS-PID{imu_suffix}"
        print(f"\nEvaluating {fll_label} controller...")
        airframe_fll = config.physics.resolve_airframe()
        adrc_fll_config = estimate_adrc_config(
            airframe_fll,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        fll_pid_config = PIDConfig(
            Cprop=args.pid_Kp,
            Cint=args.pid_Ki,
            Cderiv=args.pid_Kd,
            q_ref=args.pid_qref,
        )
        fll_config = FLLConfig(
            K_ff=args.ff_gain,
            b0=adrc_fll_config.b0,
            b0_per_pa=adrc_fll_config.b0_per_pa,
        )
        fll_ctrl = FLLController(
            fll_pid_config,
            fll_config,
            use_observations=use_imu,
        )
        fll_results = evaluate_controller(
            config,
            fll_label,
            args.wind_levels,
            args.n_episodes,
            controller=fll_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[fll_label] = fll_results

    # H-infinity / LQG/LTR (if requested)
    if args.hinf and not args.pid_only:
        hinf_label = f"H-inf{imu_suffix}"
        print(f"\nEvaluating {hinf_label} controller...")
        airframe_hinf = config.physics.resolve_airframe()
        adrc_hinf_config = estimate_adrc_config(
            airframe_hinf,
            config.physics,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        hinf_config = HinfConfig(
            b0=adrc_hinf_config.b0,
            b0_per_pa=adrc_hinf_config.b0_per_pa,
            use_observations=use_imu,
        )
        hinf_ctrl = HinfController(hinf_config)
        hinf_results = evaluate_controller(
            config,
            hinf_label,
            args.wind_levels,
            args.n_episodes,
            controller=hinf_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[hinf_label] = hinf_results

    # Optimized parameters (from bayesian_optimize.py)
    if args.optimized_params and not args.pid_only:
        import json

        with open(args.optimized_params) as f:
            lookup_data = json.load(f)
        from bayesian_optimize import ParamLookupTable

        lookup = ParamLookupTable.from_dict(lookup_data)
        ctrl_type = lookup.controller
        opt_label = f"Optimized {ctrl_type.upper().replace('-', ' ')}{imu_suffix}"
        # Shorten common names
        opt_label = opt_label.replace("GS PID", "GS-PID")
        print(f"\nEvaluating {opt_label} (from {args.optimized_params})...")

        opt_wind_results = []
        for wind_speed in args.wind_levels:
            params = lookup.get_params(wind_speed)
            print(f"  Wind {wind_speed:.0f} m/s: params={params}")

            if ctrl_type == "gs-pid":
                opt_pid_cfg = PIDConfig(
                    Cprop=params["Kp"],
                    Cint=params["Ki"],
                    Cderiv=params["Kd"],
                    q_ref=params.get("q_ref", 500.0),
                )
                opt_ctrl = GainScheduledPIDController(
                    opt_pid_cfg, use_observations=use_imu
                )
            elif ctrl_type == "adrc":
                airframe_opt = config.physics.resolve_airframe()
                opt_adrc_cfg = estimate_adrc_config(
                    airframe_opt,
                    config.physics,
                    omega_c=params["omega_c"],
                    omega_o=params["omega_o"],
                )
                opt_adrc_cfg.use_observations = use_imu
                opt_ctrl = ADRCController(opt_adrc_cfg)
            elif ctrl_type == "ensemble":
                opt_pid_cfg = PIDConfig(
                    Cprop=params["Kp"],
                    Cint=params["Ki"],
                    Cderiv=params["Kd"],
                    q_ref=params.get("q_ref", 500.0),
                )
                opt_gs_pid = GainScheduledPIDController(
                    opt_pid_cfg, use_observations=use_imu
                )
                airframe_opt = config.physics.resolve_airframe()
                opt_adrc_cfg = estimate_adrc_config(
                    airframe_opt,
                    config.physics,
                    omega_c=params["omega_c"],
                    omega_o=params["omega_o"],
                )
                opt_adrc_cfg.use_observations = use_imu
                opt_adrc = ADRCController(opt_adrc_cfg)
                opt_ctrl = EnsembleController(
                    controllers=[opt_gs_pid, opt_adrc],
                    names=["GS-PID", "ADRC"],
                    config=EnsembleConfig(),
                )
            else:
                print(f"  Unknown controller type: {ctrl_type}, skipping")
                continue

            # Evaluate at this wind level
            wind_result = evaluate_controller(
                config,
                opt_label,
                [wind_speed],
                args.n_episodes,
                controller=opt_ctrl,
                use_observations=use_imu,
                collect_spin_series=collect_spin,
            )
            opt_wind_results.extend(wind_result)

        all_results[opt_label] = opt_wind_results

    # ADRC (if requested)
    if args.adrc and not args.pid_only:
        airframe = config.physics.resolve_airframe()
        rocket_cfg = config.physics
        adrc_config = estimate_adrc_config(
            airframe,
            rocket_cfg,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        adrc_config.use_observations = use_imu
        print(
            f"  Estimated b0={adrc_config.b0:.1f} rad/s^2 per unit action"
            f" (b0_per_pa={adrc_config.b0_per_pa:.4f})"
        )

        if args.rls_b0:
            rls_label = f"ADRC+RLS{imu_suffix}"
            print(
                f"\nEvaluating {rls_label} (omega_c={args.adrc_omega_c}, omega_o={args.adrc_omega_o})..."
            )
            b0_est = B0Estimator(B0EstimatorConfig(b0_init=adrc_config.b0))
            adrc_rls_ctrl = ADRCController(adrc_config, b0_estimator=b0_est)
            adrc_rls_results = evaluate_controller(
                config,
                rls_label,
                args.wind_levels,
                args.n_episodes,
                controller=adrc_rls_ctrl,
                use_observations=use_imu,
                collect_spin_series=collect_spin,
            )
            all_results[rls_label] = adrc_rls_results
        else:
            adrc_label = f"ADRC{imu_suffix}"
            print(
                f"\nEvaluating {adrc_label} (omega_c={args.adrc_omega_c}, omega_o={args.adrc_omega_o})..."
            )
            adrc_ctrl = ADRCController(adrc_config)
            adrc_results = evaluate_controller(
                config,
                adrc_label,
                args.wind_levels,
                args.n_episodes,
                controller=adrc_ctrl,
                use_observations=use_imu,
                collect_spin_series=collect_spin,
            )
            all_results[adrc_label] = adrc_results

    # ADRC + Wind Feedforward (if requested)
    if args.adrc_ff and not args.pid_only:
        airframe_ff = config.physics.resolve_airframe()
        rocket_cfg_ff = config.physics
        adrc_ff_config = estimate_adrc_config(
            airframe_ff,
            rocket_cfg_ff,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        adrc_ff_config.use_observations = use_imu
        ff_config = WindFeedforwardConfig(K_ff=args.ff_gain)
        print(
            f"  Estimated b0={adrc_ff_config.b0:.1f} (b0_per_pa={adrc_ff_config.b0_per_pa:.4f})"
        )

        if args.rls_b0:
            ff_rls_label = f"ADRC+FF+RLS{imu_suffix}"
            print(f"\nEvaluating {ff_rls_label} (K_ff={args.ff_gain})...")
            b0_est_ff = B0Estimator(B0EstimatorConfig(b0_init=adrc_ff_config.b0))
            adrc_ff_rls_ctrl = WindFeedforwardADRC(
                adrc_ff_config, ff_config, b0_estimator=b0_est_ff
            )
            adrc_ff_rls_results = evaluate_controller(
                config,
                ff_rls_label,
                args.wind_levels,
                args.n_episodes,
                controller=adrc_ff_rls_ctrl,
                use_observations=use_imu,
                collect_spin_series=collect_spin,
            )
            all_results[ff_rls_label] = adrc_ff_rls_results
        else:
            ff_label = f"ADRC+FF{imu_suffix}"
            print(f"\nEvaluating {ff_label} (K_ff={args.ff_gain})...")
            adrc_ff_ctrl = WindFeedforwardADRC(adrc_ff_config, ff_config)
            adrc_ff_results = evaluate_controller(
                config,
                ff_label,
                args.wind_levels,
                args.n_episodes,
                controller=adrc_ff_ctrl,
                use_observations=use_imu,
                collect_spin_series=collect_spin,
            )
            all_results[ff_label] = adrc_ff_results

    # ADRC + NN Wind Estimator (if model path provided)
    if args.adrc_nn and not args.pid_only:
        nn_label = f"ADRC+NN{imu_suffix}"
        print(f"\nEvaluating {nn_label} (model={args.adrc_nn})...")
        airframe = config.physics.resolve_airframe()
        rocket_cfg = config.physics
        adrc_nn_config = estimate_adrc_config(
            airframe,
            rocket_cfg,
            omega_c=args.adrc_omega_c,
            omega_o=args.adrc_omega_o,
        )
        adrc_nn_config.use_observations = use_imu
        estimator, est_config = load_estimator(args.adrc_nn)
        print(
            f"  Estimated b0={adrc_nn_config.b0:.1f} (b0_per_pa={adrc_nn_config.b0_per_pa:.4f})"
        )
        print(
            f"  NN estimator: window={est_config.window_size}, hidden={est_config.hidden_size}"
        )
        adrc_nn_ctrl = NNFeedforwardADRC(
            adrc_nn_config,
            estimator,
            K_ff=args.ff_gain,
            warmup_steps=est_config.warmup_steps,
        )
        adrc_nn_results = evaluate_controller(
            config,
            nn_label,
            args.wind_levels,
            args.n_episodes,
            controller=adrc_nn_ctrl,
            use_observations=use_imu,
            collect_spin_series=collect_spin,
        )
        all_results[nn_label] = adrc_nn_results

    # PPO (if provided)
    if args.ppo and not args.pid_only:
        print(f"\nEvaluating PPO model: {args.ppo}")
        ppo_model, ppo_vec = load_rl_model(args.ppo, "ppo", config)
        ppo_results = evaluate_controller(
            config,
            "PPO",
            args.wind_levels,
            args.n_episodes,
            model=ppo_model,
            vec_normalize=ppo_vec,
            collect_spin_series=collect_spin,
        )
        all_results["PPO"] = ppo_results

    # SAC (if provided)
    if args.sac and not args.pid_only:
        print(f"\nEvaluating SAC model: {args.sac}")
        sac_model, sac_vec = load_rl_model(args.sac, "sac", config)
        sac_results = evaluate_controller(
            config,
            "SAC",
            args.wind_levels,
            args.n_episodes,
            model=sac_model,
            vec_normalize=sac_vec,
            collect_spin_series=collect_spin,
        )
        all_results["SAC"] = sac_results

    # Residual SAC (if provided)
    if args.residual_sac and not args.pid_only:
        print(f"\nEvaluating Residual SAC model: {args.residual_sac}")
        # Load config from model directory if available (has use_residual_pid: true),
        # otherwise fall back to the main config
        residual_model_path = Path(args.residual_sac)
        residual_config_path = residual_model_path.parent / "config.yaml"
        if residual_config_path.exists():
            residual_config = load_config(str(residual_config_path))
            print(f"  Using config from: {residual_config_path}")
        else:
            residual_config = config
            print("  No config.yaml in model dir, using --config")
        rsac_model, rsac_vec = load_rl_model(
            args.residual_sac,
            "sac",
            residual_config,
        )
        rsac_results = evaluate_controller(
            residual_config,
            "Residual SAC",
            args.wind_levels,
            args.n_episodes,
            model=rsac_model,
            vec_normalize=rsac_vec,
            collect_spin_series=collect_spin,
        )
        all_results["Residual SAC"] = rsac_results

    # DOB SAC (if provided)
    if args.dob_sac and not args.pid_only:
        print(f"\nEvaluating DOB SAC model: {args.dob_sac}")
        # Load config from model directory if available (has use_disturbance_observer: true),
        # otherwise fall back to the main config
        dob_model_path = Path(args.dob_sac)
        dob_config_path = dob_model_path.parent / "config.yaml"
        if dob_config_path.exists():
            dob_config = load_config(str(dob_config_path))
            print(f"  Using config from: {dob_config_path}")
            dob_enabled = getattr(dob_config.physics, "use_disturbance_observer", False)
            print(f"  Disturbance Observer: {'enabled' if dob_enabled else 'disabled'}")
        else:
            dob_config = config
            print("  No config.yaml in model dir, using --config")
        dob_model, dob_vec = load_rl_model(
            args.dob_sac,
            "sac",
            dob_config,
        )
        dob_results = evaluate_controller(
            dob_config,
            "DOB SAC",
            args.wind_levels,
            args.n_episodes,
            model=dob_model,
            vec_normalize=dob_vec,
            collect_spin_series=collect_spin,
        )
        all_results["DOB SAC"] = dob_results

    # Print comparison
    print_comparison_table(all_results)

    # Video quality analysis
    if args.video_quality:
        from video_quality_metric import CameraPreset, print_video_quality_table

        preset_map = {
            "1080p60": [CameraPreset.runcam_1080p60()],
            "4k30": [CameraPreset.runcam_4k30()],
            "1080p120": [CameraPreset.runcam_1080p120()],
            "all": CameraPreset.all_presets(),
        }
        camera_presets = preset_map[args.camera_preset]
        dt = getattr(config.environment, "dt", 0.01)
        print_video_quality_table(all_results, dt, camera_presets)

    # Plot
    plot_comparison(all_results, args.save_plot)


if __name__ == "__main__":
    main()
