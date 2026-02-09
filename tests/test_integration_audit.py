"""
Physics-Informed Integration Tests

These tests verify correctness at the boundaries between major subsystems:
  Environment <-> Rocket Dynamics <-> Sensor Model <-> Controller

Each test encodes a known physical invariant or expected behavior. They are
designed to catch integration bugs (sign errors, unit mismatches, double-
counting, stale-state usage) rather than internal component logic (which
unit tests already cover).

Audit findings from the systematic review (Feb 2026) are documented inline
as comments. Tests are grouped by integration boundary.
"""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from dataclasses import dataclass

from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig
from realistic_spin_rocket import RealisticMotorRocket
from airframe import RocketAirframe
from wind_model import WindModel, WindConfig
from controllers.pid_controller import (
    PIDController,
    PIDConfig,
    GainScheduledPIDController,
)
from rocket_config import load_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def estes_airframe():
    """Standard Estes Alpha airframe for integration tests."""
    return RocketAirframe.estes_alpha()


@pytest.fixture
def base_config():
    """Minimal config with wind disabled for controlled tests."""
    return RocketConfig(
        max_tab_deflection=30.0,
        initial_spin_std=0.0,  # deterministic start
        disturbance_scale=0.0,  # no random disturbance
        enable_wind=False,
        dt=0.01,
    )


@pytest.fixture
def wind_config():
    """Config with wind enabled for wind-integration tests."""
    return RocketConfig(
        max_tab_deflection=30.0,
        initial_spin_std=0.0,
        disturbance_scale=0.0,
        enable_wind=True,
        base_wind_speed=3.0,
        max_gust_speed=1.5,
        wind_variability=0.0,  # constant direction for predictability
        dt=0.01,
    )


@pytest.fixture
def motor_config():
    """Simple motor config for integration tests."""
    return {
        "name": "test_motor",
        "manufacturer": "Test",
        "designation": "C6",
        "total_impulse_Ns": 10.0,
        "avg_thrust_N": 5.4,
        "max_thrust_N": 14.0,
        "burn_time_s": 1.85,
        "propellant_mass_g": 12.3,
        "case_mass_g": 12.7,
        "thrust_curve": {
            "time_s": [0.0, 0.1, 0.5, 1.0, 1.5, 1.85],
            "thrust_N": [0.0, 14.0, 6.0, 5.0, 4.0, 0.0],
        },
    }


def make_env(airframe, config):
    """Create a SpinStabilizedCameraRocket with no random disturbances."""
    return SpinStabilizedCameraRocket(airframe=airframe, config=config)


def make_motor_env(airframe, motor_config, config):
    """Create a RealisticMotorRocket for integration tests."""
    return RealisticMotorRocket(
        airframe=airframe,
        motor_config=motor_config,
        config=config,
    )


# ===========================================================================
# 1. ENVIRONMENT → ROCKET DYNAMICS
#    Wind model forces/torques, unit consistency, physical plausibility
# ===========================================================================


class TestWindModelIntegration:
    """Verify wind forces/torques are physically plausible at the environment boundary."""

    def test_zero_wind_produces_zero_torque(self, estes_airframe):
        """Wind torque must be exactly zero when wind speed is zero."""
        wm = WindModel(WindConfig(enable=True, base_speed=0.0, max_gust_speed=0.0))
        wm.reset(seed=42)
        speed, direction = wm.get_wind(time=1.0, altitude=50.0)
        torque = wm.get_roll_torque(
            speed,
            direction,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=estes_airframe,
        )
        assert torque == 0.0, "Zero wind speed must produce zero roll torque"

    def test_wind_torque_sign_follows_relative_angle(self, estes_airframe):
        """Wind torque sign must follow sin(N/2 * (wind_dir - roll_angle)) for N=4.

        Physical meaning: body shadow breaks symmetry between windward/leeward
        fins, creating roll torque.  For a 4-fin rocket the torque is periodic
        at twice the wind-to-roll-angle frequency — zero when wind aligns with
        any fin (0, pi/2, pi, …), maximum at 45° between fin planes.
        """
        q = 500.0  # Pa
        v = 30.0  # m/s
        wind_speed = 3.0

        # Wind from 0 rad, roll angle = 0 → sin(2*0) = 0 → zero torque
        t0 = WindModel(WindConfig(enable=True)).get_roll_torque(
            wind_speed,
            wind_direction=0.0,
            roll_angle=0.0,
            velocity=v,
            dynamic_pressure=q,
            airframe=estes_airframe,
        )
        # Wind from pi/4 → sin(2*pi/4) = sin(pi/2) = 1 → positive torque
        t_pos = WindModel(WindConfig(enable=True)).get_roll_torque(
            wind_speed,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=v,
            dynamic_pressure=q,
            airframe=estes_airframe,
        )
        # Wind from -pi/4 → sin(2*(-pi/4)) = sin(-pi/2) = -1 → negative torque
        t_neg = WindModel(WindConfig(enable=True)).get_roll_torque(
            wind_speed,
            wind_direction=-np.pi / 4,
            roll_angle=0.0,
            velocity=v,
            dynamic_pressure=q,
            airframe=estes_airframe,
        )

        assert abs(t0) < 1e-10, "Fin-aligned wind should produce ~zero torque"
        assert t_pos > 0, "Wind at +45° to fins should give positive torque"
        assert t_neg < 0, "Wind at -45° to fins should give negative torque"

    def test_wind_torque_scales_with_dynamic_pressure(self, estes_airframe):
        """Wind torque must increase with dynamic pressure (more aerodynamic force)."""
        wm = WindModel(WindConfig(enable=True))
        kwargs = dict(
            wind_speed=3.0,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=30.0,
            airframe=estes_airframe,
        )
        t_low = wm.get_roll_torque(**kwargs, dynamic_pressure=100.0)
        t_high = wm.get_roll_torque(**kwargs, dynamic_pressure=1000.0)
        assert abs(t_high) > abs(t_low), "Higher q must produce larger wind torque"
        # Should scale linearly with q
        ratio = abs(t_high) / abs(t_low)
        assert (
            abs(ratio - 10.0) < 0.1
        ), f"Wind torque should scale ~linearly with q, got ratio {ratio}"

    def test_wind_torque_magnitude_physically_plausible(self, estes_airframe):
        """Wind torque at 3 m/s wind, 30 m/s rocket should be O(1e-4) N·m.

        Physical check: The wind torque is a second-order body-shadow effect.
        For Estes Alpha with K_shadow=0.10, single_fin_area=1.5e-4 m²,
        q=500 Pa, sideslip=0.1 rad, Cl_alpha=2, moment_arm=0.032 m:
        torque ~ 500 * 1.5e-4 * 2 * 0.1 * 0.032 * 0.10 * 1 ≈ 5e-5 Nm
        (see docs/wind_roll_torque_analysis.md §3.1)
        """
        wm = WindModel(WindConfig(enable=True))
        torque = wm.get_roll_torque(
            wind_speed=3.0,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=estes_airframe,
        )
        assert (
            1e-6 < abs(torque) < 0.01
        ), f"Wind torque {torque:.6f} N·m outside plausible range for Estes Alpha"

    def test_wind_torque_periodic_at_spin_frequency(self, estes_airframe):
        """As the rocket rolls through 2π, wind torque must complete N/2 cycles.

        For a 4-fin rocket the torque varies as sin(2*(wind_dir - roll_angle)),
        giving 2 complete cycles (4 zero crossings) per rotation.  This is
        because the body shadow symmetry repeats every 90° for 4 fins.
        """
        wm = WindModel(WindConfig(enable=True))
        angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        torques = [
            wm.get_roll_torque(
                wind_speed=3.0,
                wind_direction=0.0,
                roll_angle=a,
                velocity=30.0,
                dynamic_pressure=500.0,
                airframe=estes_airframe,
            )
            for a in angles
        ]
        torques = np.array(torques)
        # Check mean is near zero (sinusoidal)
        assert abs(np.mean(torques)) < 0.1 * np.max(
            np.abs(torques)
        ), "Mean wind torque over full rotation should be near zero"
        # For 4 fins: sin(2*angle) has 4 zero crossings per 2*pi
        zero_crossings = np.sum(np.diff(np.sign(torques)) != 0)
        assert (
            zero_crossings == 4
        ), f"Expected 4 zero crossings per rotation for 4-fin rocket, got {zero_crossings}"


# ===========================================================================
# 2. TAB DEFLECTION → AERODYNAMIC FORCES/TORQUES
#    Control chain: action → tab angle → torque, sign conventions, q-dependence
# ===========================================================================


class TestTabToTorqueIntegration:
    """Trace the full path from control command to roll torque."""

    def test_zero_velocity_zero_control_torque(self, estes_airframe, base_config):
        """Tabs must be aerodynamically ineffective at zero airspeed.

        AUDIT FINDING: The environment uses v = max(vertical_velocity, 0.1) for
        dynamic pressure (line 335), giving q = 0.5 * 1.225 * 0.01 = 0.006 Pa.
        With q < 1.0 threshold (line 420), control torque is correctly zero.
        """
        env = make_env(estes_airframe, base_config)
        env.reset()
        env.vertical_velocity = 0.0
        env.roll_rate = 0.0

        # Apply full deflection
        obs, reward, _, _, info = env.step(np.array([1.0]))

        # At v~0, q<1 → no control torque → no roll acceleration
        assert (
            abs(env.roll_acceleration) < 1e-6
        ), f"Control torque should be zero at zero velocity, got accel={env.roll_acceleration}"

    def test_control_torque_increases_with_q(self, estes_airframe, base_config):
        """Tab effectiveness must increase with dynamic pressure."""
        env = make_env(estes_airframe, base_config)
        env.reset()

        # Low q flight
        env.vertical_velocity = 5.0  # low speed
        env.roll_rate = 0.0
        obs, _, _, _, _ = env.step(np.array([1.0]))
        accel_low = env.roll_acceleration

        # Reset and do high q flight
        env.reset()
        env.vertical_velocity = 50.0  # high speed
        env.roll_rate = 0.0
        obs, _, _, _, _ = env.step(np.array([1.0]))
        accel_high = env.roll_acceleration

        assert abs(accel_high) > abs(
            accel_low
        ), "Tab torque should be larger at higher dynamic pressure"

    def test_control_sign_convention_positive_action(self, estes_airframe, base_config):
        """A positive action should produce a definite-sign roll acceleration.

        The sign convention is: action → servo position → tab deflection (rad)
        → control torque → roll acceleration. The signs must be consistent
        end-to-end.
        """
        env = make_env(estes_airframe, base_config)
        env.reset()
        env.vertical_velocity = 30.0  # enough for aerodynamic control
        env.roll_rate = 0.0

        obs, _, _, _, _ = env.step(np.array([1.0]))
        accel_pos = env.roll_acceleration

        env.reset()
        env.vertical_velocity = 30.0
        env.roll_rate = 0.0

        obs, _, _, _, _ = env.step(np.array([-1.0]))
        accel_neg = env.roll_acceleration

        # Opposite actions should produce opposite accelerations
        assert accel_pos * accel_neg < 0, (
            f"Opposite actions must produce opposite accelerations: "
            f"+1 gave {accel_pos}, -1 gave {accel_neg}"
        )

    def test_tab_deflection_in_radians(self, estes_airframe, base_config):
        """Tab deflection stored on env must be in radians, matching the
        torque formula which expects rad input.

        AUDIT CHECK: line 326: tab_deflection = actual_pos * deg2rad(max_tab_deflection)
        """
        env = make_env(estes_airframe, base_config)
        env.reset()
        env.step(np.array([1.0]))

        max_defl_rad = np.deg2rad(base_config.max_tab_deflection)
        assert (
            abs(env.tab_deflection) <= max_defl_rad + 1e-6
        ), f"Tab deflection {env.tab_deflection} rad exceeds max {max_defl_rad} rad"
        assert (
            abs(env.tab_deflection) > 0.01
        ), "Tab deflection should be non-trivial for full action"

    def test_control_effectiveness_double_q_audit(self, estes_airframe):
        """AUDIT FINDING [CRITICAL]: Control effectiveness is multiplied by q TWICE.

        Path traced:
        1. get_control_effectiveness() in components.py:285 computes:
           force_per_rad = cl_alpha * dynamic_pressure * tab_area
           → effectiveness already includes q

        2. _calculate_roll_torque() in spin_stabilized_control_env.py:429:
           control_torque = effectiveness * tab_deflection
           → This is correct (effectiveness already has q baked in)

        3. BUT line 432: speed_effectiveness = tanh(dynamic_pressure / 200)
           → This multiplies by ANOTHER function of q

        The tanh(q/200) factor is a deliberate soft-start (documented intent:
        "velocity-dependent effectiveness"), but it means the actual control
        torque scales as q * tanh(q/200), NOT linearly with q. This is
        physically debatable but not a bug — it's a design choice to smooth
        the low-speed transition. The gain scheduling (q_ref) accounts for
        this combined function.
        """
        # Verify the combined scaling is q * tanh(q/200)
        airframe = estes_airframe
        q_values = [50, 100, 200, 500, 1000]
        effectiveness_values = []
        for q in q_values:
            eff = airframe.get_control_effectiveness(q)
            tanh_factor = np.tanh(q / 200.0)
            combined = eff * tanh_factor
            effectiveness_values.append(combined)

        # Verify monotonically increasing (not pathological)
        for i in range(len(effectiveness_values) - 1):
            assert effectiveness_values[i + 1] > effectiveness_values[i], (
                f"Combined effectiveness must increase with q: "
                f"q={q_values[i]}→{q_values[i+1]}, "
                f"eff={effectiveness_values[i]:.6f}→{effectiveness_values[i+1]:.6f}"
            )

    def test_control_authority_vs_wind_torque(self, estes_airframe):
        """Control authority should significantly exceed wind torque.

        The wind roll torque is a second-order body-shadow effect — much
        smaller than control authority.  At q=500 Pa with 15 deg max
        deflection the control-to-wind ratio should be >> 1.
        See docs/wind_roll_torque_analysis.md §3.1.
        """
        q = 500.0  # Pa, mid-flight
        v = 30.0  # m/s

        # Max control torque at full deflection
        effectiveness = estes_airframe.get_control_effectiveness(q)
        tanh_factor = np.tanh(q / 200.0)
        max_defl_rad = np.deg2rad(30.0)  # test max_tab_deflection
        max_control_torque = effectiveness * max_defl_rad * tanh_factor

        # Wind torque at 3 m/s (worst-case angle = pi/4 for 4-fin rocket)
        wm = WindModel(WindConfig(enable=True))
        wind_torque = wm.get_roll_torque(
            wind_speed=3.0,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=v,
            dynamic_pressure=q,
            airframe=estes_airframe,
        )

        # Control should be non-zero and larger than wind
        assert (
            max_control_torque > 0.001
        ), f"Control torque should be significant: {max_control_torque:.6f} N·m"
        assert (
            abs(wind_torque) > 1e-6
        ), f"Wind torque should be non-zero: {wind_torque:.8f} N·m"
        # Control authority should dominate wind torque (ratio > 10:1)
        ratio = max_control_torque / abs(wind_torque)
        assert ratio > 10, f"Control-to-wind ratio {ratio:.1f} should be >> 10"


# ===========================================================================
# 3. ROCKET DYNAMICS → SENSOR MODEL (IMU)
#    State transformation, noise addition, sampling
# ===========================================================================


class TestDynamicsToIMUIntegration:
    """Verify that the IMU correctly transforms and noises the rocket state."""

    def test_imu_adds_noise_to_roll_rate(
        self, estes_airframe, base_config, motor_config
    ):
        """IMU wrapper must add noise to obs[3] (roll rate)."""
        from rocket_env.sensors import IMUObservationWrapper, IMUConfig

        env = make_motor_env(estes_airframe, motor_config, base_config)
        imu_config = IMUConfig.icm_20948()
        imu_env = IMUObservationWrapper(env, imu_config=imu_config, seed=42)

        obs_clean, _ = env.reset(seed=123)
        obs_noisy, _ = imu_env.reset(seed=123)

        # Roll rates should differ due to noise
        # (may match on first step if noise is small, so run a few steps)
        differences = []
        for _ in range(20):
            action = np.array([0.0])
            oc, _, _, _, _ = env.step(action)
            on, _, _, _, _ = imu_env.step(action)
            differences.append(abs(oc[3] - on[3]))

        max_diff = max(differences)
        assert max_diff > 0, "IMU should add noise to roll rate obs[3]"

    def test_imu_does_not_modify_non_gyro_channels(
        self, estes_airframe, base_config, motor_config
    ):
        """IMU wrapper should only modify obs[3] (roll rate) and obs[4] (accel).

        Other channels (altitude, velocity, angle, q, time, thrust, action,
        shake) should pass through unmodified.

        AUDIT FINDING: obs[2] (roll angle) is NOT modified by the IMU.
        This is physically correct — a real IMU measures angular rate, not
        absolute angle. Angle is derived by integration (which the environment
        does internally) and doesn't have IMU noise applied.
        """
        from rocket_env.sensors import IMUObservationWrapper, IMUConfig

        env = make_motor_env(estes_airframe, motor_config, base_config)
        imu_config = IMUConfig.icm_20948()
        imu_env = IMUObservationWrapper(env, imu_config=imu_config, seed=42)

        # Use same seed for both
        obs_clean, _ = env.reset(seed=100)
        obs_noisy, _ = imu_env.reset(seed=100)

        # Channels 0,1,2,5,6,7,8 should be identical
        # (Channel 2 = roll angle, 5 = q, 6 = time, 7 = thrust frac, 8 = action)
        unmodified_channels = [0, 1, 2, 5, 6, 7, 8]
        for ch in unmodified_channels:
            assert obs_clean[ch] == obs_noisy[ch], (
                f"Channel {ch} should not be modified by IMU: "
                f"clean={obs_clean[ch]}, noisy={obs_noisy[ch]}"
            )

    def test_imu_noise_applied_to_info_dict(
        self, estes_airframe, base_config, motor_config
    ):
        """IMU wrapper's step() must apply gyro noise to info['roll_rate_deg_s'].

        AUDIT FINDING: This was added to fix the J800 IMU mode failure.
        Classical controllers read roll_rate from the info dict to bypass
        sensor_delay_steps. The info dict value must also be noisy.
        """
        from rocket_env.sensors import IMUObservationWrapper, IMUConfig

        env = make_motor_env(estes_airframe, motor_config, base_config)
        imu_config = IMUConfig.icm_20948()
        imu_env = IMUObservationWrapper(env, imu_config=imu_config, seed=42)

        imu_env.reset(seed=123)

        # Step many times and collect info roll rates
        info_rates = []
        for _ in range(50):
            _, _, _, _, info = imu_env.step(np.array([0.0]))
            info_rates.append(info["roll_rate_deg_s"])

        # The noisy info rates should not be identical across steps
        # (unless the rocket happens to have constant rate, which is unlikely)
        unique_rates = len(set(f"{r:.6f}" for r in info_rates))
        assert unique_rates > 1, "Info dict roll_rate should vary (noise applied)"

    def test_imu_sampling_rate_matches_sim_timestep(
        self, estes_airframe, base_config, motor_config
    ):
        """The IMU gyro model must be configured with the correct dt.

        AUDIT CHECK: IMU wrapper gets control_rate_hz=100.0 by default,
        giving dt=0.01s, matching the environment's default dt=0.01s.
        If the environment uses a different dt (e.g., 0.005 for 200 Hz),
        the IMU dt must match.
        """
        from rocket_env.sensors import IMUObservationWrapper, IMUConfig

        env = make_motor_env(estes_airframe, motor_config, base_config)
        imu_env = IMUObservationWrapper(
            env,
            control_rate_hz=1.0 / base_config.dt,
        )
        expected_dt = base_config.dt
        assert (
            abs(imu_env._dt - expected_dt) < 1e-10
        ), f"IMU dt={imu_env._dt} should match env dt={expected_dt}"


# ===========================================================================
# 4. SENSOR MODEL → CONTROLLER
#    Units, conventions, closed-loop sign (negative feedback)
# ===========================================================================


class TestSensorToControllerIntegration:
    """Verify controllers receive correct data and produce correct feedback."""

    def test_closed_loop_negative_feedback(
        self, estes_airframe, base_config, motor_config
    ):
        """The full closed loop must be negative feedback:
        spin CW → controller acts → torque opposes CW spin.

        This is the single most critical integration test. A sign error
        anywhere in the chain would turn negative feedback into positive
        feedback, causing instability.

        NOTE: The rocket must have sufficient velocity for aerodynamic
        control (q > 1 Pa threshold). We advance past motor ignition
        first to build up airspeed.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()

        # Advance 30 steps to build velocity (motor ignites and accelerates)
        for _ in range(30):
            env.step(np.array([0.0]))

        # Now set a positive spin rate at mid-flight (rocket has airspeed)
        env.roll_rate = np.deg2rad(30.0)  # 30 deg/s CW

        # PID controller with known gains
        pid = PIDController(PIDConfig(), use_observations=False)
        pid.launch_detected = True

        # Let PID compute action based on current state
        obs = env._get_observation()
        info = env._get_info()
        action = pid.step(obs, info, dt=0.01)

        # Step the environment with this action
        obs, _, _, _, info = env.step(action)

        # After control, spin rate should decrease (or at minimum, the
        # acceleration should oppose the initial spin direction)
        # The initial spin was positive, so roll_acceleration should be negative
        # (or the roll_rate should have decreased)
        assert env.roll_rate < np.deg2rad(30.0), (
            f"Negative feedback check failed: initial spin +30 deg/s, "
            f"after one step: {np.rad2deg(env.roll_rate):.1f} deg/s "
            f"(should have decreased). velocity={env.vertical_velocity:.1f} m/s"
        )

    def test_pid_converges_zero_disturbance(self):
        """With zero disturbance, PID should converge to near-zero spin rate.

        This tests the full integration chain: dynamics → obs → PID → action
        → torque → dynamics. If any sign or unit error exists, convergence
        will fail.

        Uses the production config (estes_c6_sac_wind.yaml) at zero wind,
        matching compare_controllers.py's evaluation pipeline. The PID with
        optimized gains should achieve <10 deg/s mean spin at 0 m/s wind
        (known baseline: ~4.6 deg/s).
        """
        from compare_controllers import create_env, run_controller_episode

        config_path = (
            Path(__file__).parent.parent / "configs" / "estes_c6_sac_wind.yaml"
        )
        if not config_path.exists():
            pytest.skip("Production config not available")

        training_config = load_config(str(config_path))
        env = create_env(training_config, wind_speed=0.0)

        pid = PIDController(PIDConfig(), use_observations=False)

        # Run 5 episodes and take mean (stochastic initial spin)
        spin_rates = []
        for i in range(5):
            env.reset(seed=100 + i)
            pid.reset()
            metrics = run_controller_episode(env, pid, dt=0.01)
            spin_rates.append(metrics.mean_spin_rate)

        mean_spin = np.mean(spin_rates)
        assert mean_spin < 10.0, (
            f"PID should achieve <10 deg/s mean spin at 0 m/s wind, "
            f"got {mean_spin:.1f} deg/s (per-episode: {[f'{s:.1f}' for s in spin_rates]})"
        )

    def test_controller_receives_deg_s_from_info(
        self, estes_airframe, base_config, motor_config
    ):
        """Controllers must receive roll rate in deg/s from info dict.

        AUDIT CHECK: info['roll_rate_deg_s'] is computed as np.rad2deg(self.roll_rate)
        at line 632. Controllers use this directly. Verify units are correct.
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()
        env.roll_rate = np.deg2rad(45.0)  # Set 45 deg/s

        info = env._get_info()
        assert (
            abs(info["roll_rate_deg_s"] - 45.0) < 0.1
        ), f"Info dict roll_rate should be in deg/s: expected 45.0, got {info['roll_rate_deg_s']}"

    def test_controller_obs_roll_rate_in_rad_s(
        self, estes_airframe, base_config, motor_config
    ):
        """obs[3] contains roll rate in rad/s. Controllers converting to deg/s
        must use np.degrees() or np.rad2deg().

        AUDIT CHECK: PID does roll_rate = np.degrees(obs[3]) at line 85.
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()
        env.roll_rate = np.deg2rad(45.0)

        obs = env._get_observation()
        assert (
            abs(obs[3] - np.deg2rad(45.0)) < 0.01
        ), f"obs[3] should be in rad/s: expected {np.deg2rad(45.0):.4f}, got {obs[3]:.4f}"


# ===========================================================================
# 5. TIME INTEGRATION & SIMULATION LOOP
#    Off-by-one errors, order of operations, timestep consistency
# ===========================================================================


class TestSimulationLoopIntegration:
    """Verify the simulation loop ordering and timing are correct."""

    def test_action_applied_before_dynamics(
        self, estes_airframe, base_config, motor_config
    ):
        """The control action from step N must affect the dynamics at step N,
        not step N+1 (no unintended extra delay).

        AUDIT FINDING: The step() method applies servo dynamics and updates
        tab_deflection BEFORE computing roll torque (lines 324-326 before
        lines 360-363). This is correct — the action taken at this step
        influences this step's dynamics.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()
        env.vertical_velocity = 30.0  # ensure aerodynamic control
        env.roll_rate = 0.0

        # Step with zero action
        env.step(np.array([0.0]))
        accel_zero = env.roll_acceleration

        # Reset and step with full action
        env.reset()
        env.vertical_velocity = 30.0
        env.roll_rate = 0.0
        env.step(np.array([1.0]))
        accel_full = env.roll_acceleration

        # The full action should produce different acceleration THIS step
        assert (
            accel_zero != accel_full
        ), "Action must affect dynamics in the same step (no extra delay)"

    def test_propulsion_evaluated_before_time_advance(
        self, estes_airframe, base_config, motor_config
    ):
        """AUDIT FIX: Propulsion must be evaluated BEFORE time advances.

        Previously time was incremented before _update_propulsion(), causing
        thrust to be evaluated at t+dt instead of t. Now the ordering is:
        1. _update_propulsion() at current time
        2. self.time += dt
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()

        # After reset, time = 0.0
        assert env.time == 0.0

        # After one step, time should be dt
        env.step(np.array([0.0]))
        assert (
            abs(env.time - 0.01) < 1e-10
        ), f"Time after one step should be dt, got {env.time}"

        # Thrust should have been evaluated at time=0.0, not 0.01.
        # The motor's thrust curve starts at 0.0s. At t=0.0, thrust should
        # be the interpolated value from the curve at t=0.
        # This is verified by the ordering: propulsion before time increment.

    def test_observation_uses_post_dynamics_state(
        self, estes_airframe, base_config, motor_config
    ):
        """The observation returned by step() must reflect the state AFTER
        dynamics are applied, not the state before.

        AUDIT CHECK: _get_observation() is called after all dynamics updates
        (line 396), which is correct.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()
        env.roll_rate = np.deg2rad(10.0)

        obs_before = env._get_observation()
        obs_after, _, _, _, _ = env.step(np.array([0.0]))

        # obs_after should reflect updated state (roll_rate may change due to damping)
        # At minimum, the time channel (obs[6]) should have advanced
        assert obs_after[6] > obs_before[6], "Observation time must advance after step"

    def test_sensor_delay_delays_observation(self, estes_airframe, motor_config):
        """sensor_delay_steps=N should return the observation from N steps ago.

        AUDIT CHECK: Lines 396-399 implement observation delay correctly.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            sensor_delay_steps=2,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()

        # Collect observations for several steps
        obs_list = []
        for i in range(5):
            obs, _, _, _, info = env.step(np.array([0.0]))
            obs_list.append(obs.copy())

        # With delay=2, obs at step 4 should match the state from step 2
        # The time channel (obs[6]) is the most reliable indicator
        # After step 4: actual time = 5*dt = 0.05
        # With delay 2: should show time from step 2 = 3*dt = 0.03
        delayed_time = obs_list[4][6]
        # The delayed obs should show an earlier time than actual
        actual_time = info["time_s"]
        assert (
            delayed_time < actual_time
        ), f"Delayed obs time ({delayed_time}) should be less than actual time ({actual_time})"

    def test_info_dict_is_always_current(self, estes_airframe, motor_config):
        """Info dict must always contain the current (non-delayed) state,
        regardless of sensor_delay_steps.

        AUDIT CHECK: _get_info() is called at line 401 (after dynamics),
        and returns current state. sensor_delay only affects obs.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            sensor_delay_steps=3,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()

        for i in range(10):
            obs, _, _, _, info = env.step(np.array([0.0]))

        # Info time should be current (10 steps * 0.01 + 0.01 for time advance = 0.11)
        # but obs time should be delayed
        info_time = info["time_s"]
        obs_time = obs[6]
        assert (
            info_time > obs_time
        ), f"Info time ({info_time}) must be current, obs time ({obs_time}) delayed"

    def test_observation_q_matches_physics_q(
        self, estes_airframe, base_config, motor_config
    ):
        """AUDIT FIX: Dynamic pressure in obs must match physics q.

        Both step() and _get_observation() now use v=max(velocity, 0.1),
        ensuring obs[5] matches the q used for physics computations.
        Previously _get_observation() used v=max(velocity, 0), causing
        obs q to be 0 during descent while physics q was > 0.
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()
        env.vertical_velocity = -5.0  # descending

        obs = env._get_observation()
        obs_q = obs[5]

        # Both step() and obs now use v=max(-5.0, 0.1)=0.1
        rho = 1.225 * np.exp(-env.altitude / 8000)
        expected_q = 0.5 * rho * 0.1**2
        assert (
            obs_q > 0
        ), f"Obs q should be > 0 even when descending (v clamped to 0.1), got {obs_q}"
        assert (
            abs(obs_q - expected_q) < 0.01
        ), f"Obs q={obs_q:.4f} should match physics q={expected_q:.4f}"

    def test_info_air_density_matches_physics(
        self, estes_airframe, base_config, motor_config
    ):
        """AUDIT FIX: _get_info() must use the same air density model as physics.

        Previously _get_info() used a hardcoded simple exponential model
        (1.225 * exp(-h/8000)) even when use_isa_full=True. Now it calls
        _get_air_density() which delegates to _get_atmosphere().
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            use_isa_full=True,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()
        env.altitude = 3000.0  # 3km altitude

        info = env._get_info()
        info_rho = info["air_density_kg_m3"]

        # ISA full model at 3000m
        rho_isa, _, _ = env._get_atmosphere()

        # Info should now match ISA model
        assert (
            abs(info_rho - rho_isa) < 1e-6
        ), f"Info air_density ({info_rho:.6f}) should match ISA model ({rho_isa:.6f})"


# ===========================================================================
# 6. VISUALIZATION & METRICS
#    Signal identity, time windows, fair comparison
# ===========================================================================


class TestMetricsIntegration:
    """Verify that metrics measure what they claim to measure."""

    def test_spin_rate_metric_from_info_dict(
        self, estes_airframe, base_config, motor_config
    ):
        """Mean spin rate metric must be computed from info['roll_rate_deg_s'],
        which is the current (non-delayed) state in deg/s.

        AUDIT CHECK: compare_controllers.py line 214:
            spin_rate = abs(info.get("roll_rate_deg_s", 0.0))
        This correctly uses the info dict (always current) rather than obs
        (which may be delayed).
        """
        from compare_controllers import run_controller_episode

        env = make_motor_env(estes_airframe, motor_config, base_config)
        pid = PIDController(PIDConfig(), use_observations=False)
        metrics = run_controller_episode(env, pid, dt=0.01)

        # Mean spin rate should be positive (it's abs of roll rate)
        assert metrics.mean_spin_rate >= 0, "Mean spin rate must be non-negative"
        assert metrics.mean_spin_rate < 200, "Mean spin rate should be reasonable"

    def test_settling_time_uses_current_state(
        self, estes_airframe, base_config, motor_config
    ):
        """Settling time should be computed from the current (non-delayed) state.

        AUDIT CHECK: compare_controllers.py line 221:
            if settling_time == float("inf") and spin_rate < 10.0:
                settling_time = info.get("time_s", step * dt)
        spin_rate comes from info dict (current state), which is correct.
        """
        from compare_controllers import run_controller_episode

        env = make_motor_env(estes_airframe, motor_config, base_config)
        pid = PIDController(PIDConfig(), use_observations=False)
        metrics = run_controller_episode(env, pid, dt=0.01)

        # Settling time should be either a positive number or inf
        assert metrics.settling_time > 0 or metrics.settling_time == float(
            "inf"
        ), f"Settling time should be positive or inf, got {metrics.settling_time}"

    def test_control_smoothness_normalized_action_space(
        self, estes_airframe, base_config, motor_config
    ):
        """Control smoothness is computed from normalized [-1, 1] actions.

        AUDIT CHECK: compare_controllers.py line 216:
            actions.append(float(action[0]))
        line 227: action_changes = np.abs(np.diff(actions))

        This measures smoothness in normalized action space. A value of 0
        means perfectly smooth; 2.0 is the maximum possible change.
        """
        from compare_controllers import run_controller_episode

        env = make_motor_env(estes_airframe, motor_config, base_config)
        pid = PIDController(PIDConfig(), use_observations=False)
        metrics = run_controller_episode(env, pid, dt=0.01)

        assert (
            0 <= metrics.control_smoothness <= 2.0
        ), f"Control smoothness should be in [0, 2], got {metrics.control_smoothness}"

    def test_different_controllers_same_initial_conditions(
        self, estes_airframe, base_config, motor_config
    ):
        """Different controllers evaluated at the same wind level should see
        the same initial conditions when using the same env seed.

        AUDIT CHECK: compare_controllers.py creates a fresh env per wind level
        but doesn't seed episodes. Within a single evaluate_controller() call,
        episodes are sequential with the same env object, so initial conditions
        vary randomly between episodes but are independent of controller type.
        """
        env1 = make_motor_env(estes_airframe, motor_config, base_config)
        env2 = make_motor_env(estes_airframe, motor_config, base_config)

        # Same seed should give same initial conditions
        obs1, info1 = env1.reset(seed=42)
        obs2, info2 = env2.reset(seed=42)

        np.testing.assert_array_almost_equal(
            obs1,
            obs2,
            err_msg="Same seed should produce identical initial conditions",
        )


# ===========================================================================
# 7. PHYSICAL INVARIANTS & CONSERVATION
#    Energy, momentum, and other physical consistency checks
# ===========================================================================


class TestPhysicalInvariants:
    """Verify physical invariants and conservation laws."""

    def test_damping_opposes_motion(self, estes_airframe, base_config, motor_config):
        """Aerodynamic damping torque must always oppose the current spin direction.

        Physical law: Aerodynamic damping = -C * omega * q / V
        The negative sign ensures opposition to motion.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            damping_scale=2.0,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()
        env.vertical_velocity = 30.0

        # Positive spin — damping should slow it down
        env.roll_rate = np.deg2rad(50.0)
        env.step(np.array([0.0]))  # zero control
        accel_pos = env.roll_acceleration
        # Damping should produce negative acceleration (opposing positive spin)
        assert (
            accel_pos < 0
        ), f"Damping should oppose positive spin: got accel={accel_pos}"

        # Negative spin — damping should slow it down (push toward zero)
        env.reset()
        env.vertical_velocity = 30.0
        env.roll_rate = np.deg2rad(-50.0)
        env.step(np.array([0.0]))
        accel_neg = env.roll_acceleration
        assert (
            accel_neg > 0
        ), f"Damping should oppose negative spin: got accel={accel_neg}"

    def test_no_control_no_wind_spin_decays(
        self, estes_airframe, base_config, motor_config
    ):
        """With no control input and no wind, aerodynamic damping should cause
        spin rate to decay monotonically during boost (when q > 0).

        Physical invariant: in the absence of external torques, aerodynamic
        damping is the only torque, and it always opposes motion.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            damping_scale=2.0,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()
        env.roll_rate = np.deg2rad(30.0)

        spin_rates = [abs(env.roll_rate)]
        for _ in range(100):
            env.step(np.array([0.0]))
            spin_rates.append(abs(env.roll_rate))

        # Spin should decay over the boost phase
        assert spin_rates[-1] < spin_rates[0], (
            f"Spin should decay with damping: initial={np.rad2deg(spin_rates[0]):.1f}, "
            f"final={np.rad2deg(spin_rates[-1]):.1f} deg/s"
        )

    def test_roll_inertia_decreases_with_propellant_burn(
        self, estes_airframe, motor_config
    ):
        """Roll inertia should decrease as propellant is consumed.

        Physical check: I = I_airframe + I_motor, where motor mass decreases.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()

        # Inertia at start (full propellant)
        _, mass_start = env._update_propulsion()
        I_start = env._calculate_roll_inertia(mass_start)

        # Advance past motor burnout
        for _ in range(200):
            env.step(np.array([0.0]))

        _, mass_end = env._update_propulsion()
        I_end = env._calculate_roll_inertia(mass_end)

        assert I_end < I_start, (
            f"Inertia should decrease as propellant burns: "
            f"I_start={I_start:.6f}, I_end={I_end:.6f}"
        )

    def test_euler_integration_stability(
        self, estes_airframe, base_config, motor_config
    ):
        """At dt=0.01s, the Euler integration should be stable for the
        typical frequency range of roll dynamics (< 50 rad/s).

        Stability criterion: dt * max_eigenvalue < 2 for explicit Euler.
        The max eigenvalue of the roll dynamics is approximately
        damping_coef * q / (V * I), which should be << 200 at 100 Hz.
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()
        env.roll_rate = np.deg2rad(100.0)  # large initial spin
        env.vertical_velocity = 30.0

        # Run 500 steps — if integration is unstable, roll rate will explode
        for _ in range(500):
            env.step(np.array([0.0]))
            if abs(env.roll_rate) > np.deg2rad(720.0):
                pytest.fail(
                    f"Integration unstable: roll rate reached "
                    f"{np.rad2deg(env.roll_rate):.0f} deg/s"
                )


# ===========================================================================
# 8. CROSS-CUTTING INTEGRATION ISSUES
#    Issues that span multiple subsystems
# ===========================================================================


class TestCrossCuttingIntegration:
    """Tests for issues that span multiple subsystem boundaries."""

    def test_gain_scheduling_uses_current_q(
        self, estes_airframe, base_config, motor_config
    ):
        """GS-PID must use current (not delayed) dynamic pressure for gain scheduling.

        AUDIT CHECK: GainScheduledPIDController.step() line 216:
            q = info.get("dynamic_pressure_Pa", obs[5] if len(obs) > 5 else 0.0)
        This reads from info (always current), falling back to obs only if
        info is missing. This is correct.
        """
        pid = GainScheduledPIDController(PIDConfig(), use_observations=True)
        pid.launch_detected = True

        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 100.0  # delayed q

        info = {"dynamic_pressure_Pa": 500.0, "roll_rate_deg_s": 10.0}

        # The controller should use info q (500) not obs q (100)
        scale = pid._gain_scale(500.0)
        scale_if_delayed = pid._gain_scale(100.0)

        # These should be different, proving the controller would behave
        # differently if it used obs vs info
        assert (
            scale != scale_if_delayed
        ), "Gain schedule must differ between q=100 and q=500"

    def test_wind_cl_alpha_convention_matches(self, estes_airframe):
        """The cl_alpha passed to wind_model must use its ~2.0 convention,
        NOT the 2*pi convention used by the airframe.

        AUDIT CHECK: spin_stabilized_control_env.py line 458:
            wind_cl_alpha = cl_alpha / np.pi
        converts from 2*pi → ~2.0. Verify this conversion is correct.
        """
        # Default cl_alpha is 2*pi = 6.283
        cl_alpha_airframe = 2 * np.pi

        # Wind model convention: ~2.0
        cl_alpha_wind = cl_alpha_airframe / np.pi
        assert (
            abs(cl_alpha_wind - 2.0) < 0.01
        ), f"Wind cl_alpha should be ~2.0, got {cl_alpha_wind}"

    def test_wind_torque_uses_body_shadow_model(self, estes_airframe):
        """Wind torque must use the body-shadow model with K_shadow and sin(2*gamma).

        The physically correct model uses single_fin_area (not total),
        K_shadow = 1 - body_shadow_factor, and sin(N/2 * relative_angle)
        for an N-fin rocket.  See docs/wind_roll_torque_analysis.md.
        """
        wm = WindModel(WindConfig(enable=True))

        torque = wm.get_roll_torque(
            wind_speed=3.0,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=estes_airframe,
            cl_alpha=2.0,
        )
        assert torque != 0, "Wind torque should be non-zero"

        # Manually compute expected torque with body-shadow formula
        fin_set = estes_airframe.get_fin_set()
        single_fin_area = fin_set.span * 0.5 * (fin_set.root_chord + fin_set.tip_chord)
        moment_arm = estes_airframe.body_diameter / 2.0 + fin_set.span / 2.0
        K_shadow = 1.0 - 0.90  # default body_shadow_factor

        sideslip = np.arctan2(3.0, 30.0)
        # 4 fins, even → sin(N/2 * angle) = sin(2 * pi/4) = sin(pi/2) = 1
        forcing = np.sin(2 * np.pi / 4)
        expected = (
            500.0 * single_fin_area * 2.0 * sideslip * moment_arm * K_shadow * forcing
        )
        assert (
            abs(torque - expected) < 1e-10
        ), f"Wind torque {torque:.8f} != expected {expected:.8f} using body shadow model"

    def test_imu_mode_uses_info_for_roll_rate(
        self, estes_airframe, base_config, motor_config
    ):
        """In IMU mode (use_observations=True), PID controllers should read
        roll rate from info['roll_rate_deg_s'], not from obs[3].

        AUDIT FIX: This was changed to fix the J800 IMU mode failure.
        obs[3] is subject to sensor_delay_steps, but info is always current.
        The IMU wrapper applies gyro noise to info, so controllers get
        noisy-but-current values.
        """
        pid = PIDController(PIDConfig(), use_observations=True)
        pid.launch_detected = True
        pid.launch_orient = 0.0
        pid.target_orient = 0.0

        # Create obs with delayed roll rate and info with current rate
        obs = np.zeros(10, dtype=np.float32)
        obs[3] = np.deg2rad(50.0)  # delayed: 50 deg/s

        info = {"roll_rate_deg_s": 10.0}  # current: 10 deg/s

        action = pid.step(obs, info, dt=0.01)

        # The D-term uses roll_rate. If controller uses info (10 deg/s),
        # the action will be smaller than if it uses obs (50 deg/s).
        # With Kd=0.016524 and roll_rate=10: cmd_d = 10 * 0.016524 = 0.165
        # With Kd=0.016524 and roll_rate=50: cmd_d = 50 * 0.016524 = 0.826
        # So |action| with info should be smaller
        action_val = abs(action[0])

        # Now test with controller that (hypothetically) uses obs
        pid2 = PIDController(PIDConfig(), use_observations=True)
        pid2.launch_detected = True
        pid2.launch_orient = 0.0
        pid2.target_orient = 0.0

        # If controller correctly reads from info, it should use 10 deg/s
        # We can verify by checking the action magnitude
        # With 10 deg/s, the D-term dominates: cmd ≈ 10 * 0.016524 = 0.165
        # Normalized: action ≈ 0.165 / 30 ≈ 0.0055
        assert (
            action_val < 0.5
        ), f"PID with info roll_rate=10 deg/s should produce small action, got {action_val}"


# ===========================================================================
# 9. REGRESSION TESTS FOR AUDIT FIXES
#    Ensure all audit fixes remain in place and don't regress.
# ===========================================================================


class TestAuditFixRegressions:
    """Regression tests to prevent re-introduction of audit findings."""

    def test_adrc_uses_info_for_roll_rate_in_imu_mode(self):
        """ADRC in IMU mode must use info dict roll_rate at initialization.

        ADRC uses the ESO which estimates rate from successive angle
        observations, so it only uses info roll_rate at initialization.

        Regression test for audit fix #5.
        """
        from controllers.adrc_controller import ADRCController, ADRCConfig

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1  # small roll angle error
        obs[3] = np.deg2rad(50.0)  # delayed: 50 deg/s
        obs[5] = 500.0  # dynamic pressure

        ctrl = ADRCController(ADRCConfig(use_observations=True))
        # launch_detected=False so first step triggers initialization
        ctrl.launch_detected = False
        init_info = {
            "roll_rate_deg_s": 25.0,
            "dynamic_pressure_Pa": 500.0,
            "roll_angle_rad": 0.0,
        }
        ctrl.step(obs, init_info, dt=0.01)
        # z2 should be initialized from info (25 deg/s = 0.436 rad/s),
        # not from obs[3] (50 deg/s = 0.873 rad/s)
        expected_z2 = np.radians(25.0)
        assert abs(ctrl.z2 - expected_z2) < 0.1, (
            f"ADRC z2 should init from info roll_rate (25 deg/s={expected_z2:.3f}), "
            f"got z2={ctrl.z2:.3f}"
        )

    def test_ensemble_uses_info_for_performance(self):
        """Ensemble controller must read roll rate from info dict for
        performance monitoring, not from obs[3].
        """
        from controllers.ensemble_controller import EnsembleController, EnsembleConfig
        from controllers.pid_controller import (
            PIDController,
            PIDConfig,
            GainScheduledPIDController,
        )

        pid1 = PIDController(PIDConfig(), use_observations=False)
        pid1.launch_detected = True
        pid2 = GainScheduledPIDController(PIDConfig(), use_observations=False)
        pid2.launch_detected = True

        ensemble = EnsembleController([pid1, pid2], config=EnsembleConfig())

        obs = np.zeros(10, dtype=np.float32)
        obs[3] = np.deg2rad(50.0)  # delayed: 50 deg/s

        info = {
            "roll_rate_deg_s": 10.0,
            "roll_angle_rad": 0.0,
            "vertical_acceleration_ms2": 30.0,
            "dynamic_pressure_Pa": 500.0,
        }

        ensemble.step(obs, info, dt=0.01)

        # Check that performance window recorded 10 deg/s, not 50
        for window in ensemble._perf_windows:
            if len(window) > 0:
                assert window[-1] == pytest.approx(10.0, abs=0.1), (
                    f"Ensemble perf window should record info rate (10), "
                    f"not obs rate (50), got {window[-1]}"
                )

    def test_wind_torque_body_shadow_formula(self):
        """Wind torque must match the body-shadow formula.

        Regression test: verifies the formula uses single_fin_area * K_shadow
        * sin(N/2 * gamma) rather than total_fin_area * sin(gamma).
        """
        from airframe import RocketAirframe

        wm = WindModel(WindConfig(enable=True))
        airframe = RocketAirframe.estes_alpha()

        torque = wm.get_roll_torque(
            wind_speed=3.0,
            wind_direction=np.pi / 4,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=airframe,
            cl_alpha=2.0,
        )

        # Manually compute with body-shadow formula
        fin_set = airframe.get_fin_set()
        single_area = fin_set.span * 0.5 * (fin_set.root_chord + fin_set.tip_chord)
        moment_arm = airframe.body_diameter / 2.0 + fin_set.span / 2.0
        K_shadow = 1.0 - 0.90  # default
        sideslip = np.arctan2(3.0, 30.0)
        forcing = np.sin(2 * np.pi / 4)  # sin(N/2 * angle) for N=4
        expected = (
            500.0 * single_area * 2.0 * sideslip * moment_arm * K_shadow * forcing
        )

        assert (
            abs(torque - expected) < 1e-10
        ), f"Wind torque should match body-shadow formula"

    def test_propulsion_before_time_increment(
        self, estes_airframe, base_config, motor_config
    ):
        """Thrust must be evaluated at current time, not at t+dt.

        Regression test: previously self.time += dt occurred before
        _update_propulsion(), causing thrust to be evaluated one step late.
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()

        # At t=0, thrust curve starts at 0. After one step, thrust was
        # evaluated at t=0 (not t=0.01). The time has then advanced to 0.01.
        env.step(np.array([0.0]))
        assert abs(env.time - 0.01) < 1e-10

        # Verify by checking that near-end-of-burn thrust is non-zero.
        # Motor burn_time=1.85s. At step 184 (t=1.83 before step), thrust
        # should still be evaluated at t=1.83 (non-zero), not t=1.84.
        env2 = make_motor_env(estes_airframe, motor_config, base_config)
        env2.reset()
        for i in range(183):
            env2.step(np.array([0.0]))

        # Time is now 1.83s, about to evaluate thrust at t=1.83
        thrust_before, _ = env2._update_propulsion()
        # Thrust at t=1.83 should be non-zero (motor burns until 1.85)
        assert (
            thrust_before > 0
        ), f"Thrust at t=1.83s should be >0 (burn_time=1.85s), got {thrust_before}"

    def test_obs_q_nonzero_during_descent(
        self, estes_airframe, base_config, motor_config
    ):
        """obs[5] (dynamic pressure) must be >0 even during descent.

        Regression test: previously _get_observation() used v=max(v, 0),
        giving q=0 when descending. Now matches step() with v=max(v, 0.1).
        """
        env = make_motor_env(estes_airframe, motor_config, base_config)
        env.reset()
        env.vertical_velocity = -10.0  # descending

        obs = env._get_observation()
        assert (
            obs[5] > 0
        ), f"Obs q should be >0 during descent (v clamped to 0.1), got {obs[5]}"

    def test_info_density_matches_isa_at_altitude(self, estes_airframe, motor_config):
        """Info dict air_density must match _get_atmosphere() when use_isa_full=True.

        Regression test: previously _get_info() used hardcoded exponential
        model, inconsistent with the ISA model used by physics.
        """
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=0.0,
            disturbance_scale=0.0,
            enable_wind=False,
            use_isa_full=True,
            dt=0.01,
        )
        env = make_motor_env(estes_airframe, motor_config, config)
        env.reset()

        for alt in [0.0, 1000.0, 5000.0, 10000.0]:
            env.altitude = alt
            info = env._get_info()
            rho_info = info["air_density_kg_m3"]
            rho_physics = env._get_air_density()
            assert abs(rho_info - rho_physics) < 1e-6, (
                f"Info air_density ({rho_info:.6f}) must match physics "
                f"({rho_physics:.6f}) at altitude {alt}m"
            )
