"""
Tests for J800 75mm rocket physics model.

Tests atmosphere model, Mach-dependent aerodynamics, servo dynamics,
sensor latency, J800 config loading, flight profile, and backward
compatibility with the existing C6 configuration.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path

from airframe import RocketAirframe
from airframe.components import Material, TrapezoidFinSet
from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig
from rocket_config import RocketTrainingConfig, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_env(config: RocketConfig, airframe: RocketAirframe = None):
    """Create a SpinStabilizedCameraRocket with given config."""
    if airframe is None:
        airframe = RocketAirframe.estes_alpha()
    return SpinStabilizedCameraRocket(airframe=airframe, config=config)


def _j800_airframe():
    """Load the J800 75mm airframe from YAML."""
    return RocketAirframe.load("configs/airframes/j800_75mm.yaml")


def _default_j800_config():
    """Create a RocketConfig with J800 physics settings (no wind)."""
    return RocketConfig(
        max_tab_deflection=30.0,
        tab_chord_fraction=0.25,
        tab_span_fraction=0.5,
        num_controlled_fins=3,
        disturbance_scale=0.0001,
        damping_scale=1.5,
        initial_spin_std=9.0,
        max_roll_rate=720.0,
        max_episode_time=30.0,
        use_mach_aero=True,
        use_isa_full=True,
        cd_mach_table={
            "mach": [0.0, 0.6, 0.8, 0.9, 1.0, 1.05, 1.2, 1.5, 2.0],
            "cd_boost": [0.40, 0.40, 0.42, 0.55, 0.75, 0.82, 0.70, 0.55, 0.48],
            "cd_coast": [0.50, 0.50, 0.52, 0.65, 0.85, 0.92, 0.80, 0.65, 0.58],
        },
        servo_time_constant=0.020,
        servo_rate_limit=500.0,
        servo_deadband=0.5,
        sensor_delay_steps=2,
        max_velocity=400.0,
        max_dynamic_pressure=80000.0,
        dt=0.01,
        # Motor params for simple thrust model (RealisticMotorRocket overrides these)
        average_thrust=696.5,
        burn_time=1.8,
        propellant_mass=0.450,
    )


# =========================================================================
# TestAtmosphere
# =========================================================================


class TestAtmosphere:
    """Test the full ISA atmosphere model."""

    def test_sea_level(self):
        """Sea level: rho=1.225, T=288.15K, a=340.3 m/s."""
        config = RocketConfig(use_isa_full=True)
        env = _make_env(config)
        env.altitude = 0.0
        rho, T, a = env._get_atmosphere()
        assert abs(rho - 1.225) < 0.001
        assert abs(T - 288.15) < 0.01
        assert abs(a - 340.3) < 0.5

    def test_at_3000m(self):
        """At 3000m: verify rho, T, a against ISA tables."""
        config = RocketConfig(use_isa_full=True)
        env = _make_env(config)
        env.altitude = 3000.0
        rho, T, a = env._get_atmosphere()
        # ISA at 3000m: T=268.65K, rho~0.909, a~328.6
        assert abs(T - 268.65) < 0.1
        assert abs(rho - 0.909) < 0.01
        assert abs(a - 328.6) < 1.0

    def test_exponential_approx_matches_full_isa_below_1000m(self):
        """Exponential approx matches full ISA within 5% below 1000m."""
        config_full = RocketConfig(use_isa_full=True)
        config_approx = RocketConfig(use_isa_full=False)
        env_full = _make_env(config_full)
        env_approx = _make_env(config_approx)

        for alt in [0, 100, 500, 1000]:
            env_full.altitude = alt
            env_approx.altitude = alt
            rho_full = env_full._get_atmosphere()[0]
            rho_approx = env_approx._get_atmosphere()[0]
            rel_error = abs(rho_full - rho_approx) / rho_full
            assert rel_error < 0.05, f"Mismatch at {alt}m: {rel_error:.3f}"

    def test_use_isa_full_false_returns_exponential(self):
        """use_isa_full=False uses old exponential model (backward compat)."""
        config = RocketConfig(use_isa_full=False)
        env = _make_env(config)
        env.altitude = 500.0
        rho, T, a = env._get_atmosphere()
        # Should return fixed T=288.15, a=340.3 regardless of altitude
        assert T == 288.15
        assert a == 340.3
        expected_rho = 1.225 * np.exp(-500.0 / 8000)
        assert abs(rho - expected_rho) < 1e-6

    def test_negative_altitude_clamped(self):
        """Negative altitude should be clamped to 0."""
        config = RocketConfig(use_isa_full=True)
        env = _make_env(config)
        env.altitude = -100.0
        rho, T, a = env._get_atmosphere()
        # Should return sea level values
        assert abs(rho - 1.225) < 0.001
        assert abs(T - 288.15) < 0.01

    def test_high_altitude(self):
        """At high altitude, density should be much lower."""
        config = RocketConfig(use_isa_full=True)
        env = _make_env(config)
        env.altitude = 10000.0
        rho, T, a = env._get_atmosphere()
        assert rho < 0.5  # ISA at 10km: ~0.414
        assert T < 230  # ISA at 10km: ~223.25K


# =========================================================================
# TestMachDependentCd
# =========================================================================


class TestMachDependentCd:
    """Test Mach-dependent drag coefficient."""

    def _make_env_with_table(self):
        config = _default_j800_config()
        return _make_env(config, _j800_airframe())

    def test_subsonic_cd(self):
        """Subsonic M<0.8: returns low Cd value."""
        env = self._make_env_with_table()
        cd = env._get_cd(0.5, is_boost=True)
        assert abs(cd - 0.40) < 0.01

    def test_transonic_rise(self):
        """Transonic: Cd(1.0) > Cd(0.5)."""
        env = self._make_env_with_table()
        cd_sub = env._get_cd(0.5, is_boost=True)
        cd_trans = env._get_cd(1.0, is_boost=True)
        assert cd_trans > cd_sub

    def test_peak_near_m105(self):
        """Peak Cd near M=1.0-1.05."""
        env = self._make_env_with_table()
        cd_peak = env._get_cd(1.05, is_boost=True)
        cd_below = env._get_cd(0.8, is_boost=True)
        cd_above = env._get_cd(1.5, is_boost=True)
        assert cd_peak > cd_below
        assert cd_peak > cd_above

    def test_supersonic_decay(self):
        """Supersonic: Cd(1.5) < Cd(1.05)."""
        env = self._make_env_with_table()
        cd_15 = env._get_cd(1.5, is_boost=True)
        cd_105 = env._get_cd(1.05, is_boost=True)
        assert cd_15 < cd_105

    def test_boost_vs_coast(self):
        """Boost Cd < Coast Cd at same Mach."""
        env = self._make_env_with_table()
        cd_boost = env._get_cd(0.8, is_boost=True)
        cd_coast = env._get_cd(0.8, is_boost=False)
        assert cd_boost < cd_coast

    def test_interpolation(self):
        """Interpolation between table points."""
        env = self._make_env_with_table()
        # M=0.85 should be between M=0.8 (0.42) and M=0.9 (0.55)
        cd = env._get_cd(0.85, is_boost=True)
        assert 0.42 < cd < 0.55

    def test_mach_aero_disabled(self):
        """use_mach_aero=False returns constant Cd (backward compat)."""
        config = RocketConfig(use_mach_aero=False)
        env = _make_env(config)
        assert env._get_cd(0.5, is_boost=True) == 0.4
        assert env._get_cd(1.0, is_boost=True) == 0.4
        assert env._get_cd(0.5, is_boost=False) == 0.5

    def test_extrapolation_beyond_table(self):
        """Mach beyond table range should extrapolate from last segment."""
        env = self._make_env_with_table()
        cd = env._get_cd(3.0, is_boost=True)
        # np.interp clamps to last value
        assert abs(cd - 0.48) < 0.01


# =========================================================================
# TestMachDependentClAlpha
# =========================================================================


class TestMachDependentClAlpha:
    """Test Mach-dependent lift curve slope."""

    def _make_env_mach(self):
        config = RocketConfig(use_mach_aero=True)
        return _make_env(config)

    def test_incompressible(self):
        """Cl_alpha(0) = 2*pi."""
        env = self._make_env_mach()
        cl = env._get_cl_alpha(0.0)
        assert abs(cl - 2.0 * np.pi) < 0.01

    def test_prandtl_glauert_m06(self):
        """Prandtl-Glauert at M=0.6: Cl_alpha = 2*pi / sqrt(1-0.36)."""
        env = self._make_env_mach()
        cl = env._get_cl_alpha(0.6)
        expected = 2.0 * np.pi / np.sqrt(1.0 - 0.36)
        assert abs(cl - expected) < 0.01

    def test_transonic_interpolation(self):
        """Smooth interpolation between M=0.8 and M=1.2."""
        env = self._make_env_mach()
        cl_08 = env._get_cl_alpha(0.8)
        cl_10 = env._get_cl_alpha(1.0)
        cl_12 = env._get_cl_alpha(1.2)
        # At M=1.0 (midpoint), should be between M=0.8 and M=1.2 values
        assert min(cl_08, cl_12) <= cl_10 <= max(cl_08, cl_12)

    def test_ackeret_m15(self):
        """Ackeret at M=1.5: Cl_alpha = 4/sqrt(M^2-1)."""
        env = self._make_env_mach()
        cl = env._get_cl_alpha(1.5)
        expected = 4.0 / np.sqrt(1.5**2 - 1.0)
        assert abs(cl - expected) < 0.01

    def test_continuity_at_m08(self):
        """Continuity at M=0.8 boundary."""
        env = self._make_env_mach()
        cl_below = env._get_cl_alpha(0.799)
        cl_at = env._get_cl_alpha(0.8)
        assert abs(cl_below - cl_at) < 0.5  # small discontinuity acceptable

    def test_continuity_at_m12(self):
        """Continuity at M=1.2 boundary."""
        env = self._make_env_mach()
        cl_at = env._get_cl_alpha(1.2)
        cl_above = env._get_cl_alpha(1.201)
        assert abs(cl_at - cl_above) < 0.5

    def test_mach_aero_disabled(self):
        """use_mach_aero=False returns 2*pi (backward compat)."""
        config = RocketConfig(use_mach_aero=False)
        env = _make_env(config)
        assert env._get_cl_alpha(0.0) == pytest.approx(2.0 * np.pi)
        assert env._get_cl_alpha(0.8) == pytest.approx(2.0 * np.pi)
        assert env._get_cl_alpha(1.5) == pytest.approx(2.0 * np.pi)


# =========================================================================
# TestServoDynamics
# =========================================================================


class TestServoDynamics:
    """Test servo dynamics (lag, rate limit, deadband)."""

    def test_instantaneous_when_disabled(self):
        """All servo params=0: instantaneous (backward compat)."""
        config = RocketConfig(servo_time_constant=0.0, servo_rate_limit=0.0)
        env = _make_env(config)
        env.reset()
        result = env._update_servo(0.8, dt=0.01)
        assert result == 0.8

    def test_first_order_lag(self):
        """First-order lag: step response follows 1-exp(-t/tau)."""
        config = RocketConfig(
            servo_time_constant=0.05, servo_rate_limit=0.0, servo_deadband=0.0
        )
        env = _make_env(config)
        env.reset()
        # Apply step to 1.0 and run for several timesteps
        positions = []
        for _ in range(20):
            pos = env._update_servo(1.0, dt=0.01)
            positions.append(pos)
        # After 20 steps (0.2s = 4*tau), should be near 1-exp(-4) ~ 0.982
        assert positions[-1] > 0.9
        # Should be monotonically increasing
        for i in range(1, len(positions)):
            assert positions[i] >= positions[i - 1] - 1e-10

    def test_rate_limiting(self):
        """Large step capped by rate limit."""
        config = RocketConfig(
            servo_time_constant=0.0,
            servo_rate_limit=100.0,  # 100 deg/s with max_tab_deflection=15 -> 6.67 norm/s
            servo_deadband=0.0,
            max_tab_deflection=30.0,
        )
        env = _make_env(config)
        env.reset()
        # Command full deflection, rate limit should constrain movement
        pos = env._update_servo(1.0, dt=0.01)
        max_delta = (100.0 / 15.0) * 0.01  # ~0.0667
        assert abs(pos) <= max_delta + 1e-10

    def test_deadband(self):
        """Small commands below threshold ignored."""
        config = RocketConfig(
            servo_time_constant=0.0,
            servo_rate_limit=0.0,
            servo_deadband=1.0,  # 1 degree deadband
            max_tab_deflection=30.0,
        )
        env = _make_env(config)
        env.reset()
        # Command smaller than deadband normalized (1.0/30.0 = 0.0333)
        pos = env._update_servo(0.02, dt=0.01)
        assert pos == 0.0  # Should not move

    def test_combined_lag_and_rate_limit(self):
        """Combined: both lag and rate limit active."""
        config = RocketConfig(
            servo_time_constant=0.05,
            servo_rate_limit=200.0,
            servo_deadband=0.0,
            max_tab_deflection=30.0,
        )
        env = _make_env(config)
        env.reset()
        pos = env._update_servo(1.0, dt=0.01)
        # Should be limited by both mechanisms
        assert 0.0 < pos < 1.0

    def test_servo_bounded(self):
        """Servo position always bounded [-1, 1]."""
        config = RocketConfig(
            servo_time_constant=0.0, servo_rate_limit=0.0, servo_deadband=0.0
        )
        env = _make_env(config)
        env.reset()
        pos = env._update_servo(2.0, dt=0.01)
        assert -1.0 <= pos <= 1.0

    def test_reset_clears_servo(self):
        """reset() clears servo state."""
        config = RocketConfig(
            servo_time_constant=0.05, servo_rate_limit=0.0, servo_deadband=0.0
        )
        env = _make_env(config)
        env.reset()
        env._update_servo(1.0, dt=0.01)
        assert env._servo_position != 0.0
        env.reset()
        assert env._servo_position == 0.0

    def test_servo_in_step(self):
        """Servo dynamics are applied during env.step()."""
        config = RocketConfig(
            servo_time_constant=0.05,
            servo_rate_limit=0.0,
            servo_deadband=0.0,
        )
        env = _make_env(config)
        env.reset()
        # Full command should not produce full deflection due to lag
        env.step(np.array([1.0]))
        actual_defl_deg = np.rad2deg(abs(env.tab_deflection))
        max_defl = config.max_tab_deflection
        assert actual_defl_deg < max_defl * 0.5  # should be much less than full


# =========================================================================
# TestSensorLatency
# =========================================================================


class TestSensorLatency:
    """Test sensor latency (observation delay buffer)."""

    def test_zero_delay_passthrough(self):
        """Zero delay: passthrough (backward compat)."""
        config = RocketConfig(sensor_delay_steps=0)
        env = _make_env(config)
        obs1, _ = env.reset()
        obs2, _, _, _, _ = env.step(np.array([0.0]))
        # With zero delay, obs should reflect current state
        assert obs2[6] > 0  # time should be > 0

    def test_one_step_delay(self):
        """1-step delay: obs[n] = true_obs[n-1]."""
        config = RocketConfig(sensor_delay_steps=1)
        env = _make_env(config)
        obs0, _ = env.reset()
        obs1, _, _, _, _ = env.step(np.array([0.5]))
        obs2, _, _, _, _ = env.step(np.array([0.5]))
        # obs2 should be delayed by 1 step, so its time should be < current time
        # The time field is obs[6]
        # After 2 steps, current time = 0.02, delayed obs should show time = 0.01
        assert abs(obs2[6] - 0.01) < 1e-6

    def test_two_step_delay(self):
        """2-step delay: obs[n] = true_obs[n-2]."""
        config = RocketConfig(sensor_delay_steps=2)
        env = _make_env(config)
        obs0, _ = env.reset()
        obs1, _, _, _, _ = env.step(np.array([0.0]))
        obs2, _, _, _, _ = env.step(np.array([0.0]))
        obs3, _, _, _, _ = env.step(np.array([0.0]))
        # After 3 steps, current time = 0.03, delayed obs should show time = 0.01
        assert abs(obs3[6] - 0.01) < 1e-6

    def test_buffer_warmup(self):
        """First steps return best available obs (not crash)."""
        config = RocketConfig(sensor_delay_steps=5)
        env = _make_env(config)
        obs, _ = env.reset()
        # First step should work even though buffer doesn't have 5 entries yet
        obs1, _, _, _, _ = env.step(np.array([0.0]))
        assert obs1 is not None
        assert len(obs1) == 10

    def test_reset_clears_buffer(self):
        """reset() clears delay buffer."""
        config = RocketConfig(sensor_delay_steps=2)
        env = _make_env(config)
        env.reset()
        env.step(np.array([0.0]))
        env.step(np.array([0.0]))
        env.reset()
        assert len(env._obs_buffer) == 0


# =========================================================================
# TestJ800Config
# =========================================================================


class TestJ800Config:
    """Test J800 configuration loading and validation."""

    def test_config_loads(self):
        """Config loads from YAML without error."""
        config = load_config("configs/aerotech_j800_wind.yaml")
        assert config is not None
        assert config.motor.name == "aerotech_j800t"

    def test_airframe_resolves(self):
        """Airframe resolves and has correct num_fins=3."""
        config = load_config("configs/aerotech_j800_wind.yaml")
        airframe = config.physics.resolve_airframe()
        fin_set = airframe.get_fin_set()
        assert fin_set is not None
        assert fin_set.num_fins == 3

    def test_motor_loads(self):
        """Motor loads with thrust curve."""
        config = load_config("configs/aerotech_j800_wind.yaml")
        assert config.motor.total_impulse_Ns == 1229.0
        assert config.motor.thrust_curve is not None
        assert len(config.motor.thrust_curve["time_s"]) > 2

    def test_env_creates_and_steps(self):
        """Environment creates and steps without error."""
        airframe = _j800_airframe()
        config = _default_j800_config()
        env = _make_env(config, airframe)
        obs, info = env.reset()
        assert obs is not None
        obs2, reward, term, trunc, info2 = env.step(np.array([0.0]))
        assert obs2 is not None

    def test_carbon_fibre_resolves(self):
        """Carbon fibre material resolves correctly."""
        mat = Material.from_name("Carbon Fibre")
        assert mat.name == "Carbon Fibre"
        assert mat.density == 1600.0
        mat2 = Material.from_name("Carbon Fiber")
        assert mat2.density == 1600.0

    def test_mach_aero_fields_passthrough(self):
        """New physics fields pass through config loading."""
        config = load_config("configs/aerotech_j800_wind.yaml")
        assert config.physics.use_mach_aero is True
        assert config.physics.use_isa_full is True
        assert config.physics.cd_mach_table is not None
        assert config.physics.servo_time_constant == 0.020
        assert config.physics.servo_rate_limit == 500.0
        assert config.physics.servo_deadband == 0.5
        assert config.physics.sensor_delay_steps == 2
        assert config.physics.max_velocity == 400.0
        assert config.physics.max_dynamic_pressure == 80000.0


# =========================================================================
# TestJ800FlightProfile
# =========================================================================


class TestJ800FlightProfile:
    """Test J800 passive flight profile characteristics."""

    @pytest.fixture
    def flight_data(self):
        """Run a passive (zero control) J800 flight and return telemetry."""
        airframe = _j800_airframe()
        config = _default_j800_config()
        config.enable_wind = False
        config.initial_spin_std = 0.0  # no initial spin for predictable flight
        env = _make_env(config, airframe)
        np.random.seed(42)
        obs, info = env.reset()

        data = {
            "altitudes": [0.0],
            "velocities": [0.0],
            "mach_numbers": [0.0],
            "cd_values": [0.4],
            "times": [0.0],
        }

        for _ in range(2500):
            obs, reward, term, trunc, info = env.step(np.array([0.0]))
            data["altitudes"].append(info["altitude_m"])
            data["velocities"].append(info["vertical_velocity_ms"])
            data["mach_numbers"].append(info.get("mach_number", 0.0))
            data["cd_values"].append(info.get("cd", 0.4))
            data["times"].append(info["time_s"])
            if term or trunc:
                break

        return data

    def test_reaches_high_mach(self, flight_data):
        """Passive flight reaches M > 0.8."""
        max_mach = max(flight_data["mach_numbers"])
        assert max_mach > 0.8, f"Max Mach only {max_mach:.2f}"

    def test_high_apogee(self, flight_data):
        """Apogee > 1500m."""
        max_alt = max(flight_data["altitudes"])
        assert max_alt > 1500, f"Apogee only {max_alt:.0f}m"

    def test_flight_duration(self, flight_data):
        """Flight duration 10-30s."""
        duration = flight_data["times"][-1]
        assert 10 < duration <= 30, f"Duration {duration:.1f}s"

    def test_twr_at_ignition(self):
        """TWR > 2 at ignition."""
        airframe = _j800_airframe()
        config = _default_j800_config()
        total_mass = airframe.dry_mass + config.propellant_mass
        twr = config.average_thrust / (total_mass * 9.81)
        assert twr > 2.0, f"TWR only {twr:.1f}"

    def test_cd_varies_during_flight(self, flight_data):
        """Cd varies during flight (not constant) when Mach aero enabled."""
        cd_values = flight_data["cd_values"]
        unique_cds = set(round(cd, 4) for cd in cd_values)
        assert len(unique_cds) > 2, "Cd should vary with Mach"

    def test_three_fins_produce_roll_torque(self):
        """3 active fins produce roll torque."""
        airframe = _j800_airframe()
        eff = airframe.get_control_effectiveness(
            dynamic_pressure=1000.0,
            tab_chord_fraction=0.25,
            tab_span_fraction=0.5,
            num_controlled_fins=3,
        )
        assert eff > 0


# =========================================================================
# TestBackwardCompat
# =========================================================================


class TestBackwardCompat:
    """Test backward compatibility with existing C6 configuration."""

    def test_c6_uses_constant_cd(self):
        """C6 config uses constant Cd (Mach aero disabled)."""
        config = RocketConfig()  # defaults: use_mach_aero=False
        env = _make_env(config)
        cd_sub = env._get_cd(0.3, is_boost=True)
        cd_trans = env._get_cd(1.0, is_boost=True)
        assert cd_sub == cd_trans == 0.4

    def test_c6_instantaneous_servo(self):
        """C6 has instantaneous servo (servo params=0)."""
        config = RocketConfig()  # defaults: servo_time_constant=0
        env = _make_env(config)
        env.reset()
        pos = env._update_servo(0.7, dt=0.01)
        assert pos == 0.7

    def test_c6_zero_sensor_delay(self):
        """C6 has zero sensor delay."""
        config = RocketConfig()  # default: sensor_delay_steps=0
        assert config.sensor_delay_steps == 0

    def test_c6_observation_space_unchanged(self):
        """C6 observation space bounds unchanged (default config)."""
        config = RocketConfig()
        env = _make_env(config)
        # Check obs space shape
        assert env.observation_space.shape == (10,)
        # Check default velocity bound is 100
        assert env.observation_space.high[1] == 100.0
        # Check default q bound is 3000
        assert env.observation_space.high[5] == 3000.0

    def test_c6_produces_consistent_spin_rate(self):
        """C6 config produces spin rate in expected range (numerical regression)."""
        airframe = RocketAirframe.estes_alpha()
        config = RocketConfig(
            max_tab_deflection=30.0,
            initial_spin_std=9.0,
            disturbance_scale=0.0001,
            damping_scale=1.5,
            average_thrust=5.4,
            burn_time=1.85,
            propellant_mass=0.012,
        )
        env = _make_env(config, airframe)

        # Run 5 episodes with P-controller
        spin_rates = []
        for seed in range(5):
            np.random.seed(seed + 100)
            obs, _ = env.reset()
            ep_spins = []
            for _ in range(200):
                action = np.array([-0.5 * obs[3]])  # simple P control
                obs, _, term, trunc, info = env.step(np.clip(action, -1, 1))
                ep_spins.append(abs(info["roll_rate_deg_s"]))
                if term or trunc:
                    break
            spin_rates.append(np.mean(ep_spins))

        mean_spin = np.mean(spin_rates)
        # Should be in a reasonable range for controlled C6 flight
        assert mean_spin < 50, f"Mean spin {mean_spin:.1f} too high"
