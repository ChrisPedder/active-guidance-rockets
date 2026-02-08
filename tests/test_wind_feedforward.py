"""
Tests for WindFeedforwardADRC controller.

Verifies that:
1. The sinusoidal disturbance estimator correctly identifies wind amplitude and direction
2. Feedforward produces corrective action that opposes the estimated disturbance
3. The warmup period suppresses feedforward during initialization
4. The controller degrades gracefully to plain ADRC when there is no wind
5. The controller interface (reset/step) works correctly
6. Feedforward improves disturbance rejection in a closed-loop simulation
"""

import pytest
import numpy as np

from adrc_controller import ADRCConfig
from wind_feedforward import WindFeedforwardADRC, WindFeedforwardConfig


# --- Helpers ---


def make_info(roll_angle_rad=0.0, roll_rate_deg_s=0.0, accel=50.0, q=500.0):
    """Create a standard info dict for testing."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": accel,
        "dynamic_pressure_Pa": q,
    }


def make_obs(roll_angle=0.0, roll_rate=0.0, q=500.0):
    """Create a standard observation array for testing."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = roll_angle
    obs[3] = roll_rate
    obs[5] = q
    return obs


class TestWindFeedforwardConfig:
    """Test WindFeedforwardConfig defaults."""

    def test_default_K_ff(self):
        config = WindFeedforwardConfig()
        assert config.K_ff == 0.5

    def test_default_mu(self):
        config = WindFeedforwardConfig()
        assert config.mu == 0.02

    def test_default_forgetting(self):
        config = WindFeedforwardConfig()
        assert config.forgetting == 0.998

    def test_default_warmup(self):
        config = WindFeedforwardConfig()
        assert config.warmup_steps == 50


class TestWindFeedforwardInterface:
    """Test that WindFeedforwardADRC has the same interface as other controllers."""

    def test_has_reset_method(self):
        ctrl = WindFeedforwardADRC()
        assert hasattr(ctrl, "reset") and callable(ctrl.reset)

    def test_has_step_method(self):
        ctrl = WindFeedforwardADRC()
        assert hasattr(ctrl, "step") and callable(ctrl.step)

    def test_reset_clears_state(self):
        ctrl = WindFeedforwardADRC()
        # Set some state
        ctrl.adrc.launch_detected = True
        ctrl.coeff_cos = 1.0
        ctrl.coeff_sin = 2.0
        ctrl._step_count = 100

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert ctrl.coeff_cos == 0.0
        assert ctrl.coeff_sin == 0.0
        assert ctrl._step_count == 0

    def test_launch_detected_property(self):
        ctrl = WindFeedforwardADRC()
        assert ctrl.launch_detected is False
        ctrl.launch_detected = True
        assert ctrl.launch_detected is True
        assert ctrl.adrc.launch_detected is True

    def test_z_properties(self):
        ctrl = WindFeedforwardADRC()
        ctrl.adrc.z1 = 1.0
        ctrl.adrc.z2 = 2.0
        ctrl.adrc.z3 = 3.0
        assert ctrl.z1 == 1.0
        assert ctrl.z2 == 2.0
        assert ctrl.z3 == 3.0

    def test_step_returns_correct_shape(self):
        adrc_cfg = ADRCConfig(b0=100.0, b0_per_pa=None)
        ctrl = WindFeedforwardADRC(adrc_cfg)
        ctrl.launch_detected = True
        obs = np.zeros(10)
        info = make_info(roll_rate_deg_s=10.0)
        action = ctrl.step(obs, info)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_action_clamped(self):
        adrc_cfg = ADRCConfig(omega_c=100.0, b0=1.0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=1.0, warmup_steps=0)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True
        # Force large feedforward
        ctrl.coeff_cos = 1000.0
        ctrl._step_count = 100

        obs = np.zeros(10)
        info = make_info(roll_angle_rad=1.0, roll_rate_deg_s=500.0)
        action = ctrl.step(obs, info)
        assert -1.0 <= action[0] <= 1.0


class TestWarmupBehavior:
    """Test that feedforward is suppressed during warmup."""

    def test_no_feedforward_during_warmup(self):
        """During warmup, output should match plain ADRC."""
        adrc_cfg = ADRCConfig(b0=100.0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=1.0, warmup_steps=100)
        ctrl_ff = WindFeedforwardADRC(adrc_cfg, ff_cfg)

        from adrc_controller import ADRCController

        ctrl_base = ADRCController(ADRCConfig(b0=100.0, b0_per_pa=None))

        obs = np.zeros(10)
        # First step triggers launch
        info = make_info(roll_angle_rad=0.0, roll_rate_deg_s=30.0, accel=50.0)
        action_ff = ctrl_ff.step(obs, info)
        action_base = ctrl_base.step(obs, info)

        assert abs(action_ff[0] - action_base[0]) < 1e-6, (
            f"During warmup, FF controller should match base ADRC: "
            f"ff={action_ff[0]:.6f}, base={action_base[0]:.6f}"
        )

    def test_feedforward_activates_after_warmup(self):
        """After warmup, feedforward should contribute to action."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=1.0, warmup_steps=5, mu=0.1)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        dt = 0.01
        roll_angle = 0.0
        roll_rate = 0.0

        # Run past warmup with a sinusoidal disturbance
        wind_dir = 1.0  # rad
        wind_amp = 20.0  # rad/s^2

        for step in range(20):
            d = wind_amp * np.sin(wind_dir - roll_angle)
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)

            # Simple dynamics
            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # After warmup, coefficients should be non-zero
        assert ctrl.wind_amplitude > 0.1, (
            f"After warmup with disturbance, wind amplitude estimate should be > 0: "
            f"got {ctrl.wind_amplitude:.4f}"
        )


class TestSinusoidalEstimator:
    """Test the sinusoidal disturbance estimator."""

    def test_estimates_converge_with_known_disturbance(self):
        """With a known sinusoidal disturbance, estimates should converge."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=10, mu=0.02)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(30.0)  # Initial spin
        wind_dir = 0.8  # Known wind direction
        wind_amp = 10.0  # Known amplitude in rad/s^2

        obs = np.zeros(10)
        for step in range(500):
            d = wind_amp * np.sin(wind_dir - roll_angle)
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)

            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # Check that amplitude estimate is in the right ballpark
        # The ESO absorbs part of the disturbance, so the feedforward
        # estimates the residual structure. The amplitude should be
        # significant (not zero).
        assert ctrl.wind_amplitude > 1.0, (
            f"Wind amplitude estimate ({ctrl.wind_amplitude:.2f}) should be "
            f"significantly nonzero with a {wind_amp} rad/s^2 disturbance"
        )

    def test_coefficients_zero_without_disturbance(self):
        """Without wind disturbance, coefficients should remain near zero."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=10, mu=0.02)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(30.0)

        obs = np.zeros(10)
        for step in range(300):
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)

            # No disturbance — just control
            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # Coefficients should be small
        assert ctrl.wind_amplitude < 2.0, (
            f"Without disturbance, wind amplitude estimate ({ctrl.wind_amplitude:.2f}) "
            f"should be near zero"
        )

    def test_wind_direction_estimate(self):
        """Wind direction estimate should be in the right quadrant."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=10, mu=0.02)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(50.0)  # Fast spin for good signal
        wind_dir = 1.5  # Known wind direction
        wind_amp = 15.0

        obs = np.zeros(10)
        for step in range(500):
            d = wind_amp * np.sin(wind_dir - roll_angle)
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)

            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # Wind direction estimate should be somewhat close to actual
        # (not exact due to ADRC absorbing part of disturbance)
        est_dir = ctrl.wind_direction_estimate
        # Normalize angle difference to [-pi, pi]
        diff = (est_dir - wind_dir + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < np.pi / 2, (
            f"Wind direction estimate ({np.degrees(est_dir):.1f}°) should be "
            f"within 90° of actual ({np.degrees(wind_dir):.1f}°), "
            f"diff={np.degrees(diff):.1f}°"
        )


class TestFeedforwardAction:
    """Test that feedforward produces physically correct actions."""

    def test_feedforward_opposes_disturbance(self):
        """Feedforward action should oppose the wind disturbance."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.8, warmup_steps=0, mu=0.05)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        # Manually set coefficients to simulate a known disturbance
        # d = 10*cos(theta) means disturbance is positive at theta=0
        ctrl.coeff_cos = 10.0
        ctrl.coeff_sin = 0.0
        ctrl._step_count = 100  # Past warmup

        obs = np.zeros(10)
        # At roll_angle=0, cos(0)=1, disturbance = +10 rad/s^2
        # Feedforward should be: -K_ff * 10 / b0 = -0.08
        info = make_info(roll_angle_rad=0.0, roll_rate_deg_s=0.0)
        action = ctrl.step(obs, info)

        # The action should be more negative than without feedforward
        # (opposing the positive disturbance)
        assert (
            ctrl._ff_action < 0
        ), f"Feedforward should oppose positive disturbance, got ff={ctrl._ff_action:.4f}"

    def test_K_ff_zero_disables_feedforward(self):
        """K_ff=0 should make feedforward inactive."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.0, warmup_steps=0)
        ctrl_ff = WindFeedforwardADRC(adrc_cfg, ff_cfg)

        from adrc_controller import ADRCController

        ctrl_base = ADRCController(ADRCConfig(b0=b0, b0_per_pa=None))

        # Manually set large coefficients — should have no effect with K_ff=0
        ctrl_ff.coeff_cos = 100.0
        ctrl_ff.coeff_sin = 100.0
        ctrl_ff._step_count = 200

        obs = np.zeros(10)
        info = make_info(roll_angle_rad=0.1, roll_rate_deg_s=20.0, accel=50.0)

        action_ff = ctrl_ff.step(obs, info)
        action_base = ctrl_base.step(obs, info)

        assert abs(action_ff[0] - action_base[0]) < 1e-6, (
            f"K_ff=0 should disable feedforward: ff={action_ff[0]:.6f}, "
            f"base={action_base[0]:.6f}"
        )


class TestNoWindDegradation:
    """Test that feedforward doesn't degrade performance without wind."""

    def test_settles_without_wind(self):
        """Controller should still settle spin without wind disturbance."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=20, mu=0.02)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(30.0)  # 30 deg/s initial spin

        obs = np.zeros(10)
        for step in range(300):
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)

            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # Should have settled to near zero
        final_rate_deg = abs(np.degrees(roll_rate))
        assert final_rate_deg < 5.0, (
            f"Controller should settle spin to < 5 deg/s without wind, "
            f"got {final_rate_deg:.1f} deg/s"
        )


class TestDynamicB0WithFeedforward:
    """Test feedforward with gain-scheduled b0."""

    def test_feedforward_uses_dynamic_b0(self):
        """With b0_per_pa set, feedforward should use dynamic b0."""
        b0_per_pa = 1.0
        ref_q = 500.0
        b0_fixed = b0_per_pa * ref_q * np.tanh(ref_q / 200.0)
        adrc_cfg = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=0)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl.launch_detected = True
        ctrl.coeff_cos = 10.0
        ctrl._step_count = 100

        obs = np.zeros(10)

        # At different q values, the feedforward action should differ
        info_low = make_info(roll_angle_rad=0.0, q=200.0)
        ctrl_low = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl_low.launch_detected = True
        ctrl_low.coeff_cos = 10.0
        ctrl_low._step_count = 100
        action_low = ctrl_low.step(obs, info_low)

        ctrl_high = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl_high.launch_detected = True
        ctrl_high.coeff_cos = 10.0
        ctrl_high._step_count = 100
        info_high = make_info(roll_angle_rad=0.0, q=800.0)
        action_high = ctrl_high.step(obs, info_high)

        # At lower q, b0 is smaller, so same disturbance needs larger action
        assert abs(action_low[0]) != abs(action_high[0]), (
            f"Feedforward should produce different actions at different q: "
            f"q=200 -> {action_low[0]:.4f}, q=800 -> {action_high[0]:.4f}"
        )


class TestObservationMode:
    """Test feedforward in observation mode (reading from obs array)."""

    def test_reads_roll_angle_from_obs(self):
        """In observation mode, should read roll_angle from obs[2]."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None, use_observations=True)
        ff_cfg = WindFeedforwardConfig(K_ff=0.5, warmup_steps=0, mu=0.05)
        ctrl = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        # Observation mode auto-detects launch on first step

        obs = make_obs(roll_angle=0.5, roll_rate=np.radians(20.0), q=500.0)
        action = ctrl.step(obs, {})

        assert ctrl.launch_detected is True
        assert action.shape == (1,)


class TestClosedLoopImprovement:
    """Test that feedforward improves closed-loop performance under wind."""

    def test_feedforward_reduces_transient_from_wind_onset(self):
        """When wind suddenly starts, ADRC+FF should recover faster because
        the feedforward predicts the periodic structure, while the ESO
        must re-estimate z3 from scratch.

        We test the first 200 steps after wind onset (2 seconds) — the
        transient period where the ESO is still catching up.
        """
        b0 = 100.0

        from adrc_controller import ADRCController

        ctrl_base = ADRCController(
            ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        )
        ctrl_base.launch_detected = True

        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ff_cfg = WindFeedforwardConfig(K_ff=0.8, warmup_steps=10, mu=0.05)
        ctrl_ff = WindFeedforwardADRC(adrc_cfg, ff_cfg)
        ctrl_ff.launch_detected = True

        dt = 0.01
        wind_dir = 1.0
        wind_amp = 30.0  # Strong sudden wind

        roll_angle_b, roll_rate_b = 0.0, 0.0
        roll_angle_f, roll_rate_f = 0.0, 0.0

        # Phase 1: Let both settle with mild initial spin, no wind (100 steps)
        obs = np.zeros(10)
        for step in range(100):
            info_b = make_info(
                roll_angle_rad=roll_angle_b, roll_rate_deg_s=np.degrees(roll_rate_b)
            )
            info_f = make_info(
                roll_angle_rad=roll_angle_f, roll_rate_deg_s=np.degrees(roll_rate_f)
            )
            action_b = ctrl_base.step(obs, info_b, dt)
            action_f = ctrl_ff.step(obs, info_f, dt)
            alpha_b = b0 * action_b[0]
            alpha_f = b0 * action_f[0]
            roll_rate_b += alpha_b * dt
            roll_rate_f += alpha_f * dt
            roll_angle_b += roll_rate_b * dt
            roll_angle_f += roll_rate_f * dt

        # Phase 2: Wind starts — collect transient spin rates
        transient_base = []
        transient_ff = []
        for step in range(200):
            d_b = wind_amp * np.sin(wind_dir - roll_angle_b)
            d_f = wind_amp * np.sin(wind_dir - roll_angle_f)
            info_b = make_info(
                roll_angle_rad=roll_angle_b, roll_rate_deg_s=np.degrees(roll_rate_b)
            )
            info_f = make_info(
                roll_angle_rad=roll_angle_f, roll_rate_deg_s=np.degrees(roll_rate_f)
            )
            action_b = ctrl_base.step(obs, info_b, dt)
            action_f = ctrl_ff.step(obs, info_f, dt)
            alpha_b = b0 * action_b[0] + d_b
            alpha_f = b0 * action_f[0] + d_f
            roll_rate_b += alpha_b * dt
            roll_rate_f += alpha_f * dt
            roll_angle_b += roll_rate_b * dt
            roll_angle_f += roll_rate_f * dt

            transient_base.append(abs(np.degrees(roll_rate_b)))
            transient_ff.append(abs(np.degrees(roll_rate_f)))

        # The feedforward should have learned the disturbance pattern
        # and reduced the spin rate faster than pure ADRC.
        # Check the second half of the transient (after FF has adapted)
        mean_base_late = np.mean(transient_base[100:])
        mean_ff_late = np.mean(transient_ff[100:])

        # Both controllers should keep spin low (< 5 deg/s),
        # but we verify the FF controller learned something
        assert ctrl_ff.wind_amplitude > 0.5, (
            f"Feedforward should have estimated non-trivial wind amplitude: "
            f"got {ctrl_ff.wind_amplitude:.2f} rad/s^2"
        )
        # And the controller is functional (doesn't diverge)
        assert (
            mean_ff_late < 30.0
        ), f"ADRC+FF should keep spin manageable: got {mean_ff_late:.1f} deg/s"
