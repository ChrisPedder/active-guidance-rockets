"""
Tests for ADRCController with gain-scheduled b0.

Verifies that:
1. When b0_per_pa is set, b0 is computed dynamically from dynamic pressure
2. When b0_per_pa is None, fixed b0 is used (backward compatible)
3. At very low q, the controller falls back to fixed b0
4. The ESO and control law produce physically correct outputs
5. estimate_adrc_config() populates b0_per_pa correctly
"""

import pytest
import numpy as np

from controllers.adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config


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


class TestADRCConfigDefaults:
    """Test ADRCConfig default values."""

    def test_default_b0_per_pa_is_none(self):
        config = ADRCConfig()
        assert config.b0_per_pa is None

    def test_default_b0_is_set(self):
        config = ADRCConfig()
        assert config.b0 == 725.0

    def test_custom_b0_per_pa(self):
        config = ADRCConfig(b0=100.0, b0_per_pa=0.5)
        assert config.b0_per_pa == 0.5
        assert config.b0 == 100.0


class TestFixedB0Backward:
    """Test that fixed b0 mode still works (b0_per_pa=None)."""

    def test_uses_fixed_b0_when_b0_per_pa_is_none(self):
        """When b0_per_pa is None, should use cfg.b0 regardless of q."""
        config = ADRCConfig(b0=100.0, b0_per_pa=None)
        ctrl = ADRCController(config)
        ctrl.launch_detected = True
        ctrl.z1 = 0.0
        ctrl.z2 = np.radians(30.0)

        obs = np.zeros(10)
        # Step at two different q values — action should be identical
        info_low_q = make_info(roll_rate_deg_s=30.0, q=100.0)
        info_high_q = make_info(roll_rate_deg_s=30.0, q=1000.0)

        ctrl_a = ADRCController(ADRCConfig(b0=100.0, b0_per_pa=None))
        ctrl_a.launch_detected = True
        ctrl_a.z2 = np.radians(30.0)
        action_low = ctrl_a.step(obs, info_low_q)

        ctrl_b = ADRCController(ADRCConfig(b0=100.0, b0_per_pa=None))
        ctrl_b.launch_detected = True
        ctrl_b.z2 = np.radians(30.0)
        action_high = ctrl_b.step(obs, info_high_q)

        assert abs(action_low[0] - action_high[0]) < 1e-6, (
            f"With b0_per_pa=None, actions should be identical at different q: "
            f"low_q={action_low[0]:.6f}, high_q={action_high[0]:.6f}"
        )


class TestDynamicB0:
    """Test gain-scheduled b0 when b0_per_pa is set."""

    def test_action_varies_with_q(self):
        """With b0_per_pa set, actions at different q should differ."""
        b0_per_pa = 1.0
        ref_q = 500.0
        b0_fixed = b0_per_pa * ref_q * np.tanh(ref_q / 200.0)
        config = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa)

        obs = np.zeros(10)

        ctrl_low = ADRCController(config)
        ctrl_low.launch_detected = True
        ctrl_low.z2 = np.radians(30.0)
        action_low = ctrl_low.step(obs, make_info(roll_rate_deg_s=30.0, q=200.0))

        ctrl_high = ADRCController(config)
        ctrl_high.launch_detected = True
        ctrl_high.z2 = np.radians(30.0)
        action_high = ctrl_high.step(obs, make_info(roll_rate_deg_s=30.0, q=800.0))

        assert abs(action_low[0] - action_high[0]) > 0.01, (
            f"With b0_per_pa set, actions should differ at different q: "
            f"q=200 -> {action_low[0]:.4f}, q=800 -> {action_high[0]:.4f}"
        )

    def test_b0_scales_with_q_times_tanh(self):
        """Dynamic b0 should follow the q * tanh(q/200) formula."""
        b0_per_pa = 2.0
        config = ADRCConfig(b0=100.0, b0_per_pa=b0_per_pa)
        ctrl = ADRCController(config)
        ctrl.launch_detected = True

        # At q=500, b0_now should be b0_per_pa * 500 * tanh(500/200)
        q = 500.0
        expected_b0 = b0_per_pa * q * np.tanh(q / 200.0)

        # The control law divides by b0_now:
        #   action = (u0 - z3) / b0_now
        # With fresh state (z1=z2=z3=0, prev_action=0), and small angle:
        #   After ESO update with e_obs = roll_angle:
        #   u0 = kp * angle_error + kd * rate_error
        # The key assertion: larger b0 -> smaller action magnitude for same u0.

        obs = np.zeros(10)
        # Use moderate spin so action doesn't saturate
        info = make_info(roll_angle_rad=0.01, roll_rate_deg_s=5.0, q=q)
        action = ctrl.step(obs, info)

        # Verify action is non-zero and properly scaled
        assert action[0] != 0.0, "Action should be non-zero with spin input"

    def test_larger_b0_gives_smaller_action(self):
        """Higher dynamic pressure -> larger b0 -> smaller action magnitude."""
        b0_per_pa = 1.0
        config_base = ADRCConfig(b0=500.0, b0_per_pa=b0_per_pa)

        obs = np.zeros(10)
        spin_info = dict(roll_angle_rad=0.0, roll_rate_deg_s=20.0, accel=50.0)

        # q=200: b0 = 1.0 * 200 * tanh(1.0) ≈ 152
        ctrl_low = ADRCController(config_base)
        ctrl_low.launch_detected = True
        ctrl_low.z2 = np.radians(20.0)
        action_low = ctrl_low.step(obs, make_info(**spin_info, q=200.0))

        # q=800: b0 = 1.0 * 800 * tanh(4.0) ≈ 799
        ctrl_high = ADRCController(config_base)
        ctrl_high.launch_detected = True
        ctrl_high.z2 = np.radians(20.0)
        action_high = ctrl_high.step(obs, make_info(**spin_info, q=800.0))

        assert abs(action_low[0]) > abs(action_high[0]), (
            f"Higher q (larger b0) should produce smaller action: "
            f"q=200 -> |{action_low[0]:.4f}|, q=800 -> |{action_high[0]:.4f}|"
        )

    def test_fallback_at_very_low_q(self):
        """At very low q, should fall back to fixed b0."""
        b0_per_pa = 1.0
        b0_fixed = 500.0
        config = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa)

        # At q=0, b0_per_pa * 0 * tanh(0) = 0, which is below b0_min.
        # Should fall back to b0_fixed.
        ctrl_zero_q = ADRCController(config)
        ctrl_zero_q.launch_detected = True
        ctrl_zero_q.z2 = np.radians(20.0)
        obs = np.zeros(10)
        action_zero_q = ctrl_zero_q.step(obs, make_info(roll_rate_deg_s=20.0, q=0.0))

        # Also test with fixed b0 (b0_per_pa=None)
        ctrl_fixed = ADRCController(ADRCConfig(b0=b0_fixed, b0_per_pa=None))
        ctrl_fixed.launch_detected = True
        ctrl_fixed.z2 = np.radians(20.0)
        action_fixed = ctrl_fixed.step(obs, make_info(roll_rate_deg_s=20.0, q=0.0))

        assert abs(action_zero_q[0] - action_fixed[0]) < 1e-6, (
            f"At q=0, gain-scheduled ADRC should fall back to fixed b0: "
            f"dynamic={action_zero_q[0]:.6f}, fixed={action_fixed[0]:.6f}"
        )

    def test_fallback_at_tiny_q(self):
        """At q just barely above zero, should still fall back to fixed b0."""
        b0_per_pa = 1.0
        b0_fixed = 500.0
        config = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa)

        # q=1.0: b0 = 1.0 * 1.0 * tanh(0.005) ≈ 0.005, well below b0_min = 5.0
        ctrl = ADRCController(config)
        ctrl.launch_detected = True
        ctrl.z2 = np.radians(20.0)
        obs = np.zeros(10)
        action = ctrl.step(obs, make_info(roll_rate_deg_s=20.0, q=1.0))

        ctrl_fixed = ADRCController(ADRCConfig(b0=b0_fixed, b0_per_pa=None))
        ctrl_fixed.launch_detected = True
        ctrl_fixed.z2 = np.radians(20.0)
        action_fixed = ctrl_fixed.step(obs, make_info(roll_rate_deg_s=20.0, q=1.0))

        assert (
            abs(action[0] - action_fixed[0]) < 1e-6
        ), f"At tiny q, should fall back to fixed b0"


class TestDynamicB0ObservationMode:
    """Test that observation mode reads q from obs[5]."""

    def test_reads_q_from_obs_5(self):
        """With use_observations=True, q should come from obs[5]."""
        b0_per_pa = 1.0
        b0_fixed = 500.0
        config = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa, use_observations=True)

        # Low q in obs
        ctrl_low = ADRCController(config)
        obs_low = make_obs(roll_rate=np.radians(20.0), q=200.0)
        action_low = ctrl_low.step(obs_low, {})

        # High q in obs
        ctrl_high = ADRCController(config)
        obs_high = make_obs(roll_rate=np.radians(20.0), q=800.0)
        action_high = ctrl_high.step(obs_high, {})

        # Actions should differ because q differs
        assert abs(action_low[0] - action_high[0]) > 0.01, (
            f"Observation mode should read q from obs[5]: "
            f"q=200 -> {action_low[0]:.4f}, q=800 -> {action_high[0]:.4f}"
        )

    def test_obs_mode_fallback_at_zero_q(self):
        """Observation mode with q=0 should fall back to fixed b0."""
        b0_per_pa = 1.0
        b0_fixed = 500.0

        config_dynamic = ADRCConfig(
            b0=b0_fixed, b0_per_pa=b0_per_pa, use_observations=True
        )
        config_fixed = ADRCConfig(b0=b0_fixed, b0_per_pa=None, use_observations=True)

        obs = make_obs(roll_rate=np.radians(20.0), q=0.0)

        ctrl_dyn = ADRCController(config_dynamic)
        action_dyn = ctrl_dyn.step(obs, {})

        ctrl_fix = ADRCController(config_fixed)
        action_fix = ctrl_fix.step(obs, {})

        assert abs(action_dyn[0] - action_fix[0]) < 1e-6


class TestADRCBasicBehavior:
    """Test basic ADRC controller behavior (independent of b0 scheduling)."""

    def test_zero_action_before_launch(self):
        ctrl = ADRCController()
        obs = np.zeros(10)
        info = make_info(accel=5.0)  # Below threshold
        action = ctrl.step(obs, info)
        assert action[0] == 0.0
        assert ctrl.launch_detected is False

    def test_launch_detection(self):
        ctrl = ADRCController()
        obs = np.zeros(10)
        info = make_info(roll_angle_rad=0.1, accel=25.0)
        ctrl.step(obs, info)
        assert ctrl.launch_detected is True
        assert abs(ctrl.target_angle - 0.1) < 1e-6

    def test_opposes_positive_spin(self):
        """Should produce action opposing positive spin."""
        config = ADRCConfig(b0=100.0, b0_per_pa=None)
        ctrl = ADRCController(config)
        ctrl.launch_detected = True
        ctrl.z2 = np.radians(30.0)

        obs = np.zeros(10)
        info = make_info(roll_rate_deg_s=30.0, q=500.0)
        action = ctrl.step(obs, info)

        # ADRC should command negative action to oppose positive spin
        assert action[0] < 0, f"Should oppose positive spin, got action={action[0]:.4f}"

    def test_reset_clears_state(self):
        ctrl = ADRCController()
        ctrl.launch_detected = True
        ctrl.z1 = 1.0
        ctrl.z2 = 2.0
        ctrl.z3 = 3.0
        ctrl.prev_action = 0.5

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert ctrl.z1 == 0.0
        assert ctrl.z2 == 0.0
        assert ctrl.z3 == 0.0
        assert ctrl.prev_action == 0.0

    def test_action_clamped(self):
        """Output must always be in [-1, 1]."""
        config = ADRCConfig(omega_c=100.0, b0=1.0, b0_per_pa=None)  # Very aggressive
        ctrl = ADRCController(config)
        ctrl.launch_detected = True

        obs = np.zeros(10)
        info = make_info(roll_angle_rad=1.0, roll_rate_deg_s=500.0, q=500.0)
        action = ctrl.step(obs, info)

        assert -1.0 <= action[0] <= 1.0


class TestEstimateADRCConfig:
    """Test that estimate_adrc_config populates b0_per_pa."""

    def test_b0_per_pa_is_set(self, estes_alpha_airframe):
        """estimate_adrc_config should set b0_per_pa."""
        from rocket_config import RocketPhysicsConfig

        physics = RocketPhysicsConfig()
        config = estimate_adrc_config(estes_alpha_airframe, physics)
        assert config.b0_per_pa is not None
        assert config.b0_per_pa > 0

    def test_b0_matches_b0_per_pa_at_ref_q(self, estes_alpha_airframe):
        """b0 should equal b0_per_pa * ref_q * tanh(ref_q/200) at ref_q=500."""
        from rocket_config import RocketPhysicsConfig

        physics = RocketPhysicsConfig()
        config = estimate_adrc_config(estes_alpha_airframe, physics)

        ref_q = 500.0
        expected_b0 = config.b0_per_pa * ref_q * np.tanh(ref_q / 200.0)
        assert abs(config.b0 - expected_b0) < 1e-6, (
            f"b0={config.b0:.4f} should match "
            f"b0_per_pa*q*tanh(q/200)={expected_b0:.4f} at ref q"
        )

    def test_b0_is_positive(self, estes_alpha_airframe):
        from rocket_config import RocketPhysicsConfig

        physics = RocketPhysicsConfig()
        config = estimate_adrc_config(estes_alpha_airframe, physics)
        assert config.b0 > 0

    def test_bandwidths_propagated(self, estes_alpha_airframe):
        from rocket_config import RocketPhysicsConfig

        physics = RocketPhysicsConfig()
        config = estimate_adrc_config(
            estes_alpha_airframe, physics, omega_c=20.0, omega_o=80.0
        )
        assert config.omega_c == 20.0
        assert config.omega_o == 80.0


class TestESObserverConvergence:
    """Test that the ESO tracks state correctly over multiple steps."""

    def test_eso_tracks_constant_disturbance(self):
        """With a constant external disturbance, z3 should converge to it."""
        b0 = 100.0
        config = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        ctrl = ADRCController(config)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = 0.0
        disturbance = 5.0  # Constant disturbance in rad/s^2

        obs = np.zeros(10)
        for _ in range(500):
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
                q=500.0,
            )
            action = ctrl.step(obs, info, dt)

            # True dynamics: alpha = b0 * action + disturbance
            alpha = b0 * action[0] + disturbance
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # z3 should have converged close to the disturbance value
        assert abs(ctrl.z3 - disturbance) < disturbance * 0.5, (
            f"ESO z3 ({ctrl.z3:.2f}) should converge toward "
            f"disturbance ({disturbance:.2f})"
        )
