"""
Tests for sta_smc_controller.py

Tests for Super-Twisting Sliding Mode Controller (STA-SMC) for rocket roll
stabilization. Covers configuration, sliding surface, gain conditions, gain
scheduling, control output, launch detection, convergence, interface, and
dynamic b0 computation.
"""

import pytest
import numpy as np

from sta_smc_controller import STASMCConfig, STASMCController


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(roll_angle=0.0, roll_rate=0.0, dynamic_pressure=500.0, length=10):
    """Create a minimal observation vector with specified values."""
    obs = np.zeros(length, dtype=np.float32)
    obs[2] = roll_angle  # roll angle (rad)
    obs[3] = roll_rate  # roll rate (rad/s)
    obs[5] = dynamic_pressure  # dynamic pressure (Pa)
    return obs


def _make_info(
    roll_angle_rad=0.0,
    roll_rate_deg_s=0.0,
    vertical_acceleration_ms2=50.0,
    dynamic_pressure_Pa=500.0,
):
    """Create a minimal info dict with specified values."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": vertical_acceleration_ms2,
        "dynamic_pressure_Pa": dynamic_pressure_Pa,
    }


# ===========================================================================
# 1. TestSTASMCConfig
# ===========================================================================


class TestSTASMCConfig:
    """Tests for the STASMCConfig dataclass."""

    def test_default_values(self):
        """Default config should have documented values."""
        cfg = STASMCConfig()
        assert cfg.c == 10.0
        assert cfg.alpha == 5.0
        assert cfg.beta == 10.0
        assert cfg.D_max == 10.0
        assert cfg.b0 == 725.0
        assert cfg.b0_per_pa is None
        assert cfg.q_ref == 500.0
        assert cfg.max_deflection == 30.0
        assert cfg.use_observations is False

    def test_custom_values(self):
        """Custom config values should be stored correctly."""
        cfg = STASMCConfig(
            c=20.0,
            alpha=8.0,
            beta=15.0,
            D_max=5.0,
            b0=1000.0,
            b0_per_pa=1.5,
            q_ref=600.0,
            max_deflection=15.0,
            use_observations=True,
        )
        assert cfg.c == 20.0
        assert cfg.alpha == 8.0
        assert cfg.beta == 15.0
        assert cfg.D_max == 5.0
        assert cfg.b0 == 1000.0
        assert cfg.b0_per_pa == 1.5
        assert cfg.q_ref == 600.0
        assert cfg.max_deflection == 15.0
        assert cfg.use_observations is True

    def test_default_gains_validity(self):
        """Default config: alpha > sqrt(2*D_max) holds but beta == D_max
        so gains_valid is False (requires strictly greater)."""
        cfg = STASMCConfig()
        controller = STASMCController(cfg)
        # alpha=5.0 > sqrt(2*10)=4.47 => True
        # beta=10.0 > D_max=10.0 => False (not strictly greater)
        assert not controller.gains_valid


# ===========================================================================
# 2. TestSlidingSurface
# ===========================================================================


class TestSlidingSurface:
    """Tests for the sliding surface sigma = roll_rate + c * angle_error."""

    def test_sigma_zero_at_equilibrium(self):
        """At zero roll rate and zero angle error, sigma = 0 and action ~ 0."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=0.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        # At equilibrium the sliding surface is zero so the proportional
        # super-twisting term v1 is zero. Only the integrator v2 contributes,
        # and after one step from reset it should be negligible.
        assert abs(action[0]) < 0.01

    def test_sigma_positive_for_positive_rate(self):
        """Positive roll rate with zero angle error produces sigma > 0
        and should yield a negative (corrective) action."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=30.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        # Positive sigma -> v1 < 0 -> action < 0 (corrective)
        assert action[0] < 0

    def test_sigma_includes_angle_error(self):
        """Angle error contribution to sigma should affect the output."""
        cfg = STASMCConfig(c=10.0, use_observations=False)
        controller = STASMCController(cfg)
        controller.reset()

        # First step: detect launch at angle=0 to set target_angle=0
        info_launch = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=0.0, vertical_acceleration_ms2=50.0
        )
        obs_launch = _make_obs()
        controller.step(obs_launch, info_launch, dt=0.01)

        # Now drift to angle=0.2 with zero rate -> angle_error = 0.2
        angle = 0.2  # rad
        info = _make_info(
            roll_angle_rad=angle, roll_rate_deg_s=0.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        # sigma = 0 + 10 * (0.2 - 0) = 2.0 > 0 -> negative action
        assert action[0] < 0


# ===========================================================================
# 3. TestGainConditions
# ===========================================================================


class TestGainConditions:
    """Tests for gains_valid property: alpha > sqrt(2*D_max), beta > D_max."""

    def test_valid_gains(self):
        """Gains satisfying both conditions should be valid."""
        cfg = STASMCConfig(alpha=10.0, beta=20.0, D_max=5.0)
        controller = STASMCController(cfg)
        # alpha=10 > sqrt(10)=3.16, beta=20 > 5
        assert controller.gains_valid

    def test_invalid_alpha(self):
        """Alpha below sqrt(2*D_max) should fail."""
        cfg = STASMCConfig(alpha=1.0, beta=20.0, D_max=5.0)
        controller = STASMCController(cfg)
        # alpha=1 < sqrt(10)=3.16
        assert not controller.gains_valid

    def test_invalid_beta(self):
        """Beta below D_max should fail."""
        cfg = STASMCConfig(alpha=10.0, beta=3.0, D_max=5.0)
        controller = STASMCController(cfg)
        # beta=3 < D_max=5
        assert not controller.gains_valid

    def test_both_invalid(self):
        """Both conditions violated should fail."""
        cfg = STASMCConfig(alpha=0.5, beta=0.5, D_max=5.0)
        controller = STASMCController(cfg)
        assert not controller.gains_valid

    def test_boundary_alpha(self):
        """Alpha exactly at the boundary is NOT strictly greater."""
        D_max = 8.0
        boundary_alpha = np.sqrt(2 * D_max)
        cfg = STASMCConfig(alpha=boundary_alpha, beta=D_max + 1.0, D_max=D_max)
        controller = STASMCController(cfg)
        # alpha == sqrt(2*D_max), not strictly greater
        assert not controller.gains_valid


# ===========================================================================
# 4. TestGainScheduling
# ===========================================================================


class TestGainScheduling:
    """Tests for _gain_scale at different dynamic pressures."""

    def test_gain_scale_at_q_ref(self):
        """At q = q_ref, gain scale should be approximately 1.0."""
        cfg = STASMCConfig(q_ref=500.0)
        controller = STASMCController(cfg)
        scale = controller._gain_scale(500.0)
        assert abs(scale - 1.0) < 0.05

    def test_gain_scale_at_zero_q(self):
        """At q = 0, effectiveness is negligible and scale should clamp to 5.0."""
        cfg = STASMCConfig(q_ref=500.0)
        controller = STASMCController(cfg)
        scale = controller._gain_scale(0.0)
        assert scale == 5.0

    def test_gain_scale_at_low_q(self):
        """At low q, scale should be larger than 1 (gains boosted)."""
        cfg = STASMCConfig(q_ref=500.0)
        controller = STASMCController(cfg)
        scale = controller._gain_scale(100.0)
        assert scale > 1.0

    def test_gain_scale_at_high_q(self):
        """At high q, scale should be smaller than 1 (gains reduced)."""
        cfg = STASMCConfig(q_ref=500.0)
        controller = STASMCController(cfg)
        scale = controller._gain_scale(1000.0)
        assert scale < 1.0

    def test_gain_scale_clipped_low(self):
        """Scale should not go below 0.5."""
        cfg = STASMCConfig(q_ref=100.0)
        controller = STASMCController(cfg)
        # Very high q with low q_ref => scale wants to be very small
        scale = controller._gain_scale(5000.0)
        assert scale >= 0.5

    def test_gain_scale_clipped_high(self):
        """Scale should not exceed 5.0."""
        cfg = STASMCConfig(q_ref=1000.0)
        controller = STASMCController(cfg)
        # Very low q with high q_ref => scale wants to be very large
        scale = controller._gain_scale(1.0)
        assert scale <= 5.0


# ===========================================================================
# 5. TestControlOutput
# ===========================================================================


class TestControlOutput:
    """Tests for control output properties."""

    def test_zero_state_zero_action(self):
        """Zero roll angle and rate at target should produce near-zero action."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=0.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        assert abs(action[0]) < 0.01

    def test_nonzero_spin_nonzero_action(self):
        """Nonzero roll rate should produce a nonzero corrective action."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=50.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        assert abs(action[0]) > 0.001

    def test_action_bounded(self):
        """Action must always be clipped to [-1, 1]."""
        # Use high gains and large disturbance to try to exceed bounds
        cfg = STASMCConfig(
            alpha=100.0, beta=200.0, D_max=5.0, b0=10.0, use_observations=False
        )
        controller = STASMCController(cfg)
        controller.reset()

        info = _make_info(
            roll_angle_rad=3.0, roll_rate_deg_s=500.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        assert -1.0 <= action[0] <= 1.0

    def test_direction_positive_rate_negative_action(self):
        """Positive roll rate should produce a negative action tendency."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=30.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        assert action[0] < 0

    def test_direction_negative_rate_positive_action(self):
        """Negative roll rate should produce a positive action tendency."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=-30.0, vertical_acceleration_ms2=50.0
        )
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)
        assert action[0] > 0


# ===========================================================================
# 6. TestLaunchDetection
# ===========================================================================


class TestLaunchDetection:
    """Tests for launch detection in ground-truth and observation modes."""

    def test_ground_truth_requires_accel_above_threshold(self):
        """In ground-truth mode, launch requires accel > 20 m/s^2."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        # Below threshold
        info = _make_info(vertical_acceleration_ms2=10.0)
        obs = _make_obs()
        action = controller.step(obs, info, dt=0.01)

        assert not controller.launch_detected
        assert action[0] == 0.0

    def test_ground_truth_detects_launch_above_threshold(self):
        """In ground-truth mode, accel > 20 triggers launch."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(roll_angle_rad=0.5, vertical_acceleration_ms2=25.0)
        obs = _make_obs()
        action = controller.step(obs, info, dt=0.01)

        assert controller.launch_detected
        assert controller.target_angle == 0.5

    def test_obs_mode_auto_detects_launch(self):
        """In observation mode, launch is detected on the first step."""
        controller = STASMCController(STASMCConfig(use_observations=True))
        controller.reset()

        obs = _make_obs(roll_angle=0.3)
        info = _make_info()  # accel doesn't matter in obs mode
        action = controller.step(obs, info, dt=0.01)

        assert controller.launch_detected
        assert controller.target_angle == pytest.approx(0.3, abs=1e-5)

    def test_target_angle_set_on_launch(self):
        """Target angle should be captured at the moment of launch detection."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        launch_angle = 1.2
        info = _make_info(roll_angle_rad=launch_angle, vertical_acceleration_ms2=50.0)
        obs = _make_obs()
        controller.step(obs, info, dt=0.01)

        assert controller.target_angle == pytest.approx(launch_angle)


# ===========================================================================
# 7. TestConvergence
# ===========================================================================


class TestConvergence:
    """Test convergence of spin rate under simple dynamics."""

    def test_spin_rate_decreases_over_time(self):
        """Simulate roll_accel = b0 * action for ~200 steps from 30 deg/s.

        The spin rate should decrease significantly, demonstrating that
        the STA-SMC controller drives the system toward zero spin.
        """
        b0 = 725.0
        cfg = STASMCConfig(
            c=10.0,
            alpha=5.0,
            beta=10.0,
            D_max=10.0,
            b0=b0,
            use_observations=False,
            q_ref=500.0,
        )
        controller = STASMCController(cfg)
        controller.reset()

        dt = 0.01
        roll_rate_rad = np.radians(30.0)  # 30 deg/s initial spin
        roll_angle = 0.0
        q = 500.0

        # Detect launch on first step
        info = _make_info(
            roll_angle_rad=roll_angle,
            roll_rate_deg_s=np.degrees(roll_rate_rad),
            vertical_acceleration_ms2=50.0,
            dynamic_pressure_Pa=q,
        )
        obs = _make_obs(
            roll_angle=roll_angle, roll_rate=roll_rate_rad, dynamic_pressure=q
        )
        controller.step(obs, info, dt=dt)

        initial_rate = abs(roll_rate_rad)

        # Simulate for 200 steps
        for _ in range(200):
            info = _make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate_rad),
                vertical_acceleration_ms2=50.0,
                dynamic_pressure_Pa=q,
            )
            obs = _make_obs(
                roll_angle=roll_angle, roll_rate=roll_rate_rad, dynamic_pressure=q
            )

            action = controller.step(obs, info, dt=dt)

            # Simple dynamics: roll_accel = b0 * action
            roll_accel = b0 * action[0]
            roll_rate_rad += roll_accel * dt
            roll_angle += roll_rate_rad * dt

        final_rate = abs(roll_rate_rad)

        # Spin rate should have decreased significantly (at least 50%)
        assert final_rate < initial_rate * 0.5, (
            f"Spin rate did not decrease enough: {np.degrees(initial_rate):.1f} "
            f"-> {np.degrees(final_rate):.1f} deg/s"
        )

    def test_convergence_from_negative_spin(self):
        """Controller should also converge from negative initial spin."""
        b0 = 725.0
        cfg = STASMCConfig(b0=b0, use_observations=False, q_ref=500.0)
        controller = STASMCController(cfg)
        controller.reset()

        dt = 0.01
        roll_rate_rad = np.radians(-30.0)  # Negative spin
        roll_angle = 0.0
        q = 500.0

        # Launch detection step
        info = _make_info(
            roll_angle_rad=roll_angle,
            roll_rate_deg_s=np.degrees(roll_rate_rad),
            vertical_acceleration_ms2=50.0,
            dynamic_pressure_Pa=q,
        )
        obs = _make_obs(
            roll_angle=roll_angle, roll_rate=roll_rate_rad, dynamic_pressure=q
        )
        controller.step(obs, info, dt=dt)

        initial_rate = abs(roll_rate_rad)

        for _ in range(200):
            info = _make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate_rad),
                vertical_acceleration_ms2=50.0,
                dynamic_pressure_Pa=q,
            )
            obs = _make_obs(
                roll_angle=roll_angle, roll_rate=roll_rate_rad, dynamic_pressure=q
            )

            action = controller.step(obs, info, dt=dt)
            roll_accel = b0 * action[0]
            roll_rate_rad += roll_accel * dt
            roll_angle += roll_rate_rad * dt

        final_rate = abs(roll_rate_rad)
        assert final_rate < initial_rate * 0.5


# ===========================================================================
# 8. TestInterface
# ===========================================================================


class TestInterface:
    """Tests for controller interface: reset, step shape, obs mode."""

    def test_reset_clears_state(self):
        """Reset should clear launch_detected, target_angle, and v2."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.launch_detected = True
        controller.target_angle = 1.5
        controller.v2 = 42.0

        controller.reset()

        assert controller.launch_detected is False
        assert controller.target_angle == 0.0
        assert controller.v2 == 0.0

    def test_step_returns_correct_shape(self):
        """Step should return a numpy array of shape (1,)."""
        controller = STASMCController(STASMCConfig(use_observations=False))
        controller.reset()

        info = _make_info(vertical_acceleration_ms2=50.0)
        obs = _make_obs()

        action = controller.step(obs, info, dt=0.01)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_step_with_use_observations_true(self):
        """Controller should work when reading from obs instead of info."""
        controller = STASMCController(STASMCConfig(use_observations=True))
        controller.reset()

        obs = _make_obs(
            roll_angle=0.0, roll_rate=np.radians(20.0), dynamic_pressure=500.0
        )
        info = {}  # Info can be empty in obs mode

        action = controller.step(obs, info, dt=0.01)

        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        # Positive rate -> negative corrective action
        assert action[0] < 0

    def test_default_config_when_none(self):
        """Passing None as config should use defaults."""
        controller = STASMCController(None)
        assert controller.config.c == 10.0
        assert controller.config.alpha == 5.0


# ===========================================================================
# 9. TestDynamicB0
# ===========================================================================


class TestDynamicB0:
    """Tests for dynamic b0 computation via b0_per_pa."""

    def test_fixed_b0_when_b0_per_pa_is_none(self):
        """When b0_per_pa is None, _get_b0 returns the fixed b0."""
        cfg = STASMCConfig(b0=725.0, b0_per_pa=None)
        controller = STASMCController(cfg)

        obs = _make_obs(dynamic_pressure=500.0)
        info = _make_info(dynamic_pressure_Pa=500.0)

        b0 = controller._get_b0(obs, info)
        assert b0 == 725.0

    def test_dynamic_b0_with_b0_per_pa(self):
        """When b0_per_pa is set, b0 should scale with q * tanh(q/200)."""
        cfg = STASMCConfig(b0=725.0, b0_per_pa=1.5, use_observations=False)
        controller = STASMCController(cfg)

        q = 500.0
        obs = _make_obs(dynamic_pressure=q)
        info = _make_info(dynamic_pressure_Pa=q)

        b0 = controller._get_b0(obs, info)
        expected = 1.5 * q * np.tanh(q / 200.0)
        assert abs(b0 - expected) < 1e-6

    def test_dynamic_b0_fallback_at_low_q(self):
        """At very low q, dynamic b0 should fall back to fixed b0."""
        cfg = STASMCConfig(b0=725.0, b0_per_pa=1.5, use_observations=False)
        controller = STASMCController(cfg)

        q = 0.001  # Very low q
        obs = _make_obs(dynamic_pressure=q)
        info = _make_info(dynamic_pressure_Pa=q)

        b0 = controller._get_b0(obs, info)
        # b0_per_pa * q * tanh(q/200) ~ 1.5 * 0.001 * 5e-6 ~ 7.5e-9
        # This is below b0 * 0.01 = 7.25, so falls back to fixed b0
        assert b0 == cfg.b0

    def test_dynamic_b0_reads_from_obs_in_obs_mode(self):
        """In observation mode, b0 should be computed from obs[5]."""
        cfg = STASMCConfig(b0=725.0, b0_per_pa=1.5, use_observations=True)
        controller = STASMCController(cfg)

        q = 400.0
        obs = _make_obs(dynamic_pressure=q)
        info = {}  # Info not used in obs mode

        b0 = controller._get_b0(obs, info)
        expected = 1.5 * q * np.tanh(q / 200.0)
        assert abs(b0 - expected) < 1e-6

    def test_dynamic_b0_reads_from_info_in_gt_mode(self):
        """In ground-truth mode, b0 should be computed from info dict."""
        cfg = STASMCConfig(b0=725.0, b0_per_pa=1.5, use_observations=False)
        controller = STASMCController(cfg)

        q = 600.0
        obs = _make_obs(dynamic_pressure=0.0)  # obs[5] is different
        info = _make_info(dynamic_pressure_Pa=q)

        b0 = controller._get_b0(obs, info)
        expected = 1.5 * q * np.tanh(q / 200.0)
        assert abs(b0 - expected) < 1e-6
