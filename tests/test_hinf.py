"""
Tests for the H-infinity (LQG/LTR) robust controller.

Verifies:
1. HinfConfig default values and custom configuration
2. LQG/LTR synthesis (stability, gain shapes, sensitivity to weights)
3. Tustin discretization (shapes, stability, dt sensitivity)
4. Gain scheduling (_gain_scale behavior across dynamic pressure)
5. Dynamic b0 (fixed vs gain-scheduled, fallback at low q)
6. Control output (shape, dtype, range, proportionality)
7. Launch detection (ground truth vs observation mode)
8. Convergence (step response, disturbance rejection)
9. Controller interface (reset, step, multi-episode)
"""

import numpy as np
import pytest

from hinf_controller import (
    HinfController,
    HinfConfig,
    synthesize_lqg_ltr,
    _discretize_tustin,
)


# --- Helpers ---


def _make_obs(angle=0.0, rate=0.0, q=500.0):
    """Create a standard observation array for testing."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = angle
    obs[3] = rate
    obs[5] = q
    return obs


def _make_info(angle=0.0, rate_deg=0.0, accel=50.0, q=500.0):
    """Create a standard info dict for testing (ground-truth mode)."""
    return {
        "roll_angle_rad": angle,
        "roll_rate_deg_s": rate_deg,
        "vertical_acceleration_ms2": accel,
        "dynamic_pressure_Pa": q,
    }


# ======================================================================
# 1. TestHinfConfig
# ======================================================================


class TestHinfConfig:
    """Test HinfConfig defaults and custom values."""

    def test_default_values(self):
        cfg = HinfConfig()
        assert cfg.q_angle == 100.0
        assert cfg.q_rate == 10.0
        assert cfg.r_control == 0.01
        assert cfg.w_process == 100.0
        assert cfg.v_angle == 0.001
        assert cfg.v_rate == 0.001
        assert cfg.b0 == 725.0
        assert cfg.b0_per_pa is None
        assert cfg.q_ref == 500.0
        assert cfg.use_observations is False
        assert cfg.dt_design == 0.01

    def test_custom_config(self):
        cfg = HinfConfig(
            q_angle=200.0,
            q_rate=50.0,
            r_control=0.1,
            w_process=500.0,
            b0=300.0,
            b0_per_pa=1.5,
            q_ref=400.0,
            use_observations=True,
            dt_design=0.005,
        )
        assert cfg.q_angle == 200.0
        assert cfg.q_rate == 50.0
        assert cfg.r_control == 0.1
        assert cfg.w_process == 500.0
        assert cfg.b0 == 300.0
        assert cfg.b0_per_pa == 1.5
        assert cfg.q_ref == 400.0
        assert cfg.use_observations is True
        assert cfg.dt_design == 0.005

    def test_max_deflection_default(self):
        cfg = HinfConfig()
        assert cfg.max_deflection == 30.0


# ======================================================================
# 2. TestSynthesis
# ======================================================================


class TestSynthesis:
    """Test LQG/LTR synthesis function."""

    def test_synthesis_succeeds(self):
        result = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        assert result["stable"] is True

    def test_closed_loop_eigenvalues_negative_real_parts(self):
        result = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        eigs = result["cl_eigenvalues"]
        assert len(eigs) == 4  # 2 plant states + 2 controller states
        for eig in eigs:
            assert (
                np.real(eig) < 0
            ), f"Closed-loop eigenvalue {eig} should have negative real part"

    def test_lqr_gain_shape(self):
        result = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        K = result["K"]
        assert K.shape == (1, 2), f"LQR gain K should be (1,2), got {K.shape}"

    def test_kalman_gain_shape(self):
        result = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        L = result["L"]
        assert L.shape == (2, 2), f"Kalman gain L should be (2,2), got {L.shape}"

    def test_higher_q_angle_increases_angle_gain(self):
        """Higher q_angle should produce a larger K[0,0] (angle gain)."""
        r_low = synthesize_lqg_ltr(
            q_angle=10.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        r_high = synthesize_lqg_ltr(
            q_angle=1000.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        assert abs(r_high["K"][0, 0]) > abs(r_low["K"][0, 0]), (
            f"Higher q_angle should produce larger |K[0,0]|: "
            f"low={abs(r_low['K'][0, 0]):.4f}, high={abs(r_high['K'][0, 0]):.4f}"
        )

    def test_higher_w_process_increases_kalman_gain(self):
        """Higher w_process (LTR) should produce larger Kalman gain elements."""
        r_low = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=1.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        r_high = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=10000.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        L_low_norm = np.linalg.norm(r_low["L"])
        L_high_norm = np.linalg.norm(r_high["L"])
        assert L_high_norm > L_low_norm, (
            f"Higher w_process should produce larger ||L||: "
            f"low={L_low_norm:.4f}, high={L_high_norm:.4f}"
        )


# ======================================================================
# 3. TestDiscretization
# ======================================================================


class TestDiscretization:
    """Test Tustin (bilinear) discretization."""

    def _get_continuous_matrices(self):
        """Get a known-stable continuous-time controller."""
        result = synthesize_lqg_ltr(
            q_angle=100.0,
            q_rate=10.0,
            r_control=0.01,
            w_process=100.0,
            v_angle=0.001,
            v_rate=0.001,
        )
        return result["A_K"], result["B_K"], result["C_K"], result["D_K"]

    def test_discrete_matrices_shapes(self):
        A_K, B_K, C_K, D_K = self._get_continuous_matrices()
        Ad, Bd, Cd, Dd = _discretize_tustin(A_K, B_K, C_K, D_K, dt=0.01)
        assert Ad.shape == (2, 2), f"Ad shape should be (2,2), got {Ad.shape}"
        assert Bd.shape == (2, 2), f"Bd shape should be (2,2), got {Bd.shape}"
        assert Cd.shape == (1, 2), f"Cd shape should be (1,2), got {Cd.shape}"
        assert Dd.shape == (1, 2), f"Dd shape should be (1,2), got {Dd.shape}"

    def test_discrete_system_stable(self):
        """Eigenvalues of Ad should be inside the unit circle."""
        A_K, B_K, C_K, D_K = self._get_continuous_matrices()
        Ad, Bd, Cd, Dd = _discretize_tustin(A_K, B_K, C_K, D_K, dt=0.01)
        eigs = np.linalg.eigvals(Ad)
        for eig in eigs:
            assert abs(eig) < 1.0, (
                f"Discrete eigenvalue {eig} (|eig|={abs(eig):.6f}) "
                f"should be inside the unit circle"
            )

    def test_different_dt_produces_different_Ad(self):
        A_K, B_K, C_K, D_K = self._get_continuous_matrices()
        Ad_fast, _, _, _ = _discretize_tustin(A_K, B_K, C_K, D_K, dt=0.001)
        Ad_slow, _, _, _ = _discretize_tustin(A_K, B_K, C_K, D_K, dt=0.05)
        assert not np.allclose(
            Ad_fast, Ad_slow
        ), "Different dt values should produce different discrete-time Ad matrices"


# ======================================================================
# 4. TestGainScheduling
# ======================================================================


class TestGainScheduling:
    """Test _gain_scale method."""

    def test_scale_near_one_at_q_ref(self):
        """At q_ref, gain scale should be approximately 1.0."""
        ctrl = HinfController(HinfConfig(q_ref=500.0))
        scale = ctrl._gain_scale(500.0)
        assert (
            abs(scale - 1.0) < 0.05
        ), f"Scale at q_ref should be ~1.0, got {scale:.4f}"

    def test_scale_increases_at_low_q(self):
        """At low dynamic pressure, gain scale should be > 1 (more aggressive)."""
        ctrl = HinfController(HinfConfig(q_ref=500.0))
        scale_low = ctrl._gain_scale(100.0)
        scale_ref = ctrl._gain_scale(500.0)
        assert (
            scale_low > scale_ref
        ), f"Scale at low q ({scale_low:.4f}) should be > scale at q_ref ({scale_ref:.4f})"

    def test_scale_decreases_at_high_q(self):
        """At high dynamic pressure, gain scale should be < 1 (less aggressive)."""
        ctrl = HinfController(HinfConfig(q_ref=500.0))
        scale_high = ctrl._gain_scale(1000.0)
        scale_ref = ctrl._gain_scale(500.0)
        assert (
            scale_high < scale_ref
        ), f"Scale at high q ({scale_high:.4f}) should be < scale at q_ref ({scale_ref:.4f})"

    def test_scale_clamped_upper_bound(self):
        """At very low q, scale should be clamped to 5.0."""
        ctrl = HinfController(HinfConfig(q_ref=500.0))
        scale = ctrl._gain_scale(0.001)
        assert scale == 5.0, f"Scale at near-zero q should be 5.0, got {scale}"

    def test_scale_clamped_lower_bound(self):
        """At very high q, scale should be clamped to 0.5."""
        ctrl = HinfController(HinfConfig(q_ref=500.0))
        # Need extremely high q to hit the 0.5 clamp
        # effectiveness = q * tanh(q/200); for large q, tanh -> 1, so eff ~ q
        # ref_eff = 500 * tanh(500/200) ~ 500 * 0.986 ~ 493
        # scale = 493 / (q * 1.0) = 493/q; scale=0.5 => q=986
        # But clamp is at 0.5, so we need q large enough
        scale = ctrl._gain_scale(5000.0)
        assert (
            scale == 0.5
        ), f"Scale at very high q should be clamped to 0.5, got {scale}"


# ======================================================================
# 5. TestDynamicB0
# ======================================================================


class TestDynamicB0:
    """Test dynamic b0 calculation from dynamic pressure."""

    def test_fixed_b0_when_b0_per_pa_is_none(self):
        """With b0_per_pa=None, should always use fixed b0."""
        cfg = HinfConfig(b0=725.0, b0_per_pa=None)
        ctrl = HinfController(cfg)
        obs = _make_obs(q=100.0)
        info = _make_info(q=100.0)
        b0_low = ctrl._get_b0(obs, info)

        info2 = _make_info(q=1000.0)
        obs2 = _make_obs(q=1000.0)
        b0_high = ctrl._get_b0(obs2, info2)

        assert b0_low == 725.0
        assert b0_high == 725.0

    def test_dynamic_b0_from_q(self):
        """With b0_per_pa set, b0 should vary with dynamic pressure."""
        cfg = HinfConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = HinfController(cfg)

        obs_low = _make_obs(q=200.0)
        info_low = _make_info(q=200.0)
        b0_low = ctrl._get_b0(obs_low, info_low)

        obs_high = _make_obs(q=800.0)
        info_high = _make_info(q=800.0)
        b0_high = ctrl._get_b0(obs_high, info_high)

        assert (
            b0_high > b0_low
        ), f"b0 at high q ({b0_high:.2f}) should be > b0 at low q ({b0_low:.2f})"

    def test_fallback_at_very_low_q(self):
        """At very low q where b0_per_pa * q * tanh(q/200) < b0*0.01, fall back to b0."""
        cfg = HinfConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = HinfController(cfg)
        obs = _make_obs(q=0.01)
        info = _make_info(q=0.01)
        b0_val = ctrl._get_b0(obs, info)
        assert (
            b0_val == 725.0
        ), f"At near-zero q should fall back to fixed b0=725.0, got {b0_val}"

    def test_observation_mode_reads_obs5(self):
        """In observation mode, b0 should be computed from obs[5]."""
        cfg = HinfConfig(b0=725.0, b0_per_pa=1.5, use_observations=True)
        ctrl = HinfController(cfg)
        obs = _make_obs(q=600.0)
        info = {}  # Empty info in obs mode

        b0_val = ctrl._get_b0(obs, info)
        expected = 1.5 * 600.0 * np.tanh(600.0 / 200.0)
        assert (
            abs(b0_val - expected) < 1e-6
        ), f"Observation mode b0 should be {expected:.2f}, got {b0_val:.2f}"


# ======================================================================
# 6. TestControlOutput
# ======================================================================


class TestControlOutput:
    """Test step() output properties."""

    def test_action_shape_and_dtype(self):
        ctrl = HinfController(HinfConfig(b0=100.0))
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0
        obs = _make_obs(angle=0.1, rate=1.0, q=500.0)
        info = _make_info(angle=0.1, rate_deg=np.degrees(1.0), q=500.0)
        action = ctrl.step(obs, info)
        assert action.shape == (1,), f"Action shape should be (1,), got {action.shape}"
        assert (
            action.dtype == np.float32
        ), f"Action dtype should be float32, got {action.dtype}"

    def test_action_in_range(self):
        """Action should always be clipped to [-1, 1]."""
        # Use a very aggressive config to try to produce large actions
        cfg = HinfConfig(q_angle=10000.0, q_rate=1000.0, r_control=0.0001, b0=1.0)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0

        obs = _make_obs(angle=2.0, rate=50.0, q=500.0)
        info = _make_info(angle=2.0, rate_deg=np.degrees(50.0), q=500.0)
        action = ctrl.step(obs, info)
        assert -1.0 <= action[0] <= 1.0, f"Action should be in [-1,1], got {action[0]}"

    def test_zero_error_produces_near_zero_action(self):
        """With zero angle error and zero rate, action should be near zero."""
        cfg = HinfConfig(b0=100.0)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0

        obs = _make_obs(angle=0.0, rate=0.0, q=500.0)
        info = _make_info(angle=0.0, rate_deg=0.0, q=500.0)
        action = ctrl.step(obs, info)
        assert (
            abs(action[0]) < 0.01
        ), f"Zero error should produce near-zero action, got {action[0]:.6f}"

    def test_larger_error_produces_larger_action(self):
        """A larger angle error should produce a larger magnitude action."""
        cfg = HinfConfig(b0=100.0)

        # Small error
        ctrl_small = HinfController(cfg)
        ctrl_small.launch_detected = True
        ctrl_small.target_angle = 0.0
        obs_small = _make_obs(angle=0.01, rate=0.0, q=500.0)
        info_small = _make_info(angle=0.01, rate_deg=0.0, q=500.0)
        action_small = ctrl_small.step(obs_small, info_small)

        # Large error
        ctrl_large = HinfController(cfg)
        ctrl_large.launch_detected = True
        ctrl_large.target_angle = 0.0
        obs_large = _make_obs(angle=0.5, rate=0.0, q=500.0)
        info_large = _make_info(angle=0.5, rate_deg=0.0, q=500.0)
        action_large = ctrl_large.step(obs_large, info_large)

        assert abs(action_large[0]) > abs(action_small[0]), (
            f"Larger error should produce larger action: "
            f"small={abs(action_small[0]):.6f}, large={abs(action_large[0]):.6f}"
        )


# ======================================================================
# 7. TestLaunchDetection
# ======================================================================


class TestLaunchDetection:
    """Test launch detection in ground-truth and observation modes."""

    def test_pre_launch_returns_zero_ground_truth(self):
        """Before launch detection, ground-truth mode should return zero action."""
        cfg = HinfConfig(use_observations=False)
        ctrl = HinfController(cfg)
        obs = _make_obs(angle=0.1, rate=1.0, q=500.0)
        info = _make_info(angle=0.1, rate_deg=np.degrees(1.0), accel=10.0, q=500.0)

        action = ctrl.step(obs, info)
        assert action[0] == 0.0, f"Pre-launch action should be 0.0, got {action[0]}"
        assert ctrl.launch_detected is False

    def test_launch_triggers_on_high_accel(self):
        """Ground-truth mode should detect launch when accel > 20."""
        cfg = HinfConfig(use_observations=False)
        ctrl = HinfController(cfg)
        obs = _make_obs(angle=0.0, rate=0.0, q=500.0)

        # Below threshold
        info_low = _make_info(accel=15.0, q=500.0)
        ctrl.step(obs, info_low)
        assert ctrl.launch_detected is False

        # Above threshold
        info_high = _make_info(accel=25.0, q=500.0)
        ctrl.step(obs, info_high)
        assert ctrl.launch_detected is True

    def test_launch_sets_target_angle(self):
        """Launch detection should capture the current angle as target."""
        cfg = HinfConfig(use_observations=False)
        ctrl = HinfController(cfg)
        obs = _make_obs(angle=0.3, rate=0.0, q=500.0)
        info = _make_info(angle=0.3, rate_deg=0.0, accel=50.0, q=500.0)

        ctrl.step(obs, info)
        assert ctrl.launch_detected is True
        assert (
            abs(ctrl.target_angle - 0.3) < 1e-6
        ), f"Target angle should be 0.3, got {ctrl.target_angle}"

    def test_observation_mode_auto_detects_on_first_step(self):
        """In observation mode, launch should be detected on the first step."""
        cfg = HinfConfig(use_observations=True, b0=100.0, b0_per_pa=None)
        ctrl = HinfController(cfg)
        obs = _make_obs(angle=0.2, rate=1.0, q=500.0)
        info = {}

        action = ctrl.step(obs, info)
        assert ctrl.launch_detected is True
        assert abs(ctrl.target_angle - 0.2) < 1e-6


# ======================================================================
# 8. TestConvergence
# ======================================================================


class TestConvergence:
    """Test closed-loop convergence and disturbance rejection."""

    def test_step_response_converges(self):
        """Starting from an angle error, the controller should drive rate to near zero."""
        b0 = 100.0
        cfg = HinfConfig(b0=b0, b0_per_pa=None)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0

        dt = 0.01
        roll_angle = 0.3  # Initial angle offset
        roll_rate = 0.0

        obs = _make_obs(q=500.0)
        for step in range(500):
            info = _make_info(
                angle=roll_angle,
                rate_deg=np.degrees(roll_rate),
                accel=50.0,
                q=500.0,
            )
            action = ctrl.step(obs, info, dt)
            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert (
            final_rate < 5.0
        ), f"Step response should converge, final rate {final_rate:.1f} deg/s"

    def test_disturbance_rejection_bounded(self):
        """With a sinusoidal disturbance, the roll rate should remain bounded."""
        b0 = 100.0
        cfg = HinfConfig(b0=b0, b0_per_pa=None)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0

        dt = 0.01
        roll_angle = 0.0
        roll_rate = 0.0
        wind_amp = 10.0
        wind_dir = 1.0

        obs = _make_obs(q=500.0)
        max_rate = 0.0
        for step in range(500):
            d = wind_amp * np.sin(wind_dir - roll_angle)
            info = _make_info(
                angle=roll_angle,
                rate_deg=np.degrees(roll_rate),
                accel=50.0,
                q=500.0,
            )
            action = ctrl.step(obs, info, dt)
            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt
            max_rate = max(max_rate, abs(np.degrees(roll_rate)))

        assert (
            max_rate < 100.0
        ), f"Disturbance rejection: max rate {max_rate:.1f} deg/s should be bounded"

    def test_initial_rate_damped(self):
        """Starting with a non-zero rate and zero angle error, rate should decrease."""
        b0 = 100.0
        cfg = HinfConfig(b0=b0, b0_per_pa=None)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 0.0

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(30.0)  # Initial 30 deg/s spin

        obs = _make_obs(q=500.0)
        initial_rate = abs(np.degrees(roll_rate))
        for step in range(300):
            info = _make_info(
                angle=roll_angle,
                rate_deg=np.degrees(roll_rate),
                accel=50.0,
                q=500.0,
            )
            action = ctrl.step(obs, info, dt)
            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert final_rate < initial_rate * 0.3, (
            f"Rate should be damped: initial={initial_rate:.1f}, "
            f"final={final_rate:.1f} deg/s"
        )


# ======================================================================
# 9. TestInterface
# ======================================================================


class TestInterface:
    """Test the controller interface (reset, step, multi-episode)."""

    def test_reset_clears_state(self):
        cfg = HinfConfig(b0=100.0)
        ctrl = HinfController(cfg)
        ctrl.launch_detected = True
        ctrl.target_angle = 1.5
        ctrl._x_K = np.array([10.0, 20.0])

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert ctrl.target_angle == 0.0
        assert np.allclose(ctrl._x_K, [0.0, 0.0])

    def test_step_returns_correct_shape_after_reset(self):
        cfg = HinfConfig(b0=100.0)
        ctrl = HinfController(cfg)

        # First episode
        obs = _make_obs(angle=0.1, rate=1.0, q=500.0)
        info = _make_info(angle=0.1, rate_deg=np.degrees(1.0), accel=50.0, q=500.0)
        action1 = ctrl.step(obs, info)
        assert action1.shape == (1,)

        # Reset and second episode
        ctrl.reset()
        action2 = ctrl.step(obs, info)
        assert action2.shape == (1,)
        assert action2.dtype == np.float32

    def test_multi_episode_independent(self):
        """After reset, the controller should behave as freshly initialized."""
        cfg = HinfConfig(b0=100.0)

        # Run first episode and collect final state
        ctrl = HinfController(cfg)
        obs = _make_obs(angle=0.0, rate=0.0, q=500.0)
        info = _make_info(angle=0.5, rate_deg=30.0, accel=50.0, q=500.0)
        for _ in range(100):
            ctrl.step(obs, info)

        # Reset
        ctrl.reset()

        # Run same first step as a fresh controller
        ctrl_fresh = HinfController(cfg)
        info_launch = _make_info(angle=0.0, rate_deg=0.0, accel=50.0, q=500.0)
        action_reset = ctrl.step(obs, info_launch)
        action_fresh = ctrl_fresh.step(obs, info_launch)

        assert abs(action_reset[0] - action_fresh[0]) < 1e-6, (
            f"After reset, first action ({action_reset[0]:.6f}) should match "
            f"fresh controller ({action_fresh[0]:.6f})"
        )

    def test_has_required_properties(self):
        ctrl = HinfController(HinfConfig())
        assert hasattr(ctrl, "synthesis_succeeded")
        assert hasattr(ctrl, "lqr_gain")
        assert hasattr(ctrl, "kalman_gain")
        assert hasattr(ctrl, "cl_eigenvalues")
        assert isinstance(ctrl.synthesis_succeeded, bool)
        assert isinstance(ctrl.lqr_gain, np.ndarray)
        assert isinstance(ctrl.kalman_gain, np.ndarray)
        assert isinstance(ctrl.cl_eigenvalues, np.ndarray)
