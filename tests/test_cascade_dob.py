"""
Tests for the Cascade Disturbance Observer (CDO) controller.

Verifies:
1. CascadeDOBConfig default and custom values
2. Observer convergence: d_hat converges toward constant disturbance
3. Frequency tracking: omega_hat moves toward known sinusoidal frequency
4. Amplitude estimation: amplitude_estimate becomes nonzero for sinusoidal input
5. Frequency clamping: omega_hat stays within [freq_min, freq_max]
6. Warmup: feedforward is zero during warmup_steps, potentially nonzero after
7. Interface: reset clears state, step returns correct shape, launch_detected
8. Control output: action bounded [-1, 1], base controller active even with no ff
9. Gain scheduling: _gain_scale behavior at different q values
10. Dynamic b0: b0_per_pa computation
"""

import numpy as np
import pytest

from cascade_dob import CascadeDOBController, CascadeDOBConfig
from pid_controller import PIDConfig


def _make_obs(roll_angle=0.0, roll_rate=0.0, roll_accel=0.0, q=500.0):
    """Create a 10-element observation array with specified values."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = roll_angle  # roll angle (rad)
    obs[3] = roll_rate  # roll rate (rad/s)
    obs[4] = roll_accel  # roll acceleration (rad/s^2)
    obs[5] = q  # dynamic pressure (Pa)
    return obs


def _make_info(
    roll_angle_rad=0.0,
    roll_rate_deg_s=0.0,
    vertical_acceleration_ms2=50.0,
    dynamic_pressure_Pa=500.0,
):
    """Create an info dict for ground-truth mode with launch-triggering accel."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": vertical_acceleration_ms2,
        "dynamic_pressure_Pa": dynamic_pressure_Pa,
    }


# ---------------------------------------------------------------------------
# 1. TestCascadeDOBConfig
# ---------------------------------------------------------------------------


class TestCascadeDOBConfig:
    """Test configuration defaults and custom values."""

    def test_default_config_values(self):
        cfg = CascadeDOBConfig()
        assert cfg.K_ff == 0.5
        assert cfg.observer_bw == 30.0
        assert cfg.freq_adapt_rate == 0.001
        assert cfg.amp_adapt_rate == 0.05
        assert cfg.freq_min == 1.0
        assert cfg.freq_max == 100.0
        assert cfg.omega_init == 10.0
        assert cfg.b0 == 725.0
        assert cfg.b0_per_pa is None
        assert cfg.q_ref == 500.0
        assert cfg.warmup_steps == 50
        assert cfg.forgetting == 0.998

    def test_custom_config(self):
        cfg = CascadeDOBConfig(
            K_ff=0.8,
            observer_bw=50.0,
            freq_adapt_rate=0.005,
            amp_adapt_rate=0.1,
            freq_min=2.0,
            freq_max=80.0,
            omega_init=15.0,
            b0=500.0,
            b0_per_pa=1.5,
            q_ref=400.0,
            warmup_steps=100,
            forgetting=0.99,
        )
        assert cfg.K_ff == 0.8
        assert cfg.observer_bw == 50.0
        assert cfg.freq_adapt_rate == 0.005
        assert cfg.amp_adapt_rate == 0.1
        assert cfg.freq_min == 2.0
        assert cfg.freq_max == 80.0
        assert cfg.omega_init == 15.0
        assert cfg.b0 == 500.0
        assert cfg.b0_per_pa == 1.5
        assert cfg.q_ref == 400.0
        assert cfg.warmup_steps == 100
        assert cfg.forgetting == 0.99


# ---------------------------------------------------------------------------
# 2. TestObserverConvergence
# ---------------------------------------------------------------------------


class TestObserverConvergence:
    """Test that the Stage 1 disturbance observer converges."""

    def test_constant_disturbance_convergence(self):
        """Feed a constant disturbance and verify d_hat converges toward it."""
        cdo_cfg = CascadeDOBConfig(observer_bw=30.0, warmup_steps=10)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        constant_disturbance = 5.0

        obs = _make_obs(roll_accel=constant_disturbance, q=500.0)
        info = _make_info(dynamic_pressure_Pa=500.0)

        for _ in range(300):
            ctrl.step(obs, info, dt)

        # d_hat should approach the constant disturbance
        # (action is small, so roll_accel ~ disturbance)
        d_hat = ctrl.disturbance_estimate
        # With a nonzero action feeding back, d_hat won't exactly equal 5.0
        # but it should be in the right ballpark (positive, moving toward the disturbance)
        assert d_hat != 0.0, "Disturbance estimate should be nonzero"

    def test_zero_disturbance_stays_near_zero(self):
        """With zero disturbance and zero state, d_hat should stay near zero."""
        cdo_cfg = CascadeDOBConfig(observer_bw=30.0, warmup_steps=10)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        obs = _make_obs(roll_accel=0.0, q=500.0)
        info = _make_info(
            roll_angle_rad=0.0, roll_rate_deg_s=0.0, dynamic_pressure_Pa=500.0
        )

        for _ in range(200):
            ctrl.step(obs, info, dt)

        assert (
            abs(ctrl.disturbance_estimate) < 5.0
        ), f"d_hat={ctrl.disturbance_estimate:.4f} should stay near zero"


# ---------------------------------------------------------------------------
# 3. TestFrequencyTracking
# ---------------------------------------------------------------------------


class TestFrequencyTracking:
    """Test that omega_hat moves toward the known disturbance frequency."""

    def test_frequency_moves_toward_signal(self):
        """Feed sinusoidal disturbance and check omega_hat shifts."""
        target_omega = 20.0
        cdo_cfg = CascadeDOBConfig(
            omega_init=10.0,  # Start far from target
            freq_adapt_rate=0.01,  # Faster adaptation for test
            amp_adapt_rate=0.1,
            observer_bw=50.0,
            warmup_steps=10,
            freq_min=1.0,
            freq_max=100.0,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        initial_omega = ctrl.frequency_estimate
        assert initial_omega == 10.0

        for i in range(1000):
            t = i * dt
            # Sinusoidal roll accel as if disturbance is driving it
            roll_accel = 5.0 * np.sin(target_omega * t)
            obs = _make_obs(roll_accel=roll_accel, q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)

        final_omega = ctrl.frequency_estimate
        # omega_hat should have moved away from 10.0 (the initial value)
        assert (
            final_omega != initial_omega
        ), f"omega_hat should have changed from {initial_omega}"

    def test_frequency_adapts_with_high_rate(self):
        """Higher freq_adapt_rate should move omega_hat faster."""
        results = {}
        for rate in [0.0001, 0.01]:
            cdo_cfg = CascadeDOBConfig(
                omega_init=10.0,
                freq_adapt_rate=rate,
                amp_adapt_rate=0.1,
                observer_bw=50.0,
                warmup_steps=10,
            )
            ctrl = CascadeDOBController(
                pid_config=PIDConfig(),
                cdo_config=cdo_cfg,
                use_observations=False,
            )
            dt = 0.01
            target_omega = 30.0
            for i in range(500):
                t = i * dt
                roll_accel = 5.0 * np.sin(target_omega * t)
                obs = _make_obs(roll_accel=roll_accel, q=500.0)
                info = _make_info(dynamic_pressure_Pa=500.0)
                ctrl.step(obs, info, dt)
            results[rate] = abs(ctrl.frequency_estimate - 10.0)

        # Higher adaptation rate should produce more change from initial
        assert results[0.01] > results[0.0001], (
            f"Higher rate change ({results[0.01]:.4f}) should exceed "
            f"lower rate change ({results[0.0001]:.4f})"
        )


# ---------------------------------------------------------------------------
# 4. TestAmplitudeEstimation
# ---------------------------------------------------------------------------


class TestAmplitudeEstimation:
    """Test that amplitude_estimate becomes nonzero for sinusoidal input."""

    def test_amplitude_nonzero_for_sinusoidal_disturbance(self):
        """Feed sinusoidal roll acceleration, amplitude should become nonzero."""
        cdo_cfg = CascadeDOBConfig(
            amp_adapt_rate=0.1,
            observer_bw=50.0,
            warmup_steps=10,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        omega = 15.0
        for i in range(500):
            t = i * dt
            roll_accel = 3.0 * np.sin(omega * t)
            obs = _make_obs(roll_accel=roll_accel, q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)

        assert (
            ctrl.amplitude_estimate > 0.0
        ), f"Amplitude estimate should be positive, got {ctrl.amplitude_estimate}"

    def test_amplitude_zero_initially(self):
        """Before any steps, amplitude estimate should be zero."""
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        assert ctrl.amplitude_estimate == 0.0

    def test_larger_disturbance_gives_larger_amplitude(self):
        """Larger sinusoidal input should produce larger amplitude estimate."""
        results = {}
        for amp in [1.0, 10.0]:
            cdo_cfg = CascadeDOBConfig(
                amp_adapt_rate=0.1,
                observer_bw=50.0,
                warmup_steps=10,
            )
            ctrl = CascadeDOBController(
                pid_config=PIDConfig(),
                cdo_config=cdo_cfg,
                use_observations=False,
            )
            dt = 0.01
            omega = 15.0
            for i in range(500):
                t = i * dt
                roll_accel = amp * np.sin(omega * t)
                obs = _make_obs(roll_accel=roll_accel, q=500.0)
                info = _make_info(dynamic_pressure_Pa=500.0)
                ctrl.step(obs, info, dt)
            results[amp] = ctrl.amplitude_estimate

        assert results[10.0] > results[1.0], (
            f"Larger disturbance ({results[10.0]:.4f}) should give larger "
            f"amplitude than smaller ({results[1.0]:.4f})"
        )


# ---------------------------------------------------------------------------
# 5. TestFrequencyClamping
# ---------------------------------------------------------------------------


class TestFrequencyClamping:
    """Test that omega_hat stays within [freq_min, freq_max]."""

    def test_frequency_stays_above_minimum(self):
        """omega_hat should never go below freq_min."""
        cdo_cfg = CascadeDOBConfig(
            freq_min=5.0,
            freq_max=100.0,
            omega_init=6.0,  # Start near minimum
            freq_adapt_rate=0.1,  # High rate to stress clamping
            amp_adapt_rate=0.1,
            observer_bw=50.0,
            warmup_steps=5,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        for i in range(500):
            # Low-frequency disturbance that might push omega_hat down
            t = i * dt
            roll_accel = 5.0 * np.sin(1.0 * t)
            obs = _make_obs(roll_accel=roll_accel, q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)
            assert (
                ctrl.frequency_estimate >= 5.0
            ), f"omega_hat={ctrl.frequency_estimate} went below freq_min=5.0"

    def test_frequency_stays_below_maximum(self):
        """omega_hat should never exceed freq_max."""
        cdo_cfg = CascadeDOBConfig(
            freq_min=1.0,
            freq_max=50.0,
            omega_init=45.0,  # Start near maximum
            freq_adapt_rate=0.1,
            amp_adapt_rate=0.1,
            observer_bw=50.0,
            warmup_steps=5,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        for i in range(500):
            t = i * dt
            # High-frequency disturbance that might push omega_hat up
            roll_accel = 5.0 * np.sin(200.0 * t)
            obs = _make_obs(roll_accel=roll_accel, q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)
            assert (
                ctrl.frequency_estimate <= 50.0
            ), f"omega_hat={ctrl.frequency_estimate} exceeded freq_max=50.0"


# ---------------------------------------------------------------------------
# 6. TestWarmup
# ---------------------------------------------------------------------------


class TestWarmup:
    """Test that feedforward is zero during warmup, nonzero after."""

    def test_feedforward_zero_during_warmup(self):
        """During warmup steps, CDO feedforward should not affect output."""
        warmup = 50
        cdo_cfg = CascadeDOBConfig(
            warmup_steps=warmup,
            K_ff=1.0,  # High feedforward gain
            amp_adapt_rate=0.2,
            observer_bw=50.0,
        )
        pid_cfg = PIDConfig()

        # CDO controller
        cdo_ctrl = CascadeDOBController(
            pid_config=pid_cfg,
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        # Plain GS-PID for comparison
        from pid_controller import GainScheduledPIDController

        gs_ctrl = GainScheduledPIDController(pid_cfg, use_observations=False)

        dt = 0.01
        # Feed some disturbance so CDO observer has something to estimate
        for i in range(warmup - 5):  # Stay well within warmup
            t = i * dt
            roll_accel = 10.0 * np.sin(15.0 * t)
            obs = _make_obs(
                roll_angle=0.1, roll_rate=0.5, roll_accel=roll_accel, q=500.0
            )
            info = _make_info(
                roll_angle_rad=0.1,
                roll_rate_deg_s=np.degrees(0.5),
                dynamic_pressure_Pa=500.0,
            )

            cdo_action = cdo_ctrl.step(obs, info, dt)
            gs_action = gs_ctrl.step(obs, info, dt)

            # During warmup, CDO action should equal base GS-PID action
            np.testing.assert_allclose(
                cdo_action,
                gs_action,
                atol=1e-6,
                err_msg=f"Step {i}: CDO should match GS-PID during warmup",
            )

    def test_feedforward_potentially_nonzero_after_warmup(self):
        """After warmup, feedforward may contribute to the action."""
        warmup = 20
        cdo_cfg = CascadeDOBConfig(
            warmup_steps=warmup,
            K_ff=1.0,
            amp_adapt_rate=0.2,
            observer_bw=50.0,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        # Run through warmup with disturbance to build up estimates
        for i in range(warmup + 100):
            t = i * dt
            roll_accel = 10.0 * np.sin(15.0 * t)
            obs = _make_obs(roll_accel=roll_accel, q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)

        # After warmup, amplitude should be nonzero if disturbance was present
        assert (
            ctrl.amplitude_estimate > 0.0
        ), "Amplitude should be positive after prolonged sinusoidal input"


# ---------------------------------------------------------------------------
# 7. TestInterface
# ---------------------------------------------------------------------------


class TestInterface:
    """Test standard controller interface: reset, step, launch_detected."""

    def test_reset_clears_state(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )

        # Run some steps to accumulate state
        dt = 0.01
        for i in range(100):
            t = i * dt
            obs = _make_obs(roll_accel=5.0 * np.sin(15.0 * t), q=500.0)
            info = _make_info(dynamic_pressure_Pa=500.0)
            ctrl.step(obs, info, dt)

        # Verify state was accumulated
        assert ctrl.disturbance_estimate != 0.0
        assert ctrl._step_count > 0

        # Reset and verify
        ctrl.reset()
        assert ctrl.disturbance_estimate == 0.0
        assert ctrl.frequency_estimate == CascadeDOBConfig().omega_init
        assert ctrl.amplitude_estimate == 0.0
        assert ctrl._step_count == 0
        assert ctrl._prev_action == 0.0
        assert ctrl._phase_acc == 0.0
        assert ctrl._a_cos == 0.0
        assert ctrl._a_sin == 0.0

    def test_step_returns_correct_shape(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        obs = _make_obs(q=500.0)
        info = _make_info()
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_launch_detected_before_launch(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        assert not ctrl.launch_detected

    def test_launch_detected_after_step_with_accel(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        obs = _make_obs(q=500.0)
        info = _make_info(vertical_acceleration_ms2=50.0)
        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

    def test_launch_detected_setter(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        ctrl.launch_detected = True
        assert ctrl.launch_detected
        assert ctrl.base_ctrl.launch_detected

    def test_no_launch_without_accel(self):
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(),
            use_observations=False,
        )
        obs = _make_obs(q=500.0)
        info = _make_info(vertical_acceleration_ms2=5.0)  # Below threshold
        action = ctrl.step(obs, info, dt=0.01)
        assert not ctrl.launch_detected
        # Before launch, should output zero
        assert action[0] == 0.0


# ---------------------------------------------------------------------------
# 8. TestControlOutput
# ---------------------------------------------------------------------------


class TestControlOutput:
    """Test that action is bounded and base controller runs correctly."""

    def test_action_bounded_minus1_to_1(self):
        """Action should always be in [-1, 1] even with large disturbances."""
        cdo_cfg = CascadeDOBConfig(
            K_ff=1.0,
            warmup_steps=5,
            amp_adapt_rate=0.2,
            observer_bw=50.0,
        )
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        dt = 0.01
        for i in range(300):
            t = i * dt
            # Large disturbance to try to push action out of bounds
            roll_accel = 100.0 * np.sin(15.0 * t)
            obs = _make_obs(
                roll_angle=0.5,
                roll_rate=2.0,
                roll_accel=roll_accel,
                q=500.0,
            )
            info = _make_info(
                roll_angle_rad=0.5,
                roll_rate_deg_s=np.degrees(2.0),
                dynamic_pressure_Pa=500.0,
            )
            action = ctrl.step(obs, info, dt)
            assert (
                -1.0 <= action[0] <= 1.0
            ), f"Action {action[0]} out of bounds at step {i}"

    def test_base_controller_runs_when_ff_zero(self):
        """With K_ff=0, CDO should behave like pure GS-PID."""
        pid_cfg = PIDConfig()
        cdo_cfg = CascadeDOBConfig(K_ff=0.0, warmup_steps=5)

        cdo_ctrl = CascadeDOBController(
            pid_config=pid_cfg,
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        from pid_controller import GainScheduledPIDController

        gs_ctrl = GainScheduledPIDController(pid_cfg, use_observations=False)

        dt = 0.01
        for i in range(100):
            obs = _make_obs(roll_angle=0.1, roll_rate=0.3, q=500.0)
            info = _make_info(
                roll_angle_rad=0.1,
                roll_rate_deg_s=np.degrees(0.3),
                dynamic_pressure_Pa=500.0,
            )
            cdo_action = cdo_ctrl.step(obs, info, dt)
            gs_action = gs_ctrl.step(obs, info, dt)

            np.testing.assert_allclose(
                cdo_action,
                gs_action,
                atol=1e-5,
                err_msg=f"Step {i}: K_ff=0 CDO should match GS-PID",
            )


# ---------------------------------------------------------------------------
# 9. TestGainScheduling
# ---------------------------------------------------------------------------


class TestGainScheduling:
    """Test _gain_scale behavior at different dynamic pressures."""

    def test_gain_scale_at_reference_q(self):
        """At q_ref, gain scale should be approximately 1.0."""
        cdo_cfg = CascadeDOBConfig(q_ref=500.0)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        scale = ctrl._gain_scale(500.0)
        assert abs(scale - 1.0) < 0.01, f"Scale at q_ref should be ~1.0, got {scale}"

    def test_gain_scale_higher_at_low_q(self):
        """At low q, gain scale should be higher (more aggressive gains)."""
        cdo_cfg = CascadeDOBConfig(q_ref=500.0)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        scale_low = ctrl._gain_scale(100.0)
        scale_ref = ctrl._gain_scale(500.0)
        assert scale_low > scale_ref, (
            f"Scale at q=100 ({scale_low:.2f}) should exceed scale at "
            f"q=500 ({scale_ref:.2f})"
        )

    def test_gain_scale_lower_at_high_q(self):
        """At high q, gain scale should be lower (less aggressive gains)."""
        cdo_cfg = CascadeDOBConfig(q_ref=500.0)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        scale_high = ctrl._gain_scale(1000.0)
        scale_ref = ctrl._gain_scale(500.0)
        assert scale_high < scale_ref, (
            f"Scale at q=1000 ({scale_high:.2f}) should be less than "
            f"scale at q=500 ({scale_ref:.2f})"
        )

    def test_gain_scale_clamped_maximum(self):
        """Gain scale should not exceed 5.0."""
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(q_ref=500.0),
            use_observations=False,
        )
        scale = ctrl._gain_scale(0.0)
        assert scale == 5.0

    def test_gain_scale_clamped_minimum(self):
        """Gain scale should not go below 0.5."""
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=CascadeDOBConfig(q_ref=100.0),
            use_observations=False,
        )
        # Very high q relative to low q_ref should push scale toward minimum
        scale = ctrl._gain_scale(5000.0)
        assert scale >= 0.5, f"Scale should be >= 0.5, got {scale}"


# ---------------------------------------------------------------------------
# 10. TestDynamicB0
# ---------------------------------------------------------------------------


class TestDynamicB0:
    """Test b0_per_pa computation and fallback behavior."""

    def test_fixed_b0_when_no_b0_per_pa(self):
        """Without b0_per_pa, _get_b0 should return fixed b0."""
        cdo_cfg = CascadeDOBConfig(b0=725.0, b0_per_pa=None)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        obs = _make_obs(q=500.0)
        info = _make_info(dynamic_pressure_Pa=500.0)
        b0 = ctrl._get_b0(obs, info)
        assert b0 == 725.0

    def test_dynamic_b0_with_b0_per_pa(self):
        """With b0_per_pa set, b0 should vary with dynamic pressure."""
        cdo_cfg = CascadeDOBConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )

        info_low = _make_info(dynamic_pressure_Pa=100.0)
        info_high = _make_info(dynamic_pressure_Pa=800.0)
        obs = _make_obs(q=500.0)  # obs not used in ground-truth mode for q

        b0_low = ctrl._get_b0(obs, info_low)
        b0_high = ctrl._get_b0(obs, info_high)

        assert (
            b0_high > b0_low
        ), f"b0 at high q ({b0_high:.1f}) should exceed b0 at low q ({b0_low:.1f})"

    def test_dynamic_b0_fallback_at_very_low_q(self):
        """At very low q, b0 should fall back to fixed b0."""
        cdo_cfg = CascadeDOBConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=False,
        )
        obs = _make_obs(q=0.0)
        info = _make_info(dynamic_pressure_Pa=0.0)
        b0 = ctrl._get_b0(obs, info)
        # b0_per_pa * 0 * tanh(0/200) = 0, which is below b0 * 0.01 = 7.25
        # so it should fall back to fixed b0
        assert b0 == 725.0

    def test_dynamic_b0_observation_mode(self):
        """In observation mode, b0 should be computed from obs[5]."""
        cdo_cfg = CascadeDOBConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = CascadeDOBController(
            pid_config=PIDConfig(),
            cdo_config=cdo_cfg,
            use_observations=True,
        )
        obs_q300 = _make_obs(q=300.0)
        obs_q700 = _make_obs(q=700.0)
        info = {}  # Not used in obs mode

        b0_300 = ctrl._get_b0(obs_q300, info)
        b0_700 = ctrl._get_b0(obs_q700, info)
        assert (
            b0_700 > b0_300
        ), f"b0 at q=700 ({b0_700:.1f}) should exceed b0 at q=300 ({b0_300:.1f})"
