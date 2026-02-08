#!/usr/bin/env python3
"""
Comprehensive tests for the FLL (Frequency-Locked Loop) controller.

Tests cover:
1. FLLConfig defaults and custom values
2. Frequency tracking initialization and bounds
3. Frequency Hz conversion
4. Oscillator normalization
5. Amplitude estimation
6. Warmup behavior
7. Interface (reset, step shape, launch_detected)
8. Control output bounds and zero-spin behavior
9. Gain scheduling
10. Dynamic b0 computation
11. Launch detection
12. Disturbance cancellation after warmup
"""

import numpy as np
import pytest

from fll_controller import FLLConfig, FLLController
from pid_controller import PIDConfig


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _make_obs(roll_angle_rad=0.0, roll_rate_rad_s=0.0, dynamic_pressure=500.0):
    """Create a 10-element observation array with key fields set."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = roll_angle_rad
    obs[3] = roll_rate_rad_s
    obs[5] = dynamic_pressure
    return obs


def _make_info(
    roll_angle_rad=0.0,
    roll_rate_deg_s=0.0,
    vertical_acceleration_ms2=50.0,
    dynamic_pressure_Pa=500.0,
):
    """Create a ground-truth info dict with standard fields."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": vertical_acceleration_ms2,
        "dynamic_pressure_Pa": dynamic_pressure_Pa,
    }


# ===========================================================================
# 1. TestFLLConfig
# ===========================================================================


class TestFLLConfig:
    """Tests for FLLConfig dataclass defaults and custom values."""

    def test_default_values(self):
        cfg = FLLConfig()
        assert cfg.K_ff == 0.5
        assert cfg.mu_freq == 0.0005
        assert cfg.mu_amp == 0.03
        assert cfg.omega_init == 10.0
        assert cfg.freq_min == 1.0
        assert cfg.freq_max == 100.0
        assert cfg.b0 == 725.0
        assert cfg.b0_per_pa is None
        assert cfg.q_ref == 500.0
        assert cfg.warmup_steps == 50
        assert cfg.error_filter_alpha == 0.1

    def test_custom_config(self):
        cfg = FLLConfig(
            K_ff=0.8,
            mu_freq=0.001,
            mu_amp=0.05,
            omega_init=20.0,
            freq_min=2.0,
            freq_max=200.0,
            b0=500.0,
            b0_per_pa=1.5,
            q_ref=600.0,
            warmup_steps=100,
            error_filter_alpha=0.2,
        )
        assert cfg.K_ff == 0.8
        assert cfg.mu_freq == 0.001
        assert cfg.mu_amp == 0.05
        assert cfg.omega_init == 20.0
        assert cfg.freq_min == 2.0
        assert cfg.freq_max == 200.0
        assert cfg.b0 == 500.0
        assert cfg.b0_per_pa == 1.5
        assert cfg.q_ref == 600.0
        assert cfg.warmup_steps == 100
        assert cfg.error_filter_alpha == 0.2


# ===========================================================================
# 2. TestFrequencyTracking
# ===========================================================================


class TestFrequencyTracking:
    """Tests for frequency initialization and bounds enforcement."""

    def test_initial_frequency_matches_omega_init(self):
        fll_cfg = FLLConfig(omega_init=15.0)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        assert ctrl.frequency_estimate == 15.0

    def test_initial_frequency_default(self):
        ctrl = FLLController(use_observations=False)
        assert ctrl.frequency_estimate == 10.0  # default omega_init

    def test_frequency_stays_within_bounds(self):
        """After many steps the frequency must remain in [freq_min, freq_max]."""
        fll_cfg = FLLConfig(
            omega_init=50.0,
            freq_min=5.0,
            freq_max=80.0,
            mu_freq=0.01,  # aggressive adaptation
            warmup_steps=0,
        )
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)

        obs = _make_obs()
        info = _make_info()

        # Drive with large varying roll rates to push frequency adaptation
        for i in range(500):
            rate_deg = 100.0 * np.sin(2.0 * np.pi * 7.0 * i * 0.01)
            info_i = _make_info(roll_rate_deg_s=rate_deg)
            ctrl.step(obs, info_i, dt=0.01)
            assert fll_cfg.freq_min <= ctrl.frequency_estimate <= fll_cfg.freq_max, (
                f"Frequency {ctrl.frequency_estimate} outside "
                f"[{fll_cfg.freq_min}, {fll_cfg.freq_max}] at step {i}"
            )

    def test_frequency_changes_after_reset(self):
        """Reset should restore initial frequency."""
        ctrl = FLLController(
            fll_config=FLLConfig(omega_init=10.0),
            use_observations=False,
        )
        obs = _make_obs()
        info = _make_info(roll_rate_deg_s=30.0)

        for _ in range(100):
            ctrl.step(obs, info, dt=0.01)

        # Frequency may have drifted from 10.0
        ctrl.reset()
        assert ctrl.frequency_estimate == 10.0


# ===========================================================================
# 3. TestFrequencyHz
# ===========================================================================


class TestFrequencyHz:
    """Tests for the frequency_estimate_hz property."""

    def test_frequency_hz_conversion(self):
        fll_cfg = FLLConfig(omega_init=2 * np.pi * 5.0)  # 5 Hz
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        assert abs(ctrl.frequency_estimate_hz - 5.0) < 1e-10

    def test_frequency_hz_matches_rads(self):
        ctrl = FLLController(
            fll_config=FLLConfig(omega_init=20.0),
            use_observations=False,
        )
        expected_hz = 20.0 / (2 * np.pi)
        assert abs(ctrl.frequency_estimate_hz - expected_hz) < 1e-10


# ===========================================================================
# 4. TestOscillatorNormalization
# ===========================================================================


class TestOscillatorNormalization:
    """Tests that the internal oscillator (x1, x2) stays at unit norm."""

    def test_oscillator_unit_norm_after_many_steps(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info(roll_rate_deg_s=10.0)

        for i in range(500):
            ctrl.step(obs, info, dt=0.01)
            norm = np.sqrt(ctrl._x1**2 + ctrl._x2**2)
            assert abs(norm - 1.0) < 1e-6, f"Oscillator norm {norm} != 1.0 at step {i}"

    def test_oscillator_unit_norm_with_varying_rate(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()

        for i in range(300):
            rate = 50.0 * np.sin(0.1 * i)
            info = _make_info(roll_rate_deg_s=rate)
            ctrl.step(obs, info, dt=0.01)
            norm = np.sqrt(ctrl._x1**2 + ctrl._x2**2)
            assert abs(norm - 1.0) < 1e-6


# ===========================================================================
# 5. TestAmplitudeEstimate
# ===========================================================================


class TestAmplitudeEstimate:
    """Tests for the amplitude_estimate property."""

    def test_amplitude_starts_at_zero(self):
        ctrl = FLLController(use_observations=False)
        assert ctrl.amplitude_estimate == 0.0

    def test_amplitude_is_non_negative(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()

        for i in range(200):
            rate = 20.0 * np.sin(0.3 * i)
            info = _make_info(roll_rate_deg_s=rate)
            ctrl.step(obs, info, dt=0.01)
            assert (
                ctrl.amplitude_estimate >= 0.0
            ), f"Amplitude {ctrl.amplitude_estimate} < 0 at step {i}"

    def test_amplitude_resets_to_zero(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info(roll_rate_deg_s=30.0)

        for _ in range(100):
            ctrl.step(obs, info, dt=0.01)

        ctrl.reset()
        assert ctrl.amplitude_estimate == 0.0


# ===========================================================================
# 6. TestWarmup
# ===========================================================================


class TestWarmup:
    """Tests that feedforward is zero during warmup."""

    def test_output_matches_base_during_warmup(self):
        """During warmup, FLL output should equal the base GS-PID output."""
        pid_cfg = PIDConfig()
        fll_cfg = FLLConfig(warmup_steps=50)
        ctrl = FLLController(pid_cfg, fll_cfg, use_observations=False)

        # Create an independent GS-PID for comparison
        from pid_controller import GainScheduledPIDController

        base_ctrl = GainScheduledPIDController(pid_cfg, use_observations=False)

        obs = _make_obs()

        # Run through warmup period with nonzero roll rate
        for i in range(50):
            rate = 10.0 * np.sin(0.5 * i)
            info = _make_info(roll_rate_deg_s=rate)
            fll_action = ctrl.step(obs, info, dt=0.01)
            base_action = base_ctrl.step(obs, info, dt=0.01)
            np.testing.assert_allclose(
                fll_action,
                base_action,
                atol=1e-5,
                err_msg=f"FLL and base differ during warmup at step {i}",
            )

    def test_feedforward_nonzero_after_warmup(self):
        """After warmup, with a sustained disturbance, FLL should differ from base."""
        pid_cfg = PIDConfig()
        fll_cfg = FLLConfig(
            warmup_steps=20,
            mu_amp=0.1,  # fast amplitude adaptation
            K_ff=1.0,
        )
        ctrl = FLLController(pid_cfg, fll_cfg, use_observations=False)

        from pid_controller import GainScheduledPIDController

        base_ctrl = GainScheduledPIDController(pid_cfg, use_observations=False)

        obs = _make_obs()

        # Run through warmup
        for i in range(20):
            rate = 30.0 * np.sin(10.0 * i * 0.01)
            info = _make_info(roll_rate_deg_s=rate)
            ctrl.step(obs, info, dt=0.01)
            base_ctrl.step(obs, info, dt=0.01)

        # Continue past warmup with strong sinusoidal disturbance
        found_difference = False
        for i in range(200):
            rate = 30.0 * np.sin(10.0 * (i + 20) * 0.01)
            info = _make_info(roll_rate_deg_s=rate)
            fll_action = ctrl.step(obs, info, dt=0.01)
            base_action = base_ctrl.step(obs, info, dt=0.01)
            if abs(fll_action[0] - base_action[0]) > 1e-4:
                found_difference = True
                break

        assert found_difference, "FLL output never diverged from base after warmup"


# ===========================================================================
# 7. TestInterface
# ===========================================================================


class TestInterface:
    """Tests for reset, step return shape, and launch_detected property."""

    def test_reset_clears_state(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info(roll_rate_deg_s=20.0)

        for _ in range(100):
            ctrl.step(obs, info, dt=0.01)

        ctrl.reset()

        # After reset, all internal state should be at initial values
        assert ctrl.frequency_estimate == ctrl.config.omega_init
        assert ctrl.amplitude_estimate == 0.0
        assert ctrl._step_count == 0
        assert ctrl._error_filtered == 0.0
        assert ctrl._error_prev == 0.0
        assert ctrl._x1 == 1.0
        assert ctrl._x2 == 0.0
        assert not ctrl.launch_detected

    def test_step_returns_correct_shape(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info()

        action = ctrl.step(obs, info, dt=0.01)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_step_returns_ndarray_of_length_1(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info()

        action = ctrl.step(obs, info, dt=0.01)
        assert len(action) == 1

    def test_launch_detected_property(self):
        ctrl = FLLController(use_observations=False)
        # Before any steps with sufficient acceleration, not launched
        assert not ctrl.launch_detected

        # Step with low acceleration -- should not detect launch
        obs = _make_obs()
        info = _make_info(vertical_acceleration_ms2=5.0)
        ctrl.step(obs, info, dt=0.01)
        assert not ctrl.launch_detected

        # Step with high acceleration -- should detect launch
        info = _make_info(vertical_acceleration_ms2=50.0)
        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

    def test_launch_detected_setter(self):
        ctrl = FLLController(use_observations=False)
        ctrl.launch_detected = True
        assert ctrl.launch_detected
        assert ctrl.base_ctrl.launch_detected


# ===========================================================================
# 8. TestControlOutput
# ===========================================================================


class TestControlOutput:
    """Tests for action bounds and zero-spin behavior."""

    def test_action_bounded(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()

        # Try extreme roll rates to push the controller hard
        for rate in [-500, -100, -10, 0, 10, 100, 500]:
            ctrl.reset()
            info = _make_info(roll_rate_deg_s=float(rate))
            # Run enough steps to get past warmup
            for _ in range(60):
                action = ctrl.step(obs, info, dt=0.01)
                assert (
                    -1.0 <= action[0] <= 1.0
                ), f"Action {action[0]} out of [-1, 1] for rate={rate}"

    def test_zero_spin_small_action(self):
        """With zero spin rate and zero angle, action should be near zero."""
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()
        info = _make_info(roll_rate_deg_s=0.0, roll_angle_rad=0.0)

        # First step triggers launch, second computes PID from zero error
        ctrl.step(obs, info, dt=0.01)
        action = ctrl.step(obs, info, dt=0.01)
        assert (
            abs(action[0]) < 0.05
        ), f"Action {action[0]} too large for zero spin/zero angle"


# ===========================================================================
# 9. TestGainScheduling
# ===========================================================================


class TestGainScheduling:
    """Tests for the _gain_scale method."""

    def test_gain_scale_at_reference_q(self):
        fll_cfg = FLLConfig(q_ref=500.0)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        scale = ctrl._gain_scale(500.0)
        assert (
            abs(scale - 1.0) < 1e-6
        ), f"Gain scale at q_ref should be ~1.0, got {scale}"

    def test_gain_scale_at_low_q(self):
        ctrl = FLLController(use_observations=False)
        scale = ctrl._gain_scale(0.0)
        assert scale == 5.0, "Scale at q=0 should be clamped to 5.0"

    def test_gain_scale_at_very_low_q(self):
        ctrl = FLLController(use_observations=False)
        scale = ctrl._gain_scale(1e-5)
        assert scale == 5.0, "Scale at near-zero q should be 5.0"

    def test_gain_scale_increases_at_lower_q(self):
        ctrl = FLLController(fll_config=FLLConfig(q_ref=500.0), use_observations=False)
        scale_low = ctrl._gain_scale(100.0)
        scale_ref = ctrl._gain_scale(500.0)
        assert (
            scale_low > scale_ref
        ), f"Scale at q=100 ({scale_low}) should be > scale at q=500 ({scale_ref})"

    def test_gain_scale_clamped_high(self):
        ctrl = FLLController(use_observations=False)
        # Very low q but above the 1e-3 threshold
        scale = ctrl._gain_scale(0.1)
        assert scale <= 5.0


# ===========================================================================
# 10. TestDynamicB0
# ===========================================================================


class TestDynamicB0:
    """Tests for dynamic b0 computation via b0_per_pa."""

    def test_fixed_b0_when_b0_per_pa_is_none(self):
        fll_cfg = FLLConfig(b0=725.0, b0_per_pa=None)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        obs = _make_obs()
        info = _make_info(dynamic_pressure_Pa=300.0)
        b0 = ctrl._get_b0(obs, info)
        assert b0 == 725.0

    def test_dynamic_b0_with_b0_per_pa(self):
        fll_cfg = FLLConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        obs = _make_obs(dynamic_pressure=400.0)
        info = _make_info(dynamic_pressure_Pa=400.0)

        b0 = ctrl._get_b0(obs, info)
        q = 400.0
        expected = 1.5 * q * np.tanh(q / 200.0)
        assert abs(b0 - expected) < 1e-6

    def test_dynamic_b0_fallback_at_low_q(self):
        """When q is very low, b0_now < b0 * 0.01, should fall back to fixed b0."""
        fll_cfg = FLLConfig(b0=725.0, b0_per_pa=1.5)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        obs = _make_obs(dynamic_pressure=0.0)
        info = _make_info(dynamic_pressure_Pa=0.0)

        b0 = ctrl._get_b0(obs, info)
        assert (
            b0 == 725.0
        ), f"At q=0 with b0_per_pa set, should fall back to fixed b0, got {b0}"

    def test_dynamic_b0_obs_mode(self):
        """In observation mode, b0_per_pa should use obs[5] for q."""
        fll_cfg = FLLConfig(b0=725.0, b0_per_pa=2.0)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=True)

        q = 600.0
        obs = _make_obs(dynamic_pressure=q)
        info = {}

        b0 = ctrl._get_b0(obs, info)
        expected = 2.0 * q * np.tanh(q / 200.0)
        assert abs(b0 - expected) < 1e-6


# ===========================================================================
# 11. TestLaunchDetection
# ===========================================================================


class TestLaunchDetection:
    """Tests for launch detection in ground-truth and observation modes."""

    def test_ground_truth_needs_high_accel(self):
        """In ground-truth mode, launch requires accel > 20 m/s^2."""
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()

        # Low acceleration: no launch
        info = _make_info(vertical_acceleration_ms2=10.0)
        action = ctrl.step(obs, info, dt=0.01)
        assert not ctrl.launch_detected
        # Before launch, output should be zero
        assert action[0] == 0.0

        # High acceleration: launch detected
        info = _make_info(vertical_acceleration_ms2=50.0)
        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

    def test_obs_mode_auto_detects(self):
        """In observation mode, launch is auto-detected on first step."""
        ctrl = FLLController(use_observations=True)
        obs = _make_obs()
        info = {}

        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

    def test_launch_persists_after_detection(self):
        ctrl = FLLController(use_observations=False)
        obs = _make_obs()

        # Trigger launch
        info = _make_info(vertical_acceleration_ms2=50.0)
        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

        # Subsequent step with low acceleration should still be launched
        info = _make_info(vertical_acceleration_ms2=0.0)
        ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected


# ===========================================================================
# 12. TestDisturbanceCancellation
# ===========================================================================


class TestDisturbanceCancellation:
    """Tests that with a sustained sinusoidal disturbance, the FLL produces
    feedforward action after warmup."""

    def test_feedforward_active_with_sinusoidal_disturbance(self):
        """After warmup, a sinusoidal roll rate should cause measurable
        feedforward contribution."""
        fll_cfg = FLLConfig(
            warmup_steps=30,
            mu_amp=0.05,
            mu_freq=0.001,
            K_ff=0.8,
            omega_init=10.0,
        )
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        obs = _make_obs()

        dt = 0.01
        freq_hz = 5.0
        omega = 2.0 * np.pi * freq_hz

        # Run the controller with a sinusoidal roll rate for several cycles
        actions = []
        for i in range(300):
            t = i * dt
            rate = 20.0 * np.sin(omega * t)
            info = _make_info(roll_rate_deg_s=rate)
            action = ctrl.step(obs, info, dt=dt)
            actions.append(action[0])

        # After warmup (step 30+), the amplitude should have grown
        assert (
            ctrl.amplitude_estimate > 0.0
        ), "Amplitude estimate should be positive after sinusoidal input"

        # The actions after warmup should have some variation (feedforward active)
        post_warmup_actions = actions[50:]
        action_range = max(post_warmup_actions) - min(post_warmup_actions)
        assert action_range > 0.01, (
            f"Post-warmup action range {action_range} too small; "
            "feedforward may not be active"
        )

    def test_amplitude_grows_with_disturbance(self):
        """The amplitude estimate should increase when exposed to a
        sustained sinusoidal signal."""
        fll_cfg = FLLConfig(mu_amp=0.05, warmup_steps=10)
        ctrl = FLLController(fll_config=fll_cfg, use_observations=False)
        obs = _make_obs()

        dt = 0.01
        omega = 15.0  # rad/s

        amp_history = []
        for i in range(200):
            t = i * dt
            rate = 25.0 * np.sin(omega * t)
            info = _make_info(roll_rate_deg_s=rate)
            ctrl.step(obs, info, dt=dt)
            amp_history.append(ctrl.amplitude_estimate)

        # Amplitude should be larger at end than at beginning
        assert amp_history[-1] > amp_history[10], (
            f"Amplitude at end ({amp_history[-1]}) should exceed "
            f"amplitude near start ({amp_history[10]})"
        )
