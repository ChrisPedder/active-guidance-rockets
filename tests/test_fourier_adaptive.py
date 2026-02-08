"""
Tests for the Fourier-domain adaptive disturbance model.

Verifies:
1. Feature vector construction (correct dimensions, frequency components)
2. RLS convergence on known signals
3. L1 regularization produces sparsity
4. Feedforward action opposes predicted disturbance
5. Warmup suppresses feedforward
6. Controller interface (reset/step)
7. Closed-loop stability with simple dynamics
8. Multi-frequency disturbance tracking
9. Integration with compare_controllers.py
"""

import numpy as np
import pytest

from fourier_adaptive import FourierAdaptiveADRC, FourierAdaptiveConfig
from adrc_controller import ADRCConfig


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


class TestFourierAdaptiveConfig:
    """Test configuration defaults."""

    def test_default_values(self):
        cfg = FourierAdaptiveConfig()
        assert cfg.n_gust_freqs == 4
        assert cfg.include_spin_harmonics == 2
        assert cfg.K_ff == 0.5
        assert cfg.rls_forgetting == 0.995
        assert cfg.l1_lambda == 0.001
        assert cfg.warmup_steps == 50
        assert cfg.predict_steps == 3
        assert cfg.omega_spin_smoothing == 0.9

    def test_custom_config(self):
        cfg = FourierAdaptiveConfig(n_gust_freqs=6, K_ff=0.8, l1_lambda=0.01)
        assert cfg.n_gust_freqs == 6
        assert cfg.K_ff == 0.8
        assert cfg.l1_lambda == 0.01


class TestFeatureVector:
    """Test feature vector construction."""

    def test_feature_dimension(self):
        """Feature vector should have correct length."""
        fc = FourierAdaptiveConfig(n_gust_freqs=4, include_spin_harmonics=2)
        ctrl = FourierAdaptiveADRC(fourier_config=fc)
        # 1 (DC) + 2*2 (spin harmonics) + 2*4 (gust freqs) = 13
        expected = 1 + 2 * 2 + 2 * 4
        assert ctrl._n_features == expected
        phi = ctrl._build_features(0.0, 10.0)
        assert len(phi) == expected

    def test_dc_component_always_one(self):
        """DC component (index 0) should always be 1.0."""
        ctrl = FourierAdaptiveADRC()
        for t in [0.0, 0.5, 1.0, 10.0]:
            phi = ctrl._build_features(t, 10.0)
            assert phi[0] == 1.0

    def test_features_at_time_zero(self):
        """At t=0, cos terms should be 1 and sin terms should be 0."""
        fc = FourierAdaptiveConfig(n_gust_freqs=2, include_spin_harmonics=1)
        ctrl = FourierAdaptiveADRC(fourier_config=fc)
        phi = ctrl._build_features(0.0, 10.0)
        # phi = [1, cos(0)=1, sin(0)=0, cos(0)=1, sin(0)=0, cos(0)=1, sin(0)=0]
        assert phi[0] == 1.0
        assert phi[1] == 1.0  # cos(omega*0)
        assert abs(phi[2]) < 1e-10  # sin(omega*0)

    def test_features_bounded(self):
        """All features should be in [-1, 1] (trig functions + DC=1)."""
        ctrl = FourierAdaptiveADRC()
        for _ in range(100):
            t = np.random.uniform(0, 10)
            omega = np.random.uniform(1, 100)
            phi = ctrl._build_features(t, omega)
            assert np.all(np.abs(phi) <= 1.0 + 1e-10)

    def test_zero_gust_freqs(self):
        """With no gust frequencies, only DC + spin harmonics."""
        fc = FourierAdaptiveConfig(n_gust_freqs=0, include_spin_harmonics=2)
        ctrl = FourierAdaptiveADRC(fourier_config=fc)
        # 1 (DC) + 2*2 (spin harmonics) = 5
        assert ctrl._n_features == 5

    def test_zero_spin_harmonics(self):
        """With no spin harmonics, only DC + gust freqs."""
        fc = FourierAdaptiveConfig(n_gust_freqs=3, include_spin_harmonics=0)
        ctrl = FourierAdaptiveADRC(fourier_config=fc)
        # 1 (DC) + 2*3 (gust freqs) = 7
        assert ctrl._n_features == 7


class TestRLSConvergence:
    """Test RLS weight estimation."""

    def test_rls_converges_to_dc_signal(self):
        """RLS should converge when target is a constant (DC only)."""
        fc = FourierAdaptiveConfig(n_gust_freqs=0, include_spin_harmonics=0)
        ctrl = FourierAdaptiveADRC(fourier_config=fc)
        # Only DC feature: n_features = 1
        dc_target = 5.0
        for _ in range(100):
            phi = np.array([1.0])
            ctrl._update_rls(phi, dc_target)

        assert (
            abs(ctrl._weights[0] - dc_target) < 0.1
        ), f"DC weight {ctrl._weights[0]:.3f} should converge to {dc_target}"

    def test_rls_converges_to_sinusoidal_signal(self):
        """RLS should identify the coefficients of a known sinusoidal signal."""
        fc = FourierAdaptiveConfig(
            n_gust_freqs=0,
            include_spin_harmonics=1,
            l1_lambda=0.0,
            rls_forgetting=0.995,
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        omega = 10.0
        a_true, b_true = 3.0, -2.0  # Target: 3*cos(omega*t) - 2*sin(omega*t)
        dt = 0.01
        for i in range(500):
            t = i * dt
            target = a_true * np.cos(omega * t) + b_true * np.sin(omega * t)
            phi = ctrl._build_features(t, omega)
            ctrl._update_rls(phi, target)

        # Weights should be [~0 (DC), ~3 (cos), ~-2 (sin)]
        assert abs(ctrl._weights[0]) < 0.5  # DC should be near zero
        assert (
            abs(ctrl._weights[1] - a_true) < 0.5
        ), f"cos weight {ctrl._weights[1]:.2f} should be near {a_true}"
        assert (
            abs(ctrl._weights[2] - b_true) < 0.5
        ), f"sin weight {ctrl._weights[2]:.2f} should be near {b_true}"

    def test_rls_prediction_error_decreases(self):
        """Prediction error should decrease as RLS adapts."""
        fc = FourierAdaptiveConfig(
            n_gust_freqs=2,
            include_spin_harmonics=1,
            l1_lambda=0.0,
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        omega = 10.0
        dt = 0.01
        errors_early = []
        errors_late = []

        for i in range(400):
            t = i * dt
            target = 5.0 * np.cos(omega * t) + 2.0 * np.sin(omega * t)
            phi = ctrl._build_features(t, omega)
            ctrl._update_rls(phi, target)
            if i < 50:
                errors_early.append(abs(ctrl._prediction_error))
            elif i >= 350:
                errors_late.append(abs(ctrl._prediction_error))

        early_mean = np.mean(errors_early)
        late_mean = np.mean(errors_late)
        assert (
            late_mean < early_mean * 0.5
        ), f"Late error ({late_mean:.3f}) should be < 50% of early error ({early_mean:.3f})"


class TestL1Regularization:
    """Test L1 regularization for sparsity."""

    def test_l1_drives_irrelevant_weights_to_zero(self):
        """With L1, weights for unused frequencies should be near zero."""
        fc = FourierAdaptiveConfig(
            n_gust_freqs=4,
            include_spin_harmonics=2,
            l1_lambda=0.01,
            rls_forgetting=0.995,
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        # Feed a pure DC signal — all frequency weights should get suppressed
        for _ in range(200):
            phi = ctrl._build_features(0.01 * _, 10.0)
            ctrl._update_rls(phi, 3.0)  # DC target

        # Non-DC weights should be near zero
        non_dc_weights = ctrl._weights[1:]
        max_non_dc = np.max(np.abs(non_dc_weights))
        assert (
            max_non_dc < 1.0
        ), f"Non-DC weights should be small with L1: max={max_non_dc:.4f}"

    def test_l1_preserves_dc_weight(self):
        """L1 should not regularize the DC term."""
        fc = FourierAdaptiveConfig(
            n_gust_freqs=2,
            include_spin_harmonics=1,
            l1_lambda=0.1,  # Strong L1
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        for i in range(200):
            phi = ctrl._build_features(i * 0.01, 10.0)
            ctrl._update_rls(phi, 5.0)  # DC target

        # DC weight should still be near 5.0
        assert (
            abs(ctrl._weights[0] - 5.0) < 2.0
        ), f"DC weight {ctrl._weights[0]:.2f} should be near 5.0 even with L1"

    def test_no_l1_preserves_all_weights(self):
        """Without L1, all weights can be non-zero."""
        fc = FourierAdaptiveConfig(
            n_gust_freqs=2,
            include_spin_harmonics=1,
            l1_lambda=0.0,
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        omega = 10.0
        for i in range(300):
            t = i * 0.01
            target = 2.0 + 3.0 * np.cos(omega * t) + 1.0 * np.sin(omega * t)
            phi = ctrl._build_features(t, omega)
            ctrl._update_rls(phi, target)

        # Multiple weights should be non-zero
        assert (
            ctrl.active_features >= 3
        ), f"Without L1, at least 3 features should be active, got {ctrl.active_features}"


class TestControllerInterface:
    """Test the full controller interface."""

    def test_has_required_methods(self):
        ctrl = FourierAdaptiveADRC()
        assert hasattr(ctrl, "reset") and callable(ctrl.reset)
        assert hasattr(ctrl, "step") and callable(ctrl.step)

    def test_reset_clears_state(self):
        ctrl = FourierAdaptiveADRC()
        ctrl.adrc.launch_detected = True
        ctrl._weights[0] = 10.0
        ctrl._step_count = 200
        ctrl._time = 5.0

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert np.all(ctrl._weights == 0.0)
        assert ctrl._step_count == 0
        assert ctrl._time == 0.0

    def test_step_returns_correct_shape(self):
        adrc_cfg = ADRCConfig(b0=100.0, b0_per_pa=None)
        ctrl = FourierAdaptiveADRC(adrc_cfg)
        ctrl.launch_detected = True
        obs = np.zeros(10)
        info = make_info(roll_rate_deg_s=10.0)
        action = ctrl.step(obs, info)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_action_clamped_to_range(self):
        adrc_cfg = ADRCConfig(omega_c=100.0, b0=1.0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=1.0, warmup_steps=0)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl.launch_detected = True
        ctrl._weights[:] = 1000.0  # Force large prediction
        ctrl._step_count = 100

        obs = np.zeros(10)
        info = make_info(roll_angle_rad=1.0, roll_rate_deg_s=500.0)
        action = ctrl.step(obs, info)
        assert -1.0 <= action[0] <= 1.0

    def test_launch_detection_ground_truth(self):
        ctrl = FourierAdaptiveADRC()
        obs = np.zeros(10)
        info = make_info(accel=10.0)  # Below threshold
        action = ctrl.step(obs, info)
        assert not ctrl.launch_detected
        assert action[0] == 0.0

        info = make_info(accel=50.0)  # Above threshold
        action = ctrl.step(obs, info)
        assert ctrl.launch_detected

    def test_launch_detection_obs_mode(self):
        adrc_cfg = ADRCConfig(b0=100.0, b0_per_pa=None, use_observations=True)
        ctrl = FourierAdaptiveADRC(adrc_cfg)
        obs = make_obs(roll_angle=0.1, roll_rate=0.5, q=500.0)
        ctrl.step(obs, {})
        assert ctrl.launch_detected

    def test_z_properties(self):
        ctrl = FourierAdaptiveADRC()
        ctrl.adrc.z1 = 1.0
        ctrl.adrc.z2 = 2.0
        ctrl.adrc.z3 = 3.0
        assert ctrl.z1 == 1.0
        assert ctrl.z2 == 2.0
        assert ctrl.z3 == 3.0


class TestWarmupBehavior:
    """Test warmup period suppresses feedforward."""

    def test_no_feedforward_during_warmup(self):
        adrc_cfg = ADRCConfig(b0=100.0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=1.0, warmup_steps=100)
        ctrl_ff = FourierAdaptiveADRC(adrc_cfg, fc)

        from adrc_controller import ADRCController

        ctrl_base = ADRCController(ADRCConfig(b0=100.0, b0_per_pa=None))

        obs = np.zeros(10)
        info = make_info(roll_rate_deg_s=30.0, accel=50.0)
        action_ff = ctrl_ff.step(obs, info)
        action_base = ctrl_base.step(obs, info)

        assert (
            abs(action_ff[0] - action_base[0]) < 1e-6
        ), f"During warmup, should match base ADRC: ff={action_ff[0]:.6f}, base={action_base[0]:.6f}"

    def test_feedforward_activates_after_warmup(self):
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=1.0, warmup_steps=10, l1_lambda=0.0)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = np.radians(20.0)
        wind_dir = 1.0
        wind_amp = 20.0

        obs = np.zeros(10)
        for step in range(50):
            d = wind_amp * np.sin(wind_dir - roll_angle)
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)
            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        # After warmup, weights should be non-zero
        assert np.any(
            np.abs(ctrl._weights) > 0.1
        ), f"After warmup with disturbance, weights should be non-zero"
        assert ctrl._ff_action != 0.0


class TestFeedforwardAction:
    """Test feedforward physics."""

    def test_feedforward_opposes_disturbance(self):
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=0.8, warmup_steps=0)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl.launch_detected = True
        ctrl._step_count = 100

        # Manually set weights so prediction is positive
        ctrl._weights[0] = 10.0  # DC = +10 rad/s^2

        obs = np.zeros(10)
        info = make_info(roll_rate_deg_s=0.0)
        ctrl.step(obs, info)

        # Feedforward should be negative (opposing positive disturbance)
        assert (
            ctrl._ff_action < 0
        ), f"Feedforward should oppose positive disturbance, got ff={ctrl._ff_action:.4f}"

    def test_K_ff_zero_disables_feedforward(self):
        b0 = 100.0
        adrc_cfg = ADRCConfig(b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=0.0, warmup_steps=0)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)

        from adrc_controller import ADRCController

        ctrl_base = ADRCController(ADRCConfig(b0=b0, b0_per_pa=None))

        ctrl._weights[:] = 100.0
        ctrl._step_count = 200

        obs = np.zeros(10)
        info = make_info(roll_angle_rad=0.1, roll_rate_deg_s=20.0, accel=50.0)

        action_ff = ctrl.step(obs, info)
        action_base = ctrl_base.step(obs, info)

        assert abs(action_ff[0] - action_base[0]) < 1e-6


class TestDynamicB0:
    """Test feedforward with gain-scheduled b0."""

    def test_feedforward_scales_with_q(self):
        b0_per_pa = 1.0
        ref_q = 500.0
        b0_fixed = b0_per_pa * ref_q * np.tanh(ref_q / 200.0)
        adrc_cfg = ADRCConfig(b0=b0_fixed, b0_per_pa=b0_per_pa)
        fc = FourierAdaptiveConfig(K_ff=0.5, warmup_steps=0)

        obs = np.zeros(10)

        # Low q
        ctrl_low = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl_low.launch_detected = True
        ctrl_low._weights[0] = 10.0
        ctrl_low._step_count = 100
        info_low = make_info(roll_angle_rad=0.0, q=200.0)
        action_low = ctrl_low.step(obs, info_low)

        # High q
        ctrl_high = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl_high.launch_detected = True
        ctrl_high._weights[0] = 10.0
        ctrl_high._step_count = 100
        info_high = make_info(roll_angle_rad=0.0, q=800.0)
        action_high = ctrl_high.step(obs, info_high)

        assert abs(action_low[0]) != abs(
            action_high[0]
        ), f"Actions should differ at different q: low={action_low[0]:.4f}, high={action_high[0]:.4f}"


class TestClosedLoopStability:
    """Test closed-loop stability."""

    def test_stable_without_disturbance(self):
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=0.5, warmup_steps=20)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
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
            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert (
            final_rate < 5.0
        ), f"Should settle without disturbance, got {final_rate:.1f} deg/s"

    def test_stable_with_sinusoidal_disturbance(self):
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(K_ff=0.5, warmup_steps=20, l1_lambda=0.0)
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = 0.0
        wind_amp = 15.0
        wind_dir = 1.0

        obs = np.zeros(10)
        max_rate = 0.0
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
            max_rate = max(max_rate, abs(np.degrees(roll_rate)))

        assert (
            max_rate < 100.0
        ), f"Controller should stay bounded, max rate {max_rate:.1f} deg/s"

    def test_stable_with_multi_frequency_disturbance(self):
        """Dual-frequency disturbance mimicking the wind model."""
        b0 = 100.0
        adrc_cfg = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, b0_per_pa=None)
        fc = FourierAdaptiveConfig(
            K_ff=0.5,
            warmup_steps=20,
            n_gust_freqs=4,
            l1_lambda=0.0,
        )
        ctrl = FourierAdaptiveADRC(adrc_cfg, fc)
        ctrl.launch_detected = True

        dt = 0.01
        roll_angle = 0.0
        roll_rate = 0.0
        gust_freq = 1.5  # Hz
        wind_amp = 10.0

        obs = np.zeros(10)
        max_rate = 0.0
        for step in range(500):
            t = step * dt
            # Dual-frequency gust like wind_model.py
            gust = wind_amp * (
                np.sin(2 * np.pi * gust_freq * t)
                + 0.5 * np.sin(2 * np.pi * gust_freq * 2.3 * t)
            )
            d = gust * np.sin(1.0 - roll_angle)
            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
            )
            action = ctrl.step(obs, info, dt)
            alpha = b0 * action[0] + d
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt
            max_rate = max(max_rate, abs(np.degrees(roll_rate)))

        assert (
            max_rate < 120.0
        ), f"Multi-freq disturbance: max rate {max_rate:.1f} deg/s too large"


class TestDiagnostics:
    """Test diagnostic properties."""

    def test_active_features_starts_zero(self):
        ctrl = FourierAdaptiveADRC()
        assert ctrl.active_features == 0

    def test_active_features_increases_with_training(self):
        fc = FourierAdaptiveConfig(
            n_gust_freqs=2, include_spin_harmonics=1, l1_lambda=0.0
        )
        ctrl = FourierAdaptiveADRC(fourier_config=fc)

        omega = 10.0
        for i in range(100):
            t = i * 0.01
            target = 2.0 + 3.0 * np.cos(omega * t)
            phi = ctrl._build_features(t, omega)
            ctrl._update_rls(phi, target)

        assert ctrl.active_features > 0

    def test_disturbance_estimate_property(self):
        ctrl = FourierAdaptiveADRC()
        ctrl._weights[0] = 5.0  # DC only
        est = ctrl.disturbance_estimate
        assert abs(est - 5.0) < 0.1


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_controller_importable(self):
        from fourier_adaptive import FourierAdaptiveADRC

        ctrl = FourierAdaptiveADRC()
        assert hasattr(ctrl, "step")
        assert hasattr(ctrl, "reset")
