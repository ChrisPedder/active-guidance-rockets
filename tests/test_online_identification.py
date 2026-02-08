"""
Tests for the online RLS b0 identification module.

Verifies:
1. RLS converges to the correct b0 with known plant
2. Clamping prevents divergence
3. Persistent excitation guard skips low-action updates
4. Forgetting factor allows tracking of time-varying b0
5. Integration with ADRC controller
6. Reset clears state properly
"""

import numpy as np
import pytest

from online_identification import B0Estimator, B0EstimatorConfig


class TestB0EstimatorBasic:
    """Test basic B0Estimator behavior."""

    def test_initial_b0_matches_config(self):
        cfg = B0EstimatorConfig(b0_init=200.0)
        est = B0Estimator(cfg)
        assert est.b0_hat == 200.0

    def test_initial_bias_is_zero(self):
        est = B0Estimator()
        assert est.c_hat == 0.0

    def test_reset_restores_initial_state(self):
        est = B0Estimator(B0EstimatorConfig(b0_init=100.0))
        # Perturb state via updates with data implying b0=500
        for _ in range(100):
            est.update(roll_accel=500.0 * 0.5, action=0.5)
        assert (
            abs(est.b0_hat - 100.0) > 10.0
        ), f"b0_hat should have moved from init, got {est.b0_hat}"
        # Reset should restore
        est.reset()
        assert est.b0_hat == 100.0
        assert est.c_hat == 0.0
        assert est.n_updates == 0

    def test_update_count_increments(self):
        est = B0Estimator()
        assert est.n_updates == 0
        est.update(roll_accel=10.0, action=0.5)
        assert est.n_updates == 1
        est.update(roll_accel=10.0, action=0.5)
        assert est.n_updates == 2


class TestRLSConvergence:
    """Test that RLS converges to the correct b0."""

    def test_converges_to_true_b0_no_noise(self):
        """With noiseless data from a known plant, RLS should converge exactly."""
        true_b0 = 150.0
        true_c = 5.0

        cfg = B0EstimatorConfig(b0_init=100.0, forgetting=0.99, p_init=1000.0)
        est = B0Estimator(cfg)

        # Generate data with known b0
        np.random.seed(42)
        for _ in range(200):
            action = np.random.uniform(-1.0, 1.0)
            if abs(action) < 0.05:
                action = 0.1  # Ensure excitation
            roll_accel = true_b0 * action + true_c
            est.update(roll_accel, action)

        assert (
            abs(est.b0_hat - true_b0) < 1.0
        ), f"b0_hat={est.b0_hat:.1f} should be close to true b0={true_b0}"
        assert (
            abs(est.c_hat - true_c) < 1.0
        ), f"c_hat={est.c_hat:.1f} should be close to true c={true_c}"

    def test_converges_with_noise(self):
        """With noisy data, RLS should still converge reasonably."""
        true_b0 = 130.0
        true_c = 0.0
        noise_std = 5.0

        cfg = B0EstimatorConfig(b0_init=130.0, forgetting=0.99)
        est = B0Estimator(cfg)

        np.random.seed(42)
        for _ in range(500):
            action = np.random.uniform(-1.0, 1.0)
            if abs(action) < 0.05:
                action = 0.1
            noise = np.random.normal(0, noise_std)
            roll_accel = true_b0 * action + true_c + noise
            est.update(roll_accel, action)

        # Should be within 10% of true value
        assert (
            abs(est.b0_hat - true_b0) / true_b0 < 0.1
        ), f"b0_hat={est.b0_hat:.1f}, true={true_b0}"

    def test_converges_from_wrong_initial(self):
        """Should converge even when initial guess is very wrong."""
        true_b0 = 150.0

        cfg = B0EstimatorConfig(b0_init=50.0, forgetting=0.99, p_init=5000.0)
        est = B0Estimator(cfg)

        np.random.seed(42)
        for _ in range(300):
            action = np.random.uniform(-1.0, 1.0)
            if abs(action) < 0.05:
                action = 0.1
            roll_accel = true_b0 * action
            est.update(roll_accel, action)

        assert abs(est.b0_hat - true_b0) < 5.0, f"b0_hat={est.b0_hat:.1f}"


class TestClamping:
    """Test b0 clamping behavior."""

    def test_clamp_prevents_extreme_values(self):
        cfg = B0EstimatorConfig(b0_init=100.0, clamp_ratio=10.0)
        est = B0Estimator(cfg)

        # Feed data that would drive b0 to extreme values
        for _ in range(200):
            est.update(roll_accel=100000.0, action=0.1)

        assert est.b0_hat <= 100.0 * 10.0
        assert est.b0_hat >= 100.0 / 10.0

    def test_clamp_prevents_negative_b0(self):
        cfg = B0EstimatorConfig(b0_init=100.0, clamp_ratio=10.0)
        est = B0Estimator(cfg)

        # Feed contradictory data (negative b0)
        for _ in range(200):
            est.update(roll_accel=-1000.0, action=1.0)

        assert est.b0_hat >= 100.0 / 10.0


class TestPersistentExcitation:
    """Test the persistent excitation guard."""

    def test_small_action_skips_update(self):
        cfg = B0EstimatorConfig(excitation_threshold=0.05)
        est = B0Estimator(cfg)

        initial_b0 = est.b0_hat
        est.update(roll_accel=999.0, action=0.01)
        assert est.b0_hat == initial_b0
        assert est.n_updates == 0

    def test_large_action_updates(self):
        cfg = B0EstimatorConfig(excitation_threshold=0.05)
        est = B0Estimator(cfg)

        initial_b0 = est.b0_hat
        est.update(roll_accel=999.0, action=0.5)
        # b0 should change
        assert est.n_updates == 1

    def test_threshold_exactly_at_boundary(self):
        """Action exactly at threshold should be skipped."""
        cfg = B0EstimatorConfig(excitation_threshold=0.05)
        est = B0Estimator(cfg)
        est.update(roll_accel=100.0, action=0.05)
        assert est.n_updates == 0

    def test_threshold_just_above(self):
        """Action just above threshold should update."""
        cfg = B0EstimatorConfig(excitation_threshold=0.05)
        est = B0Estimator(cfg)
        est.update(roll_accel=100.0, action=0.051)
        assert est.n_updates == 1


class TestForgettingFactor:
    """Test that forgetting factor allows tracking time-varying b0."""

    def test_tracks_step_change(self):
        """B0 estimate should track a step change in true b0."""
        cfg = B0EstimatorConfig(b0_init=100.0, forgetting=0.98, p_init=1000.0)
        est = B0Estimator(cfg)

        np.random.seed(42)

        # Phase 1: true b0 = 100
        for _ in range(200):
            action = np.random.uniform(-1.0, 1.0)
            if abs(action) < 0.05:
                action = 0.1
            est.update(100.0 * action, action)

        b0_phase1 = est.b0_hat
        assert abs(b0_phase1 - 100.0) < 5.0

        # Phase 2: true b0 jumps to 200
        for _ in range(200):
            action = np.random.uniform(-1.0, 1.0)
            if abs(action) < 0.05:
                action = 0.1
            est.update(200.0 * action, action)

        b0_phase2 = est.b0_hat
        assert (
            abs(b0_phase2 - 200.0) < 10.0
        ), f"b0_hat={b0_phase2:.1f} should track new b0=200"

    def test_higher_forgetting_tracks_faster(self):
        """Lower forgetting factor should track changes faster."""
        results = {}
        for lam in [0.95, 0.99]:
            cfg = B0EstimatorConfig(b0_init=100.0, forgetting=lam, p_init=1000.0)
            est = B0Estimator(cfg)

            np.random.seed(42)
            # Train on b0=100
            for _ in range(100):
                a = np.random.uniform(0.2, 1.0)
                est.update(100.0 * a, a)
            # Switch to b0=200
            for _ in range(50):
                a = np.random.uniform(0.2, 1.0)
                est.update(200.0 * a, a)

            results[lam] = est.b0_hat

        # Lower forgetting (0.95) should be closer to 200 after 50 steps
        assert results[0.95] > results[0.99], (
            f"lam=0.95 ({results[0.95]:.1f}) should track faster than "
            f"lam=0.99 ({results[0.99]:.1f})"
        )


class TestADRCIntegration:
    """Test integration with ADRC controller."""

    def test_adrc_accepts_b0_estimator(self):
        from adrc_controller import ADRCController, ADRCConfig

        est = B0Estimator(B0EstimatorConfig(b0_init=130.0))
        ctrl = ADRCController(ADRCConfig(b0=130.0), b0_estimator=est)
        assert ctrl.b0_estimator is est

    def test_adrc_reset_resets_estimator(self):
        from adrc_controller import ADRCController, ADRCConfig

        est = B0Estimator(B0EstimatorConfig(b0_init=130.0))
        ctrl = ADRCController(ADRCConfig(b0=130.0), b0_estimator=est)

        # Perturb estimator
        est.update(1000.0, 0.5)
        assert est.n_updates == 1

        ctrl.reset()
        assert est.n_updates == 0
        assert est.b0_hat == 130.0

    def test_adrc_with_rls_produces_valid_action(self):
        from adrc_controller import ADRCController, ADRCConfig

        est = B0Estimator(B0EstimatorConfig(b0_init=130.0))
        ctrl = ADRCController(ADRCConfig(b0=130.0), b0_estimator=est)

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1  # roll angle
        obs[3] = 0.5  # roll rate
        obs[4] = 10.0  # roll accel
        obs[5] = 500.0  # dynamic pressure
        obs[8] = 0.3  # last action
        ctrl.config.use_observations = True

        action = ctrl.step(obs, {}, dt=0.01)
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_adrc_without_estimator_unchanged(self):
        """ADRC without b0_estimator should behave exactly as before."""
        from adrc_controller import ADRCController, ADRCConfig

        ctrl = ADRCController(ADRCConfig(b0=130.0))
        assert ctrl.b0_estimator is None

        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        ctrl.config.use_observations = True
        action = ctrl.step(obs, {}, dt=0.01)
        assert action.shape == (1,)

    def test_wind_feedforward_accepts_b0_estimator(self):
        from wind_feedforward import WindFeedforwardADRC, WindFeedforwardConfig
        from adrc_controller import ADRCConfig

        est = B0Estimator(B0EstimatorConfig(b0_init=130.0))
        ctrl = WindFeedforwardADRC(
            ADRCConfig(b0=130.0),
            WindFeedforwardConfig(),
            b0_estimator=est,
        )
        assert ctrl.adrc.b0_estimator is est


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_rls_b0_flag_exists(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "--rls-b0" in source

    def test_rls_colors_defined(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "ADRC+RLS" in source

    def test_imports_exist(self):
        from online_identification import B0Estimator, B0EstimatorConfig

        est = B0Estimator(B0EstimatorConfig())
        assert hasattr(est, "update")
        assert hasattr(est, "reset")
        assert hasattr(est, "b0_hat")
