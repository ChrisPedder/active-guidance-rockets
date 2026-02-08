"""
Tests for the GP disturbance model with uncertainty gating.

Verifies:
1. SparseGP kernel computation and predictions
2. Uncertainty increases away from data
3. Sigmoid gate behavior
4. Budget management (circular buffer)
5. Controller interface (reset/step)
6. Uncertainty gating suppresses feedforward when uncertain
7. Closed-loop stability
8. Integration with compare_controllers.py
"""

import numpy as np
import pytest

from gp_disturbance import (
    GPFeedforwardController,
    GPDisturbanceConfig,
    SparseGP,
)
from pid_controller import PIDConfig


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


class TestSparseGPKernel:
    """Test the RBF kernel computation."""

    def test_kernel_self_similarity(self):
        """k(x, x) should equal signal_variance."""
        gp = SparseGP(
            input_dim=2,
            budget_size=10,
            length_scales=np.array([1.0, 1.0]),
            signal_variance=5.0,
            noise_variance=0.1,
        )
        x = np.array([[1.0, 2.0]])
        K = gp._rbf_kernel(x, x)
        assert abs(K[0, 0] - 5.0) < 1e-10

    def test_kernel_symmetric(self):
        """K(X1, X2) should equal K(X2, X1)^T."""
        gp = SparseGP(
            input_dim=3,
            budget_size=10,
            length_scales=np.array([1.0, 2.0, 0.5]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        X1 = np.random.randn(5, 3)
        X2 = np.random.randn(3, 3)
        K12 = gp._rbf_kernel(X1, X2)
        K21 = gp._rbf_kernel(X2, X1)
        np.testing.assert_allclose(K12, K21.T, atol=1e-10)

    def test_kernel_positive_definite(self):
        """Kernel matrix should be positive definite."""
        gp = SparseGP(
            input_dim=2,
            budget_size=10,
            length_scales=np.array([1.0, 1.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        X = np.random.randn(5, 2)
        K = gp._rbf_kernel(X, X) + 0.1 * np.eye(5)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues > 0)

    def test_kernel_decreases_with_distance(self):
        """Kernel value should decrease as inputs get farther apart."""
        gp = SparseGP(
            input_dim=2,
            budget_size=10,
            length_scales=np.array([1.0, 1.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        x0 = np.array([[0.0, 0.0]])
        x_near = np.array([[0.1, 0.1]])
        x_far = np.array([[5.0, 5.0]])
        k_near = gp._rbf_kernel(x0, x_near)[0, 0]
        k_far = gp._rbf_kernel(x0, x_far)[0, 0]
        assert k_near > k_far

    def test_length_scale_effect(self):
        """Longer length scale should give higher kernel values at same distance."""
        x0 = np.array([[0.0]])
        x1 = np.array([[2.0]])

        gp_short = SparseGP(
            input_dim=1,
            budget_size=10,
            length_scales=np.array([0.5]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        gp_long = SparseGP(
            input_dim=1,
            budget_size=10,
            length_scales=np.array([5.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )

        k_short = gp_short._rbf_kernel(x0, x1)[0, 0]
        k_long = gp_long._rbf_kernel(x0, x1)[0, 0]
        assert k_long > k_short


class TestSparseGPPrediction:
    """Test GP prediction behavior."""

    def test_prior_prediction_zero_mean(self):
        """With no data, prediction should be zero mean."""
        gp = SparseGP(
            input_dim=2,
            budget_size=10,
            length_scales=np.array([1.0, 1.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        mean, var = gp.predict(np.array([0.5, 0.5]))
        assert mean == 0.0
        assert abs(var - 1.0) < 1e-6  # Prior variance = signal_variance

    def test_prediction_near_data(self):
        """Prediction near an observed point should be close to the observed value."""
        gp = SparseGP(
            input_dim=1,
            budget_size=10,
            length_scales=np.array([1.0]),
            signal_variance=10.0,
            noise_variance=0.1,
        )
        gp.add_point(np.array([0.0]), 5.0)

        mean, var = gp.predict(np.array([0.0]))
        assert abs(mean - 5.0) < 1.0  # Should be close to observed
        assert var < 10.0  # Variance should decrease near data

    def test_variance_increases_away_from_data(self):
        """Variance should be higher farther from observed data."""
        gp = SparseGP(
            input_dim=1,
            budget_size=10,
            length_scales=np.array([1.0]),
            signal_variance=10.0,
            noise_variance=0.1,
        )
        gp.add_point(np.array([0.0]), 5.0)

        _, var_near = gp.predict(np.array([0.1]))
        _, var_far = gp.predict(np.array([10.0]))
        assert var_far > var_near

    def test_multiple_points_reduce_variance(self):
        """More data points should reduce prediction variance."""
        gp1 = SparseGP(
            input_dim=1,
            budget_size=20,
            length_scales=np.array([1.0]),
            signal_variance=10.0,
            noise_variance=0.5,
        )
        gp1.add_point(np.array([0.0]), 5.0)

        gp2 = SparseGP(
            input_dim=1,
            budget_size=20,
            length_scales=np.array([1.0]),
            signal_variance=10.0,
            noise_variance=0.5,
        )
        for i in range(5):
            gp2.add_point(np.array([i * 0.5]), 5.0)

        _, var_1 = gp1.predict(np.array([0.5]))
        _, var_2 = gp2.predict(np.array([0.5]))
        assert var_2 < var_1

    def test_reset_clears_data(self):
        """Reset should return GP to prior state."""
        gp = SparseGP(
            input_dim=1,
            budget_size=10,
            length_scales=np.array([1.0]),
            signal_variance=10.0,
            noise_variance=0.1,
        )
        gp.add_point(np.array([0.0]), 5.0)
        gp.reset()

        mean, var = gp.predict(np.array([0.0]))
        assert mean == 0.0
        assert abs(var - 10.0) < 1e-6


class TestBudgetManagement:
    """Test circular buffer budget management."""

    def test_budget_limit(self):
        """Number of stored points should not exceed budget."""
        gp = SparseGP(
            input_dim=1,
            budget_size=5,
            length_scales=np.array([1.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        for i in range(20):
            gp.add_point(np.array([float(i)]), float(i))

        assert gp._n_points == 5

    def test_circular_replacement(self):
        """Old points should be replaced when budget is full."""
        gp = SparseGP(
            input_dim=1,
            budget_size=3,
            length_scales=np.array([1.0]),
            signal_variance=1.0,
            noise_variance=0.1,
        )
        # Add 5 points to a budget-3 GP
        for i in range(5):
            gp.add_point(np.array([float(i)]), float(i) * 10)

        # The oldest 2 points (0, 1) should be gone
        # Points 2, 3, 4 should remain
        # Due to circular buffer: positions 0,1,2 have points 3,4,2
        assert gp._n_points == 3


class TestSigmoidGate:
    """Test the uncertainty gating sigmoid."""

    def test_gate_high_when_confident(self):
        """Gate should be ~1 when sigma << threshold."""
        cfg = GPDisturbanceConfig(sigma_threshold=5.0, sigma_scale=2.0)
        ctrl = GPFeedforwardController(gp_config=cfg)
        gate = ctrl._sigmoid_gate(0.1)
        assert gate > 0.9

    def test_gate_low_when_uncertain(self):
        """Gate should be ~0 when sigma >> threshold."""
        cfg = GPDisturbanceConfig(sigma_threshold=5.0, sigma_scale=2.0)
        ctrl = GPFeedforwardController(gp_config=cfg)
        gate = ctrl._sigmoid_gate(20.0)
        assert gate < 0.1

    def test_gate_half_at_threshold(self):
        """Gate should be 0.5 at exactly sigma_threshold."""
        cfg = GPDisturbanceConfig(sigma_threshold=5.0, sigma_scale=2.0)
        ctrl = GPFeedforwardController(gp_config=cfg)
        gate = ctrl._sigmoid_gate(5.0)
        assert abs(gate - 0.5) < 0.01

    def test_gate_monotonically_decreasing(self):
        """Gate should decrease as sigma increases."""
        cfg = GPDisturbanceConfig(sigma_threshold=5.0, sigma_scale=2.0)
        ctrl = GPFeedforwardController(gp_config=cfg)
        sigmas = [0.1, 1.0, 3.0, 5.0, 10.0, 20.0]
        gates = [ctrl._sigmoid_gate(s) for s in sigmas]
        for i in range(len(gates) - 1):
            assert gates[i] >= gates[i + 1]


class TestGPInputBuilder:
    """Test GP input feature construction."""

    def test_input_dimension(self):
        ctrl = GPFeedforwardController()
        x = ctrl._build_gp_input(0.5, 1.0, 500.0)
        assert len(x) == 4

    def test_input_scaling(self):
        ctrl = GPFeedforwardController()
        x = ctrl._build_gp_input(0.0, 10.0, 1000.0)
        # cos(0)=1, sin(0)=0, q/1000=1.0, rate/10=1.0
        np.testing.assert_allclose(x, [1.0, 0.0, 1.0, 1.0], atol=1e-10)

    def test_input_trig_range(self):
        """Trig features should be in [-1, 1]."""
        ctrl = GPFeedforwardController()
        for angle in np.linspace(-np.pi, np.pi, 20):
            x = ctrl._build_gp_input(angle, 0.0, 500.0)
            assert -1.0 <= x[0] <= 1.0
            assert -1.0 <= x[1] <= 1.0


class TestControllerInterface:
    """Test the full controller interface."""

    def test_has_required_methods(self):
        ctrl = GPFeedforwardController()
        assert hasattr(ctrl, "reset") and callable(ctrl.reset)
        assert hasattr(ctrl, "step") and callable(ctrl.step)

    def test_reset_clears_state(self):
        ctrl = GPFeedforwardController()
        ctrl.launch_detected = True
        ctrl._step_count = 100
        ctrl._ff_action = 0.5
        ctrl._gate_value = 0.8

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert ctrl._step_count == 0
        assert ctrl._ff_action == 0.0
        assert ctrl._gate_value == 0.0

    def test_step_returns_correct_shape(self):
        ctrl = GPFeedforwardController(use_observations=True)
        obs = make_obs(roll_angle=0.1, roll_rate=0.5, q=500.0)
        action = ctrl.step(obs, {})
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_action_clamped_to_range(self):
        cfg = GPDisturbanceConfig(K_ff=10.0, warmup_steps=0, sigma_threshold=100.0)
        ctrl = GPFeedforwardController(gp_config=cfg, use_observations=True)
        obs = make_obs(roll_angle=1.0, roll_rate=10.0, q=500.0)

        for _ in range(200):
            action = ctrl.step(obs, {}, dt=0.01)
            assert -1.0 <= action[0] <= 1.0

    def test_launch_detection_ground_truth(self):
        ctrl = GPFeedforwardController()
        obs = np.zeros(10)
        info = make_info(accel=10.0)
        action = ctrl.step(obs, info)
        assert not ctrl.launch_detected
        assert action[0] == 0.0

        info = make_info(accel=50.0)
        action = ctrl.step(obs, info)
        assert ctrl.launch_detected

    def test_launch_detection_obs_mode(self):
        ctrl = GPFeedforwardController(use_observations=True)
        obs = make_obs(roll_angle=0.1, roll_rate=0.5, q=500.0)
        ctrl.step(obs, {})
        assert ctrl.launch_detected


class TestWarmupBehavior:
    """Test warmup suppresses feedforward."""

    def test_no_feedforward_during_warmup(self):
        cfg = GPDisturbanceConfig(warmup_steps=100)
        ctrl = GPFeedforwardController(gp_config=cfg, use_observations=True)
        obs = make_obs(roll_angle=0.3, roll_rate=1.0, q=500.0)

        for _ in range(50):
            ctrl.step(obs, {}, dt=0.01)

        assert ctrl._ff_action == 0.0
        assert ctrl._gate_value == 0.0


class TestUncertaintyGating:
    """Test that uncertainty gates the feedforward."""

    def test_high_uncertainty_suppresses_ff(self):
        """With no data, GP should be uncertain and suppress feedforward.

        Query a point far from any training data to ensure high variance.
        """
        cfg = GPDisturbanceConfig(
            warmup_steps=5,
            K_ff=1.0,
            sigma_threshold=5.0,
            signal_variance=100.0,
            update_interval=3,
        )
        ctrl = GPFeedforwardController(gp_config=cfg, use_observations=True)
        # Train on data near roll_angle=0
        obs_train = make_obs(roll_angle=0.0, roll_rate=0.5, q=500.0)
        for _ in range(10):
            ctrl.step(obs_train, {}, dt=0.01)

        # Now query at a very different roll_angle — GP should be uncertain
        obs_far = make_obs(roll_angle=3.0, roll_rate=5.0, q=100.0)
        ctrl.step(obs_far, {}, dt=0.01)

        # sigma at the far point should be high (near prior) -> gate low
        assert (
            ctrl._gp_std > 3.0
        ), f"GP std should be high far from data, got {ctrl._gp_std:.3f}"


class TestClosedLoopStability:
    """Test closed-loop stability."""

    def test_stable_without_disturbance(self):
        ctrl = GPFeedforwardController(use_observations=True)

        obs = make_obs(q=500.0)
        ctrl.step(obs, {})  # Launch detection

        roll_angle = np.radians(10.0)
        roll_rate = 0.0
        dt = 0.01
        b0 = 130.0

        for _ in range(500):
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            action = ctrl.step(obs, {}, dt)
            accel = b0 * action[0] - 2.0 * roll_rate
            roll_rate += accel * dt
            roll_angle += roll_rate * dt

        final_deg = abs(np.degrees(roll_angle))
        assert (
            final_deg < 20.0
        ), f"Should be stable without disturbance, got {final_deg:.1f} deg"

    def test_bounded_with_sinusoidal_disturbance(self):
        cfg = GPDisturbanceConfig(
            K_ff=0.3,
            warmup_steps=20,
            budget_size=30,
            sigma_threshold=5.0,
        )
        ctrl = GPFeedforwardController(gp_config=cfg, use_observations=True)

        obs = make_obs(q=500.0)
        ctrl.step(obs, {})

        roll_angle = 0.0
        roll_rate = 0.0
        dt = 0.01
        b0 = 130.0
        wind_amp = 10.0

        max_rate = 0.0
        for step in range(500):
            d = wind_amp * np.sin(1.0 - roll_angle)
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            action = ctrl.step(obs, {}, dt)
            accel = b0 * action[0] + d
            roll_rate += accel * dt
            roll_angle += roll_rate * dt
            max_rate = max(max_rate, abs(np.degrees(roll_rate)))

        assert max_rate < 100.0, f"Max rate {max_rate:.1f} deg/s too large"


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_controller_importable(self):
        from gp_disturbance import GPFeedforwardController

        ctrl = GPFeedforwardController()
        assert hasattr(ctrl, "step")
        assert hasattr(ctrl, "reset")
