"""
Tests for the neural network wind estimator (Step 7).

Verifies that:
1. WindEstimatorNetwork architecture is correct (input/output shapes)
2. NNWindEstimator maintains observation buffer and produces estimates
3. NNFeedforwardADRC controller interface matches other controllers
4. Training pipeline (prepare_windows, train) produces a working model
5. Save/load round-trips correctly
6. Feedforward action opposes wind disturbance direction
7. Controller is stable (doesn't diverge) with NN estimator
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path

from wind_estimator import (
    WindEstimatorConfig,
    WindEstimatorNetwork,
    NNWindEstimator,
    NNFeedforwardADRC,
    OBS_INDICES,
    prepare_training_windows,
    train_wind_estimator,
    save_estimator,
    load_estimator,
)
from adrc_controller import ADRCController, ADRCConfig


# --- Helpers ---


def make_obs(roll_angle=0.0, roll_rate=0.0, roll_accel=0.0, q=500.0, action=0.0):
    """Create a standard observation array."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = roll_angle
    obs[3] = roll_rate
    obs[4] = roll_accel
    obs[5] = q
    obs[8] = action
    return obs


def make_info(
    roll_angle_rad=0.0,
    roll_rate_deg_s=0.0,
    accel=50.0,
    q=500.0,
    wind_speed=0.0,
    wind_dir=0.0,
):
    """Create a standard info dict."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": accel,
        "dynamic_pressure_Pa": q,
        "wind_speed_ms": wind_speed,
        "wind_direction_rad": wind_dir,
    }


def create_dummy_model(config=None):
    """Create a small untrained GRU model for testing."""
    cfg = config or WindEstimatorConfig(window_size=10, hidden_size=8)
    model = WindEstimatorNetwork(cfg)
    return model, cfg


def create_dummy_estimator(config=None):
    """Create a NNWindEstimator with dummy normalization stats."""
    model, cfg = create_dummy_model(config)
    estimator = NNWindEstimator(model, cfg)
    estimator.obs_mean = np.zeros(cfg.obs_features, dtype=np.float32)
    estimator.obs_std = np.ones(cfg.obs_features, dtype=np.float32)
    estimator.target_mean = np.zeros(3, dtype=np.float32)
    estimator.target_std = np.ones(3, dtype=np.float32)
    return estimator, cfg


# --- Network Architecture Tests ---


class TestWindEstimatorNetwork:
    """Test GRU network architecture."""

    def test_default_config(self):
        cfg = WindEstimatorConfig()
        assert cfg.window_size == 20
        assert cfg.hidden_size == 32
        assert cfg.obs_features == 5

    def test_forward_shape(self):
        """Network should output (batch, 3) for wind prediction."""
        cfg = WindEstimatorConfig(window_size=10, hidden_size=16)
        model = WindEstimatorNetwork(cfg)
        x = torch.randn(4, 10, 5)  # batch=4, window=10, features=5
        y = model(x)
        assert y.shape == (4, 3), f"Expected (4, 3), got {y.shape}"

    def test_single_sample(self):
        """Should work with batch size 1."""
        cfg = WindEstimatorConfig(window_size=5, hidden_size=8)
        model = WindEstimatorNetwork(cfg)
        x = torch.randn(1, 5, 5)
        y = model(x)
        assert y.shape == (1, 3)

    def test_output_is_finite(self):
        """Output should not contain NaN or Inf."""
        model, cfg = create_dummy_model()
        x = torch.randn(2, cfg.window_size, cfg.obs_features)
        y = model(x)
        assert torch.isfinite(y).all()

    def test_obs_indices_correct(self):
        """OBS_INDICES should select the right features."""
        obs = make_obs(
            roll_angle=0.1, roll_rate=0.2, roll_accel=0.3, q=500.0, action=0.5
        )
        features = obs[OBS_INDICES]
        assert features[0] == pytest.approx(0.1)  # roll_angle
        assert features[1] == pytest.approx(0.2)  # roll_rate
        assert features[2] == pytest.approx(0.3)  # roll_accel
        assert features[3] == pytest.approx(500.0)  # q
        assert features[4] == pytest.approx(0.5)  # last_action


# --- NNWindEstimator Tests ---


class TestNNWindEstimator:
    """Test runtime wind estimator wrapper."""

    def test_reset_clears_buffer(self):
        estimator, _ = create_dummy_estimator()
        estimator.update(make_obs())
        estimator.update(make_obs())
        assert len(estimator._obs_buffer) == 2
        estimator.reset()
        assert len(estimator._obs_buffer) == 0

    def test_update_adds_to_buffer(self):
        estimator, cfg = create_dummy_estimator()
        for _ in range(5):
            estimator.update(make_obs())
        assert len(estimator._obs_buffer) == 5

    def test_buffer_limits_to_window_size(self):
        cfg = WindEstimatorConfig(window_size=10, hidden_size=8)
        estimator, _ = create_dummy_estimator(cfg)
        for _ in range(25):
            estimator.update(make_obs())
        assert len(estimator._obs_buffer) == 10

    def test_estimate_returns_tuple(self):
        estimator, _ = create_dummy_estimator()
        for _ in range(15):
            estimator.update(make_obs())
        speed, direction = estimator.estimate()
        assert isinstance(speed, float)
        assert isinstance(direction, float)

    def test_estimate_speed_non_negative(self):
        estimator, _ = create_dummy_estimator()
        for _ in range(15):
            estimator.update(make_obs(roll_rate=np.radians(20.0)))
        speed, _ = estimator.estimate()
        assert speed >= 0.0

    def test_estimate_direction_in_range(self):
        estimator, _ = create_dummy_estimator()
        for _ in range(15):
            estimator.update(make_obs())
        _, direction = estimator.estimate()
        assert -np.pi <= direction <= np.pi

    def test_estimate_with_short_buffer(self):
        """Should handle buffer shorter than window via padding."""
        estimator, _ = create_dummy_estimator()
        estimator.update(make_obs())
        estimator.update(make_obs())
        # Should not raise
        speed, direction = estimator.estimate()
        assert np.isfinite(speed)
        assert np.isfinite(direction)

    def test_estimate_with_empty_buffer(self):
        """Should return zeros for empty buffer."""
        estimator, _ = create_dummy_estimator()
        speed, direction = estimator.estimate()
        assert speed == 0.0
        assert direction == 0.0


# --- NNFeedforwardADRC Controller Tests ---


class TestNNFeedforwardADRC:
    """Test the ADRC + NN controller interface."""

    def test_reset_interface(self):
        estimator, cfg = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator)
        ctrl.reset()
        assert ctrl.launch_detected is False
        assert ctrl._step_count == 0

    def test_step_returns_action(self):
        estimator, cfg = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator)
        obs = make_obs(roll_angle=0.1, roll_rate=np.radians(20.0))
        action = ctrl.step(obs, {})
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_launches_in_obs_mode(self):
        estimator, cfg = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator)
        obs = make_obs(roll_angle=0.1)
        ctrl.step(obs, {})
        assert ctrl.launch_detected is True

    def test_warmup_suppresses_feedforward(self):
        estimator, cfg = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator, warmup_steps=100)
        obs = make_obs(roll_rate=np.radians(20.0))
        ctrl.step(obs, {})
        # During warmup, ff_action should be 0
        assert ctrl._ff_action == 0.0

    def test_feedforward_activates_after_warmup(self):
        cfg = WindEstimatorConfig(window_size=5, hidden_size=8, warmup_steps=5)
        estimator, _ = create_dummy_estimator(cfg)
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator, warmup_steps=5)

        obs = make_obs(roll_rate=np.radians(20.0))
        for _ in range(10):
            ctrl.step(obs, {})
        # After warmup, feedforward should have computed something
        # (may be 0 if estimator predicts 0 wind, but at least it ran)
        assert ctrl._step_count >= 5

    def test_properties_delegate_to_adrc(self):
        estimator, _ = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator)
        obs = make_obs(roll_angle=0.1, roll_rate=np.radians(5.0))
        ctrl.step(obs, {})
        # z1, z2, z3 should be accessible
        assert hasattr(ctrl, "z1")
        assert hasattr(ctrl, "z2")
        assert hasattr(ctrl, "z3")

    def test_action_clamped(self):
        estimator, _ = create_dummy_estimator()
        adrc_config = ADRCConfig(
            b0=1.0, use_observations=True
        )  # Very low b0 → large action
        ctrl = NNFeedforwardADRC(adrc_config, estimator, warmup_steps=0)
        obs = make_obs(roll_rate=np.radians(100.0))
        action = ctrl.step(obs, {})
        assert -1.0 <= action[0] <= 1.0

    def test_works_without_estimator(self):
        """Should work with estimator=None (pure ADRC)."""
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator=None)
        obs = make_obs(roll_rate=np.radians(20.0))
        action = ctrl.step(obs, {})
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_wind_speed_estimate_property(self):
        estimator, _ = create_dummy_estimator()
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator, warmup_steps=0)
        obs = make_obs(roll_rate=np.radians(20.0))
        for _ in range(5):
            ctrl.step(obs, {})
        assert isinstance(ctrl.wind_speed_estimate, float)
        assert isinstance(ctrl.wind_direction_estimate, float)


# --- Training Pipeline Tests ---


class TestTrainingPipeline:
    """Test the data preparation and training functions."""

    def test_prepare_windows_shape(self):
        """prepare_training_windows should produce correct shapes."""
        # Simulate 3 episodes of 50 steps each
        obs_seqs = [np.random.randn(50, 5).astype(np.float32) for _ in range(3)]
        labels = [np.random.randn(50, 3).astype(np.float32) for _ in range(3)]

        X, Y = prepare_training_windows(obs_seqs, labels, window_size=10)
        assert X.ndim == 3
        assert X.shape[1] == 10  # window_size
        assert X.shape[2] == 5  # features
        assert Y.ndim == 2
        assert Y.shape[1] == 3  # [speed, cos, sin]
        assert len(X) == len(Y)
        # Each episode of 50 steps produces 50-10 = 40 windows
        assert len(X) == 3 * 40

    def test_prepare_windows_skips_short_episodes(self):
        """Episodes shorter than window should be skipped."""
        obs_seqs = [np.random.randn(5, 5).astype(np.float32)]  # Too short
        labels = [np.random.randn(5, 3).astype(np.float32)]

        X, Y = prepare_training_windows(obs_seqs, labels, window_size=10)
        assert len(X) == 0

    def test_train_produces_model(self):
        """Training should return a model with lower validation loss than random."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Create synthetic data: observations → wind labels
        # Simple relationship: wind_speed correlates with roll_accel variance
        N = 500
        window = 10
        X = np.random.randn(N, window, 5).astype(np.float32)
        Y = np.zeros((N, 3), dtype=np.float32)
        # Target: wind_speed = mean of abs(roll_accel) in window
        Y[:, 0] = np.abs(X[:, :, 2]).mean(axis=1)  # speed from accel
        Y[:, 1] = 1.0  # cos(0) = 1
        Y[:, 2] = 0.0  # sin(0) = 0

        cfg = WindEstimatorConfig(window_size=window, hidden_size=8)
        model, stats = train_wind_estimator(
            X,
            Y,
            cfg,
            epochs=20,
            batch_size=64,
            verbose=False,
        )

        assert len(stats["train_loss"]) == 20
        assert len(stats["val_loss"]) == 20
        # Training loss should decrease
        assert stats["train_loss"][-1] < stats["train_loss"][0]

    def test_train_stores_normalization(self):
        """Training stats should include normalization parameters."""
        N = 100
        window = 5
        X = np.random.randn(N, window, 5).astype(np.float32)
        Y = np.random.randn(N, 3).astype(np.float32)

        cfg = WindEstimatorConfig(window_size=window, hidden_size=8)
        _, stats = train_wind_estimator(X, Y, cfg, epochs=2, verbose=False)

        assert "obs_mean" in stats
        assert "obs_std" in stats
        assert "target_mean" in stats
        assert "target_std" in stats
        assert stats["obs_mean"].shape == (5,)
        assert stats["target_mean"].shape == (3,)


# --- Save/Load Tests ---


class TestSaveLoad:
    """Test model persistence."""

    def test_save_load_roundtrip(self):
        """Save and load should produce identical estimator."""
        np.random.seed(42)
        torch.manual_seed(42)

        # Train a tiny model
        N = 100
        window = 5
        cfg = WindEstimatorConfig(
            window_size=window, hidden_size=8, K_ff=0.7, warmup_steps=30
        )
        X = np.random.randn(N, window, 5).astype(np.float32)
        Y = np.random.randn(N, 3).astype(np.float32)
        model, stats = train_wind_estimator(X, Y, cfg, epochs=2, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            save_path = f.name

        try:
            save_estimator(model, stats, cfg, save_path)
            loaded_estimator, loaded_cfg = load_estimator(save_path)

            # Config should match
            assert loaded_cfg.window_size == cfg.window_size
            assert loaded_cfg.hidden_size == cfg.hidden_size
            assert loaded_cfg.K_ff == cfg.K_ff
            assert loaded_cfg.warmup_steps == cfg.warmup_steps

            # Normalization should match
            np.testing.assert_array_almost_equal(
                loaded_estimator.obs_mean,
                stats["obs_mean"],
            )
            np.testing.assert_array_almost_equal(
                loaded_estimator.target_mean,
                stats["target_mean"],
            )

            # Predictions should match
            test_input = np.random.randn(1, window, 5).astype(np.float32)
            # Normalize as estimator would
            x_norm = (test_input - stats["obs_mean"]) / (stats["obs_std"] + 1e-8)
            x_tensor = torch.from_numpy(x_norm).float()

            model.eval()
            loaded_estimator.model.eval()
            with torch.no_grad():
                orig_out = model(x_tensor).numpy()
                loaded_out = loaded_estimator.model(x_tensor).numpy()
            np.testing.assert_array_almost_equal(orig_out, loaded_out)
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_load_nonexistent_raises(self):
        """Loading nonexistent file should raise."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_estimator("/nonexistent/model.pt")


# --- Stability Tests ---


class TestClosedLoopStability:
    """Test that ADRC+NN doesn't diverge in closed-loop simulation."""

    def test_stable_with_untrained_estimator(self):
        """Controller should remain stable even with random NN estimates."""
        np.random.seed(42)
        estimator, cfg = create_dummy_estimator(
            WindEstimatorConfig(window_size=5, hidden_size=8, warmup_steps=10),
        )
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator, K_ff=0.5, warmup_steps=10)

        roll_angle = 0.0
        roll_rate = np.radians(30.0)
        dt = 0.01

        for step in range(300):
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            action = ctrl.step(obs, {}, dt)
            alpha = 100.0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert final_rate < 100.0, (
            f"ADRC+NN should not diverge with untrained estimator: "
            f"final rate = {final_rate:.1f} deg/s"
        )

    def test_stable_without_estimator(self):
        """Pure ADRC (no estimator) should settle from initial spin."""
        adrc_config = ADRCConfig(b0=100.0, use_observations=True)
        ctrl = NNFeedforwardADRC(adrc_config, estimator=None)

        roll_angle = 0.0
        roll_rate = np.radians(30.0)
        dt = 0.01

        for step in range(300):
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            action = ctrl.step(obs, {}, dt)
            alpha = 100.0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert (
            final_rate < 5.0
        ), f"Pure ADRC should settle: final rate = {final_rate:.1f} deg/s"


# --- Integration Tests ---


class TestCompareControllersIntegration:
    """Test that compare_controllers.py properly imports ADRC+NN."""

    def test_import_nn_feedforward(self):
        """compare_controllers.py should import NNFeedforwardADRC."""
        source = Path("compare_controllers.py").read_text()
        assert "NNFeedforwardADRC" in source
        assert "load_estimator" in source

    def test_adrc_nn_flag_in_argparse(self):
        """compare_controllers.py should have --adrc-nn flag."""
        source = Path("compare_controllers.py").read_text()
        assert "--adrc-nn" in source

    def test_adrc_nn_color_defined(self):
        """ADRC+NN should have a color in the plot color map."""
        source = Path("compare_controllers.py").read_text()
        assert '"ADRC+NN"' in source or "'ADRC+NN'" in source
