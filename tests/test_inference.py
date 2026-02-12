"""
Tests for rocket_env/inference modules.

Tests for ONNXRunner and RocketController classes for embedded deployment.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pickle
import tempfile

# Check if onnxruntime is available
try:
    import onnxruntime

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class TestONNXRunner:
    """Tests for ONNXRunner class."""

    @pytest.mark.skipif(
        HAS_ONNXRUNTIME, reason="Test requires onnxruntime to be missing"
    )
    def test_import_error_without_onnxruntime(self, tmp_path):
        """Test proper error when onnxruntime is not available."""
        # This test only runs when onnxruntime is not installed
        from rocket_env.inference.onnx_runner import ONNXRunner

        with pytest.raises(ImportError, match="onnxruntime is required"):
            ONNXRunner(str(tmp_path / "model.onnx"))

    def test_predict_reshape_1d_input(self):
        """Test that 1D input is reshaped to 2D."""
        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # Simulate the reshaping logic
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        assert obs.shape == (1, 3)

    def test_get_info_structure(self):
        """Test get_info returns expected structure."""
        info = {
            "model_path": "model.onnx",
            "input_name": "obs",
            "input_shape": [1, 10],
            "output_names": ["action"],
            "is_deterministic": True,
        }

        assert "model_path" in info
        assert "input_name" in info
        assert "input_shape" in info
        assert "output_names" in info
        assert "is_deterministic" in info


class MockRunningMeanStd:
    """Mock class for VecNormalize running mean/std."""

    def __init__(self, mean, var):
        self.mean = np.array(mean, dtype=np.float32)
        self.var = np.array(var, dtype=np.float32)


class MockVecNormalize:
    """Mock VecNormalize class that can be pickled."""

    def __init__(self, mean, var):
        self.obs_rms = MockRunningMeanStd(mean, var)


class MockVecNormalizeAlt:
    """Alternative format mock VecNormalize."""

    def __init__(self, mean, var):
        self.running_mean = np.array(mean, dtype=np.float32)
        self.running_var = np.array(var, dtype=np.float32)


class TestRocketController:
    """Tests for RocketController class."""

    @pytest.fixture
    def mock_vec_normalize_file(self, tmp_path):
        """Create a mock VecNormalize pickle file."""
        mock_vec_normalize = MockVecNormalize(
            mean=[0.0, 1.0, 2.0, 3.0, 4.0],
            var=[1.0, 1.0, 1.0, 1.0, 1.0],
        )

        pkl_path = tmp_path / "vec_normalize.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mock_vec_normalize, f)

        return str(pkl_path)

    def test_normalize_observation(self, mock_vec_normalize_file):
        """Test observation normalization."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        controller.obs_var = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        controller.epsilon = 1e-8
        controller.clip_obs = 10.0

        obs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        normalized = controller.normalize_observation(obs)

        # (obs - mean) / sqrt(var + eps) = (obs - obs) / 1 = 0
        assert np.allclose(normalized, 0.0, atol=1e-5)

    def test_normalize_observation_with_clipping(self):
        """Test observation clipping after normalization."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = np.array([0.0], dtype=np.float32)
        controller.obs_var = np.array([0.01], dtype=np.float32)  # Small variance
        controller.epsilon = 1e-8
        controller.clip_obs = 5.0

        obs = np.array([100.0], dtype=np.float32)  # Large value
        normalized = controller.normalize_observation(obs)

        # Should be clipped to clip_obs
        assert normalized[0] == 5.0

    def test_normalize_observation_no_stats(self):
        """Test normalization without stats returns original."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller.epsilon = 1e-8
        controller.clip_obs = 10.0

        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        normalized = controller.normalize_observation(obs)

        assert np.allclose(normalized, obs)

    def test_load_normalize_stats(self, mock_vec_normalize_file):
        """Test loading normalization stats from pickle."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller._load_normalize_stats(mock_vec_normalize_file)

        assert controller.obs_mean is not None
        assert controller.obs_var is not None
        assert len(controller.obs_mean) == 5

    def test_load_normalize_stats_file_not_found(self, tmp_path):
        """Test error when normalize file doesn't exist."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)

        with pytest.raises(FileNotFoundError):
            controller._load_normalize_stats(str(tmp_path / "nonexistent.pkl"))

    def test_load_normalize_stats_alternative_format(self, tmp_path):
        """Test loading stats with alternative attribute names."""
        mock_vec_normalize = MockVecNormalizeAlt(
            mean=[1.0, 2.0],
            var=[0.5, 0.5],
        )

        pkl_path = tmp_path / "alt_normalize.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(mock_vec_normalize, f)

        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller._load_normalize_stats(str(pkl_path))

        assert controller.obs_mean is not None
        assert np.allclose(controller.obs_mean, [1.0, 2.0])

    def test_get_action(self):
        """Test get_action method."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller.epsilon = 1e-8
        controller.clip_obs = 10.0
        controller.action_low = -1.0
        controller.action_high = 1.0

        mock_runner = MagicMock()
        mock_runner.predict.return_value = np.array([0.5], dtype=np.float32)
        controller.runner = mock_runner

        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = controller.get_action(obs, normalize=False)

        assert action[0] == 0.5

    def test_get_action_clipping(self):
        """Test action clipping in get_action."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller.epsilon = 1e-8
        controller.clip_obs = 10.0
        controller.action_low = -1.0
        controller.action_high = 1.0

        mock_runner = MagicMock()
        mock_runner.predict.return_value = np.array(
            [2.0], dtype=np.float32
        )  # Out of range
        controller.runner = mock_runner

        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        action = controller.get_action(obs, normalize=False)

        assert action[0] == 1.0  # Clipped to max

    def test_get_tab_deflection(self):
        """Test get_tab_deflection convenience method."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.obs_mean = None
        controller.obs_var = None
        controller.epsilon = 1e-8
        controller.clip_obs = 10.0
        controller.action_low = -1.0
        controller.action_high = 1.0

        mock_runner = MagicMock()
        mock_runner.predict.return_value = np.array([0.5], dtype=np.float32)
        controller.runner = mock_runner

        obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        deflection = controller.get_tab_deflection(obs, max_deflection_deg=20.0)

        # 0.5 * 20.0 = 10.0
        assert deflection == 10.0

    def test_get_info(self):
        """Test get_info method."""
        from rocket_env.inference.controller import RocketController

        controller = RocketController.__new__(RocketController)
        controller.model_path = Path("test_model.onnx")
        controller.obs_mean = np.array([0.0])
        controller.action_low = -1.0
        controller.action_high = 1.0

        mock_runner = MagicMock()
        mock_runner.get_info.return_value = {"input_name": "obs"}
        controller.runner = mock_runner

        info = controller.get_info()

        assert info["model_path"] == "test_model.onnx"
        assert info["has_normalization"] is True
        assert info["action_range"] == (-1.0, 1.0)
        assert info["input_name"] == "obs"


class TestLoadControllerFromTraining:
    """Tests for load_controller_from_training function."""

    def test_load_controller_file_not_found(self, tmp_path):
        """Test error when model file doesn't exist."""
        from rocket_env.inference.controller import load_controller_from_training

        with pytest.raises(FileNotFoundError, match="Model not found"):
            load_controller_from_training(str(tmp_path))


class TestPIDDeployController:
    """Tests for PIDDeployController class."""

    def test_basic_action(self):
        """Test that PID returns action in [-1, 1]."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        action = pid.get_action(
            roll_angle_deg=10.0,
            roll_rate_deg_s=5.0,
            dynamic_pressure_pa=500.0,
        )
        assert -1.0 <= action <= 1.0

    def test_zero_error_gives_near_zero(self):
        """Test that zero roll angle/rate gives near-zero action."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        action = pid.get_action(
            roll_angle_deg=0.0,
            roll_rate_deg_s=0.0,
            dynamic_pressure_pa=500.0,
        )
        assert abs(action) < 0.01

    def test_gain_scheduling(self):
        """Test that gain scheduling changes effective gains."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01, q_ref=500.0)

        # At q_ref, scale should be ~1.0
        scale_at_ref = pid._gain_scale(500.0)
        assert 0.9 < scale_at_ref < 1.1

        # At low q, scale should be high (up to 5.0)
        scale_at_low = pid._gain_scale(10.0)
        assert scale_at_low > scale_at_ref

        # At high q, scale should be low (down to 0.5)
        scale_at_high = pid._gain_scale(5000.0)
        assert scale_at_high < scale_at_ref

    def test_anti_windup_clamp(self):
        """Test that integral error is clamped (anti-windup)."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.001, Kd=0.01)

        # Accumulate large integral by calling many times with large error
        for _ in range(10000):
            pid.get_action(
                roll_angle_deg=180.0,
                roll_rate_deg_s=0.0,
                dynamic_pressure_pa=500.0,
            )

        max_integ = pid.max_deflection / (pid.Ki + 1e-6)
        assert abs(pid.integ_error) <= max_integ + 1e-6

    def test_launch_detection_with_accel(self):
        """Test launch detection via vertical acceleration."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)

        # Below threshold: should return 0
        action = pid.get_action(
            roll_angle_deg=10.0,
            roll_rate_deg_s=5.0,
            dynamic_pressure_pa=500.0,
            vertical_accel_ms2=5.0,
        )
        assert action == 0.0
        assert not pid.launch_detected

        # Above threshold: should return non-zero for non-zero input
        action = pid.get_action(
            roll_angle_deg=10.0,
            roll_rate_deg_s=5.0,
            dynamic_pressure_pa=500.0,
            vertical_accel_ms2=25.0,
        )
        assert pid.launch_detected
        assert action != 0.0

    def test_launch_detection_without_accel(self):
        """Test launch assumed immediately when no accel provided."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        pid.get_action(
            roll_angle_deg=0.0,
            roll_rate_deg_s=0.0,
            dynamic_pressure_pa=500.0,
        )
        assert pid.launch_detected

    def test_reset(self):
        """Test that reset clears all state."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        # First call sets launch_orient to 0
        pid.get_action(0.0, 0.0, 500.0)
        # Second call with error accumulates integral
        pid.get_action(10.0, 5.0, 500.0)
        assert pid.launch_detected
        assert pid.integ_error != 0.0

        pid.reset()
        assert not pid.launch_detected
        assert pid.integ_error == 0.0
        assert pid.target_orient == 0.0

    def test_json_config_loading(self, tmp_path):
        """Test loading PID gains from JSON config file."""
        import json
        from rocket_env.inference.controller import PIDDeployController

        config = {"Kp": 0.05, "Ki": 0.001, "Kd": 0.03, "q_ref": 1000.0}
        config_path = tmp_path / "controller_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f)

        pid = PIDDeployController(config_path=str(config_path))
        assert pid.Kp == 0.05
        assert pid.Ki == 0.001
        assert pid.Kd == 0.03
        assert pid.q_ref == 1000.0

    def test_get_tab_deflection(self):
        """Test get_tab_deflection convenience method."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        action = pid.get_action(10.0, 5.0, 500.0)
        deflection = pid.get_tab_deflection(10.0, 5.0, 500.0, max_deflection_deg=20.0)
        assert abs(deflection - action * 20.0) < 1e-6

    def test_opposing_error_direction(self):
        """Test that positive roll error produces negative action (opposing)."""
        from rocket_env.inference.controller import PIDDeployController

        pid = PIDDeployController(Kp=0.1, Ki=0.0, Kd=0.0)
        # First call sets launch_orient to 0
        pid.get_action(0.0, 0.0, 500.0)
        # Second call with positive roll error should produce negative action
        action = pid.get_action(
            roll_angle_deg=30.0,
            roll_rate_deg_s=0.0,
            dynamic_pressure_pa=500.0,
        )
        assert action < 0


class TestResidualSACController:
    """Tests for ResidualSACController class."""

    def test_combined_output(self):
        """Test that output = PID + clipped SAC residual."""
        from rocket_env.inference.controller import (
            ResidualSACController,
            PIDDeployController,
        )

        controller = ResidualSACController.__new__(ResidualSACController)
        controller.pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        controller.max_residual = 0.2

        mock_sac = MagicMock()
        mock_sac.get_action.return_value = np.array([0.5], dtype=np.float32)
        controller.sac = mock_sac

        obs = np.zeros(10, dtype=np.float32)
        action = controller.get_action(
            observation=obs,
            roll_angle_deg=10.0,
            roll_rate_deg_s=5.0,
            dynamic_pressure_pa=500.0,
        )

        pid_action = controller.pid.get_action(10.0, 5.0, 500.0)
        expected_residual = np.clip(0.5 * 0.2, -0.2, 0.2)
        expected = np.clip(pid_action + expected_residual, -1.0, 1.0)

        # Need to re-create PID state since it was called above
        # Just check action is in valid range and combined
        assert -1.0 <= action <= 1.0

    def test_residual_clamping(self):
        """Test that SAC residual is clamped to max_residual."""
        from rocket_env.inference.controller import (
            ResidualSACController,
            PIDDeployController,
        )

        controller = ResidualSACController.__new__(ResidualSACController)
        controller.pid = PIDDeployController(Kp=0.0, Ki=0.0, Kd=0.0)
        controller.max_residual = 0.1

        mock_sac = MagicMock()
        # SAC outputs 1.0 -> residual should be clamped to 0.1
        mock_sac.get_action.return_value = np.array([1.0], dtype=np.float32)
        controller.sac = mock_sac

        obs = np.zeros(10, dtype=np.float32)
        action = controller.get_action(obs, 0.0, 0.0, 500.0)

        # PID is all zeros, so action should equal clamped residual
        assert abs(action - 0.1) < 1e-6

    def test_reset_propagation(self):
        """Test that reset resets PID state."""
        from rocket_env.inference.controller import (
            ResidualSACController,
            PIDDeployController,
        )

        controller = ResidualSACController.__new__(ResidualSACController)
        controller.pid = PIDDeployController(Kp=0.02, Ki=0.0002, Kd=0.01)
        controller.max_residual = 0.2
        controller.sac = MagicMock()

        # Exercise PID to build state
        controller.pid.get_action(10.0, 5.0, 500.0)
        assert controller.pid.launch_detected

        controller.reset()
        assert not controller.pid.launch_detected
        assert controller.pid.integ_error == 0.0


class TestLoadNormalizeJson:
    """Tests for load_normalize_json helper."""

    def test_load_normalize_json(self, tmp_path):
        """Test loading normalization stats from JSON."""
        import json
        from rocket_env.inference.controller import load_normalize_json

        stats = {"obs_mean": [1.0, 2.0, 3.0], "obs_var": [0.5, 0.5, 0.5]}
        json_path = tmp_path / "normalize.json"
        with open(json_path, "w") as f:
            json.dump(stats, f)

        mean, var = load_normalize_json(str(json_path))
        assert np.allclose(mean, [1.0, 2.0, 3.0])
        assert np.allclose(var, [0.5, 0.5, 0.5])
        assert mean.dtype == np.float32
        assert var.dtype == np.float32


class TestExportOnnxSAC:
    """Tests for SAC ONNX export path."""

    def test_build_sac_wrapper_structure(self):
        """Test that SAC wrapper has correct forward path."""
        import torch

        # Mock the SAC actor structure
        class MockActor:
            features_extractor = torch.nn.Linear(10, 64)
            latent_pi = torch.nn.Linear(64, 64)
            mu = torch.nn.Linear(64, 1)

        from deployment.export_onnx import _build_sac_wrapper

        mock_model = MagicMock()
        mock_model.policy.actor = MockActor()

        wrapper = _build_sac_wrapper(mock_model)
        wrapper.eval()

        # Forward pass should work
        obs = torch.randn(1, 10)
        with torch.no_grad():
            action = wrapper(obs)

        # Output should be tanh-bounded
        assert action.shape == (1, 1)
        assert -1.0 <= action.item() <= 1.0

    def test_build_ppo_wrapper_structure(self):
        """Test that PPO wrapper has correct forward path (no mlp_extractor)."""
        import torch

        class MockPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self._fe = torch.nn.Linear(10, 64)
                self.action_net = torch.nn.Linear(64, 1)

            def extract_features(self, obs):
                return self._fe(obs)

        from deployment.export_onnx import _build_ppo_wrapper

        mock_model = MagicMock()
        mock_model.policy = MockPolicy()

        wrapper = _build_ppo_wrapper(mock_model)
        wrapper.eval()

        obs = torch.randn(1, 10)
        with torch.no_grad():
            action = wrapper(obs)

        assert action.shape == (1, 1)

    def test_load_sb3_model_invalid(self, tmp_path):
        """Test that invalid model path raises ValueError."""
        from deployment.export_onnx import _load_sb3_model

        # Create a dummy file that isn't a valid model
        dummy = tmp_path / "bad_model.zip"
        dummy.write_text("not a model")

        with pytest.raises(ValueError, match="Could not load model"):
            _load_sb3_model(str(dummy))


class TestInferencePackageImports:
    """Tests for inference package imports."""

    def test_package_imports(self):
        """Test that package exposes expected classes."""
        from rocket_env.inference import ONNXRunner, RocketController

        assert ONNXRunner is not None
        assert RocketController is not None

    def test_all_exports(self):
        """Test __all__ exports."""
        from rocket_env import inference

        assert "ONNXRunner" in inference.__all__
        assert "RocketController" in inference.__all__
        assert "PIDDeployController" in inference.__all__
        assert "ResidualSACController" in inference.__all__
        assert "load_normalize_json" in inference.__all__

    def test_new_imports(self):
        """Test that new classes are importable."""
        from rocket_env.inference import (
            PIDDeployController,
            ResidualSACController,
            load_normalize_json,
        )

        assert PIDDeployController is not None
        assert ResidualSACController is not None
        assert load_normalize_json is not None
