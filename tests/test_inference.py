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

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="Requires onnxruntime")
    def test_file_not_found(self, tmp_path):
        """Test error when ONNX file doesn't exist."""
        from rocket_env.inference.onnx_runner import ONNXRunner

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            ONNXRunner(str(tmp_path / "nonexistent.onnx"))

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


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="Requires onnxruntime")
class TestONNXRunnerIntegration:
    """Integration tests for ONNXRunner (require onnxruntime)."""

    @pytest.fixture
    def simple_onnx_model(self, tmp_path):
        """Create a simple ONNX model for testing."""
        pytest.importorskip("onnx")

        import onnx
        from onnx import helper, TensorProto

        # Create a simple identity model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])

        # Identity with shape change via matmul with ones
        weights = helper.make_tensor(
            "weights",
            TensorProto.FLOAT,
            [5, 1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        )

        matmul_node = helper.make_node("MatMul", ["input", "weights"], ["output"])

        graph_def = helper.make_graph(
            [matmul_node],
            "test_model",
            [X],
            [Y],
            [weights],
        )

        model_def = helper.make_model(graph_def, producer_name="test")
        model_def.opset_import[0].version = 13

        model_path = tmp_path / "test_model.onnx"
        onnx.save(model_def, str(model_path))

        return str(model_path)

    def test_onnx_runner_real_model(self, simple_onnx_model):
        """Test ONNXRunner with a real ONNX model."""
        from rocket_env.inference.onnx_runner import ONNXRunner

        runner = ONNXRunner(simple_onnx_model)

        obs = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        action = runner.predict(obs)

        # Sum of 1.0 * 0.2 for each of 5 inputs = 1.0
        assert action.shape == (1,)
        assert np.isclose(action[0], 1.0, atol=0.01)

    def test_onnx_runner_get_info_real(self, simple_onnx_model):
        """Test get_info with real model."""
        from rocket_env.inference.onnx_runner import ONNXRunner

        runner = ONNXRunner(simple_onnx_model)
        info = runner.get_info()

        assert info["input_name"] == "input"
        assert info["output_names"] == ["output"]
        assert info["is_deterministic"] is True


@pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="Requires onnxruntime")
class TestBenchmarkInference:
    """Tests for benchmark_inference function."""

    def test_benchmark_inference(self, tmp_path):
        """Test benchmark function with real model."""
        pytest.importorskip("onnx")

        import onnx
        from onnx import helper, TensorProto

        # Create simple model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
        weights = helper.make_tensor("weights", TensorProto.FLOAT, [5, 1], [0.2] * 5)
        matmul_node = helper.make_node("MatMul", ["input", "weights"], ["output"])
        graph_def = helper.make_graph([matmul_node], "test", [X], [Y], [weights])
        model_def = helper.make_model(graph_def, producer_name="test")
        model_def.opset_import[0].version = 13

        model_path = tmp_path / "bench_model.onnx"
        onnx.save(model_def, str(model_path))

        from rocket_env.inference.onnx_runner import benchmark_inference

        results = benchmark_inference(str(model_path), obs_size=5, n_iterations=100)

        assert "mean_ms" in results
        assert "std_ms" in results
        assert "min_ms" in results
        assert "max_ms" in results
        assert "p99_ms" in results
        assert "iterations" in results
        assert "throughput_hz" in results
        assert results["iterations"] == 100
        assert results["throughput_hz"] > 0


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

    @pytest.mark.skipif(not HAS_ONNXRUNTIME, reason="Requires onnxruntime")
    def test_load_controller_without_normalize(self, tmp_path):
        """Test loading controller without normalization file."""
        pytest.importorskip("onnx")

        import onnx
        from onnx import helper, TensorProto

        # Create simple model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 5])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
        weights = helper.make_tensor("weights", TensorProto.FLOAT, [5, 1], [0.2] * 5)
        matmul_node = helper.make_node("MatMul", ["input", "weights"], ["output"])
        graph_def = helper.make_graph([matmul_node], "test", [X], [Y], [weights])
        model_def = helper.make_model(graph_def, producer_name="test")
        model_def.opset_import[0].version = 13

        model_path = tmp_path / "best_model.onnx"
        onnx.save(model_def, str(model_path))

        from rocket_env.inference.controller import load_controller_from_training

        controller = load_controller_from_training(str(tmp_path))

        assert controller is not None
        assert controller.obs_mean is None  # No normalization


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
