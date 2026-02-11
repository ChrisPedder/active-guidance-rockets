"""
Tests for training scripts: train_sac.py and train_residual_sac.py.

Covers: WindCurriculumCallback, MovingAverageEarlyStoppingCallback,
create_sac_environment, and import verification.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, PropertyMock

from rocket_config import load_config


class TestWindCurriculumCallback:
    """Test WindCurriculumCallback from train_sac."""

    def test_import(self):
        from training.train_sac import WindCurriculumCallback

    def test_init(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)
        assert cb.current_stage == 0
        assert len(cb.stages) == 4

    def test_stages_have_increasing_thresholds(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config)
        thresholds = [s[0] for s in cb.stages]
        assert thresholds == sorted(thresholds)

    def test_stages_have_increasing_wind(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config)
        winds = [s[1] for s in cb.stages]
        assert winds == sorted(winds)

    def test_on_step_returns_true(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)
        cb.num_timesteps = 0
        result = cb._on_step()
        assert result is True

    def test_stage_advancement(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        # Mock model for _update_wind
        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 0
        mock_vec_env.envs = []
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        # At step 0 we're in stage 0
        cb.num_timesteps = 0
        cb._on_step()
        assert cb.current_stage == 0

        # At step 300_001 should advance to stage 1
        cb.num_timesteps = 300_001
        cb._on_step()
        assert cb.current_stage == 1

        # At step 800_001 should advance to stage 2
        cb.num_timesteps = 800_001
        cb._on_step()
        assert cb.current_stage == 2

        # At step 1_500_001 should advance to stage 3
        cb.num_timesteps = 1_500_001
        cb._on_step()
        assert cb.current_stage == 3


class TestMovingAverageEarlyStopping:
    """Test MovingAverageEarlyStoppingCallback from train_residual_sac."""

    def test_import(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

    def test_init(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            window_size=20,
            max_no_improvement_evals=40,
            min_evals=20,
            verbose=0,
        )
        assert cb.window_size == 20
        assert cb.max_no_improvement_evals == 40
        assert cb.min_evals == 20
        assert cb.n_evals == 0
        assert cb.best_moving_avg == -np.inf

    def test_warmup_phase(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            min_evals=5, max_no_improvement_evals=10, verbose=0
        )

        # Mock parent
        mock_parent = MagicMock()
        mock_parent.last_mean_reward = 100.0
        cb.parent = mock_parent

        # During warmup, should always return True
        for _ in range(4):
            assert cb._on_step() is True
        assert cb.n_evals == 4

    def test_ema_tracking(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            min_evals=2, max_no_improvement_evals=100, verbose=0
        )

        mock_parent = MagicMock()
        cb.parent = mock_parent

        # Feed improving rewards
        for reward in [50.0, 60.0, 70.0, 80.0, 90.0]:
            mock_parent.last_mean_reward = reward
            result = cb._on_step()
            assert result is True

        assert cb.ema_reward is not None
        assert cb.ema_reward > 50.0  # EMA should have increased

    def test_early_stopping_triggers(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            min_evals=2, max_no_improvement_evals=3, verbose=0
        )

        mock_parent = MagicMock()
        cb.parent = mock_parent

        # Feed a good reward first, then decreasing
        rewards = [100.0, 100.0, 100.0, 50.0, 40.0, 30.0, 20.0, 10.0]
        results = []
        for r in rewards:
            mock_parent.last_mean_reward = r
            results.append(cb._on_step())

        # Should eventually return False (stopped)
        assert False in results

    def test_reset(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(min_evals=2, verbose=0)

        mock_parent = MagicMock()
        cb.parent = mock_parent
        mock_parent.last_mean_reward = 100.0

        # Accumulate some state
        cb._on_step()
        cb._on_step()
        cb._on_step()
        assert cb.n_evals == 3

        # Reset should clear state
        cb.reset()
        assert cb.n_evals == 0
        assert cb.best_moving_avg == -np.inf
        assert len(cb.eval_rewards) == 0
        assert cb.ema_reward is None


class TestCreateSACEnvironment:
    """Test create_sac_environment from train_sac."""

    def test_create_sac_environment_basic(self):
        from training.train_sac import create_sac_environment

        config = load_config("configs/estes_c6_sac_wind.yaml")
        env = create_sac_environment(config)
        assert env is not None

    def test_create_sac_environment_with_wind_override(self):
        from training.train_sac import create_sac_environment
        import copy

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config_copy = copy.deepcopy(config)
        env = create_sac_environment(
            config_copy,
            wind_override={"base_wind_speed": 0.0, "max_gust_speed": 0.0},
        )
        assert env is not None
        assert config_copy.physics.base_wind_speed == 0.0


class TestUpdateWind:
    """Test WindCurriculumCallback._update_wind method."""

    def test_update_wind_with_wind_model(self):
        """Lines 112-121: _update_wind when wind_model exists."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        # Create a mock env with wind_model
        mock_base_env = MagicMock()
        mock_base_env.wind_model = MagicMock()
        mock_base_env.wind_model.config = MagicMock()
        # No nested .env to unwrap
        del mock_base_env.env

        mock_env = MagicMock()
        mock_env.env = mock_base_env
        # mock_env has .env, but mock_base_env does not

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [mock_env]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(2.0, 1.5)
        assert mock_base_env.wind_model.config.base_speed == 2.0
        assert mock_base_env.wind_model.config.max_gust_speed == 1.5

    def test_update_wind_enable_wind(self):
        """Lines 122-136: _update_wind when wind_model is None but config exists."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        mock_base_env = MagicMock()
        mock_base_env.wind_model = None
        mock_base_env.config = MagicMock()
        mock_base_env.config.wind_variability = 0.3
        del mock_base_env.env

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [mock_base_env]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(3.0, 2.0)
        assert mock_base_env.config.enable_wind is True
        assert mock_base_env.config.base_wind_speed == 3.0
        assert mock_base_env.config.max_gust_speed == 2.0
        # wind_model should have been created
        assert mock_base_env.wind_model is not None


class TestTrainSacFunction:
    """Test train_sac function error paths."""

    def test_train_sac_no_sac_config(self):
        """Line 175-178: train_sac with no sac section returns None."""
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.sac = None
        result = train_sac(config)
        assert result is None

    def test_train_sac_critical_config_issue(self):
        """Lines 180-189: train_sac returns None on critical config issues."""
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        # Remove airframe file to cause critical validation error
        config.physics.airframe_file = None
        result = train_sac(config)
        assert result is None


class TestTrainResidualSacFunction:
    """Test train_residual_sac function error paths."""

    def test_train_residual_sac_no_sac_config(self):
        """train_residual_sac with no sac section returns None."""
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.sac = None
        result = train_residual_sac(config)
        assert result is None

    def test_train_residual_sac_no_residual_pid(self):
        """Lines 169-173: returns None when use_residual_pid is False."""
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.physics.use_residual_pid = False
        result = train_residual_sac(config)
        assert result is None


class TestTrainingScriptImports:
    """Verify all training script modules can be imported."""

    def test_train_sac_imports(self):
        from training.train_sac import (
            WindCurriculumCallback,
            create_sac_environment,
            train_sac,
        )

    def test_train_residual_sac_imports(self):
        from training.train_residual_sac import (
            MovingAverageEarlyStoppingCallback,
            train_residual_sac,
        )
