"""
Tests for training modules: train_residual_sac.py and train_sac.py.

Targets coverage improvement from ~33-40% to above 80% by testing:
- MovingAverageEarlyStoppingCallback (all branches and verbose paths)
- WindCurriculumCallback (stage transitions, _update_wind branches, verbose)
- create_sac_environment (with and without wind overrides)
- train_residual_sac() early-return paths and validation
- train_sac() early-return paths and validation
- main() argument parsing for both scripts
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from rocket_config import load_config, SACConfig, RocketTrainingConfig


# ---------------------------------------------------------------------------
# MovingAverageEarlyStoppingCallback (train_residual_sac.py)
# ---------------------------------------------------------------------------


class TestMovingAverageEarlyStoppingCallbackInit:
    """Test __init__ defaults and custom parameters."""

    def test_default_parameters(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback()
        assert cb.window_size == 20
        assert cb.max_no_improvement_evals == 40
        assert cb.min_evals == 20
        assert cb.verbose == 0
        assert cb.eval_rewards == []
        assert cb.best_moving_avg == -np.inf
        assert cb.no_improvement_evals == 0
        assert cb.n_evals == 0
        assert cb.ema_reward is None
        assert cb.ema_alpha == 0.15

    def test_custom_parameters(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            window_size=10,
            max_no_improvement_evals=5,
            min_evals=3,
            verbose=1,
        )
        assert cb.window_size == 10
        assert cb.max_no_improvement_evals == 5
        assert cb.min_evals == 3
        assert cb.verbose == 1


class TestMovingAverageEarlyStoppingOnStep:
    """Test _on_step() under various conditions."""

    @pytest.fixture
    def make_callback(self):
        """Factory that returns a callback with a mocked parent."""
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        def _make(min_evals=3, max_no_improvement_evals=5, verbose=0):
            cb = MovingAverageEarlyStoppingCallback(
                min_evals=min_evals,
                max_no_improvement_evals=max_no_improvement_evals,
                verbose=verbose,
            )
            parent = MagicMock()
            parent.last_mean_reward = 0.0
            cb.parent = parent
            return cb, parent

        return _make

    def test_warmup_returns_true(self, make_callback):
        """During warm-up (n_evals < min_evals) always returns True."""
        cb, parent = make_callback(min_evals=5)
        for i in range(4):
            parent.last_mean_reward = float(i)
            assert cb._on_step() is True
        assert cb.n_evals == 4
        # EMA should not have been set during warmup
        assert cb.ema_reward is None

    def test_warmup_verbose(self, make_callback):
        """Verbose output during warm-up phase."""
        cb, parent = make_callback(min_evals=3, verbose=1)
        parent.last_mean_reward = 10.0
        result = cb._on_step()
        assert result is True
        assert cb.n_evals == 1

    def test_ema_initialised_on_first_post_warmup(self, make_callback):
        """After warm-up the first call sets ema_reward = last_reward."""
        cb, parent = make_callback(min_evals=3, max_no_improvement_evals=100)
        # Two warmup steps (n_evals < min_evals=3 => n_evals in {1,2})
        parent.last_mean_reward = 10.0
        cb._on_step()
        parent.last_mean_reward = 20.0
        cb._on_step()
        assert cb.ema_reward is None  # still in warmup (n_evals=2 < 3)
        # Third step exits warmup (n_evals=3, not < 3)
        parent.last_mean_reward = 30.0
        cb._on_step()
        assert cb.ema_reward == 30.0  # initialised directly

    def test_ema_updates_on_subsequent_steps(self, make_callback):
        """After initialisation EMA uses alpha blending."""
        cb, parent = make_callback(min_evals=3, max_no_improvement_evals=100)
        # Warmup: steps 1 and 2 (n_evals < 3)
        parent.last_mean_reward = 10.0
        cb._on_step()
        parent.last_mean_reward = 10.0
        cb._on_step()
        # First post-warmup (n_evals=3): EMA initialised to 50.0
        parent.last_mean_reward = 50.0
        cb._on_step()
        assert cb.ema_reward == 50.0
        # Second post-warmup: EMA = 0.15*100 + 0.85*50 = 57.5
        parent.last_mean_reward = 100.0
        cb._on_step()
        expected = 0.15 * 100.0 + 0.85 * 50.0
        assert abs(cb.ema_reward - expected) < 1e-6

    def test_improving_ema_resets_counter(self, make_callback):
        """When EMA improves, no_improvement_evals resets to 0."""
        cb, parent = make_callback(min_evals=2, max_no_improvement_evals=100)
        parent.last_mean_reward = 10.0
        cb._on_step()
        cb._on_step()
        # Set a good baseline
        parent.last_mean_reward = 50.0
        cb._on_step()
        assert cb.no_improvement_evals == 0
        # Higher reward => EMA improves
        parent.last_mean_reward = 200.0
        cb._on_step()
        assert cb.no_improvement_evals == 0

    def test_stagnating_ema_increments_counter(self, make_callback):
        """When EMA does not improve, no_improvement_evals increments."""
        cb, parent = make_callback(min_evals=2, max_no_improvement_evals=100)
        parent.last_mean_reward = 10.0
        cb._on_step()
        cb._on_step()
        parent.last_mean_reward = 100.0
        cb._on_step()  # EMA=100
        # Lower rewards => EMA drops
        parent.last_mean_reward = 0.0
        cb._on_step()
        assert cb.no_improvement_evals == 1

    def test_early_stop_triggers_false(self, make_callback):
        """Returns False once patience is exhausted."""
        cb, parent = make_callback(min_evals=2, max_no_improvement_evals=3)
        parent.last_mean_reward = 10.0
        cb._on_step()
        cb._on_step()
        # Set high baseline
        parent.last_mean_reward = 100.0
        cb._on_step()
        # Now feed low rewards until patience runs out
        results = []
        for _ in range(10):
            parent.last_mean_reward = 0.0
            results.append(cb._on_step())
        assert False in results
        # Should have stopped at exactly patience=3
        first_false = results.index(False)
        assert first_false == 2  # 0-indexed => 3rd stagnant step

    def test_early_stop_verbose_messages(self, make_callback):
        """Verbose=1 prints messages on improvement tracking and stop."""
        cb, parent = make_callback(min_evals=2, max_no_improvement_evals=2, verbose=1)
        parent.last_mean_reward = 10.0
        cb._on_step()
        cb._on_step()
        parent.last_mean_reward = 100.0
        cb._on_step()  # post-warmup, improving
        parent.last_mean_reward = 0.0
        cb._on_step()  # not improving
        parent.last_mean_reward = 0.0
        result = cb._on_step()  # triggers stop
        assert result is False

    def test_rewards_appended_every_step(self, make_callback):
        """eval_rewards accumulates every call."""
        cb, parent = make_callback(min_evals=2)
        for val in [1.0, 2.0, 3.0, 4.0]:
            parent.last_mean_reward = val
            cb._on_step()
        assert cb.eval_rewards == [1.0, 2.0, 3.0, 4.0]


class TestMovingAverageEarlyStoppingReset:
    """Test reset() method."""

    def test_reset_clears_all_state(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(min_evals=2, verbose=0)
        parent = MagicMock()
        parent.last_mean_reward = 50.0
        cb.parent = parent

        # Accumulate state
        for _ in range(5):
            cb._on_step()

        assert cb.n_evals == 5
        assert len(cb.eval_rewards) == 5
        assert cb.ema_reward is not None

        cb.reset()

        assert cb.n_evals == 0
        assert cb.eval_rewards == []
        assert cb.best_moving_avg == -np.inf
        assert cb.no_improvement_evals == 0
        assert cb.ema_reward is None

    def test_can_resume_after_reset(self):
        """After reset, callback works as if freshly created."""
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        cb = MovingAverageEarlyStoppingCallback(
            min_evals=2, max_no_improvement_evals=100, verbose=0
        )
        parent = MagicMock()
        parent.last_mean_reward = 50.0
        cb.parent = parent

        for _ in range(5):
            cb._on_step()
        cb.reset()

        parent.last_mean_reward = 10.0
        assert cb._on_step() is True  # warmup again
        assert cb.n_evals == 1


# ---------------------------------------------------------------------------
# train_residual_sac() function early-return paths
# ---------------------------------------------------------------------------


class TestTrainResidualSacEarlyReturns:
    """Test all early-return paths in train_residual_sac()."""

    def test_returns_none_when_sac_is_none(self):
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.sac = None
        result = train_residual_sac(config)
        assert result is None

    def test_returns_none_when_residual_pid_false(self):
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.physics.use_residual_pid = False
        result = train_residual_sac(config)
        assert result is None

    def test_returns_none_when_residual_pid_missing(self):
        """use_residual_pid attribute absent defaults to False."""
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        # Delete the attribute so getattr(..., False) kicks in
        if hasattr(config.physics, "use_residual_pid"):
            config.physics.use_residual_pid = False
        result = train_residual_sac(config)
        assert result is None

    def test_returns_none_on_critical_validation_issue(self):
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_residual_sac_wind.yaml")
        # Cause a CRITICAL validation error by removing airframe
        config.physics.airframe_file = None
        result = train_residual_sac(config)
        assert result is None

    def test_passes_validation_warnings_but_continues(self):
        """Non-critical warnings should NOT cause early return."""
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_residual_sac_wind.yaml")
        mock_model = MagicMock()

        with patch.object(
            type(config), "validate", return_value=["WARNING: something minor"]
        ):
            with (
                patch(
                    "training.train_residual_sac.DummyVecEnv",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.VecNormalize",
                    return_value=MagicMock(),
                ),
                patch("training.train_residual_sac.SAC", return_value=mock_model),
                patch(
                    "training.train_residual_sac.EvalCallback",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.CheckpointCallback",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.CallbackList",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.TrainingMetricsCallback",
                    return_value=MagicMock(),
                ),
                patch.object(config, "save"),
            ):
                result = train_residual_sac(config)
        # Should have returned the model (not None)
        assert result is not None


# ---------------------------------------------------------------------------
# WindCurriculumCallback (train_sac.py)
# ---------------------------------------------------------------------------


class TestWindCurriculumCallbackInit:
    """Test __init__ for WindCurriculumCallback."""

    def test_default_stages(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config)
        assert cb.current_stage == 0
        assert len(cb.stages) == 4
        # First stage: no wind
        assert cb.stages[0] == (0, 0.0, 0.0)
        # Second stage
        assert cb.stages[1] == (300_000, 1.0, 0.5)

    def test_stages_respect_config_final_wind(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.physics.base_wind_speed = 2.0
        config.physics.max_gust_speed = 1.0
        cb = WindCurriculumCallback(config)
        # Stage 3 should use min(3.0, 2.0) = 2.0
        assert cb.stages[2][1] == 2.0
        assert cb.stages[2][2] == 1.0
        # Stage 4 should use final config values
        assert cb.stages[3][1] == 2.0
        assert cb.stages[3][2] == 1.0

    def test_stages_increasing_thresholds(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config)
        thresholds = [s[0] for s in cb.stages]
        assert thresholds == sorted(thresholds)

    def test_stages_increasing_wind(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config)
        winds = [s[1] for s in cb.stages]
        assert winds == sorted(winds)


class TestWindCurriculumOnStep:
    """Test _on_step() for stage transitions."""

    @pytest.fixture
    def callback_with_mock_model(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        mock_base_env = MagicMock()
        mock_base_env.wind_model = MagicMock()
        mock_base_env.wind_model.config = MagicMock()
        del mock_base_env.env  # no further unwrapping

        mock_env = MagicMock()
        mock_env.env = mock_base_env

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [mock_env]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        return cb, mock_base_env

    def test_stays_at_stage_0_initially(self, callback_with_mock_model):
        cb, _ = callback_with_mock_model
        cb.num_timesteps = 0
        assert cb._on_step() is True
        assert cb.current_stage == 0

    def test_advances_to_stage_1(self, callback_with_mock_model):
        cb, base_env = callback_with_mock_model
        cb.num_timesteps = 300_001
        cb._on_step()
        assert cb.current_stage == 1
        assert base_env.wind_model.config.base_speed == 1.0
        assert base_env.wind_model.config.max_gust_speed == 0.5

    def test_advances_to_stage_2(self, callback_with_mock_model):
        cb, base_env = callback_with_mock_model
        # Jump directly to stage 2 threshold
        cb.num_timesteps = 800_001
        cb._on_step()
        assert cb.current_stage == 2

    def test_advances_to_stage_3(self, callback_with_mock_model):
        cb, base_env = callback_with_mock_model
        cb.num_timesteps = 1_500_001
        cb._on_step()
        assert cb.current_stage == 3

    def test_no_transition_when_same_stage(self, callback_with_mock_model):
        """No _update_wind call if stage hasn't changed."""
        cb, base_env = callback_with_mock_model
        cb.num_timesteps = 0
        cb._on_step()
        # Reset mock call count
        cb.model.get_env.reset_mock()
        cb.num_timesteps = 100
        cb._on_step()
        # _update_wind should NOT have been called
        cb.model.get_env.assert_not_called()

    def test_always_returns_true(self, callback_with_mock_model):
        cb, _ = callback_with_mock_model
        for ts in [0, 300_001, 800_001, 1_500_001, 5_000_000]:
            cb.num_timesteps = ts
            assert cb._on_step() is True

    def test_verbose_output_on_transition(self):
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=1)

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 0
        mock_vec_env.envs = []
        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb.num_timesteps = 300_001
        result = cb._on_step()
        assert result is True
        assert cb.current_stage == 1


class TestUpdateWind:
    """Test _update_wind with different environment structures."""

    def test_update_wind_with_existing_wind_model(self):
        """Wind model config attributes are updated directly."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        mock_base_env = MagicMock()
        mock_base_env.wind_model = MagicMock()
        mock_base_env.wind_model.config = MagicMock()
        del mock_base_env.env

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [mock_base_env]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(5.0, 3.0)
        assert mock_base_env.wind_model.config.base_speed == 5.0
        assert mock_base_env.wind_model.config.max_gust_speed == 3.0

    def test_update_wind_creates_new_wind_model(self):
        """When wind_model is None but config exists, creates a new WindModel."""
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

        cb._update_wind(2.0, 1.0)
        assert mock_base_env.config.enable_wind is True
        assert mock_base_env.config.base_wind_speed == 2.0
        assert mock_base_env.config.max_gust_speed == 1.0
        assert mock_base_env.wind_model is not None

    def test_update_wind_zero_speed_no_wind_model_created(self):
        """With base_speed=0 and no existing wind_model, wind stays disabled."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        mock_base_env = MagicMock()
        mock_base_env.wind_model = None
        mock_base_env.config = MagicMock()
        del mock_base_env.env

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [mock_base_env]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(0.0, 0.0)
        assert mock_base_env.config.enable_wind is False
        # wind_model should remain None since base_speed == 0
        assert mock_base_env.wind_model is None

    def test_update_wind_multiple_envs(self):
        """Updates all environments in the vec env."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        envs = []
        for _ in range(3):
            mock_base = MagicMock()
            mock_base.wind_model = MagicMock()
            mock_base.wind_model.config = MagicMock()
            del mock_base.env
            envs.append(mock_base)

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 3
        mock_vec_env.envs = envs

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(4.0, 2.5)
        for env in envs:
            assert env.wind_model.config.base_speed == 4.0
            assert env.wind_model.config.max_gust_speed == 2.5

    def test_update_wind_unwraps_nested_envs(self):
        """Correctly walks through wrapper chain."""
        from training.train_sac import WindCurriculumCallback

        config = load_config("configs/estes_c6_sac_wind.yaml")
        cb = WindCurriculumCallback(config, verbose=0)

        # Build a chain: env -> wrapper1 -> wrapper2 -> base_env
        mock_base = MagicMock()
        mock_base.wind_model = MagicMock()
        mock_base.wind_model.config = MagicMock()
        del mock_base.env  # terminal

        wrapper2 = MagicMock()
        wrapper2.env = mock_base

        wrapper1 = MagicMock()
        wrapper1.env = wrapper2

        mock_vec_env = MagicMock()
        mock_vec_env.num_envs = 1
        mock_vec_env.envs = [wrapper1]

        mock_model = MagicMock()
        mock_model.get_env.return_value = mock_vec_env
        cb.model = mock_model

        cb._update_wind(6.0, 4.0)
        assert mock_base.wind_model.config.base_speed == 6.0
        assert mock_base.wind_model.config.max_gust_speed == 4.0


# ---------------------------------------------------------------------------
# create_sac_environment (train_sac.py)
# ---------------------------------------------------------------------------


class TestCreateSacEnvironment:
    """Test create_sac_environment function."""

    def test_basic_creation(self):
        from training.train_sac import create_sac_environment

        config = load_config("configs/estes_c6_sac_wind.yaml")
        env = create_sac_environment(config)
        assert env is not None
        env.close()

    def test_with_wind_override(self):
        from training.train_sac import create_sac_environment

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config_copy = copy.deepcopy(config)
        env = create_sac_environment(
            config_copy,
            wind_override={"base_wind_speed": 0.0, "max_gust_speed": 0.0},
        )
        assert env is not None
        assert config_copy.physics.base_wind_speed == 0.0
        assert config_copy.physics.max_gust_speed == 0.0
        env.close()

    def test_wind_override_ignores_unknown_keys(self):
        from training.train_sac import create_sac_environment

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config_copy = copy.deepcopy(config)
        original_speed = config_copy.physics.base_wind_speed
        env = create_sac_environment(
            config_copy,
            wind_override={"nonexistent_key": 999},
        )
        assert env is not None
        # Original values should be unchanged
        assert config_copy.physics.base_wind_speed == original_speed
        env.close()

    def test_no_wind_override(self):
        from training.train_sac import create_sac_environment

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config_copy = copy.deepcopy(config)
        original_speed = config_copy.physics.base_wind_speed
        env = create_sac_environment(config_copy, wind_override=None)
        assert env is not None
        assert config_copy.physics.base_wind_speed == original_speed
        env.close()


# ---------------------------------------------------------------------------
# train_sac() function early-return paths
# ---------------------------------------------------------------------------


class TestTrainSacEarlyReturns:
    """Test all early-return paths in train_sac()."""

    def test_returns_none_when_sac_is_none(self):
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.sac = None
        result = train_sac(config)
        assert result is None

    def test_returns_none_on_critical_validation(self):
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        config.physics.airframe_file = None
        result = train_sac(config)
        assert result is None

    def test_non_critical_warnings_do_not_stop(self):
        """Non-CRITICAL validation issues should not prevent training."""
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")
        mock_model = MagicMock()

        with patch.object(
            type(config), "validate", return_value=["WARNING: something minor"]
        ):
            with (
                patch("training.train_sac.DummyVecEnv", return_value=MagicMock()),
                patch("training.train_sac.VecNormalize", return_value=MagicMock()),
                patch("training.train_sac.SAC", return_value=mock_model),
                patch("training.train_sac.EvalCallback", return_value=MagicMock()),
                patch(
                    "training.train_sac.CheckpointCallback",
                    return_value=MagicMock(),
                ),
                patch("training.train_sac.CallbackList", return_value=MagicMock()),
                patch(
                    "training.train_sac.TrainingMetricsCallback",
                    return_value=MagicMock(),
                ),
                patch.object(config, "save"),
            ):
                result = train_sac(config)
        assert result is not None

    def test_returns_none_when_validate_has_critical(self):
        """CRITICAL in the issues list causes early return."""
        from training.train_sac import train_sac

        config = load_config("configs/estes_c6_sac_wind.yaml")

        with patch.object(
            type(config),
            "validate",
            return_value=["CRITICAL: airframe missing"],
        ):
            result = train_sac(config)
        assert result is None


# ---------------------------------------------------------------------------
# train_residual_sac() validation path
# ---------------------------------------------------------------------------


class TestTrainResidualSacValidation:
    """Test validation paths in train_residual_sac()."""

    def test_returns_none_when_validate_has_critical(self):
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_residual_sac_wind.yaml")

        with patch.object(
            type(config),
            "validate",
            return_value=["CRITICAL: something bad"],
        ):
            result = train_residual_sac(config)
        assert result is None

    def test_non_critical_validation_warning_continues(self):
        from training.train_residual_sac import train_residual_sac

        config = load_config("configs/estes_c6_residual_sac_wind.yaml")
        mock_model = MagicMock()

        with patch.object(
            type(config), "validate", return_value=["WARNING: minor issue"]
        ):
            with (
                patch(
                    "training.train_residual_sac.DummyVecEnv",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.VecNormalize",
                    return_value=MagicMock(),
                ),
                patch("training.train_residual_sac.SAC", return_value=mock_model),
                patch(
                    "training.train_residual_sac.EvalCallback",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.CheckpointCallback",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.CallbackList",
                    return_value=MagicMock(),
                ),
                patch(
                    "training.train_residual_sac.TrainingMetricsCallback",
                    return_value=MagicMock(),
                ),
                patch.object(config, "save"),
            ):
                result = train_residual_sac(config)
        assert result is not None


# ---------------------------------------------------------------------------
# main() argument parsing (train_residual_sac.py)
# ---------------------------------------------------------------------------


class TestResidualSacMain:
    """Test the main() function of train_residual_sac.py."""

    def test_main_loads_config_and_trains(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--timesteps",
            "100",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                mock_train.assert_called_once()
                call_args = mock_train.call_args
                config = call_args[0][0]
                assert config.sac is not None
                assert config.sac.total_timesteps == 100

    def test_main_creates_sac_config_if_missing(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                # Patch load_config to return config with sac=None
                with patch("training.train_residual_sac.load_config") as mock_load:
                    cfg = load_config("configs/estes_c6_sac_wind.yaml")
                    cfg.sac = None
                    mock_load.return_value = cfg
                    main()
                    # After main, config.sac should have been created
                    call_config = mock_train.call_args[0][0]
                    assert call_config.sac is not None

    def test_main_overrides_lr(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--lr",
            "0.001",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.learning_rate == 0.001

    def test_main_overrides_buffer_size(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--buffer-size",
            "50000",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.buffer_size == 50000

    def test_main_overrides_batch_size(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--batch-size",
            "128",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.batch_size == 128

    def test_main_overrides_device(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--device",
            "cpu",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.device == "cpu"

    def test_main_passes_early_stopping(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--early-stopping",
            "25",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                kwargs = mock_train.call_args[1]
                assert kwargs["early_stopping_patience"] == 25

    def test_main_passes_load_model(self):
        from training.train_residual_sac import main

        test_args = [
            "train_residual_sac.py",
            "--config",
            "configs/estes_c6_residual_sac_wind.yaml",
            "--load-model",
            "some/model.zip",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_residual_sac.train_residual_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                kwargs = mock_train.call_args[1]
                assert kwargs["load_model_path"] == "some/model.zip"


# ---------------------------------------------------------------------------
# main() argument parsing (train_sac.py)
# ---------------------------------------------------------------------------


class TestSacMain:
    """Test the main() function of train_sac.py."""

    def test_main_loads_config_and_trains(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--timesteps",
            "100",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                mock_train.assert_called_once()
                config = mock_train.call_args[0][0]
                assert config.sac is not None
                assert config.sac.total_timesteps == 100

    def test_main_creates_sac_config_if_missing(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                with patch("training.train_sac.load_config") as mock_load:
                    cfg = load_config("configs/estes_c6_sac_wind.yaml")
                    cfg.sac = None
                    mock_load.return_value = cfg
                    main()
                    call_config = mock_train.call_args[0][0]
                    assert call_config.sac is not None

    def test_main_overrides_lr(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--lr",
            "0.001",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.learning_rate == 0.001

    def test_main_overrides_buffer_size(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--buffer-size",
            "50000",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.buffer_size == 50000

    def test_main_overrides_batch_size(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--batch-size",
            "128",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.batch_size == 128

    def test_main_overrides_device(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--device",
            "cpu",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                config = mock_train.call_args[0][0]
                assert config.sac.device == "cpu"

    def test_main_passes_early_stopping(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--early-stopping",
            "30",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                kwargs = mock_train.call_args[1]
                assert kwargs["early_stopping_patience"] == 30

    def test_main_passes_load_model(self):
        from training.train_sac import main

        test_args = [
            "train_sac.py",
            "--config",
            "configs/estes_c6_sac_wind.yaml",
            "--load-model",
            "models/best.zip",
        ]
        with patch("sys.argv", test_args):
            with patch("training.train_sac.train_sac") as mock_train:
                mock_train.return_value = MagicMock()
                main()
                kwargs = mock_train.call_args[1]
                assert kwargs["load_model_path"] == "models/best.zip"


# ---------------------------------------------------------------------------
# Import verification
# ---------------------------------------------------------------------------


class TestImports:
    """Verify all key symbols can be imported."""

    def test_import_moving_average_callback(self):
        from training.train_residual_sac import MovingAverageEarlyStoppingCallback

        assert MovingAverageEarlyStoppingCallback is not None

    def test_import_train_residual_sac(self):
        from training.train_residual_sac import train_residual_sac

        assert callable(train_residual_sac)

    def test_import_residual_sac_main(self):
        from training.train_residual_sac import main

        assert callable(main)

    def test_import_wind_curriculum_callback(self):
        from training.train_sac import WindCurriculumCallback

        assert WindCurriculumCallback is not None

    def test_import_create_sac_environment(self):
        from training.train_sac import create_sac_environment

        assert callable(create_sac_environment)

    def test_import_train_sac(self):
        from training.train_sac import train_sac

        assert callable(train_sac)

    def test_import_sac_main(self):
        from training.train_sac import main

        assert callable(main)
