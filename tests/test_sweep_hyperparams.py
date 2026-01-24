"""
Tests for sweep_hyperparams module - hyperparameter sweep functionality.
"""
import pytest
import tempfile
from pathlib import Path
import json
import yaml


class TestGenerateSweepConfigs:
    """Tests for generate_sweep_configs function."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig

        # Use default config
        config = RocketTrainingConfig()
        config.motor.name = "estes_c6"
        return config

    def test_generate_reward_sweep(self, base_config):
        """Test generating reward sweep configurations."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("reward", base_config)

        assert len(sweeps) > 0
        # Should have spin/altitude sweeps and bonus sweeps
        spin_sweeps = [s for s in sweeps if 'spin' in s.get('name', '')]
        bonus_sweeps = [s for s in sweeps if 'bonus' in s.get('name', '')]

        assert len(spin_sweeps) > 0, "Should have spin penalty sweep configs"
        assert len(bonus_sweeps) > 0, "Should have bonus sweep configs"

    def test_generate_ppo_sweep(self, base_config):
        """Test generating PPO sweep configurations."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("ppo", base_config)

        assert len(sweeps) > 0
        # Should have LR/clip/batch sweeps and architecture sweeps
        lr_sweeps = [s for s in sweeps if 'lr' in s.get('name', '')]
        arch_sweeps = [s for s in sweeps if 'arch' in s.get('name', '')]

        assert len(lr_sweeps) > 0, "Should have learning rate sweep configs"
        assert len(arch_sweeps) > 0, "Should have architecture sweep configs"

    def test_generate_motors_sweep(self, base_config):
        """Test generating motors sweep configurations."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("motors", base_config)

        assert len(sweeps) > 0
        motor_sweeps = [s for s in sweeps if 'motor' in s.get('name', '')]
        assert len(motor_sweeps) > 0, "Should have motor sweep configs"

    def test_generate_quick_sweep(self, base_config):
        """Test generating quick sweep configurations."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("quick", base_config)

        assert len(sweeps) > 0
        assert len(sweeps) <= 10, "Quick sweep should have few configs"

        # Should have baseline
        names = [s.get('name', '') for s in sweeps]
        assert 'baseline' in names

    def test_unknown_sweep_type_error(self, base_config):
        """Test that unknown sweep type raises error."""
        from sweep_hyperparams import generate_sweep_configs

        with pytest.raises(ValueError, match="Unknown sweep type"):
            generate_sweep_configs("nonexistent", base_config)


class TestApplyConfigOverrides:
    """Tests for apply_config_overrides function."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig
        return RocketTrainingConfig()

    def test_apply_single_override(self, base_config):
        """Test applying a single override."""
        from sweep_hyperparams import apply_config_overrides

        overrides = {
            'name': 'test',
            'physics.max_tab_deflection': 20.0,
        }

        result = apply_config_overrides(base_config, overrides)

        assert result.physics.max_tab_deflection == 20.0
        # Original should be unchanged
        assert base_config.physics.max_tab_deflection == 15.0

    def test_apply_multiple_overrides(self, base_config):
        """Test applying multiple overrides."""
        from sweep_hyperparams import apply_config_overrides

        overrides = {
            'name': 'multi_test',
            'physics.disturbance_scale': 0.0005,
            'ppo.learning_rate': 1e-4,
            'reward.spin_penalty_scale': -0.2,
        }

        result = apply_config_overrides(base_config, overrides)

        assert result.physics.disturbance_scale == 0.0005
        assert result.ppo.learning_rate == 1e-4
        assert result.reward.spin_penalty_scale == -0.2

    def test_name_and_description_ignored(self, base_config):
        """Test that name and description keys are ignored."""
        from sweep_hyperparams import apply_config_overrides

        original_deflection = base_config.physics.max_tab_deflection

        overrides = {
            'name': 'test_name',
            'description': 'Test description',
        }

        result = apply_config_overrides(base_config, overrides)

        # Config should be unchanged
        assert result.physics.max_tab_deflection == original_deflection


class TestLoadSweepFromYaml:
    """Tests for load_sweep_from_yaml function."""

    def test_load_valid_yaml(self, tmp_path):
        """Test loading a valid sweep YAML file."""
        from sweep_hyperparams import load_sweep_from_yaml

        sweep_content = {
            'description': 'Test sweep',
            'sweeps': [
                {'name': 'config1', 'physics.dry_mass': 0.1},
                {'name': 'config2', 'physics.dry_mass': 0.2},
            ]
        }

        yaml_file = tmp_path / "test_sweep.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sweep_content, f)

        sweeps = load_sweep_from_yaml(str(yaml_file))

        assert len(sweeps) == 2
        assert sweeps[0]['name'] == 'config1'
        assert sweeps[1]['name'] == 'config2'

    def test_load_yaml_missing_sweeps(self, tmp_path):
        """Test loading YAML without sweeps key."""
        from sweep_hyperparams import load_sweep_from_yaml

        content = {'description': 'No sweeps here'}

        yaml_file = tmp_path / "empty.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(content, f)

        sweeps = load_sweep_from_yaml(str(yaml_file))

        assert sweeps == []


class TestRunSweepDryRun:
    """Tests for run_sweep in dry run mode."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig
        return RocketTrainingConfig()

    def test_run_sweep_dry_run(self, tmp_path, base_config):
        """Test running sweep in dry run mode."""
        from sweep_hyperparams import run_sweep

        sweep_configs = [
            {'name': 'test1', 'description': 'First test'},
            {'name': 'test2', 'description': 'Second test'},
        ]

        results = run_sweep(
            sweep_configs=sweep_configs,
            base_config=base_config,
            output_dir=str(tmp_path),
            dry_run=True,
        )

        # In dry run mode, no results are returned
        assert len(results) == 0

        # But sweep info should be saved
        assert (tmp_path / "sweep_info.json").exists()

        with open(tmp_path / "sweep_info.json") as f:
            info = json.load(f)

        assert info['num_configs'] == 2

    def test_run_sweep_creates_config_files(self, tmp_path, base_config):
        """Test that sweep creates config files."""
        from sweep_hyperparams import run_sweep

        sweep_configs = [
            {'name': 'config_a', 'physics.dry_mass': 0.1},
            {'name': 'config_b', 'physics.dry_mass': 0.2},
        ]

        run_sweep(
            sweep_configs=sweep_configs,
            base_config=base_config,
            output_dir=str(tmp_path),
            dry_run=True,
        )

        # Config files should be created
        assert (tmp_path / "config_a_config.yaml").exists()
        assert (tmp_path / "config_b_config.yaml").exists()


class TestSweepConfigContent:
    """Tests for sweep config content."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig
        return RocketTrainingConfig()

    def test_reward_sweep_has_valid_scales(self, base_config):
        """Test that reward sweep generates valid scale values."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("reward", base_config)

        for sweep in sweeps:
            if 'reward.spin_penalty_scale' in sweep:
                scale = sweep['reward.spin_penalty_scale']
                assert scale <= 0, f"Spin penalty {scale} should be negative"

            if 'reward.altitude_reward_scale' in sweep:
                scale = sweep['reward.altitude_reward_scale']
                assert scale > 0, f"Altitude reward {scale} should be positive"

    def test_ppo_sweep_has_valid_lr(self, base_config):
        """Test that PPO sweep generates valid learning rates."""
        from sweep_hyperparams import generate_sweep_configs

        sweeps = generate_sweep_configs("ppo", base_config)

        for sweep in sweeps:
            if 'ppo.learning_rate' in sweep:
                lr = sweep['ppo.learning_rate']
                assert 1e-5 < lr < 1e-2, f"LR {lr} should be reasonable"


class TestSweepConfigDescriptions:
    """Tests for sweep config descriptions."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig
        return RocketTrainingConfig()

    def test_sweeps_have_descriptions(self, base_config):
        """Test that sweep configs have descriptions."""
        from sweep_hyperparams import generate_sweep_configs

        # Only test sweep types that don't require motor specs
        for sweep_type in ['reward', 'ppo', 'motors', 'quick']:
            sweeps = generate_sweep_configs(sweep_type, base_config)

            for sweep in sweeps:
                assert 'name' in sweep, f"Sweep should have name: {sweep}"
                assert 'description' in sweep, f"Sweep should have description: {sweep}"


class TestSweepIntegration:
    """Integration tests for sweep functionality."""

    @pytest.fixture
    def base_config(self):
        """Create a base config for testing."""
        from rocket_config import RocketTrainingConfig
        return RocketTrainingConfig()

    def test_apply_and_save_configs(self, tmp_path, base_config):
        """Test applying overrides and saving configs."""
        from sweep_hyperparams import generate_sweep_configs, apply_config_overrides

        sweeps = generate_sweep_configs("quick", base_config)

        for sweep in sweeps:
            config = apply_config_overrides(base_config, sweep)

            # Should be saveable
            config_path = tmp_path / f"{sweep['name']}.yaml"
            config.save(config_path)

            assert config_path.exists()
