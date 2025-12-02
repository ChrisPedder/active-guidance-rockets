# Active Guidance Rockets

Reinforcement learning for active spin control of model rockets using control surface tabs on fins.

## Overview

This project trains RL agents to stabilize rocket roll (spin) during flight using small deflectable tabs on the fins. The goal is to maintain stable camera footage from an onboard horizontal camera by minimizing spin rate.

## Quick Start

```bash
# 1. Setup environment
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"

# 2. Generate config and run training (recommended)
./run_experiment.sh --motor estes_c6 --difficulty easy --generate-config --timesteps 500000

# Or manually:
# 3. Generate motor config
uv run python generate_motor_config.py generate estes_c6 --output configs/

# 4. Run training
uv run python train_improved.py --config configs/estes_c6_easy.yaml

# 5. Visualize results
uv run python visualize_spin_agent.py models/best_model.zip --n-episodes 50
```

---

## Installation

### Requirements
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with all dependencies
uv pip install -e ".[all,dev]"

# Optional: Install requests for ThrustCurve.org API access
pip install requests
```

---

## Motor Config Generator

The `generate_motor_config.py` script automatically creates properly-tuned training configurations from motor specifications. This ensures configs are physics-appropriate for each motor's characteristics.

### Why Use the Config Generator?

Different motors have vastly different characteristics. A config tuned for Estes C6 (5N avg thrust) won't work for an Aerotech H128 (128N avg thrust). The generator:

1. **Analyzes motor physics** - thrust, impulse, burn time
2. **Calculates safe parameters** - tab deflection, damping, timestep
3. **Auto-sizes the rocket** - dry mass for target TWR
4. **Creates difficulty progression** - easy → medium → full configs

### Commands

```bash
# List motors in offline database (works without internet)
uv run python generate_motor_config.py list-popular

# Verify a motor exists
uv run python generate_motor_config.py verify "Estes C6"
uv run python generate_motor_config.py verify "Aerotech H128"

# Search ThrustCurve.org (requires 'requests' library)
uv run python generate_motor_config.py search "F40"
uv run python generate_motor_config.py search --impulse-class G
uv run python generate_motor_config.py search --manufacturer AeroTech

# Generate configs for a motor
uv run python generate_motor_config.py generate estes_c6
uv run python generate_motor_config.py generate aerotech_f40 --output configs/
uv run python generate_motor_config.py generate cesaroni_g79 --difficulty easy --dry-mass 0.8
```

### Available Offline Motors

| Key | Motor | Class | Impulse | Avg Thrust |
|-----|-------|-------|---------|------------|
| `estes_a8` | Estes A8 | A | 2.5 N·s | 5 N |
| `estes_b6` | Estes B6 | B | 5.0 N·s | 6 N |
| `estes_c6` | Estes C6 | C | 10 N·s | 5.4 N |
| `estes_d12` | Estes D12 | D | 20 N·s | 12.5 N |
| `aerotech_f40` | Aerotech F40 | F | 80 N·s | 40 N |
| `aerotech_h128` | Aerotech H128 | H | 219 N·s | 128 N |
| `cesaroni_g79` | Cesaroni G79 | G | 130 N·s | 79 N |

### Example: Adding a New Motor

```bash
# 1. Verify the motor exists
uv run python generate_motor_config.py verify "Aerotech H128"

# 2. Generate all difficulty configs
uv run python generate_motor_config.py generate aerotech_h128

# 3. Review generated config
cat configs/aerotech_h128_easy.yaml

# 4. Train with the new motor
./run_experiment.sh --motor aerotech_h128 --difficulty easy --timesteps 500000
```

### Physics-Based Parameter Calculation

The generator calculates parameters based on motor characteristics:

| Motor | Class | Tab Deflection | Damping | Notes |
|-------|-------|----------------|---------|-------|
| Estes C6 | C | 1.2° | 3.0 | Low impulse, small rocket |
| Aerotech F40 | F | 3.0° | 3.0 | High velocity motor |
| Cesaroni G79 | G | 1.3° | 3.0 | Very high velocity |
| Aerotech H128 | H | 5.0° | 3.0 | Heavy rocket = higher inertia |

**Key insight:** Larger/faster motors often need *lower* control authority because aerodynamic forces scale with velocity squared.

---

## Complete Workflow

### Automated Pipeline (Recommended)

Use the included shell script to run the complete training and evaluation pipeline:

```bash
# Make script executable
chmod +x run_experiment.sh

# Basic training with built-in motor
./run_experiment.sh --motor estes_c6 --difficulty easy --timesteps 500000

# Auto-generate config for any motor
./run_experiment.sh --motor aerotech_h128 --generate-config --timesteps 500000

# Generate config with custom dry mass
./run_experiment.sh --motor cesaroni_g79 --generate-config --dry-mass 0.9

# Generate all difficulty levels at once
./run_experiment.sh --motor aerotech_f40 --generate-config --difficulty all

# Quick test run
./run_experiment.sh --motor estes_c6 --difficulty easy --timesteps 50000

# Evaluate existing model only
./run_experiment.sh --eval-only --model-path models/best_model.zip --config configs/estes_c6_easy.yaml
```

The pipeline automates:
0. **Config Generation** - Create physics-tuned config from motor data (optional)
1. **Motor Visualization** - Plot thrust curves and characteristics
2. **Environment Diagnostics** - Verify configuration
3. **Training** - Run PPO training
4. **Evaluation** - Test trained agent
5. **Reporting** - Generate plots and statistics

### Pipeline Options

```
MOTOR OPTIONS:
    -m, --motor MOTOR       Motor name (e.g., estes_c6, aerotech_h128)
    -d, --difficulty LEVEL  easy, medium, full, or all
    
CONFIG OPTIONS:
    -g, --generate-config   Auto-generate config if not found
    --dry-mass KG           Override dry mass for config generation
    -c, --config FILE       Use specific config file
    
TRAINING OPTIONS:
    -t, --timesteps N       Training timesteps (default: 500000)
    -n, --n-envs N          Parallel environments (default: 8)
    --eval-only             Skip training
    --model-path PATH       Existing model for evaluation
    
OUTPUT OPTIONS:
    -o, --output-dir DIR    Results directory (default: experiments/)
    -e, --eval-episodes N   Evaluation episodes (default: 50)
    --skip-motor-viz        Skip motor visualization
    --skip-eval             Skip evaluation
```

### Manual Step-by-Step

#### Step 1: Generate Motor Config

```bash
# Generate configs for your motor
uv run python generate_motor_config.py generate estes_c6 --output configs/

# Review the physics analysis
# The script will show:
# - Motor specifications
# - Recommended rocket configuration  
# - Physics analysis (control authority, disturbance levels)
# - Generated config files
```

#### Step 2: Visualize Motor Characteristics

```bash
# Single motor profile
uv run python visualize_motor.py --motor estes_c6 --save motor_c6.png

# Compare all motors
uv run python visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79
```

#### Step 3: Train the Agent

```bash
# Start with easy config (recommended)
uv run python train_improved.py --config configs/estes_c6_easy.yaml

# With custom parameters
uv run python train_improved.py --config configs/estes_c6_easy.yaml \
    --timesteps 500000 \
    --n-envs 8
```

#### Step 4: Evaluate and Visualize

```bash
# Run evaluation with visualizations
uv run python visualize_spin_agent.py models/best_model.zip \
    --config configs/estes_c6_easy.yaml \
    --n-episodes 100 \
    --save-dir evaluation_results/
```

---

## Training

### Recommended: Start with Easy Config

For successful training, **start with the easy configuration** which uses reduced control authority:

```bash
# Generate easy config and train
./run_experiment.sh --motor estes_c6 --generate-config --difficulty easy

# Or manually
uv run python generate_motor_config.py generate estes_c6 --difficulty easy
uv run python train_improved.py --config configs/estes_c6_easy.yaml
```

This config is tuned so that even random actions from an untrained agent don't immediately destabilize the rocket.

### Training Progression Strategy

| Phase | Config | Timesteps | What Agent Learns |
|-------|--------|-----------|-------------------|
| 1. Easy | `*_easy.yaml` | 500k | Basic stabilization |
| 2. Medium | `*_medium.yaml` | 500k | Stronger control inputs |
| 3. Full | `*_full.yaml` | 500k | Full difficulty with wind |

```bash
# Phase 1: Easy start
./run_experiment.sh --motor estes_c6 --generate-config --difficulty easy

# Phase 2: Load best model, continue with medium difficulty
uv run python train_improved.py --config configs/estes_c6_medium.yaml \
    --load-model experiments/estes_c6_easy_*/best_model.zip

# Phase 3: Full difficulty
uv run python train_improved.py --config configs/estes_c6_full.yaml \
    --load-model models/rocket_spin_control_medium/best_model.zip
```

### Config Parameter Reference

Key parameters that affect training success:

| Parameter | Easy | Medium | Full | Effect |
|-----------|------|--------|------|--------|
| `max_tab_deflection` | 1-5° | 2-10° | 3-15° | Control authority |
| `initial_spin_std` | 3-5°/s | 6-10°/s | 9-15°/s | Starting difficulty |
| `max_roll_rate` | 720-900°/s | 576-720°/s | 360-450°/s | Termination threshold |
| `damping_scale` | 2.0-3.0 | 1.5-2.2 | 1.0-1.5 | Aerodynamic stability |
| `enable_wind` | false | true | true | Environmental challenge |

---

## Monitoring Training

### TensorBoard

```bash
uv run tensorboard --logdir logs
```

### Training Output Interpretation

Good training progress looks like:
```
Episode 500 | Timestep 50000
Rewards (last 10): 45.2 ± 12.3      # Increasing over time
Max Altitude: 89.4 ± 15.2 m         # Should reach 80-150m
Final Spin Rate: 25.3 ± 18.1 °/s    # Decreasing over time
High altitude (>50m): 85%           # Should increase
Low spin (<30°/s): 70%              # Should increase
```

Warning signs of failed training:
```
Max Altitude: 6.6 ± 2.2 m           # ❌ Too low
Final Spin Rate: 395.3 ± 12.3 °/s   # ❌ Too high
ep_len_mean: 37                     # ❌ Episodes too short
```

---

## Visualization Tools

### Motor Visualization (`visualize_motor.py`)

```bash
# View single motor
uv run python visualize_motor.py --motor aerotech_f40

# Compare motors
uv run python visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79

# Save without displaying
uv run python visualize_motor.py --motor estes_c6 --save motor.png --no-show
```

**Output plots:**
- Thrust curve with average/peak values
- Mass profile showing propellant consumption
- Impulse accumulation
- Motor specifications table

### Agent Visualization (`visualize_spin_agent.py`)

```bash
# Basic evaluation
uv run python visualize_spin_agent.py models/best_model.zip

# Full evaluation with report
uv run python visualize_spin_agent.py models/best_model.zip \
    --config configs/estes_c6_easy.yaml \
    --n-episodes 100 \
    --save-dir results/
```

**Output plots:**
- Altitude and spin rate distributions
- Camera quality breakdown
- Best trajectory detailed analysis
- Control action patterns
- Reward accumulation

---

## Physics Details

### The Control Problem

The rocket experiences random roll disturbances from:
- Asymmetric thrust
- Wind gusts
- Manufacturing imperfections

The agent controls two tabs on opposite fins that deflect to create a roll torque counteracting disturbances.

### Key Physics Parameters

```yaml
physics:
  disturbance_scale: 0.0001   # Random torque magnitude
  damping_scale: 1.0          # Aerodynamic roll damping
  max_tab_deflection: 15.0    # Control authority (degrees)
```

### Critical Fix: Control Authority Scaling

**Problem:** The original code had control authority calibrated for large rockets. For small rockets at high velocity, excessive tab deflection creates catastrophic angular acceleration.

**Solution:** The config generator analyzes physics and sets appropriate `max_tab_deflection` for each motor/rocket combination, ensuring random actions from an untrained agent don't immediately terminate episodes.

---

## File Structure

```
active-guidance-rockets/
├── configs/
│   ├── estes_c6_easy.yaml
│   ├── estes_c6_medium.yaml
│   ├── estes_c6_full.yaml
│   ├── aerotech_f40_easy.yaml
│   └── ...                          # Generated by generate_motor_config.py
├── spin_stabilized_control_env.py   # Environment with physics fixes
├── realistic_spin_rocket.py         # Motor integration
├── thrustcurve_motor_data.py        # Motor data parsing
├── train_improved.py                # Training script
├── rocket_config.py                 # Configuration system
├── diagnose_rocket.py               # Diagnostic tool
├── sweep_hyperparams.py             # Hyperparameter sweeps
│
├── generate_motor_config.py         # ★ Motor config generator
├── visualize_motor.py               # ★ Motor visualization
├── visualize_spin_agent.py          # ★ Agent evaluation & plots
├── run_experiment.sh                # ★ Automated pipeline
│
└── experiments/                     # Output directory
    └── {motor}_{difficulty}_{timestamp}/
        ├── config.yaml
        ├── plots/
        │   ├── motor_profile.png
        │   └── motor_comparison.png
        ├── evaluation/
        │   ├── performance_overview_*.png
        │   ├── best_trajectory_*.png
        │   └── evaluation_report_*.txt
        ├── logs/
        │   ├── diagnostics.log
        │   ├── training.log
        │   └── evaluation.log
        ├── best_model.zip
        └── SUMMARY.md
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| Altitude ~6m, spin ~400°/s | Control authority too high | Use `--generate-config` for physics-tuned config |
| Episodes end in <50 steps | `max_roll_rate` threshold too low | Increase in config or regenerate |
| Agent never improves | Reward too sparse | Reduce `spin_penalty_scale`, increase `low_spin_threshold` |
| Config not found error | Missing config file | Add `--generate-config` flag |
| Motor not found | Not in offline database | Install `requests` for API access |

### Verifying Environment Works

Test that episodes survive with random actions:

```bash
# Quick test using the pipeline
./run_experiment.sh --motor estes_c6 --generate-config --skip-training --skip-eval

# Or manually
python -c "
from spin_stabilized_control_env import RocketConfig
from realistic_spin_rocket import RealisticMotorRocket, CommonMotors
import numpy as np

config = RocketConfig(
    dry_mass=0.1, max_tab_deflection=5.0,
    damping_scale=2.0, initial_spin_std=5.0, max_roll_rate=720.0
)
env = RealisticMotorRocket(CommonMotors.estes_c6(), config)

for ep in range(5):
    obs, _ = env.reset()
    for step in range(200):
        obs, _, term, trunc, _ = env.step(np.random.uniform(-1, 1, size=(1,)))
        if term or trunc: break
    print(f'Episode {ep+1}: {step+1} steps')
# Should see 150+ steps per episode
"
```

---

## Expected Results

With proper training (500k+ steps on easy config):

| Metric | Untrained | Trained |
|--------|-----------|---------|
| Episode length | 30-50 steps | 200+ steps |
| Max altitude | 5-20m | 80-150m |
| Final spin rate | 300-400°/s | <30°/s |
| Camera quality | Poor | Good/Excellent |

---

## Hyperparameter Sweeps

```bash
# Preview sweep configurations (dry run)
uv run python sweep_hyperparams.py --sweep physics --dry-run

# Run sweeps
uv run python sweep_hyperparams.py --sweep physics --timesteps 100000
uv run python sweep_hyperparams.py --sweep reward --timesteps 100000
uv run python sweep_hyperparams.py --sweep ppo --timesteps 100000
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{active_guidance_rockets,
  title = {Active Guidance Rockets: RL for Spin Control},
  year = {2024},
  url = {https://github.com/your-repo/active-guidance-rockets}
}
```

---

## License

MIT License - see LICENSE file for details.
