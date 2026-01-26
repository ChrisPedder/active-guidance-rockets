# Active Guidance Rockets

Reinforcement learning for active spin control of model rockets using control surface tabs on fins.

## Overview

This project trains RL agents to stabilize rocket roll (spin) during flight using small deflectable tabs on the fins. The goal is to maintain stable camera footage from an onboard horizontal camera by minimizing spin rate.

**Key Features:**
- Import rocket designs from **OpenRocket** (.ork files) for accurate physics
- Automatic motor configuration from ThrustCurve.org database
- Physics-based moment of inertia and control effectiveness calculations
- Progressive difficulty training pipeline

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
uv run python visualizations/visualize_spin_agent.py models/best_model.zip --n-episodes 50
```

### Using Your OpenRocket Design

```python
from airframe import RocketAirframe

# Import your rocket design from OpenRocket
airframe = RocketAirframe.load("my_rocket.ork")
print(airframe.summary())

# Use in training (add airframe_file to your config YAML)
# physics:
#   airframe_file: "my_rocket.ork"
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

## Rocket Airframe Definition

The `airframe` module allows you to define rocket geometry separately from training configuration. This enables:
- **Importing designs from OpenRocket** (.ork files)
- **Accurate physics** based on actual component geometry
- **Reusable airframe definitions** across multiple experiments

### Loading from OpenRocket

If you design your rockets in [OpenRocket](https://openrocket.info/), you can import them directly:

```python
from airframe import RocketAirframe

# Load your OpenRocket design
airframe = RocketAirframe.load("my_rocket.ork")

# View the imported geometry
print(airframe.summary())
# Airframe: My Custom Rocket
#   Length: 450.0 mm
#   Diameter: 29.0 mm
#   Dry mass: 156.3 g
#   Components: 5
#   Fins: 4x, span=50.0mm
```

The parser extracts:
- Nose cone shape and dimensions
- Body tube lengths and diameters
- Fin geometry (trapezoidal fins)
- Material properties and masses
- Motor mount dimensions

### Using Airframes in Training

**Airframe is REQUIRED.** Every training config must specify an airframe file:

```yaml
# configs/my_training_config.yaml
physics:
  airframe_file: "configs/airframes/my_rocket.ork"  # REQUIRED (.ork or .yaml)

  # Control tab geometry (applied to fins)
  tab_chord_fraction: 0.25
  tab_span_fraction: 0.5
  max_tab_deflection: 10.0

  # Physics tuning (training-specific)
  disturbance_scale: 0.0001
  damping_scale: 1.5

motor:
  name: "aerotech_f40"
  # ... motor config
```

Or create the environment programmatically:

```python
from airframe import RocketAirframe
from realistic_spin_rocket import RealisticMotorRocket

# Airframe is REQUIRED as first argument
airframe = RocketAirframe.load("my_rocket.ork")
env = RealisticMotorRocket(
    airframe=airframe,  # REQUIRED
    motor_config=motor_cfg
)
```

### Defining Airframes in YAML

You can also define airframes in YAML (see `configs/airframes/estes_alpha.yaml`):

```yaml
name: My Custom Rocket
description: A 29mm minimum diameter rocket

components:
  - type: NoseCone
    name: Nose Cone
    position: 0.0
    length: 0.10
    base_diameter: 0.029
    shape: ogive
    thickness: 0.002
    material: Fiberglass

  - type: BodyTube
    name: Body Tube
    position: 0.10
    length: 0.35
    outer_diameter: 0.029
    inner_diameter: 0.027
    material: Fiberglass

  - type: TrapezoidFinSet
    name: Fins
    position: 0.35
    num_fins: 4
    root_chord: 0.06
    tip_chord: 0.03
    span: 0.05
    thickness: 0.003
    material: Fiberglass
```

### Physics from Airframe Geometry

When an airframe is provided, the simulation calculates:

| Property | Calculation |
|----------|-------------|
| **Roll Inertia** | Sum of component inertias using parallel axis theorem |
| **Control Effectiveness** | From fin area, tab size, and moment arm |
| **Aerodynamic Damping** | From total fin area and geometry |
| **Disturbance Scaling** | Based on body diameter (volume scaling) |

This gives more accurate physics than the simplified inline parameters.

### Factory Methods

Common airframes are available as factory methods:

```python
from airframe import RocketAirframe

# Estes Alpha III (24mm, C motors)
alpha = RocketAirframe.estes_alpha()

# Minimum diameter rocket
min_d = RocketAirframe.high_power_minimum_diameter(motor_diameter=0.038)
```

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
uv run python visualizations/visualize_motor.py --motor estes_c6 --save motor_c6.png

# Compare all motors
uv run python visualizations/visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79
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
uv run python visualizations/visualize_spin_agent.py models/best_model.zip \
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

Progressive training from easy to hard difficulty produces the best results. Each stage fine-tunes from the previous stage's model.

| Phase | Config | Timesteps | What Agent Learns |
|-------|--------|-----------|-------------------|
| 1. Easy | `*_easy.yaml` | 1.5M | Basic stabilization, smooth control |
| 2. Medium | `*_medium.yaml` | 1M | Tighter spin control, wind handling |
| 3. Full | `*_full.yaml` | 1M | Full difficulty, <3°/s target spin |

#### Automated Progressive Training (Recommended)

Use the progressive training script to run all stages automatically:

```bash
# Run full pipeline (easy -> medium -> full)
./train_progressive.sh

# Run with custom timesteps
./train_progressive.sh --timesteps 2000000

# Run only a specific stage
./train_progressive.sh --stage easy
./train_progressive.sh --stage medium
./train_progressive.sh --stage full

# Quick test run (reduced timesteps)
./train_progressive.sh --timesteps-easy 100000 --timesteps-medium 50000 --timesteps-full 50000
```

#### Manual Progressive Training

```bash
# Phase 1: Easy start (from scratch)
uv run python train_improved.py --config configs/estes_c6_easy.yaml --timesteps 1500000

# Phase 2: Fine-tune on medium difficulty
uv run python train_improved.py --config configs/estes_c6_medium.yaml --load-model models/rocket_estes_c6_easy_*/best_model.zip

# Phase 3: Fine-tune on full difficulty
uv run python train_improved.py --config configs/estes_c6_full.yaml --load-model models/rocket_estes_c6_medium_*/best_model.zip
```

### Config Parameter Reference

Key parameters that affect training success:

**Physics Parameters:**

| Parameter | Easy | Medium | Full | Effect |
|-----------|------|--------|------|--------|
| `max_tab_deflection` | 1-5° | 2-10° | 3-15° | Control authority |
| `initial_spin_std` | 3-5°/s | 6-10°/s | 9-15°/s | Starting difficulty |
| `max_roll_rate` | 720-900°/s | 576-720°/s | 360-450°/s | Termination threshold |
| `damping_scale` | 2.0-3.0 | 1.5-2.2 | 1.0-1.5 | Aerodynamic stability |
| `enable_wind` | false | true | true | Environmental challenge |

**Reward Parameters (anti-oscillation tuning):**

| Parameter | Easy | Medium | Full | Effect |
|-----------|------|--------|------|--------|
| `control_smoothness_penalty` | -0.10 | -0.12 | -0.15 | Penalizes rapid control changes |
| `spin_oscillation_penalty` | -0.03 | -0.04 | -0.05 | Penalizes spin rate oscillation |
| `low_spin_threshold` | 10°/s | 5°/s | 3°/s | Target spin rate for bonus |

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

## PID Controller Baseline

### Why Compare with Classical Control?

While reinforcement learning can discover sophisticated control strategies, it's valuable to compare against classical control theory approaches like PID (Proportional-Integral-Derivative) control:

1. **Baseline Performance**: PID provides a well-understood baseline to measure RL improvements against
2. **Interpretability**: PID gains have clear physical meaning, while RL policies are black boxes
3. **Tuning Effort**: PID requires manual tuning; RL learns automatically but needs more compute
4. **Real-world Validation**: Many existing rocket stabilization systems use PID, so comparison validates our simulation
5. **Failure Modes**: Understanding where PID fails helps explain what the RL agent learns

### PID Controller Model

The PID controller (`pid_controller.py`) mimics a real Arduino-based rocket stabilization system:

```
Control = Cprop × (θ - θ_target) + Cint × ∫(θ - θ_target)dt + Cderiv × ω
```

Where:
- **θ**: Current roll orientation (degrees)
- **θ_target**: Target orientation (captured at launch)
- **ω**: Roll rate (degrees/second)
- **Cprop, Cint, Cderiv**: Tunable gains

**Key Features:**
- **Launch Detection**: Activates when vertical acceleration exceeds 20 m/s²
- **Target Lock**: Stores roll orientation at launch as the setpoint
- **Anti-windup**: Integral term is clamped to prevent saturation
- **Output Limiting**: Control surface deflection is bounded

### Running PID Simulations

```bash
# Basic PID evaluation
uv run python pid_controller.py --config configs/estes_c6_easy.yaml --n-episodes 50

# Custom PID gains
uv run python pid_controller.py --config configs/estes_c6_easy.yaml \
    --Cprop 0.02 --Cint 0.005 --Cderiv 0.05

# Compare PID vs trained RL agent
uv run python pid_controller.py --config configs/estes_c6_easy.yaml \
    --compare models/rocket_estes_c6_easy_*/best_model.zip

# Save comparison plot
uv run python pid_controller.py --config configs/estes_c6_easy.yaml \
    --compare models/rocket_estes_c6_easy_*/best_model.zip \
    --save-plot comparison.png
```

### PID Tuning Guide

| Gain | Effect | Too Low | Too High |
|------|--------|---------|----------|
| `Cprop` | Response speed | Slow correction, drift | Overshoot, oscillation |
| `Cint` | Steady-state error | Persistent offset | Windup, slow recovery |
| `Cderiv` | Damping | Overshoot | Noise sensitivity, jitter |

**Recommended starting values:**
- `Cprop = 0.02` - Moderate proportional response
- `Cint = 0.005` - Small integral to eliminate drift
- `Cderiv = 0.05` - Light damping based on roll rate

### Expected Comparison Results

| Metric | PID (tuned) | RL Agent |
|--------|-------------|----------|
| Mean Spin Rate | 15-30°/s | 2-10°/s |
| Consistency | Variable | Consistent |
| Adaptability | Fixed response | Adapts to conditions |
| Control Smoothness | Can oscillate | Learned smooth control |

The RL agent typically outperforms PID because it:
- Learns non-linear control strategies
- Adapts control authority to flight phase (boost vs coast)
- Optimizes for the specific reward function
- Handles the full state space, not just roll error

---

## Visualization Tools

### Motor Visualization (`visualizations/visualize_motor.py`)

```bash
# View single motor
uv run python visualizations/visualize_motor.py --motor aerotech_f40

# Compare motors
uv run python visualizations/visualize_motor.py --compare estes_c6 aerotech_f40 cesaroni_g79

# Save without displaying
uv run python visualizations/visualize_motor.py --motor estes_c6 --save motor.png --no-show
```

**Output plots:**
- Thrust curve with average/peak values
- Mass profile showing propellant consumption
- Impulse accumulation
- Motor specifications table

### Agent Visualization (`visualizations/visualize_spin_agent.py`)

```bash
# Basic evaluation
uv run python visualizations/visualize_spin_agent.py models/best_model.zip

# Full evaluation with report
uv run python visualizations/visualize_spin_agent.py models/best_model.zip \
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

### Physics Calculations

When using an airframe definition (from OpenRocket or YAML), the simulation computes:

**Roll Moment of Inertia:**
```
I_total = I_nosecone + I_bodytube + I_fins + I_motor

I_fins = Σ (I_cm + m_fin × d²)  # Parallel axis theorem
```

**Control Torque:**
```
τ_control = 2 × F_tab × r_moment
F_tab = ½ × Cl × q × A_tab
Cl = 2π × sin(δ)  # Thin airfoil theory
```

**Aerodynamic Damping:**
```
τ_damping = -C_damp × ω × q / V
C_damp = A_fins × r_moment²
```

### Control Authority Scaling

Control authority must be calibrated for rocket size. For small rockets at high velocity, excessive tab deflection creates large angular accelerations.

The config generator analyzes motor physics and sets appropriate `max_tab_deflection` for each motor/rocket combination, ensuring the control authority is appropriate for the rocket's inertia.

When using an airframe, the actual fin geometry determines control effectiveness automatically.

---

## File Structure

```
active-guidance-rockets/
├── configs/                          # Training configurations
│   ├── estes_c6.yaml
│   ├── estes_c6_easy_start.yaml
│   ├── estes_c6_medium.yaml
│   ├── aerotech_f40_easy.yaml
│   ├── ...                          # Generated by generate_motor_config.py
│   └── airframes/                   # Airframe definitions
│       └── estes_alpha.yaml         # Example airframe
│
├── airframe/                        # Rocket geometry module
│   ├── __init__.py
│   ├── components.py                # NoseCone, BodyTube, TrapezoidFinSet, etc.
│   ├── airframe.py                  # RocketAirframe class
│   └── openrocket_parser.py         # .ork file parser (pure Python)
│
├── spin_stabilized_control_env.py   # Spin-stabilized rocket environment
├── realistic_spin_rocket.py         # Motor integration
├── thrustcurve_motor_data.py        # Motor data parsing
├── train_improved.py                # Training script
├── rocket_config.py                 # Configuration system
├── motor_loader.py                  # Motor loading utilities
├── sweep_hyperparams.py             # Hyperparameter sweeps
│
├── generate_motor_config.py         # Motor config generator
├── run_experiment.sh                # Automated pipeline
│
├── visualizations/
│   ├── visualize_motor.py           # Motor visualization
│   └── visualize_spin_agent.py      # Agent evaluation & plots
│
├── models/                          # Trained models
├── logs/                            # Training logs
└── experiments/                     # Output directory
    └── {motor}_{difficulty}_{timestamp}/
        ├── config.yaml
        ├── plots/
        ├── evaluation/
        ├── logs/
        ├── best_model.zip
        └── SUMMARY.md
```

---

## Troubleshooting

| Symptom | Cause | Solution |
|---------|-------|----------|
| "RocketAirframe is required" error | No airframe specified | Add `airframe_file` to config or use `RocketAirframe.load()` |
| "airframe_file is REQUIRED" error | Config missing airframe | Add `physics.airframe_file: "path/to/rocket.ork"` |
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
from airframe import RocketAirframe
from spin_stabilized_control_env import RocketConfig
from realistic_spin_rocket import RealisticMotorRocket
import numpy as np

# Load airframe (REQUIRED)
airframe = RocketAirframe.estes_alpha()

# Motor config
motor_config = {
    'name': 'estes_c6',
    'manufacturer': 'Estes', 'designation': 'C6',
    'total_impulse_Ns': 10.0, 'avg_thrust_N': 5.4, 'max_thrust_N': 14.0,
    'burn_time_s': 1.85, 'propellant_mass_g': 12.3, 'case_mass_g': 12.7,
    'thrust_curve': {'time_s': [0,0.1,1.85], 'thrust_N': [0,14,0]}
}

config = RocketConfig(
    max_tab_deflection=5.0, damping_scale=2.0,
    initial_spin_std=5.0, max_roll_rate=720.0
)
env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config, config=config)

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
