# Active Guidance Rockets

Spin stabilization for model rockets with onboard cameras. Uses adjustable fin tabs to minimize roll rate during flight for stable video footage.

## Overview

**Target rockets:** Estes C6 (~10 N-s impulse) and AeroTech J800T (~800 N-s impulse)

**Control mechanism:** 2 controllable fin tabs (of 4 fins total) that create differential drag torque. A 100 Hz control loop reads gyro data and adjusts tab deflection to counteract spin.

**Controllers:**
- **PID** — simple rate-damping baseline
- **Gain-Scheduled PID (GS-PID)** — PID gains scaled by dynamic pressure to handle ~20x variation in control effectiveness during flight. **Recommended for deployment.**
- **Ensemble** — online switching between GS-PID and ADRC, reduces worst-case episodes
- **ADRC** — active disturbance rejection, retained as research baseline (fails on J800)

## Results

All results from 50-episode evaluations with ICM-20948 IMU noise simulation, February 2026. Two physics bugfixes were applied before these runs: corrected wind torque model (body-shadow) and tab deflection (3.6deg to 30deg).

### Estes C6

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 3.7 ± 0.6 | **3.4 ± 0.5** | **3.4 ± 0.5** | < 5 |
| 1 | 7.6 ± 3.8 | **6.0 ± 2.7** | 6.6 ± 2.7 | < 10 |
| 2 | 10.4 ± 5.4 | 9.5 ± 5.2 | **9.4 ± 4.9** | < 15 |
| 3 | 14.7 ± 7.5 | **11.7 ± 6.7** | 11.5 ± 6.1 | < 20 |
| 5 | **14.1 ± 7.5** | 17.8 ± 8.6 | 14.8 ± 7.6 | — |

All targets met up to 3 m/s wind. 100% success rate at 0-2 m/s across all controllers.

### AeroTech J800T

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 12.8 ± 0.6 | **10.5 ± 0.8** | **10.5 ± 0.7** | < 5 |
| 1 | 13.6 ± 1.0 | 11.0 ± 0.8 | **10.8 ± 0.7** | < 10 |
| 2 | 14.1 ± 1.5 | 11.3 ± 0.9 | **11.2 ± 0.8** | < 15 |
| 3 | 14.6 ± 2.0 | **11.7 ± 1.2** | 12.0 ± 1.2 | < 20 |
| 5 | 15.1 ± 1.9 | 12.7 ± 1.7 | **12.4 ± 1.5** | — |

J800 baseline (10.5 deg/s at 0 m/s) is higher than Estes — gains may need re-optimization for post-bugfix physics.

### Hardware Parameter Studies (Estes C6, GS-PID)

| Config | 0 m/s | 1 m/s | 2 m/s | 3 m/s | 5 m/s |
|--------|-------|-------|-------|-------|-------|
| Baseline (100 Hz, 2 fins) | 3.4 | 5.6 | 9.2 | 12.1 | 17.9 |
| 4 fins | 3.5 | 6.4 | 9.1 | 12.7 | 16.4 |
| 200 Hz | 2.2 | 5.3 | 8.2 | 11.4 | 14.8 |
| 500 Hz | 1.5 | 5.0 | 8.6 | 10.5 | 17.0 |
| **4 fins + 200 Hz** | **1.7** | **3.7** | **7.1** | **8.4** | **10.8** |

Higher loop rate is the single highest-impact hardware change. The 4-fin + 200 Hz configuration outperforms every controller at baseline hardware.

### Video Quality

At all spin rates achieved (3-18 deg/s), Gyroflow post-stabilization produces **Excellent** video quality across all camera presets (RunCam 1080p60, 4K30, 1080p120). Motion blur is 0.03-0.14 deg/frame. Further spin rate reduction has diminishing returns for video quality.

---

## Quick Start

### Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

### Running Experiments

#### Controller Comparison

The main tool for evaluating controllers under wind:

```bash
# Estes C6: PID baseline
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --pid-only --wind-levels 0 1 2 3 5 --n-episodes 50

# Estes C6: all retained controllers with IMU noise
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 with optimized gains
uv run python compare_controllers.py --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# Video quality analysis
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --imu --video-quality --camera-preset all \
    --wind-levels 0 1 2 3 5 --n-episodes 20
```

Key flags: `--pid-only`, `--gain-scheduled`, `--adrc`, `--ensemble`, `--imu`, `--video-quality`, `--camera-preset`, `--wind-levels`, `--n-episodes`, `--save-plot`.

Custom PID gains: `--pid-Kp`, `--pid-Ki`, `--pid-Kd`, `--pid-qref`.

#### PID Gain Optimization

Optimize PID/GS-PID gains using Latin Hypercube Sampling + Nelder-Mead local search:

```bash
# Estes C6
uv run python optimization/optimize_pid.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization.json

# Bayesian per-condition optimization (differential evolution)
uv run python optimization/bayesian_optimize.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 1 2 3 5 --n-episodes 20 \
    --output optimization_results/bo_gs_pid_params.json
```

#### RL Training

```bash
# PPO training
uv run python training/train_improved.py --config configs/estes_c6_sac_wind.yaml

# SAC training with wind curriculum
uv run python training/train_sac.py --config configs/estes_c6_sac_wind.yaml \
    --timesteps 2000000 --early-stopping 30

# Hyperparameter sweep
uv run python training/sweep_hyperparams.py --sweep physics --timesteps 100000
```

#### Hardware Parameter Studies

Pre-configured configs for different hardware setups:

```bash
# 4 active fins
uv run python compare_controllers.py --config configs/estes_c6_4fin.yaml \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# 200 Hz control loop
uv run python compare_controllers.py --config configs/estes_c6_200hz.yaml \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# 4 fins + 200 Hz (best hardware config)
uv run python compare_controllers.py --config configs/estes_c6_4fin_200hz.yaml \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# Tab deflection sweep (10, 15, 25, 30 degrees)
uv run python compare_controllers.py --config configs/estes_c6_tab10.yaml \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50
```

Re-optimize gains for each hardware config:

```bash
uv run python optimization/optimize_pid.py --config configs/estes_c6_4fin_200hz.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization_4fin_200hz.json
```

#### Dryden Turbulence

Test under MIL-HDBK-1797 continuous turbulence instead of sinusoidal wind:

```bash
uv run python compare_controllers.py --config configs/estes_c6_dryden_moderate.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50
```

Severities: `dryden_light`, `dryden_moderate`, `dryden_severe`.

### Importing Custom Rockets

Import rocket designs from [OpenRocket](https://openrocket.info/) .ork files or define airframes in YAML:

```python
from airframe import RocketAirframe

# From OpenRocket
airframe = RocketAirframe.load("my_rocket.ork")

# From YAML
airframe = RocketAirframe.load("configs/airframes/estes_alpha.yaml")

# Built-in factory
airframe = RocketAirframe.estes_alpha()
```

Configs reference airframes via `physics.airframe_file` in the YAML.

### Generating Motor Configs

Auto-generate training configs from motor specs:

```bash
uv run python generate_motor_config.py list-popular
uv run python generate_motor_config.py generate estes_c6 --output configs/
```

### Tests

```bash
uv run python -m pytest tests/ -v
```

662 tests, covering controllers, physics simulation, optimization, configuration, and IMU modeling.

---

## Optimized Gains

| Platform | Kp | Ki | Kd | q_ref |
|----------|------|--------|--------|-------|
| Estes C6 (PID) | 0.0203 | 0.0002 | 0.0118 | — |
| J800 (GS-PID) | 0.0213 | 0.0050 | 0.0271 | 13268 |
| Estes 4-fin | 0.0177 | 0.0001 | 0.0054 | — |
| Estes 4-fin+200Hz | 0.0237 | 0.0043 | 0.0178 | — |
| Estes Tab10 | 0.0079 | 0.0060 | 0.0148 | 5549 |

Stored in `optimization_results/*.json`.

---

## Visualizations

Three animated visualization scripts in `visualizations/` provide insight into simulation behavior. All support `--rocket estes_alpha` or `--rocket j800`, `--controller pid` or `--controller gs-pid`, and a `--save` flag to write output files to `visualizations/outputs/`.

### Roll Rate Monte Carlo (`roll_rate_montecarlo.py`)

Shows |roll rate| vs time for multiple simulation runs under increasing wind. Traces are drawn one at a time with a brief pause, then the plot clears and repeats at the next wind level. The fixed y-axis makes wind-degradation visible at a glance.

```bash
# Display on screen
uv run python visualizations/roll_rate_montecarlo.py --rocket estes_alpha --controller gs-pid

# Save as GIF
uv run python visualizations/roll_rate_montecarlo.py --rocket j800 --controller pid --save

# Custom settings
uv run python visualizations/roll_rate_montecarlo.py --rocket estes_alpha --controller pid \
    --n-runs 10 --wind-levels 1 2 3 --format mp4 --save
```

**Arguments:** `--rocket`, `--controller`, `--n-runs` (default 10), `--wind-levels` (default 1 2 3), `--save`, `--format` (gif/mp4).

### Wind Field Visualization (`wind_field_visualization.py`)

Visualizes the wind model (sinusoidal or Dryden) to show what disturbances the rocket faces. Three panels: wind speed vs altitude, wind direction vs altitude, and a time-series at a fixed altitude. Altitude profiles accumulate as time-stamped snapshots.

```bash
# Display on screen
uv run python visualizations/wind_field_visualization.py --rocket estes_alpha

# Dryden turbulence at 5 m/s
uv run python visualizations/wind_field_visualization.py --rocket j800 \
    --wind-speed 5.0 --dryden --severity moderate --save

# Custom fixed altitude for time-series panel
uv run python visualizations/wind_field_visualization.py --rocket estes_alpha \
    --wind-speed 3.0 --fixed-altitude 100 --save
```

**Arguments:** `--rocket`, `--wind-speed` (default 2.0), `--dryden`, `--severity` (light/moderate/severe), `--fixed-altitude`, `--save`, `--format`.

### Trajectory Monte Carlo (`trajectory_montecarlo.py`)

Shows 3D flight paths (or 2D panel views) for multiple runs under varying wind. Lateral drift is estimated from wind speed and direction. The 2D mode shows x-z, y-z, and x-y (ground track) subplots which are often easier to interpret than the 3D view.

```bash
# 3D animated plot
uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid

# 2D panel view, saved
uv run python visualizations/trajectory_montecarlo.py --rocket j800 --controller pid \
    --mode 2d --save

# Both 3D and 2D
uv run python visualizations/trajectory_montecarlo.py --rocket estes_alpha --controller gs-pid \
    --mode both --wind-levels 1 2 3 --n-runs 10 --save
```

**Arguments:** `--rocket`, `--controller`, `--n-runs` (default 10), `--wind-levels` (default 1 2 3), `--mode` (3d/2d/both), `--save`, `--format`.

---

## Project Structure

```
├── compare_controllers.py          # Main evaluation tool
├── realistic_spin_rocket.py        # Physics simulation
├── spin_stabilized_control_env.py  # Gym environment
├── rocket_config.py                # Configuration management
├── wind_model.py                   # Wind disturbance model
├── motor_loader.py                 # Motor thrust curve loading
├── thrustcurve_motor_data.py       # Thrust curve data parsing
├── generate_motor_config.py        # Auto-generate configs from motor specs
│
├── controllers/                    # Control algorithms
│   ├── pid_controller.py           # PID and Gain-Scheduled PID
│   ├── adrc_controller.py          # ADRC (research baseline)
│   ├── ensemble_controller.py      # GS-PID + ADRC online switching
│   ├── disturbance_observer.py     # DOB for training wrappers
│   └── video_quality_metric.py     # Gyroflow post-stabilization analysis
│
├── training/                       # RL training pipelines
│   ├── train_improved.py           # PPO training
│   ├── train_sac.py                # SAC with wind curriculum
│   ├── train_residual_sac.py       # Residual SAC (PID + RL)
│   └── sweep_hyperparams.py        # Hyperparameter sweeps
│
├── optimization/                   # Classical gain optimization
│   ├── optimize_pid.py             # LHS + Nelder-Mead
│   └── bayesian_optimize.py        # Per-condition Bayesian optimization
│
├── airframe/                       # Rocket geometry & OpenRocket import
├── rocket_env/                     # ONNX inference & sensor simulation
├── configs/                        # YAML environment configs & airframes
├── optimization_results/           # Stored gain optimization results (JSON)
├── tests/                          # Test suite (662 tests)
├── visualizations/                 # Motor & agent visualization
├── scripts/                        # ONNX export
├── docs/                           # Wind torque analysis
├── camera_electronics/             # Hardware modification guides
├── rocket-fin-servo-mount/         # Mechanical design docs
├── models/                         # Trained RL models
└── experimental_results.md         # Full history of all 17 approaches tested
```
