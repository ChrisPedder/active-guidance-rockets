# Active Guidance Rockets

Spin stabilization for model rockets with onboard cameras. Adjustable fin tabs create differential drag torque to minimize roll rate during powered flight, producing stable video footage suitable for post-stabilization with Gyroflow.

## Overview

Model rockets spin during flight due to manufacturing asymmetries and aerodynamic disturbances. Even small spin rates (>5 deg/s) degrade onboard camera footage. This project implements a closed-loop roll rate controller using deflectable tabs on the trailing edges of the rocket's fins.

**Control mechanism:** 2-3 controllable fin tabs (of 4 fins total) with 30-degree max deflection. A 100 Hz control loop reads an ICM-20948 IMU gyroscope and adjusts tab deflection to counteract spin.

**Goal:** Mean spin rate < 5 deg/s in calm conditions, graceful degradation up to 3 m/s wind.

### Supported Rockets

| Parameter | Estes Alpha (C6) | AeroTech J800T |
|-----------|-------------------|----------------|
| Total impulse | 10 N-s | 1229 N-s |
| Burn time | 1.85 s | 1.80 s |
| Launch mass | 122 g | 2613 g |
| Max velocity | ~40 m/s | ~300 m/s (transonic) |
| Dynamic pressure range | ~1x variation | ~20x variation |
| Controlled fins | 2 of 4 | 3 of 4 |
| Tab sizing | 25% chord, 50% span | 25% chord, 50% span |

The J800 presents a harder control problem: higher dynamic pressure range means control effectiveness varies ~20x during flight, and the rocket reaches transonic speeds where aerodynamic coefficients change rapidly.

## Controllers

- **PID** — proportional-integral-derivative rate damper. Simplest option, most robust at high wind.
- **Gain-Scheduled PID (GS-PID)** — PID gains scaled by dynamic pressure to handle varying control effectiveness. Recommended classical controller.
- **Ensemble** — online switching between GS-PID and ADRC. Reduces worst-case episodes on Estes.
- **ADRC** — active disturbance rejection control. Strong on Estes but catastrophically fails on J800 (0% success). Retained as research baseline.
- **Residual SAC** — SAC reinforcement learning agent adds small corrections (10-20% max deflection) on top of optimized GS-PID. Best performer on J800.
- **Standalone SAC** — direct RL control without PID. Requires careful hyperparameter tuning (alpha=0.5, ent_coef=0.01).

## Results

All results from 50-episode evaluations with ICM-20948 IMU noise simulation. Wind model: sinusoidal gust model (dual-frequency gusts with configurable base speed, gust amplitude, and variability). Wind speed values below represent the mean wind speed for each condition.

### Estes C6

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 3.7 ± 0.6 | **3.4 ± 0.5** | **3.4 ± 0.5** | < 5 |
| 1 | 7.6 ± 3.8 | **6.0 ± 2.7** | 6.6 ± 2.7 | < 10 |
| 2 | 10.4 ± 5.4 | 9.5 ± 5.2 | **9.4 ± 4.9** | < 15 |
| 3 | 14.7 ± 7.5 | **11.7 ± 6.7** | 11.5 ± 6.1 | < 20 |
| 5 | **14.1 ± 7.5** | 17.8 ± 8.6 | 14.8 ± 7.6 | — |

All targets met up to 3 m/s wind. GS-PID is the recommended controller for Estes. PID is more robust at 5 m/s.

### AeroTech J800T

| Wind (m/s) | PID | GS-PID | Residual SAC (2M) | Standalone SAC | Target |
|------------|-----------|-----------|-------------------|----------------|--------|
| 0 | 12.8 ± 0.8 | 10.5 ± 0.8 | **3.7 ± 0.1** | 5.4 ± 0.1 | < 5 |
| 1 | 13.2 ± 0.9 | 11.0 ± 0.8 | **3.9 ± 0.3** | 5.6 ± 0.2 | < 10 |
| 2 | 14.4 ± 1.8 | 11.3 ± 0.9 | **4.0 ± 0.4** | 5.7 ± 0.4 | < 15 |
| 3 | 14.6 ± 1.8 | 11.7 ± 1.2 | **4.5 ± 0.8** | 6.1 ± 0.6 | < 20 |
| 5 | 15.9 ± 2.8 | 12.7 ± 1.7 | **5.6 ± 1.5** | 6.8 ± 1.2 | — |

Classical controllers plateau at ~10.5 deg/s on J800 (0 m/s wind) — they cannot meet the < 5 target. Residual SAC (2M training steps) achieves 3.7 deg/s at 0 m/s and meets the target at 0-2 m/s. Standalone SAC (alpha=0.5, ent_coef=0.01) also beats PID by 2-3x.

### Hardware Parameter Studies (Estes C6, GS-PID)

| Config | 0 m/s | 1 m/s | 2 m/s | 3 m/s | 5 m/s |
|--------|-------|-------|-------|-------|-------|
| Baseline (100 Hz, 2 fins) | 3.4 | 5.6 | 9.2 | 12.1 | 17.9 |
| 4 fins | 3.5 | 6.4 | 9.1 | 12.7 | 16.4 |
| 200 Hz | 2.2 | 5.3 | 8.2 | 11.4 | 14.8 |
| **4 fins + 200 Hz** | **1.7** | **3.7** | **7.1** | **8.4** | **10.8** |

Higher loop rate is the highest-impact hardware change. The 4-fin + 200 Hz configuration outperforms every classical controller at baseline hardware.

### Video Quality

At all spin rates achieved (3-18 deg/s), Gyroflow post-stabilization produces **Excellent** video quality across all camera presets (RunCam 1080p60, 4K30, 1080p120). Motion blur is 0.03-0.14 deg/frame. Further spin rate reduction has diminishing returns for video quality.

### Wind Model Note

The primary results above use a **sinusoidal gust model** (deterministic dual-frequency gusts). Dryden continuous turbulence (MIL-HDBK-1797) was evaluated separately — see `experimental_results.md` for full Dryden results.

All RL models (Residual SAC, Standalone SAC) and PID gain optimizations were trained under the sinusoidal wind model. When evaluated under Dryden moderate turbulence on J800, the RL models generalize well:

| Wind (m/s) | PID (Dryden) | GS-PID (Dryden) | Residual SAC (Dryden) | Standalone SAC (Dryden) |
|------------|-------------|-----------------|----------------------|------------------------|
| 0 | 12.8 ± 1.0 | 10.5 ± 0.5 | **3.7 ± 0.2** | 5.4 ± 0.2 |
| 1 | 14.1 ± 1.3 | 11.2 ± 1.0 | **4.3 ± 0.5** | 5.7 ± 0.4 |
| 2 | 15.2 ± 2.6 | 12.0 ± 1.4 | **4.8 ± 1.2** | 6.3 ± 0.8 |
| 3 | 15.5 ± 2.9 | 13.5 ± 2.2 | **6.1 ± 2.2** | 7.3 ± 1.8 |
| 5 | 15.9 ± 4.1 | 13.3 ± 2.4 | **8.5 ± 4.0** | 9.3 ± 2.9 |

Residual SAC maintains < 5 deg/s at 0-1 m/s under Dryden (vs 0-2 m/s under sinusoidal). Performance degrades 35-50% at 3-5 m/s due to the stochastic nature of Dryden turbulence, but RL still outperforms classical controllers by 2-3x. On Estes, ADRC outperforms GS-PID under Dryden turbulence (broadband disturbance plays to the ESO's strengths) but still catastrophically fails on J800 regardless of wind model.

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

### Controller Comparison

The main evaluation tool:

```bash
# Estes C6: all retained controllers with IMU noise
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 with optimized gains
uv run python compare_controllers.py --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# Evaluate a trained RL model
uv run python compare_controllers.py --config configs/aerotech_j800_wind.yaml \
    --sac models/rocket_residual_sac_j800_wind_aerotech_j800t_20260209_222006/best_model.zip \
    --imu --wind-levels 0 1 2 3 5 --n-episodes 50
```

Key flags: `--pid-only`, `--gain-scheduled`, `--adrc`, `--ensemble`, `--imu`, `--sac PATH`, `--video-quality`, `--camera-preset`, `--wind-levels`, `--n-episodes`, `--save-plot`.

Custom PID gains: `--pid-Kp`, `--pid-Ki`, `--pid-Kd`, `--pid-qref`.

### PID Gain Optimization

```bash
# LHS + Nelder-Mead
uv run python optimization/optimize_pid.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization.json

# Bayesian per-condition optimization
uv run python optimization/bayesian_optimize.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 1 2 3 5 --n-episodes 20 \
    --output optimization_results/bo_gs_pid_params.json
```

### RL Training

```bash
# SAC training with wind curriculum
uv run python training/train_sac.py --config configs/estes_c6_sac_wind.yaml \
    --timesteps 2000000 --early-stopping 30

# Residual SAC (PID + RL corrections)
uv run python training/train_residual_sac.py --config configs/aerotech_j800_wind.yaml \
    --timesteps 2000000

# PPO training
uv run python training/train_improved.py --config configs/estes_c6_sac_wind.yaml

# Hyperparameter sweep
uv run python training/sweep_hyperparams.py --sweep physics --timesteps 100000
```

### Visualizations

Three animated visualization scripts in `visualizations/`. All support `--rocket estes_alpha` or `--rocket j800`, PID controllers via `--controller pid`/`--controller gs-pid`, and RL models via `--sac`/`--residual-sac`/`--ppo` with a model path.

```bash
# Roll rate Monte Carlo (PID)
uv run python visualizations/roll_rate_montecarlo.py --rocket estes_alpha --controller gs-pid

# Roll rate Monte Carlo (RL model)
uv run python visualizations/roll_rate_montecarlo.py --rocket j800 \
    --residual-sac models/rocket_residual_sac_j800_wind_aerotech_j800t_20260209_222006/best_model.zip \
    --save

# Trajectory Monte Carlo (3D and 2D)
uv run python visualizations/trajectory_montecarlo.py --rocket j800 --controller gs-pid \
    --mode both --save

# Wind field visualization
uv run python visualizations/wind_field_visualization.py --rocket estes_alpha \
    --wind-speed 3.0 --save

# Wind field with Dryden turbulence
uv run python visualizations/wind_field_visualization.py --rocket j800 \
    --wind-speed 5.0 --dryden --severity moderate --save
```

### Dryden Turbulence

Test under MIL-HDBK-1797 continuous turbulence instead of sinusoidal wind:

```bash
# Estes C6 with Dryden turbulence
uv run python compare_controllers.py --config configs/estes_c6_dryden_moderate.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 with Dryden turbulence (RL models)
uv run python compare_controllers.py --config configs/aerotech_j800_dryden_moderate.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50
```

Severity options (Estes): `dryden_light`, `dryden_moderate`, `dryden_severe`.

### Hardware Parameter Studies

```bash
# 4 fins + 200 Hz (best hardware config)
uv run python compare_controllers.py --config configs/estes_c6_4fin_200hz.yaml \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50
```

### Tests

```bash
uv run python -m pytest tests/ -v
```

816 tests across 39 files covering controllers, physics simulation, optimization, configuration, wind models, sensors, training, and visualizations.

### Hardware Deployment

See **[`deployment/RASPBERRY_PI_GUIDE.md`](deployment/RASPBERRY_PI_GUIDE.md)** for a step-by-step guide covering:

- Parts list (Pi Zero 2 W, ICM-20948 IMU, micro servos, power distribution)
- Raspberry Pi OS setup and performance tuning
- Wiring diagrams (IMU, servos, power — see `deployment/diagrams/`)
- ONNX model export and deployment
- Servo and IMU calibration
- Bench testing procedure
- Pre-flight checklist and post-flight data review

The camera system (RunCam + DollaTek WiFi trigger + MOSFET modification) is documented separately in [`camera_electronics/`](camera_electronics/).

---

## Optimized Gains

| Platform | Kp | Ki | Kd | q_ref |
|----------|------|--------|--------|-------|
| Estes C6 (PID) | 0.0203 | 0.0002 | 0.0118 | — |
| J800 (GS-PID) | 0.0213 | 0.0050 | 0.0271 | 13268 |
| Estes 4-fin | 0.0177 | 0.0001 | 0.0054 | — |
| Estes 4-fin+200Hz | 0.0237 | 0.0043 | 0.0178 | — |

Stored in `optimization_results/*.json`.

---

## Project Structure

```
├── compare_controllers.py          # Main evaluation tool
├── realistic_spin_rocket.py        # Physics simulation
├── spin_stabilized_control_env.py  # Gym environment
├── rocket_config.py                # Configuration management
├── wind_model.py                   # Sinusoidal + Dryden wind disturbance models
├── motor_loader.py                 # Motor thrust curve loading
├── thrustcurve_motor_data.py       # Thrust curve data parsing
├── generate_motor_config.py        # Auto-generate configs from motor specs
│
├── controllers/                    # Control algorithms
│   ├── pid_controller.py           # PID and Gain-Scheduled PID
│   ├── adrc_controller.py          # ADRC (research baseline, Estes-only)
│   ├── ensemble_controller.py      # GS-PID + ADRC online switching
│   ├── disturbance_observer.py     # DOB for training wrappers
│   └── video_quality_metric.py     # Gyroflow post-stabilization analysis
│
├── training/                       # RL training pipelines
│   ├── train_improved.py           # PPO training
│   ├── train_sac.py                # SAC with wind curriculum
│   ├── train_residual_sac.py       # Residual SAC (PID + RL corrections)
│   └── sweep_hyperparams.py        # Hyperparameter sweeps
│
├── optimization/                   # Classical gain optimization
│   ├── optimize_pid.py             # LHS + Nelder-Mead
│   └── bayesian_optimize.py        # Per-condition Bayesian optimization
│
├── airframe/                       # Rocket geometry & OpenRocket import
│   ├── airframe.py                 # RocketAirframe class
│   ├── components.py               # NoseCone, BodyTube, TrapezoidFinSet
│   └── openrocket_parser.py        # .ork file parser
│
├── rocket_env/                     # Deployment & sensor simulation
│   ├── inference/                  # ONNX inference for embedded deployment
│   └── sensors/                    # IMU noise simulation (ICM-20948)
│
├── configs/                        # YAML environment configs
│   ├── estes_c6_sac_wind.yaml      # Main Estes config (sinusoidal wind)
│   ├── aerotech_j800_wind.yaml     # Main J800 config (sinusoidal wind)
│   ├── estes_c6_dryden_*.yaml      # Estes Dryden turbulence (light/moderate/severe)
│   ├── aerotech_j800_dryden_moderate.yaml  # J800 Dryden moderate turbulence
│   ├── estes_c6_4fin.yaml          # Hardware study: 4 active fins
│   ├── estes_c6_200hz.yaml         # Hardware study: 200 Hz loop
│   ├── estes_c6_4fin_200hz.yaml    # Hardware study: combined
│   ├── estes_c6_tab{10,15,25,30}.yaml  # Tab deflection sweep
│   └── airframes/                  # Airframe YAML definitions
│
├── models/                         # Trained RL models & wind estimator weights
├── optimization_results/           # Stored gain optimization results (JSON)
├── tests/                          # 39 test files, 816 tests
├── visualizations/                 # Animated Monte Carlo visualizations
├── deployment/                     # ONNX export, deployment bundles & Pi guide
├── docs/                           # Wind torque analysis
├── camera_electronics/             # Hardware modification guides
├── rocket-fin-servo-mount/         # Mechanical design docs
├── evaluation_results/             # Past evaluation plots/reports
└── experimental_results.md         # Full history of all 17 approaches tested
```
