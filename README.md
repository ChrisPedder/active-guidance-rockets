# Active Guidance Rockets

Closed-loop spin stabilization for model rockets with onboard cameras. Deflectable tabs on fin trailing edges create differential drag torque to minimize roll rate during powered flight, producing stable video footage suitable for post-stabilization with Gyroflow.

## Project Goal

**Primary objective:** Mean spin rate < 5 deg/s during powered flight for stable onboard camera footage.

**Success criteria by wind level:**

| Wind (m/s) | Target (deg/s) |
|------------|----------------|
| 0          | < 5            |
| 1          | < 10           |
| 2          | < 15           |
| 3          | < 20           |

### Supported Rockets

| Parameter | Estes Alpha (C6) | AeroTech J800T |
|-----------|-------------------|----------------|
| Total impulse | 10 N-s | 1229 N-s |
| Burn time | 1.85 s | 1.80 s |
| Launch mass | 122 g | 2613 g |
| Max velocity | ~40 m/s | ~300 m/s (transonic) |
| Dynamic pressure range | ~1x variation | ~20x variation |
| Controlled fins | 2 of 4 | 3 of 4 |

#### Estes Alpha III (C6)

![Estes Alpha III airframe](estes_alpha_airframe.png)

#### AeroTech J800T (75mm Carbon Fibre)

![J800 75mm airframe](j800_airframe.png)

## Results Summary

All results from 50-episode evaluations with ICM-20948 IMU noise. Full details in [`experimental_results.md`](experimental_results.md).

### Estes C6

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 3.4 | **3.4** | 3.3 | < 5 |
| 1 | 6.8 | **5.6** | 6.0 | < 10 |
| 2 | 10.3 | 9.2 | **9.0** | < 15 |
| 3 | 11.4 | 12.1 | **10.8** | < 20 |

All targets met up to 3 m/s. GS-PID recommended for flight.

### AeroTech J800T

| Wind (m/s) | PID | GS-PID | Residual SAC (2M) | Standalone SAC | Target |
|------------|-----|--------|-------------------|----------------|--------|
| 0 | 12.8 | 10.5 | **3.7** | 5.4 | < 5 |
| 1 | 13.2 | 10.7 | **3.9** | 5.6 | < 10 |
| 2 | 14.4 | 11.3 | **4.0** | 5.7 | < 15 |
| 3 | 14.6 | 11.7 | **4.5** | 6.1 | < 20 |

Classical controllers plateau at ~10.5 deg/s on J800. Residual SAC meets the < 5 target at 0-2 m/s.

---

## Prerequisites

- **Python 3.11+** (tested with 3.12; `requires-python = ">=3.11,<4.0"`)
- **[uv](https://github.com/astral-sh/uv)** package manager
- ~4 GB disk space for dependencies (PyTorch, Stable-Baselines3, etc.)
- GPU optional but recommended for RL training (CUDA-compatible; CPU works for evaluation)

## Setup

```bash
# Clone and enter the project
git clone <repository-url>
cd active-guidance-rockets

# Create virtual environment and install all dependencies
uv venv && source .venv/bin/activate
uv pip install -e ".[all,dev]"
```

The `[all]` extra installs: `stable-baselines3[extra]`, `torch==2.5.1`, `tensorboard`, `matplotlib`, `pygame`, `plotly`, `seaborn`, `pandas`, `lxml`. The `[dev]` extra adds: `pytest`, `pytest-cov`, `black`, `mypy`.

Verify the installation:

```bash
uv run python -m pytest tests/ -v --tb=short
```

---

## Reproducing the Results

The results in [`experimental_results.md`](experimental_results.md) are produced by the pipeline below. Steps are listed in dependency order: later steps require outputs from earlier steps.

### Step 0: Verify Configs Exist

All required YAML configs are checked into `configs/`. No generation step is needed.

Key configs used throughout:
- `configs/estes_c6_sac_wind.yaml` — Main Estes C6 config (sinusoidal wind)
- `configs/aerotech_j800_wind.yaml` — Main J800 config (sinusoidal wind)
- `configs/aerotech_j800_residual_sac_wind.yaml` — J800 Residual SAC training config
- `configs/aerotech_j800_sac_wind.yaml` — J800 Standalone SAC training config
- `configs/estes_c6_residual_sac_wind.yaml` — Estes Residual SAC training config
- `configs/estes_c6_dryden_{light,moderate,severe}.yaml` — Estes Dryden turbulence
- `configs/aerotech_j800_dryden_moderate.yaml` — J800 Dryden turbulence
- `configs/estes_c6_{4fin,200hz,500hz,4fin_200hz,tab10}.yaml` — Hardware studies
- `configs/sac_sweep/j800_sac_a{50,80,100}_e{001,005}.yaml` — Standalone SAC sweep

### Step 1: PID Gain Optimization

Finds optimal PID/GS-PID gains using Latin Hypercube Sampling + Nelder-Mead refinement. These gains are used in all subsequent evaluation and RL training steps.

**Script:** `optimization/optimize_pid.py`

```bash
# Estes C6 PID gains (produces Kp=0.0203, Ki=0.0002, Kd=0.0118)
uv run python optimization/optimize_pid.py \
    --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 --imu \
    --output optimization_results/pid_optimization.json

# J800 GS-PID gains (produces Kp=0.0213, Ki=0.0050, Kd=0.0271, q_ref=13268)
uv run python optimization/optimize_pid.py \
    --config configs/aerotech_j800_wind.yaml \
    --gain-scheduled --optimize-qref --imu \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/gs_pid_j800_optimization.json

# Hardware study configs (4fin, 200hz, 4fin+200hz, tab10)
uv run python optimization/optimize_pid.py \
    --config configs/estes_c6_4fin.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 --imu \
    --output optimization_results/pid_optimization_4fin.json

uv run python optimization/optimize_pid.py \
    --config configs/estes_c6_200hz.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 --imu \
    --output optimization_results/pid_optimization_200hz.json

uv run python optimization/optimize_pid.py \
    --config configs/estes_c6_4fin_200hz.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 --imu \
    --output optimization_results/pid_optimization_4fin_200hz.json

uv run python optimization/optimize_pid.py \
    --config configs/estes_c6_tab10.yaml \
    --gain-scheduled --optimize-qref --imu \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization_tab10.json

# Dryden turbulence PID gains (one per severity)
for severity in light moderate severe; do
    uv run python optimization/optimize_pid.py \
        --config configs/estes_c6_dryden_${severity}.yaml \
        --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 --imu \
        --output optimization_results/pid_optimization_dryden_${severity}.json
done
```

**Inputs:** YAML config file
**Outputs:** JSON file in `optimization_results/` with optimized gains and evaluation scores
**Note:** Stochastic — results vary between runs due to random episode generation. The gains documented in `experimental_results.md` are the specific values found during the original optimization.

### Step 2: Bayesian Per-Condition Optimization

Finds per-wind-level optimized parameters using differential evolution. Produces parameter lookup tables used by `compare_controllers.py --optimized-params`.

**Script:** `optimization/bayesian_optimize.py`

```bash
# Estes C6 GS-PID per-condition
uv run python optimization/bayesian_optimize.py \
    --config configs/estes_c6_sac_wind.yaml \
    --controller gs-pid --wind-level 0 1 2 3 5 --n-episodes 20 --n-trials 80 \
    --output optimization_results/bo_gs_pid_params.json

# Estes C6 ADRC per-condition
uv run python optimization/bayesian_optimize.py \
    --config configs/estes_c6_sac_wind.yaml \
    --controller adrc --wind-level 0 1 2 3 5 --n-episodes 20 --n-trials 80 \
    --output optimization_results/bo_adrc_params.json

# J800 GS-PID per-condition
uv run python optimization/bayesian_optimize.py \
    --config configs/aerotech_j800_wind.yaml \
    --controller gs-pid --wind-level 0 1 2 3 5 --n-episodes 20 --n-trials 80 \
    --output optimization_results/bo_gs_pid_j800_params.json
```

**Inputs:** YAML config file
**Outputs:** JSON parameter lookup table in `optimization_results/`
**Note:** Independent of Step 1 (uses its own default gains as starting point).

### Step 3: RL Model Training

Trains the RL models evaluated in `experimental_results.md`. Training produces model directories under `models/` (gitignored).

**Residual SAC (PID + RL corrections):**

```bash
# Estes C6 Residual SAC (~470K steps, early-stops)
uv run python training/train_residual_sac.py \
    --config configs/estes_c6_residual_sac_wind.yaml \
    --timesteps 500000 --early-stopping 40

# J800 Residual SAC (2M steps — main result: 3.7 deg/s at 0 m/s)
uv run python training/train_residual_sac.py \
    --config configs/aerotech_j800_residual_sac_wind.yaml \
    --timesteps 2000000
```

**Standalone SAC (direct RL, no PID):**

```bash
# J800 Standalone SAC hyperparameter sweep (6 configs)
for config in configs/sac_sweep/j800_sac_a*.yaml; do
    uv run python training/train_sac.py \
        --config "$config" \
        --timesteps 2000000
done
```

**Inputs:** YAML config files. Residual SAC configs reference PID gains inline.
**Outputs:** Model directory under `models/` containing `best_model.zip`, `final_model.zip`, `config.yaml`, `vec_normalize.pkl`, and `checkpoints/` subdirectory. Directory names are auto-generated with timestamps.
**Note:** Training is stochastic. Exact spin rates will differ between runs, but the qualitative conclusions (Residual SAC >> PID on J800) should hold. J800 Residual SAC at 2M steps requires several hours on GPU.

### Step 4: Evaluate Classical Controllers

The core evaluation comparing PID, GS-PID, ADRC, Ensemble, and other classical controllers.

**Script:** `compare_controllers.py`

```bash
# Estes C6 — Full controller comparison (Section "Estes C6 — Full Controller Comparison")
# PID gains: Kp=0.0203, Ki=0.0002, Kd=0.0118 (from Step 1)
uv run python compare_controllers.py \
    --config configs/estes_c6_sac_wind.yaml \
    --pid-Kp 0.0203 --pid-Ki 0.0002 --pid-Kd 0.0118 \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 — Full controller comparison (Section "J800 — Full Controller Comparison")
# GS-PID gains: Kp=0.0213, Ki=0.0050, Kd=0.0271, q_ref=13268 (from Step 1)
uv run python compare_controllers.py \
    --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50
```

**Inputs:** YAML config. PID gains from Step 1 (passed as CLI args or use defaults).
**Outputs:** Console table with mean spin rate, std, success rate, settling time, control smoothness. Optional `--save-plot PATH` generates a PNG comparison plot.

### Step 5: Evaluate RL Models

Evaluates trained RL models from Step 3.

```bash
# Residual SAC on J800 (Section "Residual SAC — AeroTech J800T")
uv run python compare_controllers.py \
    --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --residual-sac models/<j800_residual_sac_directory>/best_model.zip \
    --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# Residual SAC on Estes (Section "Residual SAC — Estes C6")
uv run python compare_controllers.py \
    --config configs/estes_c6_sac_wind.yaml \
    --pid-Kp 0.0203 --pid-Ki 0.0002 --pid-Kd 0.0118 \
    --residual-sac models/<estes_residual_sac_directory>/best_model.zip \
    --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# Standalone SAC sweep on J800 (Section "Standalone SAC — Hyperparameter Sweep")
for model_dir in models/rocket_sac_j800_a*; do
    uv run python compare_controllers.py \
        --config configs/aerotech_j800_wind.yaml \
        --sac "$model_dir/best_model.zip" \
        --imu --wind-levels 0 1 2 3 5 --n-episodes 50
done
```

**Inputs:** YAML config, trained model `.zip` file from Step 3, `vec_normalize.pkl` (auto-loaded from model directory).
**Outputs:** Same console table as Step 4.
**Note:** Model directory names contain timestamps. Replace `<j800_residual_sac_directory>` with the actual directory name from your training run.

### Step 6: Hardware Parameter Studies

Re-runs evaluation with hardware-variant configs (Estes C6 only).

```bash
# Each config uses gains from its own optimization (Step 1)
uv run python compare_controllers.py \
    --config configs/estes_c6_4fin.yaml \
    --pid-Kp 0.0177 --pid-Ki 0.0001 --pid-Kd 0.0054 \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_200hz.yaml \
    --pid-Kp 0.0203 --pid-Ki 0.0002 --pid-Kd 0.0118 \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_500hz.yaml \
    --pid-Kp 0.0203 --pid-Ki 0.0002 --pid-Kd 0.0118 \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_4fin_200hz.yaml \
    --pid-Kp 0.0237 --pid-Ki 0.0043 --pid-Kd 0.0178 \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_tab10.yaml \
    --pid-Kp 0.0079 --pid-Ki 0.0060 --pid-Kd 0.0148 --pid-qref 5549 \
    --gain-scheduled --imu --wind-levels 0 1 2 3 5 --n-episodes 50
```

### Step 7: Dryden Turbulence Validation

Re-runs evaluation under MIL-HDBK-1797 Dryden continuous turbulence.

```bash
# Estes C6 — Dryden light/moderate/severe (uses per-severity optimized gains from Step 1)
uv run python compare_controllers.py \
    --config configs/estes_c6_dryden_light.yaml \
    --pid-Kp 0.0108 --pid-Ki 0.0033 --pid-Kd 0.0088 \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_dryden_moderate.yaml \
    --pid-Kp 0.0218 --pid-Ki 0.0052 --pid-Kd 0.0074 \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

uv run python compare_controllers.py \
    --config configs/estes_c6_dryden_severe.yaml \
    --pid-Kp 0.0017 --pid-Ki 0.0052 --pid-Kd 0.0154 \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 — Dryden moderate (RL models generalization test)
uv run python compare_controllers.py \
    --config configs/aerotech_j800_dryden_moderate.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --imu \
    --residual-sac models/<j800_residual_sac_directory>/best_model.zip \
    --wind-levels 0 1 2 3 5 --n-episodes 50
```

### Step 8: Video Quality Analysis

```bash
uv run python compare_controllers.py \
    --config configs/estes_c6_sac_wind.yaml \
    --pid-Kp 0.0203 --pid-Ki 0.0002 --pid-Kd 0.0118 \
    --gain-scheduled --imu --video-quality --camera-preset all \
    --wind-levels 0 1 2 3 5 --n-episodes 20
```

### Step 9: Bayesian Optimized Evaluation

Evaluates controllers using per-condition optimized parameters from Step 2.

```bash
# Estes — BO GS-PID
uv run python compare_controllers.py \
    --config configs/estes_c6_sac_wind.yaml \
    --optimized-params optimization_results/bo_gs_pid_params.json \
    --imu --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 — BO GS-PID
uv run python compare_controllers.py \
    --config configs/aerotech_j800_wind.yaml \
    --optimized-params optimization_results/bo_gs_pid_j800_params.json \
    --imu --wind-levels 0 1 2 3 5 --n-episodes 50
```

---

## Known Gotchas

1. **Stochastic results.** All evaluations use random wind episodes. Exact numbers will not match `experimental_results.md` precisely, but should be within the reported standard deviations. Use `--n-episodes 50` or more for stable statistics.

2. **PID gains must match.** The documented results use specific optimized gains. Using default gains (which differ) will produce different numbers. Always pass the gains listed in `experimental_results.md` via `--pid-Kp`, `--pid-Ki`, `--pid-Kd`, and `--pid-qref`.

3. **RL model paths contain timestamps.** Training generates directories like `models/rocket_residual_sac_j800_wind_aerotech_j800t_20260209_222006/`. The timestamp in your run will differ. Replace model paths accordingly.

4. **J800 Residual SAC training is long.** 2M timesteps on J800 can take several hours on GPU. The 500K-step checkpoint (auto-saved) is also usable but will show ~4.8 deg/s at 0 m/s instead of the 3.7 deg/s reported for 2M steps.

5. **ADRC catastrophically fails on J800.** Do not pass `--adrc` when evaluating J800 unless you specifically want to reproduce the 0% success rate result. It will not break anything, but episodes will hit the spin-out termination condition.

6. **Dryden configs use different PID gains.** Each Dryden severity level has its own optimized PID gains (Step 1). Using the sinusoidal-wind gains will produce worse results.

7. **`models/` and `optimization_results/` are gitignored.** You must run Steps 1-3 to generate these artifacts, or obtain them separately. The code itself (configs, scripts, modules) is all in git.

8. **The `vec_normalize.pkl` file matters.** RL models use observation normalization. The `vec_normalize.pkl` in the model directory is auto-loaded when you pass a model to `compare_controllers.py`. If it is missing, the model will perform poorly.

---

## Optimized Gains Reference

| Platform / Config | Kp | Ki | Kd | q_ref |
|-------------------|------|--------|--------|-------|
| Estes C6 (PID) | 0.0203 | 0.0002 | 0.0118 | -- |
| J800 (GS-PID) | 0.0213 | 0.0050 | 0.0271 | 13268 |
| Estes 4-fin | 0.0177 | 0.0001 | 0.0054 | -- |
| Estes 4-fin+200Hz | 0.0237 | 0.0043 | 0.0178 | -- |
| Estes Tab 10 | 0.0079 | 0.0060 | 0.0148 | 5549 |
| Dryden Light | 0.0108 | 0.0033 | 0.0088 | -- |
| Dryden Moderate | 0.0218 | 0.0052 | 0.0074 | -- |
| Dryden Severe | 0.0017 | 0.0052 | 0.0154 | -- |

---

## Running Tests

```bash
uv run python -m pytest tests/ -v
```

---

## Project Structure

```
compare_controllers.py          Main evaluation tool (Step 4-9)
realistic_spin_rocket.py        6-DOF physics with real motor data
spin_stabilized_control_env.py  Gymnasium environment for RL training
rocket_config.py                YAML config loading and dataclasses
wind_model.py                   Sinusoidal + Dryden wind disturbance models
motor_loader.py                 Motor thrust curve loading
thrustcurve_motor_data.py       ThrustCurve.org data parsing

controllers/
  pid_controller.py             PID, GS-PID, Lead-compensated GS-PID
  adrc_controller.py            ADRC with Extended State Observer
  ensemble_controller.py        GS-PID + ADRC online switching
  disturbance_observer.py       DOB for RL training wrappers
  video_quality_metric.py       Gyroflow post-stabilization quality analysis

training/
  train_improved.py             PPO training + environment factory (create_environment)
  train_sac.py                  SAC training with wind curriculum (Step 3)
  train_residual_sac.py         Residual SAC training (Step 3)
  sweep_hyperparams.py          Hyperparameter sweep automation

optimization/
  optimize_pid.py               LHS + Nelder-Mead gain optimization (Step 1)
  bayesian_optimize.py          Per-condition Bayesian optimization (Step 2)

airframe/
  airframe.py                   RocketAirframe class (geometry, mass, aerodynamics)
  components.py                 NoseCone, BodyTube, TrapezoidFinSet
  openrocket_parser.py          OpenRocket .ork file parser

rocket_env/
  sensors/                      IMU noise simulation (ICM-20948)
    gyro_model.py               Gyroscope noise model (ARW, bias, drift)
    imu_config.py               IMU configuration presets
    imu_wrapper.py              Gymnasium ObservationWrapper for IMU noise
  inference/                    ONNX inference for embedded deployment
    controller.py               Deployment-ready controller wrappers
    onnx_runner.py              Lightweight ONNX model runner

configs/                        YAML environment configs
  estes_c6_sac_wind.yaml        Main Estes config
  aerotech_j800_wind.yaml       Main J800 config
  estes_c6_dryden_*.yaml        Dryden turbulence variants
  estes_c6_{4fin,200hz,...}.yaml Hardware study variants
  sac_sweep/                    Standalone SAC hyperparameter sweep configs
  airframes/                    Airframe YAML definitions

models/                         Trained RL models (gitignored, generated by Step 3)
optimization_results/           Gain optimization JSON outputs (gitignored, generated by Steps 1-2)
tests/                          Pytest test suite
visualizations/                 Plotting and animation scripts
deployment/                     ONNX export for Raspberry Pi deployment
```

---

## Hardware Deployment

See [`deployment/RASPBERRY_PI_GUIDE.md`](deployment/RASPBERRY_PI_GUIDE.md) for deploying to Raspberry Pi Zero 2 W with ICM-20948 IMU and micro servos. Camera system documented in [`camera_electronics/`](camera_electronics/).
