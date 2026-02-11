# Active Guidance Rockets — Project Context

Spin stabilization system for model rockets with onboard cameras. Minimizes roll rate during flight using adjustable fin tabs that create differential drag.

**Rockets:** Estes C6 (~10 N-s, 1.85s burn) and AeroTech J800T (~800 N-s)
**Primary goal:** Mean spin rate < 5 deg/s (calm), graceful degradation up to 3 m/s wind
**Recommended controller:** Gain-Scheduled PID (GS-PID)

---

## Results (50 episodes, IMU noise, Feb 2026)

### Estes C6

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 3.7 ± 0.6 | **3.4 ± 0.5** | **3.4 ± 0.5** | < 5 |
| 1 | 7.6 ± 3.8 | **6.0 ± 2.7** | 6.6 ± 2.7 | < 10 |
| 2 | 10.4 ± 5.4 | 9.5 ± 5.2 | **9.4 ± 4.9** | < 15 |
| 3 | 14.7 ± 7.5 | **11.7 ± 6.7** | 11.5 ± 6.1 | < 20 |
| 5 | **14.1 ± 7.5** | 17.8 ± 8.6 | 14.8 ± 7.6 | — |

All targets met on Estes up to 3 m/s. PID is most robust at 5 m/s.

### AeroTech J800T

| Wind (m/s) | PID | GS-PID | Residual SAC (2M) | Standalone SAC | Target |
|------------|-----------|-----------|-------------------|----------------|--------|
| 0 | 12.8 ± 0.8 | 10.5 ± 0.8 | **3.7 ± 0.1** | 5.4 ± 0.1 | < 5 |
| 1 | 13.2 ± 0.9 | 11.0 ± 0.8 | **3.9 ± 0.3** | 5.6 ± 0.2 | < 10 |
| 2 | 14.4 ± 1.8 | 11.3 ± 0.9 | **4.0 ± 0.4** | 5.7 ± 0.4 | < 15 |
| 3 | 14.6 ± 1.8 | 11.7 ± 1.2 | **4.5 ± 0.8** | 6.1 ± 0.6 | < 20 |
| 5 | 15.9 ± 2.8 | 12.7 ± 1.7 | **5.6 ± 1.5** | 6.8 ± 1.2 | — |

Residual SAC (2M steps) meets < 5 target at 0-2 m/s. Standalone SAC (alpha=0.5, ent_coef=0.01, 2M steps) also beats PID by 2-3x. Classical controllers (GS-PID) plateau at ~10.5 deg/s.

---

## Key Design Decisions

- **Simple controllers win (mostly).** 17 advanced approaches were tested — most eliminated. Residual SAC and standalone SAC (with correct hyperparameters) beat PID on J800 by 2-3x, but require careful tuning. See `experimental_results.md` for the full record.
- **Gain scheduling** addresses the ~20x variation in control effectiveness during flight (varying dynamic pressure).
- **Wind torque is periodic** at the spin frequency (`torque ∝ sin(wind_dir - roll_angle)`). No feedforward/estimation approach reliably tracked this.
- **Hardware over algorithms:** 4-fin + 200 Hz GS-PID (8.4 deg/s at 3 m/s) outperforms every classical controller at baseline hardware. Residual SAC (4.5 deg/s at 3 m/s) now surpasses even hardware upgrades.
- **IMU noise is negligible.** ICM-20948 at 100 Hz → ~0.15 deg/s RMS, irrelevant vs 5-30 deg/s spin rates.

### Known Issues

- J800 classical controllers (GS-PID: 10.5 deg/s at 0 m/s) don't meet the < 5 target — only Residual SAC achieves this (3.7 deg/s)
- ADRC is retained as a research baseline only — it catastrophically fails on J800

---

## Repository Structure

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
├── controllers/
│   ├── pid_controller.py           # PID and Gain-Scheduled PID
│   ├── adrc_controller.py          # ADRC with dynamic b0 (research baseline)
│   ├── ensemble_controller.py      # GS-PID + ADRC online switching
│   ├── disturbance_observer.py     # DOB for training wrappers
│   └── video_quality_metric.py     # Gyroflow post-stabilization analysis
│
├── training/
│   ├── train_improved.py           # PPO training
│   ├── train_sac.py                # SAC training with wind curriculum
│   ├── train_residual_sac.py       # Residual SAC (PID + RL corrections)
│   └── sweep_hyperparams.py        # Hyperparameter sweep automation
│
├── optimization/
│   ├── optimize_pid.py             # LHS + Nelder-Mead PID gain optimization
│   └── bayesian_optimize.py        # Per-condition Bayesian optimization
│
├── airframe/
│   ├── airframe.py                 # RocketAirframe class
│   ├── components.py               # NoseCone, BodyTube, TrapezoidFinSet
│   └── openrocket_parser.py        # .ork file parser
│
├── rocket_env/
│   ├── inference/                  # ONNX inference for embedded deployment
│   │   ├── controller.py
│   │   └── onnx_runner.py
│   └── sensors/                    # Sensor simulation
│       ├── gyro_model.py
│       ├── imu_config.py
│       └── imu_wrapper.py
│
├── configs/
│   ├── estes_c6_sac_wind.yaml      # Main Estes config (sinusoidal wind)
│   ├── aerotech_j800_wind.yaml     # J800 config
│   ├── estes_c6_dryden_*.yaml      # Estes Dryden turbulence (light/moderate/severe)
│   ├── aerotech_j800_dryden_moderate.yaml  # J800 Dryden moderate turbulence
│   ├── estes_c6_4fin.yaml          # Hardware study: 4 active fins
│   ├── estes_c6_200hz.yaml         # Hardware study: 200 Hz loop
│   ├── estes_c6_500hz.yaml         # Hardware study: 500 Hz loop
│   ├── estes_c6_4fin_200hz.yaml    # Hardware study: combined
│   ├── estes_c6_tab{10,15,25,30}.yaml  # Tab deflection sweep
│   ├── estes_c6_bigtab.yaml        # 4x tab area
│   ├── estes_c6_residual*.yaml     # Residual RL configs
│   ├── estes_c6_dob_sac.yaml       # DOB-SAC config
│   └── airframes/                  # Airframe YAML definitions
│
├── optimization_results/           # Stored gain optimization results (JSON)
├── tests/                          # 30 test files, 662 tests
├── visualizations/                 # Motor & agent visualization scripts
├── scripts/                        # export_onnx.py
├── docs/                           # Wind torque analysis
├── camera_electronics/             # Hardware modification guides
├── rocket-fin-servo-mount/         # Mechanical design docs
├── slides/                         # Presentation materials
├── models/                         # Trained RL models & wind estimator weights
├── sweeps/                         # Hyperparameter sweep results
└── evaluation_results/             # Past evaluation plots/reports
```

---

## Key Commands

```bash
# Full comparison (Estes, all retained controllers, IMU)
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 comparison
uv run python compare_controllers.py --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# PID gain optimization
uv run python optimization/optimize_pid.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization.json

# Video quality analysis
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --imu --video-quality --camera-preset all \
    --wind-levels 0 1 2 3 5 --n-episodes 20

# Run tests
uv run python -m pytest tests/ -v
```

## Optimized Gains

| Platform | Kp | Ki | Kd | q_ref |
|----------|------|--------|--------|-------|
| Estes C6 (PID) | 0.0203 | 0.0002 | 0.0118 | — |
| J800 (GS-PID) | 0.0213 | 0.0050 | 0.0271 | 13268 |

## Hardware Parameters (from `configs/estes_c6_sac_wind.yaml`)

- `max_tab_deflection: 30` (degrees)
- `tab_chord_fraction: 0.25`, `tab_span_fraction: 0.5`
- `num_controlled_fins: 2` (of 4 total)
- `dt: 0.01` (100 Hz control loop)
