#!/bin/bash
#
# run_experiment.sh - Complete Rocket Spin Control Training & Evaluation Pipeline
#
# This script automates the full workflow:
# 0. Generate motor config (optional - from ThrustCurve.org data)
# 1. Visualize motor characteristics
# 2. Run diagnostics
# 3. Train the agent
# 4. Evaluate and visualize results
#
# Usage:
#   ./run_experiment.sh --motor estes_c6 --difficulty easy --timesteps 500000
#   ./run_experiment.sh --motor aerotech_f40 --difficulty easy --generate-config
#   ./run_experiment.sh --motor aerotech_h128 --generate-config --dry-mass 2.5
#   ./run_experiment.sh --help
#

set -e  # Exit on error

# Default values
MOTOR="estes_c6"
DIFFICULTY="easy"
TIMESTEPS=500000
EVAL_EPISODES=50
N_ENVS=8
OUTPUT_DIR="experiments"
SKIP_TRAINING=false
SKIP_MOTOR_VIZ=false
SKIP_EVAL=false
GENERATE_CONFIG=false
DRY_MASS=""
MODEL_PATH=""
CONFIG_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print colored message
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Help message
show_help() {
    cat << EOF
Rocket Spin Control Training Pipeline
=====================================

Automates the complete workflow from motor selection to trained agent evaluation.

Usage: $0 [OPTIONS]

MOTOR OPTIONS:
    -m, --motor MOTOR       Motor name/key. Built-in options:
                              estes_a8, estes_b6, estes_c6, estes_d12
                              aerotech_f40, aerotech_h128
                              cesaroni_g79
                            Or any motor from ThrustCurve.org with --generate-config
                            (default: estes_c6)

    -d, --difficulty LEVEL  Difficulty: easy, medium, full, or all
                            (default: easy)

CONFIG OPTIONS:
    -g, --generate-config   Auto-generate config from motor data if not found
                            Uses generate_motor_config.py to create physics-tuned config

    --dry-mass KG           Override rocket dry mass (kg) for config generation
                            If not specified, auto-calculates for TWR ≈ 5

    -c, --config FILE       Use specific config file (overrides motor/difficulty)

TRAINING OPTIONS:
    -t, --timesteps N       Total training timesteps (default: 500000)
    -n, --n-envs N          Number of parallel environments (default: 8)
    --eval-only             Skip training, only run evaluation
    --model-path PATH       Path to existing model for evaluation

OUTPUT OPTIONS:
    -o, --output-dir DIR    Output directory for results (default: experiments/)
    -e, --eval-episodes N   Number of evaluation episodes (default: 50)
    --skip-motor-viz        Skip motor visualization step
    --skip-eval             Skip evaluation step

    -h, --help              Show this help message

EXAMPLES:
    # Basic training with built-in motor
    $0 --motor estes_c6 --difficulty easy --timesteps 500000

    # Auto-generate config for any motor
    $0 --motor aerotech_h128 --generate-config --timesteps 500000

    # Generate config with custom dry mass
    $0 --motor cesaroni_g79 --generate-config --dry-mass 0.9 --timesteps 300000

    # Generate all difficulty levels
    $0 --motor aerotech_f40 --generate-config --difficulty all

    # Evaluate existing model
    $0 --eval-only --model-path models/best_model.zip --config configs/estes_c6_easy.yaml

    # Quick test run
    $0 --motor estes_c6 --difficulty easy --timesteps 50000

WORKFLOW:
    0. Config Generation    - Create physics-tuned config from motor data (optional)
    1. Motor Visualization  - Plot thrust curves and characteristics
    2. Diagnostics          - Verify environment configuration
    3. Training             - Train PPO agent
    4. Evaluation           - Run evaluation episodes
    5. Visualization        - Generate performance plots and reports

MOTOR CONFIG GENERATION:
    The --generate-config flag uses generate_motor_config.py to:
    - Look up motor specifications (offline database or ThrustCurve.org API)
    - Analyze physics to determine safe training parameters
    - Auto-calculate appropriate tab deflection, damping, timestep
    - Create difficulty-appropriate reward scaling

    This ensures configs are properly tuned for each motor's characteristics.

    List available offline motors:
      python generate_motor_config.py list-popular

    Search ThrustCurve.org (requires 'requests' library):
      python generate_motor_config.py search --impulse-class G

    Verify a motor exists:
      python generate_motor_config.py verify "Estes D12"

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--motor)
            MOTOR="$2"
            shift 2
            ;;
        -d|--difficulty)
            DIFFICULTY="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -t|--timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        -e|--eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        -n|--n-envs)
            N_ENVS="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--generate-config)
            GENERATE_CONFIG=true
            shift
            ;;
        --dry-mass)
            DRY_MASS="$2"
            shift 2
            ;;
        --eval-only)
            SKIP_TRAINING=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --skip-motor-viz)
            SKIP_MOTOR_VIZ=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate difficulty
if [[ ! "$DIFFICULTY" =~ ^(easy|medium|full|all)$ ]]; then
    print_error "Invalid difficulty: $DIFFICULTY"
    echo "Valid options: easy, medium, full, all"
    exit 1
fi

# Normalize motor name (lowercase, underscores)
MOTOR_NORMALIZED=$(echo "$MOTOR" | tr '[:upper:]' '[:lower:]' | tr ' -' '_')

# Determine config file path
if [[ -n "$CONFIG_FILE" ]]; then
    # User specified explicit config
    if [[ ! -f "$CONFIG_FILE" ]]; then
        print_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
elif [[ "$DIFFICULTY" == "all" ]]; then
    # Will generate/use multiple configs
    CONFIG_FILE="configs/${MOTOR_NORMALIZED}_easy.yaml"
else
    CONFIG_FILE="configs/${MOTOR_NORMALIZED}_${DIFFICULTY}.yaml"
fi

# ════════════════════════════════════════════════════════════════
# STEP 0: Config Generation (if needed)
# ════════════════════════════════════════════════════════════════

# Check if config exists or needs generation
CONFIG_EXISTS=true
if [[ "$DIFFICULTY" == "all" ]]; then
    # Check if any configs exist for this motor
    if [[ ! -f "configs/${MOTOR_NORMALIZED}_easy.yaml" ]]; then
        CONFIG_EXISTS=false
    fi
else
    if [[ ! -f "$CONFIG_FILE" ]]; then
        CONFIG_EXISTS=false
    fi
fi

if [[ "$CONFIG_EXISTS" == false ]]; then
    if [[ "$GENERATE_CONFIG" == true ]]; then
        print_header "STEP 0: Generating Motor Configuration"

        print_step "Verifying motor: $MOTOR"

        # Build generation command
        GEN_CMD="uv run python generate_motor_config.py generate $MOTOR_NORMALIZED --output configs/"

        if [[ "$DIFFICULTY" != "all" ]]; then
            GEN_CMD="$GEN_CMD --difficulty $DIFFICULTY"
        fi

        if [[ -n "$DRY_MASS" ]]; then
            GEN_CMD="$GEN_CMD --dry-mass $DRY_MASS"
        fi

        print_step "Running: $GEN_CMD"
        eval $GEN_CMD

        # Verify config was created
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_error "Config generation failed - config file not created"
            exit 1
        fi

        print_success "Config generated: $CONFIG_FILE"
    else
        print_error "Config file not found: $CONFIG_FILE"
        echo ""
        echo "Options:"
        echo "  1. Add --generate-config to auto-generate from motor data"
        echo "  2. Create config manually in configs/ directory"
        echo "  3. Use --config to specify an existing config file"
        echo ""
        echo "Example:"
        echo "  $0 --motor $MOTOR --difficulty $DIFFICULTY --generate-config"
        echo ""
        echo "Available offline motors:"
        uv run python generate_motor_config.py list-popular 2>/dev/null || echo "  Run: python generate_motor_config.py list-popular"
        exit 1
    fi
else
    print_info "Using existing config: $CONFIG_FILE"
fi

# Handle "all" difficulty - train easy first, then can continue manually
if [[ "$DIFFICULTY" == "all" ]]; then
    print_info "Difficulty 'all' specified - will train with 'easy' config first"
    print_info "Generated configs: easy, medium, full"
    print_info "After training, progress with: --config configs/${MOTOR_NORMALIZED}_medium.yaml --load-model ..."
    DIFFICULTY="easy"
    CONFIG_FILE="configs/${MOTOR_NORMALIZED}_easy.yaml"
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_DIR="${OUTPUT_DIR}/${MOTOR_NORMALIZED}_${DIFFICULTY}_${TIMESTAMP}"
mkdir -p "$EXPERIMENT_DIR"
mkdir -p "$EXPERIMENT_DIR/plots"
mkdir -p "$EXPERIMENT_DIR/logs"

# Copy config to experiment directory
cp "$CONFIG_FILE" "$EXPERIMENT_DIR/config.yaml"

# Save experiment configuration
cat > "$EXPERIMENT_DIR/experiment_config.txt" << EOF
Experiment Configuration
========================
Date: $(date)
Motor: $MOTOR ($MOTOR_NORMALIZED)
Difficulty: $DIFFICULTY
Config File: $CONFIG_FILE
Config Generated: $GENERATE_CONFIG
Dry Mass Override: ${DRY_MASS:-"auto"}
Timesteps: $TIMESTEPS
Eval Episodes: $EVAL_EPISODES
N Envs: $N_ENVS
Output Dir: $EXPERIMENT_DIR
EOF

print_header "ROCKET SPIN CONTROL EXPERIMENT"
echo "Motor:       $MOTOR ($MOTOR_NORMALIZED)"
echo "Difficulty:  $DIFFICULTY"
echo "Config:      $CONFIG_FILE"
echo "Timesteps:   $TIMESTEPS"
echo "Output:      $EXPERIMENT_DIR"
echo ""

# ════════════════════════════════════════════════════════════════
# STEP 1: Motor Visualization
# ════════════════════════════════════════════════════════════════
if [[ "$SKIP_MOTOR_VIZ" == false ]]; then
    print_header "STEP 1: Motor Visualization"

    print_step "Generating motor profile..."
    if uv run python visualize_motor.py \
        --motor "$MOTOR_NORMALIZED" \
        --save "$EXPERIMENT_DIR/plots/motor_profile.png" \
        --no-show 2>/dev/null; then
        print_success "Motor profile saved"
    else
        print_info "Motor visualization skipped (motor not in visualize_motor.py database)"
    fi

    print_step "Generating motor comparison..."
    if uv run python visualize_motor.py \
        --compare estes_c6 aerotech_f40 cesaroni_g79 \
        --save "$EXPERIMENT_DIR/plots/motor_comparison.png" \
        --no-show 2>/dev/null; then
        print_success "Motor comparison saved"
    fi

    print_success "Motor visualizations saved to $EXPERIMENT_DIR/plots/"
else
    echo "Skipping motor visualization..."
fi

# ════════════════════════════════════════════════════════════════
# STEP 2: Environment Diagnostics
# ════════════════════════════════════════════════════════════════
print_header "STEP 2: Environment Diagnostics"

print_step "Running environment diagnostics..."

# Extract motor name from config for diagnostics
MOTOR_FROM_CONFIG=$(grep -A5 "^motor:" "$CONFIG_FILE" | grep "name:" | head -1 | awk '{print $2}' | tr -d '"' || echo "$MOTOR_NORMALIZED")

uv run python -c "
import sys
import yaml
import numpy as np

# Load config
with open('$CONFIG_FILE', 'r') as f:
    config_data = yaml.safe_load(f)

physics = config_data.get('physics', {})
env_cfg = config_data.get('environment', {})
motor_cfg = config_data.get('motor', {})

print('\\nConfiguration Summary')
print('=' * 50)
print(f'Motor: ${MOTOR_NORMALIZED}')
print(f'Dry mass: {physics.get(\"dry_mass\", 0.1)*1000:.0f}g')
print(f'Propellant mass: {physics.get(\"propellant_mass\", 0.01)*1000:.1f}g')
print(f'Diameter: {physics.get(\"diameter\", 0.024)*1000:.0f}mm')
print(f'Max tab deflection: {physics.get(\"max_tab_deflection\", 15.0):.1f}°')
print(f'Damping scale: {physics.get(\"damping_scale\", 1.0):.1f}')
print(f'Initial spin std: {physics.get(\"initial_spin_std\", 15.0):.1f}°/s')
print(f'Max roll rate: {physics.get(\"max_roll_rate\", 360.0):.0f}°/s')
print(f'Timestep: {env_cfg.get(\"dt\", 0.01)}s')
print('=' * 50)

# Try to run environment test
try:
    from spin_stabilized_control_env import RocketConfig
    from realistic_spin_rocket import RealisticMotorRocket, CommonMotors

    # Map motor names to CommonMotors
    motor_map = {
        'estes_c6': CommonMotors.estes_c6,
        'estes_a8': CommonMotors.estes_c6,  # Fallback
        'estes_b6': CommonMotors.estes_c6,  # Fallback
        'estes_d12': CommonMotors.estes_c6,  # Fallback
        'aerotech_f40': CommonMotors.aerotech_f40,
        'aerotech_h128': CommonMotors.aerotech_f40,  # Fallback to similar
        'cesaroni_g79': CommonMotors.cesaroni_g79,
    }

    motor_key = '$MOTOR_NORMALIZED'
    if motor_key not in motor_map:
        print(f'\\nNote: Motor {motor_key} not in environment motor database')
        print('Using generic motor for diagnostics')
        motor_func = CommonMotors.estes_c6
    else:
        motor_func = motor_map[motor_key]

    config = RocketConfig(
        dry_mass=physics.get('dry_mass', 0.1),
        diameter=physics.get('diameter', 0.024),
        max_tab_deflection=physics.get('max_tab_deflection', 15.0),
        disturbance_scale=physics.get('disturbance_scale', 0.0001),
        damping_scale=physics.get('damping_scale', 1.0),
        initial_spin_std=physics.get('initial_spin_std', 15.0),
        max_roll_rate=physics.get('max_roll_rate', 360.0),
        dt=env_cfg.get('dt', 0.01),
    )

    motor = motor_func()
    env = RealisticMotorRocket(motor, config)

    print('\\nRandom Action Survival Test')
    print('-' * 50)

    # Test random action survival
    survived = 0
    total_steps = []
    max_alts = []
    for ep in range(10):
        np.random.seed(ep * 100)
        obs, _ = env.reset()
        max_alt = 0
        for step in range(300):
            action = np.random.uniform(-1, 1, size=(1,))
            obs, _, term, trunc, info = env.step(action)
            if hasattr(env, 'altitude'):
                max_alt = max(max_alt, env.altitude)
            if term or trunc:
                break
        total_steps.append(step + 1)
        max_alts.append(max_alt)
        if step > 100:
            survived += 1

    print(f'Episodes surviving >100 steps: {survived}/10')
    print(f'Average episode length: {np.mean(total_steps):.0f} steps')
    print(f'Average max altitude: {np.mean(max_alts):.1f}m')
    print('-' * 50)

    if survived >= 8:
        print('✓ Environment is READY for training')
    elif survived >= 5:
        print('⚠ Environment is MARGINAL - training may be slow')
    else:
        print('✗ Environment needs tuning - reduce max_tab_deflection or increase damping')

except ImportError as e:
    print(f'\\nNote: Could not run full diagnostics: {e}')
    print('Environment modules not available in this context')
except Exception as e:
    print(f'\\nDiagnostics error: {e}')

print('=' * 50)
" 2>&1 | tee "$EXPERIMENT_DIR/logs/diagnostics.log"

print_success "Diagnostics complete"

# ════════════════════════════════════════════════════════════════
# STEP 3: Training
# ════════════════════════════════════════════════════════════════
if [[ "$SKIP_TRAINING" == false ]]; then
    print_header "STEP 3: Training Agent"

    print_step "Starting PPO training with $TIMESTEPS timesteps..."

    TRAIN_CMD="uv run python train_improved.py --config $CONFIG_FILE --timesteps $TIMESTEPS --n-envs $N_ENVS"

    if [[ -n "$MODEL_PATH" ]]; then
        TRAIN_CMD="$TRAIN_CMD --load-model $MODEL_PATH"
    fi

    eval $TRAIN_CMD 2>&1 | tee "$EXPERIMENT_DIR/logs/training.log"

    # Find the latest model (use ls -t for macOS compatibility instead of find -printf)
    LATEST_MODEL=$(ls -t models/*/best_model.zip 2>/dev/null | head -1)

    if [[ -n "$LATEST_MODEL" ]]; then
        MODEL_PATH="$LATEST_MODEL"
        print_success "Training complete. Model saved to: $MODEL_PATH"

        # Copy model to experiment directory
        cp "$MODEL_PATH" "$EXPERIMENT_DIR/"

        # Also copy config from model directory if exists
        MODEL_DIR=$(dirname "$MODEL_PATH")
        if [[ -f "$MODEL_DIR/config.yaml" ]]; then
            cp "$MODEL_DIR/config.yaml" "$EXPERIMENT_DIR/training_config.yaml"
        fi
    else
        print_error "No model found after training!"
        exit 1
    fi
else
    if [[ -z "$MODEL_PATH" ]]; then
        print_error "--eval-only requires --model-path"
        exit 1
    fi
    echo "Skipping training, using existing model: $MODEL_PATH"
fi

# ════════════════════════════════════════════════════════════════
# STEP 4: Evaluation
# ════════════════════════════════════════════════════════════════
if [[ "$SKIP_EVAL" == false ]] && [[ -n "$MODEL_PATH" ]]; then
    print_header "STEP 4: Agent Evaluation"

    print_step "Running $EVAL_EPISODES evaluation episodes..."

    # Use experiment config for evaluation
    EVAL_CONFIG="$EXPERIMENT_DIR/config.yaml"

    mkdir -p "$EXPERIMENT_DIR/evaluation"

    uv run python visualize_spin_agent.py \
        "$MODEL_PATH" \
        --config "$EVAL_CONFIG" \
        --n-episodes "$EVAL_EPISODES" \
        --save-dir "$EXPERIMENT_DIR/evaluation" \
        --no-show \
        2>&1 | tee "$EXPERIMENT_DIR/logs/evaluation.log"

    print_success "Evaluation complete"
else
    echo "Skipping evaluation..."
fi

# ════════════════════════════════════════════════════════════════
# STEP 5: Generate Summary
# ════════════════════════════════════════════════════════════════
print_header "STEP 5: Generating Summary"

# Extract key metrics from evaluation if available
EVAL_METRICS=""
if [[ -f "$EXPERIMENT_DIR/logs/evaluation.log" ]]; then
    EVAL_METRICS=$(grep -E "(Success Rate|Mean Altitude|Mean Spin|Camera Quality)" "$EXPERIMENT_DIR/logs/evaluation.log" | head -10 || true)
fi

cat > "$EXPERIMENT_DIR/SUMMARY.md" << EOF
# Experiment Summary

## Configuration

| Setting | Value |
|---------|-------|
| **Motor** | $MOTOR ($MOTOR_NORMALIZED) |
| **Difficulty** | $DIFFICULTY |
| **Config File** | $CONFIG_FILE |
| **Config Generated** | $GENERATE_CONFIG |
| **Dry Mass** | ${DRY_MASS:-"auto-calculated"} |
| **Training Timesteps** | $TIMESTEPS |
| **Evaluation Episodes** | $EVAL_EPISODES |
| **Date** | $(date) |

## Key Results

\`\`\`
$EVAL_METRICS
\`\`\`

## Files Generated

### Configuration
- \`config.yaml\` - Training configuration used

### Plots
- \`plots/motor_profile.png\` - Motor thrust curve and characteristics
- \`plots/motor_comparison.png\` - Comparison of all motors

### Evaluation
- \`evaluation/performance_overview_*.png\` - Agent performance metrics
- \`evaluation/best_trajectory_*.png\` - Best flight visualization
- \`evaluation/trajectory_comparison_*.png\` - Best vs worst comparison
- \`evaluation/evaluation_report_*.txt\` - Detailed text report

### Logs
- \`logs/diagnostics.log\` - Environment diagnostics output
- \`logs/training.log\` - Training progress log
- \`logs/evaluation.log\` - Evaluation output

### Models
- \`best_model.zip\` - Trained agent model

## Quick Commands

\`\`\`bash
# View TensorBoard logs
tensorboard --logdir logs/

# Run additional evaluation
python visualize_spin_agent.py $EXPERIMENT_DIR/best_model.zip \\
    --config $EXPERIMENT_DIR/config.yaml \\
    --n-episodes 100

# Continue training with medium difficulty
python train_improved.py \\
    --config configs/${MOTOR_NORMALIZED}_medium.yaml \\
    --load-model $EXPERIMENT_DIR/best_model.zip

# Visualize motor
python visualize_motor.py --motor $MOTOR_NORMALIZED
\`\`\`

## Reproduction

\`\`\`bash
# Reproduce this experiment
./run_experiment.sh \\
    --motor $MOTOR \\
    --difficulty $DIFFICULTY \\
    --timesteps $TIMESTEPS \\
    --eval-episodes $EVAL_EPISODES \\
    ${GENERATE_CONFIG:+--generate-config} \\
    ${DRY_MASS:+--dry-mass $DRY_MASS}
\`\`\`
EOF

print_success "Summary saved to $EXPERIMENT_DIR/SUMMARY.md"

# ════════════════════════════════════════════════════════════════
# Final Summary
# ════════════════════════════════════════════════════════════════
print_header "EXPERIMENT COMPLETE"

echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "Generated files:"
find "$EXPERIMENT_DIR" -type f \( -name "*.png" -o -name "*.txt" -o -name "*.zip" -o -name "*.yaml" -o -name "*.md" -o -name "*.log" \) 2>/dev/null | sort | while read f; do
    SIZE=$(du -h "$f" 2>/dev/null | cut -f1)
    echo "  - $f ($SIZE)"
done

echo ""
echo "Next steps:"
echo "  1. Review evaluation plots in $EXPERIMENT_DIR/evaluation/"
echo "  2. Check training logs: less $EXPERIMENT_DIR/logs/training.log"
echo "  3. View TensorBoard: tensorboard --logdir logs/"
if [[ "$DIFFICULTY" == "easy" ]]; then
    echo "  4. Progress to medium: ./run_experiment.sh --motor $MOTOR --difficulty medium --model-path $EXPERIMENT_DIR/best_model.zip"
fi
echo ""

print_success "All done!"
