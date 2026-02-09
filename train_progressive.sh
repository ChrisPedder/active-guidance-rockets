#!/bin/bash
#
# Progressive Training Pipeline for Rocket Spin Control
#
# Trains sequentially from easy -> medium -> full difficulty,
# fine-tuning each stage from the previous one.
#
# Usage:
#   ./train_progressive.sh                    # Run full pipeline
#   ./train_progressive.sh --stage easy       # Run only easy stage
#   ./train_progressive.sh --stage medium     # Run medium (requires easy model)
#   ./train_progressive.sh --stage full       # Run full (requires medium model)
#   ./train_progressive.sh --timesteps 2000000  # Override timesteps per stage
#

set -e  # Exit on error

# Default settings
MOTOR="estes_c6"
TIMESTEPS_EASY=1500000      # More steps for from-scratch training
TIMESTEPS_MEDIUM=1000000    # Fine-tuning needs fewer steps
TIMESTEPS_FULL=1000000
LEARNING_RATE=""            # Empty = use config default
STAGE="all"                 # Which stage(s) to run

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --motor)
            MOTOR="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS_EASY="$2"
            TIMESTEPS_MEDIUM="$2"
            TIMESTEPS_FULL="$2"
            shift 2
            ;;
        --timesteps-easy)
            TIMESTEPS_EASY="$2"
            shift 2
            ;;
        --timesteps-medium)
            TIMESTEPS_MEDIUM="$2"
            shift 2
            ;;
        --timesteps-full)
            TIMESTEPS_FULL="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --motor NAME           Motor name (default: estes_c6)"
            echo "  --timesteps N          Timesteps for all stages"
            echo "  --timesteps-easy N     Timesteps for easy stage (default: 1500000)"
            echo "  --timesteps-medium N   Timesteps for medium stage (default: 1000000)"
            echo "  --timesteps-full N     Timesteps for full stage (default: 1000000)"
            echo "  --lr RATE              Override learning rate"
            echo "  --stage STAGE          Run specific stage: easy, medium, full, or all"
            echo "  -h, --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build learning rate flag if specified
LR_FLAG=""
if [[ -n "$LEARNING_RATE" ]]; then
    LR_FLAG="--lr $LEARNING_RATE"
fi

echo "========================================================================"
echo "PROGRESSIVE TRAINING PIPELINE"
echo "========================================================================"
echo "Motor: $MOTOR"
echo "Stage: $STAGE"
echo "Timesteps: Easy=$TIMESTEPS_EASY, Medium=$TIMESTEPS_MEDIUM, Full=$TIMESTEPS_FULL"
if [[ -n "$LEARNING_RATE" ]]; then
    echo "Learning rate override: $LEARNING_RATE"
fi
echo "========================================================================"
echo ""

# Helper function to find the most recent model
find_latest_model() {
    local pattern="$1"
    # Find most recently modified best_model.zip matching pattern
    local model=$(ls -t $pattern 2>/dev/null | head -1)
    echo "$model"
}

# Helper function to train a stage
train_stage() {
    local stage="$1"
    local config="$2"
    local timesteps="$3"
    local load_model="$4"

    echo ""
    echo "========================================================================"
    echo "STAGE: $stage"
    echo "Config: $config"
    echo "Timesteps: $timesteps"
    if [[ -n "$load_model" ]]; then
        echo "Loading model: $load_model"
    fi
    echo "========================================================================"
    echo ""

    # Build command
    local cmd="uv run python training/train_improved.py --config $config --timesteps $timesteps $LR_FLAG"
    if [[ -n "$load_model" ]]; then
        cmd="$cmd --load-model $load_model"
    fi

    echo "Running: $cmd"
    echo ""

    # Execute training
    eval $cmd

    echo ""
    echo "Stage $stage complete!"
    echo ""
}

# ============================================================================
# STAGE 1: EASY (from scratch)
# ============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "easy" ]]; then
    CONFIG_EASY="configs/${MOTOR}_easy.yaml"

    if [[ ! -f "$CONFIG_EASY" ]]; then
        echo "Config not found: $CONFIG_EASY"
        echo "Generating configs..."
        uv run python generate_motor_config.py generate "$MOTOR" --output configs/
    fi

    train_stage "EASY" "$CONFIG_EASY" "$TIMESTEPS_EASY" ""
fi

# ============================================================================
# STAGE 2: MEDIUM (fine-tune from easy)
# ============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "medium" ]]; then
    CONFIG_MEDIUM="configs/${MOTOR}_medium.yaml"

    if [[ ! -f "$CONFIG_MEDIUM" ]]; then
        echo "Config not found: $CONFIG_MEDIUM"
        echo "Generating configs..."
        uv run python generate_motor_config.py generate "$MOTOR" --output configs/
    fi

    # Find the easy model to load
    EASY_MODEL=$(find_latest_model "models/rocket_${MOTOR}_easy_*/best_model.zip")

    if [[ -z "$EASY_MODEL" ]]; then
        echo "ERROR: No easy model found. Run easy stage first."
        echo "Looking for: models/rocket_${MOTOR}_easy_*/best_model.zip"
        exit 1
    fi

    train_stage "MEDIUM" "$CONFIG_MEDIUM" "$TIMESTEPS_MEDIUM" "$EASY_MODEL"
fi

# ============================================================================
# STAGE 3: FULL (fine-tune from medium)
# ============================================================================
if [[ "$STAGE" == "all" || "$STAGE" == "full" ]]; then
    CONFIG_FULL="configs/${MOTOR}_full.yaml"

    if [[ ! -f "$CONFIG_FULL" ]]; then
        echo "Config not found: $CONFIG_FULL"
        echo "Generating configs..."
        uv run python generate_motor_config.py generate "$MOTOR" --output configs/
    fi

    # Find the medium model to load
    MEDIUM_MODEL=$(find_latest_model "models/rocket_${MOTOR}_medium_*/best_model.zip")

    if [[ -z "$MEDIUM_MODEL" ]]; then
        echo "ERROR: No medium model found. Run medium stage first."
        echo "Looking for: models/rocket_${MOTOR}_medium_*/best_model.zip"
        exit 1
    fi

    train_stage "FULL" "$CONFIG_FULL" "$TIMESTEPS_FULL" "$MEDIUM_MODEL"
fi

echo ""
echo "========================================================================"
echo "PROGRESSIVE TRAINING COMPLETE"
echo "========================================================================"
echo ""
echo "Models saved in: models/"
echo ""
echo "To evaluate the final model:"
FINAL_MODEL=$(find_latest_model "models/rocket_${MOTOR}_full_*/best_model.zip")
if [[ -z "$FINAL_MODEL" ]]; then
    FINAL_MODEL=$(find_latest_model "models/rocket_${MOTOR}_medium_*/best_model.zip")
fi
if [[ -z "$FINAL_MODEL" ]]; then
    FINAL_MODEL=$(find_latest_model "models/rocket_${MOTOR}_easy_*/best_model.zip")
fi
if [[ -n "$FINAL_MODEL" ]]; then
    FINAL_DIR=$(dirname "$FINAL_MODEL")
    echo "  uv run python visualizations/visualize_spin_agent.py $FINAL_MODEL --config $FINAL_DIR/config.yaml"
fi
echo ""
echo "To view tensorboard:"
echo "  uv run tensorboard --logdir logs/"
echo "========================================================================"
