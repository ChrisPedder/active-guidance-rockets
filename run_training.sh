#!/bin/bash
# run_training.sh - Training script with uv

echo "==================================="
echo "Rocket Control RL Training"
echo "==================================="

# Function to display help
show_help() {
    echo "Usage: ./run_training.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --original        Train original 6DOF rocket environment (default)"
    echo "  --spin           Train spin-stabilized camera rocket"
    echo "  --motor NAME     Motor configuration for spin env"
    echo "                   Options: estes_c6, aerotech_f40, cesaroni_g79"
    echo "  --timesteps N    Number of training timesteps (default: 500000)"
    echo "  --device DEVICE  Training device: auto, cpu, cuda (default: auto)"
    echo "  --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_training.sh --original"
    echo "  ./run_training.sh --spin --motor estes_c6"
    echo "  ./run_training.sh --spin --motor aerotech_f40 --timesteps 1000000"
}

# Default values
ENV_TYPE="original"
MOTOR=""
TIMESTEPS=500000
DEVICE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original)
            ENV_TYPE="original"
            shift
            ;;
        --spin)
            ENV_TYPE="spin"
            shift
            ;;
        --motor)
            MOTOR="$2"
            shift 2
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --help)
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

# Build command
CMD="uv run python train_unified.py --env $ENV_TYPE --timesteps $TIMESTEPS --device $DEVICE"

if [ -n "$MOTOR" ]; then
    CMD="$CMD --motor $MOTOR"
fi

echo "Running command: $CMD"
echo ""

# Run training
$CMD
