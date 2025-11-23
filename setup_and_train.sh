#!/bin/bash
# setup_and_train.sh - Complete setup and training script

echo "==================================="
echo "Setting up Rocket Control RL Environment"
echo "==================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment and installing dependencies..."
uv venv
uv pip install -e .

# Create necessary directories
mkdir -p models logs motors

echo ""
echo "==================================="
echo "Setup Complete! Choose your training:"
echo "==================================="
echo ""
echo "1. Train Original 6DOF Rocket Control"
echo "   uv run python train_ppo.py --env original"
echo ""
echo "2. Train Spin-Stabilized Camera Rocket (Basic)"
echo "   uv run python train_ppo.py --env spin"
echo ""
echo "3. Train Spin-Stabilized Camera Rocket (Estes C6 Motor)"
echo "   uv run python train_ppo.py --env spin --motor estes_c6"
echo ""
echo "4. Train Spin-Stabilized Camera Rocket (Aerotech F40 Motor)"
echo "   uv run python train_ppo.py --env spin --motor aerotech_f40"
echo ""
echo "5. Monitor Training Progress (in another terminal)"
echo "   uv run tensorboard --logdir logs"
echo ""

# Ask user what to train
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        uv run python train_ppo.py --env original
        ;;
    2)
        uv run python train_ppo.py --env spin
        ;;
    3)
        uv run python train_ppo.py --env spin --motor estes_c6
        ;;
    4)
        uv run python train_ppo.py --env spin --motor aerotech_f40
        ;;
    5)
        echo "Starting TensorBoard..."
        uv run tensorboard --logdir logs
        ;;
    *)
        echo "Invalid choice. Please run the commands manually."
        ;;
esac
