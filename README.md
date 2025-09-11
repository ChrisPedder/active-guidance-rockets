# active-guidance-rockets
A repository for code to do active guidance for rockets.

## Requirements installation
Packages in this repo are managed with uv

### 1. Create a new virtual environment with uv
uv venv

### 2. Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

### 3. Install base dependencies
uv pip install -e .

### 4. Install with specific extras (choose what you need)
#### For RL dependencies:
uv pip install -e ".[rl]"

#### For visualization:
uv pip install -e ".[viz]"

#### For simulation:
uv pip install -e ".[sim]"

#### For development dependencies:
uv pip install -e ".[dev]"

#### Or install everything:
uv pip install -e ".[all,dev]"

## Running Training

### 1. Run scripts using uv:
bash# Training the agent
uv run python train_ppo.py --timesteps 500000 --save-dir models

### With custom parameters
uv run python train_ppo.py --timesteps 1000000 --learning-rate 1e-4 --n-envs 8

### Evaluate existing model
uv run python train_ppo.py --eval-only --model-path models/best_model_20241205_143022.zip

### Run visualization
uv run python visualize_agent.py models/best_model_[timestamp].zip --n-episodes 100

### View tensorboard logs
uv run tensorboard --logdir logs
