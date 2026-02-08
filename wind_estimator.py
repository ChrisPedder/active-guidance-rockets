#!/usr/bin/env python3
"""
Neural Network Wind Estimator for Rocket Roll Stabilization

Trains a small GRU network to estimate wind speed and direction from
the observation history. The estimator is used as a drop-in replacement
for the analytical sinusoidal estimator in wind_feedforward.py.

Architecture:
    - Input: sliding window of recent observations (10 features x W steps)
    - GRU: small recurrent network (1 layer, 32 hidden units)
    - Output: [wind_speed, wind_dir_cos, wind_dir_sin]
      (direction encoded as sin/cos to avoid angle wrapping issues)

Training:
    Supervised learning using ground-truth wind data from the environment's
    info dict (wind_speed_ms, wind_direction_rad). Data is collected by
    running episodes with a base controller (GS-PID) under varying wind.

Usage:
    # Train a new estimator
    uv run python wind_estimator.py --train --episodes 500 --wind-levels 0 1 2 3 5

    # Evaluate a trained estimator
    uv run python wind_estimator.py --evaluate --model models/wind_estimator.pt
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config
from wind_feedforward import WindFeedforwardConfig


@dataclass
class WindEstimatorConfig:
    """Configuration for the GRU wind estimator.

    Attributes:
        window_size: Number of past observations to feed to the GRU.
        hidden_size: GRU hidden state dimension.
        num_layers: Number of GRU layers.
        obs_features: Number of features per observation step.
            Uses indices [2,3,4,5,8] = [roll_angle, roll_rate,
            roll_accel, dynamic_pressure, last_action].
        K_ff: Feedforward gain (same role as in WindFeedforwardConfig).
        warmup_steps: Steps before feedforward activates.
    """

    window_size: int = 20
    hidden_size: int = 32
    num_layers: int = 1
    obs_features: int = 5  # [roll_angle, roll_rate, roll_accel, q, last_action]
    K_ff: float = 0.5
    warmup_steps: int = 50


# Observation indices used by the estimator
OBS_INDICES = [2, 3, 4, 5, 8]  # roll_angle, roll_rate, roll_accel, q, last_action


class WindEstimatorNetwork(nn.Module):
    """GRU-based wind estimator network.

    Takes a sequence of observation features and predicts wind speed
    and direction (as sin/cos components).
    """

    def __init__(self, config: WindEstimatorConfig = None):
        super().__init__()
        cfg = config or WindEstimatorConfig()
        self.config = cfg

        self.gru = nn.GRU(
            input_size=cfg.obs_features,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            batch_first=True,
        )
        # Output: [wind_speed, wind_dir_cos, wind_dir_sin]
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, window_size, obs_features) tensor

        Returns:
            (batch, 3) tensor: [wind_speed, cos(wind_dir), sin(wind_dir)]
        """
        # GRU output: (batch, window_size, hidden_size)
        gru_out, _ = self.gru(x)
        # Use last timestep's hidden state
        last_hidden = gru_out[:, -1, :]
        return self.head(last_hidden)


class NNWindEstimator:
    """Runtime wind estimator that maintains an observation buffer
    and runs the GRU to produce wind estimates.

    This is the inference-time wrapper around WindEstimatorNetwork.
    """

    def __init__(
        self,
        model: WindEstimatorNetwork,
        config: WindEstimatorConfig = None,
    ):
        self.model = model
        self.model.eval()
        self.config = config or model.config
        self._obs_buffer: List[np.ndarray] = []
        # Normalization stats (set during training)
        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std: Optional[np.ndarray] = None
        self.target_mean: Optional[np.ndarray] = None
        self.target_std: Optional[np.ndarray] = None

    def reset(self):
        """Clear observation buffer for new episode."""
        self._obs_buffer = []

    def update(self, obs: np.ndarray):
        """Add a new observation to the buffer.

        Args:
            obs: Full 10-element observation vector from environment.
        """
        features = obs[OBS_INDICES].astype(np.float32)
        self._obs_buffer.append(features)
        # Keep only the most recent window_size observations
        if len(self._obs_buffer) > self.config.window_size:
            self._obs_buffer = self._obs_buffer[-self.config.window_size :]

    def estimate(self) -> Tuple[float, float]:
        """Estimate wind speed and direction from buffered observations.

        Returns:
            (wind_speed, wind_direction) in (m/s, radians)
        """
        if len(self._obs_buffer) < 2:
            return 0.0, 0.0

        # Pad if we don't have a full window yet
        window = self.config.window_size
        if len(self._obs_buffer) < window:
            pad_count = window - len(self._obs_buffer)
            padded = [self._obs_buffer[0]] * pad_count + self._obs_buffer
        else:
            padded = self._obs_buffer[-window:]

        # Stack into (window, features) array
        x = np.stack(padded, axis=0).astype(np.float32)

        # Normalize
        if self.obs_mean is not None and self.obs_std is not None:
            x = (x - self.obs_mean) / (self.obs_std + 1e-8)

        # Run inference
        x_tensor = torch.from_numpy(x).unsqueeze(0)  # (1, window, features)
        with torch.no_grad():
            raw_output = self.model(x_tensor).squeeze(0).numpy()  # (3,)

        # Denormalize output
        if self.target_mean is not None and self.target_std is not None:
            raw_output = raw_output * (self.target_std + 1e-8) + self.target_mean

        wind_speed = max(0.0, float(raw_output[0]))
        wind_dir = float(np.arctan2(raw_output[2], raw_output[1]))

        return wind_speed, wind_dir


class NNFeedforwardADRC:
    """ADRC controller with neural network wind feedforward.

    Replaces the analytical sinusoidal estimator from WindFeedforwardADRC
    with a learned GRU-based wind estimator. The feedforward structure
    is the same: predict the roll-angle-dependent wind torque and
    cancel it via feedforward action.

    The key difference is that the GRU can learn complex wind dynamics
    (multi-frequency gusts, direction drift, altitude-dependent speed)
    that the simple sin/cos decomposition cannot capture.
    """

    def __init__(
        self,
        adrc_config: ADRCConfig = None,
        estimator: NNWindEstimator = None,
        K_ff: float = 0.5,
        warmup_steps: int = 50,
    ):
        self.adrc = ADRCController(adrc_config)
        self.config = adrc_config or ADRCConfig()
        self.estimator = estimator
        self.K_ff = K_ff
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """Reset controller, observer, and wind estimator."""
        self.adrc.reset()
        if self.estimator is not None:
            self.estimator.reset()
        self._step_count = 0
        self._ff_action = 0.0
        self._last_wind_speed = 0.0
        self._last_wind_dir = 0.0

    @property
    def launch_detected(self):
        return self.adrc.launch_detected

    @launch_detected.setter
    def launch_detected(self, value):
        self.adrc.launch_detected = value

    @property
    def target_angle(self):
        return self.adrc.target_angle

    @target_angle.setter
    def target_angle(self, value):
        self.adrc.target_angle = value

    @property
    def z1(self):
        return self.adrc.z1

    @property
    def z2(self):
        return self.adrc.z2

    @property
    def z3(self):
        return self.adrc.z3

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: ADRC base + NN wind feedforward.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        cfg = self.config

        # --- Run base ADRC controller ---
        base_action = self.adrc.step(obs, info, dt)

        if not self.adrc.launch_detected:
            return base_action

        self._step_count += 1

        # --- Update wind estimator with current observation ---
        if self.estimator is not None:
            self.estimator.update(obs)

        # --- Skip feedforward during warmup ---
        if self._step_count < self.warmup_steps or self.estimator is None:
            self._ff_action = 0.0
            return base_action

        # --- Get wind estimate from GRU ---
        wind_speed, wind_dir = self.estimator.estimate()
        self._last_wind_speed = wind_speed
        self._last_wind_dir = wind_dir

        # --- Read roll angle ---
        if cfg.use_observations:
            roll_angle = obs[2] if len(obs) > 2 else 0.0
        else:
            roll_angle = info.get("roll_angle_rad", 0.0)

        # --- Get current b0 for scaling feedforward action ---
        if cfg.b0_per_pa is not None:
            if cfg.use_observations:
                q = float(obs[5]) if len(obs) > 5 else 0.0
            else:
                q = info.get("dynamic_pressure_Pa", 0.0)
            q_effectiveness = q * np.tanh(q / 200.0)
            b0_now = cfg.b0_per_pa * q_effectiveness
            b0_min = cfg.b0 * 0.01
            if b0_now < b0_min:
                b0_now = cfg.b0
        else:
            b0_now = cfg.b0

        # --- Compute feedforward from wind estimate ---
        # Wind torque ~ wind_speed * sin(wind_dir - roll_angle)
        # We predict and cancel this periodic component
        if wind_speed < 0.01:
            self._ff_action = 0.0
            return base_action

        # Predicted wind torque direction (relative to current roll angle)
        relative_angle = wind_dir - roll_angle
        # Scale: use wind_speed as proxy for disturbance amplitude
        # The actual scaling depends on velocity, fin geometry, etc.
        # We let the GRU learn the effective amplitude and use K_ff to tune
        ff_disturbance = wind_speed * np.sin(relative_angle)

        # Scale to action units
        ff_action = -self.K_ff * ff_disturbance / b0_now

        self._ff_action = float(ff_action)

        # Combine base ADRC action with feedforward
        total_action = float(base_action[0]) + ff_action
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)

    @property
    def wind_speed_estimate(self) -> float:
        """Last estimated wind speed in m/s."""
        return self._last_wind_speed

    @property
    def wind_direction_estimate(self) -> float:
        """Last estimated wind direction in radians."""
        return self._last_wind_dir


def collect_training_data(
    config,
    wind_levels: List[float],
    episodes_per_level: int = 100,
    controller=None,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect observation sequences and ground-truth wind labels.

    Runs episodes with a base controller (default: GS-PID) under varying
    wind conditions and records the observation history alongside the
    ground-truth wind speed and direction from the info dict.

    Args:
        config: RocketTrainingConfig
        wind_levels: List of wind speeds to sample
        episodes_per_level: Number of episodes per wind level
        controller: Base controller to use (default: GS-PID)

    Returns:
        (obs_sequences, wind_labels):
            obs_sequences: List of (T, obs_features) arrays
            wind_labels: List of (T, 3) arrays [speed, cos(dir), sin(dir)]
    """
    from pid_controller import PIDController, GainScheduledPIDController, PIDConfig

    obs_sequences = []
    wind_labels = []

    for wind_speed in wind_levels:
        for ep in range(episodes_per_level):
            # Create environment with this wind level
            env = _create_env_for_collection(config, wind_speed)
            obs, info = env.reset()

            if controller is not None:
                ctrl = controller
            else:
                ctrl = GainScheduledPIDController(use_observations=True)
            ctrl.reset()

            ep_obs = []
            ep_labels = []

            while True:
                # Record observation features
                features = obs[OBS_INDICES].astype(np.float32)
                ep_obs.append(features)

                # Record ground-truth wind
                ws = info.get("wind_speed_ms", 0.0)
                wd = info.get("wind_direction_rad", 0.0)
                ep_labels.append([ws, np.cos(wd), np.sin(wd)])

                # Step
                action = ctrl.step(obs, info, 0.01)
                obs, _, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            env.close()

            if len(ep_obs) > 5:  # Skip very short episodes
                obs_sequences.append(np.array(ep_obs, dtype=np.float32))
                wind_labels.append(np.array(ep_labels, dtype=np.float32))

    return obs_sequences, wind_labels


def _create_env_for_collection(config, wind_speed: float):
    """Create environment for data collection with IMU observations."""
    from compare_controllers import create_env_with_imu

    return create_env_with_imu(config, wind_speed)


def prepare_training_windows(
    obs_sequences: List[np.ndarray],
    wind_labels: List[np.ndarray],
    window_size: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert variable-length sequences into fixed-size training windows.

    Args:
        obs_sequences: List of (T, features) arrays
        wind_labels: List of (T, 3) arrays
        window_size: Window length

    Returns:
        (X, Y): X is (N, window_size, features), Y is (N, 3)
    """
    X_windows = []
    Y_windows = []

    for obs_seq, label_seq in zip(obs_sequences, wind_labels):
        T = len(obs_seq)
        if T < window_size:
            continue
        for t in range(window_size, T):
            X_windows.append(obs_seq[t - window_size : t])
            Y_windows.append(label_seq[t])

    if not X_windows:
        return np.empty(
            (0, window_size, obs_sequences[0].shape[-1] if obs_sequences else 5)
        ), np.empty((0, 3))

    X = np.stack(X_windows, axis=0)
    Y = np.stack(Y_windows, axis=0)

    return X, Y


def train_wind_estimator(
    X: np.ndarray,
    Y: np.ndarray,
    config: WindEstimatorConfig = None,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    validation_fraction: float = 0.1,
    verbose: bool = True,
) -> Tuple[WindEstimatorNetwork, dict]:
    """Train the GRU wind estimator.

    Args:
        X: (N, window_size, features) training inputs
        Y: (N, 3) training targets [speed, cos(dir), sin(dir)]
        config: Network configuration
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
        validation_fraction: Fraction of data for validation
        verbose: Print training progress

    Returns:
        (model, stats): Trained model and training statistics
    """
    cfg = config or WindEstimatorConfig()

    # Normalize inputs and targets
    obs_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    obs_std = X.reshape(-1, X.shape[-1]).std(axis=0)
    target_mean = Y.mean(axis=0)
    target_std = Y.std(axis=0)

    X_norm = (X - obs_mean) / (obs_std + 1e-8)
    Y_norm = (Y - target_mean) / (target_std + 1e-8)

    # Train/val split
    n = len(X_norm)
    n_val = max(1, int(n * validation_fraction))
    indices = np.random.permutation(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train = torch.from_numpy(X_norm[train_idx]).float()
    Y_train = torch.from_numpy(Y_norm[train_idx]).float()
    X_val = torch.from_numpy(X_norm[val_idx]).float()
    Y_val = torch.from_numpy(Y_norm[val_idx]).float()

    # Create model
    model = WindEstimatorNetwork(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    stats = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle training data
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        Y_train = Y_train[perm]

        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            Y_batch = Y_train[i : i + batch_size]

            pred = model(X_batch)
            loss = loss_fn(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        stats["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()
        stats["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(
                f"  Epoch {epoch:3d}/{epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}"
            )

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Store normalization stats
    stats["obs_mean"] = obs_mean
    stats["obs_std"] = obs_std
    stats["target_mean"] = target_mean
    stats["target_std"] = target_std

    return model, stats


def save_estimator(
    model: WindEstimatorNetwork,
    stats: dict,
    config: WindEstimatorConfig,
    path: str,
):
    """Save trained estimator to disk.

    Args:
        model: Trained GRU network
        stats: Training statistics (includes normalization params)
        config: Estimator configuration
        path: File path for the saved model
    """
    save_dict = {
        "model_state": model.state_dict(),
        "config": {
            "window_size": config.window_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "obs_features": config.obs_features,
            "K_ff": config.K_ff,
            "warmup_steps": config.warmup_steps,
        },
        # Convert numpy arrays to tensors for weights_only=True compatibility
        "obs_mean": torch.from_numpy(np.asarray(stats["obs_mean"], dtype=np.float32)),
        "obs_std": torch.from_numpy(np.asarray(stats["obs_std"], dtype=np.float32)),
        "target_mean": torch.from_numpy(
            np.asarray(stats["target_mean"], dtype=np.float32)
        ),
        "target_std": torch.from_numpy(
            np.asarray(stats["target_std"], dtype=np.float32)
        ),
    }
    torch.save(save_dict, path)


def load_estimator(path: str) -> Tuple[NNWindEstimator, WindEstimatorConfig]:
    """Load a trained estimator from disk.

    Args:
        path: Path to saved model file

    Returns:
        (estimator, config): Ready-to-use NNWindEstimator and its config
    """
    save_dict = torch.load(path, map_location="cpu", weights_only=True)

    cfg_dict = save_dict["config"]
    config = WindEstimatorConfig(**cfg_dict)

    model = WindEstimatorNetwork(config)
    model.load_state_dict(save_dict["model_state"])
    model.eval()

    estimator = NNWindEstimator(model, config)
    # Convert tensors back to numpy arrays
    estimator.obs_mean = save_dict["obs_mean"].numpy()
    estimator.obs_std = save_dict["obs_std"].numpy()
    estimator.target_mean = save_dict["target_mean"].numpy()
    estimator.target_std = save_dict["target_std"].numpy()

    return estimator, config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/evaluate wind estimator")
    parser.add_argument("--train", action="store_true", help="Train a new estimator")
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate trained estimator"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/wind_estimator.pt",
        help="Model save/load path",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/estes_c6_sac_wind.yaml",
        help="Environment config YAML",
    )
    parser.add_argument(
        "--wind-levels",
        type=float,
        nargs="+",
        default=[0, 1, 2, 3, 5],
        help="Wind speeds for data collection",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per wind level for data collection",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument(
        "--window-size", type=int, default=20, help="Observation window size"
    )
    args = parser.parse_args()

    from rocket_config import load_config

    if args.train:
        print("=" * 60)
        print("Wind Estimator Training")
        print("=" * 60)

        config = load_config(args.config)
        est_config = WindEstimatorConfig(window_size=args.window_size)

        print(
            f"\nCollecting data: {len(args.wind_levels)} wind levels x "
            f"{args.episodes} episodes each..."
        )
        obs_seqs, wind_labels = collect_training_data(
            config,
            args.wind_levels,
            args.episodes,
        )
        print(
            f"Collected {len(obs_seqs)} episodes, "
            f"{sum(len(s) for s in obs_seqs)} total steps"
        )

        print(f"\nPreparing training windows (size={args.window_size})...")
        X, Y = prepare_training_windows(obs_seqs, wind_labels, args.window_size)
        print(f"Training samples: {len(X)}")

        print(f"\nTraining GRU estimator ({args.epochs} epochs)...")
        model, stats = train_wind_estimator(
            X,
            Y,
            est_config,
            epochs=args.epochs,
        )

        # Save model
        save_path = Path(args.model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_estimator(model, stats, est_config, str(save_path))
        print(f"\nModel saved to {save_path}")
        print(f"Final val loss: {stats['val_loss'][-1]:.4f}")

    if args.evaluate:
        print("=" * 60)
        print("Wind Estimator Evaluation")
        print("=" * 60)

        estimator, est_config = load_estimator(args.model)
        print(f"Loaded model from {args.model}")
        print(
            f"Config: window={est_config.window_size}, "
            f"hidden={est_config.hidden_size}"
        )

        config = load_config(args.config)

        # Quick evaluation: run a few episodes and check estimation accuracy
        from pid_controller import GainScheduledPIDController

        for wind_speed in [0, 1, 2, 3]:
            env = _create_env_for_collection(config, wind_speed)
            ctrl = GainScheduledPIDController(use_observations=True)
            estimator.reset()
            ctrl.reset()
            obs, info = env.reset()

            speed_errors = []
            dir_errors = []

            for step in range(500):
                estimator.update(obs)
                if step > est_config.warmup_steps:
                    est_speed, est_dir = estimator.estimate()
                    true_speed = info.get("wind_speed_ms", 0.0)
                    true_dir = info.get("wind_direction_rad", 0.0)

                    speed_errors.append(abs(est_speed - true_speed))
                    # Angular error
                    dir_err = abs(
                        np.arctan2(
                            np.sin(est_dir - true_dir),
                            np.cos(est_dir - true_dir),
                        )
                    )
                    dir_errors.append(dir_err)

                action = ctrl.step(obs, info, 0.01)
                obs, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

            env.close()

            if speed_errors:
                print(
                    f"  Wind={wind_speed} m/s: "
                    f"speed_MAE={np.mean(speed_errors):.3f} m/s, "
                    f"dir_MAE={np.degrees(np.mean(dir_errors)):.1f} deg"
                )
