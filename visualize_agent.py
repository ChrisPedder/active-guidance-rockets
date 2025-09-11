#!/usr/bin/env python3
"""
Visualization Script for Rocket Control Agent

This script runs 100 episodes with a trained agent across different
environmental conditions and creates comprehensive visualizations
of the agent's performance.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from stable_baselines3 import PPO
import gymnasium as gym

# Import your environment
from rocket_boost_control_env import RocketBoostControlEnv


@dataclass
class EpisodeData:
    """Data structure to store episode results"""
    episode_id: int
    trajectory: np.ndarray  # [time, x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz]
    actions: np.ndarray     # [time, flap_angle]
    rewards: np.ndarray     # [time, reward]
    wind_conditions: np.ndarray  # [time, wind_x, wind_y, wind_z]
    final_altitude: float
    final_horizontal_distance: float
    max_attitude_deviation: float
    total_reward: float
    success: bool
    episode_length: int


class RocketVisualizationRunner:
    """Class to run episodes and collect data for visualization"""

    def __init__(self, model_path: str, env_config: Dict = None):
        """
        Initialize the visualization runner

        Args:
            model_path: Path to trained PPO model
            env_config: Environment configuration override
        """
        self.model = PPO.load(model_path)
        self.env_config = env_config or {}
        print(f"Loaded model from: {model_path}")

    def run_episode(self, episode_id: int, deterministic: bool = True) -> EpisodeData:
        """Run a single episode and collect data"""
        env = RocketBoostControlEnv(self.env_config)

        # Storage for episode data
        trajectory = []
        actions = []
        rewards = []
        wind_conditions = []

        obs, info = env.reset()
        max_attitude_dev = 0.0
        total_reward = 0.0

        for step in range(1000):  # Max episode length
            # Get action from agent
            action, _states = self.model.predict(obs, deterministic=deterministic)
            actions.append(action[0])  # Flap angle

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            total_reward += reward

            # Store trajectory data
            trajectory_point = np.concatenate([
                info['position'],           # x, y, z
                info['velocity'],           # vx, vy, vz
                info['attitude_degrees'],   # roll, pitch, yaw (converted to degrees)
                info['angular_velocity']    # wx, wy, wz
            ])
            trajectory.append(trajectory_point)

            # Store wind data
            wind_conditions.append(info['wind'])

            # Track max attitude deviation
            attitude_dev = np.max(np.abs(info['attitude_degrees']))
            max_attitude_dev = max(max_attitude_dev, attitude_dev)

            if terminated or truncated:
                break

        env.close()

        # Convert to numpy arrays
        trajectory = np.array(trajectory)
        actions = np.array(actions)
        rewards = np.array(rewards)
        wind_conditions = np.array(wind_conditions)

        # Determine success (reached high altitude with good control)
        success = (info['altitude'] > 2500 and max_attitude_dev < 30 and
                  info['horizontal_distance'] < 500)

        return EpisodeData(
            episode_id=episode_id,
            trajectory=trajectory,
            actions=actions,
            rewards=rewards,
            wind_conditions=wind_conditions,
            final_altitude=info['altitude'],
            final_horizontal_distance=info['horizontal_distance'],
            max_attitude_deviation=max_attitude_dev,
            total_reward=total_reward,
            success=success,
            episode_length=len(trajectory)
        )

    def run_multiple_episodes(self, n_episodes: int = 100) -> List[EpisodeData]:
        """Run multiple episodes with varying environmental conditions"""
        print(f"Running {n_episodes} episodes...")

        episodes_data = []
        success_count = 0

        for i in range(n_episodes):
            episode_data = self.run_episode(i)
            episodes_data.append(episode_data)

            if episode_data.success:
                success_count += 1

            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{n_episodes} episodes. "
                      f"Success rate so far: {success_count/(i+1)*100:.1f}%")

        print(f"\nCompleted all {n_episodes} episodes!")
        print(f"Overall success rate: {success_count/n_episodes*100:.1f}%")

        return episodes_data


class RocketVisualizationPlotter:
    """Class to create comprehensive visualizations"""

    def __init__(self, episodes_data: List[EpisodeData]):
        self.episodes_data = episodes_data
        self.n_episodes = len(episodes_data)

        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def plot_performance_overview(self, save_path: str = None):
        """Create overview plot of agent performance"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Rocket Control Agent Performance Overview ({self.n_episodes} Episodes)',
                     fontsize=16, fontweight='bold')

        # Extract metrics
        altitudes = [ep.final_altitude for ep in self.episodes_data]
        horizontal_distances = [ep.final_horizontal_distance for ep in self.episodes_data]
        attitude_deviations = [ep.max_attitude_deviation for ep in self.episodes_data]
        total_rewards = [ep.total_reward for ep in self.episodes_data]
        successes = [ep.success for ep in self.episodes_data]
        episode_lengths = [ep.episode_length for ep in self.episodes_data]

        # 1. Altitude distribution
        axes[0,0].hist(altitudes, bins=30, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(3000, color='red', linestyle='--', label='Target (3000m)')
        axes[0,0].set_xlabel('Final Altitude (m)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Final Altitude Distribution')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Horizontal drift
        axes[0,1].hist(horizontal_distances, bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(500, color='red', linestyle='--', label='Success threshold')
        axes[0,1].set_xlabel('Final Horizontal Distance (m)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Horizontal Drift Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Attitude control
        axes[0,2].hist(attitude_deviations, bins=30, alpha=0.7, edgecolor='black')
        axes[0,2].axvline(30, color='red', linestyle='--', label='Success threshold')
        axes[0,2].set_xlabel('Max Attitude Deviation (°)')
        axes[0,2].set_ylabel('Frequency')
        axes[0,2].set_title('Attitude Control Performance')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        # 4. Rewards
        axes[1,0].hist(total_rewards, bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Total Episode Reward')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Reward Distribution')
        axes[1,0].grid(True, alpha=0.3)

        # 5. Success rate pie chart
        success_counts = [sum(successes), len(successes) - sum(successes)]
        labels = ['Successful', 'Failed']
        colors = ['green', 'red']
        axes[1,1].pie(success_counts, labels=labels, colors=colors, autopct='%1.1f%%',
                     startangle=90)
        axes[1,1].set_title('Mission Success Rate')

        # 6. Episode length vs success
        successful_lengths = [ep.episode_length for ep in self.episodes_data if ep.success]
        failed_lengths = [ep.episode_length for ep in self.episodes_data if not ep.success]

        axes[1,2].hist([successful_lengths, failed_lengths], bins=20, alpha=0.7,
                      label=['Successful', 'Failed'], color=['green', 'red'])
        axes[1,2].set_xlabel('Episode Length (steps)')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Episode Length by Outcome')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_trajectory_analysis(self, n_sample_episodes: int = 10, save_path: str = None):
        """Plot trajectory analysis for sample episodes"""
        # Select sample episodes (mix of successful and failed)
        successful_episodes = [ep for ep in self.episodes_data if ep.success]
        failed_episodes = [ep for ep in self.episodes_data if not ep.success]

        n_successful = min(n_sample_episodes // 2, len(successful_episodes))
        n_failed = min(n_sample_episodes - n_successful, len(failed_episodes))

        sample_episodes = (successful_episodes[:n_successful] +
                          failed_episodes[:n_failed])

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trajectory Analysis (Sample Episodes)', fontsize=16, fontweight='bold')

        # Time vector for plotting
        max_length = max([ep.episode_length for ep in sample_episodes])

        for i, episode in enumerate(sample_episodes):
            time = np.arange(episode.episode_length) * 0.02  # dt = 0.02s
            color = 'green' if episode.success else 'red'
            alpha = 0.6

            # 1. Altitude vs time
            axes[0,0].plot(time, episode.trajectory[:, 2], color=color, alpha=alpha, linewidth=1)

            # 2. Horizontal position (x-y plane)
            axes[0,1].plot(episode.trajectory[:, 0], episode.trajectory[:, 1],
                          color=color, alpha=alpha, linewidth=1)

            # 3. Attitude angles
            axes[0,2].plot(time, episode.trajectory[:, 6], color=color, alpha=alpha,
                          linewidth=1, label='Roll' if i == 0 else "")
            axes[0,2].plot(time, episode.trajectory[:, 7], color=color, alpha=alpha,
                          linewidth=1, linestyle='--', label='Pitch' if i == 0 else "")
            axes[0,2].plot(time, episode.trajectory[:, 8], color=color, alpha=alpha,
                          linewidth=1, linestyle=':', label='Yaw' if i == 0 else "")

            # 4. Control actions
            axes[1,0].plot(time, episode.actions, color=color, alpha=alpha, linewidth=1)

            # 5. Velocities
            axes[1,1].plot(time, episode.trajectory[:, 5], color=color, alpha=alpha,
                          linewidth=1, label='Vertical' if i == 0 else "")
            horizontal_vel = np.sqrt(episode.trajectory[:, 3]**2 + episode.trajectory[:, 4]**2)
            axes[1,1].plot(time, horizontal_vel, color=color, alpha=alpha,
                          linewidth=1, linestyle='--', label='Horizontal' if i == 0 else "")

            # 6. Wind conditions
            wind_magnitude = np.linalg.norm(episode.wind_conditions, axis=1)
            axes[1,2].plot(time, wind_magnitude, color=color, alpha=alpha, linewidth=1)

        # Formatting
        axes[0,0].set_xlabel('Time (s)')
        axes[0,0].set_ylabel('Altitude (m)')
        axes[0,0].set_title('Altitude Trajectories')
        axes[0,0].grid(True, alpha=0.3)

        axes[0,1].set_xlabel('X Position (m)')
        axes[0,1].set_ylabel('Y Position (m)')
        axes[0,1].set_title('Horizontal Drift (Top View)')
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].axis('equal')

        axes[0,2].set_xlabel('Time (s)')
        axes[0,2].set_ylabel('Attitude (°)')
        axes[0,2].set_title('Attitude Angles')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)

        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Flap Angle (°)')
        axes[1,0].set_title('Control Actions')
        axes[1,0].grid(True, alpha=0.3)

        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Velocity (m/s)')
        axes[1,1].set_title('Velocities')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('Wind Speed (m/s)')
        axes[1,2].set_title('Wind Conditions')
        axes[1,2].grid(True, alpha=0.3)

        # Add legend for success/failure
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='green', label='Successful'),
                          Line2D([0], [0], color='red', label='Failed')]
        fig.legend(legend_elements, ['Successful', 'Failed'],
                  loc='upper right', bbox_to_anchor=(0.99, 0.99))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_wind_correlation_analysis(self, save_path: str = None):
        """Analyze how wind conditions affect performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Wind Condition vs Performance Analysis', fontsize=16, fontweight='bold')

        # Extract wind statistics for each episode
        wind_magnitudes = []
        wind_variabilities = []
        altitudes = []
        attitude_deviations = []
        successes = []

        for episode in self.episodes_data:
            wind_mag = np.linalg.norm(episode.wind_conditions, axis=1)
            wind_magnitudes.append(np.mean(wind_mag))
            wind_variabilities.append(np.std(wind_mag))
            altitudes.append(episode.final_altitude)
            attitude_deviations.append(episode.max_attitude_deviation)
            successes.append(episode.success)

        wind_magnitudes = np.array(wind_magnitudes)
        wind_variabilities = np.array(wind_variabilities)
        altitudes = np.array(altitudes)
        attitude_deviations = np.array(attitude_deviations)
        successes = np.array(successes)

        # 1. Wind magnitude vs altitude
        successful_mask = successes == True
        failed_mask = successes == False

        axes[0,0].scatter(wind_magnitudes[successful_mask], altitudes[successful_mask],
                         c='green', alpha=0.6, label='Successful', s=30)
        axes[0,0].scatter(wind_magnitudes[failed_mask], altitudes[failed_mask],
                         c='red', alpha=0.6, label='Failed', s=30)
        axes[0,0].set_xlabel('Mean Wind Magnitude (m/s)')
        axes[0,0].set_ylabel('Final Altitude (m)')
        axes[0,0].set_title('Wind Magnitude vs Final Altitude')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # 2. Wind variability vs attitude control
        axes[0,1].scatter(wind_variabilities[successful_mask], attitude_deviations[successful_mask],
                         c='green', alpha=0.6, label='Successful', s=30)
        axes[0,1].scatter(wind_variabilities[failed_mask], attitude_deviations[failed_mask],
                         c='red', alpha=0.6, label='Failed', s=30)
        axes[0,1].set_xlabel('Wind Variability (std of magnitude)')
        axes[0,1].set_ylabel('Max Attitude Deviation (°)')
        axes[0,1].set_title('Wind Variability vs Attitude Control')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # 3. Success rate by wind magnitude bins
        wind_bins = np.linspace(0, np.max(wind_magnitudes), 10)
        bin_centers = (wind_bins[:-1] + wind_bins[1:]) / 2
        success_rates = []

        for i in range(len(wind_bins)-1):
            mask = (wind_magnitudes >= wind_bins[i]) & (wind_magnitudes < wind_bins[i+1])
            if np.sum(mask) > 0:
                success_rates.append(np.mean(successes[mask]))
            else:
                success_rates.append(0)

        axes[1,0].bar(bin_centers, success_rates, width=np.diff(wind_bins)[0]*0.8,
                     alpha=0.7, edgecolor='black')
        axes[1,0].set_xlabel('Wind Magnitude (m/s)')
        axes[1,0].set_ylabel('Success Rate')
        axes[1,0].set_title('Success Rate vs Wind Conditions')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 1)

        # 4. Control effort vs wind conditions
        control_efforts = []
        for episode in self.episodes_data:
            control_effort = np.mean(np.abs(episode.actions))
            control_efforts.append(control_effort)

        control_efforts = np.array(control_efforts)

        axes[1,1].scatter(wind_magnitudes[successful_mask], control_efforts[successful_mask],
                         c='green', alpha=0.6, label='Successful', s=30)
        axes[1,1].scatter(wind_magnitudes[failed_mask], control_efforts[failed_mask],
                         c='red', alpha=0.6, label='Failed', s=30)
        axes[1,1].set_xlabel('Mean Wind Magnitude (m/s)')
        axes[1,1].set_ylabel('Mean Control Effort (°)')
        axes[1,1].set_title('Control Effort vs Wind Conditions')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_3d_trajectories(self, n_trajectories: int = 20, save_path: str = None):
        """Create 3D visualization of rocket trajectories"""
        from mpl_toolkits.mplot3d import Axes3D

        # Select diverse episodes
        successful_episodes = [ep for ep in self.episodes_data if ep.success]
        failed_episodes = [ep for ep in self.episodes_data if not ep.success]

        n_successful = min(n_trajectories // 2, len(successful_episodes))
        n_failed = min(n_trajectories - n_successful, len(failed_episodes))

        sample_episodes = (successful_episodes[:n_successful] +
                          failed_episodes[:n_failed])

        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        for episode in sample_episodes:
            x = episode.trajectory[:, 0]
            y = episode.trajectory[:, 1]
            z = episode.trajectory[:, 2]

            color = 'green' if episode.success else 'red'
            alpha = 0.6

            ax.plot(x, y, z, color=color, alpha=alpha, linewidth=2)

            # Mark start and end points
            ax.scatter([x[0]], [y[0]], [z[0]], color='blue', s=50, alpha=0.8)
            ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=100,
                      marker='*' if episode.success else 'x')

        # Set labels and title
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Altitude (m)')
        ax.set_title(f'3D Rocket Trajectories ({n_trajectories} episodes)')

        # Add target altitude plane
        max_xy = max(np.max(np.abs([ep.trajectory[:, 0] for ep in sample_episodes])),
                    np.max(np.abs([ep.trajectory[:, 1] for ep in sample_episodes])))
        xx, yy = np.meshgrid(np.linspace(-max_xy, max_xy, 10),
                           np.linspace(-max_xy, max_xy, 10))
        zz = np.ones_like(xx) * 3000
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', marker='o', linestyle='None',
                   markersize=8, label='Launch Point'),
            Line2D([0], [0], color='green', linewidth=2, label='Successful Flight'),
            Line2D([0], [0], color='red', linewidth=2, label='Failed Flight'),
            Line2D([0], [0], color='gray', alpha=0.5, label='Target Altitude (3000m)')
        ]
        ax.legend(handles=legend_elements, loc='upper left')

        # Set equal aspect ratio
        max_range = max_xy
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([0, 4000])

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def create_comprehensive_report(self, save_dir: str = "visualization_results"):
        """Create all visualizations and save to directory"""
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"Creating comprehensive visualization report...")
        print(f"Results will be saved to: {save_dir}")

        # 1. Performance overview
        print("Creating performance overview...")
        self.plot_performance_overview(
            os.path.join(save_dir, f"performance_overview_{timestamp}.png")
        )

        # 2. Trajectory analysis
        print("Creating trajectory analysis...")
        self.plot_trajectory_analysis(
            n_sample_episodes=15,
            save_path=os.path.join(save_dir, f"trajectory_analysis_{timestamp}.png")
        )

        # 3. Wind correlation analysis
        print("Creating wind correlation analysis...")
        self.plot_wind_correlation_analysis(
            save_path=os.path.join(save_dir, f"wind_correlation_{timestamp}.png")
        )

        # 4. 3D trajectories
        print("Creating 3D trajectory visualization...")
        self.plot_3d_trajectories(
            n_trajectories=25,
            save_path=os.path.join(save_dir, f"3d_trajectories_{timestamp}.png")
        )

        # 5. Generate summary statistics
        self._generate_summary_report(save_dir, timestamp)

        print(f"\nVisualization report completed!")
        print(f"All files saved to: {save_dir}")

    def _generate_summary_report(self, save_dir: str, timestamp: str):
        """Generate text summary of results"""
        # Calculate summary statistics
        altitudes = [ep.final_altitude for ep in self.episodes_data]
        horizontal_distances = [ep.final_horizontal_distance for ep in self.episodes_data]
        attitude_deviations = [ep.max_attitude_deviation for ep in self.episodes_data]
        total_rewards = [ep.total_reward for ep in self.episodes_data]
        successes = [ep.success for ep in self.episodes_data]

        success_rate = np.mean(successes) * 100

        report = f"""
ROCKET CONTROL AGENT PERFORMANCE REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Episodes Analyzed: {self.n_episodes}

=== MISSION SUCCESS ===
Success Rate: {success_rate:.1f}%
Successful Missions: {sum(successes)}/{self.n_episodes}

=== ALTITUDE PERFORMANCE ===
Mean Final Altitude: {np.mean(altitudes):.1f} ± {np.std(altitudes):.1f} m
Target Altitude (3000m) Achievement Rate: {sum(alt > 3000 for alt in altitudes)/len(altitudes)*100:.1f}%
Maximum Altitude Reached: {np.max(altitudes):.1f} m
Minimum Altitude Reached: {np.min(altitudes):.1f} m

=== ATTITUDE CONTROL ===
Mean Max Attitude Deviation: {np.mean(attitude_deviations):.1f} ± {np.std(attitude_deviations):.1f}°
Episodes with <30° Max Deviation: {sum(dev < 30 for dev in attitude_deviations)/len(attitude_deviations)*100:.1f}%
Episodes with <10° Max Deviation: {sum(dev < 10 for dev in attitude_deviations)/len(attitude_deviations)*100:.1f}%

=== TRAJECTORY ACCURACY ===
Mean Horizontal Drift: {np.mean(horizontal_distances):.1f} ± {np.std(horizontal_distances):.1f} m
Episodes with <500m Drift: {sum(dist < 500 for dist in horizontal_distances)/len(horizontal_distances)*100:.1f}%
Episodes with <100m Drift: {sum(dist < 100 for dist in horizontal_distances)/len(horizontal_distances)*100:.1f}%

=== REWARD PERFORMANCE ===
Mean Episode Reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}
Best Episode Reward: {np.max(total_rewards):.1f}
Worst Episode Reward: {np.min(total_rewards):.1f}

=== WIND RESILIENCE ===
Episodes tested across varying wind conditions
Wind speeds: 0-20 m/s base wind + 0-10 m/s gusts
Agent demonstrates {'excellent' if success_rate > 80 else 'good' if success_rate > 60 else 'moderate'} performance across wind spectrum

=== RECOMMENDATIONS ===
"""

        if success_rate > 80:
            report += "- Agent shows excellent performance and is ready for deployment\n"
        elif success_rate > 60:
            report += "- Agent shows good performance but may benefit from additional training\n"
        else:
            report += "- Agent needs significant improvement before deployment\n"

        if np.mean(attitude_deviations) > 30:
            report += "- Focus additional training on attitude control\n"

        if np.mean(horizontal_distances) > 500:
            report += "- Improve trajectory tracking performance\n"

        # Save report
        with open(os.path.join(save_dir, f"summary_report_{timestamp}.txt"), 'w') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Visualize rocket control agent performance")
    parser.add_argument("model_path", type=str, help="Path to trained PPO model")
    parser.add_argument("--n-episodes", type=int, default=100,
                       help="Number of episodes to run")
    parser.add_argument("--save-dir", type=str, default="visualization_results",
                       help="Directory to save visualizations")
    parser.add_argument("--env-config", type=str, help="JSON file with environment config")

    args = parser.parse_args()

    # Load environment config if provided
    env_config = None
    if args.env_config:
        import json
        with open(args.env_config, 'r') as f:
            env_config = json.load(f)

    # Create runner and collect data
    runner = RocketVisualizationRunner(args.model_path, env_config)
    episodes_data = runner.run_multiple_episodes(args.n_episodes)

    # Create visualizations
    plotter = RocketVisualizationPlotter(episodes_data)
    plotter.create_comprehensive_report(args.save_dir)


if __name__ == "__main__":
    main()
