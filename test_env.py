# test_env.py
from rocket_boost_control_env import RocketBoostControlEnv
import numpy as np

env = RocketBoostControlEnv()
obs, info = env.reset()
print(f"Initial observation: {obs}")
print(f"Action space: {env.action_space}")
print(f"Action space bounds: low={env.action_space.low}, high={env.action_space.high}")

total_reward = 0
for i in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if i % 20 == 0:
        print(f"Step {i}: reward={reward:.2f}, altitude={info['altitude']:.1f}m")

    if terminated or truncated:
        print(f"Episode ended at step {i}")
        break

print(f"Total reward: {total_reward:.2f}")
