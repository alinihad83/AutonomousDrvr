import gymnasium as gym
from matplotlib import pyplot as plt
import json

def run_env(env_config):
    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(env_config)
    env.reset()
    for _ in range(10):
        action = env.action_type.actions_indexes["IDLE"]
        obs, reward, done, truncated, info = env.step(action)
        env.render()

    plt.imshow(env.render())