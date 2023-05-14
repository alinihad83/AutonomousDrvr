import gymnasium as gym
from matplotlib import pyplot as plt
from .agent import Agent

def run_env(env_config, agent_config):

    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(env_config)
    obs, info = env.reset()
    next_obs = obs
    done = False
    terminated = False
    agent = Agent(**agent_config)

    while not terminated:

        # Predict the next action knowing the current state
        obs = next_obs
        action = agent.get_action(obs, **agent_config)

        # Do the next action and get the next state
        next_obs, reward, done, truncated, info = env.step(action)

        # Update the model based on the next state
        agent.train(obs, next_obs, reward, action)

        # Update number of iterations if car crashed
        if done == True:
            agent.iter += 1
            obs, info = env.reset()

        env.render()

    plt.imshow(env.render())