import gymnasium as gym
from matplotlib import pyplot as plt
from .agent import Agent

def print_info(next_obs, reward, done, truncated, info, action):
    print(f"================================================")
    print(f"next_obs: {next_obs}\nreward: {reward}\ndone: {done}\ntruncated: {truncated}\ninfo: {info}")

def print_iter(iter):
    print(f"ITERATION NUMBER: {iter}")

def run_env(env_config, agent_config):

    env = gym.make('highway-v0', render_mode='rgb_array')
    env.configure(env_config)
    obs, info = env.reset()
    done = False
    truncated = False
    terminated = False
    agent = Agent(**agent_config)

    while not terminated:

        # Predict the next action knowing the current state
        action = agent.get_action(obs)

        # Do the next action and get the next state
        next_obs, reward, done, truncated, info = env.step(action)
        print_info(next_obs, reward, done, truncated, info, action)

        # Update the model based on the next state
        agent.train(obs, next_obs, reward, action)

        # Update number of iterations if car crashed
        if done == True:
            agent.iter += 1
            obs, info = env.reset()
            print_iter(agent.iter)

        env.render()

    plt.imshow(env.render())