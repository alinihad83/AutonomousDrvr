import gymnasium as gym
import random
import torch
from .model import MLP, Trainer

class Agent:
    def __init__(self, **kwargs):
        self.iter = 0
        self.epsilon = kwargs['epsilon']
        self.gamma = kwargs['gamma']
        self.model = MLP(kwargs['input_size'], kwargs['layer_1_size'], kwargs['layer_2_size'], kwargs['output_size'])
        self.trainer = Trainer(self.model, lr=kwargs['lr'], gamma=self.gamma)

    def train(self, obs, next_obs, reward, action):
        self.trainer.train_step(obs, next_obs, reward, action)

    def get_action(self, obs, **kwargs):
        obs = obs.flatten()
        obs = torch.from_numpy(obs)
        obs = torch.round(obs, decimals=1)

        self.epsilon = 100 - self.iter

        # exploration/exploitation trade-off
        if random.randint(0, kwargs['rand_range']) < self.epsilon:
            move = random.randint(0, 4)
        else:
            prediction = self.model(obs)
            move = torch.argmax(prediction).item()

        return move