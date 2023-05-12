import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class MLP(nn.Module):
    def __init__(self, input_size, layer_1_size, layer_2_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, layer_1_size)
        self.linear2 = nn.Linear(layer_1_size, layer_2_size)
        self.linear3 = nn.Linear(layer_2_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x
    
    def save(self, file_name='model.pth'):
        model_folder_path = '../cache'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class Trainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, obs, next_obs, reward, action):
        obs = obs.flatten()
        obs = torch.from_numpy(obs)
        next_obs = next_obs.flatten()
        next_obs = torch.from_numpy(next_obs)

        pred = self.model(obs)
        target = pred.clone()

        q_new = reward + self.gamma * torch.max(self.model(next_obs))
        target[action] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)

        loss.backward()
        self.optimizer.step()

