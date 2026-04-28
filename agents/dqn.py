# agentss/dqn.py
import torch
import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

def forward(self, x):
    return self.net(x)