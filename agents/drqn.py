import torch
import torch.nn as nn

class DRQNNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        encoded = self.encoder(x)
        lstm_out, hidden = self.lstm(encoded, hidden)
        q_values = self.decoder(lstm_out)
        return q_values, hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        return (
            torch.zeros(1, batch_size, self.hidden_dim).to(device),
            torch.zeros(1, batch_size, self.hidden_dim).to(device)
        )
