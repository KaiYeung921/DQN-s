import torch
import torch.nn as nn

class DRQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(72, 128),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.decoder = nn.Linear(128, 7)

    def forward(self, x, hidden=None):
        encoded = self.encoder(x)
        lstm_out, hidden = self.lstm(encoded, hidden)
        q_values = self.decoder(lstm_out)
        return q_values, hidden
    
    def init_hidden(self, batch_size=1, device="cpu"):
        # h_0 and c_0 both start as zeros
        return (
            torch.zeros(1, batch_size, 128).to(device),
            torch.zeros(1, batch_size, 128).to(device)
        )
