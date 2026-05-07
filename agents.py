"""
agents.py — Neural networks and replay buffers for DQN and DRQN.
"""

import random
import numpy as np
import torch
import torch.nn as nn


# ── Networks ──────────────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """Feedforward Q-network: state → Q-values for each action."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DRQNNetwork(nn.Module):
    """Recurrent Q-network: encodes state, passes through LSTM, decodes Q-values.
    The LSTM hidden state carries temporal context across steps within an episode."""

    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden_dim), nn.ReLU())
        self.lstm    = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden=None):
        encoded       = self.encoder(x)
        lstm_out, hidden = self.lstm(encoded, hidden)
        return self.decoder(lstm_out), hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        z = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return (z, z)


# ── Replay Buffers ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Circular buffer storing individual (s, a, r, s', done) transitions for DQN."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.pos      = 0
        self.full     = False
        self._s = self._a = self._r = self._ns = self._d = None  # lazy init

    def push(self, state, action, reward, next_state, done):
        s  = np.asarray(state,      dtype=np.float32)
        ns = np.asarray(next_state, dtype=np.float32)
        if self._s is None:
            d = s.shape[0]
            self._s  = np.zeros((self.capacity, d), dtype=np.float32)
            self._a  = np.zeros((self.capacity,),   dtype=np.int64)
            self._r  = np.zeros((self.capacity,),   dtype=np.float32)
            self._ns = np.zeros((self.capacity, d), dtype=np.float32)
            self._d  = np.zeros((self.capacity,),   dtype=np.float32)
        self._s[self.pos]  = s;  self._a[self.pos]  = action
        self._r[self.pos]  = reward
        self._ns[self.pos] = ns; self._d[self.pos]  = float(done)
        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        upper = self.capacity if self.full else self.pos
        idx   = np.random.randint(0, upper, size=batch_size)
        return (
            torch.from_numpy(self._s[idx]),
            torch.from_numpy(self._a[idx]),
            torch.from_numpy(self._r[idx]),
            torch.from_numpy(self._ns[idx]),
            torch.from_numpy(self._d[idx]),
        )

    def __len__(self):
        return self.capacity if self.full else self.pos

    def is_ready(self, batch_size):
        return len(self) >= batch_size


class SequenceReplayBuffer:
    """Episode buffer for DRQN. Stores full episodes; samples fixed-length windows
    so the LSTM can warm up its hidden state before the target timestep."""

    def __init__(self, capacity, seq_len):
        self._episodes  = [None] * capacity
        self.capacity   = capacity
        self.seq_len    = seq_len
        self._pos       = 0
        self._n_stored  = 0
        self._n_trans   = 0   # total transitions (used for is_ready check)
        self._ep = {"s": [], "a": [], "r": [], "ns": [], "d": []}

    def push(self, state, action, reward, next_state, done):
        self._ep["s"].append(np.asarray(state,      dtype=np.float32))
        self._ep["a"].append(action)
        self._ep["r"].append(float(reward))
        self._ep["ns"].append(np.asarray(next_state, dtype=np.float32))
        self._ep["d"].append(float(done))
        if done:
            if len(self._ep["s"]) >= self.seq_len:
                ep = {k: np.stack(v) if k in ("s", "ns") else np.array(v)
                      for k, v in self._ep.items()}
                if self._episodes[self._pos] is not None:
                    self._n_trans -= len(self._episodes[self._pos]["a"])
                self._episodes[self._pos] = ep
                self._n_trans  += len(self._ep["a"])
                self._pos       = (self._pos + 1) % self.capacity
                self._n_stored  = min(self._n_stored + 1, self.capacity)
            self._ep = {"s": [], "a": [], "r": [], "ns": [], "d": []}

    def sample(self, batch_size):
        idxs   = np.random.randint(0, self._n_stored, size=batch_size)
        starts = [random.randint(0, len(self._episodes[i]["a"]) - self.seq_len)
                  for i in idxs]
        sl = self.seq_len
        s  = np.stack([self._episodes[i]["s"] [t:t+sl] for i, t in zip(idxs, starts)])
        a  = np.stack([self._episodes[i]["a"] [t:t+sl] for i, t in zip(idxs, starts)])
        r  = np.stack([self._episodes[i]["r"] [t:t+sl] for i, t in zip(idxs, starts)])
        ns = np.stack([self._episodes[i]["ns"][t:t+sl] for i, t in zip(idxs, starts)])
        d  = np.stack([self._episodes[i]["d"] [t:t+sl] for i, t in zip(idxs, starts)])
        return (torch.from_numpy(s), torch.from_numpy(a),
                torch.from_numpy(r), torch.from_numpy(ns), torch.from_numpy(d))

    def __len__(self):
        return self._n_stored

    def is_ready(self, batch_size):
        return self._n_trans >= batch_size
