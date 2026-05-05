import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:
    """Standard transition buffer for DQN.
    Pre-allocates numpy arrays and uses a circular buffer with a position
    pointer — O(1) random access vs the previous deque which was O(n) to
    sample because random.sample had to convert the deque to a list first."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.pos      = 0
        self.full     = False

        # arrays initialised lazily on first push so obs_dim doesn't need
        # to be passed to __init__
        self._states      = None
        self._actions     = None
        self._rewards     = None
        self._next_states = None
        self._dones       = None

    def push(self, state, action, reward, next_state, done):
        state      = np.asarray(state,      dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)

        if self._states is None:
            obs_dim = state.shape[0]
            self._states      = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self._actions     = np.zeros((self.capacity,),         dtype=np.int64)
            self._rewards     = np.zeros((self.capacity,),         dtype=np.float32)
            self._next_states = np.zeros((self.capacity, obs_dim), dtype=np.float32)
            self._dones       = np.zeros((self.capacity,),         dtype=np.float32)

        self._states[self.pos]      = state
        self._actions[self.pos]     = action
        self._rewards[self.pos]     = reward
        self._next_states[self.pos] = next_state
        self._dones[self.pos]       = float(done)

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, batch_size):
        upper   = self.capacity if self.full else self.pos
        indices = np.random.randint(0, upper, size=batch_size)
        return (
            torch.from_numpy(self._states[indices]),
            torch.from_numpy(self._actions[indices]),
            torch.from_numpy(self._rewards[indices]),
            torch.from_numpy(self._next_states[indices]),
            torch.from_numpy(self._dones[indices]),
        )

    def __len__(self):
        return self.capacity if self.full else self.pos

    def is_ready(self, batch_size):
        return len(self) >= batch_size


class SequenceReplayBuffer:
    """Sequence buffer for DRQN.
    Stores full episodes as pre-stacked numpy arrays so sampling is direct
    array slicing — no Python-level tuple unpacking at sample time."""

    def __init__(self, capacity, seq_len):
        self._episodes          = [None] * capacity
        self.capacity           = capacity
        self.seq_len            = seq_len
        self._pos               = 0
        self._n_stored          = 0
        self._total_transitions = 0

        # accumulators for the episode currently being built
        self._ep_states      = []
        self._ep_actions     = []
        self._ep_rewards     = []
        self._ep_next_states = []
        self._ep_dones       = []

    def push(self, state, action, reward, next_state, done):
        self._ep_states.append(np.asarray(state,      dtype=np.float32))
        self._ep_actions.append(action)
        self._ep_rewards.append(float(reward))
        self._ep_next_states.append(np.asarray(next_state, dtype=np.float32))
        self._ep_dones.append(float(done))

        if done:
            if len(self._ep_states) >= self.seq_len:
                # stack into arrays once per episode, not once per sample
                episode = {
                    "states":      np.stack(self._ep_states),       # (T, obs_dim)
                    "actions":     np.array(self._ep_actions,     dtype=np.int64),
                    "rewards":     np.array(self._ep_rewards,     dtype=np.float32),
                    "next_states": np.stack(self._ep_next_states),  # (T, obs_dim)
                    "dones":       np.array(self._ep_dones,       dtype=np.float32),
                }
                if self._episodes[self._pos] is not None:
                    self._total_transitions -= len(self._episodes[self._pos]["actions"])
                self._episodes[self._pos]    = episode
                self._total_transitions     += len(self._ep_actions)
                self._pos                    = (self._pos + 1) % self.capacity
                self._n_stored               = min(self._n_stored + 1, self.capacity)

            self._ep_states      = []
            self._ep_actions     = []
            self._ep_rewards     = []
            self._ep_next_states = []
            self._ep_dones       = []

    def sample(self, batch_size):
        indices = np.random.randint(0, self._n_stored, size=batch_size)
        starts  = [
            random.randint(0, len(self._episodes[i]["actions"]) - self.seq_len)
            for i in indices
        ]

        # direct numpy slicing — no zip, no tuple unpacking
        states      = np.stack([self._episodes[i]["states"]     [s:s+self.seq_len] for i, s in zip(indices, starts)])
        actions     = np.stack([self._episodes[i]["actions"]    [s:s+self.seq_len] for i, s in zip(indices, starts)])
        rewards     = np.stack([self._episodes[i]["rewards"]    [s:s+self.seq_len] for i, s in zip(indices, starts)])
        next_states = np.stack([self._episodes[i]["next_states"][s:s+self.seq_len] for i, s in zip(indices, starts)])
        dones       = np.stack([self._episodes[i]["dones"]      [s:s+self.seq_len] for i, s in zip(indices, starts)])

        return (
            torch.from_numpy(states),               # (batch, seq_len, obs_dim)
            torch.from_numpy(actions),              # (batch, seq_len)
            torch.from_numpy(rewards),              # (batch, seq_len)
            torch.from_numpy(next_states),          # (batch, seq_len, obs_dim)
            torch.from_numpy(dones),                # (batch, seq_len)
        )

    def __len__(self):
        return self._n_stored

    def is_ready(self, batch_size):
        return self._total_transitions >= batch_size



if __name__ == "__main__":
    # test ReplayBuffer
    rb = ReplayBuffer(capacity=1000)
    for _ in range(100):
        rb.push(
            np.zeros(72), 
            random.randint(0, 6),
            0.0, 
            np.zeros(72), 
            False
        )
    states, actions, rewards, next_states, dones = rb.sample(32)
    print("ReplayBuffer sample shapes:")
    print("  states:     ", states.shape)      # (32, 72)
    print("  actions:    ", actions.shape)     # (32,)
    print("  rewards:    ", rewards.shape)     # (32,)
    print("  next_states:", next_states.shape) # (32, 72)
    print("  dones:      ", dones.shape)       # (32,)

    # test SequenceReplayBuffer
    srb = SequenceReplayBuffer(capacity=100, seq_len=8)
    # push enough full episodes to fill buffer
    for ep in range(20):
        for step in range(30):
            done = (step == 29)
            srb.push(np.zeros(72), random.randint(0, 6), 0.0, np.zeros(72), done)
    states, actions, rewards, next_states, dones = srb.sample(32)
    print("\nSequenceReplayBuffer sample shapes:")
    print("  states:     ", states.shape)      # (32, 8, 72)
    print("  actions:    ", actions.shape)     # (32, 8)
    print("  rewards:    ", rewards.shape)     # (32, 8)
    print("  next_states:", next_states.shape) # (32, 8, 72)
    print("  dones:      ", dones.shape)       # (32, 8)