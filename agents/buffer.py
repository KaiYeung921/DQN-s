import random
from collections import deque
import torch
import numpy as np


class ReplayBuffer:
    """Standard transition buffer for DQN.
    Stores individual (s, a, r, s', done) transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones))
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self) >= batch_size


class SequenceReplayBuffer:
    """Sequence buffer for DRQN.
    Stores full episodes, samples fixed-length sequences from them."""

    def __init__(self, capacity, seq_len):
        self.buffer = deque(maxlen=capacity)
        self.seq_len = seq_len
        self.current_episode = []

    def push(self, state, action, reward, next_state, done):
        self.current_episode.append((
            state,
            action,
            reward,
            next_state,
            float(done)
        ))
        if done:
            if len(self.current_episode) >= self.seq_len:
                self.buffer.append(self.current_episode)
            self.current_episode = []   # reset for next episode

    def sample(self, batch_size):
        sequences = []
        while len(sequences) < batch_size:
            episode = random.choice(self.buffer)
            # pick a random starting point that leaves room for seq_len steps
            start = random.randint(0, len(episode) - self.seq_len)
            sequences.append(episode[start : start + self.seq_len])

        # sequences is (batch_size, seq_len, 5)
        # we need to transpose to (5, batch_size, seq_len) then tensorize
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for seq in sequences:
            s, a, r, ns, d = zip(*seq)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            torch.FloatTensor(np.array(states)),       # (batch, seq_len, 72)
            torch.LongTensor(np.array(actions)),        # (batch, seq_len)
            torch.FloatTensor(np.array(rewards)),       # (batch, seq_len)
            torch.FloatTensor(np.array(next_states)),   # (batch, seq_len, 72)
            torch.FloatTensor(np.array(dones))          # (batch, seq_len)
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        return len(self) >= batch_size
    


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