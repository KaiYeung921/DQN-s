# training/train.py
import torch
import torch.nn.functional as F
import numpy as np
import random

from env import hexxed
from agents.dqn import DQNNetwork
from agents.drqn import DRQNNetwork
from agents.buffer import ReplayBuffer, SequenceReplayBuffer
from config import ENV_CONFIG, TRAIN_CONFIG

def train_dqn(lr, gamma, batch_size, buffer_size, target_update, hidden_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup

    env = hexxed()
    env.ready(**ENV_CONFIG)

    policy_net = DQNNetwork(obs_dim=72, action_dim=7, hidden_dim=hidden_dim).to(device)
    target_net = DQNNetwork(obs_dim=72, action_dim=7, hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer    = ReplayBuffer(capacity=buffer_size)

    epsilon      = TRAIN_CONFIG["epsilon_start"]
    epsilon_end  = TRAIN_CONFIG["epsilon_end"]
    epsilon_decay= TRAIN_CONFIG["epsilon_decay"]
    total_steps  = TRAIN_CONFIG["total_timesteps"]

    step        = 0
    episode     = 0
    reward_log  = []   # tracks per-episode reward for Optuna objective

    #traing loop

    while step < total_steps:
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = policy_net(state_t)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1

            # train only when buffer has enough transitions
            if buffer.is_ready(batch_size):
                loss = _dqn_train_step(
                    policy_net, target_net, optimizer,
                    buffer, batch_size, gamma, device
                )

            # sync target network
            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        reward_log.append(episode_reward)
        episode += 1

def _dqn_train_step(policy_net, target_net, optimizer,
                    buffer, batch_size, gamma, device):
    """Single gradient update for DQN."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states      = states.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)
    next_states = next_states.to(device)
    dones       = dones.to(device)

    # Q(s, a) for actions actually taken
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + γ * max_a Q_target(s', a)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(dim=1)[0]
        target_q   = rewards + gamma * max_next_q * (1 - dones)

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def _drqn_train_step(policy_net, target_net, optimizer,
                     buffer, batch_size, gamma, device):
    """Single gradient update for DRQN."""
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    # all shapes: states/next_states (batch, seq_len, 72)
    #             actions/rewards/dones (batch, seq_len)
    states      = states.to(device)
    actions     = actions.to(device)
    rewards     = rewards.to(device)
    next_states = next_states.to(device)
    dones       = dones.to(device)

    # hidden starts at zeros for every training sequence
    # this is the zero-start approach — simple and standard
    hidden      = policy_net.init_hidden(batch_size=batch_size, device=device)
    target_hidden = target_net.init_hidden(batch_size=batch_size, device=device)

    # forward pass through full sequence
    q_all, _        = policy_net(states, hidden)           # (batch, seq_len, 7)
    next_q_all, _   = target_net(next_states, target_hidden) # (batch, seq_len, 7)

    # only train on the LAST step of each sequence
    # earlier steps were burn-in to build up hidden state
    q_last      = q_all[:, -1, :]        # (batch, 7)
    next_q_last = next_q_all[:, -1, :]   # (batch, 7)
    actions_last = actions[:, -1]         # (batch,)
    rewards_last = rewards[:, -1]         # (batch,)
    dones_last   = dones[:, -1]           # (batch,)

    # Q(s, a) for actions actually taken at the last step
    current_q = q_last.gather(1, actions_last.unsqueeze(1)).squeeze(1)

    # Bellman target using last step
    with torch.no_grad():
        max_next_q = next_q_last.max(dim=1)[0]
        target_q   = rewards_last + gamma * max_next_q * (1 - dones_last)

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()

    # gradient clipping — important for LSTMs to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)

    optimizer.step()
    return loss.item()
