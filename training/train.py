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

class ProgressTracker:
    """
    Tracks learning speed metrics during a training run.
    Records step-level data so we can plot learning curves
    and measure how fast the agent reaches competency.
    """
    def __init__(self, rolling_window=10):
        self.rolling_window   = rolling_window

        # per-step logs
        self.steps            = []        # global step number
        self.episode_rewards  = []        # reward each episode
        self.episode_steps    = []        # which step each episode ended on
        self.level_at_step    = []        # what level agent is on each episode
        self.rolling_reward   = []        # smoothed reward

        # milestone logs
        self.first_level_clear   = None   # step when level 1 first cleared
        self.level_clear_steps   = {}     # step when each level first cleared
        self.episodes_to_level   = {}     # episode number when each level first cleared

        self._episode_count   = 0

    def log_episode(self, step, reward, level, env):
        self._episode_count += 1
        self.episode_rewards.append(reward)
        self.episode_steps.append(step)
        self.level_at_step.append(level)

        # rolling average
        window = self.episode_rewards[-self.rolling_window:]
        self.rolling_reward.append(np.mean(window))

        # check if a new level was cleared
        if level not in self.level_clear_steps:
            if env.wave_reward >= env.max_reward - len(env.pattern_list) * 6:
                self.level_clear_steps[level]   = step
                self.episodes_to_level[level]   = self._episode_count
                if level == 1 and self.first_level_clear is None:
                    self.first_level_clear = step
                print(f"  ★ Level {level} cleared at step {step} "
                      f"(episode {self._episode_count})")

    def summary(self):
        return {
            "first_level_clear_step":    self.first_level_clear,
            "level_clear_steps":         self.level_clear_steps,
            "episodes_to_level":         self.episodes_to_level,
            "final_rolling_reward":      self.rolling_reward[-1] if self.rolling_reward else 0,
            "mean_reward":               np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }

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

    tracker = ProgressTracker(rolling_window=TRAIN_CONFIG["rolling_window"])

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
        tracker.log_episode(step, episode_reward, env.curr_wave, env)

    eval_episodes = TRAIN_CONFIG["eval_episodes"]
    final_reward = float(np.mean(tracker.episode_rewards[-eval_episodes:]))
    return final_reward, tracker

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

def train_drqn(lr, gamma, batch_size, buffer_size, target_update, hidden_dim, seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup
    env = hexxed()
    env.ready(**ENV_CONFIG)

    policy_net = DRQNNetwork(obs_dim=72, action_dim=7, hidden_dim=hidden_dim).to(device)
    target_net = DRQNNetwork(obs_dim=72, action_dim=7, hidden_dim=hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer    = SequenceReplayBuffer(capacity=buffer_size, seq_len=seq_len)

    tracker = ProgressTracker(rolling_window=TRAIN_CONFIG["rolling_window"])


    epsilon       = TRAIN_CONFIG["epsilon_start"]
    epsilon_end   = TRAIN_CONFIG["epsilon_end"]
    epsilon_decay = TRAIN_CONFIG["epsilon_decay"]
    total_steps   = TRAIN_CONFIG["total_timesteps"]

    step       = 0
    reward_log = []

    # training loop
    while step < total_steps:
        state  = env.reset()
        done   = False
        episode_reward = 0

        # hidden resets every episode, persists across steps within it
        hidden = policy_net.init_hidden(batch_size=1, device=device)

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = env.action_space.sample()
                # step hidden forward even on random actions
                with torch.no_grad():
                    _, hidden = policy_net(state_t, hidden)
            else:
                with torch.no_grad():
                    q_values, hidden = policy_net(state_t, hidden)
                    action = q_values[0, 0, :].argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            step += 1

            if buffer.is_ready(batch_size):
                loss = _drqn_train_step(
                    policy_net, target_net, optimizer,
                    buffer, batch_size, gamma, device
                )

            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        reward_log.append(episode_reward)
        tracker.log_episode(step, episode_reward, env.curr_wave, env)


    eval_episodes = TRAIN_CONFIG["eval_episodes"]
    final_reward  = float(np.mean(tracker.episode_rewards[-eval_episodes:]))
    return final_reward, tracker


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
