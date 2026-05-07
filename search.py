"""
search.py — Optuna hyperparameter search for DQN and DRQN on Hexxed.

Usage:
    python search.py --agent dqn   --trials 20
    python search.py --agent drqn  --trials 20
"""

import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # silence per-trial Optuna logs

from hexxed import hexxed
from agents import DQNNetwork, DRQNNetwork, ReplayBuffer, SequenceReplayBuffer


# ── Environment config ────────────────────────────────────────────────────────

ENV_CONFIG = dict(
    num_vertices    = 6,
    step_per_pattern= 6,
    levels          = 6,
    shuffle_patterns= True,
    random_rolls    = False,
    perfect_bonus   = False,
    render_mode     = 0,   # 0 = silent
)

# Derived from ENV_CONFIG
OBS_DIM    = ENV_CONFIG["num_vertices"] * 2 * ENV_CONFIG["num_vertices"]  # 72
ACTION_DIM = ENV_CONFIG["num_vertices"] + 1                                # 7


# ── Training config ───────────────────────────────────────────────────────────

TOTAL_STEPS   = 30_000   # steps per trial
EPSILON_START = 1.0      # start fully random
EPSILON_END   = 0.05     # minimum exploration
EPSILON_DECAY = 0.99985  # multiplicative decay per step (~hits 0.05 at 20k steps)
TRAIN_FREQ    = 4        # gradient update every N environment steps


# ── Hyperparameter search spaces ──────────────────────────────────────────────
# Each entry: (type, low, high, log-scale?)
# "log=True" means Optuna samples on a log scale — good for learning rate.

DQN_SEARCH_SPACE = {
    "lr"           : ("float", 1e-4, 1e-2,  True),
    "gamma"        : ("float", 0.90, 0.999, False),
    "batch_size"   : ("int",   32,   256,   False),
    "buffer_size"  : ("int",   5000, 25000, False),
    "target_update": ("int",   10,   500,   False),
    "hidden_dim"   : ("int",   64,   512,   False),
}

DRQN_SEARCH_SPACE = {
    **DQN_SEARCH_SPACE,
    "seq_len": ("int", 4, 12, False),  # LSTM burn-in sequence length
}


# ── Objective metric ──────────────────────────────────────────────────────────

def learning_speed_score(level_clear_steps, total_steps):
    """Higher = faster. Sum of remaining budget when each level was first cleared.
    A trial that clears all 6 levels early scores higher than one that barely clears level 1."""
    if not level_clear_steps:
        return 0.0
    return float(sum(total_steps - step for step in level_clear_steps.values()))


# ── Training loops ────────────────────────────────────────────────────────────

def train_dqn(lr, gamma, batch_size, buffer_size, target_update, hidden_dim):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = hexxed()
    env.ready(**ENV_CONFIG)

    policy_net = DQNNetwork(OBS_DIM, ACTION_DIM, hidden_dim).to(device)
    target_net = DQNNetwork(OBS_DIM, ACTION_DIM, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer    = ReplayBuffer(capacity=buffer_size)

    epsilon           = EPSILON_START
    level_clear_steps = {}   # {level: step_when_cleared}
    step = 0

    while step < TOTAL_STEPS:
        state = env.reset()
        done  = False

        while not done:
            # epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q = policy_net(torch.FloatTensor(state).unsqueeze(0).to(device))
                    action = q.argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state  = next_state
            step  += 1
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            # gradient update every TRAIN_FREQ steps
            if step % TRAIN_FREQ == 0 and buffer.is_ready(batch_size):
                _dqn_update(policy_net, target_net, optimizer,
                            buffer, batch_size, gamma, device)

            # sync target network
            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # check for level clear at end of episode
        level = env.curr_wave
        if level not in level_clear_steps:
            all_subwaves_done = env.subwave_num == len(env.pattern_list)
            passed_threshold  = env.wave_reward >= env.max_reward - len(env.pattern_list) * 15
            if all_subwaves_done and passed_threshold:
                level_clear_steps[level] = step
                print(f"  ★ Level {level} cleared at step {step}")

        # stop once all levels cleared
        if ENV_CONFIG["levels"] in level_clear_steps:
            break

    return learning_speed_score(level_clear_steps, TOTAL_STEPS)


def _dqn_update(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
    next_states, dones       = next_states.to(device), dones.to(device)

    # Q(s, a) for the actions actually taken
    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: r + γ * max_a' Q_target(s', a')
    with torch.no_grad():
        target_q = rewards + gamma * target_net(next_states).max(dim=1)[0] * (1 - dones)

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()


def train_drqn(lr, gamma, batch_size, buffer_size, target_update, hidden_dim, seq_len):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = hexxed()
    env.ready(**ENV_CONFIG)

    policy_net = DRQNNetwork(OBS_DIM, ACTION_DIM, hidden_dim).to(device)
    target_net = DRQNNetwork(OBS_DIM, ACTION_DIM, hidden_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    buffer    = SequenceReplayBuffer(capacity=buffer_size, seq_len=seq_len)

    epsilon           = EPSILON_START
    level_clear_steps = {}
    step = 0

    while step < TOTAL_STEPS:
        state  = env.reset()
        done   = False
        # hidden state resets each episode, persists across steps within it
        hidden = policy_net.init_hidden(batch_size=1, device=device)

        while not done:
            state_t = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)

            if random.random() < epsilon:
                action = env.action_space.sample()
                # LSTM must step even on random actions to keep hidden state consistent
                with torch.no_grad():
                    _, hidden = policy_net(state_t, hidden)
            else:
                with torch.no_grad():
                    q_vals, hidden = policy_net(state_t, hidden)
                    action = q_vals[0, 0, :].argmax().item()

            next_state, reward, done, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state  = next_state
            step  += 1
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            if step % TRAIN_FREQ == 0 and buffer.is_ready(batch_size):
                _drqn_update(policy_net, target_net, optimizer,
                             buffer, batch_size, gamma, device)

            if step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        level = env.curr_wave
        if level not in level_clear_steps:
            all_subwaves_done = env.subwave_num == len(env.pattern_list)
            passed_threshold  = env.wave_reward >= env.max_reward - len(env.pattern_list) * 15
            if all_subwaves_done and passed_threshold:
                level_clear_steps[level] = step
                print(f"  ★ Level {level} cleared at step {step}")

        if ENV_CONFIG["levels"] in level_clear_steps:
            break

    return learning_speed_score(level_clear_steps, TOTAL_STEPS)


def _drqn_update(policy_net, target_net, optimizer, buffer, batch_size, gamma, device):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    # shapes: states/next_states (batch, seq_len, 72)
    #         actions/rewards/dones (batch, seq_len)
    states, actions, rewards = states.to(device), actions.to(device), rewards.to(device)
    next_states, dones       = next_states.to(device), dones.to(device)

    # zero hidden state for each training sequence
    h  = policy_net.init_hidden(batch_size=batch_size, device=device)
    th = target_net.init_hidden(batch_size=batch_size, device=device)

    # forward the full sequence; only train on the LAST step
    # (earlier steps = LSTM burn-in to build up hidden state)
    q_all,  _ = policy_net(states,      h)
    nq_all, _ = target_net(next_states, th)

    q_last  = q_all[:, -1, :]      # (batch, 7)
    nq_last = nq_all[:, -1, :]     # (batch, 7)
    a_last  = actions[:, -1]       # (batch,)
    r_last  = rewards[:, -1]       # (batch,)
    d_last  = dones[:, -1]         # (batch,)

    current_q = q_last.gather(1, a_last.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        target_q = r_last + gamma * nq_last.max(dim=1)[0] * (1 - d_last)

    loss = F.mse_loss(current_q, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10)
    optimizer.step()


# ── Optuna objectives ─────────────────────────────────────────────────────────

def _suggest_params(trial, search_space):
    """Sample one set of hyperparameters from the search space."""
    params = {}
    for name, (kind, low, high, log) in search_space.items():
        if kind == "float":
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif kind == "int":
            params[name] = trial.suggest_int(name, low, high)
    return params


def dqn_objective(trial):
    params = _suggest_params(trial, DQN_SEARCH_SPACE)
    print(f"\nTrial {trial.number} | params: {params}")
    score = train_dqn(**params)
    print(f"Trial {trial.number} | score: {score:.1f}")
    return score


def drqn_objective(trial):
    params = _suggest_params(trial, DRQN_SEARCH_SPACE)
    print(f"\nTrial {trial.number} | params: {params}")
    score = train_drqn(**params)
    print(f"Trial {trial.number} | score: {score:.1f}")
    return score


# ── Study runner ──────────────────────────────────────────────────────────────

def run_study(agent_type, n_trials):
    objective = dqn_objective if agent_type == "dqn" else drqn_objective

    # In-memory storage — no files needed. Use JournalStorage with a file path
    # if you want to resume an interrupted run across sessions.
    study = optuna.create_study(
        direction="maximize",                      # maximize learning_speed_score
        sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
    )

    print(f"\n{'='*60}")
    print(f"Optuna search: {agent_type.upper()}  |  {n_trials} trials")
    print(f"{'='*60}")

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n{'='*60}")
    print(f"Best {agent_type.upper()} hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:15s}: {v}")
    print(f"  {'score':15s}: {study.best_value:.1f}")
    print(f"{'='*60}\n")

    return study.best_params


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for DQN / DRQN on Hexxed.")
    parser.add_argument("--agent",  choices=["dqn", "drqn"], default="dqn",
                        help="Which agent to search (default: dqn)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials to run (default: 20)")
    args = parser.parse_args()

    run_study(args.agent, args.trials)
