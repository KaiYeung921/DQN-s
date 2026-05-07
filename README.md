# DQN vs DRQN on Hexxed

Compares a standard Deep Q-Network (DQN) and a Deep Recurrent Q-Network (DRQN) on Hexxed, a hexagonal puzzle game. Hyperparameters are tuned automatically with Optuna.

---

## Files

```
hexxed.py   — Gym environment (professor's implementation)
agents.py   — DQN and DRQN networks + replay buffers
search.py   — Training loops and Optuna hyperparameter search
spawn.txt   — Puzzle pattern data (required by the environment)
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running

```bash
python3 search.py --agent dqn  --trials 20
python3 search.py --agent drqn --trials 20
```

Each trial trains the agent for 30,000 steps and returns a **learning speed score** (sum of remaining budget when each level was first cleared — higher is faster). Optuna uses this score to decide which hyperparameters to try next.

Best parameters are printed at the end.

---

## How It Works

```
search.py
  └── Optuna suggests a set of hyperparameters for each trial
        └── train_dqn / train_drqn runs one 30k-step training run
              ├── hexxed env    → states and rewards
              ├── ReplayBuffer  → stores transitions, samples batches
              ├── DQNNetwork    → predicts Q-values, updated via Bellman loss
              └── returns learning_speed_score
        └── Optuna receives the score and selects the next hyperparameters
  └── after N trials, prints best hyperparameters found
```

---

## Hyperparameter Search Space

| Parameter | Range | Notes |
|---|---|---|
| `lr` | 1e-4 – 1e-2 | log scale |
| `gamma` | 0.90 – 0.999 | discount factor |
| `batch_size` | 32 – 256 | |
| `buffer_size` | 5,000 – 25,000 | |
| `target_update` | 10 – 500 | steps between target net syncs |
| `hidden_dim` | 64 – 512 | network width |
| `seq_len` | 4 – 12 | DRQN only — LSTM burn-in length |

---

## Architecture

**DQN** — each state processed independently:
```
state (72,) → Linear → ReLU → Linear → Q-values (7,)
```

**DRQN** — hidden state carries context across timesteps:
```
state (72,) → Linear → ReLU → LSTM → Linear → Q-values (7,)
                                ↑
                    hidden state persists within an episode
```
