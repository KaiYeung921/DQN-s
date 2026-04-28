# RLCompNeuro — DQN vs DRQN on Hexxed

A comparison of a standard Deep Q-Network (DQN) and a Deep Recurrent Q-Network (DRQN) playing Hexxed, a hexagonal puzzle game. Hyperparameter search is handled by Optuna and all experiments are tracked with MLflow.

---

## Project Structure

```
RLCompNeuro/
├── env/
│   ├── hexxed.py           # Gym environment (professor's implementation)
│   ├── spawn.txt           # Puzzle pattern data, required by the environment
│   └── __init__.py
│
├── agents/
│   ├── dqn.py              # DQNNetwork — standard MLP Q-network
│   ├── drqn.py             # DRQNNetwork — LSTM-based Q-network
│   ├── buffer.py           # ReplayBuffer (DQN) and SequenceReplayBuffer (DRQN)
│   └── __init__.py
│
├── training/
│   ├── train.py            # train_dqn and train_drqn — full training loops
│   ├── optuna_search.py    # Optuna study setup, objective functions
│   └── __init__.py
│
├── tracking/
│   ├── mlflow_logger.py    # MLflow helpers — setup, logging params and metrics
│   └── __init__.py
│
├── analysis/
│   └── plots.py            # Human vs agent comparison plots
│
├── config.py               # All hyperparameters and Optuna search spaces
├── main.py                 # Entry point
├── requirements.txt
└── README.md
```

---

## How It All Connects

```
config.py
  └── defines hyperparameter search spaces, env settings, MLflow/Optuna config

main.py --agent dqn
  └── calls run_study("dqn") in optuna_search.py
        └── Optuna suggests hyperparameters for each trial
              └── calls train_dqn(**params) in train.py
                    ├── hexxed env    → produces states and rewards
                    ├── ReplayBuffer  → stores transitions, samples batches
                    ├── DQNNetwork    → predicts Q-values, updated via Bellman loss
                    └── returns final_reward
              └── MLflow logs params + metrics for the trial
              └── Optuna receives final_reward, decides next hyperparameters
        └── after N_TRIALS, returns best hyperparameters found
```

The DRQN path is identical, substituting `DRQNNetwork` and `SequenceReplayBuffer` and adding hidden state management in the training loop.

---

## Setup

**1. Clone the repo and create a virtual environment:**
```bash
git clone <your-repo-url>
cd RLCompNeuro
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Download puzzle data:**
```bash
wget --no-check-certificate 'https://drive.google.com/uc?id=1sesWIlEC2WPLa8hJQlmp-SwX7OwCpUPI' -O spawn.txt
```

---

## Running

**Train DQN:**
```bash
python main.py --agent dqn
```

**Train DRQN:**
```bash
python main.py --agent drqn
```

Each run launches an Optuna study that trains the agent across `N_TRIALS` trials (set in `config.py`). Results are saved automatically — if a run is interrupted, it resumes from the last completed trial.

**View results in MLflow:**
```bash
mlflow ui
```
Then open `http://localhost:5000`. DQN and DRQN runs are separated into their own experiments.

---

## Key Hyperparameters

All hyperparameters live in `config.py`. Optuna searches within the bounds defined in `DQN_SEARCH_SPACE` and `DRQN_SEARCH_SPACE`.

| Parameter | Description | Searched range |
|---|---|---|
| `lr` | Learning rate | 1e-4 to 1e-2 (log scale) |
| `gamma` | Discount factor | 0.90 to 0.999 |
| `buffer_size` | Replay buffer capacity | 5,000 to 50,000 |
| `batch_size` | Training batch size | 32 to 256 |
| `target_update` | Steps between target net syncs | 10 to 500 |
| `hidden_dim` | Network hidden layer size | 64 to 512 |
| `seq_len` | Sequence length (DRQN only) | 4 to 32 |

---

## Architecture

**DQN:**
```
state (72,) → Linear → ReLU → Linear → ReLU → Linear → Q-values (7,)
```

**DRQN:**
```
state (72,) → Linear → ReLU → LSTM → Linear → Q-values (7,)
                                ↑
                    hidden state carries temporal context
                    forward across steps within an episode
```

The key difference is memory. The DQN sees each game state independently. The DRQN maintains a hidden state across timesteps, giving it context about what happened earlier in the episode.

---

## Resuming Interrupted Runs

Optuna studies are persisted to SQLite in `optuna_studies/`. If a run is interrupted, just re-run the same command — Optuna will detect the existing study and resume from the last completed trial.

```bash
python main.py --agent dqn   # resumes automatically if interrupted
```

MLflow runs that were open when a crash occurred will appear as incomplete in the UI — this is expected and does not affect Optuna's ability to resume.

---

## Comparing Results

After both studies complete, the best parameters for each agent can be retrieved directly from the Optuna database:

```python
import optuna

dqn_study = optuna.load_study(
    study_name="dqn_study",
    storage="sqlite:///optuna_studies/dqn_study.db"
)
drqn_study = optuna.load_study(
    study_name="drqn_study",
    storage="sqlite:///optuna_studies/drqn_study.db"
)

print("Best DQN reward: ", dqn_study.best_value)
print("Best DRQN reward:", drqn_study.best_value)
print("Best DQN params: ", dqn_study.best_params)
print("Best DRQN params:", drqn_study.best_params)
```

Use `analysis/plots.py` to generate human vs agent comparison plots mirroring your professor's notebook.
