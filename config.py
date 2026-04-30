# config.py

ENV_CONFIG = {
    "num_vertices": 6,
    "step_per_pattern": 6,
    "levels": 6,
    "shuffle_patterns": True,
    "random_rolls": False,
    "perfect_bonus": False,
    "render_mode": 0
}

TRAIN_CONFIG = {
    "total_timesteps": 10000,
    "eval_episodes":   20,
    "epsilon_start":   1.0,
    "epsilon_end":     0.05,
    "epsilon_decay":   0.99,
    "log_every":       100,
    "rolling_window":  10,
}

# Optuna search spaces — (type, low, high, log_scale)
# These are the bounds Optuna explores, not fixed values
DQN_SEARCH_SPACE = {
    "lr":           ("float", 1e-4, 1e-2, True),
    "gamma":        ("float", 0.90, 0.999, False),
    "buffer_size":  ("int",   5_000, 50_000, False),
    "batch_size":   ("int",   32, 256, False),
    "target_update":("int",   10, 500, False),
    "hidden_dim":   ("int",   64, 512, False),
}

DRQN_SEARCH_SPACE = {
    "lr":           ("float", 1e-4, 1e-2, True),
    "gamma":        ("float", 0.90, 0.999, False),
    "buffer_size":  ("int",   5_000, 50_000, False),
    "batch_size":   ("int",   32, 256, False),
    "target_update":("int",   10, 500, False),
    "hidden_dim":   ("int",   64, 512, False),
    "seq_len":      ("int",   4, 12, False),    # DRQN only — capped to match episode lengths at early levels
}

# MLflow
MLFLOW_TRACKING_URI  = "mlruns"
DQN_EXPERIMENT_NAME  = "hexxed_dqn"
DRQN_EXPERIMENT_NAME = "hexxed_drqn"

# Optuna
OPTUNA_DIR      = "optuna_studies"
DQN_STUDY_NAME  = "dqn_study"
DRQN_STUDY_NAME = "drqn_study"
N_TRIALS        = 50
