# training/sb3_optuna_search.py
import os
import mlflow
import optuna

from config import (
    OPTUNA_DIR,
    DQN_SEARCH_SPACE,
    MLFLOW_TRACKING_URI,
    TRAIN_CONFIG,
)
from tracking.mlflow_logger import (
    start_run,
    end_run,
    log_trial_result,
    log_summary,
)
from training.sb3_train import train_sb3_dqn

SB3_DQN_STUDY_NAME  = "sb3_dqn_study"
SB3_EXPERIMENT_NAME = "hexxed_sb3_dqn"
SB3_N_TRIALS        = 20


def build_params(trial, search_space):
    params = {}
    for name, spec in search_space.items():
        kind, low, high, log = spec
        if kind == "float":
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif kind == "int":
            params[name] = trial.suggest_int(name, low, high)
    return params


def make_sb3_dqn_objective():
    def objective(trial):
        params = build_params(trial, DQN_SEARCH_SPACE)

        # set experiment directly — keeps sb3 runs in their own MLflow experiment
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(SB3_EXPERIMENT_NAME)
        start_run(params)

        try:
            score, tracker = train_sb3_dqn(**params)
            log_trial_result(score)
            log_summary(tracker)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("-inf")
        finally:
            end_run()

    return objective


def run_sb3_study():
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    db_path = f"sqlite:///{OPTUNA_DIR}/{SB3_DQN_STUDY_NAME}.db"

    study = optuna.create_study(
        study_name=SB3_DQN_STUDY_NAME,
        storage=db_path,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting SB3 DQN sanity check — {SB3_N_TRIALS} trials")
    print(f"Resuming from trial {len(study.trials)}" if study.trials else "Fresh study")
    print(f"Steps per trial: {TRAIN_CONFIG['search_timesteps']:,}")

    study.optimize(
        make_sb3_dqn_objective(),
        n_trials=SB3_N_TRIALS,
        show_progress_bar=True,
    )

    print(f"\nBest SB3 DQN params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  → learning_speed_score: {study.best_value:.2f}")

    return study.best_params
