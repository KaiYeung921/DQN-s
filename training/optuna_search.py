# training/optuna_search.py
import os
import optuna
import mlflow
from config import (
    OPTUNA_DIR,
    DQN_STUDY_NAME,
    DRQN_STUDY_NAME,
    DQN_SEARCH_SPACE,
    DRQN_SEARCH_SPACE,
    N_TRIALS,
    N_JOBS,
    MLFLOW_TRACKING_URI,
)
from tracking.mlflow_logger import (
    setup_mlflow,
    start_run,
    end_run,
    log_trial_result,
    log_summary,
)
from training.train import train_dqn, train_drqn


def build_params(trial, search_space):
    params = {}
    for name, spec in search_space.items():
        kind = spec[0]
        low  = spec[1]
        high = spec[2]
        log  = spec[3]
        if kind == "float":
            params[name] = trial.suggest_float(name, low, high, log=log)
        elif kind == "int":
            params[name] = trial.suggest_int(name, low, high)
    return params


def make_dqn_objective(experiment_name):
    """
    Returns the DQN objective for Optuna.
    experiment_name is passed explicitly so each worker process can set up
    MLflow independently without relying on shared global state.
    """
    def objective(trial):
        params = build_params(trial, DQN_SEARCH_SPACE)

        # each worker process calls this independently — safe because the
        # experiment already exists (created in run_study before workers start)
        setup_mlflow("dqn", experiment_name=experiment_name)
        start_run(params)

        try:
            score, tracker = train_dqn(**params)
            log_trial_result(score)
            log_summary(tracker)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("-inf")
        finally:
            end_run()

    return objective


def make_drqn_objective(experiment_name):
    """
    Returns the DRQN objective for Optuna.
    Same parallel-safe pattern as DQN objective.
    """
    def objective(trial):
        params = build_params(trial, DRQN_SEARCH_SPACE)

        setup_mlflow("drqn", experiment_name=experiment_name)
        start_run(params)

        try:
            score, tracker = train_drqn(**params)
            log_trial_result(score)
            log_summary(tracker)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return float("-inf")
        finally:
            end_run()

    return objective


def run_study(agent_type, study_name=None, n_jobs=None):
    """
    Creates or resumes an Optuna study and runs N_TRIALS trials.

    n_jobs controls parallel trials. With n_jobs > 1:
      - Uses JournalStorage (file-based, concurrent-safe) instead of SQLite
      - MLflow experiment is pre-created in this process before workers start
        to avoid a race condition on first experiment creation
      - Progress bar is disabled (Optuna does not support it with n_jobs > 1)

    study_name overrides config defaults — use it to run a new isolated
    experiment without touching existing storage or MLflow runs.
    """
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    n_jobs = n_jobs if n_jobs is not None else N_JOBS

    if agent_type == "dqn":
        study_name = study_name or DQN_STUDY_NAME
        objective  = make_dqn_objective(experiment_name=study_name)
    else:
        study_name = study_name or DRQN_STUDY_NAME
        objective  = make_drqn_objective(experiment_name=study_name)

    # JournalStorage uses file-based locking — safe for concurrent writers,
    # no database server required. SQLite serializes writes and raises
    # locking errors when multiple processes hit it simultaneously.
    journal_path = f"{OPTUNA_DIR}/{study_name}.log"
    # JournalFileBackend (Optuna <4) was renamed to JournalFileStorage (Optuna 4+)
    _backend_cls = getattr(optuna.storages, "JournalFileStorage",
                           optuna.storages.JournalFileBackend)
    storage = optuna.storages.JournalStorage(_backend_cls(journal_path))

    # pre-create the MLflow experiment in this process before workers start
    # so parallel workers never race to create it simultaneously
    setup_mlflow(agent_type, experiment_name=study_name)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Starting {agent_type.upper()} study — {N_TRIALS} trials, {n_jobs} parallel")
    print(f"Resuming from {completed} completed trials" if completed else "Fresh study")
    print(f"Storage: {journal_path}")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        n_jobs=n_jobs,
        # progress bar does not work with n_jobs > 1
        show_progress_bar=(n_jobs == 1),
    )

    print(f"\nBest {agent_type.upper()} params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  -> learning_speed_score: {study.best_value:.2f}")

    return study.best_params
