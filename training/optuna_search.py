# training/optuna_search.py
import os
import optuna
from config import (
    OPTUNA_DIR,
    DQN_STUDY_NAME,
    DRQN_STUDY_NAME,
    DQN_SEARCH_SPACE,
    DRQN_SEARCH_SPACE,
    N_TRIALS
)
from tracking.mlflow_logger import (
    setup_mlflow,
    start_run,
    end_run,
    log_trial_result,
    log_summary
)
from training.train import train_dqn, train_drqn


def build_params(trial, search_space):
    """
    Translates your config search space into actual Optuna suggestions.
    For each hyperparameter, asks Optuna to suggest a value within the bounds.
    """
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


def make_dqn_objective():
    """
    Returns the objective function for DQN Optuna study.
    Objective functions must take exactly one argument — the trial object.
    """
    def objective(trial):
        # ask Optuna to suggest hyperparameters for this trial
        params = build_params(trial, DQN_SEARCH_SPACE)

        # open an MLflow run for this trial
        setup_mlflow("dqn")
        start_run(params)

        try:
            # run full training, get back the objective score
            final_reward, tracker = train_dqn(**params)
            log_trial_result(final_reward)
            log_summary(tracker)
            return final_reward

        except Exception as e:
            # if a trial crashes, tell Optuna to mark it as failed
            # rather than crashing the whole study
            print(f"Trial failed: {e}")
            return float("-inf")

        finally:
            # always close the MLflow run even if training crashed
            end_run()

    return objective


def make_drqn_objective():
    """
    Returns the objective function for DRQN Optuna study.
    Identical structure to DQN objective — just different search space
    and training function.
    """
    def objective(trial):
        params = build_params(trial, DRQN_SEARCH_SPACE)

        setup_mlflow("drqn")
        start_run(params)

        try:
            final_reward, tracker = train_drqn(**params)
            log_trial_result(final_reward)
            log_summary(tracker)
            return final_reward

        except Exception as e:
            print(f"Trial failed: {e}")
            return float("-inf")

        finally:
            end_run()

    return objective


def run_study(agent_type):
    """
    Creates or resumes an Optuna study for the given agent type,
    runs N_TRIALS trials, and returns the best hyperparameters found.
    """
    os.makedirs(OPTUNA_DIR, exist_ok=True)

    if agent_type == "dqn":
        study_name = DQN_STUDY_NAME
        db_path    = f"sqlite:///{OPTUNA_DIR}/{study_name}.db"
        objective  = make_dqn_objective()
    else:
        study_name = DRQN_STUDY_NAME
        db_path    = f"sqlite:///{OPTUNA_DIR}/{study_name}.db"
        objective  = make_drqn_objective()

    # create_study with load_if_exists=True means:
    # - first run: creates a fresh study and saves to db
    # - subsequent runs: loads existing study and resumes from where it left off
    # this is what makes Optuna survive crashes and session restarts
    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        direction="maximize",       # we want to maximize final_reward
        load_if_exists=True,        # resume if interrupted
        sampler=optuna.samplers.TPESampler(seed=42)  # reproducible sampling
    )

    # optionally silence Optuna's per-trial logging since MLflow handles it
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print(f"Starting {agent_type.upper()} study — {N_TRIALS} trials")
    print(f"Resuming from trial {len(study.trials)}" if study.trials else "Fresh study")

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True      # shows a tqdm bar across trials
    )

    print(f"\nBest {agent_type.upper()} params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"  → final_reward: {study.best_value:.2f}")

    return study.best_params