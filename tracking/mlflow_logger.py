# tracking/mlflow_logger.py
import mlflow
from config import (
    MLFLOW_TRACKING_URI,
    DQN_EXPERIMENT_NAME,
    DRQN_EXPERIMENT_NAME
)


def setup_mlflow(agent_type):
    """
    Call once before any training run.
    Tells MLflow where to save data and which experiment to log under.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    if agent_type == "dqn":
        mlflow.set_experiment(DQN_EXPERIMENT_NAME)
    else:
        mlflow.set_experiment(DRQN_EXPERIMENT_NAME)


def start_run(params):
    """
    Opens a new MLflow run and logs the hyperparameters for this trial.
    Returns the run object — caller is responsible for ending it.
    """
    run = mlflow.start_run()
    mlflow.log_params(params)
    return run


def end_run():
    """Closes the current active MLflow run."""
    mlflow.end_run()


def log_episode(episode, reward, epsilon, loss=None):
    """
    Call at the end of every episode during training.
    Logs training metrics at a specific step so you get a curve over time.
    """
    mlflow.log_metric("episode_reward", reward,   step=episode)
    mlflow.log_metric("epsilon",        epsilon,  step=episode)
    if loss is not None:
        mlflow.log_metric("loss", loss, step=episode)


def log_trial_result(final_reward):
    """
    Call once at the end of a full training run.
    Logs the single number Optuna uses to compare trials.
    """
    mlflow.log_metric("final_reward", final_reward)