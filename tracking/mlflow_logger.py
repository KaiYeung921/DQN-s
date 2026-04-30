# tracking/mlflow_logger.py
import mlflow
import numpy as np
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


def log_episode(episode, reward, epsilon, level, loss=None):
    """Updated to also log level and rolling reward."""
    mlflow.log_metric("episode_reward", reward,  step=episode)
    mlflow.log_metric("epsilon",        epsilon, step=episode)
    mlflow.log_metric("level",          level,   step=episode)
    if loss is not None:
        mlflow.log_metric("loss", loss, step=episode)


def log_milestone(name, step):
    """
    Logs a one-time milestone event — e.g. when level 1 is first cleared.
    Stored as a param since it's a single value not a curve.
    """
    mlflow.log_param(name, step)


def log_training_phase_rewards(early, mid, late):
    """Logs the three-phase reward breakdown as summary metrics."""
    mlflow.log_metric("reward_early", early)
    mlflow.log_metric("reward_mid",   mid)
    mlflow.log_metric("reward_late",  late)


def log_trial_result(score):
    """Logs the scalar objective value Optuna optimizes — the learning speed score."""
    mlflow.log_metric("learning_speed_score", score)


def log_summary(tracker):
    """
    Call once at the end of a training run.
    Logs everything from the ProgressTracker as a structured summary.
    """
    summary = tracker.summary()

    # first level clear step — most important single milestone
    if summary["first_level_clear_step"] is not None:
        mlflow.log_metric("first_level_clear_step", summary["first_level_clear_step"])
    else:
        mlflow.log_metric("first_level_clear_step", -1)  # -1 = never cleared

    # step when each level was first cleared
    for level, step in summary["level_clear_steps"].items():
        mlflow.log_metric(f"level_{level}_clear_step", step)

    # episode when each level was first cleared
    for level, ep in summary["episodes_to_level"].items():
        mlflow.log_metric(f"level_{level}_clear_episode", ep)

    # reward summary
    mlflow.log_metric("final_rolling_reward", summary["final_rolling_reward"])
    mlflow.log_metric("mean_reward",          summary["mean_reward"])

    # training phase breakdown
    rewards = tracker.episode_rewards
    if len(rewards) >= 3:
        n = len(rewards) // 3
        log_training_phase_rewards(
            early=float(np.mean(rewards[:n])),
            mid=float(np.mean(rewards[n:2*n])),
            late=float(np.mean(rewards[2*n:]))
        )