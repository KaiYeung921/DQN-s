# analysis/plots.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import requests
import optuna
from io import BytesIO

import mlflow
from mlflow.tracking import MlflowClient

from config import (
    MLFLOW_TRACKING_URI,
    DQN_EXPERIMENT_NAME,
    DRQN_EXPERIMENT_NAME,
    TRAIN_CONFIG,
    OPTUNA_DIR,
    DQN_STUDY_NAME,
    DRQN_STUDY_NAME,
)


# ── mlflow helpers ────────────────────────────────────────────────────────────

def get_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_runs(client, experiment_name):
    """Returns all completed runs for an experiment, sorted by final_reward."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in mlruns/")
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.final_reward DESC"]
    )
    print(f"{experiment_name}: {len(runs)} completed runs")
    return runs


def get_best_run(runs):
    """Returns the single best run by final_reward."""
    return runs[0] if runs else None


def get_metric_curve(client, run_id, metric_name):
    """
    Returns (steps, values) for a metric that was logged per episode.
    Used for curves like episode_reward, level, epsilon.
    """
    history = client.get_metric_history(run_id, metric_name)
    if not history:
        return [], []
    steps  = [m.step  for m in history]
    values = [m.value for m in history]
    return steps, values


def get_scalar(run, metric_name, default=None):
    """Returns a single summary scalar from a run."""
    return run.data.metrics.get(metric_name, default)


def get_all_scalars(runs, metric_name):
    """Returns a list of scalar values across all runs — for distribution plots."""
    return [
        r.data.metrics[metric_name]
        for r in runs
        if metric_name in r.data.metrics
    ]


def smooth(values, window=10):
    """Rolling average to smooth noisy curves."""
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid')


# ── human data ────────────────────────────────────────────────────────────────

def load_human_data():
    url = 'https://drive.google.com/uc?export=download&id=1m4RPxkOraYFfZLHQ7yhGdM6Y3oeTP8YP'
    r   = requests.get(url, allow_redirects=True)
    df  = pd.read_parquet(BytesIO(r.content))
    winners = df[(df['learning'] == True) & (df['learned'] == True)]
    print(f"Human data loaded: {len(winners)} winner rows")
    print(f"Columns: {list(winners.columns)}")
    return winners


# ── plot 1 — learning curves ──────────────────────────────────────────────────

def plot_learning_curves(client, dqn_runs, drqn_runs, ax, human_data=None):
    """
    Rolling reward over episodes for best DQN and best DRQN run.
    Vertical dashed line marks when each agent first cleared level 1.
    Directly answers: who learns faster?
    """
    dqn_best  = get_best_run(dqn_runs)
    drqn_best = get_best_run(drqn_runs)

    for run, label, color in [
        (dqn_best,  "DQN",  "#DD8452"),
        (drqn_best, "DRQN", "#55A868")
    ]:
        if run is None:
            continue
        steps, values = get_metric_curve(client, run.info.run_id, "episode_reward")
        if not values:
            continue

        smoothed = smooth(values, window=10)
        ax.plot(range(len(smoothed)), smoothed,
                label=label, color=color, linewidth=2)

        # mark first level clear
        clear_ep = get_scalar(run, "level_1_clear_episode")
        if clear_ep is not None:
            ax.axvline(clear_ep, color=color, linestyle="--", alpha=0.6,
                       label=f"{label} clears L1 (ep {int(clear_ep)})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("1 — Learning curves (best trial each)")
    ax.legend(fontsize=8)


# ── plot 2 — training phase breakdown ────────────────────────────────────────

def plot_phase_breakdown(dqn_runs, drqn_runs, ax):
    """
    Average reward in early / mid / late thirds of training.
    Averaged across ALL trials not just best — shows consistent advantage.
    Answers: where in training does the advantage show up?
    """
    phases = ["early", "mid", "late"]
    labels = ["Early\n(first third)", "Mid\n(second third)", "Late\n(final third)"]

    dqn_means  = [np.mean(get_all_scalars(dqn_runs,  f"reward_{p}")) for p in phases]
    drqn_means = [np.mean(get_all_scalars(drqn_runs, f"reward_{p}")) for p in phases]
    dqn_stds   = [np.std(get_all_scalars(dqn_runs,   f"reward_{p}")) for p in phases]
    drqn_stds  = [np.std(get_all_scalars(drqn_runs,  f"reward_{p}")) for p in phases]

    x     = np.arange(3)
    width = 0.35

    ax.bar(x - width/2, dqn_means,  width,
           yerr=dqn_stds,  label="DQN",  color="#DD8452", alpha=0.85,
           capsize=4, error_kw={"linewidth": 1})
    ax.bar(x + width/2, drqn_means, width,
           yerr=drqn_stds, label="DRQN", color="#55A868", alpha=0.85,
           capsize=4, error_kw={"linewidth": 1})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Average reward")
    ax.set_title("2 — Reward by training phase (all trials)")
    ax.legend(fontsize=8)


# ── plot 3 — first level clear distribution ───────────────────────────────────

def plot_first_clear_distribution(dqn_runs, drqn_runs, ax):
    """
    Histogram of first_level_clear_step across all Optuna trials.
    Shows how consistently each agent clears level 1 and how quickly.
    Answers: is DRQN's advantage consistent or just lucky?
    """
    dqn_clears  = get_all_scalars(dqn_runs,  "first_level_clear_step")
    drqn_clears = get_all_scalars(drqn_runs, "first_level_clear_step")

    # filter out -1 (never cleared)
    dqn_clears  = [v for v in dqn_clears  if v > 0]
    drqn_clears = [v for v in drqn_clears if v > 0]

    if not dqn_clears and not drqn_clears:
        ax.text(0.5, 0.5, "No level clears recorded yet",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("3 — First level clear distribution")
        return

    all_vals = dqn_clears + drqn_clears
    bins     = np.linspace(min(all_vals), max(all_vals), 20)

    ax.hist(dqn_clears,  bins=bins, alpha=0.6,
            label=f"DQN  (n={len(dqn_clears)})",  color="#DD8452", density=True)
    ax.hist(drqn_clears, bins=bins, alpha=0.6,
            label=f"DRQN (n={len(drqn_clears)})", color="#55A868", density=True)

    # mean lines
    if dqn_clears:
        ax.axvline(np.mean(dqn_clears),  color="#DD8452",
                   linestyle="--", linewidth=1.5,
                   label=f"DQN mean:  {np.mean(dqn_clears):.0f}")
    if drqn_clears:
        ax.axvline(np.mean(drqn_clears), color="#55A868",
                   linestyle="--", linewidth=1.5,
                   label=f"DRQN mean: {np.mean(drqn_clears):.0f}")

    ax.set_xlabel("Training steps to clear level 1")
    ax.set_ylabel("Density")
    ax.set_title("3 — First level clear distribution (all trials)")
    ax.legend(fontsize=8)


# ── plot 4 — level progression ────────────────────────────────────────────────

def plot_level_progression(client, dqn_runs, drqn_runs, ax):
    """
    Which level is the agent on at each episode, averaged across all trials.
    Answers: does DRQN advance through levels faster?
    """
    def avg_level_curve(runs):
        all_curves = []
        for run in runs:
            _, values = get_metric_curve(client, run.info.run_id, "level")
            if values:
                all_curves.append(values)
        if not all_curves:
            return [], []
        # pad shorter curves with their last value
        max_len = max(len(c) for c in all_curves)
        padded  = [c + [c[-1]] * (max_len - len(c)) for c in all_curves]
        mean    = np.mean(padded, axis=0)
        std     = np.std(padded,  axis=0)
        return mean, std

    dqn_mean,  dqn_std  = avg_level_curve(dqn_runs)
    drqn_mean, drqn_std = avg_level_curve(drqn_runs)

    if len(dqn_mean):
        eps = range(len(dqn_mean))
        ax.plot(eps, dqn_mean, label="DQN",  color="#DD8452", linewidth=2)
        ax.fill_between(eps,
                        dqn_mean - dqn_std,
                        dqn_mean + dqn_std,
                        color="#DD8452", alpha=0.15)

    if len(drqn_mean):
        eps = range(len(drqn_mean))
        ax.plot(eps, drqn_mean, label="DRQN", color="#55A868", linewidth=2)
        ax.fill_between(eps,
                        drqn_mean - drqn_std,
                        drqn_mean + drqn_std,
                        color="#55A868", alpha=0.15)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Level")
    ax.set_title("4 — Level progression (mean ± std, all trials)")
    ax.set_yticks(range(1, 7))
    ax.legend(fontsize=8)


# ── plot 5 — puzzles completed over timesteps ─────────────────────────────────

def plot_puzzles_over_timesteps(dqn_runs, drqn_runs, ax, human_data=None):
    """
    Cumulative levels (puzzles) cleared over timesteps, averaged across all trials.
    X-axis is actual environment timesteps — directly shows sample efficiency.
    """
    total_steps = TRAIN_CONFIG["total_timesteps"]
    sample_points = np.arange(0, total_steps + 1, 500)

    def cumulative_levels(runs):
        all_curves = []
        for run in runs:
            clear_steps = {}
            for level in range(1, 7):
                step = run.data.metrics.get(f"level_{level}_clear_step")
                if step is not None and step > 0:
                    clear_steps[level] = int(step)
            if not clear_steps:
                continue
            curve = np.array([
                sum(1 for s in clear_steps.values() if s <= t)
                for t in sample_points
            ])
            all_curves.append(curve)
        if not all_curves:
            return None, None
        arr = np.array(all_curves)
        return np.mean(arr, axis=0), np.std(arr, axis=0)

    dqn_mean,  dqn_std  = cumulative_levels(dqn_runs)
    drqn_mean, drqn_std = cumulative_levels(drqn_runs)

    if dqn_mean is not None:
        ax.plot(sample_points, dqn_mean, label="DQN",  color="#DD8452", linewidth=2)
        ax.fill_between(sample_points,
                        dqn_mean - dqn_std, dqn_mean + dqn_std,
                        color="#DD8452", alpha=0.15)

    if drqn_mean is not None:
        ax.plot(sample_points, drqn_mean, label="DRQN", color="#55A868", linewidth=2)
        ax.fill_between(sample_points,
                        drqn_mean - drqn_std, drqn_mean + drqn_std,
                        color="#55A868", alpha=0.15)

    # human reference line — median trials to complete each puzzle
    if human_data is not None:
        _add_human_puzzle_reference(human_data, ax, total_steps)

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Puzzles (levels) completed")
    ax.set_title("5 — Puzzles completed over timesteps (mean ± std, all trials)")
    ax.set_yticks(range(0, 7))
    ax.legend(fontsize=8)


def _add_human_puzzle_reference(human_data, ax, total_steps):
    """
    Overlays human performance on the puzzles-over-timesteps plot.
    Tries common column names; skips silently if columns aren't found.
    """
    cols = human_data.columns.tolist()

    # find a level/puzzle column
    level_col = next((c for c in cols if c in ("level", "wave", "puzzle", "curr_wave")), None)
    # find a timestep/trial column
    step_col  = next((c for c in cols if c in ("step", "timestep", "trial", "t")), None)

    if level_col is None or step_col is None:
        print(f"Human data: couldn't find level/step columns in {cols}")
        return

    # median level reached by humans at each step
    grouped = human_data.groupby(step_col)[level_col].median()
    ax.plot(grouped.index, grouped.values,
            label="Human (median)", color="#4C72B0",
            linewidth=2, linestyle="--")


# ── plot 6 — hyperparameter importance ───────────────────────────────────────

def plot_hyperparameter_importance(ax):
    """
    Optuna FAnova importance scores for each hyperparameter.
    Shows which knobs actually drove performance differences.
    """
    dqn_imp  = {}
    drqn_imp = {}

    try:
        dqn_study = optuna.load_study(
            study_name=DQN_STUDY_NAME,
            storage=f"sqlite:///{OPTUNA_DIR}/{DQN_STUDY_NAME}.db"
        )
        dqn_imp = optuna.importance.get_param_importances(dqn_study)
    except Exception as e:
        print(f"Could not load DQN study: {e}")

    try:
        drqn_study = optuna.load_study(
            study_name=DRQN_STUDY_NAME,
            storage=f"sqlite:///{OPTUNA_DIR}/{DRQN_STUDY_NAME}.db"
        )
        drqn_imp = optuna.importance.get_param_importances(drqn_study)
    except Exception as e:
        print(f"Could not load DRQN study: {e}")

    if not dqn_imp and not drqn_imp:
        ax.text(0.5, 0.5, "No Optuna studies found.\nRun hyperparameter search first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("6 — Hyperparameter importance (Optuna)")
        return

    all_params = sorted(set(list(dqn_imp.keys()) + list(drqn_imp.keys())))
    x     = np.arange(len(all_params))
    width = 0.35

    dqn_vals  = [dqn_imp.get(p, 0)  for p in all_params]
    drqn_vals = [drqn_imp.get(p, 0) for p in all_params]

    if dqn_imp:
        ax.bar(x - width/2, dqn_vals,  width, label="DQN",  color="#DD8452", alpha=0.85)
    if drqn_imp:
        ax.bar(x + width/2, drqn_vals, width, label="DRQN", color="#55A868", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_params, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Importance score")
    ax.set_title("6 — Hyperparameter importance (Optuna FAnova)")
    ax.legend(fontsize=8)


# ── summary table ─────────────────────────────────────────────────────────────

def print_summary_table(dqn_runs, drqn_runs):
    """
    Prints a clean comparison table of key metrics across all trials.
    This is what goes in your paper's results section.
    """
    metrics = [
        ("final_reward",            "Final reward"),
        ("first_level_clear_step",  "Steps to clear L1"),
        ("reward_early",            "Reward (early)"),
        ("reward_mid",              "Reward (mid)"),
        ("reward_late",             "Reward (late)"),
        ("final_rolling_reward",    "Final rolling reward"),
        ("mean_reward",             "Mean reward"),
    ]

    print("\n" + "=" * 65)
    print(f"{'Metric':<30} {'DQN':>15} {'DRQN':>15}")
    print("=" * 65)

    for metric_key, metric_label in metrics:
        dqn_vals  = get_all_scalars(dqn_runs,  metric_key)
        drqn_vals = get_all_scalars(drqn_runs, metric_key)

        # filter -1 sentinel values
        dqn_vals  = [v for v in dqn_vals  if v >= 0]
        drqn_vals = [v for v in drqn_vals if v >= 0]

        if not dqn_vals and not drqn_vals:
            continue

        dqn_str  = f"{np.mean(dqn_vals):.2f} ± {np.std(dqn_vals):.2f}"  if dqn_vals  else "n/a"
        drqn_str = f"{np.mean(drqn_vals):.2f} ± {np.std(drqn_vals):.2f}" if drqn_vals else "n/a"

        if dqn_vals and drqn_vals:
            if "step" in metric_key:
                winner = "←" if np.mean(dqn_vals) < np.mean(drqn_vals) else "→"
            else:
                winner = "←" if np.mean(dqn_vals) > np.mean(drqn_vals) else "→"
        else:
            winner = ""

        print(f"{metric_label:<30} {dqn_str:>15} {drqn_str:>15}  {winner}")

    print("=" * 65)
    print("← DQN wins   → DRQN wins\n")


# ── main ───────────────────────────────────────────────────────────────────────

def run_analysis(save=True, include_human=False):
    client    = get_client()
    dqn_runs  = get_runs(client, DQN_EXPERIMENT_NAME)
    drqn_runs = get_runs(client, DRQN_EXPERIMENT_NAME)

    if not dqn_runs:
        print("No DQN runs found — run main.py --agent dqn first")
        return
    if not drqn_runs:
        print("No DRQN runs found — run main.py --agent drqn first")
        return

    human_data = None
    if include_human:
        try:
            human_data = load_human_data()
        except Exception as e:
            print(f"Could not load human data: {e}")

    print_summary_table(dqn_runs, drqn_runs)

    # 3x2 grid
    fig = plt.figure(figsize=(14, 15))
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    print("\nBuilding plots...")
    plot_learning_curves(client, dqn_runs, drqn_runs, ax1, human_data=human_data)
    plot_phase_breakdown(dqn_runs, drqn_runs, ax2)
    plot_first_clear_distribution(dqn_runs, drqn_runs, ax3)
    plot_level_progression(client, dqn_runs, drqn_runs, ax4)
    plot_puzzles_over_timesteps(dqn_runs, drqn_runs, ax5, human_data=human_data)
    plot_hyperparameter_importance(ax6)

    fig.suptitle(
        "DQN vs DRQN — Learning speed on Hexxed",
        fontsize=14, y=1.01
    )

    if save:
        os.makedirs("analysis", exist_ok=True)
        plt.savefig("analysis/comparison.png", dpi=150, bbox_inches="tight")
        print("Saved → analysis/comparison.png")

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-save",      action="store_true")
    parser.add_argument("--human",        action="store_true", help="include human data overlay")
    args = parser.parse_args()
    run_analysis(save=not args.no_save, include_human=args.human)
