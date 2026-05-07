# eval.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import multiprocessing
import mlflow
from config import MLFLOW_TRACKING_URI
from training.train import train_dqn, train_drqn

DQN_BEST = {
    "lr":            0.002839219709446442,
    "gamma":         0.9079193236500384,
    "buffer_size":   12741,
    "batch_size":    91,
    "target_update": 417,
    "hidden_dim":    460,
}

DRQN_BEST = {
    "lr":            0.00046852819707594433,
    "gamma":         0.9325970031762794,
    "buffer_size":   24626,
    "batch_size":    32,
    "target_update": 115,
    "hidden_dim":    468,
    "seq_len":       4,
}


def _run_trial(args):
    agent, run_id = args
    params   = DQN_BEST  if agent == "dqn"  else DRQN_BEST
    train_fn = train_dqn if agent == "dqn"  else train_drqn

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(f"{agent}_eval")
    mlflow.start_run(run_name=f"eval_{run_id}")
    mlflow.log_params(params)
    mlflow.log_param("eval_run_id", run_id)

    try:
        score, tracker = train_fn(**params)
        mlflow.log_metric("learning_speed_score", score)
        summary = tracker.summary()
        for level, step in summary["level_clear_steps"].items():
            mlflow.log_metric(f"level_{level}_clear_step", step)
        mlflow.log_metric("levels_cleared", len(summary["level_clear_steps"]))
        mlflow.log_metric("mean_reward",    summary["mean_reward"])
        return score, summary
    except Exception as e:
        print(f"Run {run_id} failed: {e}")
        return 0.0, {}
    finally:
        mlflow.end_run()


def main():
    parser = argparse.ArgumentParser(description="Evaluate optimal DQN or DRQN hyperparameters")
    parser.add_argument("--agent",  choices=["dqn", "drqn"], required=True)
    parser.add_argument("--n-runs", type=int, default=50)
    parser.add_argument("--n-jobs", type=int, default=20)
    args = parser.parse_args()

    print(f"Running {args.n_runs} {args.agent.upper()} eval runs, {args.n_jobs} parallel")

    tasks = [(args.agent, i) for i in range(args.n_runs)]
    with multiprocessing.Pool(processes=args.n_jobs) as pool:
        results = pool.map(_run_trial, tasks)

    scores        = [r[0] for r in results]
    non_zero      = sum(1 for s in scores if s > 0)
    levels_counts = [len(r[1].get("level_clear_steps", {})) for r in results]

    print(f"\n{'='*50}")
    print(f"{args.agent.upper()} Eval — {args.n_runs} runs")
    print(f"  Mean score:      {sum(scores)/len(scores):.1f}")
    print(f"  Max score:       {max(scores):.1f}")
    print(f"  Min score:       {min(scores):.1f}")
    print(f"  Runs clearing ≥1 level: {non_zero}/{args.n_runs}")
    print(f"  Avg levels cleared:     {sum(levels_counts)/len(levels_counts):.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
