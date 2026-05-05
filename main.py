# main.py
import argparse
from training.optuna_search import run_study


def main():
    parser = argparse.ArgumentParser(description="Train DQN or DRQN on Hexxed")
    parser.add_argument(
        "--agent",
        choices=["dqn", "drqn"],
        required=True,
        help="Which agent to train"
    )
    parser.add_argument(
        "--study-name",
        default=None,
        help="Override the Optuna study name and MLflow experiment name "
             "(default: use names from config.py)"
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel Optuna trials (default: N_JOBS from config.py)"
    )
    args = parser.parse_args()
    best_params = run_study(args.agent, study_name=args.study_name, n_jobs=args.n_jobs)
    print(f"\nBest params found for {args.agent}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()