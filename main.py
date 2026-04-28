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
    args = parser.parse_args()
    best_params = run_study(args.agent)
    print(f"\nBest params found for {args.agent}:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()