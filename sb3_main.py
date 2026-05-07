# sb3_main.py — entry point for the SB3 DQN sanity check
# Run: python sb3_main.py
from training.sb3_optuna_search import run_sb3_study

if __name__ == "__main__":
    best = run_sb3_study()
    print("\nDone. Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")
