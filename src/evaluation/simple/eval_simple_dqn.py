"""
Evaluate a trained DQN agent on test seasons.

Usage (from repo root):
    python src/evaluation/simple/eval_simple_dqn.py

Loads the saved model from experiments/ and runs it against
random and always-skip baselines on each test season.
"""

import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

# repo root is 3 directories up from this file
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.envs.simple_env import SimpleBettingEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/processed/kaggle"
TEST_SEASONS = [2021, 2022, 2023]

INITIAL_BANKROLL = 500.0
BET_SIZE = 1.0

MODEL_SAVE_PATH = "experiments/simple_dqn_model"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_season(season: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"season_{season}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded season {season}: {len(df)} games")
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_policy(model, season_df, initial_bankroll, bet_size):
    """Run trained model deterministically on one season."""
    env = SimpleBettingEnv(
        games_df=season_df, initial_bankroll=initial_bankroll, bet_size=bet_size
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return env.get_bet_statistics()


def evaluate_baseline(policy_fn, season_df, initial_bankroll, bet_size):
    """Run a simple policy function (obs -> action) on one season."""
    env = SimpleBettingEnv(
        games_df=season_df, initial_bankroll=initial_bankroll, bet_size=bet_size
    )
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.get_bet_statistics()


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def print_stats(label, stats):
    if not stats:
        print(f"  {label}: no bets recorded")
        return
    print(
        f"  {label}: "
        f"ROI={stats.get('roi', 0):.2f}%  "
        f"bets={stats.get('total_bets', 0)}  "
        f"win%={stats.get('win_rate', 0):.1f}  "
        f"profit={stats.get('total_profit', 0):.2f}  "
        f"bankroll={stats.get('final_bankroll', 0):.2f}"
    )


def print_separator(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print_separator("Loading model")
    model = DQN.load(MODEL_SAVE_PATH)
    print(f"  Loaded {MODEL_SAVE_PATH}.zip")

    print_separator("Loading test data")
    test_dfs = {s: load_season(s) for s in TEST_SEASONS}

    print_separator("Evaluation on test seasons")

    rng = np.random.default_rng(42)
    all_roi = {"dqn": [], "random": [], "skip": []}

    for season in TEST_SEASONS:
        print(f"\n--- Season {season} ---")
        df = test_dfs[season]

        dqn_stats = evaluate_policy(model, df, INITIAL_BANKROLL, BET_SIZE)
        random_stats = evaluate_baseline(
            lambda obs, rng=rng: rng.integers(3), df, INITIAL_BANKROLL, BET_SIZE
        )
        skip_stats = evaluate_baseline(lambda obs: 0, df, INITIAL_BANKROLL, BET_SIZE)

        print_stats("DQN   ", dqn_stats)
        print_stats("Random", random_stats)
        print_stats("Skip  ", skip_stats)

        all_roi["dqn"].append(dqn_stats.get("roi", 0))
        all_roi["random"].append(random_stats.get("roi", 0))
        all_roi["skip"].append(skip_stats.get("roi", 0))

    # ----- summary ---------------------------------------------------------
    print_separator("Summary — Average ROI across test seasons")
    print(f"  DQN:    {np.mean(all_roi['dqn']):+.2f}%")
    print(f"  Random: {np.mean(all_roi['random']):+.2f}%")
    print(f"  Skip:   {np.mean(all_roi['skip']):+.2f}%")
    print()


if __name__ == "__main__":
    main()
