"""
Train a DQN agent on SimpleBettingEnv using processed kaggle data.

Usage (from repo root):
    python scripts/train_simple_dqn.py
"""

import os
import sys

import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs.simple_env import SimpleBettingEnv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = "data/processed/kaggle"
TRAIN_SEASONS = list(range(2008, 2021))  # 2008 through 2020

INITIAL_BANKROLL = 500.0
BET_SIZE = 1.0

TOTAL_TIMESTEPS = 100_000
MODEL_SAVE_PATH = "experiments/simple_dqn_model"


# ---------------------------------------------------------------------------
# Multi-season wrapper
# ---------------------------------------------------------------------------
class MultiSeasonWrapper(SimpleBettingEnv):
    """Cycles through a list of season DataFrames, one per episode."""

    def __init__(self, season_dfs, **kwargs):
        self.season_dfs = season_dfs
        self.season_idx = 0
        super().__init__(games_df=season_dfs[0], **kwargs)

    def reset(self, seed=None, options=None):
        self.games = self.season_dfs[self.season_idx].reset_index(drop=True)
        self.season_idx = (self.season_idx + 1) % len(self.season_dfs)
        return super().reset(seed=seed, options=options)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_season(season: int) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"season_{season}.csv")
    df = pd.read_csv(path)
    print(f"  Loaded season {season}: {len(df)} games")
    return df


def load_seasons(season_list: list) -> list:
    return [load_season(s) for s in season_list]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{'=' * 60}")
    print("  Loading training data")
    print("=" * 60)
    train_dfs = load_seasons(TRAIN_SEASONS)

    print(f"\n{'=' * 60}")
    print("  Training DQN")
    print("=" * 60)

    env = DummyVecEnv(
        [
            lambda dfs=train_dfs: MultiSeasonWrapper(
                dfs, initial_bankroll=INITIAL_BANKROLL, bet_size=BET_SIZE
            )
        ]
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_starts=1000,
        buffer_size=15000,
        batch_size=32,
        gamma=0.99,
        tau=0.01,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}.zip")


if __name__ == "__main__":
    main()
