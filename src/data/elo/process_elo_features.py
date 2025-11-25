"""
Process Kaggle all_seasons data and add ELO features
"""

import os
from pathlib import Path
import pandas as pd

from elo_calculator import calculate_elo_features


def load_all_seasons(
    data_file: str = "data/processed/kaggle/all_seasons.csv",
) -> pd.DataFrame:
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"File not found: {data_file}")

    df = pd.read_csv(data_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def add_elo_features_and_save(
    data_file: str = "data/processed/kaggle/all_seasons.csv",
    output_dir: str = "data/features",
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_games = load_all_seasons(data_file)

    games_with_elo = calculate_elo_features(
        all_games, k_factor=k_factor, home_advantage=home_advantage
    )
    print("\nELO calculation complete!")

    combined_output = os.path.join(output_dir, "all_seasons_with_elo.csv")
    games_with_elo.to_csv(combined_output, index=False)
    print(f"  Saved: {combined_output}")

    # save by season for convenience
    for season in sorted(games_with_elo["season"].unique()):
        season_data = games_with_elo[games_with_elo["season"] == season]
        season_file = os.path.join(output_dir, f"season_{season}_elo.csv")
        season_data.to_csv(season_file, index=False)


if __name__ == "__main__":
    add_elo_features_and_save(
        data_file="data/processed/kaggle/all_seasons.csv",
        output_dir="data/features",
        k_factor=20.0,
        home_advantage=100.0,
    )
