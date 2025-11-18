import pandas as pd
import os
from src.data.kaggle_cleaner import load_and_split_by_season


def main():
    raw_path = "data/raw/kaggle/nba_2008-2025.csv"
    output_dir = "data/processed/kaggle/"
    os.makedirs(output_dir, exist_ok=True)

    df_all = load_and_split_by_season(raw_path, season=None, regular_season=True)

    # Save by season
    for season in sorted(df_all["season"].unique()):
        df_season = df_all[df_all["season"] == season]
        output_path = f"{output_dir}season_{season}.csv"
        df_season.to_csv(output_path, index=False)
        print(f"Saved {len(df_season):4d} games → season_{season}.csv")

    # Also save full dataset
    df_all.to_csv(f"{output_dir}all_seasons.csv", index=False)
    print(f"\nSaved {len(df_all)} total games → all_seasons.csv")


if __name__ == "__main__":
    main()
