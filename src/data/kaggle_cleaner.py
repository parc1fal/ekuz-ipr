import pandas as pd


def load_and_split_by_season(raw_csv_path, season=None, regular_season=True):
    df = pd.read_csv(raw_csv_path)

    df["date"] = pd.to_datetime(df["date"])

    if regular_season:
        df = df[df["regular"] == True].copy()

    if season is not None:
        df = df[df["season"] == season].copy()

    df["home_won"] = df["score_home"] > df["score_away"]

    df = df.rename(
        columns={
            "away": "away_team",
            "home": "home_team",
            "moneyline_away": "away_ml",
            "moneyline_home": "home_ml",
        }
    )

    df = df.sort_values("date").reset_index(drop=True)
    df = clean_invalid_games(df)
    return df


def clean_invalid_games(df):
    original_length = len(df)
    df = df.dropna(subset=["away_ml", "home_ml"])
    df = df[(df["away_ml"] != 0) & (df["home_ml"] != 0)]

    removed = original_length - len(df)
    if removed > 0:
        print(f"Removed {removed} invalid games due to missing or zero moneyline odds.")

    return df
