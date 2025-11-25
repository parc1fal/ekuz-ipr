"""
ELO Rating Calculator for NBA
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class NBAEloCalculator:

    def __init__(
        self,
        k_factor: float = 20.0,
        home_advantage: float = 100.0,
        initial_elo: float = 1500.0,
        season_carryover: float = 0.75,
    ):
        """
        Args:
            k_factor: How much ratings change per game (FiveThirtyEight uses 20)
            home_advantage: Rating boost for home team
            initial_elo: Starting rating for new teams
            season_carryover: How much of rating carries over between seasons
                             (0.75 = regress 25% toward mean of 1500)
        """
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.initial_elo = initial_elo
        self.season_carryover = season_carryover

        # Track current ratings for all teams
        self.ratings: Dict[str, float] = {}

    def _get_team_rating(self, team: str) -> float:
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
        return self.ratings[team]

    def _expected_score(self, rating_a: float, rating_b: float) -> float:

        # Based on FiveThirtyEight formula
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _mov_multiplier(self, score_diff: int, elo_diff: float) -> float:
        """
        Margin of victory multiplier
        Larger wins = bigger rating changes
        But favorites beating underdogs by a lot doesn't count as much

        Based on FiveThirtyEight's formula
        """
        mov = abs(score_diff)

        # Base multiplier increases with margin
        multiplier = np.log(max(mov, 1) + 1) * (
            2.2 / (1 if elo_diff < 0 else (elo_diff * 0.001 + 1))
        )

        return multiplier

    def update_ratings(
        self, home_team: str, away_team: str, home_score: int, away_score: int
    ) -> Tuple[float, float, float, float]:

        home_elo_pre = self._get_team_rating(home_team)
        away_elo_pre = self._get_team_rating(away_team)

        home_elo_adjusted = home_elo_pre + self.home_advantage

        # Calculate expected win probability
        expected_home_win = self._expected_score(home_elo_adjusted, away_elo_pre)

        actual_home_win = 1.0 if home_score > away_score else 0.0

        # Margin of victory multiplier
        score_diff = home_score - away_score
        elo_diff = home_elo_pre - away_elo_pre
        mov_mult = self._mov_multiplier(score_diff, elo_diff)

        # Rating changes
        home_change = self.k_factor * mov_mult * (actual_home_win - expected_home_win)
        away_change = -home_change

        # Update ratings
        self.ratings[home_team] = home_elo_pre + home_change
        self.ratings[away_team] = away_elo_pre + away_change

        return (
            home_elo_pre,
            away_elo_pre,
            self.ratings[home_team],
            self.ratings[away_team],
        )

    def season_adjustment(self):
        for team in self.ratings:
            # Regress toward 1500
            self.ratings[team] = self.ratings[
                team
            ] * self.season_carryover + self.initial_elo * (1 - self.season_carryover)

    def get_ratings_snapshot(self) -> Dict[str, float]:
        return self.ratings.copy()


def calculate_elo_features(
    games_df: pd.DataFrame,
    k_factor: float = 20.0,
    home_advantage: float = 100.0,
    initial_elo: float = 1500.0,
    season_carryover: float = 0.75,
) -> pd.DataFrame:
    """
    Calculate ELO ratings for all games chronologically

    CRITICAL: Games must be sorted by date to avoid look-ahead bias

    Args:
        games_df: DataFrame with columns [date, season, home_team, away_team,
                                          score_home, score_away]
        k_factor: ELO k-factor (default 20)
        home_advantage: Home court advantage in ELO points (default 100)
        initial_elo: Starting rating (default 1500)
        season_carryover: Rating carryover between seasons (default 0.75)

    Returns:
        DataFrame with added columns:
            - home_elo_before: Home team ELO before game
            - away_elo_before: Away team ELO before game
            - home_elo_after: Home team ELO after game
            - away_elo_after: Away team ELO after game
            - elo_diff: home_elo_before - away_elo_before
            - elo_prob_home: Probability home wins based on ELO
    """
    # Ensure chronological order
    df = games_df.sort_values("date").copy()

    # Initialize calculator
    calculator = NBAEloCalculator(
        k_factor=k_factor,
        home_advantage=home_advantage,
        initial_elo=initial_elo,
        season_carryover=season_carryover,
    )

    # Track previous season for season adjustments
    prev_season = None

    # Lists to store ELO values
    home_elo_before_list = []
    away_elo_before_list = []
    home_elo_after_list = []
    away_elo_after_list = []

    # Process games chronologically
    for idx, row in df.iterrows():
        # Apply season adjustment if new season
        if prev_season is not None and row["season"] != prev_season:
            calculator.season_adjustment()
        prev_season = row["season"]

        # Update ratings and store pre/post values
        home_pre, away_pre, home_post, away_post = calculator.update_ratings(
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_score=row["score_home"],
            away_score=row["score_away"],
        )

        home_elo_before_list.append(home_pre)
        away_elo_before_list.append(away_pre)
        home_elo_after_list.append(home_post)
        away_elo_after_list.append(away_post)

    # Add ELO features to dataframe
    df["home_elo_before"] = home_elo_before_list
    df["away_elo_before"] = away_elo_before_list
    df["home_elo_after"] = home_elo_after_list
    df["away_elo_after"] = away_elo_after_list

    # Derived features
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]

    # Probability home wins (with home advantage applied)
    df["elo_prob_home"] = 1 / (
        1
        + 10
        ** ((df["away_elo_before"] - (df["home_elo_before"] + home_advantage)) / 400)
    )

    return df


if __name__ == "__main__":
    # Example usage
    print("ELO Calculator Module")
    print("=" * 50)
    print("\nUsage example:")
    print(
        """
    from elo_calculator import calculate_elo_features
    import pandas as pd
    
    # Load games
    games = pd.read_csv('data/processed/kaggle/season_2021.csv')
    
    # Calculate ELO features
    games_with_elo = calculate_elo_features(games)
    
    # Save
    games_with_elo.to_csv('data/features/season_2021_elo.csv', index=False)
    """
    )
