"""
ELO-Enhanced Betting Environment
Uses pre-calculated ELO ratings as features
"""

import gymnasium as gym
import numpy as np
import pandas as pd


class EloBettingEnv(gym.Env):
    """
    Betting environment with ELO features

    State features:
        - bankroll: Current bankroll
        - home_elo_before: Home team ELO rating before game
        - away_elo_before: Away team ELO rating before game
        - elo_diff: home_elo - away_elo (positive = home favored by ELO)
        - elo_prob_home: ELO-implied probability home wins
        - home_ml: Home team money line odds
        - away_ml: Away team money line odds
        - ml_prob_home: Market-implied probability home wins
        - elo_market_residual: elo_prob_home - ml_prob_home (KEY FEATURE)

    Actions: 0=Skip, 1=Bet Home, 2=Bet Away
    """

    def __init__(
        self,
        games_df: pd.DataFrame,
        initial_bankroll: float = 500.0,
        bet_size: float = 1.0,
    ):
        """
        Args:
            games_df: DataFrame with ELO features and betting odds
                     Must include: home_elo_before, away_elo_before, elo_prob_home,
                                   home_ml, away_ml, score_home, score_away
            initial_bankroll: Starting bankroll
            bet_size: Fixed bet size (for now)
        """
        super().__init__()

        self.games = games_df.reset_index(drop=True)
        self.initial_bankroll = initial_bankroll
        self.bet_size = bet_size

        # Verify required columns exist
        required_cols = [
            "home_elo_before",
            "away_elo_before",
            "elo_prob_home",
            "home_ml",
            "away_ml",
            "score_home",
            "score_away",
        ]
        missing = [col for col in required_cols if col not in self.games.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Action space: 0=Skip, 1=Bet Home, 2=Bet Away
        self.action_space = gym.spaces.Discrete(3)

        # State space: 9 features
        # [bankroll, home_elo, away_elo, elo_diff, elo_prob_home,
        #  home_ml, away_ml, ml_prob_home, elo_market_residual]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

    def _calculate_market_features(self):
        """Calculate market-implied probabilities and ELO-market residual"""

        def ml_to_prob(ml_odds: float) -> float:
            """Convert American money line odds to implied probability"""
            if ml_odds < 0:
                # Favorite: prob = |odds| / (|odds| + 100)
                return abs(ml_odds) / (abs(ml_odds) + 100)
            else:
                # Underdog: prob = 100 / (odds + 100)
                return 100 / (ml_odds + 100)

        # Market-implied probabilities (with vig)
        self.games["ml_prob_home_raw"] = self.games["home_ml"].apply(ml_to_prob)
        self.games["ml_prob_away_raw"] = self.games["away_ml"].apply(ml_to_prob)

        # Remove vig (normalize so probabilities sum to 1)
        total_prob = self.games["ml_prob_home_raw"] + self.games["ml_prob_away_raw"]
        self.games["ml_prob_home"] = self.games["ml_prob_home_raw"] / total_prob
        self.games["ml_prob_away"] = self.games["ml_prob_away_raw"] / total_prob

        # KEY FEATURE: ELO-market residual
        # Positive = ELO thinks home is more likely to win than market does
        # This captures market inefficiency
        self.games["elo_market_residual"] = (
            self.games["elo_prob_home"] - self.games["ml_prob_home"]
        )

        # Also calculate ELO diff if not present
        if "elo_diff" not in self.games.columns:
            self.games["elo_diff"] = (
                self.games["home_elo_before"] - self.games["away_elo_before"]
            )

    def reset(self, seed=None, options=None):
        """Reset environment to start of season"""
        super().reset(seed=seed)
        self._calculate_market_features()
        self.current_game_idx = 0
        self.bankroll = self.initial_bankroll
        self.bet_history = []
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        """Get current observation (state)"""
        if self.current_game_idx >= len(self.games):
            return np.zeros(9, dtype=np.float32)

        game = self.games.iloc[self.current_game_idx]

        return np.array(
            [
                self.bankroll,
                game["home_elo_before"],
                game["away_elo_before"],
                game["elo_diff"],
                game["elo_prob_home"],
                game["home_ml"],
                game["away_ml"],
                game["ml_prob_home"],
                game["elo_market_residual"],
            ],
            dtype=np.float32,
        )

    def _calculate_payout(self, odds: float) -> float:
        """
        Convert American odds to payout multiplier

        Examples:
            -110 odds: bet $1, win $0.91 (need to risk $110 to win $100)
            +150 odds: bet $1, win $1.50 (win $150 on $100 bet)
        """
        if odds < 0:
            return 100 / abs(odds)
        else:
            return odds / 100

    def step(self, action: int):
        """Execute one step in the environment"""
        game = self.games.iloc[self.current_game_idx]

        reward = 0.0

        # Determine outcome
        home_won = game["score_home"] > game["score_away"]

        if action == 1:  # Bet on home team
            if home_won:
                payout = self._calculate_payout(game["home_ml"])
                reward = self.bet_size * payout
            else:
                reward = -self.bet_size

        elif action == 2:  # Bet on away team
            if not home_won:
                payout = self._calculate_payout(game["away_ml"])
                reward = self.bet_size * payout
            else:
                reward = -self.bet_size

        # Update bankroll
        self.bankroll += reward

        # Record bet
        self.bet_history.append(
            {
                "game_idx": self.current_game_idx,
                "date": game.get("date", None),
                "home_team": game.get("home_team", None),
                "away_team": game.get("away_team", None),
                "action": action,
                "reward": reward,
                "bankroll": self.bankroll,
                "elo_market_residual": game["elo_market_residual"],
            }
        )

        # Move to next game
        self.current_game_idx += 1

        # Check if episode is done
        terminated = self.current_game_idx >= len(self.games) or self.bankroll <= 0
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def get_roi(self) -> float:
        """Calculate return on investment"""
        return (self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100

    def get_bet_statistics(self) -> dict:
        """Calculate betting statistics"""
        if not self.bet_history:
            return {}

        bets_df = pd.DataFrame(self.bet_history)
        actual_bets = bets_df[bets_df["action"] != 0]

        if len(actual_bets) == 0:
            return {
                "total_bets": 0,
                "win_rate": 0.0,
                "roi": self.get_roi(),
                "final_bankroll": self.bankroll,
            }

        wins = actual_bets[actual_bets["reward"] > 0]

        return {
            "total_bets": len(actual_bets),
            "win_rate": len(wins) / len(actual_bets) * 100,
            "roi": self.get_roi(),
            "final_bankroll": self.bankroll,
            "avg_reward": actual_bets["reward"].mean(),
            "total_profit": actual_bets["reward"].sum(),
        }


if __name__ == "__main__":
    print("ELO Betting Environment")
    print("=" * 70)
    print("\nUsage example:")
    print(
        """
    import pandas as pd
    from envs.betting_env_elo import EloBettingEnv
    
    # Load data with ELO features
    games = pd.read_csv('data/features/season_2021_elo.csv')
    
    # Create environment
    env = EloBettingEnv(games, initial_bankroll=500)
    
    # Reset and get initial observation
    obs, info = env.reset()
    print(f"State shape: {obs.shape}")
    print(f"State features: {obs}")
    
    # Take actions
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
    # Get statistics
    stats = env.get_bet_statistics()
    print(f"Final ROI: {stats['roi']:.2f}%")
    print(f"Win rate: {stats['win_rate']:.2f}%")
    """
    )
