"""
ELO-Enhanced Betting Environment
Uses pre-calculated ELO ratings as features
"""

import gymnasium as gym
import numpy as np
import pandas as pd


FATIGUE_COLS = [
    "home_days_since_last", "away_days_since_last",
    "home_is_back_to_back", "away_is_back_to_back",
    "home_win_pct_last5",   "away_win_pct_last5",
    "home_point_diff_last5", "away_point_diff_last5",
]


class EloBettingEnv(gym.Env):
    """
    Betting environment with ELO features and optional fatigue features.

    Base observation (7 features):
        - bankroll
        - home_ml / away_ml
        - elo_market_residual   (elo_prob_home - ml_prob_home)
        - elo_prob_home
        - elo_ev_home / elo_ev_away  (ELO-implied EV per $1 bet)

    Extended observation (+8 fatigue features, auto-enabled when columns present):
        - home/away_days_since_last
        - home/away_is_back_to_back
        - home/away_win_pct_last5
        - home/away_point_diff_last5

    Actions: 0=Skip, 1=Bet Home, 2=Bet Away
    """

    def __init__(
        self,
        games_df: pd.DataFrame,
        initial_bankroll: float = 500.0,
        bet_size: float = 1.0,
        edge_threshold: float = 0.0,
        no_edge_penalty: float = 0.0,
    ):
        """
        Args:
            games_df: DataFrame with ELO features and betting odds
                     Must include: elo_prob_home, home_ml, away_ml,
                                   score_home, score_away
            initial_bankroll: Starting bankroll
            bet_size: Fixed bet size (for now)
            edge_threshold: |elo_market_residual| below which a bet is penalised
            no_edge_penalty: Extra cost subtracted from shaped reward when betting
                            without edge.  Does not affect bankroll — shaping only.
        """
        super().__init__()

        self.games = games_df.reset_index(drop=True)
        self.initial_bankroll = initial_bankroll
        self.bet_size = bet_size
        self.edge_threshold = edge_threshold
        self.no_edge_penalty = no_edge_penalty

        # Verify required columns exist
        required_cols = [
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

        # Auto-detect fatigue features; obs is 7 (base) or 15 (base + fatigue)
        self.has_fatigue = all(col in self.games.columns for col in FATIGUE_COLS)
        obs_dim = 15 if self.has_fatigue else 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
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

        # Payout multipliers (American odds → decimal win amount per $1 bet)
        self.games["payout_home"] = self.games["home_ml"].apply(self._calculate_payout)
        self.games["payout_away"] = self.games["away_ml"].apply(self._calculate_payout)

        # ELO-implied expected value for each side
        # EV = P(win)*payout - P(lose)  (per $1 wagered)
        # Positive EV = profitable bet under the ELO model
        elo_h = self.games["elo_prob_home"]
        elo_a = 1 - elo_h
        self.games["elo_ev_home"] = elo_h * self.games["payout_home"] - elo_a
        self.games["elo_ev_away"] = elo_a * self.games["payout_away"] - elo_h

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
        obs_dim = 15 if self.has_fatigue else 7
        if self.current_game_idx >= len(self.games):
            return np.zeros(obs_dim, dtype=np.float32)

        game = self.games.iloc[self.current_game_idx]

        obs = [
            self.bankroll,
            game["home_ml"],
            game["away_ml"],
            game["elo_market_residual"],
            game["elo_prob_home"],
            game["elo_ev_home"],
            game["elo_ev_away"],
        ]

        if self.has_fatigue:
            obs.extend([
                game["home_days_since_last"],
                game["away_days_since_last"],
                game["home_is_back_to_back"],
                game["away_is_back_to_back"],
                game["home_win_pct_last5"],
                game["away_win_pct_last5"],
                game["home_point_diff_last5"],
                game["away_point_diff_last5"],
            ])

        return np.array(obs, dtype=np.float32)

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

        # Bankroll tracks actual outcomes only
        self.bankroll += reward

        # Reward shaping: penalise betting when edge is below threshold.
        # Does not touch bankroll — ROI / stats remain ground-truth.
        shaped_reward = reward
        if action != 0 and self.no_edge_penalty > 0:
            if abs(game["elo_market_residual"]) < self.edge_threshold:
                shaped_reward -= self.no_edge_penalty

        # Record bet (actual reward, not shaped)
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

        return self._get_obs(), shaped_reward, terminated, truncated, {}

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
