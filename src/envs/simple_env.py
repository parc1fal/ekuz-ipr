"""
simple betting env with three features
"""

import gymnasium as gym
import numpy as np
import pandas as pd


class SimpleBettingEnv(gym.Env):
    """
    betting environment with three basic features

    state features:
        - bankroll
        - home_ml: home team moneyline odds
        - away_ml: away team moneyline odds

    actions: 0=skip, 1=bet home, 2=bet away
    """

    def __init__(
        self,
        games_df: pd.DataFrame,
        initial_bankroll: float = 500.0,
        bet_size: float = 1.0,
    ):
        super().__init__()
        self.games = games_df.reset_index(drop=True)
        self.initial_bankroll = initial_bankroll
        self.bet_size = bet_size

        required_cols = [
            "home_ml",
            "away_ml",
            "home_won",
        ]

        missing = [col for col in required_cols if col not in self.games]

        if missing:
            raise ValueError(f"Missing columns in games dataset: {missing}")

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_game_idx = 0
        self.bankroll = self.initial_bankroll
        self.bet_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        "get current observation state"

        if self.current_game_idx >= len(self.games):
            return np.zeros(3, dtype=np.float32)

        game = self.games.iloc[self.current_game_idx]

        return np.array(
            [
                self.bankroll,
                game["home_ml"],
                game["away_ml"],
            ],
            dtype=np.float32,
        )

    def step(self, action):
        "execute one betting decision"

        game = self.games.iloc[self.current_game_idx]
        home_won = game["home_won"]

        reward = 0.0

        # actions: 0 = skip, 1 = bet home, 2 = bet away
        if action == 0:
            reward = 0.0
        elif action == 1:
            if home_won:
                payout = self._calculate_payout(game["home_ml"])
                reward = self.bet_size * payout
            else:
                reward = -self.bet_size
        else:
            if home_won:
                reward = -self.bet_size
            else:
                payout = self._calculate_payout(game["away_ml"])
                reward = self.bet_size * payout

        # update bankroll
        self.bankroll += reward

        # record bet
        self.bet_history.append(
            {
                "game_idx": self.current_game_idx,
                "date": game.get("date", None),
                "home_team": game.get("home_team", None),
                "away_team": game.get("away_team", None),
                "action": action,
                "reward": reward,
                "bankroll": self.bankroll,
            }
        )

        # advance to next game
        self.current_game_idx += 1

        # check termination
        terminated = self.current_game_idx >= len(self.games) or self.bankroll <= 0
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def _calculate_payout(self, ml_odds):
        """
        Convert American moneyline odds to profit multiplier per unit bet.

            -110 odds: risk $110 to win $100 → profit $0.909 per $1 bet
            +150 odds: risk $100 to win $150 → profit $1.50 per $1 bet
        """
        if ml_odds >= 0:
            return ml_odds / 100
        else:
            return 100 / abs(ml_odds)

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
