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
            ]
        )

    def step(self, action):
        "execute one betting decision"

        # get game
        game = self.games.iloc[self.current_game_idx]

        home_won = game["home_won"]

        # calculate rewards

        reward = 0.0

        # actions: 0 = skip, 1 = bet home, 2 = bet away
        if action == 0:
            reward = 0.0
        elif action == 1:
            if home_won:
                reward = self._calculate_payout(game["home_ml"])
            else:
                reward = -self.bet_size
        else:
            if home_won:
                reward = -self.bet_size
            else:
                reward = self._calculate_payout(game["away_ml"])

    def _calculate_payout(self, ml_odds, bet_size=1):

        if ml_odds >= 0:
            payout = (ml_odds / 100) * bet_size
        else:
            payout = (100 / abs(ml_odds)) * bet_size
