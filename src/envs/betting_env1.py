"""
SimpleBettingEnv: Minimal environment to start testing
Features: [bankroll, home_elo, away_elo, home_ml, away_ml]
"""

import gymnasium as gym
import numpy as np
import pandas as pd


class SimpleBettingEnv(gym.Env):
    """
    Simple betting environment with fixed $1 bets

    State: [bankroll, home_elo, away_elo, home_ml, away_ml]
    Actions: 0=Skip, 1=Bet Home, 2=Bet Away
    """

    def __init__(self, games_df, initial_bankroll=500):
        super().__init__()

        self.games = games_df.reset_index(drop=True)
        self.initial_bankroll = initial_bankroll

        self.action_space = gym.spaces.Discrete(3)

        # State space: 5 features
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
        )

        # Add placeholder ELO ratings (all teams = 1500 for now)
        self.games["home_elo"] = 1500
        self.games["away_elo"] = 1500

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_game_idx = 0
        self.bankroll = self.initial_bankroll
        self.bet_history = []
        return self._get_obs(), {}

    def _get_obs(self):
        """Return current observation"""
        if self.current_game_idx >= len(self.games):
            return np.zeros(5, dtype=np.float32)

        game = self.games.iloc[self.current_game_idx]
        return np.array(
            [
                self.bankroll,
                game["home_elo"],
                game["away_elo"],
                game["home_ml"],
                game["away_ml"],
            ],
            dtype=np.float32,
        )

    def _calculate_payout(self, odds):
        """
        Convert American odds to payout multiplier

        Examples:
            -110 odds: bet $1, win $0.91 (need to risk $110 to win $100)
            +150 odds: bet $1, win $1.50 (win $150 on $100 bet)
        """
        if odds < 0:
            # Favorite: need to bet |odds| to win $100
            return 100 / abs(odds)
        else:
            # Underdog: win odds amount on $100 bet
            return odds / 100

    def step(self, action):
        """Execute one step"""
        game = self.games.iloc[self.current_game_idx]

        reward = 0
        bet_size = 1

        if action == 1:  # Bet on home team
            if game["home_won"]:
                payout = self._calculate_payout(game["home_ml"])
                reward = bet_size * payout
            else:
                reward = -bet_size
        elif action == 2:  # Bet on away team
            if not game["home_won"]:
                payout = self._calculate_payout(game["away_ml"])
                reward = bet_size * payout
            else:
                reward = -bet_size

        self.bankroll += reward
        self.bet_history.append(
            {
                "game_idx": self.current_game_idx,
                "action": action,
                "reward": reward,
                "bankroll": self.bankroll,
            }
        )

        self.current_game_idx += 1

        terminated = (self.current_game_idx >= len(self.games)) or (self.bankroll <= 0)
        truncated = False

        return self._get_obs(), reward, terminated, truncated, {}

    def get_roi(self):
        """Calculate return on investment"""
        return (self.bankroll - self.initial_bankroll) / self.initial_bankroll * 100
