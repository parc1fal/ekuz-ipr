"""
Variable bet-sizing environment.

Same 15-dim observation as EloBettingEnv.  Supports two action-space modes
selected by the *bet_tiers* constructor parameter:

Discrete  (DQN-compatible) — bet_tiers = [0.5, 1.0, 2.0, 5.0]
    Action space: Discrete(1 + 2 * len(bet_tiers))
        0           →  Skip
        1 .. N      →  Bet Home  @ bet_tiers[action - 1]
        N+1 .. 2N   →  Bet Away  @ bet_tiers[action - 1 - N]

Continuous (PPO-compatible) — bet_tiers = None
    Action space: Box([-max_bet, +max_bet])
        |action| < min_bet   →  Skip
        action  >=  min_bet  →  Bet Home,  wager = action
        action  <= -min_bet  →  Bet Away,  wager = |action|

In both modes wager is clamped to the current bankroll, and the reward-
shaping penalty scales linearly with wager.
"""

import gymnasium as gym
import numpy as np
import pandas as pd

from src.envs.betting_env_elo import EloBettingEnv


class ContinuousBettingEnv(EloBettingEnv):
    """EloBettingEnv with variable bet sizing (discrete tiers or continuous)."""

    def __init__(
        self,
        games_df: pd.DataFrame,
        initial_bankroll: float = 500.0,
        bet_size: float = 1.0,
        edge_threshold: float = 0.0,
        no_edge_penalty: float = 0.0,
        bet_tiers: list | None = None,
        max_bet_multiplier: float = 5.0,
        min_bet_multiplier: float = 0.25,
        risk_coeff: float = 0.0,
        ev_bonus: float = 0.0,
    ):
        """
        Args:
            bet_tiers: If provided, list of wager amounts → Discrete action space.
                       If None → Box action space (continuous).
            max_bet_multiplier: (continuous only) largest wager = bet_size * this
            min_bet_multiplier: (continuous only) dead-zone = bet_size * this
            risk_coeff: quadratic risk penalty coefficient.
                        shaped_reward -= risk_coeff * (wager / bet_size)²
                        Breaks the linear wager-scaling symmetry.
            ev_bonus:   EV-sizing bonus coefficient.
                        shaped_reward += ev_bonus * ev * wager  (only when ev > 0)
                        Teaches the agent to bet more on high-EV games.
        """
        super().__init__(
            games_df,
            initial_bankroll=initial_bankroll,
            bet_size=bet_size,
            edge_threshold=edge_threshold,
            no_edge_penalty=no_edge_penalty,
        )
        self.bet_tiers = bet_tiers
        self.risk_coeff = risk_coeff
        self.ev_bonus   = ev_bonus

        if bet_tiers is not None:
            self.n_tiers = len(bet_tiers)
            self.action_space = gym.spaces.Discrete(1 + 2 * self.n_tiers)
        else:
            self.max_bet = bet_size * max_bet_multiplier
            self.min_bet = bet_size * min_bet_multiplier
            self.action_space = gym.spaces.Box(
                low=np.array([-self.max_bet], dtype=np.float32),
                high=np.array([self.max_bet], dtype=np.float32),
            )

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------
    def _parse_action(self, action) -> tuple[int, float]:
        """Map raw action → (side, wager).  side: 0=skip, 1=home, 2=away."""
        if self.bet_tiers is not None:
            a = int(action)
            if a == 0:
                return 0, 0.0
            elif a <= self.n_tiers:
                return 1, min(self.bet_tiers[a - 1], self.bankroll)
            else:
                return 2, min(self.bet_tiers[a - 1 - self.n_tiers], self.bankroll)
        else:
            raw = float(action[0]) if hasattr(action, "__len__") else float(action)
            if abs(raw) < self.min_bet:
                return 0, 0.0
            elif raw >= self.min_bet:
                return 1, min(raw, self.bankroll)
            else:
                return 2, min(-raw, self.bankroll)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------
    def step(self, action):
        side, wager = self._parse_action(action)
        game = self.games.iloc[self.current_game_idx]

        # --- outcome ---
        home_won = game["score_home"] > game["score_away"]
        reward = 0.0

        if side == 1:  # Bet Home
            if home_won:
                reward = wager * self._calculate_payout(game["home_ml"])
            else:
                reward = -wager
        elif side == 2:  # Bet Away
            if not home_won:
                reward = wager * self._calculate_payout(game["away_ml"])
            else:
                reward = -wager

        # Bankroll tracks actual outcomes only
        self.bankroll += reward

        # --- reward shaping (does not touch bankroll) ---
        shaped_reward = reward
        if side != 0:
            wager_norm = wager / self.bet_size

            # 1) Edge penalty (linear) — discourages betting without edge
            if self.no_edge_penalty > 0:
                if abs(game["elo_market_residual"]) < self.edge_threshold:
                    shaped_reward -= self.no_edge_penalty * wager_norm

            # 2) Quadratic risk penalty — breaks linear scaling symmetry;
            #    makes large bets super-linearly expensive
            if self.risk_coeff > 0:
                shaped_reward -= self.risk_coeff * wager_norm ** 2

            # 3) EV-sizing bonus — rewards betting more on positive-EV games;
            #    combined with (2) the optimal wager ∝ EV (Kelly-like)
            if self.ev_bonus > 0:
                ev = game["elo_ev_home"] if side == 1 else game["elo_ev_away"]
                if ev > 0:
                    shaped_reward += self.ev_bonus * ev * wager

        # Record (side stored as 0/1/2 so get_bet_statistics works unchanged)
        self.bet_history.append(
            {
                "game_idx": self.current_game_idx,
                "date": game.get("date", None),
                "home_team": game.get("home_team", None),
                "away_team": game.get("away_team", None),
                "action": side,
                "wager": wager,
                "reward": reward,
                "bankroll": self.bankroll,
                "elo_market_residual": game["elo_market_residual"],
            }
        )

        self.current_game_idx += 1
        terminated = self.current_game_idx >= len(self.games) or self.bankroll <= 0
        truncated = False

        return self._get_obs(), shaped_reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Extended statistics
    # ------------------------------------------------------------------
    def get_bet_statistics(self) -> dict:
        """Extends parent stats with average and max wager."""
        stats = super().get_bet_statistics()
        if not self.bet_history or stats.get("total_bets", 0) == 0:
            return stats

        bets_df = pd.DataFrame(self.bet_history)
        actual_bets = bets_df[bets_df["action"] != 0]
        stats["avg_wager"] = float(actual_bets["wager"].mean())
        stats["max_wager"] = float(actual_bets["wager"].max())
        return stats
