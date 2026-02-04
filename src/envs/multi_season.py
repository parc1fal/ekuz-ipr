"""
Generic multi-season wrapper factory.

Usage:
    Wrapper = create_multi_season_wrapper(EloBettingEnv)
    env = Wrapper(season_dfs, initial_bankroll=500, bet_size=1.0)

The returned class cycles through the provided season DataFrames, one per
episode.  It uses the same inheritance + self.games-swap pattern that was
validated in train_simple_dqn.py, just parameterised over the base env class.
"""

import pandas as pd


def create_multi_season_wrapper(env_class):
    """Return a subclass of *env_class* that cycles through season DataFrames."""

    class MultiSeasonWrapper(env_class):
        """Cycles through a list of season DataFrames, one per episode."""

        def __init__(self, season_dfs: list[pd.DataFrame], **kwargs):
            self.season_dfs = season_dfs
            self.season_idx = 0
            super().__init__(games_df=season_dfs[0], **kwargs)

        def reset(self, seed=None, options=None):
            self.games = self.season_dfs[self.season_idx].reset_index(drop=True)
            self.season_idx = (self.season_idx + 1) % len(self.season_dfs)
            return super().reset(seed=seed, options=options)

    return MultiSeasonWrapper
