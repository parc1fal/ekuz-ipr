"""
Collect offline transitions for CQL training.

Supports two behavioral policies:
  random : uniform random over {skip, home, away}
  dqn    : trained DQN model (requires offline_data.dqn_model_path in config)

Usage:
    python scripts/collect_offline_data.py configs/cql_elo_fatigue.yaml
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DQN

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import get_env_class


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Collect offline transitions for CQL.")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    offline_cfg = cfg["offline_data"]
    save_path = offline_cfg["save_path"]
    seed = offline_cfg.get("seed", 42)
    policy_type = offline_cfg.get("policy", "random")

    env_class = get_env_class(cfg["env"]["name"])
    env_params = dict(cfg["env_params"])

    rng = np.random.default_rng(seed)

    # Load DQN behavioral policy if specified
    dqn_model = None
    if policy_type == "dqn":
        model_path = offline_cfg["dqn_model_path"]
        print(f"\nLoading DQN behavioral policy from {model_path}...")
        dqn_model = DQN.load(model_path)

    observations, actions, rewards, terminals = [], [], [], []

    print(f"\nCollecting offline data ({policy_type} policy, seed={seed})...")
    total_games = 0
    for season in cfg["data"]["train_seasons"]:
        path = os.path.join(
            cfg["data"]["dir"],
            cfg["data"]["filename_pattern"].format(season=season),
        )
        df = pd.read_csv(path)
        env = env_class(games_df=df, **env_params)
        obs, _ = env.reset()
        done = False
        season_games = 0

        while not done:
            if policy_type == "dqn":
                action = int(dqn_model.predict(obs, deterministic=True)[0])
            else:
                action = int(rng.integers(3))

            next_obs, reward, terminated, truncated, _ = env.step(action)

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            terminals.append(terminated or truncated)

            obs = next_obs
            done = terminated or truncated
            season_games += 1

        total_games += season_games
        print(f"  Season {season}: {season_games} transitions")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(
        save_path,
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool),
    )

    # Report action distribution
    acts = np.array(actions)
    skip_pct = (acts == 0).mean() * 100
    home_pct = (acts == 1).mean() * 100
    away_pct = (acts == 2).mean() * 100
    print(f"\nSaved {total_games} transitions to {save_path}")
    print(f"  Action distribution: skip={skip_pct:.1f}% home={home_pct:.1f}% away={away_pct:.1f}%")


if __name__ == "__main__":
    main()
