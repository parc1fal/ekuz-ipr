"""
Generic DQN training pipeline.

Usage:
    python scripts/train.py configs/elo.yaml
    python scripts/train.py configs/elo.yaml agent.total_timesteps=200000 agent.gamma=0.95
"""

import argparse
import os
import sys

import pandas as pd
import yaml
import wandb
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import get_env_class
from src.envs.multi_season import create_multi_season_wrapper


# ---------------------------------------------------------------------------
# Config loading + CLI overrides
# ---------------------------------------------------------------------------
def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _coerce(s: str):
    """Try int -> float -> str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply dotted key=value overrides to a nested config dict."""
    for item in overrides:
        key_path, value_str = item.split("=", 1)
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = _coerce(value_str)
    return config


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_season_dfs(data_cfg: dict) -> list:
    dfs = []
    for season in data_cfg["train_seasons"]:
        path = os.path.join(
            data_cfg["dir"],
            data_cfg["filename_pattern"].format(season=season),
        )
        df = pd.read_csv(path)
        print(f"  Loaded season {season}: {len(df)} games")
        dfs.append(df)
    return dfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train a betting agent from a YAML config.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides, e.g. agent.gamma=0.95")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    # --- resolve env class from registry ---
    env_class = get_env_class(cfg["env"]["name"])

    # --- load training data ---
    print("\nLoading training data...")
    train_dfs = load_season_dfs(cfg["data"])

    # --- wandb ---
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["wandb"].get("name"),
        entity=cfg["wandb"].get("entity"),
        config=cfg,
        resume="allow",
    )

    # --- build vectorised env ---
    Wrapper = create_multi_season_wrapper(env_class)
    env_params = cfg["env_params"]
    env = DummyVecEnv([
        lambda dfs=train_dfs: Wrapper(dfs, **env_params)
    ])

    # --- DQN: everything under agent except algorithm + total_timesteps
    #     passes through directly to the SB3 constructor ---
    algo_cfg = {
        k: v for k, v in cfg["agent"].items()
        if k not in ("algorithm", "total_timesteps")
    }
    assert cfg["agent"]["algorithm"] == "DQN", (
        f"Unsupported algorithm: {cfg['agent']['algorithm']}"
    )

    model = DQN("MlpPolicy", env, **algo_cfg)
    model.learn(
        total_timesteps=cfg["agent"]["total_timesteps"],
        callback=WandbCallback(),
    )

    # --- save ---
    save_path = cfg["model"]["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}.zip")

    wandb.finish()


if __name__ == "__main__":
    main()
