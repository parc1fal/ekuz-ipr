"""
Train an ensemble of K independent DQN agents with different seeds.

Usage:
    python scripts/train_ensemble.py configs/ensemble_dqn_tiered.yaml
    python scripts/train_ensemble.py configs/ensemble_dqn_tiered.yaml agent.total_timesteps=100000
"""

import argparse
import os
import sys

import pandas as pd
import yaml
import wandb
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback

ALGO_REGISTRY = {
    "DQN": DQN,
    "PPO": PPO,
}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import get_env_class
from src.envs.multi_season import create_multi_season_wrapper


def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _coerce(s: str):
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
    for item in overrides:
        key_path, value_str = item.split("=", 1)
        keys = key_path.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = _coerce(value_str)
    return config


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


def main():
    parser = argparse.ArgumentParser(description="Train an ensemble of betting agents.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    ensemble_cfg = cfg["ensemble"]
    seeds = ensemble_cfg["seeds"]
    save_dir = ensemble_cfg["save_dir"]
    snapshot_freq = ensemble_cfg.get("snapshot_freq", None)  # None = no checkpoint saving
    os.makedirs(save_dir, exist_ok=True)

    env_class = get_env_class(cfg["env"]["name"])

    print(f"\nLoading training data...")
    train_dfs = load_season_dfs(cfg["data"])

    algo_name = cfg["agent"]["algorithm"]
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(f"Unsupported algorithm: '{algo_name}'. Available: {list(ALGO_REGISTRY.keys())}")

    algo_cfg = {k: v for k, v in cfg["agent"].items() if k not in ("algorithm", "total_timesteps")}
    total_timesteps = cfg["agent"]["total_timesteps"]
    base_wandb_name = cfg["wandb"].get("name", "ensemble")

    print(f"\nTraining ensemble of {len(seeds)} agents ({algo_name})")
    print(f"Seeds: {seeds}")
    print(f"Save directory: {save_dir}/\n")

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Training agent {i+1}/{len(seeds)} — seed {seed}")
        print(f"{'='*60}")

        wandb.init(
            project=cfg["wandb"]["project"],
            name=f"{base_wandb_name}-s{seed}",
            entity=cfg["wandb"].get("entity"),
            config={**cfg, "seed": seed},
            resume="allow",
            group=base_wandb_name,
        )

        Wrapper = create_multi_season_wrapper(env_class)
        env = DummyVecEnv([lambda dfs=train_dfs: Wrapper(dfs, **cfg["env_params"])])

        model = ALGO_REGISTRY[algo_name](
            "MlpPolicy", env, seed=seed, **{k: v for k, v in algo_cfg.items() if k != "seed"},
        )

        callbacks = [WandbCallback()]
        if snapshot_freq is not None:
            ckpt_dir = os.path.join(save_dir, f"seed_{seed}")
            os.makedirs(ckpt_dir, exist_ok=True)
            callbacks.append(CheckpointCallback(
                save_freq=snapshot_freq,
                save_path=ckpt_dir,
                name_prefix="checkpoint",
                verbose=0,
            ))

        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        # Save final model: either in the checkpoint dir (averaged_dqn) or flat (outer_ensemble)
        if snapshot_freq is not None:
            save_path = os.path.join(save_dir, f"seed_{seed}", f"checkpoint_{total_timesteps}_steps")
        else:
            save_path = os.path.join(save_dir, f"seed_{seed}")
        model.save(save_path)
        print(f"  Saved to {save_path}.zip")

        wandb.finish()

    print(f"\n{'='*60}")
    print(f"  Ensemble training complete — {len(seeds)} models saved to {save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
