"""
Generic evaluation script.

Usage:
    python scripts/eval.py configs/elo.yaml
    python scripts/eval.py configs/elo.yaml model.save_path=experiments/my_model
"""

import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import yaml
import wandb
from stable_baselines3 import DQN, PPO

ALGO_REGISTRY = {
    "DQN": DQN,
    "PPO": PPO,
}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import get_env_class


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
def load_test_dfs(data_cfg: dict) -> dict:
    """Returns {season_year: DataFrame} for all test seasons."""
    result = {}
    for season in data_cfg["test_seasons"]:
        path = os.path.join(
            data_cfg["dir"],
            data_cfg["filename_pattern"].format(season=season),
        )
        df = pd.read_csv(path)
        print(f"  Loaded season {season}: {len(df)} games")
        result[season] = df
    return result


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def run_episode(env, action_fn) -> dict:
    """Run one full episode. action_fn(obs) -> action. Returns bet statistics."""
    obs, _ = env.reset()
    done = False
    while not done:
        action = action_fn(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.get_bet_statistics()


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def print_stats(label, stats):
    if not stats:
        print(f"  {label}: no bets recorded")
        return
    msg = (
        f"  {label}: "
        f"ROI={stats.get('roi', 0):.2f}%  "
        f"bets={stats.get('total_bets', 0)}  "
        f"win%={stats.get('win_rate', 0):.1f}  "
        f"profit={stats.get('total_profit', 0):.2f}  "
        f"bankroll={stats.get('final_bankroll', 0):.2f}"
    )
    if "avg_wager" in stats:
        msg += f"  avg_wager={stats['avg_wager']:.2f}"
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained betting agent.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    env_class = get_env_class(cfg["env"]["name"])
    env_params = cfg["env_params"]

    # --- load model ---
    algo_name = cfg["agent"]["algorithm"]
    if algo_name not in ALGO_REGISTRY:
        raise ValueError(
            f"Unsupported algorithm: '{algo_name}'. "
            f"Available: {list(ALGO_REGISTRY.keys())}"
        )

    print(f"\nLoading {algo_name} model from {cfg['model']['save_path']}...")
    model = ALGO_REGISTRY[algo_name].load(cfg["model"]["save_path"])

    # --- load test data ---
    print("\nLoading test data...")
    test_dfs = load_test_dfs(cfg["data"])

    # --- wandb (separate eval run) ---
    base_name = cfg["wandb"].get("name")
    wandb.init(
        project=cfg["wandb"]["project"],
        name=(base_name + "-eval") if base_name else None,
        entity=cfg["wandb"].get("entity"),
        config=cfg,
        tags=["eval"],
        resume="allow",
    )

    # --- detect action-space type from a probe env ---
    first_season = cfg["data"]["test_seasons"][0]
    _probe = env_class(games_df=test_dfs[first_season], **env_params)
    continuous = isinstance(_probe.action_space, gym.spaces.Box)

    # Baselines adapt to action-space type
    rng = np.random.default_rng(42)
    if continuous:
        random_fn = lambda obs, _as=_probe.action_space: _as.sample()
        skip_fn   = lambda obs: np.array([0.0], dtype=np.float32)
    else:
        random_fn = lambda obs, _rng=rng: _rng.integers(3)
        skip_fn   = lambda obs: 0

    algo_key = algo_name.lower()          # "dqn" | "ppo"
    all_roi  = {"agent": [], "random": [], "skip": []}

    for season in cfg["data"]["test_seasons"]:
        print(f"\n--- Season {season} ---")
        df = test_dfs[season]

        agent_stats = run_episode(
            env_class(games_df=df, **env_params),
            lambda obs: model.predict(obs, deterministic=True)[0],
        )
        random_stats = run_episode(
            env_class(games_df=df, **env_params),
            random_fn,
        )
        skip_stats = run_episode(
            env_class(games_df=df, **env_params),
            skip_fn,
        )

        print_stats(f"{algo_name:7s}", agent_stats)
        print_stats("Random ", random_stats)
        print_stats("Skip   ", skip_stats)

        wandb.log({
            f"eval/season_{season}/{algo_key}_roi":        agent_stats.get("roi", 0),
            f"eval/season_{season}/{algo_key}_win_rate":   agent_stats.get("win_rate", 0),
            f"eval/season_{season}/{algo_key}_total_bets": agent_stats.get("total_bets", 0),
            f"eval/season_{season}/{algo_key}_profit":     agent_stats.get("total_profit", 0),
            f"eval/season_{season}/random_roi":            random_stats.get("roi", 0),
            f"eval/season_{season}/skip_roi":              skip_stats.get("roi", 0),
        })

        all_roi["agent"].append(agent_stats.get("roi", 0))
        all_roi["random"].append(random_stats.get("roi", 0))
        all_roi["skip"].append(skip_stats.get("roi", 0))

    # --- summary ---
    avg_agent  = float(np.mean(all_roi["agent"]))
    avg_random = float(np.mean(all_roi["random"]))
    avg_skip   = float(np.mean(all_roi["skip"]))

    print(f"\n{'=' * 60}")
    print("  Summary — Average ROI across test seasons")
    print("=" * 60)
    print(f"  {algo_name:7s} {avg_agent:+.2f}%")
    print(f"  Random  {avg_random:+.2f}%")
    print(f"  Skip    {avg_skip:+.2f}%")

    wandb.log({
        f"eval/avg_roi_{algo_key}": avg_agent,
        "eval/avg_roi_random":     avg_random,
        "eval/avg_roi_skip":       avg_skip,
    })

    wandb.finish()


if __name__ == "__main__":
    main()
