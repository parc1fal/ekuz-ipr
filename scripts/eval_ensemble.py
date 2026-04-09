"""
Evaluate an ensemble of DQN agents via Q-value averaging (Anschel 2017).

Two modes controlled by ensemble.aggregation in config:

  outer_ensemble   : K independently trained models, average Q-values (Anschel Alg 3 / MeanQ).
                     Expects seed_*.zip files in save_dir.

  averaged_dqn     : For each seed, average the last K snapshots from that seed's training run
                     (Anschel Algorithm 2), then outer-ensemble across seeds.
                     Expects seed_*/checkpoint_*.zip files in save_dir.

Usage:
    WANDB_MODE=disabled python scripts/eval_ensemble.py configs/ensemble_dqn_tiered.yaml
    WANDB_MODE=disabled python scripts/eval_ensemble.py configs/ensemble_averaged_dqn.yaml
"""

import argparse
import glob
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import yaml
import wandb
from stable_baselines3 import DQN, PPO

ALGO_REGISTRY = {"DQN": DQN, "PPO": PPO}

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.envs import get_env_class


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def _coerce(s: str):
    for fn in (int, float):
        try:
            return fn(s)
        except ValueError:
            pass
    if s in ("null", "None"):
        return None
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_dfs(data_cfg: dict) -> dict:
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
# Model loading
# ---------------------------------------------------------------------------
def load_outer_ensemble(save_dir: str, algo_cls) -> list:
    """Load seed_*.zip models → list of SB3 models."""
    paths = sorted(glob.glob(os.path.join(save_dir, "seed_*.zip")))
    if not paths:
        raise FileNotFoundError(f"No seed_*.zip found in {save_dir}")
    models = []
    for p in paths:
        models.append(algo_cls.load(p[:-4]))
        print(f"  Loaded {os.path.basename(p)}")
    return models


def load_averaged_dqn_members(save_dir: str, algo_cls, last_k: int) -> list:
    """
    For each seed_N/ subdirectory in save_dir, load the last `last_k` checkpoint zips.
    Returns list of lists: [[snap1, snap2, ..., snapK], [snap1, ...], ...]
    One inner list per seed (outer ensemble member).
    """
    seed_dirs = sorted(glob.glob(os.path.join(save_dir, "seed_*")))
    seed_dirs = [d for d in seed_dirs if os.path.isdir(d)]
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* subdirectories found in {save_dir}")

    all_members = []
    for sd in seed_dirs:
        ckpts = sorted(glob.glob(os.path.join(sd, "checkpoint_*.zip")))
        if not ckpts:
            # Fall back to final model in parent dir named after this seed
            seed_name = os.path.basename(sd)
            final = os.path.join(os.path.dirname(sd), f"{seed_name}.zip")
            if os.path.exists(final):
                ckpts = [final]
            else:
                print(f"  [warn] No checkpoints found in {sd}, skipping")
                continue
        selected = ckpts[-last_k:]
        snapshots = [algo_cls.load(p[:-4]) for p in selected]
        all_members.append(snapshots)
        print(f"  {os.path.basename(sd)}: averaged {len(snapshots)} snapshots "
              f"(of {len(ckpts)} available)")
    return all_members


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def q_average_action(models: list, obs, uncertainty_threshold=None) -> int:
    """Anschel Alg 3 / MeanQ: average Q-values across K models, act greedy on mean."""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_stack = torch.stack([m.policy.q_net(obs_t) for m in models])  # (K, 1, n_actions)
    mean_q = q_stack.mean(dim=0)  # (1, n_actions)
    best = mean_q.argmax(dim=1).item()

    if uncertainty_threshold is not None and best != 0:
        std_q = q_stack.std(dim=0)
        if std_q[0, best].item() > uncertainty_threshold:
            return 0  # skip when ensemble disagrees

    return best


def averaged_dqn_action(members: list, obs, uncertainty_threshold=None) -> int:
    """
    Anschel Alg 2 + outer ensemble:
      - For each outer member: average Q-values across its K snapshots (Averaged-DQN).
      - Outer-ensemble: average those per-member averaged Q-values.
    """
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        member_qs = []
        for snapshots in members:
            snap_q = torch.stack([m.policy.q_net(obs_t) for m in snapshots])
            member_qs.append(snap_q.mean(dim=0))  # avg within member
        outer_q = torch.stack(member_qs).mean(dim=0)  # avg across members

    best = outer_q.argmax(dim=1).item()

    if uncertainty_threshold is not None and best != 0:
        member_q_stack = torch.stack(member_qs)
        std_q = member_q_stack.std(dim=0)
        if std_q[0, best].item() > uncertainty_threshold:
            return 0

    return best


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(env, action_fn) -> dict:
    obs, _ = env.reset()
    done = False
    while not done:
        action = action_fn(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return env.get_bet_statistics()


def print_stats(label, stats):
    if not stats:
        print(f"  {label}: no bets recorded")
        return
    msg = (
        f"  {label}: "
        f"ROI={stats.get('roi', 0):+.2f}%  "
        f"bets={stats.get('total_bets', 0)}  "
        f"win%={stats.get('win_rate', 0):.1f}  "
        f"profit={stats.get('total_profit', 0):+.2f}  "
        f"bankroll={stats.get('final_bankroll', 0):.2f}"
    )
    if "avg_wager" in stats:
        msg += f"  avg_wager={stats['avg_wager']:.2f}"
    print(msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    ensemble_cfg = cfg["ensemble"]
    save_dir = ensemble_cfg["save_dir"]
    aggregation = ensemble_cfg.get("aggregation", "outer_ensemble")
    uncertainty_threshold = ensemble_cfg.get("uncertainty_threshold")
    last_k = ensemble_cfg.get("last_k_snapshots", 5)

    env_class = get_env_class(cfg["env"]["name"])
    env_params = cfg["env_params"]
    algo_cls = ALGO_REGISTRY[cfg["agent"]["algorithm"]]

    print(f"\nAggregation mode: {aggregation}")

    # --- load models ---
    print(f"\nLoading ensemble from {save_dir}/...")
    if aggregation == "averaged_dqn":
        members = load_averaged_dqn_members(save_dir, algo_cls, last_k)
        K = len(members)
        print(f"  Outer ensemble size: K={K}  (each with up to {last_k} snapshots)")
    else:
        models = load_outer_ensemble(save_dir, algo_cls)
        K = len(models)
        print(f"  Ensemble size: K={K}")

    if uncertainty_threshold is not None:
        print(f"  Uncertainty threshold: {uncertainty_threshold}")

    # --- load test data ---
    print("\nLoading test data...")
    test_dfs = load_test_dfs(cfg["data"])

    # --- wandb ---
    base_name = cfg["wandb"].get("name", "ensemble")
    wandb.init(
        project=cfg["wandb"]["project"],
        name=f"{base_name}-eval",
        entity=cfg["wandb"].get("entity"),
        config={**cfg, "ensemble_size": K, "aggregation": aggregation},
        tags=["eval", "ensemble"],
        resume="allow",
    )

    # --- baselines ---
    rng = np.random.default_rng(42)
    first_season = cfg["data"]["test_seasons"][0]
    _probe = env_class(games_df=test_dfs[first_season], **env_params)
    if isinstance(_probe.action_space, gym.spaces.Box):
        random_fn = lambda obs, _as=_probe.action_space: _as.sample()
        skip_fn = lambda obs: np.array([0.0], dtype=np.float32)
    else:
        random_fn = lambda obs: rng.integers(3)
        skip_fn = lambda obs: 0

    # --- build action function ---
    if aggregation == "averaged_dqn":
        action_fn = lambda obs: averaged_dqn_action(members, obs, uncertainty_threshold)
        # individual model ROIs: use final model per seed (first snapshot in each member)
        individual_models = [snaps[-1] for snaps in members]  # final checkpoint per seed
    else:
        action_fn = lambda obs: q_average_action(models, obs, uncertainty_threshold)
        individual_models = models

    # --- evaluate ---
    ensemble_rois = []
    individual_rois = {i: [] for i in range(len(individual_models))}

    for season in cfg["data"]["test_seasons"]:
        print(f"\n{'='*60}")
        print(f"  Season {season}")
        print(f"{'='*60}")
        df = test_dfs[season]

        ensemble_stats = run_episode(env_class(games_df=df, **env_params), action_fn)
        label = f"Ensemble({aggregation},K={K})"
        print_stats(label, ensemble_stats)

        ind_season_rois = []
        for i, model in enumerate(individual_models):
            stats = run_episode(
                env_class(games_df=df, **env_params),
                lambda obs, m=model: m.predict(obs, deterministic=True)[0],
            )
            roi = stats.get("roi", 0)
            ind_season_rois.append(roi)
            individual_rois[i].append(roi)

        print(f"  Individual ROIs: [{', '.join(f'{r:+.2f}%' for r in ind_season_rois)}]")
        print(f"  Individual mean={np.mean(ind_season_rois):+.2f}%  std={np.std(ind_season_rois):.2f}pp")

        random_stats = run_episode(env_class(games_df=df, **env_params), random_fn)
        skip_stats = run_episode(env_class(games_df=df, **env_params), skip_fn)
        print_stats("Random ", random_stats)
        print_stats("Skip   ", skip_stats)

        ensemble_rois.append(ensemble_stats.get("roi", 0))

        wandb.log({
            f"eval/season_{season}/ensemble_roi":       ensemble_stats.get("roi", 0),
            f"eval/season_{season}/ensemble_bets":      ensemble_stats.get("total_bets", 0),
            f"eval/season_{season}/individual_mean":    np.mean(ind_season_rois),
            f"eval/season_{season}/individual_std":     np.std(ind_season_rois),
            f"eval/season_{season}/individual_min":     np.min(ind_season_rois),
            f"eval/season_{season}/individual_max":     np.max(ind_season_rois),
            f"eval/season_{season}/random_roi":         random_stats.get("roi", 0),
            f"eval/season_{season}/skip_roi":           skip_stats.get("roi", 0),
        })

    # --- summary ---
    avg_ensemble = float(np.mean(ensemble_rois))
    per_model_avg = [float(np.mean(individual_rois[i])) for i in range(len(individual_models))]

    print(f"\n{'='*60}")
    print(f"  Summary — Average ROI across test seasons")
    print(f"{'='*60}")
    print(f"  {label:<30} {avg_ensemble:+.2f}%")
    print(f"  Individual mean:               {np.mean(per_model_avg):+.2f}%")
    print(f"  Individual std:                {np.std(per_model_avg):.2f}pp")
    print(f"  Individual range:              [{np.min(per_model_avg):+.2f}%, {np.max(per_model_avg):+.2f}%]")
    print(f"\n  Per-model avg ROIs:")
    for i, roi in enumerate(per_model_avg):
        print(f"    Model {i}: {roi:+.2f}%")

    wandb.log({
        "eval/avg_ensemble_roi":       avg_ensemble,
        "eval/individual_mean_roi":    np.mean(per_model_avg),
        "eval/individual_std_roi":     np.std(per_model_avg),
        "eval/individual_min_roi":     np.min(per_model_avg),
        "eval/individual_max_roi":     np.max(per_model_avg),
    })
    wandb.finish()


if __name__ == "__main__":
    main()
