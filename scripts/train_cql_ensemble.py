"""
Train an ensemble of K independent CQL models with different seeds.

All seeds share the same offline dataset — only network initialisation
and minibatch sampling order vary.

Usage:
    python scripts/train_cql_ensemble.py configs/cql_elo_fatigue.yaml
"""

import argparse
import os
import sys

import d3rlpy
import numpy as np
import wandb
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def load_dataset(npz_path: str) -> d3rlpy.dataset.MDPDataset:
    data = np.load(npz_path)
    return d3rlpy.dataset.MDPDataset(
        observations=data["observations"],
        actions=data["actions"],
        rewards=data["rewards"],
        terminals=data["terminals"],
        action_space=d3rlpy.ActionSpace.DISCRETE,
    )


def main():
    parser = argparse.ArgumentParser(description="Train a CQL ensemble.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)

    ensemble_cfg = cfg["ensemble"]
    seeds = ensemble_cfg["seeds"]
    save_dir = ensemble_cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    cql_cfg = cfg["cql"]
    base_wandb_name = cfg["wandb"].get("name", "cql-ensemble")

    # Load offline dataset (shared across all seeds)
    data_path = cfg["offline_data"]["save_path"]
    print(f"\nLoading offline data from {data_path}...")
    dataset = load_dataset(data_path)
    print(f"  {len(dataset.episodes)} episodes, {sum(len(e) for e in dataset.episodes)} transitions")

    print(f"\nTraining CQL ensemble of {len(seeds)} agents")
    print(f"  alpha={cql_cfg['alpha']}, seeds={seeds}")
    print(f"  Save directory: {save_dir}/\n")

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"  Training CQL agent {i+1}/{len(seeds)} — seed {seed}")
        print(f"{'='*60}")

        d3rlpy.seed(seed)

        wandb.init(
            project=cfg["wandb"]["project"],
            name=f"{base_wandb_name}-s{seed}",
            entity=cfg["wandb"].get("entity"),
            config={**cfg, "seed": seed},
            group=base_wandb_name,
            resume="allow",
        )

        cql = d3rlpy.algos.DiscreteCQLConfig(
            learning_rate=cql_cfg["learning_rate"],
            batch_size=cql_cfg["batch_size"],
            gamma=cql_cfg["gamma"],
            alpha=cql_cfg["alpha"],
            target_update_interval=cql_cfg["target_update_interval"],
            n_critics=cql_cfg.get("n_critics", 1),
        ).create(device="cpu:0")

        cql.fit(
            dataset,
            n_steps=cql_cfg["n_steps"],
            n_steps_per_epoch=cql_cfg["n_steps_per_epoch"],
        )

        save_path = os.path.join(save_dir, f"seed_{seed}.d3")
        cql.save(save_path)
        print(f"  Saved to {save_path}")

        wandb.finish()

    print(f"\n{'='*60}")
    print(f"  CQL ensemble complete — {len(seeds)} models saved to {save_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
