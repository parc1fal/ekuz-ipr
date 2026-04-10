"""
Train a single CQL model on offline data.

Usage:
    python scripts/train_cql.py configs/cql_elo_fatigue.yaml
    python scripts/train_cql.py configs/cql_elo_fatigue.yaml cql.alpha=0.5
"""

import argparse
import os
import sys

import d3rlpy
import numpy as np
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
    parser = argparse.ArgumentParser(description="Train a CQL model on offline data.")
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    cql_cfg = cfg["cql"]

    # Load offline dataset
    data_path = cfg["offline_data"]["save_path"]
    print(f"\nLoading offline data from {data_path}...")
    dataset = load_dataset(data_path)
    print(f"  {len(dataset.episodes)} episodes, {sum(len(e) for e in dataset.episodes)} transitions")

    # Seed
    seed = cfg.get("agent", {}).get("seed", 42)
    d3rlpy.seed(seed)

    # Build CQL
    cql = d3rlpy.algos.DiscreteCQLConfig(
        learning_rate=cql_cfg["learning_rate"],
        batch_size=cql_cfg["batch_size"],
        gamma=cql_cfg["gamma"],
        alpha=cql_cfg["alpha"],
        target_update_interval=cql_cfg["target_update_interval"],
        n_critics=cql_cfg.get("n_critics", 1),
    ).create(device="cpu:0")

    print(f"\nTraining CQL (alpha={cql_cfg['alpha']}, seed={seed})...")
    print(f"  n_steps={cql_cfg['n_steps']}, batch_size={cql_cfg['batch_size']}")

    cql.fit(
        dataset,
        n_steps=cql_cfg["n_steps"],
        n_steps_per_epoch=cql_cfg["n_steps_per_epoch"],
    )

    # Save
    save_path = cfg["model"]["save_path"]
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cql.save(save_path + ".d3")
    print(f"\nModel saved to {save_path}.d3")


if __name__ == "__main__":
    main()
