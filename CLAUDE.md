# CLAUDE.md — NBA RL Betting Codebase

## What this project is

Deep reinforcement learning agents that learn NBA moneyline betting strategies from historical data (2007–2023). Uses DQN and PPO via Stable-Baselines3, with Gymnasium-based custom environments.

---

## How to work in this codebase

### Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

### Workflow

1. **Plan first**: Enter plan mode for any non-trivial task (3+ steps or architectural decisions). Write a spec upfront. If something goes sideways, STOP and re-plan — don't keep pushing.
2. **Use subagents**: Offload research, exploration, and parallel analysis to subagents. One task per subagent. Keep the main context window clean.
3. **Track progress**: Write plans to `tasks/todo.md` with checkable items. Mark items complete as you go. Summarise changes at each step.
4. **Verify before done**: Never mark a task complete without proving it works. Run the relevant train/eval scripts. Check logs. Ask: "Would a staff engineer approve this?"
5. **Demand elegance**: For non-trivial changes, pause and ask "is there a more elegant way?" But skip this for simple, obvious fixes — don't over-engineer.
6. **Fix bugs autonomously**: When given a bug report, just fix it. Point at logs, errors, failing tests — then resolve. Zero hand-holding required from the user.

### Self-Improvement

- After ANY correction from the user: update `tasks/lessons.md` with the pattern.
- Write rules that prevent the same mistake. Review lessons at session start.

---

## Project structure

```
ekuz-ipr/
├── configs/                    # YAML experiment configs (env, agent, data paths)
│   ├── simple.yaml             # 3-feature baseline, DQN
│   ├── elo.yaml                # 7-feature ELO env, DQN
│   ├── elo_ppo.yaml            # 7-feature ELO env, PPO
│   ├── elo_fatigue.yaml        # 15-feature enriched env, DQN + reward shaping
│   ├── elo_ppo_fatigue.yaml    # 15-feature enriched env, PPO + reward shaping
│   ├── elo_dqn_tiered.yaml     # 15-feature, tiered bet sizing, DQN (best model)
│   └── elo_ppo_continuous.yaml # 15-feature, continuous bet sizing, PPO
│
├── data/
│   ├── raw/kaggle/             # Source CSV: nba_2008-2025.csv
│   ├── processed/kaggle/       # Cleaned per-season CSVs (season_YYYY.csv)
│   └── features/
│       ├── season_YYYY_elo.csv          # ELO-only features (7-dim obs)
│       ├── all_seasons_with_elo.csv
│       └── enriched/
│           ├── season_YYYY_enriched.csv # ELO + fatigue features (15-dim obs)
│           └── all_seasons_enriched.csv
│
├── src/
│   ├── data/
│   │   ├── kaggle_cleaner.py           # Raw data → cleaned per-season CSVs
│   │   ├── elo/
│   │   │   ├── elo_calculator.py       # NBAEloCalculator class (FiveThirtyEight method)
│   │   │   └── process_elo_features.py # Orchestration: compute ELO, save per-season
│   │   └── fatigue/
│   │       ├── fatigue_calculator.py   # Rest days, back-to-back, rolling stats
│   │       └── process_fatigue_features.py  # Orchestration: add fatigue to ELO data
│   │
│   ├── envs/
│   │   ├── __init__.py                 # ENV_REGISTRY: "simple", "elo", "elo_continuous"
│   │   ├── simple_env.py               # SimpleBettingEnv — 3 features, flat $1 bets
│   │   ├── betting_env_elo.py          # EloBettingEnv — 7 or 15 features, reward shaping
│   │   ├── betting_env_continuous.py   # ContinuousBettingEnv — variable bet sizing
│   │   └── multi_season.py             # MultiSeasonWrapper factory (cycles seasons)
│   │
│   └── evaluation/                     # Legacy eval code (not actively used)
│
├── scripts/
│   ├── train.py                # Generic training: `python scripts/train.py configs/elo.yaml`
│   ├── eval.py                 # Generic eval: `python scripts/eval.py configs/elo.yaml`
│   ├── train_simple_dqn.py     # Legacy standalone training script
│   └── save_processed_data.py  # Data preprocessing script
│
├── experiments/                # Saved model checkpoints (.zip files)
│   ├── elo_dqn_tiered.zip      # Best model (seed 42, +6.17% ROI)
│   ├── elo_dqn_tiered_s7.zip   # Same config, seed 7
│   ├── elo_dqn_tiered_s123.zip # Same config, seed 123
│   └── ...                     # ~25 total saved models
│
├── wandb/                      # Local W&B run logs (run-YYYYMMDD_HHMMSS-*/files/)
├── venv/                       # Python 3.11 virtual environment
└── requirements.txt
```

---

## How to run

**Activate the environment:**
```bash
source venv/bin/activate
# or use: venv/bin/python directly
```

**Train a model:**
```bash
python scripts/train.py configs/elo_dqn_tiered.yaml
# Override params: python scripts/train.py configs/elo.yaml agent.gamma=0.95
```

**Evaluate a model:**
```bash
WANDB_MODE=disabled python scripts/eval.py configs/elo_dqn_tiered.yaml
# Override model path: ... model.save_path=experiments/elo_dqn_tiered_s7
```

**Reprocess data pipeline:**
```bash
python scripts/save_processed_data.py                           # Stage 1: raw → cleaned
python src/data/elo/process_elo_features.py                     # Stage 2: add ELO
python src/data/fatigue/process_fatigue_features.py              # Stage 3: add fatigue
```

---

## Key design decisions

- **Environment registry**: `src/envs/__init__.py` maps string names → env classes. Configs reference envs by name ("simple", "elo", "elo_continuous").
- **YAML-driven experiments**: All hyperparameters live in config files, not code. New experiments = new YAML file.
- **Reward shaping is training-only**: Shaped rewards (edge penalty, risk penalty, EV bonus) modify the learning signal but never touch the bankroll. Eval metrics always use ground-truth financial outcomes.
- **Multi-season wrapper**: `create_multi_season_wrapper(env_class)` dynamically subclasses any env to cycle through seasons. Used during training only.
- **Observation auto-detection**: EloBettingEnv automatically detects fatigue columns in the data and adjusts observation dimension (7 vs 15).

---

## Current best results

| Model | Avg ROI (test) | Notes |
|---|---|---|
| Tiered DQN (seed 42) | +6.17% | Best single model |
| Tiered DQN (mean, 3 seeds) | +0.09% | High seed variance |
| ELO DQN EV (seed 42) | +2.36% | Best flat-bet model |
| Random baseline | ~-4.4% | Confirms house edge |

---

## Important notes

- The venv uses Python 3.11. Always use `venv/bin/python` or activate the venv.
- W&B logging is on by default. Use `WANDB_MODE=disabled` to skip it.
- Some older saved models (elo_dqn_model, elo_ppo_model, elo_dqn_s42/s7/s123) were trained with earlier env versions and have observation shape mismatches with the current code. The tiered and EV models work with the current env.
- Season numbers in configs represent the ending year (e.g., 2008 = 2007-08 season).
- Train seasons: 2008–2020 (13 seasons). Test seasons: 2021–2023 (3 seasons).
