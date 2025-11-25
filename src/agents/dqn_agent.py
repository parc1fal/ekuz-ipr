"""
DQN Agent for NBA Betting

Trains a Deep Q-Network to learn profitable betting strategies using ELO features.
Based on Stanford CS224R paper approach but with our ELO implementation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import sys
from datetime import datetime

# Add paths for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "data" / "elo"))
sys.path.insert(0, str(project_root / "src" / "envs"))

from betting_env_elo import EloBettingEnv  # ignore pylance error
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class BettingMetricsCallback:
    """
    Custom callback to track betting-specific metrics during training
    """

    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.verbose = verbose
        self.evaluations = []

    def __call__(self, locals_dict, globals_dict):
        """Called during training"""
        if locals_dict.get("num_timesteps", 0) % self.eval_freq == 0:
            metrics = evaluate_agent(
                locals_dict["self"], self.eval_env, n_episodes=1, deterministic=True
            )

            self.evaluations.append(
                {"timestep": locals_dict["num_timesteps"], **metrics}
            )

            if self.verbose:
                print(f"\n[Eval @ {locals_dict['num_timesteps']} steps]")
                print(f"  ROI: {metrics['roi']:.2f}%")
                print(f"  Win Rate: {metrics['win_rate']:.2f}%")
                print(f"  Bets: {metrics['total_bets']}/{metrics['total_games']}")
                print(f"  Final Bankroll: ${metrics['final_bankroll']:.2f}")


def load_data_splits(features_dir: Path):
    """
    Load and split data into train/validation/test sets

    Data split strategy:
    - Train: 2008-2018 (11 seasons) - Learn patterns
    - Validation: 2019-2020 (2 seasons) - Hyperparameter tuning
    - Test: 2021-2024 (4 seasons) - Final evaluation

    Args:
        features_dir: Directory containing all_seasons_with_elo.csv

    Returns:
        train_df, val_df, test_df
    """
    data_path = features_dir / "all_seasons_with_elo.csv"

    print("=" * 70)
    print("Loading Data Splits")
    print("=" * 70)
    print(f"\nLoading from: {data_path}")

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to regular season only (no playoffs for now)
    if "regular" in df.columns:
        df = df[df["regular"] == True].copy()

    print(f"Total games: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Seasons: {sorted(df['season'].unique())}")

    # Split by season
    train_df = df[df["season"] <= 2018].copy()
    val_df = df[(df["season"] >= 2019) & (df["season"] <= 2020)].copy()
    test_df = df[df["season"] >= 2021].copy()

    print(
        f"\nTrain: {len(train_df)} games ({train_df['season'].min()}-{train_df['season'].max()})"
    )
    print(
        f"Val:   {len(val_df)} games ({val_df['season'].min()}-{val_df['season'].max()})"
    )
    print(
        f"Test:  {len(test_df)} games ({test_df['season'].min()}-{test_df['season'].max()})"
    )

    return train_df, val_df, test_df


def evaluate_agent(model, env, n_episodes=1, deterministic=True):
    """
    Evaluate agent and return detailed metrics

    Args:
        model: Trained DQN model
        env: EloBettingEnv instance
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions (no exploration)

    Returns:
        dict with metrics: roi, win_rate, total_bets, etc.
    """
    all_rewards = []
    all_stats = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        all_rewards.append(episode_reward)
        stats = env.get_bet_statistics()
        all_stats.append(stats)

    # Aggregate metrics
    avg_stats = {
        "roi": np.mean([s["roi"] for s in all_stats]),
        "win_rate": np.mean([s["win_rate"] for s in all_stats]),
        "total_bets": int(np.mean([s["total_bets"] for s in all_stats])),
        "final_bankroll": np.mean([s["final_bankroll"] for s in all_stats]),
        "avg_reward": np.mean([s.get("avg_reward", 0) for s in all_stats]),
        "total_profit": np.mean([s.get("total_profit", 0) for s in all_stats]),
        "total_games": len(env.games),
        "bet_frequency": np.mean([s["total_bets"] for s in all_stats])
        / len(env.games)
        * 100,
    }

    return avg_stats


def print_evaluation_report(metrics, split_name="Test"):
    """Print formatted evaluation report"""
    print(f"\n{'=' * 70}")
    print(f"{split_name} Set Evaluation")
    print("=" * 70)
    print(f"\n📊 Performance Metrics:")
    print(f"  ROI:              {metrics['roi']:>8.2f}%")
    print(f"  Win Rate:         {metrics['win_rate']:>8.2f}%")
    print(f"  Final Bankroll:   ${metrics['final_bankroll']:>8.2f}")
    print(f"  Total Profit:     ${metrics['total_profit']:>8.2f}")

    print(f"\n🎯 Betting Behavior:")
    print(f"  Total Bets:       {metrics['total_bets']:>8} / {metrics['total_games']}")
    print(f"  Bet Frequency:    {metrics['bet_frequency']:>8.1f}%")
    print(f"  Avg Reward/Bet:   ${metrics['avg_reward']:>8.2f}")

    print(f"\n💡 Interpretation:")

    # ROI interpretation
    if metrics["roi"] > 5:
        print(f"  ✅ Strong positive ROI - agent found profitable patterns")
    elif metrics["roi"] > 0:
        print(f"  ✓  Positive ROI - modest profit")
    elif metrics["roi"] > -5:
        print(f"  ⚠️  Small loss - close to break-even")
    else:
        print(f"  ❌ Negative ROI - agent struggled")

    # Win rate interpretation
    breakeven_wr = 52.4  # Need >52.4% to overcome -110 vig
    if metrics["win_rate"] > breakeven_wr:
        print(f"  ✅ Win rate above breakeven ({breakeven_wr}%)")
    else:
        print(f"  ❌ Win rate below breakeven ({breakeven_wr}%)")

    # Betting frequency interpretation
    if metrics["bet_frequency"] < 10:
        print(f"  📉 Very conservative - betting rarely")
    elif metrics["bet_frequency"] < 30:
        print(f"  📊 Selective - betting on strong signals")
    elif metrics["bet_frequency"] < 60:
        print(f"  📈 Active - betting frequently")
    else:
        print(f"  ⚠️  Over-active - may be betting too often")

    print()


def train_dqn_agent(
    train_df,
    val_df,
    output_dir: Path,
    total_timesteps: int = 500_000,
    learning_rate: float = 1e-4,
    buffer_size: int = 100_000,
    batch_size: int = 32,
    gamma: float = 0.99,
    exploration_fraction: float = 0.3,
    exploration_final_eps: float = 0.01,
    target_update_interval: int = 1000,
    seed: int = 42,
):
    """
    Train DQN agent on NBA betting task

    Args:
        train_df: Training data
        val_df: Validation data
        output_dir: Where to save models and logs
        total_timesteps: Total training steps (500k = ~40 full seasons)
        learning_rate: Adam learning rate
        buffer_size: Replay buffer size
        batch_size: Minibatch size
        gamma: Discount factor (0.99 = long-term planning)
        exploration_fraction: Fraction of training for epsilon decay
        exploration_final_eps: Final epsilon (1% random actions)
        target_update_interval: Update target network every N steps
        seed: Random seed

    Returns:
        Trained model
    """
    print("\n" + "=" * 70)
    print("Training DQN Agent")
    print("=" * 70)

    # Create environments
    print("\nCreating environments...")
    train_env = EloBettingEnv(train_df, initial_bankroll=500)
    train_env = Monitor(train_env)

    val_env = EloBettingEnv(val_df, initial_bankroll=500)

    print(f"  Train env: {len(train_df)} games")
    print(f"  Val env:   {len(val_df)} games")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure DQN
    print("\nConfiguring DQN...")
    print(f"  Total timesteps:    {total_timesteps:,}")
    print(f"  Learning rate:      {learning_rate}")
    print(f"  Buffer size:        {buffer_size:,}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Gamma:              {gamma}")
    print(
        f"  Exploration:        {exploration_fraction * 100}% → {exploration_final_eps * 100}%"
    )

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,  # Start training after 1000 steps
        batch_size=batch_size,
        gamma=gamma,
        train_freq=4,  # Train every 4 steps
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=dict(net_arch=[256, 256]),  # 2 hidden layers, 256 units each
        verbose=1,
        seed=seed,
        tensorboard_log=str(output_dir / "tensorboard"),
    )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="dqn_betting",
    )

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    # Train
    print("\n🚀 Starting training...")
    print(f"Estimated time: ~{total_timesteps / len(train_df) * 2:.0f} minutes")
    print("(Training will pause every 10k steps for validation)\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=100,
        progress_bar=True,
    )

    # Save final model
    final_model_path = output_dir / "final_model.zip"
    model.save(final_model_path)
    print(f"\n✓ Training complete! Model saved to: {final_model_path}")

    return model


def run_baseline_comparison(test_df):
    """
    Run baseline strategies for comparison

    Baselines:
    1. Random betting (50/50 home/away, skip 33% of time)
    2. Always bet on favorites
    3. Always bet on underdogs

    Returns:
        dict of baseline results
    """
    print("\n" + "=" * 70)
    print("Baseline Comparison")
    print("=" * 70)

    baselines = {}

    # 1. Random baseline
    print("\n1. Random Betting Strategy...")
    env = EloBettingEnv(test_df, initial_bankroll=500)
    obs, _ = env.reset()
    done = False

    while not done:
        action = np.random.choice([0, 1, 2])  # Random action
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    baselines["random"] = env.get_bet_statistics()
    print_evaluation_report(baselines["random"], "Random Strategy")

    # 2. Always bet favorites
    print("\n2. Always Bet Favorites...")
    env = EloBettingEnv(test_df, initial_bankroll=500)
    obs, _ = env.reset()
    done = False

    while not done:
        # Favorite has lower (more negative) money line
        home_ml = obs[5]
        away_ml = obs[6]
        action = 1 if home_ml < away_ml else 2
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    baselines["favorites"] = env.get_bet_statistics()
    print_evaluation_report(baselines["favorites"], "Always Bet Favorites")

    # 3. Always bet underdogs
    print("\n3. Always Bet Underdogs...")
    env = EloBettingEnv(test_df, initial_bankroll=500)
    obs, _ = env.reset()
    done = False

    while not done:
        # Underdog has higher (more positive) money line
        home_ml = obs[5]
        away_ml = obs[6]
        action = 1 if home_ml > away_ml else 2
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    baselines["underdogs"] = env.get_bet_statistics()
    print_evaluation_report(baselines["underdogs"], "Always Bet Underdogs")

    return baselines


def main():
    """Main training and evaluation pipeline"""

    # Setup paths
    project_root = Path(__file__).resolve().parent.parent.parent
    features_dir = project_root / "data" / "features"
    output_dir = (
        project_root
        / "models"
        / f"dqn_betting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    print("=" * 70)
    print("DQN NBA BETTING AGENT")
    print("=" * 70)
    print(f"\nProject root: {project_root}")
    print(f"Features dir: {features_dir}")
    print(f"Output dir:   {output_dir}")

    # Load data
    train_df, val_df, test_df = load_data_splits(features_dir)

    # Train agent
    model = train_dqn_agent(
        train_df=train_df,
        val_df=val_df,
        output_dir=output_dir,
        total_timesteps=500_000,  # 500k steps ≈ 40 full training seasons
        learning_rate=1e-4,
        buffer_size=100_000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        seed=42,
    )

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Final Evaluation on Test Set (2021-2024)")
    print("=" * 70)

    test_env = EloBettingEnv(test_df, initial_bankroll=500)
    test_metrics = evaluate_agent(model, test_env, n_episodes=1, deterministic=True)
    print_evaluation_report(test_metrics, "DQN Agent - Test Set")

    # Run baselines for comparison
    baseline_metrics = run_baseline_comparison(test_df)

    # Summary comparison
    print("\n" + "=" * 70)
    print("Performance Summary")
    print("=" * 70)
    print(
        f"\n{'Strategy':<20} {'ROI':>10} {'Win Rate':>10} {'Bets':>8} {'Final $':>10}"
    )
    print("-" * 70)

    strategies = {
        "DQN Agent": test_metrics,
        "Random": baseline_metrics["random"],
        "Favorites": baseline_metrics["favorites"],
        "Underdogs": baseline_metrics["underdogs"],
    }

    for name, metrics in strategies.items():
        print(
            f"{name:<20} {metrics['roi']:>9.2f}% {metrics['win_rate']:>9.2f}% "
            f"{metrics['total_bets']:>8} ${metrics['final_bankroll']:>9.2f}"
        )

    # Save results
    results_path = output_dir / "results.txt"
    with open(results_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("DQN NBA BETTING AGENT - RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write(
            f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("Test Set Performance (2021-2024):\n")
        for key, value in test_metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\nBaseline Comparisons:\n")
        for name, metrics in baseline_metrics.items():
            f.write(f"\n{name}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")

    print(f"\n✓ Results saved to: {results_path}")
    print("\n🎉 Training and evaluation complete!")

    return model, test_metrics, baseline_metrics


if __name__ == "__main__":
    model, test_metrics, baselines = main()
