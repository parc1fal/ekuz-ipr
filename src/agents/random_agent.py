"""
Random Agent for NBA Betting - Environment Debugging

This agent takes random actions to verify:
1. The environment works correctly
2. Betting mechanics are sound
3. Metrics are calculated properly

If this doesn't work, the environment has issues.
If this works but DQN doesn't, DQN has issues.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add paths
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "data" / "elo"))
sys.path.insert(0, str(project_root / "src" / "envs"))

from betting_env_elo import EloBettingEnv


def random_agent(env, n_episodes=1, verbose=True):
    """
    Run random agent on environment

    Args:
        env: EloBettingEnv instance
        n_episodes: Number of episodes to run
        verbose: Print detailed output

    Returns:
        dict with metrics
    """
    all_results = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        action_counts = {0: 0, 1: 0, 2: 0}  # Skip, Bet Home, Bet Away

        if verbose:
            print(f"\nEpisode {episode + 1}/{n_episodes}")
            print("-" * 50)

        while not done:
            # Random action
            action = env.action_space.sample()
            action_counts[action] += 1

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

            # Print occasional updates
            if verbose and step_count % 500 == 0:
                print(
                    f"  Step {step_count}: Bankroll=${env.bankroll:.2f}, "
                    f"Reward={episode_reward:.2f}"
                )

        # Get final statistics
        stats = env.get_bet_statistics()

        if verbose:
            print(f"\n  Episode Complete!")
            print(f"  Total Steps: {step_count}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Final Bankroll: ${stats['final_bankroll']:.2f}")
            print(f"  ROI: {stats['roi']:.2f}%")
            print(
                f"  Actions: Skip={action_counts[0]}, Home={action_counts[1]}, Away={action_counts[2]}"
            )
            print(f"  Bets Placed: {stats['total_bets']}")
            print(f"  Win Rate: {stats['win_rate']:.2f}%")

        all_results.append(
            {
                "episode": episode,
                "episode_reward": episode_reward,
                "steps": step_count,
                "action_counts": action_counts,
                **stats,
            }
        )

    return all_results


def print_summary(results):
    """Print summary statistics across episodes"""
    print("\n" + "=" * 70)
    print("RANDOM AGENT SUMMARY")
    print("=" * 70)

    avg_roi = np.mean([r["roi"] for r in results])
    avg_win_rate = np.mean([r["win_rate"] for r in results])
    avg_bets = np.mean([r["total_bets"] for r in results])
    avg_final = np.mean([r["final_bankroll"] for r in results])

    print(f"\nAveraged over {len(results)} episode(s):")
    print(f"  Average ROI:              {avg_roi:>8.2f}%")
    print(f"  Average Win Rate:         {avg_win_rate:>8.2f}%")
    print(f"  Average Bets:             {avg_bets:>8.0f}")
    print(f"  Average Final Bankroll:   ${avg_final:>8.2f}")

    print(f"\n📊 Expected Results for Random Agent:")
    print(f"  ROI: Should be around -5% to -10% (house edge)")
    print(f"  Win Rate: Should be around 45-50%")
    print(f"  Bets: Should be around 2/3 of total games")

    if avg_roi < -15:
        print(f"\n⚠️  ROI is very negative - environment might have issues")
    elif avg_roi > 0:
        print(f"\n⚠️  Positive ROI from random betting - suspicious!")
    else:
        print(f"\n✅ ROI looks reasonable for random betting")

    if avg_bets < 100:
        print(f"⚠️  Very few bets placed - check action space")
    else:
        print(f"✅ Betting frequency looks reasonable")


def test_environment_mechanics(env):
    """
    Test basic environment mechanics

    Checks:
    1. Reset works
    2. Step works for all actions
    3. Rewards are calculated
    4. Episode terminates
    """
    print("\n" + "=" * 70)
    print("ENVIRONMENT MECHANICS TEST")
    print("=" * 70)

    # Test reset
    print("\n1. Testing reset()...")
    obs, info = env.reset()
    print(f"   ✓ Reset works")
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial observation: {obs}")
    print(f"   Initial bankroll: ${env.bankroll:.2f}")

    # Test each action
    print("\n2. Testing each action...")

    actions = {0: "Skip", 1: "Bet Home", 2: "Bet Away"}

    for action, name in actions.items():
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"   ✓ Action {action} ({name}): reward={reward:.4f}, "
            f"bankroll=${env.bankroll:.2f}"
        )

    # Test full episode
    print("\n3. Testing full episode (10 steps)...")
    obs, info = env.reset()
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"   Episode terminated early at step {step}")
            break
    else:
        print(f"   ✓ Completed 10 steps")

    # Test statistics
    print("\n4. Testing get_bet_statistics()...")
    stats = env.get_bet_statistics()
    print(f"   ✓ Statistics calculated")
    print(f"   Keys: {list(stats.keys())}")

    print("\n✅ All environment mechanics working!")


def main():
    """Main testing function"""

    print("=" * 70)
    print("RANDOM AGENT - ENVIRONMENT DEBUGGING")
    print("=" * 70)

    # Load data
    project_root = Path(__file__).resolve().parent.parent.parent
    features_dir = project_root / "data" / "features"
    data_path = features_dir / "all_seasons_with_elo.csv"

    print(f"\nLoading data from: {data_path}")

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"   Please run: cd src/data/elo && python process_elo_features.py")
        return

    df = pd.read_csv(data_path)

    # Use a small subset for testing (2021 season only)
    test_df = df[df["season"] == 2021].copy()

    if "regular" in test_df.columns:
        test_df = test_df[test_df["regular"] == True].copy()

    print(f"✓ Loaded {len(test_df)} games from 2021 season")

    # Create environment
    print(f"\nCreating environment...")
    env = EloBettingEnv(test_df, initial_bankroll=500)
    print(f"✓ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")

    # Test mechanics
    test_environment_mechanics(env)

    # Run random agent
    print("\n" + "=" * 70)
    print("RUNNING RANDOM AGENT")
    print("=" * 70)

    results = random_agent(env, n_episodes=1, verbose=True)
    print_summary(results)

    # Detailed action analysis
    print("\n" + "=" * 70)
    print("ACTION ANALYSIS")
    print("=" * 70)

    result = results[0]
    total_actions = sum(result["action_counts"].values())

    print(f"\nAction Distribution:")
    for action, name in {0: "Skip", 1: "Bet Home", 2: "Bet Away"}.items():
        count = result["action_counts"][action]
        pct = count / total_actions * 100
        print(f"  {name:12s}: {count:>5} ({pct:>5.1f}%)")

    print(f"\nBetting Details:")
    print(f"  Total Games:     {total_actions}")
    print(f"  Bets Placed:     {result['total_bets']}")
    print(f"  Bets Won:        {int(result['total_bets'] * result['win_rate'] / 100)}")
    print(
        f"  Bets Lost:       {int(result['total_bets'] * (1 - result['win_rate'] / 100))}"
    )

    if result["total_bets"] > 0:
        print(
            f"  Avg Profit/Bet:  ${result['total_profit'] / result['total_bets']:.2f}"
        )

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)

    if result["roi"] < -15:
        print("\n❌ Random agent ROI is very negative")
        print("   → Environment might have issues")
        print("   → Check reward calculations in betting_env_elo.py")
    elif result["roi"] > 0:
        print("\n⚠️  Random agent has positive ROI (suspicious)")
        print("   → Check if rewards are correct")
        print("   → Verify odds calculations")
    else:
        print("\n✅ Random agent results look reasonable")
        print("   → Environment is working correctly")
        print("   → Safe to train DQN agent")
        print("\n   Run: python dqn_agent_simple.py")


if __name__ == "__main__":
    main()
