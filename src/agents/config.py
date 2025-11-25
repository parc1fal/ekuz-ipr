"""
Training Configuration for DQN Betting Agent

Adjust these parameters to tune model performance
"""

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================

DATA_SPLIT = {
    "train_years": (2008, 2018),  # 11 seasons for training
    "val_years": (2019, 2020),  # 2 seasons for validation
    "test_years": (2021, 2023),  # 4 seasons for final evaluation
    "use_regular_season_only": True,  # Exclude playoffs
}

# ============================================================================
# DQN HYPERPARAMETERS
# ============================================================================

DQN_CONFIG = {
    # Training duration
    "total_timesteps": 500_000,  # 500k steps ≈ 40 full training seasons
    # Increase to 1M for longer training
    # Learning rate
    "learning_rate": 1e-4,  # Adam optimizer learning rate
    # Lower (1e-5) = slower but more stable
    # Higher (1e-3) = faster but less stable
    # Experience replay
    "buffer_size": 100_000,  # How many experiences to store
    # Larger = better sample diversity, more memory
    "batch_size": 32,  # Minibatch size for training
    # Common values: 32, 64, 128
    "learning_starts": 1000,  # Start training after N random steps
    # Update frequencies
    "train_freq": 4,  # Train every N steps
    "gradient_steps": 1,  # Gradient updates per training step
    "target_update_interval": 1000,  # Update target network every N steps
    # Discount factor
    "gamma": 0.99,  # Discount factor for future rewards
    # 0.99 = long-term planning (recommended)
    # 0.95 = more short-term
    # Exploration (epsilon-greedy)
    "exploration_fraction": 0.3,  # Decay epsilon over 30% of training
    "exploration_initial_eps": 1.0,  # Start with 100% random
    "exploration_final_eps": 0.01,  # End with 1% random
    # Neural network architecture
    "policy_kwargs": {
        "net_arch": [256, 256]  # 2 hidden layers, 256 units each
        # Alternatives: [128, 128], [256, 256, 128]
    },
    # Misc
    "seed": 42,
    "verbose": 1,
}

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

ENV_CONFIG = {
    "initial_bankroll": 500.0,  # Starting bankroll per episode
    "bet_size": 1.0,  # Fixed bet size (for now)
}

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    "eval_freq": 10_000,  # Evaluate every N timesteps
    "n_eval_episodes": 1,  # Episodes per evaluation
    "deterministic": True,  # Use deterministic policy for eval
}

# ============================================================================
# CHECKPOINT CONFIGURATION
# ============================================================================

CHECKPOINT_CONFIG = {
    "save_freq": 50_000,  # Save checkpoint every N timesteps
    "keep_best_only": True,  # Only keep best model based on val performance
}

# ============================================================================
# INTERPRETATION THRESHOLDS
# ============================================================================

INTERPRETATION = {
    "breakeven_win_rate": 52.4,  # Need >52.4% to beat -110 vig
    "strong_roi_threshold": 5.0,  # ROI > 5% is strong
    "conservative_bet_freq": 10.0,  # <10% bet frequency is very conservative
    "active_bet_freq": 60.0,  # >60% bet frequency is very active
}

# ============================================================================
# BASELINE STRATEGIES
# ============================================================================

BASELINES = [
    "random",  # Random betting (33% skip, 33% home, 33% away)
    "favorites",  # Always bet on favorites
    "underdogs",  # Always bet on underdogs
]

# ============================================================================
# TRAINING NOTES
# ============================================================================

"""
RECOMMENDED CONFIGURATIONS:

1. QUICK TEST (5 minutes):
   - total_timesteps: 50_000
   - Use to verify everything works

2. BASELINE (30 minutes):
   - total_timesteps: 500_000
   - Good starting point

3. EXTENDED (2 hours):
   - total_timesteps: 1_000_000
   - For better convergence

4. THOROUGH (4+ hours):
   - total_timesteps: 2_000_000+
   - For publication-quality results


TYPICAL RESULTS TO EXPECT:

With 500k timesteps on 2008-2018 training data:
- Random strategy:    ~-5% ROI (house edge)
- Favorites strategy: ~-2% ROI (slight edge)
- Underdogs strategy: ~-8% ROI (worse)
- DQN (uninformative features): ~-3% ROI (learned to skip most games)
- DQN (with ELO features): 5-15% ROI (if it finds patterns)

Stanford CS224R achieved 34.7% ROI, but that's exceptional.
Anything > 5% ROI is very good for sports betting.


HYPERPARAMETER TUNING GUIDANCE:

If agent is too conservative (betting <5% of games):
- Increase exploration_final_eps to 0.05
- Decrease gamma to 0.95 (focus on immediate rewards)

If agent is too aggressive (betting >80% of games):
- Increase learning_rate to 1e-3
- Add penalty for bankruptcy in reward function

If training is unstable (reward bouncing around):
- Decrease learning_rate to 1e-5
- Increase batch_size to 64 or 128

If agent isn't learning:
- Increase total_timesteps to 1M+
- Check that ELO features are present
- Verify data is sorted chronologically
"""
