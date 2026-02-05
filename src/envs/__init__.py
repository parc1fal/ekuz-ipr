from src.envs.simple_env import SimpleBettingEnv
from src.envs.betting_env_elo import EloBettingEnv
from src.envs.betting_env_continuous import ContinuousBettingEnv

ENV_REGISTRY = {
    "simple": SimpleBettingEnv,
    "elo": EloBettingEnv,
    "elo_continuous": ContinuousBettingEnv,
}


def get_env_class(name: str):
    """Look up an env class by its registry key."""
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"Unknown environment '{name}'. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name]
