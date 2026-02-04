from src.envs.simple_env import SimpleBettingEnv
from src.envs.betting_env_elo import EloBettingEnv

ENV_REGISTRY = {
    "simple": SimpleBettingEnv,
    "elo": EloBettingEnv,
}


def get_env_class(name: str):
    """Look up an env class by its registry key."""
    if name not in ENV_REGISTRY:
        raise KeyError(
            f"Unknown environment '{name}'. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    return ENV_REGISTRY[name]
