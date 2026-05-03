"""Simulation package."""

__all__ = []

try:
    from .thor_env import ThorEnv, RewardConfig

    __all__ = ["ThorEnv", "RewardConfig"]
except ModuleNotFoundError:
    # Allow importing the package in environments without AI2-THOR.
    pass
