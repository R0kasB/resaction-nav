"""Simulation package."""

from .PPO import PPO, PPOMetrics

__all__ = ["PPO", "PPOMetrics"]

try:
    from .thor_env import ThorEnv, RewardConfig

    __all__ += ["ThorEnv", "RewardConfig"]
except ModuleNotFoundError:
    # Allow importing PPO utilities without requiring AI2-THOR.
    pass
