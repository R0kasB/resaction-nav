"""
High-resolution baseline.

This agent never uses SENSE. The environment must provide full-resolution
observations directly via fixed_high_res=True.

Modes:
- poc: simplified setup, usually minimal action space.
- baseline: final high-res baseline, usually full navigation action space.
"""

from src.agents.ppo_agent import PPOAgent


class HighResBaseline(PPOAgent):
    VALID_MODES = {"poc", "baseline"}

    def __init__(self, *args, mode: str = "baseline", **kwargs):
        super().__init__(*args, **kwargs)

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid HighResBaseline mode={mode!r}. "
                f"Choose from {sorted(self.VALID_MODES)}."
            )

        self.mode = mode

    def act(self, image, aux_features, hidden=None):
        return super().act(image, aux_features, hidden)