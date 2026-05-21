"""
Low-resolution baseline.

PPO agent that always operates at the worst resolution (base_downgrade).
SENSE is permanently masked out — the agent never improves its resolution.

Only `sense_action_idx` is injected; budget tracking is irrelevant since
SENSE is never taken.
"""

import torch
from torch.distributions import Categorical

from src.agents.ppo_agent import PPOAgent


class LowResBaseline(PPOAgent):

    def __init__(self, *args, sense_action_idx: int, **kwargs):
        super().__init__(*args, **kwargs)
        # sense_action_idx may be -1 if action_set has no SENSE (e.g. navigation).
        # In that case there is nothing to mask, which is fine.
        self.sense_action_idx = sense_action_idx

    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)

        if self.sense_action_idx >= 0:
            logits = logits.clone()
            logits[..., self.sense_action_idx] = float("-inf")

        dist       = Categorical(logits=logits)
        action_t   = dist.sample()
        action_idx = action_t.item()
        log_prob   = dist.log_prob(action_t).squeeze(-1)

        return action_idx, log_prob, value.squeeze(), hidden