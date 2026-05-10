"""
Low-resolution baseline.

PPO agent that never uses SENSE: the SENSE logit is masked to -inf at
inference time, so the agent always navigates at base_downgrade resolution.
"""

import torch
from torch.distributions import Categorical

from src.agents.ppo_agent import PPOAgent

SENSE_IDX = 8  # position of SENSE in ACTION_LIST


class LowResBaseline(PPOAgent):

    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)

        # Mask SENSE so it is never sampled
        logits = logits.clone()
        logits[..., SENSE_IDX] = float("-inf")

        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(-1), value.squeeze(), hidden
