"""
Random-sensing baseline.

PPO agent that injects SENSE randomly: with probability `sense_prob` the
agent SENSEs instead of following its policy, provided budget remains.
When the budget is exhausted, SENSE is masked out as usual.

The indices `sense_action_idx` and `budget_feature_idx` are injected at
construction time. They depend on the env's action_set (which determines
where SENSE sits in ACTION_LIST) and on the aux-feature layout (which
determines the offset of the budget scalar). We don't hardcode them so
that changing action_set or aux layout in YAML does not silently break
the baselines.

Config:
  agent:
    sense_prob: 0.2   # default
"""

import random
import torch
from torch.distributions import Categorical

from src.agents.ppo_agent import PPOAgent


class RandomSensingBaseline(PPOAgent):

    def __init__(
        self,
        *args,
        sense_action_idx: int,
        budget_feature_idx: int,
        sense_prob: float = 0.2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sense_action_idx < 0:
            raise ValueError(
                "RandomSensingBaseline requires SENSE in the action set; got "
                "action_list without SENSE. Use action_set with SENSE enabled."
            )
        self.sense_action_idx   = sense_action_idx
        self.budget_feature_idx = budget_feature_idx
        self.sense_prob         = sense_prob

    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)

        flat             = aux_features.squeeze(0) if aux_features.dim() == 2 else aux_features
        budget_available = flat[self.budget_feature_idx].item() > 0

        if budget_available and random.random() < self.sense_prob:
            action_idx = self.sense_action_idx
            dist       = Categorical(logits=logits)
            action_t   = torch.tensor(action_idx, device=logits.device)
            log_prob   = dist.log_prob(action_t).squeeze(-1)
        else:
            if not budget_available:
                logits = logits.clone()
                logits[..., self.sense_action_idx] = float("-inf")
            dist       = Categorical(logits=logits)
            action_t   = dist.sample()
            action_idx = action_t.item()
            log_prob   = dist.log_prob(action_t).squeeze(-1)

        return action_idx, log_prob, value.squeeze(), hidden