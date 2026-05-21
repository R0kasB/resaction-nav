"""
Fixed-schedule baseline.

PPO agent that forces SENSE every `sense_interval` steps (default 10),
provided budget remains. Between forced-sense steps SENSE is masked out
so the policy cannot accidentally use it and double-spend the budget.

The indices `sense_action_idx` and `budget_feature_idx` are injected at
construction time — same rationale as RandomSensingBaseline.

Config:
  agent:
    sense_interval: 10   # default
"""

import torch
from torch.distributions import Categorical

from src.agents.ppo_agent import PPOAgent


class FixedScheduleBaseline(PPOAgent):

    def __init__(
        self,
        *args,
        sense_action_idx: int,
        budget_feature_idx: int,
        sense_interval: int = 10,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sense_action_idx < 0:
            raise ValueError(
                "FixedScheduleBaseline requires SENSE in the action set; got "
                "action_list without SENSE. Use action_set with SENSE enabled."
            )
        self.sense_action_idx   = sense_action_idx
        self.budget_feature_idx = budget_feature_idx
        self.sense_interval     = sense_interval
        self._step              = 0

    def store_initial_hidden(self, hidden):
        # Reset step counter at the start of each episode.
        super().store_initial_hidden(hidden)
        self._step = 0

    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)

        flat             = aux_features.squeeze(0) if aux_features.dim() == 2 else aux_features
        budget_available = flat[self.budget_feature_idx].item() > 0

        force_sense = (self._step % self.sense_interval == 0) and budget_available
        self._step += 1

        if force_sense:
            action_idx = self.sense_action_idx
            dist       = Categorical(logits=logits)
            action_t   = torch.tensor(action_idx, device=logits.device)
            log_prob   = dist.log_prob(action_t).squeeze(-1)
        else:
            # Mask SENSE between scheduled steps.
            logits = logits.clone()
            logits[..., self.sense_action_idx] = float("-inf")
            dist       = Categorical(logits=logits)
            action_t   = dist.sample()
            action_idx = action_t.item()
            log_prob   = dist.log_prob(action_t).squeeze(-1)

        return action_idx, log_prob, value.squeeze(), hidden