"""
Random-sensing baseline.
 
PPO agent that injects SENSE randomly: with probability `sense_prob` the
agent SENSEs instead of following its policy, provided budget remains.
When the budget is exhausted, SENSE is masked out as usual.
 
Config:
  agent:
    sense_prob: 0.2   # default
"""
 
import random
 
import torch
from torch.distributions import Categorical
 
from src.agents.ppo_agent import PPOAgent
 
# Position of SENSE in ACTION_LIST (thor_env.py):
SENSE_IDX  = 8
 
# Indices in the aux_features vector (thor_env.get_aux_features):
#   [0:3]  gps (x, y, z)
#   [3:5]  compass (yaw, horizon)
#   [5:15] prev_action one-hot (10 actions)
#   [15]   resolution_level  (current_downgrade / base_downgrade)
#   [16]   sensing_budget    (remaining / max)
BUDGET_IDX = 16
 
 
class RandomSensingBaseline(PPOAgent):
 
    def __init__(self, *args, sense_prob: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sense_prob = sense_prob
 
    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)
 
        flat             = aux_features.squeeze(0) if aux_features.dim() == 2 else aux_features
        budget_available = flat[BUDGET_IDX].item() > 0
 
        if budget_available and random.random() < self.sense_prob:
            # Random SENSE injection
            action_idx = SENSE_IDX
            dist       = Categorical(logits=logits)
            action_t   = torch.tensor(action_idx, device=logits.device)
            log_prob   = dist.log_prob(action_t).squeeze(-1)
        else:
            # Normal policy; mask SENSE if no budget left
            if not budget_available:
                logits = logits.clone()
                logits[..., SENSE_IDX] = float("-inf")
            dist       = Categorical(logits=logits)
            action_t   = dist.sample()
            action_idx = action_t.item()
            log_prob   = dist.log_prob(action_t).squeeze(-1)
 
        return action_idx, log_prob, value.squeeze(), hidden
