"""
High-resolution baseline.
 
PPO agent that always forces SENSE when resolution is not yet maximal and
budget remains, simulating an agent that observes at full resolution whenever
possible. When already at max res (or budget exhausted), SENSE is masked out
and the policy acts freely.
 
aux_features layout (see thor_env.get_aux_features):
  [0:3]  gps
  [3:5]  compass
  [5:15] prev_action one-hot  (n_actions = 10)
  [15]   resolution_level     (current_downgrade / base_downgrade)
  [16]   sensing_budget       (remaining / max)
"""
 
import torch
from torch.distributions import Categorical
 
from src.agents.ppo_agent import PPOAgent
 
SENSE_IDX  = 8
 
# Indices in the aux_features vector (thor_env.get_aux_features):
#   [0:3]  gps (x, y, z)
#   [3:5]  compass (yaw, horizon)
#   [5:15] prev_action one-hot (10 actions)
#   [15]   resolution_level  (current_downgrade / base_downgrade)
#   [16]   sensing_budget    (remaining / max)
RES_IDX    = 15
BUDGET_IDX = 16

 
 
class HighResBaseline(PPOAgent):
 
    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)
 
        flat = aux_features.squeeze(0) if aux_features.dim() == 2 else aux_features
        resolution_level = flat[RES_IDX].item()    # 0 = full res, 1 = worst
        sensing_budget   = flat[BUDGET_IDX].item()
 
        if resolution_level > 0 and sensing_budget > 0:
            # Force SENSE; compute log_prob under the current distribution for PPO consistency
            action_idx = SENSE_IDX
            dist       = Categorical(logits=logits)
            action_t   = torch.tensor(action_idx, device=logits.device)
            log_prob   = dist.log_prob(action_t).squeeze(-1)
        else:
            # Already at max res or budget gone: mask SENSE and let the policy choose
            logits = logits.clone()
            logits[..., SENSE_IDX] = float("-inf")
            dist       = Categorical(logits=logits)
            action_t   = dist.sample()
            action_idx = action_t.item()
            log_prob   = dist.log_prob(action_t).squeeze(-1)
 
        return action_idx, log_prob, value.squeeze(), hidden
 
