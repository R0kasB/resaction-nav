"""
Fixed-schedule baseline.
 
PPO agent that forces SENSE every `sense_interval` steps (default 10),
provided budget remains. Between forced-sense steps SENSE is masked out
so the policy cannot accidentally use it and double-spend the budget.
 
Config:
  agent:
    sense_interval: 10   # default
"""
 
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
 
 
class FixedScheduleBaseline(PPOAgent):
 
    def __init__(self, *args, sense_interval: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.sense_interval = sense_interval
        self._step = 0  # step counter within the current episode
 
    def store_initial_hidden(self, hidden):
        # Reset the step counter at the start of each episode
        super().store_initial_hidden(hidden)
        self._step = 0
 
    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)
 
        flat             = aux_features.squeeze(0) if aux_features.dim() == 2 else aux_features
        budget_available = flat[BUDGET_IDX].item() > 0
 
        force_sense = (self._step % self.sense_interval == 0) and budget_available
        self._step += 1
 
        if force_sense:
            action_idx = SENSE_IDX
            dist       = Categorical(logits=logits)
            action_t   = torch.tensor(action_idx, device=logits.device)
            log_prob   = dist.log_prob(action_t).squeeze(-1)
        else:
            # Mask SENSE between scheduled steps
            logits = logits.clone()
            logits[..., SENSE_IDX] = float("-inf")
            dist       = Categorical(logits=logits)
            action_t   = dist.sample()
            action_idx = action_t.item()
            log_prob   = dist.log_prob(action_t).squeeze(-1)
 
        return action_idx, log_prob, value.squeeze(), hidden
