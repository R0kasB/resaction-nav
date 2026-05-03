import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    """
    Proximal Policy Optimization agent with LSTM policy, GAE, and value clipping.

    Expects the policy network (src/models/lstm.py PolicyLSTM) to produce
    (action_logits, value, hidden) from a flat observation vector.

    act(obs, hidden) -> (action_idx, log_prob, value, next_hidden)
        obs:         flat observation tensor from the environment
        hidden:      LSTM hidden state; pass None at episode start
        action_idx:  discrete action int to pass to env.step()
        log_prob:    log probability of the sampled action (old policy, for PPO ratio)
        value:       critic's state value estimate
        next_hidden: updated LSTM hidden state

    store_initial_hidden(hidden)
        Save the hidden state at rollout start so update() can replay the
        sequence from the same LSTM context across all PPO epochs.

    store(obs, action, log_prob, reward, value, done)
        Buffers one transition. Call after every env.step().

    update() -> {"policy_loss", "value_loss", "entropy"}
        Computes GAE advantages, runs PPO epochs with policy ratio clipping
        and value clipping, clears the buffer. Call once per episode/rollout.
        Returns loss components averaged over epochs, for logging.

    Usage:
        agent = PPOAgent(policy=PolicyLSTM(...), lr=3e-4)
        hidden = None
        agent.store_initial_hidden(hidden)                        # before episode loop
        action, log_prob, value, hidden = agent.act(obs, hidden)
        agent.store(obs, action, log_prob, reward, value, done)  # each step
        losses = agent.update(next_obs, hidden)                  # each episode end
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.5,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.buffer: list[dict] = []
        self.initial_hidden = None

    def act(self, obs: torch.Tensor, hidden=None):
        """
        Sample an action from the policy.
        Returns: (action_idx, log_prob, value, next_hidden)
        """
        with torch.no_grad():
            logits, value, hidden = self.policy(obs, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(), value.squeeze(-1), hidden

    def store_initial_hidden(self, hidden):
        """Call with the hidden state at the start of each rollout."""
        self.initial_hidden = hidden

    def store(
        self,
        obs: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        """Buffer one transition. Call after every env.step()."""
        self.buffer.append({
            "obs":      obs,
            "action":   action,
            "log_prob": log_prob,
            "reward":   reward,
            "value":    value,
            "done":     done,
        })

    def update(self, next_obs: torch.Tensor, final_hidden=None) -> dict:
        """
        Compute advantages over the stored rollout, run PPO epochs, clear the buffer.
        next_obs:     first observation after the rollout ends (for bootstrapping V(s_{T+1}))
        final_hidden: LSTM hidden state after the last act() call
        Call once per episode/rollout — after all transitions have been stored.
        Returns a dict with loss components for logging.
        """
        trajectory = self.buffer
        self.buffer = []

        obs      = torch.stack([t["obs"]      for t in trajectory])
        old_lps  = torch.stack([t["log_prob"]  for t in trajectory]).detach()
        rewards  = [t["reward"] for t in trajectory]
        dones    = [t["done"]   for t in trajectory]
        old_vals = torch.stack([t["value"]     for t in trajectory]).detach()

        device  = old_vals.device
        obs     = obs.to(device)
        actions = torch.tensor([t["action"] for t in trajectory], dtype=torch.long, device=device)

        with torch.no_grad():
            _, last_value, _ = self.policy(next_obs.to(device), hidden=final_hidden)
        last_value = last_value.squeeze()

        returns, advantages = self._compute_gae(rewards, dones, old_vals, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = total_value_loss = total_entropy = 0.0

        for _ in range(self.epochs):
            logits, values, _ = self.policy(obs.unsqueeze(0), hidden=self.initial_hidden)
            logits = logits.squeeze(0)  # (T, n_actions)
            values = values.squeeze(0)  # (T, 1)
            dist = Categorical(logits=logits)
            new_lps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_lps - old_lps).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            v_pred = values.squeeze(-1)
            v_clipped = old_vals + (v_pred - old_vals).clamp(-self.value_clip_eps, self.value_clip_eps)
            value_loss = torch.max((v_pred - returns).pow(2), (v_clipped - returns).pow(2)).mean()

            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss  += value_loss.item()
            total_entropy     += entropy.item()

        n = self.epochs
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss":  total_value_loss  / n,
            "entropy":     total_entropy      / n,
        }

    def _compute_gae(
        self, rewards, dones, values: torch.Tensor, last_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = []
        gae = 0.0
        next_val = last_value.item() if not dones[-1] else 0.0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(values.tolist())):
            delta = r + self.gamma * next_val * (1 - float(d)) - v
            gae = delta + self.gamma * self.gae_lambda * (1 - float(d)) * gae
            advantages.append(gae)
            next_val = v
        advantages.reverse()
        advantages = torch.tensor(advantages, dtype=torch.float32).to(last_value.device)
        returns = advantages + values
        return returns, advantages
