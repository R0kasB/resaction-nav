import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Expects the policy network (src/models/lstm.py PolicyLSTM) to produce
    (action_logits, value, hidden) from a flat observation vector.

    act(obs, hidden) -> (action_idx, log_prob, value, next_hidden)
        obs:         flat observation tensor from the environment
        hidden:      LSTM hidden state; pass None at episode start
        action_idx:  discrete action int to pass to env.step()
        log_prob:    log probability of the sampled action
        value:       critic's state value estimate
        next_hidden: updated LSTM hidden state

    store(obs, action, log_prob, reward, value, done)
        Buffers one transition. Call after every env.step().

    update() -> {"policy_loss", "value_loss", "entropy"}
        Computes advantages over the stored rollout, runs PPO epochs,
        clears the buffer. Call once per episode/rollout.
        Returns loss components averaged over epochs, for logging.

    Usage:
        agent = PPOAgent(policy=PolicyLSTM(...), lr=3e-4)
        action, log_prob, value, hidden = agent.act(obs, hidden)
        agent.store(obs, action, log_prob, reward, value, done)  # each step
        losses = agent.update()                                   # each episode end
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
    ):
        self.policy = policy
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.buffer: list[dict] = []

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

    def update(self) -> dict:
        """
        Compute advantages over the stored rollout, run PPO epochs, clear the buffer.
        Call once per episode/rollout — after all transitions have been stored.
        Returns a dict with loss components for logging.
        """
        trajectory = self.buffer
        self.buffer = []

        obs      = torch.stack([t["obs"]      for t in trajectory])
        actions  = torch.tensor([t["action"]   for t in trajectory], dtype=torch.long)
        old_lps  = torch.stack([t["log_prob"]  for t in trajectory]).detach()
        rewards  = [t["reward"] for t in trajectory]
        dones    = [t["done"]   for t in trajectory]
        old_vals = torch.stack([t["value"]     for t in trajectory]).detach()

        returns = self._compute_returns(rewards, dones, old_vals[-1])
        advantages = (returns - old_vals).detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = total_value_loss = total_entropy = 0.0

        for _ in range(self.epochs):
            # hidden state reset per epoch is acceptable for now;
            # full BPTT would require storing initial hidden state per episode
            logits, values, _ = self.policy(obs, hidden=None)
            dist = Categorical(logits=logits)
            new_lps = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_lps - old_lps).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)

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

    def _compute_returns(self, rewards, dones, last_value: torch.Tensor) -> torch.Tensor:
        returns = []
        R = last_value.item() if not dones[-1] else 0.0
        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - float(d))
            returns.append(R)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)
