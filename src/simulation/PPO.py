from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical


@dataclass
class PPOMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


class PPO:
    """
    PPO trainer operating directly on rollout trajectories.

    Expected transition keys in each trajectory item:
        - obs: torch.Tensor (state at time t)
        - action: int or action name (str)
        - reward: float
        - done: bool

    Optional keys:
        - next_obs: torch.Tensor (used for bootstrap when final done=False)
        - log_prob: torch.Tensor (old-policy log probability)
        - value: torch.Tensor (old value estimate)

    The policy should implement:
        policy(obs_batch) -> (logits, values) or (logits, values, hidden)
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        minibatch_size: int = 64,
        normalize_advantages: bool = True,
        device: str | torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.normalize_advantages = normalize_advantages

        if device is None:
            try:
                first_param = next(policy.parameters())
                self.device = first_param.device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _policy_forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.policy(obs)
        if not isinstance(out, tuple) or len(out) < 2:
            raise TypeError(
                "Policy forward must return (logits, values) or (logits, values, hidden)."
            )
        logits = out[0]
        values = out[1]
        if values.dim() > 1 and values.shape[-1] == 1:
            values = values.squeeze(-1)
        return logits, values

    def _action_to_id(self, action: Any, action2id: dict[str, int] | None) -> int:
        if isinstance(action, int):
            return action
        if torch.is_tensor(action):
            return int(action.item())
        if isinstance(action, str):
            if action2id is None:
                raise ValueError(
                    "Trajectory contains string actions but action2id mapping was not provided."
                )
            if action not in action2id:
                raise KeyError(f"Unknown action '{action}' in trajectory.")
            return action2id[action]
        raise TypeError(f"Unsupported action type: {type(action)!r}")

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=rewards.device, dtype=rewards.dtype)

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        returns = advantages + values
        return returns, advantages

    def loss(
        self,
        logits: torch.Tensor,
        values: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = (new_log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_pred_clipped = old_values + (values - old_values).clamp(
            -self.value_clip_eps, self.value_clip_eps
        )
        value_losses = (values - returns).pow(2)
        value_losses_clipped = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        return total_loss, policy_loss, value_loss, entropy

    def update(
        self,
        trajectory: list[dict[str, Any]],
        action2id: dict[str, int] | None = None,
        next_obs: torch.Tensor | None = None,
    ) -> PPOMetrics:
        if not trajectory:
            return PPOMetrics(0.0, 0.0, 0.0, 0.0)

        def _get_item(step: dict[str, Any], *names: str) -> Any:
            for name in names:
                if name in step:
                    return step[name]
            joined = ", ".join(names)
            raise KeyError(f"Missing required trajectory key. Expected one of: {joined}")

        obs = torch.stack([t["obs"] for t in trajectory]).to(self.device)
        rewards = torch.tensor(
            [float(t["reward"]) for t in trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [float(_get_item(t, "done", "is_model_done")) for t in trajectory],
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [self._action_to_id(_get_item(t, "action", "Action"), action2id) for t in trajectory],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            if "log_prob" in trajectory[0] and "value" in trajectory[0]:
                old_log_probs = torch.stack([t["log_prob"] for t in trajectory]).to(self.device)
                old_values = torch.stack([t["value"] for t in trajectory]).to(self.device).squeeze(-1)
            else:
                old_logits, old_values = self._policy_forward(obs)
                old_dist = Categorical(logits=old_logits)
                old_log_probs = old_dist.log_prob(actions)

            if bool(dones[-1].item()):
                bootstrap_value = torch.zeros(1, device=self.device, dtype=torch.float32)
            else:
                if next_obs is None:
                    if "next_obs" in trajectory[-1]:
                        next_obs = trajectory[-1]["next_obs"]
                    elif "nex_obs" in trajectory[-1]:
                        next_obs = trajectory[-1]["nex_obs"]
                    else:
                        raise ValueError(
                            "Missing bootstrap state: pass next_obs or include next_obs/nex_obs in trajectory[-1]."
                        )
                next_obs_tensor = next_obs.to(self.device).unsqueeze(0)
                _, next_value = self._policy_forward(next_obs_tensor)
                bootstrap_value = next_value.squeeze(-1)

            returns, advantages = self._compute_gae(
                rewards=rewards,
                dones=dones,
                values=old_values,
                next_value=bootstrap_value,
            )
            if self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.shape[0]
        mini_size = min(self.minibatch_size, batch_size)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_size):
                mb_idx = permutation[start : start + mini_size]

                logits, values = self._policy_forward(obs[mb_idx])
                loss, policy_loss, value_loss, entropy = self.loss(
                    logits=logits,
                    values=values,
                    actions=actions[mb_idx],
                    old_log_probs=old_log_probs[mb_idx],
                    old_values=old_values[mb_idx],
                    returns=returns[mb_idx],
                    advantages=advantages[mb_idx],
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += float(loss.item())
                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                n_updates += 1

        if n_updates == 0:
            return PPOMetrics(0.0, 0.0, 0.0, 0.0)

        return PPOMetrics(
            policy_loss=total_policy_loss / n_updates,
            value_loss=total_value_loss / n_updates,
            entropy=total_entropy / n_updates,
            total_loss=total_loss / n_updates,
        )
