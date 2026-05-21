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
        obs:         observation tensor, shape (1, obs_dim)
        hidden:      LSTM hidden state; pass None at episode start
        action_idx:  discrete action int to pass to env.step()
        log_prob:    log probability of the sampled action (old policy, for PPO ratio)
        value:       critic's state value estimate (scalar tensor)
        next_hidden: updated LSTM hidden state

    store_initial_hidden(hidden)
        Save the hidden state before the first store() of each rollout.
        Must be called before any store() calls.

    store(obs, action, log_prob, reward, value, done)
        Buffers one transition. Call after every env.step().

    update(next_obs, final_hidden) -> {"policy_loss", "value_loss", "entropy"}
        Computes GAE advantages, runs PPO epochs with TBPTT chunking,
        clears the buffer. Call once per rollout.
        Returns loss components averaged over all gradient steps, for logging.

    Usage:
        agent = PPOAgent(policy=PolicyLSTM(...), lr=3e-4, total_updates=1000)
        hidden = None
        agent.store_initial_hidden(hidden)
        while not done:
            action, log_prob, value, hidden = agent.act(obs, hidden)
            agent.store(obs, action, log_prob, reward, value, done)
        losses = agent.update(obs, hidden)
    """

    def __init__(
        self,
        policy: nn.Module,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_clip_eps: float = 0.2,
        use_value_clip: bool = True,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
        chunk_size: int = 0,       # 0 = full BPTT; >0 = TBPTT segment length
        total_updates: int = 0,    # >0 enables linear LR decay to 0
    ):
        self.policy = policy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_clip_eps = value_clip_eps
        self.use_value_clip = use_value_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.chunk_size = chunk_size
        self.optimizer = optim.Adam(policy.parameters(), lr=lr)
        self.scheduler = (
            optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_updates
            )
            if total_updates > 0
            else None
        )
        self.buffer: list[dict] = []
        self.initial_hidden = None

    def act(self, image: torch.Tensor, aux_features: torch.Tensor,  hidden=None):
        """
        Sample an action from the policy.
        obs: (1, obs_dim)
        Returns: (action_idx, log_prob, value, next_hidden)
        """
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(-1), value.squeeze(), hidden

    def store_initial_hidden(self, hidden):
        """Call with the hidden state before the first store() of each rollout."""
        if self.buffer:
            raise RuntimeError("store_initial_hidden must be called before any store() calls for the rollout")
        if hidden is None:
            self.initial_hidden = None
        else:
            h, c = hidden
            self.initial_hidden = (h.detach(), c.detach())

    def store(self,
        image: torch.Tensor,
        aux_features: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool):
        """Buffer one transition. Call after every env.step()."""
        self.buffer.append({
            "image": image.detach(),
            "aux_features": aux_features.detach(),
            "action":   action,
            "log_prob": log_prob,
            "reward":   reward,
            "value":    value,
            "done":     done,
        })

    def update(self, next_image: torch.Tensor, next_aux_features: torch.Tensor, final_hidden=None) -> dict:
        """
        Compute advantages over the stored rollout, run PPO updates, and clear the buffer.

        The rollout stores raw images and auxiliary features separately because the policy
        is now responsible for encoding the image internally.
        """
        trajectory = self.buffer

        if not trajectory:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        images = torch.stack([t["image"] for t in trajectory])
        aux_features = torch.stack([t["aux_features"] for t in trajectory])

        old_lps = torch.stack([t["log_prob"] for t in trajectory]).detach()
        rewards = [t["reward"] for t in trajectory]
        dones = [t["done"] for t in trajectory]
        old_vals = torch.stack([t["value"] for t in trajectory]).detach()

        device = old_vals.device

        images = images.to(device)
        aux_features = aux_features.to(device)
        next_image = next_image.to(device)
        next_aux_features = next_aux_features.to(device)

        actions = torch.tensor([t["action"] for t in trajectory], dtype=torch.long, device=device,)

        with torch.no_grad():
            _, last_value, _ = self.policy(
                next_image,
                next_aux_features,
                hidden=final_hidden,
            )

        last_value = last_value.squeeze()

        returns, advantages = self._compute_gae(rewards=rewards, dones=dones, values=old_vals, last_value=last_value)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        T = images.shape[0]
        seg = self.chunk_size if self.chunk_size > 0 else T

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.epochs):
            # Hidden state propagated through the full sequence within this epoch.
            # initial_hidden was captured from the *old* policy at rollout start; we
            # bootstrap from None (or initial_hidden) and let dones reset it.
            h = self.initial_hidden

            self.optimizer.zero_grad()

            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_chunks = 0

            for chunk_start in range(0, T, seg):
                chunk_end = min(chunk_start + seg, T)
                sl = slice(chunk_start, chunk_end)

                # Detach h at chunk boundary so the graph for this chunk is bounded
                # by `seg` steps (this is the whole point of TBPTT).
                if h is not None:
                    hh, cc = h
                    h = (hh.detach(), cc.detach())

                all_logits = []
                all_values = []

                for i in range(chunk_start, chunk_end):
                    if i > 0 and dones[i - 1]:
                        h = None

                    logit, val, h = self.policy(
                        images[i],
                        aux_features[i],
                        hidden=h,
                    )
                    all_logits.append(logit.squeeze(0))
                    all_values.append(val.squeeze())

                logits = torch.stack(all_logits)
                v_pred = torch.stack(all_values)

                dist = Categorical(logits=logits)
                new_lps = dist.log_prob(actions[sl])
                entropy = dist.entropy().mean()

                ratio = (new_lps - old_lps[sl]).exp()
                surr1 = ratio * advantages[sl]
                surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages[sl]
                policy_loss = -torch.min(surr1, surr2).mean()

                if self.use_value_clip:
                    v_clipped = old_vals[sl] + (v_pred - old_vals[sl]).clamp(
                        -self.value_clip_eps, self.value_clip_eps,
                    )
                    value_loss = 0.5 * torch.max(
                        (v_pred - returns[sl]).pow(2),
                        (v_clipped - returns[sl]).pow(2),
                    ).mean()
                else:
                    value_loss = 0.5 * (v_pred - returns[sl]).pow(2).mean()

                # Weight each chunk's contribution by its length so the total
                # is the *mean over the whole rollout*, not the mean over chunks.
                weight = (chunk_end - chunk_start) / T
                chunk_loss = weight * (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                chunk_loss.backward()

                epoch_policy_loss += weight * policy_loss.item()
                epoch_value_loss  += weight * value_loss.item()
                epoch_entropy     += weight * entropy.item()
                epoch_chunks      += 1

            # One optimizer step per epoch, after gradients are accumulated
            # across all chunks of the rollout.
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += epoch_policy_loss
            total_value_loss  += epoch_value_loss
            total_entropy     += epoch_entropy
            n_updates         += 1

        if self.scheduler is not None:
            self.scheduler.step()

        self.initial_hidden = None
        self.buffer = []

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss":  total_value_loss  / max(n_updates, 1),
            "entropy":     total_entropy     / max(n_updates, 1),
        }
    
    def _compute_gae(
        self, rewards, dones, values: torch.Tensor, last_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T          = len(rewards)
        advantages = torch.zeros(T, dtype=torch.float32, device=values.device)
        gae        = 0.0
        next_val   = last_value.item() if not dones[-1] else 0.0
        for t in reversed(range(T)):
            d        = float(dones[t])
            delta    = rewards[t] + self.gamma * next_val * (1 - d) - values[t].item()
            gae      = delta + self.gamma * self.gae_lambda * (1 - d) * gae
            advantages[t] = gae
            next_val = values[t].item()
        returns = advantages + values
        return returns, advantages
