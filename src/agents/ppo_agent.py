import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PPOAgent:
    """
    Proximal Policy Optimization agent with LSTM policy, GAE, and value clipping.

    Two update paths:

      - update(next_image, next_aux_features, final_hidden)
          Legacy single-rollout update. Computes GAE for the buffered trajectory,
          runs `epochs` PPO passes over it, clears the buffer, steps the scheduler.

      - update_from_rollouts(rollouts)
          Multi-rollout update. Takes a list of fully-formed rollouts (see the
          rollout dict schema below), computes per-rollout GAE, concatenates
          everything into one batch, then runs `epochs` PPO passes that iterate
          over rollouts sequentially (preserving per-rollout LSTM hidden flow)
          but accumulate gradients into a *single* optimizer.step() per epoch.
          Steps the scheduler exactly once. This is the correct way when
          collecting `num_parallel_envs` rollouts per iteration.

    Rollout dict schema (for update_from_rollouts):
        {
          "images":          (T, 3, H, W),
          "aux_features":    (T, D_aux),
          "actions":         (T,)   long,
          "log_probs":       (T,)   float (old policy),
          "rewards":         list[float] length T,
          "values":          (T,)   float (old critic),
          "dones":           list[bool] length T,
          "initial_hidden":  None or (h, c) tensors of shape (num_layers, 1, hidden_dim),
          "next_image":      (3, H, W) or (1, 3, H, W),
          "next_aux_features": (D_aux,) or (1, D_aux),
          "final_hidden":    None or (h, c),
        }
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
        chunk_size: int = 0,
        total_updates: int = 0,
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


    def act(self, image: torch.Tensor, aux_features: torch.Tensor, hidden=None):
        with torch.no_grad():
            logits, value, hidden = self.policy(image, aux_features, hidden)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).squeeze(-1), value.squeeze(), hidden


    def store_initial_hidden(self, hidden):
        if self.buffer:
            raise RuntimeError(
                "store_initial_hidden must be called before any store() calls for the rollout"
            )
        if hidden is None:
            self.initial_hidden = None
        else:
            h, c = hidden
            self.initial_hidden = (h.detach(), c.detach())

    def store(
        self,
        image: torch.Tensor,
        aux_features: torch.Tensor,
        action: int,
        log_prob: torch.Tensor,
        reward: float,
        value: torch.Tensor,
        done: bool,
    ):
        self.buffer.append({
            "image":        image.detach(),
            "aux_features": aux_features.detach(),
            "action":       action,
            "log_prob":     log_prob,
            "reward":       reward,
            "value":        value,
            "done":         done,
        })

    def update(self, next_image: torch.Tensor, next_aux_features: torch.Tensor, final_hidden=None) -> dict:
        """Legacy single-rollout update. Implemented as a 1-rollout call to
        update_from_rollouts so behaviour stays identical."""
        trajectory = self.buffer
        if not trajectory:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        rollout = self._buffer_to_rollout(
            trajectory,
            initial_hidden=self.initial_hidden,
            next_image=next_image,
            next_aux_features=next_aux_features,
            final_hidden=final_hidden,
        )

        self.buffer = []
        self.initial_hidden = None

        return self.update_from_rollouts([rollout])

    def _buffer_to_rollout(
        self,
        trajectory: list[dict],
        initial_hidden,
        next_image: torch.Tensor,
        next_aux_features: torch.Tensor,
        final_hidden,
    ) -> dict:
        images = torch.stack([t["image"] for t in trajectory])
        aux_features = torch.stack([t["aux_features"] for t in trajectory])
        log_probs = torch.stack([t["log_prob"] for t in trajectory]).detach()
        values = torch.stack([t["value"] for t in trajectory]).detach()
        actions = torch.tensor(
            [t["action"] for t in trajectory], dtype=torch.long, device=values.device,
        )
        rewards = [float(t["reward"]) for t in trajectory]
        dones = [bool(t["done"]) for t in trajectory]
        return {
            "images":            images,
            "aux_features":      aux_features,
            "actions":           actions,
            "log_probs":         log_probs,
            "rewards":           rewards,
            "values":            values,
            "dones":             dones,
            "initial_hidden":    initial_hidden,
            "next_image":        next_image,
            "next_aux_features": next_aux_features,
            "final_hidden":      final_hidden,
        }


    def update_from_rollouts(self, rollouts: list[dict]) -> dict:
        """Run PPO on a batch of rollouts.

        Implementation notes:
        - GAE is computed per-rollout (since advantages depend on the within-rollout
          done sequence and bootstrap value).
        - Advantages are normalised *jointly* across the concatenated batch — this
          is the standard PPO behaviour and reduces variance vs per-rollout norm.
        - Each epoch iterates over rollouts sequentially (the LSTM hidden flow
          must be preserved within a rollout) but calls backward() per chunk and
          a single optimizer.step() at the end of the epoch.
        - One scheduler.step() per call to update_from_rollouts(), not per rollout.
        """
        if not rollouts:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        device = rollouts[0]["values"].device

        all_returns = []
        all_advantages = []
        rollout_lengths = []

        for ro in rollouts:
            values = ro["values"].to(device)
            rewards = ro["rewards"]
            dones = ro["dones"]
            T = len(rewards)
            rollout_lengths.append(T)

            with torch.no_grad():
                _, last_value, _ = self.policy(
                    ro["next_image"].to(device),
                    ro["next_aux_features"].to(device),
                    hidden=ro["final_hidden"],
                )
            last_value = last_value.squeeze()

            returns, advantages = self._compute_gae(
                rewards=rewards, dones=dones, values=values, last_value=last_value,
            )
            all_returns.append(returns)
            all_advantages.append(advantages)
         
        with torch.no_grad():
            flat_returns = torch.cat(all_returns, dim=0)
            flat_values = torch.cat(
                [ro["values"].to(device) for ro in rollouts], dim=0
            )
            var_returns = flat_returns.var(unbiased=False)
            explained_var = (
                1.0 - (flat_returns - flat_values).var(unbiased=False) / (var_returns + 1e-8)
            ).item()

        flat_adv = torch.cat(all_advantages, dim=0)
        if flat_adv.numel() > 1:
            adv_mean = flat_adv.mean()
            adv_std = flat_adv.std(unbiased=False) + 1e-8
            normalised = [(a - adv_mean) / adv_std for a in all_advantages]
        else:
            normalised = all_advantages
        all_advantages = normalised

        total_T = sum(rollout_lengths)
        if total_T == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0

            for ro_idx, ro in enumerate(rollouts):
                T = rollout_lengths[ro_idx]
                seg = self.chunk_size if self.chunk_size > 0 else T

                images = ro["images"].to(device)
                aux_features = ro["aux_features"].to(device)
                actions = ro["actions"].to(device)
                old_lps = ro["log_probs"].to(device)
                old_vals = ro["values"].to(device)
                returns = all_returns[ro_idx].to(device)
                advantages = all_advantages[ro_idx].to(device)
                dones = ro["dones"]
                h = ro["initial_hidden"]

                # Per-rollout weight = T_i / total_T so the epoch loss is the
                # mean over *all* transitions in the batch.
                rollout_weight = T / total_T

                for chunk_start in range(0, T, seg):
                    chunk_end = min(chunk_start + seg, T)
                    sl = slice(chunk_start, chunk_end)

                    if h is not None:
                        hh, cc = h
                        h = (hh.detach(), cc.detach())

                    all_logits = []
                    all_values = []

                    for i in range(chunk_start, chunk_end):
                        if i > 0 and dones[i - 1]:
                            h = None

                        logit, val, h = self.policy(
                            images[i], aux_features[i], hidden=h,
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

                    chunk_weight = (chunk_end - chunk_start) / T
                    chunk_loss = rollout_weight * chunk_weight * (
                        policy_loss
                        + self.value_coef * value_loss
                        - self.entropy_coef * entropy
                    )
                    chunk_loss.backward()

                    total_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), max_norm=float('inf')
                    ).item()
                    # log it

                    epoch_policy_loss += rollout_weight * chunk_weight * policy_loss.item()
                    epoch_value_loss  += rollout_weight * chunk_weight * value_loss.item()
                    epoch_entropy     += rollout_weight * chunk_weight * entropy.item()

            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += epoch_policy_loss
            total_value_loss  += epoch_value_loss
            total_entropy     += epoch_entropy

        # One scheduler step per *batch* update, regardless of rollout count.
        if self.scheduler is not None:
            self.scheduler.step()

        return {
            "policy_loss": total_policy_loss / max(self.epochs, 1),
            "value_loss":  total_value_loss  / max(self.epochs, 1),
            "entropy":     total_entropy     / max(self.epochs, 1),
            "R^2 | explained_variance": explained_var,
            "returns_var": var_returns.item(),
            "returns_mean": flat_returns.mean().item(),
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
