"""
Training loop for ThorEnv trajectories with PPO updates.

This script uses:
  - Environment: src/simulation/thor_env.py (ThorEnv)
  - Optimizer:   src/simulation/PPO.py (PPO)

Usage examples:
  uv run python scripts/train_controller_ppo.py --episodes 100 --scenes FloorPlan1 FloorPlan2
  uv run python scripts/train_controller_ppo.py --episodes 50 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.simulation.PPO import PPO


class MLPActorCritic(nn.Module):
    """Simple actor-critic over the flattened observation vector."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        x = self.backbone(obs)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        return logits, values


class ActionSampler:
    """Callable wrapper to sample an action index from policy logits."""

    def __init__(self, policy: nn.Module, device: torch.device) -> None:
        self.policy = policy
        self.device = device

    def __call__(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            obs_batch = obs.to(self.device).float().unsqueeze(0)
            logits, _ = self.policy(obs_batch)
            dist = Categorical(logits=logits.squeeze(0))
            return int(dist.sample().item())


def build_trajectory_step(
    obs: torch.Tensor,
    action_idx: int,
    action_name: str,
    reward: float,
    next_obs: torch.Tensor,
    done: bool,
) -> dict:
    """Build canonical + legacy trajectory schema for PPO compatibility."""
    return {
        # Canonical keys
        "obs": obs,
        "Action": action_name,
        "reward": reward,
        "nex_obs": next_obs,
        "is_model_done": done,
        # Legacy aliases
        "action": action_idx,
        "next_obs": next_obs,
        "done": done,
    }


def save_checkpoint(
    path: Path,
    episode: int,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "policy": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


def train(args: argparse.Namespace) -> None:
    from src.simulation.thor_env import ThorEnv

    device = torch.device(args.device)

    env = ThorEnv(
        target_object_types=args.target_object_types,
        base_resolution=(args.width, args.height),
        max_steps=args.max_steps,
        device=device.type,
        seed=args.seed,
    )

    try:
        # Probe observation size from one reset before creating the policy.
        init_obs = env.reset(scene=args.scenes[0], target_obj_type=args.target_obj_type)
        obs_dim = int(init_obs.numel())
        n_actions = len(env.action_list)

        policy = MLPActorCritic(
            obs_dim=obs_dim,
            n_actions=n_actions,
            hidden_dim=args.hidden_dim,
        ).to(device)
        ppo = PPO(
            policy=policy,
            lr=args.lr,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            max_grad_norm=args.max_grad_norm,
            ppo_epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            device=device,
        )
        sampler = ActionSampler(policy=policy, device=device)

        rewards: list[float] = []
        lengths: list[int] = []
        successes: list[bool] = []

        for episode in range(args.episodes):
            scene = args.scenes[episode % len(args.scenes)]
            obs = env.reset(scene=scene, target_obj_type=args.target_obj_type)
            trajectory: list[dict] = []
            episode_reward = 0.0
            episode_len = 0
            episode_success = False

            while True:
                action_idx = sampler(obs)
                action_name = env.action_list[action_idx]
                next_obs, reward, terminated, truncated, info = env.step(action_idx)
                done = bool(terminated or truncated)

                trajectory.append(
                    build_trajectory_step(
                        obs=obs,
                        action_idx=action_idx,
                        action_name=action_name,
                        reward=float(reward),
                        next_obs=next_obs,
                        done=done,
                    )
                )

                episode_reward += float(reward)
                episode_len += 1
                obs = next_obs

                if done:
                    episode_success = bool(terminated and info.get("success", False))
                    break

            metrics = ppo.update(trajectory=trajectory, action2id=env.action2id)

            rewards.append(episode_reward)
            lengths.append(episode_len)
            successes.append(episode_success)

            if (episode + 1) % args.print_every == 0:
                print(
                    f"[ep {episode + 1:4d}] "
                    f"reward10={np.mean(rewards[-10:]):.3f} | "
                    f"len10={np.mean(lengths[-10:]):.1f} | "
                    f"success10={np.mean(successes[-10:]):.0%} | "
                    f"pi={metrics.policy_loss:.4f} | "
                    f"v={metrics.value_loss:.4f} | "
                    f"H={metrics.entropy:.4f}"
                )

            if args.checkpoint_dir and (episode + 1) % args.checkpoint_every == 0:
                ckpt_path = Path(args.checkpoint_dir) / f"ep{episode + 1}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    episode=episode + 1,
                    policy=policy,
                    optimizer=ppo.optimizer,
                )

        print(
            "Training done. "
            f"mean_reward={np.mean(rewards):.3f}, "
            f"mean_len={np.mean(lengths):.1f}, "
            f"success_rate={np.mean(successes):.1%}"
        )
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on ThorEnv trajectories.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenes", nargs="+", default=["FloorPlan1"])
    parser.add_argument("--target-obj-type", default=None)
    parser.add_argument("--target-object-types", nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=200)

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)

    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--checkpoint-dir", default="checkpoints/controller_ppo")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
