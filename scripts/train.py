"""
Training entry point for the dynamic-resolution navigation agent.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --cfg cfgs/train_rl.yaml
    uv run python scripts/train.py --cfg cfgs/train_rl.yaml --resume checkpoints/ep100.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from huggingface_hub import HfApi
from tqdm import tqdm

from src.agents.ppo_agent import PPOAgent
from src.models.lstm import PolicyLSTM
from src.simulation.thor_env import RewardConfig, ThorEnv


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(policy, optimizer, episode, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
    }, path)


def push_to_hub(api: HfApi, repo_id: str, local_path: str, episode: int):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"checkpoints/ep{episode}.pt",
        repo_id=repo_id,
        repo_type="model",
    )


def run_agent(env: ThorEnv, agent: PPOAgent, cfg: dict, start_episode: int = 0):
    tcfg = cfg["training"]
    scenes = tcfg["scenes"]
    num_episodes = tcfg["num_episodes"]
    print_every = tcfg["print_every"]
    checkpoint_every = tcfg["checkpoint_every"]
    checkpoint_dir = tcfg["checkpoint_dir"]

    hf_cfg = cfg.get("huggingface", {})
    hf_push = hf_cfg.get("push", False)
    hf_push_every = hf_cfg.get("push_every", 100)
    hf_repo_id = hf_cfg.get("repo_id")
    hf_api = HfApi() if hf_push else None

    use_wandb = cfg.get("wandb", {}).get("enabled", False)

    rewards = []
    episode_lengths = []
    successes = []

    for episode in tqdm(range(start_episode, num_episodes)):
        scene = scenes[episode % len(scenes)]
        obs = env.reset(scene)
        obs = obs.flatten().unsqueeze(0)  # (1, obs_dim) — TODO: replace with encoder

        done = False
        episode_reward = 0.0
        episode_success = False
        hidden = None
        agent.store_initial_hidden(hidden)

        while not done:
            action_idx, log_prob, value, hidden = agent.act(obs, hidden)
            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            next_obs = next_obs.flatten().unsqueeze(0)

            agent.store(obs, action_idx, log_prob, reward, value, terminated or truncated)

            obs = next_obs
            episode_reward += reward
            done = terminated or truncated
            if terminated and info["success"]:
                episode_success = True

        loss_dict = agent.update(obs, hidden)
        rewards.append(episode_reward)
        episode_lengths.append(info["step"])
        successes.append(episode_success)

        if use_wandb:
            wandb.log({
                "episode": episode,
                "reward": episode_reward,
                "episode_length": info["step"],
                "success": float(episode_success),
                "success_rate_10": float(np.mean(successes[-10:])),
                **loss_dict,
            })

        if episode % print_every == 0:
            print(
                f"[ep {episode:4d}] "
                f"reward (last 10): {np.mean(rewards[-10:]):.3f} | "
                f"steps: {np.mean(episode_lengths[-10:]):.1f} | "
                f"success: {np.mean(successes[-10:]):.0%}"
            )

        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = f"{checkpoint_dir}/ep{episode + 1}.pt"
            save_checkpoint(agent.policy, agent.optimizer, episode + 1, ckpt_path)
            if hf_push and hf_api and (episode + 1) % hf_push_every == 0:
                push_to_hub(hf_api, hf_repo_id, ckpt_path, episode + 1)

    return rewards, episode_lengths, successes


def output_data(rewards, successes, episode_lengths, run_type, params, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    mean_reward = float(np.mean(rewards))
    success_pct = float(np.mean(successes) * 100)
    mean_ep_len = float(np.mean(episode_lengths))

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Type: {run_type}\n")
        f.write(f"Parameters: {params}\n")
        f.write(f"Episodes: {len(rewards)}\n")
        f.write(f"Mean Reward: {mean_reward:.4f}\n")
        f.write(f"Final Mean Reward (last 10): {np.mean(rewards[-10:]):.4f}\n")
        f.write(f"Success Rate: {success_pct:.2f}%\n")
        f.write(f"Mean Episode Length: {mean_ep_len:.4f}\n")
        f.write("-" * 40 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfgs/train_rl.yaml", help="path to YAML config")
    parser.add_argument("--resume", default=None, help="path to checkpoint .pt to resume from")
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    tcfg = cfg["training"]
    env_cfg = dict(cfg["env"])
    agent_cfg = cfg["agent"]
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})

    Path(tcfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config=cfg,
        )

    reward_cfg = RewardConfig(**env_cfg.pop("reward_cfg", {}))
    env = ThorEnv(**env_cfg, reward_cfg=reward_cfg)

    base_res = env.base_resolution
    flat_obs_dim = 3 * base_res[0] * base_res[1]  # TODO: replace with encoder output dim
    policy = PolicyLSTM(
        vis_dim=flat_obs_dim,
        n_actions=len(env.action_list),
        **model_cfg,
    )
    agent = PPOAgent(policy=policy, **agent_cfg)

    start_episode = 0
    resume_path = args.resume or tcfg.get("resume_from")
    if resume_path:
        ckpt = torch.load(resume_path, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"]
        print(f"Resumed from {resume_path} (episode {start_episode})")

    try:
        rewards, episode_lengths, successes = run_agent(env, agent, cfg, start_episode)
        output_data(
            rewards, successes, episode_lengths,
            run_type="ppo_baseline",
            params=agent_cfg,
            filename=f"{tcfg['output_dir']}/training_log.txt",
        )
    finally:
        env.close()
        if wandb_cfg.get("enabled", False):
            wandb.finish()
