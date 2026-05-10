"""
Training entry point for the dynamic-resolution navigation agent.

Usage:
    uv run python scripts/train.py
    uv run python scripts/train.py --cfg cfgs/train_rl.yaml
    uv run python scripts/train.py --cfg cfgs/train_rl.yaml --resume checkpoints/ep100.pt
    or 
    sbatch train_all.sh to train adaptative agent + 4 baselines
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from huggingface_hub import HfApi
from tqdm import tqdm

from src.agents.baselines import (
    LowResBaseline,
    HighResBaseline,
    RandomSensingBaseline,
    FixedScheduleBaseline,
)

from src.agents.ppo_agent import PPOAgent
from src.models.agent_policy import AgentPolicy
from src.simulation.thor_env import RewardConfig, ThorEnv


AGENT_REGISTRY = {
    "adaptive":       PPOAgent,               # default: full adaptive PPO
    "low_res":        LowResBaseline,
    "high_res":       HighResBaseline,
    "random_sensing": RandomSensingBaseline,
    "fixed_schedule": FixedScheduleBaseline,
}

def build_agent(agent_type: str, **kwargs):
    """Instantiate the right agent from a string key."""
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent_type '{agent_type}'. "
            f"Choose from: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_type](**kwargs)
 


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(policy, optimizer, episode, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    torch.save({
        "policy":    policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode":   episode,
    }, tmp_path)
    Path(tmp_path).rename(path)

def push_to_hub(api: HfApi, repo_id: str, local_path: str, episode: int):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"checkpoints/ep{episode}.pt",
        repo_id=repo_id,
        repo_type="model",
    )


def run_agent(env: ThorEnv, agent: PPOAgent, cfg: dict, device: torch.device, start_episode: int = 0):
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

        image = env.reset(scene).to(device)
        prev_action_idx = None
        aux_features = env.get_aux_features(prev_action_idx=prev_action_idx).to(device)

        done = False
        episode_reward = 0.0
        episode_success = False
        episode_num_sense_actions = 0

        hidden = None
        agent.store_initial_hidden(hidden)

        while not done:
            action_idx, log_prob, value, hidden = agent.act(image=image, aux_features=aux_features, hidden=hidden)

            next_image, reward, terminated, truncated, info = env.step(action_idx)
            next_image = next_image.to(device)

            next_aux_features = env.get_aux_features(prev_action_idx=action_idx).to(device)

            agent.store(
                image=image,
                aux_features=aux_features,
                action=action_idx,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=terminated or truncated,
            )

            if env.action_list[action_idx] == "SENSE":
                episode_num_sense_actions += 1

            image = next_image
            aux_features = next_aux_features
            prev_action_idx = action_idx

            episode_reward += reward
            done = terminated or truncated

            if terminated and info["success"]:
                episode_success = True

        loss_dict = agent.update(next_image=image, next_aux_features=aux_features, final_hidden=hidden)

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
                "num_sense_actions": episode_num_sense_actions,
                "final_downgrade": info["downgrade"],
                "final_sensing_budget": info["sensing_budget"],
                **loss_dict,
            })

        if episode % print_every == 0:
            print(
                f"[ep {episode:4d}] "
                f"reward (last 10): {np.mean(rewards[-10:]):.3f} | "
                f"steps: {np.mean(episode_lengths[-10:]):.1f} | "
                f"success: {np.mean(successes[-10:]):.0%} | "
                f"sense actions: {episode_num_sense_actions}"
            )

        if (episode + 1) % checkpoint_every == 0:
            ckpt_path = f"{checkpoint_dir}/ep{episode + 1}.pt"
            save_checkpoint(agent.policy, agent.optimizer, episode + 1, ckpt_path)

            if hf_push and hf_api and (episode + 1) % hf_push_every == 0:
                push_to_hub(hf_api, hf_repo_id, ckpt_path, episode + 1)

    return rewards, episode_lengths, successes


def output_data(rewards, successes, episode_lengths, run_type, params, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if len(rewards) == 0:
        mean_reward = 0.0
        success_pct = 0.0
        mean_ep_len = 0.0
        final_mean_reward = 0.0
    else:
        mean_reward = float(np.mean(rewards))
        success_pct = float(np.mean(successes) * 100)
        mean_ep_len = float(np.mean(episode_lengths))
        final_mean_reward = float(np.mean(rewards[-10:]))

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Type: {run_type}\n")
        f.write(f"Parameters: {params}\n")
        f.write(f"Episodes: {len(rewards)}\n")
        f.write(f"Mean Reward: {mean_reward:.4f}\n")
        f.write(f"Final Mean Reward (last 10): {final_mean_reward:.4f}\n")
        f.write(f"Success Rate: {success_pct:.2f}%\n")
        f.write(f"Mean Episode Length: {mean_ep_len:.4f}\n")
        f.write("-" * 40 + "\n")


def run_training_pipeline(cfg: dict, resume_path: str | None = None, agent_type: str | None = None) -> dict:
    tcfg = cfg["training"]
    env_cfg = dict(cfg["env"])
    agent_cfg = cfg["agent"]
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})

    if agent_type is None:
        agent_type = cfg.get("agent_type", "adaptive")

    tcfg["checkpoint_dir"] = str(Path(tcfg["checkpoint_dir"]) / agent_type)
    tcfg["output_dir"]     = str(Path(tcfg["output_dir"])     / agent_type)
    Path(tcfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if wandb_cfg.get("enabled", False):
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name", agent_type),
            config={**cfg, "agent_type": agent_type},
        )


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reward_cfg = RewardConfig(**env_cfg.pop("reward_cfg", {}))
    env = ThorEnv(**env_cfg, reward_cfg=reward_cfg)

    policy = AgentPolicy(
        n_actions=len(env.action_list),
        encoder_name=cfg.get("visual_encoder", {}).get("model_name", "dinov2_vitb14"),
        **model_cfg,
        device=device,
    ).to(device)

    agent = build_agent(agent_type, policy=policy, **agent_cfg)

    start_episode = 0
    resume_path = resume_path or tcfg.get("resume_from")

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        policy.load_state_dict(ckpt["policy"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"]
        print(f"Resumed from {resume_path} (episode {start_episode})")

    try:
        rewards, episode_lengths, successes = run_agent(
            env=env,
            agent=agent,
            cfg=cfg,
            device=device,
            start_episode=start_episode,
        )

        output_data(
            rewards=rewards,
            successes=successes,
            episode_lengths=episode_lengths,
            run_type=agent_type,
            params=agent_cfg,
            filename=f"{tcfg['output_dir']}/training_log.txt",
        )
    finally:
        env.close()

        if wandb_cfg.get("enabled", False):
            wandb.finish()

    return {
        "episodes": len(rewards),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "output_log": f"{tcfg['output_dir']}/training_log.txt",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfgs/train_rl.yaml", help="path to YAML config")
    parser.add_argument("--resume", default=None, help="path to checkpoint .pt to resume from")
    parser.add_argument("--agent", default=None, choices=list(AGENT_REGISTRY.keys()), help="agent type (overrides config agent_type field)")

    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    run_training_pipeline(cfg=cfg, resume_path=args.resume, agent_type=args.agent)