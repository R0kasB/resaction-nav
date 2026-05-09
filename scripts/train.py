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
import yaml
from huggingface_hub import HfApi
from tqdm import tqdm

from src.agents.ppo_agent import PPOAgent
from src.models.agent_policy import AgentPolicy
from src.simulation.thor_env import RewardConfig, ThorEnv

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency in test/runtime environments
    wandb = None


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


def _resolve_target_setup(training_cfg: dict, env_cfg: dict) -> tuple[str | None, list[str] | None]:
    target_cfg = training_cfg.get("target")
    fixed_target_object = None
    target_object_cycle = None

    # Backward compatibility with older flat config keys.
    if target_cfg is None:
        fixed_target_object = training_cfg.get("target_object_type")
        target_object_cycle = training_cfg.get("target_object_cycle")
        if fixed_target_object and target_object_cycle:
            raise ValueError("Use either training.target_object_type or training.target_object_cycle, not both.")
        if target_object_cycle is not None and len(target_object_cycle) == 0:
            raise ValueError("training.target_object_cycle cannot be empty.")
        return fixed_target_object, target_object_cycle

    mode = target_cfg.get("mode", "random")
    candidates = target_cfg.get("candidates")
    object_type = target_cfg.get("object_type")
    cycle = target_cfg.get("cycle")

    if mode == "fixed":
        if not object_type:
            raise ValueError("training.target.object_type is required when mode='fixed'.")
        fixed_target_object = object_type
    elif mode == "cycle":
        if not cycle:
            raise ValueError("training.target.cycle must be a non-empty list when mode='cycle'.")
        target_object_cycle = list(cycle)
    elif mode == "random":
        pass
    else:
        raise ValueError(f"Unsupported training.target.mode: {mode!r}")

    if candidates is not None and len(candidates) == 0:
        raise ValueError("training.target.candidates cannot be empty when provided.")

    derived_candidates = None
    if mode == "fixed":
        derived_candidates = [fixed_target_object]
    elif mode == "cycle":
        derived_candidates = list(dict.fromkeys(target_object_cycle))

    if candidates is None:
        candidates = derived_candidates
    elif mode == "fixed" and fixed_target_object not in candidates:
        raise ValueError("training.target.object_type must be included in training.target.candidates.")
    elif mode == "cycle":
        missing = [obj for obj in target_object_cycle if obj not in candidates]
        if missing:
            raise ValueError("All training.target.cycle objects must be included in training.target.candidates.")

    env_candidates = env_cfg.get("target_object_types")
    if candidates is not None:
        if env_candidates is not None and list(env_candidates) != list(candidates):
            raise ValueError(
                "Conflicting target candidates: use training.target.candidates as the single source of truth "
                "or make env.target_object_types match exactly."
            )
        env_cfg["target_object_types"] = list(candidates)

    return fixed_target_object, target_object_cycle


def run_agent(
    env: ThorEnv,
    agent: PPOAgent,
    cfg: dict,
    device: torch.device,
    start_episode: int = 0,
    fixed_target_object: str | None = None,
    target_object_cycle: list[str] | None = None,
):
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
    if use_wandb and wandb is None:
        raise RuntimeError("wandb logging is enabled, but wandb could not be imported.")

    rewards = []
    episode_lengths = []
    successes = []

    for episode in tqdm(range(start_episode, num_episodes)):
        scene = scenes[episode % len(scenes)]
        target_obj_type = fixed_target_object
        if target_object_cycle:
            target_obj_type = target_object_cycle[episode % len(target_object_cycle)]

        image = env.reset(scene, target_obj_type=target_obj_type).to(device)
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

        if use_wandb and wandb is not None:
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


def run_training_pipeline(cfg: dict, resume_path: str | None = None) -> dict:
    tcfg = cfg["training"]
    env_cfg = dict(cfg["env"])
    agent_cfg = cfg["agent"]
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})

    Path(tcfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if wandb_cfg.get("enabled", False):
        if wandb is None:
            raise RuntimeError("wandb is enabled in config, but wandb could not be imported.")
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("run_name"),
            config=cfg,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fixed_target_object, target_object_cycle = _resolve_target_setup(
        training_cfg=tcfg,
        env_cfg=env_cfg,
    )

    reward_cfg = RewardConfig(**env_cfg.pop("reward_cfg", {}))
    env = ThorEnv(**env_cfg, reward_cfg=reward_cfg)
    env_target_embed_dim = int(getattr(env, "target_object_embed_dim", 0))

    model_target_embed_dim = model_cfg.get("target_object_embed_dim")
    if model_target_embed_dim is not None and int(model_target_embed_dim) != env_target_embed_dim:
        raise ValueError(
            "model.target_object_embed_dim must match env.target_object_embed_dim. "
            f"Got model={model_target_embed_dim}, env={env_target_embed_dim}."
        )
    model_cfg["target_object_embed_dim"] = env_target_embed_dim

    policy = AgentPolicy(
        n_actions=len(env.action_list),
        encoder_name=cfg.get("visual_encoder", {}).get("model_name", "dinov2_vitb14"),
        **model_cfg,
        device=device,
    ).to(device)

    agent = PPOAgent(policy=policy, **agent_cfg)

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
            fixed_target_object=fixed_target_object,
            target_object_cycle=target_object_cycle,
        )

        output_data(
            rewards=rewards,
            successes=successes,
            episode_lengths=episode_lengths,
            run_type="ppo_dynamic_resolution",
            params=agent_cfg,
            filename=f"{tcfg['output_dir']}/training_log.txt",
        )
    finally:
        env.close()

        if wandb_cfg.get("enabled", False) and wandb is not None:
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
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    run_training_pipeline(cfg=cfg, resume_path=args.resume)