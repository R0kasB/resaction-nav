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
import os
from pathlib import Path

import numpy as np
import torch
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
from src.utils.trajectory_logger import TrajectoryLogger

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

VIDEO_EVERY = 50

AGENT_REGISTRY = {
    "adaptive":       PPOAgent,
    "low_res":        LowResBaseline,
    "high_res":       HighResBaseline,
    "random_sensing": RandomSensingBaseline,
    "fixed_schedule": FixedScheduleBaseline,
}


def build_agent(agent_type: str, **kwargs):
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(
            f"Unknown agent_type '{agent_type}'. "
            f"Choose from: {list(AGENT_REGISTRY.keys())}"
        )
    return AGENT_REGISTRY[agent_type](**kwargs)


def load_cfg(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(policy, optimizer, episode, path, retries=3):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(path) + ".tmp"
    for attempt in range(retries):
        try:
            torch.save({
                "policy":              policy.state_dict(),
                "optimizer":           optimizer.state_dict(),
                "episode":             episode,
                "target_object_vocab": policy.target_object_vocab,
            }, tmp_path)
            Path(tmp_path).rename(path)
            return
        except RuntimeError as e:
            Path(tmp_path).unlink(missing_ok=True)
            if attempt == retries - 1:
                print(f"[checkpoint] failed after {retries} attempts: {e}")
            else:
                print(f"[checkpoint] write failed (attempt {attempt+1}), retrying...")


def push_to_hub(api: HfApi, repo_id: str, local_path: str, episode: int):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"checkpoints/ep{episode}.pt",
        repo_id=repo_id,
        repo_type="model",
    )


def _build_cluster_run_id(training_cfg: dict) -> str | None:
    explicit_run_id = training_cfg.get("run_id")
    if explicit_run_id:
        return str(explicit_run_id)

    auto_cluster_id = training_cfg.get("auto_cluster_run_id", True)
    if not auto_cluster_id:
        return None

    job_id = (
        os.getenv("SLURM_JOB_ID")
        or os.getenv("PBS_JOBID")
        or os.getenv("LSB_JOBID")
        or os.getenv("JOB_ID")
    )
    if not job_id:
        return None

    rank = (
        os.getenv("SLURM_PROCID")
        or os.getenv("SLURM_ARRAY_TASK_ID")
        or os.getenv("OMPI_COMM_WORLD_RANK")
    )
    suffix = f"job-{job_id}"
    if rank is not None:
        suffix += f"-rank-{rank}"
    return suffix


def _apply_run_id_to_training_dirs(training_cfg: dict) -> str | None:
    run_id = _build_cluster_run_id(training_cfg)
    if not run_id:
        return None
    output_dir = Path(training_cfg["output_dir"]) / run_id
    checkpoint_dir = Path(training_cfg["checkpoint_dir"]) / run_id
    training_cfg["output_dir"] = str(output_dir)
    training_cfg["checkpoint_dir"] = str(checkpoint_dir)
    return run_id


def _resolve_target_setup(training_cfg: dict, env_cfg: dict) -> tuple[str | None, list[str] | None]:
    target_cfg = training_cfg.get("target")
    fixed_target_object = None
    target_object_cycle = None

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
    envs: ThorEnv | list[ThorEnv],
    agent: PPOAgent,
    cfg: dict,
    device: torch.device,
    start_episode: int = 0,
    fixed_target_object: str | None = None,
    target_object_cycle: list[str] | None = None,
    trajectory_logger: TrajectoryLogger | None = None,
):
    tcfg = cfg["training"]
    scenes = tcfg["scenes"]
    num_episodes = tcfg["num_episodes"]
    num_parallel_envs = int(tcfg.get("num_parallel_envs", 1))
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

    video_every = int(tcfg.get("video_every", VIDEO_EVERY))

    rewards = []
    episode_lengths = []
    successes = []
    spls = []

    if isinstance(envs, ThorEnv):
        env_list = [envs]
    else:
        env_list = list(envs)

    if not env_list:
        raise ValueError("At least one environment instance is required.")
    if num_parallel_envs < 1:
        raise ValueError("training.num_parallel_envs must be >= 1.")
    if num_parallel_envs > len(env_list):
        raise ValueError(
            f"training.num_parallel_envs={num_parallel_envs} exceeds available environments ({len(env_list)})."
        )

    episode_table = (
        wandb.Table(
            columns=[
                "episode", "scene", "target", "success", "spl", "reward",
                "steps", "n_sense", "initial_dist", "final_dist",
            ]
        )
        if use_wandb and wandb is not None
        else None
    )

    episode = start_episode
    progress = tqdm(total=max(num_episodes - start_episode, 0))

    while episode < num_episodes:
        remaining = num_episodes - episode
        batch_size = min(num_parallel_envs, remaining)
        slots = []

        # ----- Reset all slots for this iteration -----
        for slot_idx in range(batch_size):
            current_episode = episode + slot_idx
            env = env_list[slot_idx]
            scene = scenes[current_episode % len(scenes)]

            target_obj_type = fixed_target_object
            if target_object_cycle:
                target_obj_type = target_object_cycle[current_episode % len(target_object_cycle)]

            image = env.reset(scene, target_obj_type=target_obj_type, episode=current_episode).to(device)
            aux_features = env.get_aux_features(prev_action_idx=None).to(device)
            if trajectory_logger is not None:
                trajectory_logger.save_layout_once(scene, env)

            slots.append({
                "env": env,
                "episode": current_episode,
                "image": image,
                "aux_features": aux_features,
                "hidden": None,
                "initial_hidden": None,   # captured once, before any act()
                "transitions": [],
                "frames": [],
                "episode_reward": 0.0,
                "episode_success": False,
                "episode_spl": 0.0,
                "episode_num_sense_actions": 0,
                "done": False,
                "info": None,
                "final_image": image,
                "final_aux_features": aux_features,
                "final_hidden": None,
            })

        # ----- Roll out all slots until all done. Sequential per-slot inference
        # to keep things deterministic and avoid the non-thread-safe controller.
        while any(not slot["done"] for slot in slots):
            active_slots = [slot for slot in slots if not slot["done"]]

            for slot in active_slots:
                action_idx, log_prob, value, hidden = agent.act(
                    image=slot["image"],
                    aux_features=slot["aux_features"],
                    hidden=slot["hidden"],
                )

                next_image_raw, reward, terminated, truncated, info = slot["env"].step(action_idx)
                next_image = next_image_raw.to(device)
                next_aux_features = slot["env"].get_aux_features(
                    prev_action_idx=action_idx,
                ).to(device)

                done = terminated or truncated
                slot["transitions"].append({
                    "image":        slot["image"],
                    "aux_features": slot["aux_features"],
                    "action":       action_idx,
                    "log_prob":     log_prob,
                    "reward":       reward,
                    "value":        value,
                    "done":         done,
                })

                if use_wandb and slot["episode"] % video_every == 0:
                    frame = (
                        slot["image"].detach().permute(1, 2, 0).cpu().numpy() * 255.0
                    ).clip(0, 255).astype(np.uint8)
                    slot["frames"].append(frame)

                action_name = slot["env"].action_list[action_idx]
                if action_name == "SENSE":
                    slot["episode_num_sense_actions"] += 1
                if trajectory_logger is not None:
                    trajectory_logger.log_step(
                        slot["episode"], info["step"], slot["env"].scene,
                        slot["env"].target_obj_type, slot["env"],
                        action_name, reward, info,
                    )

                slot["image"] = next_image
                slot["aux_features"] = next_aux_features
                slot["hidden"] = hidden
                slot["episode_reward"] += reward
                slot["info"] = info

                if terminated and info["success"]:
                    slot["episode_success"] = True
                if done:
                    slot["episode_spl"] = float(info.get("spl", 0.0))
                    slot["done"] = True
                    slot["final_image"] = next_image
                    slot["final_aux_features"] = next_aux_features
                    slot["final_hidden"] = hidden

        # ----- Build one rollout dict per slot and run a SINGLE batched update. -----
        rollouts = []
        for slot in slots:
            trans = slot["transitions"]
            if not trans:
                continue
            images = torch.stack([t["image"] for t in trans]).to(device)
            aux_feats = torch.stack([t["aux_features"] for t in trans]).to(device)
            log_probs = torch.stack([t["log_prob"] for t in trans]).detach().to(device)
            values = torch.stack([t["value"] for t in trans]).detach().to(device)
            actions = torch.tensor(
                [t["action"] for t in trans], dtype=torch.long, device=device,
            )
            rewards_list = [float(t["reward"]) for t in trans]
            dones_list = [bool(t["done"]) for t in trans]

            rollouts.append({
                "images":            images,
                "aux_features":      aux_feats,
                "actions":           actions,
                "log_probs":         log_probs,
                "rewards":           rewards_list,
                "values":            values,
                "dones":             dones_list,
                "initial_hidden":    slot["initial_hidden"],  # None here (start of ep)
                "next_image":        slot["final_image"],
                "next_aux_features": slot["final_aux_features"],
                "final_hidden":      slot["final_hidden"],
            })

        if rollouts:
            loss_dict = agent.update_from_rollouts(rollouts)
        else:
            loss_dict = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        if trajectory_logger is not None:
            trajectory_logger.flush()

        # ----- Logging / checkpointing -----
        for slot in slots:
            current_episode = slot["episode"]
            info = slot["info"]
            rewards.append(slot["episode_reward"])
            episode_lengths.append(info["step"])
            successes.append(slot["episode_success"])
            spls.append(slot["episode_spl"])

            if use_wandb and wandb is not None:
                log_dict = {
                    "episode": current_episode,
                    "reward": slot["episode_reward"],
                    "episode_length": info["step"],
                    "success": float(slot["episode_success"]),
                    "success_rate_10": float(np.mean(successes[-10:])),
                    "spl": slot["episode_spl"],
                    "spl_10": float(np.mean(spls[-10:])),
                    "num_sense_actions": slot["episode_num_sense_actions"],
                    "final_downgrade": info["downgrade"],
                    "final_sensing_budget": info["sensing_budget"],
                    "initial_distance": info["initial_distance"],
                    "final_distance": info["current_distance"],
                    "min_distance_seen": info["min_distance_seen"],
                    "optimal_path_length": info.get("optimal_path_length", float("nan")),
                    "path_length": info.get("path_length", float("nan")),
                    **loss_dict,
                }

                if current_episode % video_every == 0 and slot["frames"]:
                    video_np = np.stack(slot["frames"]).transpose(0, 3, 1, 2)
                    log_dict["video"] = wandb.Video(
                        video_np, fps=4, format="mp4",
                        caption=(
                            f"ep{current_episode} | "
                            f"{slot['env'].target_obj_type} | "
                            f"{'success' if slot['episode_success'] else 'fail'} | "
                            f"reward={slot['episode_reward']:.2f} | "
                            f"spl={slot['episode_spl']:.2f}"
                        ),
                    )

                wandb.log(log_dict)

                if episode_table is not None:
                    episode_table.add_data(
                        current_episode,
                        slot["env"].scene,
                        slot["env"].target_obj_type,
                        slot["episode_success"],
                        slot["episode_spl"],
                        slot["episode_reward"],
                        info["step"],
                        slot["episode_num_sense_actions"],
                        info["initial_distance"],
                        info["current_distance"],
                    )

            if current_episode % print_every == 0:
                print(
                    f"[ep {current_episode:4d}] "
                    f"reward (last 10): {np.mean(rewards[-10:]):.3f} | "
                    f"steps: {np.mean(episode_lengths[-10:]):.1f} | "
                    f"success: {np.mean(successes[-10:]):.0%} | "
                    f"spl: {np.mean(spls[-10:]):.3f} | "
                    f"sense actions: {slot['episode_num_sense_actions']}"
                )

            if (current_episode + 1) % checkpoint_every == 0:
                ckpt_path = f"{checkpoint_dir}/ep{current_episode + 1}.pt"
                save_checkpoint(agent.policy, agent.optimizer, current_episode + 1, ckpt_path)
                if hf_push and hf_api and (current_episode + 1) % hf_push_every == 0:
                    push_to_hub(hf_api, hf_repo_id, ckpt_path, current_episode + 1)

        episode += batch_size
        progress.update(batch_size)

    progress.close()

    if use_wandb and wandb is not None and episode_table is not None:
        wandb.log({"episodes_table": episode_table})

    return rewards, episode_lengths, successes, spls


def output_data(rewards, successes, episode_lengths, spls, run_type, params, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if len(rewards) == 0:
        mean_reward = success_pct = mean_ep_len = final_mean_reward = mean_spl = 0.0
    else:
        mean_reward = float(np.mean(rewards))
        success_pct = float(np.mean(successes) * 100)
        mean_ep_len = float(np.mean(episode_lengths))
        final_mean_reward = float(np.mean(rewards[-10:]))
        mean_spl = float(np.mean(spls)) if spls else 0.0

    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"Type: {run_type}\n")
        f.write(f"Parameters: {params}\n")
        f.write(f"Episodes: {len(rewards)}\n")
        f.write(f"Mean Reward: {mean_reward:.4f}\n")
        f.write(f"Final Mean Reward (last 10): {final_mean_reward:.4f}\n")
        f.write(f"Success Rate: {success_pct:.2f}%\n")
        f.write(f"Mean SPL: {mean_spl:.4f}\n")
        f.write(f"Mean Episode Length: {mean_ep_len:.4f}\n")
        f.write("-" * 40 + "\n")


def _apply_agent_mode_defaults(agent_type: str, agent_cfg: dict, env_cfg: dict) -> str:
    agent_label = agent_type

    if agent_type != "high_res":
        return agent_label

    mode = agent_cfg.get("mode", "baseline")
    if mode not in {"poc", "poc_rokas", "baseline"}:
        raise ValueError(f"Invalid high_res mode={mode!r}. Use 'poc', 'poc_rokas', or 'baseline'.")

    agent_label = f"high_res_{mode}"
    env_cfg["fixed_high_res"] = True
    env_cfg["max_sensing_budget"] = 0

    reward_cfg = dict(env_cfg.get("reward_cfg", {}))
    reward_cfg["sense_penalty"] = 0.0
    reward_cfg["oversensing_penalty"] = 0.0
    env_cfg["reward_cfg"] = reward_cfg

    if mode == "poc":
        env_cfg["action_set"] = "minimal"
        env_cfg["auto_success_on_goal"] = True
    elif mode == "poc_rokas":
        env_cfg["action_set"] = "minimal"
        env_cfg["auto_success_on_goal"] = False
    else:
        env_cfg["action_set"] = "navigation"
        env_cfg["auto_success_on_goal"] = False

    return agent_label


def run_training_pipeline(cfg: dict, resume_path: str | None = None, agent_type: str | None = None) -> dict:
    tcfg = cfg["training"]
    env_cfg = dict(cfg["env"])
    agent_cfg = cfg["agent"]
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {})

    run_id = _apply_run_id_to_training_dirs(tcfg)

    if agent_type is None:
        agent_type = cfg.get("agent_type", "adaptive")

    agent_label = _apply_agent_mode_defaults(
        agent_type=agent_type, agent_cfg=agent_cfg, env_cfg=env_cfg,
    )

    tcfg["checkpoint_dir"] = str(Path(tcfg["checkpoint_dir"]) / agent_label)
    tcfg["output_dir"]     = str(Path(tcfg["output_dir"])     / agent_label)
    Path(tcfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)
    Path(tcfg["output_dir"]).mkdir(parents=True, exist_ok=True)

    if wandb_cfg.get("enabled", False):
        if wandb is None:
            raise RuntimeError("wandb is enabled in config, but wandb could not be imported.")
        wandb.init(
            project=wandb_cfg["project"],
            entity=wandb_cfg.get("entity"),
            name=f"{wandb_cfg['run_name']}-{agent_label}" if wandb_cfg.get("run_name") else agent_label,
            group=wandb_cfg.get("group"),
            config={**cfg, "agent_type": agent_type, "agent_label": agent_label},
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device=", device)

    fixed_target_object, target_object_cycle = _resolve_target_setup(
        training_cfg=tcfg, env_cfg=env_cfg,
    )

    reward_cfg = RewardConfig(**env_cfg.pop("reward_cfg", {}))
    num_parallel_envs = int(tcfg.get("num_parallel_envs", 1))
    if num_parallel_envs < 1:
        raise ValueError("training.num_parallel_envs must be >= 1.")

    base_seed = env_cfg.get("seed")
    envs: list[ThorEnv] = []
    for env_idx in range(num_parallel_envs):
        env_instance_cfg = dict(env_cfg)
        if base_seed is not None:
            env_instance_cfg["seed"] = int(base_seed) + env_idx
        controller_kwargs = env_instance_cfg.get("controller_kwargs")
        if isinstance(controller_kwargs, dict):
            controller_kwargs = dict(controller_kwargs)
            if controller_kwargs.get("port") is not None:
                controller_kwargs["port"] = int(controller_kwargs["port"]) + env_idx
            env_instance_cfg["controller_kwargs"] = controller_kwargs
        env = ThorEnv(**env_instance_cfg, reward_cfg=reward_cfg)
        envs.append(env)

    env_target_embed_dim = int(getattr(envs[0], "target_object_embed_dim", 0))
    model_target_embed_dim = model_cfg.get("target_object_embed_dim")
    if model_target_embed_dim is not None and int(model_target_embed_dim) != env_target_embed_dim:
        raise ValueError(
            "model.target_object_embed_dim must match env.target_object_embed_dim. "
            f"Got model={model_target_embed_dim}, env={env_target_embed_dim}."
        )
    model_cfg["target_object_embed_dim"] = env_target_embed_dim

    target_cfg = cfg["training"].get("target") or {}
    target_vocab = (
        target_cfg.get("candidates")
        or target_cfg.get("cycle")
        or ([target_cfg["object_type"]] if target_cfg.get("object_type") else None)
        or tcfg.get("target_object_cycle")
        or ([tcfg["target_object_type"]] if tcfg.get("target_object_type") else None)
    )
    if not target_vocab:
        raise ValueError(
            "training.target must specify `candidates`, `cycle`, or `object_type` "
            "so the policy knows the target-object vocabulary."
        )
    target_vocab = sorted(set(target_vocab))
    print(f"[target-vocab] {len(target_vocab)} object(s): {target_vocab}")
    for env in envs:
        env.set_target_vocab(target_vocab)

    policy = AgentPolicy(
        n_actions=len(envs[0].action_list),
        encoder_name=cfg.get("visual_encoder", {}).get("model_name", "dinov2_vitb14"),
        target_object_vocab=target_vocab,
        **model_cfg,
        device=device,
    ).to(device)

    ref_env = envs[0]
    sense_action_idx   = ref_env.action_list.index("SENSE") if "SENSE" in ref_env.action_list else -1
    budget_feature_idx = ref_env.budget_idx

    ppo_kwargs = {
        k: v for k, v in agent_cfg.items()
        if k not in {"sense_prob", "sense_interval", "mode"}
    }

    if agent_type == "random_sensing":
        agent = RandomSensingBaseline(
            policy=policy, sense_action_idx=sense_action_idx,
            budget_feature_idx=budget_feature_idx,
            sense_prob=agent_cfg.get("sense_prob", 0.2),
            **ppo_kwargs,
        )
    elif agent_type == "fixed_schedule":
        agent = FixedScheduleBaseline(
            policy=policy, sense_action_idx=sense_action_idx,
            budget_feature_idx=budget_feature_idx,
            sense_interval=agent_cfg.get("sense_interval", 10),
            **ppo_kwargs,
        )
    elif agent_type == "low_res":
        agent = LowResBaseline(
            policy=policy, sense_action_idx=sense_action_idx, **ppo_kwargs,
        )
    elif agent_type == "high_res":
        agent = HighResBaseline(policy=policy, **ppo_kwargs)
    elif agent_type == "adaptive":
        agent = PPOAgent(policy=policy, **ppo_kwargs)
    else:
        raise ValueError(f"Unknown agent_type '{agent_type}'.")

    start_episode = 0
    resume_path = resume_path or tcfg.get("resume_from")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ckpt_vocab = ckpt.get("target_object_vocab")
        if ckpt_vocab is not None and ckpt_vocab != target_vocab:
            raise ValueError(
                f"Vocabulary mismatch with checkpoint.\n"
                f"  checkpoint: {ckpt_vocab}\n"
                f"  yaml:       {target_vocab}\n"
                "Either restore the original YAML, or train from scratch."
            )
        policy.load_state_dict(ckpt["policy"])
        agent.optimizer.load_state_dict(ckpt["optimizer"])
        start_episode = ckpt["episode"]
        print(f"Resumed from {resume_path} (episode {start_episode})")

    traj_logger = TrajectoryLogger.from_cfg(
        output_dir=tcfg["output_dir"],
        traj_cfg=cfg.get("trajectory_logging", {}),
    )

    try:
        rewards, episode_lengths, successes, spls = run_agent(
            envs=envs, agent=agent, cfg=cfg, device=device,
            start_episode=start_episode,
            fixed_target_object=fixed_target_object,
            target_object_cycle=target_object_cycle,
            trajectory_logger=traj_logger,
        )

        output_data(
            rewards=rewards, successes=successes,
            episode_lengths=episode_lengths, spls=spls,
            run_type=agent_label, params=agent_cfg,
            filename=f"{tcfg['output_dir']}/training_log.txt",
        )
    finally:
        if traj_logger is not None:
            traj_logger.close()
        for env in envs:
            env.close()
        if wandb_cfg.get("enabled", False) and wandb is not None:
            wandb.finish()

    if run_id:
        print(f"Cluster run id: {run_id}")

    return {
        "episodes":             len(rewards),
        "mean_reward":          float(np.mean(rewards))        if rewards else 0.0,
        "success_rate":         float(np.mean(successes))      if successes else 0.0,
        "mean_spl":             float(np.mean(spls))           if spls else 0.0,
        "mean_episode_length":  float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "output_log":           f"{tcfg['output_dir']}/training_log.txt",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="cfgs/train_rl.yaml", help="path to YAML config")
    parser.add_argument("--resume", default=None, help="path to checkpoint .pt to resume from")
    parser.add_argument("--agent", default=None, choices=list(AGENT_REGISTRY.keys()),
                        help="agent type (overrides config agent_type field)")

    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    run_training_pipeline(cfg=cfg, resume_path=args.resume, agent_type=args.agent)
