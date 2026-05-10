"""
Unified training pipeline launcher.

Usage:
  uv run python scripts/run_pipeline.py --cfg cfgs/train_rl.yaml --agent <agent-name>
  uv run python scripts/run_pipeline.py --cfg cfgs/train_rl.yaml --agent <agent-name> --smoke
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path


def _load_train_module():
    module_path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("train_script", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def apply_smoke_overrides(cfg: dict) -> dict:
    smoke_cfg = copy.deepcopy(cfg)

    smoke_cfg.setdefault("training", {})
    output_dir = Path(smoke_cfg["training"].get("output_dir", "output")) / "smoke"
    checkpoint_dir = Path(smoke_cfg["training"].get("checkpoint_dir", "checkpoints")) / "smoke"
    smoke_cfg["training"]["num_episodes"] = 10
    smoke_cfg["training"]["print_every"] = 1
    smoke_cfg["training"]["checkpoint_every"] = 9999
    smoke_cfg["training"]["output_dir"] = str(output_dir)
    smoke_cfg["training"]["checkpoint_dir"] = str(checkpoint_dir)
    smoke_cfg["training"]["resume_from"] = None

    smoke_cfg.setdefault("env", {})
    smoke_cfg["env"]["max_steps"] = min(int(smoke_cfg["env"].get("max_steps", 200)), 8)
    smoke_cfg["env"]["base_resolution"] = [64, 64]

    smoke_cfg.setdefault("wandb", {})
    smoke_cfg["wandb"]["enabled"] = False

    smoke_cfg.setdefault("huggingface", {})
    smoke_cfg["huggingface"]["push"] = False

    return smoke_cfg


def run_pipeline(cfg_path: str, resume: str | None = None, smoke: bool = False, agent_type: str | None = None) -> dict:
    train_module = _load_train_module()
    cfg = train_module.load_cfg(cfg_path)
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    return train_module.run_training_pipeline(cfg=cfg, resume_path=resume, agent_type=agent_type)


def parse_args() -> argparse.Namespace:
    _train        = _load_train_module()
    agent_choices = list(_train.AGENT_REGISTRY.keys())


    parser = argparse.ArgumentParser(description="Run the full resaction-nav training pipeline.")
    parser.add_argument("--cfg", default="cfgs/train_rl.yaml", help="Path to YAML config.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint .pt to resume from.")
    parser.add_argument("--agent", default=None, choices=agent_choices, help=f"Agent type (overrides config agent_type). Choices: {agent_choices}")

    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny end-to-end smoke pipeline with reduced episode/step counts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run_pipeline(cfg_path=args.cfg, resume=args.resume, smoke=args.smoke, agent_type=args.agent)    
    print(
        "Pipeline done | "
        f"episodes={summary['episodes']} "
        f"mean_reward={summary['mean_reward']:.3f} "
        f"success_rate={summary['success_rate']:.1%} "
        f"log={summary['output_log']}"
    )
