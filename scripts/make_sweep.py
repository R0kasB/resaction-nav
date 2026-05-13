"""
Generate a parameter-grid sweep: one YAML config per combination.

Edit GRID below to change the search space, then run:
    uv run python scripts/make_sweep.py

The script prints the exact --array range to paste into train_sweep.sh.

Usage:
    uv run python scripts/make_sweep.py
    uv run python scripts/make_sweep.py --base cfgs/train_rl.yaml --out cfgs/sweep
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path

import yaml

# ── grid definition ──────────────────────────────────────────────────────────
GRID: dict[str, list] = {
    "entropy_coef":   [0.01, 0.05, 0.1],   # exploration pressure
    "distance_scale": [0.01, 0.1,  0.5],   # approach-shaping reward
    "lr":             [1e-4, 3e-4],         # learning rate
}

SWEEP_EPISODES   = 1500   # enough to see if learning starts, cheaper than full 5000
SWEEP_PARALLEL   = 4      # parallel envs per task
BASE_PORT        = 9200   # above train_rokas.sh range (8200-8299)
PORT_BLOCK       = 20     # ports reserved per task (covers SWEEP_PARALLEL + headroom)
# ─────────────────────────────────────────────────────────────────────────────


def make_configs(base_cfg_path: str, out_dir: str) -> int:
    base = yaml.safe_load(Path(base_cfg_path).read_text(encoding="utf-8"))
    out  = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys   = list(GRID.keys())
    combos = list(itertools.product(*GRID.values()))

    index: list[dict] = []
    for task_id, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg    = copy.deepcopy(base)

        # training overrides
        cfg["training"]["num_episodes"]        = SWEEP_EPISODES
        cfg["training"]["num_parallel_envs"]   = SWEEP_PARALLEL
        cfg["training"]["auto_cluster_run_id"] = True
        cfg["training"]["checkpoint_every"]    = SWEEP_EPISODES  # checkpoint at end only

        # reward overrides: success must clearly dominate fail+steps (-1.0 - 0.4 max)
        cfg["env"]["reward_cfg"]["success_reward"] = 10.0

        # agent params
        cfg["agent"]["entropy_coef"] = params["entropy_coef"]
        cfg["agent"]["lr"]           = params["lr"]

        # reward shaping
        cfg["env"]["reward_cfg"]["distance_scale"] = params["distance_scale"]

        # unique port block so parallel array tasks never collide
        port = BASE_PORT + PORT_BLOCK * task_id
        cfg["env"].setdefault("controller_kwargs", {})
        cfg["env"]["controller_kwargs"]["platform"] = "CloudRendering"
        cfg["env"]["controller_kwargs"]["port"]     = port

        # descriptive W&B run name
        tag = (
            f"ec{params['entropy_coef']}"
            f"_ds{params['distance_scale']}"
            f"_lr{params['lr']:.0e}"
        )
        cfg["wandb"]["run_name"] = f"sweep-{task_id:03d}-{tag}"
        cfg["wandb"]["group"]    = "sweep"

        path = out / f"run_{task_id:03d}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        index.append({"task_id": task_id, "yaml": str(path), **params})
        print(f"  [{task_id:3d}]  {tag}")

    (out / "grid.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    n = len(combos)
    print(f"\n{n} configs written to {out}/")
    print(f"Update train_sweep.sh:  #SBATCH --array=0-{n - 1}")
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="cfgs/train_rl.yaml", help="Base config to build from.")
    parser.add_argument("--out",  default="cfgs/sweep",         help="Output directory for generated configs.")
    args = parser.parse_args()
    make_configs(args.base, args.out)
