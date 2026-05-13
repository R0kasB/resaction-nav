#!/bin/bash
#SBATCH --job-name=resnav-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:25:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

set -euo pipefail

# Run from repo root if launched via `sbatch scripts/run_izar_smoke.sbatch`.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not available in PATH. Install uv or load the module/environment first."
  exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# On multi-GPU / multi-task nodes, reserve disjoint port blocks per local rank.
LOCAL_RANK="${SLURM_LOCALID:-0}"
NUM_ENVS="${NUM_ENVS:-2}"
BASE_PORT="${BASE_PORT:-$((8200 + 20 * LOCAL_RANK))}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID:-none}"
echo "SLURM_LOCALID=${LOCAL_RANK}"
echo "NUM_ENVS=${NUM_ENVS}"
echo "BASE_PORT=${BASE_PORT}"

# Generate a temporary config tuned for IZAR headless smoke testing.
GENERATED_CFG="$(python - <<'PY'
from pathlib import Path
import os
import yaml

cfg_path = Path("cfgs/train_rl.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

cfg.setdefault("training", {})
cfg["training"]["num_parallel_envs"] = int(os.environ.get("NUM_ENVS", "2"))
cfg["training"]["auto_cluster_run_id"] = True

cfg.setdefault("env", {})
controller_kwargs = dict(cfg["env"].get("controller_kwargs") or {})
controller_kwargs["platform"] = "CloudRendering"
controller_kwargs["port"] = int(os.environ["BASE_PORT"])
cfg["env"]["controller_kwargs"] = controller_kwargs

out_path = Path("cfgs/train_rl.izar.smoke.generated.yaml")
out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(str(out_path))
PY
)"

echo "Generated config: ${GENERATED_CFG}"
uv run python scripts/run_pipeline.py --cfg "${GENERATED_CFG}" --smoke
