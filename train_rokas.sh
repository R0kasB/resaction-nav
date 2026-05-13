#!/bin/bash
#SBATCH --job-name=resnav-rokas
#SBATCH --output=logs/rokas_%A_%a.out
#SBATCH --error=logs/rokas_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

AGENTS=(adaptive low_res high_res random_sensing fixed_schedule)
AGENT=${AGENTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting agent: $AGENT"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# Initialize conda so its binaries (including uv if installed there) are on PATH.
CONDA_BASE=$(conda info --base 2>/dev/null || true)
if [ -n "$CONDA_BASE" ]; then
  eval "$("$CONDA_BASE/bin/conda" shell.bash hook 2>/dev/null)"
fi

# uv is typically installed in ~/.local/bin on this cluster.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Tried PATH: $PATH"
  exit 1
fi

echo "uv: $(which uv)"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
unset DISPLAY

# Each array task gets a unique port block to avoid collisions between agents.
BASE_PORT=$((8200 + 20 * SLURM_ARRAY_TASK_ID))
GENERATED_CFG="cfgs/train_rl.rokas_task${SLURM_ARRAY_TASK_ID}.generated.yaml"

python3 - <<PY
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("cfgs/train_rl.yaml").read_text(encoding="utf-8"))

cfg.setdefault("env", {})
ckw = dict(cfg["env"].get("controller_kwargs") or {})
ckw["platform"] = "CloudRendering"
ckw["port"] = ${BASE_PORT}
cfg["env"]["controller_kwargs"] = ckw

Path("${GENERATED_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "Config: ${GENERATED_CFG}  base_port: ${BASE_PORT}"

uv run python scripts/run_pipeline.py \
    --cfg "${GENERATED_CFG}" \
    --agent "$AGENT"

rm -f "${GENERATED_CFG}"
echo "Done: $AGENT"
