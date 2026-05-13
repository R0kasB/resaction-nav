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

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# Stub vulkaninfo: ai2thor calls subprocess.run(["vulkaninfo"]) and fails if it
# returns non-zero or is missing. Real vulkaninfo isn't installed, so we emit
# the UUIDs from nvidia-smi in the exact format ai2thor expects.
# Write into .venv/bin so uv run always finds it (uv puts .venv/bin first in PATH).
# Also write to $HOME/bin for train_all.sh which already has $HOME/bin in PATH.
mkdir -p "$HOME/bin" .venv/bin
for _stub_dest in "$HOME/bin/vulkaninfo" ".venv/bin/vulkaninfo"; do
cat > "$_stub_dest" << 'STUB'
#!/bin/bash
nvidia-smi -L 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" =~ GPU\ ([0-9]+):.+UUID:\ GPU-([^)]+) ]]; then
        echo "GPU${BASH_REMATCH[1]}:"
        echo "        deviceUUID = ${BASH_REMATCH[2]}"
    fi
done
STUB
chmod +x "$_stub_dest"
done
unset _stub_dest
export PATH="$HOME/bin:$PATH"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json
echo "vulkaninfo stub: $(which vulkaninfo 2>&1)"
echo "stub output: $($HOME/bin/vulkaninfo | head -2 || echo '(empty)')"

# Each array task gets a unique port block to avoid collisions between agents.
BASE_PORT=$((8200 + 20 * SLURM_ARRAY_TASK_ID))
GENERATED_CFG="cfgs/train_rl.rokas_task${SLURM_ARRAY_TASK_ID}.generated.yaml"

uv run python - <<PY
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
