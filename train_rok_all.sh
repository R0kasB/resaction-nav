#!/bin/bash
#SBATCH --job-name=resnav-rok
#SBATCH --output=logs/rok_%A_%a.out
#SBATCH --error=logs/rok_%A_%a.err
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

CONFIGS=(rok_adaptive rok_multi rok_multi_adaptive rok_multi_adaptive_worst)
CFG_NAME=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
CFG="cfgs/train_rl.${CFG_NAME}.yaml"

echo "Config: $CFG"
echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$HOME/bin" .venv/bin
for _stub_dest in "$HOME/bin/vulkaninfo" ".venv/bin/vulkaninfo"; do
cat > "$_stub_dest" << 'STUB'
#!/bin/bash
_re='GPU ([0-9]+):.+UUID: GPU-([^)]+)'
nvidia-smi -L 2>/dev/null | while IFS= read -r line; do
    if [[ "$line" =~ $_re ]]; then
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

# Each array task gets its own port to avoid collisions
BASE_PORT=$((8200 + 100 * SLURM_ARRAY_TASK_ID))
GENERATED_CFG="cfgs/train_rl.${CFG_NAME}.generated.yaml"

uv run python - <<PY
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("${CFG}").read_text(encoding="utf-8"))
cfg.setdefault("env", {})
ckw = dict(cfg["env"].get("controller_kwargs") or {})
ckw["platform"] = "CloudRendering"
ckw["port"] = ${BASE_PORT}
cfg["env"]["controller_kwargs"] = ckw
Path("${GENERATED_CFG}").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

echo "Generated config: ${GENERATED_CFG}  port: ${BASE_PORT}"
uv run python scripts/train.py --cfg "${GENERATED_CFG}" --agent high_res

rm -f "${GENERATED_CFG}"
echo "Done: ${CFG_NAME}"
