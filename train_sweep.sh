#!/bin/bash
#SBATCH --job-name=resnav-sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --array=0-17        # update this after running make_sweep.py
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00

# Run make_sweep.py first to generate cfgs/sweep/run_*.yaml, then:
#   sbatch train_sweep.sh

set -euo pipefail

CFG="cfgs/sweep/run_$(printf '%03d' "$SLURM_ARRAY_TASK_ID").yaml"

echo "Task: $SLURM_ARRAY_TASK_ID  Config: $CFG"
echo "Job ID: $SLURM_JOB_ID  Node: $(hostname)"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# vulkaninfo stub (same logic as train_rokas.sh)
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

uv run python scripts/run_pipeline.py \
    --cfg "$CFG" \
    --agent adaptive

echo "Done: task $SLURM_ARRAY_TASK_ID"
