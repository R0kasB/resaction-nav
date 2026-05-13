#!/bin/bash
#SBATCH --job-name=resnav-sweep
#SBATCH --output=logs/sweep_%j.out
#SBATCH --error=logs/sweep_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --mem=375G
#SBATCH --time=12:00:00

# Run make_sweep.py first to generate cfgs/sweep/run_*.yaml, then:
#   sbatch train_sweep.sh

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

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

run_config() {
    local i=$1
    local gpu=$2
    local cfg="cfgs/sweep/run_$(printf '%03d' "$i").yaml"
    echo "=== Task $i  GPU $gpu  Config: $cfg ==="
    CUDA_VISIBLE_DEVICES=$gpu uv run python scripts/run_pipeline.py \
        --cfg "$cfg" \
        --agent adaptive \
        > "logs/sweep_${SLURM_JOB_ID}_task${i}.out" 2>&1
    echo "=== Done: task $i ==="
}

# Run all 18 configs at once: even → GPU 0, odd → GPU 1
pids=()
for i in $(seq 0 17); do
    gpu=$((i % 2))
    run_config "$i" "$gpu" &
    pids+=($!)
done
wait "${pids[@]}"

echo "All sweep configs complete."
