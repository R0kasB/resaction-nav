#!/bin/bash
#SBATCH --job-name=resaction-nav
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
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

cd /home/sauser/resaction-nav

mkdir -p logs
mkdir -p output/$AGENT
mkdir -p checkpoints/$AGENT

CONDA_BASE=$(conda info --base)

export PATH="$HOME/bin:$CONDA_BASE/bin:$PATH"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

unset DISPLAY
unset LD_LIBRARY_PATH

WANDB_RUN_NAME="$AGENT" uv run python scripts/run_pipeline.py \
    --cfg cfgs/train_rl.yaml \
    --agent "$AGENT"

echo "Done successfully: $AGENT"