#!/usr/bin/env bash
set -euo pipefail

eval "$(conda shell.bash hook)"

conda create -n resaction-nav python=3.13 -y
conda activate resaction-nav

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ai2thor huggingface-hub numpy pyyaml transformers wandb tqdm
pip install -e .

wandb login

echo "Setup complete. Activate with: conda activate resaction-nav"
