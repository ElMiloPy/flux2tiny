#!/usr/bin/env bash
# flux2tiny — setup environment using conda
# Requires miniconda/anaconda to be installed
set -euo pipefail

ENV_NAME="${1:-flux2tiny}"

echo "=== flux2tiny conda environment setup ==="

# Create conda env with Python 3.11
echo "Creating conda env: ${ENV_NAME}..."
conda create -n "${ENV_NAME}" python=3.11 -y

echo "Activating env..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# Install PyTorch directly via PyTorch pip index to avoid Conda MKL symbol conflicts
echo "Installing PyTorch CUDA 12.4 via pip..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install core pip packages
echo "Installing pip dependencies..."
pip install accelerate safetensors Pillow tqdm datasets

# Bleeding-edge diffusers (required for Flux2KleinPipeline)
echo "Installing diffusers from git (for Flux2KleinPipeline)..."
pip install git+https://github.com/huggingface/diffusers.git

# Bleeding-edge transformers (required for LlamaForCausalLM / Qwen3 compat)
echo "Installing transformers from git..."
pip install git+https://github.com/huggingface/transformers.git

echo ""
echo "=== Setup complete ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. python train_adapter.py   # Train the projection adapter"
echo "  2. python generate.py        # Generate images"
