#!/bin/bash
set -euo pipefail
#SBATCH --job-name=ocpq-serve
#SBATCH --partition=c23g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem-per-gpu=32GB
#SBATCH --time=01:00:00
#SBATCH --output=logs/%j_serve.log
#SBATCH --error=logs/%j_serve.err

# Resolve the project directory from this script location so sbatch can be run
# from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"

# Set up environment variables to minimize CUDA fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=${HF_TOKEN:-}
export CUDA_VISIBLE_DEVICES=0

# Create logs directory if it doesn't exist
mkdir -p logs

# Load necessary modules on RWTH HPC
module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

# Create and activate a Python virtual environment
echo "Setting up Python virtual environment..."
if [ ! -d "venv_ocpq" ]; then
    python -m venv venv_ocpq
fi

source venv_ocpq/bin/activate

# Upgrade pip and install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
pip install -r "$PROJECT_DIR/requirements.colab-serve.txt"

# Start the inference server
echo "Starting OCPQ LoRA inference server on $(hostname) at $(date)"
echo "Server will be available at http://localhost:8000 on compute node $(hostname)"
echo "Remember to tunnel port 8000 back to your local machine!"

cd "$PROJECT_DIR"

python "$PROJECT_DIR/colab_serve.py" \
  --host 127.0.0.1 \
  --port 8000 \
  --base-model unsloth/llama-3-8b-Instruct-bnb-4bit \
  --adapter-path ./lora_model_ocpq \
  --load-in-4bit

echo "Server shut down at $(date)"
