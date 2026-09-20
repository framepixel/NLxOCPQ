#!/bin/bash
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

set -euo pipefail

# Run from the directory where sbatch was invoked, and derive the project root
# relative to that. The script itself is copied into Slurm's spool directory,
# so BASH_SOURCE points at /var/spool/... and cannot be used for repo paths.
# Resolve the repo root from this script's own location, so the job works whether it was
# submitted from the repo root (`sbatch slurm/<this>.sh`) or from inside `slurm/`, and
# regardless of where $SLURM_SUBMIT_DIR points. Everything below is repo-root-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

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
pip install -r "$PROJECT_DIR/requirements/requirements.colab-serve.txt"

# Start the inference server
echo "Starting OCPQ LoRA inference server on $(hostname) at $(date)"
echo "Server will be available at http://localhost:8000 on compute node $(hostname)"
echo "Remember to tunnel port 8000 back to your local machine!"

cd "$PROJECT_DIR"

python "$PROJECT_DIR/src/colab_serve.py" \
  --host 0.0.0.0 \
  --port 8000 \
  --base-model unsloth/llama-3-8b-Instruct-bnb-4bit \
  --adapter-path ./models/lora_model_ocpq \
  --load-in-4bit

echo "Server shut down at $(date)"
