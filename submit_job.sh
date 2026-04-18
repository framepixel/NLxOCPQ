#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora
#SBATCH --output=ocpq-lora-%J.out
#SBATCH --partition=c25g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00

set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load the cluster Python/CUDA stack if these module names exist on the cluster.
# If the module names differ, replace them with the RWTH-provided equivalents.
module load python/3.11 cuda/12.1

# Activate the project virtual environment.
# Create it once on the cluster with: python3 -m venv .venv
source .venv/bin/activate

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Run script
python -u train_unsloth.py
