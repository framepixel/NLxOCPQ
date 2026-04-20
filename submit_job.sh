#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora
#SBATCH --output=ocpq-lora-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00

set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
# Run from the directory where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

# Activate the project virtual environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate apm_env

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Run script
python --version
python -u train_unsloth.py
