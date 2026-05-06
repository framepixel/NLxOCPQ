#!/usr/bin/env zsh
#SBATCH --job-name=query_run
#SBATCH --output=query-run-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:06:00

set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
# Run from the directory where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

# CRITICAL: Purge all modules first, then reload only what we need
module purge
module load GCC/13.3.0
module load CUDA/12.4.0

# Activate the project virtual environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train

# Set compilation flags to prevent Intel-optimized symbol dependencies
export CFLAGS="-march=x86-64 -mtune=generic"
export CXXFLAGS="-march=x86-64 -mtune=generic"
export RUSTFLAGS="-C target-cpu=generic"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG
export LD_LIBRARY_PATH=""

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Run script
python --version
python cli.py --backend local-gpu --model lora_model_ocpq --prompt-strategy schema-allowed-entities-only --ocel ../OCEL_Logs/hospital_log_high.json  "For every person, the diagnosis must occur exactly once."
