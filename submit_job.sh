#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora
#SBATCH --output=ocpq-lora-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
# Run from the directory where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

# Activate the project virtual environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train

# The PyTorch wheel used by train_unsloth.py needs a newer libstdc++.so.6 than
# the default cluster runtime provides. Load the newer GCC toolchain and keep
# its lib directory ahead of the conda environment libraries.
module load GCC/13.3.0
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Fail fast if the env drifted away from the torch/torchao pair this job expects.
# python -c "import importlib.metadata as m, sys; v = m.version('torchao'); print('torchao', v); sys.exit(0 if v == '0.12.0' else 1)"

# Run script
python --version
OCPQ_FORCE_4BIT=1 OCPQ_FULL_PRECISION_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" python -u train_unsloth.py
