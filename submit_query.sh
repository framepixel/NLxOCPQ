#!/usr/bin/env zsh
#SBATCH --job-name=query_run
#SBATCH --output=query-run-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:04:00

set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
# Run from the directory where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

# Activate the project virtual environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Fail fast if the env drifted away from the torch/torchao pair this job expects.
# python -c "import importlib.metadata as m, sys; v = m.version('torchao'); print('torchao', v); sys.exit(0 if v == '0.12.0' else 1)"

# Run script
python --version
python cli.py --backend local-gpu --model lora_model_ocpq_v0-1 --prompt-strategy schema-allowed-entities-only --ocel ../OCEL_Logs/hospital_log_high.json  "For any person, the diagnosis must be completed within 3 days after the registration at ER."
