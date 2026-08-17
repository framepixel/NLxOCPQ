#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_model_check
#SBATCH --output=nlxocpq-model-check-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --account=thes2358

# Load each model once and generate one short reply. Minutes, against the ~1.5 h a full smoke
# run costs, so a loader or adapter problem is found before booking the long job.
#
#   sbatch submit_model_check.sh              # all six models
#   OCPQ_MODEL_IDX="3" sbatch submit_model_check.sh    # just ministral3-14b-finetuned
#
# Uses the same environment block as the eval jobs, so a green result here means the eval jobs
# will load the same way.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

if [ -d "$SLURM_SUBMIT_DIR/NLxOCPQ" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR/NLxOCPQ"
else
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
fi

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi
export OCPQ_SCHEMA_TOKEN_BUDGET="${OCPQ_SCHEMA_TOKEN_BUDGET:-17000}"

echo "Checking model load at: $(date)"
echo "Python: $(which python)"; python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo

# OCPQ_MODEL_CHECK_CONFIG points at a candidate config, so a model can be proven to load before
# ~8h of training is queued against it. Defaults to the real matrix config.
CHECK_CONFIG="${OCPQ_MODEL_CHECK_CONFIG:-eval_config.json}"
case "$CHECK_CONFIG" in
  /*) ;;
  *) CHECK_CONFIG="$PROJECT_DIR/$CHECK_CONFIG" ;;
esac
if [ ! -f "$CHECK_CONFIG" ]; then
  echo "FATAL: config not found: $CHECK_CONFIG" >&2
  exit 1
fi
echo "Config: $CHECK_CONFIG"

if [ -n "${OCPQ_MODEL_IDX:-}" ]; then
  # shellcheck disable=SC2086
  python "$PROJECT_DIR/check_model_load.py" --config "$CHECK_CONFIG" --model-idx ${OCPQ_MODEL_IDX}
else
  python "$PROJECT_DIR/check_model_load.py" --config "$CHECK_CONFIG"
fi
