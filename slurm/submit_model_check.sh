#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_model_check
#SBATCH --output=logs/nlxocpq-model-check-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --account=thes2358
# Site default may be --export=NONE; options are passed as arguments regardless, this is belt-and-braces.
#SBATCH --export=ALL

# Load each model once and generate one short reply. Minutes, against the ~1.5 h a full smoke
# run costs, so a loader or adapter problem is found before booking the long job.
#
#   sbatch slurm/submit_model_check.sh                          # all six models
#   sbatch slurm/submit_model_check.sh --model-idx 3            # just ministral3-14b-finetuned
#   sbatch slurm/submit_model_check.sh --config configs/eval_config.candidate.json --model-idx 0
#
# Options are ARGUMENTS, not environment variables: this cluster does not propagate the
# submitting environment into jobs.
#
# Uses the same environment block as the eval jobs, so a green result here means the eval jobs
# will load the same way.

set -euo pipefail

# Resolve the repo root from this script's own location, so the job works whether it was
# submitted from the repo root (`sbatch slurm/<this>.sh`) or from inside `slurm/`, and
# regardless of where $SLURM_SUBMIT_DIR points. Everything below is repo-root-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

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

# Options come from ARGUMENTS, not the environment. This cluster does not propagate the
# submitting environment into jobs: `OCPQ_MODEL_IDX=... sbatch ...` arrived empty in jobs
# 3015021 and 3015720, which silently checked all six models against the default config instead
# of the one requested. Anything after the script name is always passed through by SLURM, so
# that is the reliable channel. The environment variables are kept as a fallback for anyone
# running this script directly rather than through sbatch.
#
#   sbatch slurm/submit_model_check.sh                                   # all models, matrix config
#   sbatch slurm/submit_model_check.sh --model-idx 4 5                   # just those two
#   sbatch slurm/submit_model_check.sh --config configs/eval_config.candidate.json --model-idx 0
CHECK_ARGS=("$@")
if [ ${#CHECK_ARGS[@]} -eq 0 ]; then
  [ -n "${OCPQ_MODEL_CHECK_CONFIG:-}" ] && CHECK_ARGS+=(--config "$OCPQ_MODEL_CHECK_CONFIG")
  # shellcheck disable=SC2206
  [ -n "${OCPQ_MODEL_IDX:-}" ] && CHECK_ARGS+=(--model-idx ${OCPQ_MODEL_IDX})
fi
echo "check_model_load args: ${CHECK_ARGS[*]:-(none — all models, default config)}"

python "$PROJECT_DIR/src/check_model_load.py" "${CHECK_ARGS[@]}"
