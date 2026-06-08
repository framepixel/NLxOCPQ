#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_array
#SBATCH --output=nlxocpq-eval-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-5

# One task per expanded model:
#   0 = llama3-8b-base       1 = llama3-8b-finetuned
#   2 = gpt-oss-20b          3 = gpt-oss-20b-finetuned
#   4 = gemma3-4b            5 = gemma3-4b-finetuned

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

source "$HOME/envs/nlxocpq-venv/bin/activate"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:$HOME/envs/nlxocpq-venv/lib:${LD_LIBRARY_PATH:-}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

# Pre-compile the Rust CLI binary once (no-op if already up to date).
OCPQ_CLI_DIR="$SLURM_SUBMIT_DIR/OCPQ/backend/ocpq_cli"
if [ -d "$OCPQ_CLI_DIR" ]; then
  echo "Pre-compiling OCPQ Rust CLI..."
  cargo build -q --release --manifest-path "$OCPQ_CLI_DIR/Cargo.toml"
fi

echo "Starting shard model-idx=${SLURM_ARRAY_TASK_ID} at: $(date)"
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/eval_testset.py" \
    --config "$PROJECT_DIR/eval_config.json" \
    --model-idx "${SLURM_ARRAY_TASK_ID}" \
    2>&1 | tee "nlxocpq-eval-shard${SLURM_ARRAY_TASK_ID}.log"
rc=${PIPESTATUS[0]:-0}
echo "Shard ${SLURM_ARRAY_TASK_ID} exit code: $rc"
exit $rc
