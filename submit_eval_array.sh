#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_array
#SBATCH --output=nlxocpq-eval-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --array=0-35

# 36 tasks = 6 models x 3 prompt strategies x 2 schema settings.
# Each task evaluates ONE (model, strategy, schema) cell over all test cases,
# so even the heaviest model finishes within the wall-time limit. Recombine the
# per-shard run dirs afterwards with:  eval_testset.py --merge-shards <dirs...>
#
# Task -> (model, strategy, schema) mapping:
#   MODEL_IDX    = TASK / 6      (0..5)  0,1=llama base/ft  2,3=phi base/ft  4,5=gemma base/ft
#   STRATEGY_IDX = (TASK % 6) / 2 (0..2) per eval_config.prompt_strategies order
#   SCHEMA_IDX   = TASK % 2       (0..1) per eval_config.schema_enforced order

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

# SLURM does not source ~/.bashrc — load the token explicitly.
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

# Pre-compile the Rust CLI binary once (no-op if already up to date).
OCPQ_CLI_DIR="$SLURM_SUBMIT_DIR/OCPQ/backend/ocpq_cli"
if [ -d "$OCPQ_CLI_DIR" ]; then
  echo "Pre-compiling OCPQ Rust CLI..."
  cargo build -q --release --manifest-path "$OCPQ_CLI_DIR/Cargo.toml"
fi

TASK="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=$(( TASK / 6 ))
REM=$(( TASK % 6 ))
STRATEGY_IDX=$(( REM / 2 ))
SCHEMA_IDX=$(( REM % 2 ))

echo "Starting shard task=${TASK} (model=${MODEL_IDX} strategy=${STRATEGY_IDX} schema=${SCHEMA_IDX}) at: $(date)"
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/eval_testset.py" \
    --config "$PROJECT_DIR/eval_config.json" \
    --model-idx "${MODEL_IDX}" \
    --strategy-idx "${STRATEGY_IDX}" \
    --schema-idx "${SCHEMA_IDX}" \
    2>&1 | tee "nlxocpq-eval-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Shard ${TASK} exit code: $rc"
exit $rc
