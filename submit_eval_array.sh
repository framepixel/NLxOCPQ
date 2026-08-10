#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_array
#SBATCH --output=nlxocpq-eval-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --array=0-71%24

# 72 tasks = 36 conditions x 2 case-halves.
# 36 conditions = 6 models x 3 prompt strategies x 2 schema settings.
# The model always receives the full OCEL schema (entities + attributes + relations);
# schema richness is a fixed pipeline behaviour, not a design variable. With 150 test
# cases a full-condition shard (150 runs) would push the slowest model past the wall-time
# limit, so each condition is split into two case halves (75 runs each). %24 caps
# concurrent tasks. Recombine afterwards with:
#   python eval_testset.py --merge-shards eval_runs/shard_* --merge-output eval_runs/merged
#
# Task -> (model, strategy, schema, case-half) mapping:
#   MODEL_IDX    = TASK / 12       (0..5) 0,1=llama base/ft 2,3=phi base/ft 4,5=gemma base/ft
#   STRATEGY_IDX = (TASK % 12) / 4 (0..2) per eval_config.prompt_strategies order
#   SCHEMA_IDX   = (TASK % 4) / 2  (0..1) per eval_config.schema_enforced order
#   CASE_HALF    = TASK % 2        (0..1) case slice i of 2

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
MODEL_IDX=$(( TASK / 12 ))
STRATEGY_IDX=$(( (TASK % 12) / 4 ))
SCHEMA_IDX=$(( (TASK % 4) / 2 ))
CASE_HALF=$(( TASK % 2 ))

echo "Starting shard task=${TASK} (model=${MODEL_IDX} strategy=${STRATEGY_IDX} schema=${SCHEMA_IDX} case-half=${CASE_HALF}/2) at: $(date)"
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/eval_testset.py" \
    --config "$PROJECT_DIR/eval_config.json" \
    --model-idx "${MODEL_IDX}" \
    --strategy-idx "${STRATEGY_IDX}" \
    --schema-idx "${SCHEMA_IDX}" \
    --case-shard "${CASE_HALF}/2" \
    2>&1 | tee "nlxocpq-eval-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Shard ${TASK} exit code: $rc"
exit $rc
