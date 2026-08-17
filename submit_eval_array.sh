#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_array
#SBATCH --output=nlxocpq-eval-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
# 64G, matching the training jobs. Parsing an ocpq_cli result export costs a measured 11.7x its
# size in RAM; the largest gold export is 745 MB (~8.5 GB resident), and an over-selecting
# prediction can exceed gold. 32G left no margin for that plus the 4-bit model.
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# 14h: the previous 10h was sized for 150 test cases (75 per case-half). The testset is now 169
# (85/84 per half), ~13% more work, and Ministral 3 14B is the slowest model in the set.
#SBATCH --time=14:00:00
#SBATCH --array=0-71%24
# Required: the personal allocation is exhausted (-101%, 0 GPU-h consumable) while the project
# still holds ~945 GPU-h. Without this the array is rejected at submission.
#SBATCH --account=thes2358
# Site default may be --export=NONE; options are passed as arguments regardless, this is belt-and-braces.
#SBATCH --export=ALL

# 72 tasks = 36 conditions x 2 case-halves.
# 36 conditions = 6 models x 3 prompt strategies x 2 schema settings.
# The model always receives the full OCEL schema (entities + attributes + relations);
# schema richness is a fixed pipeline behaviour, not a design variable. With 169 test
# cases a full-condition shard (169 runs) would push the slowest model past the wall-time
# limit, so each condition is split into two case halves (85 and 84 runs). %24 caps
# concurrent tasks. Recombine afterwards with:
#   python eval_testset.py --merge-shards eval_runs/shard_* --merge-output eval_runs/merged
# The merge also unions each shard's result_store/, so the merged run can be rescored later
# with --recompute without re-running anything.
#
# Task -> (model, strategy, schema, case-half) mapping:
#   MODEL_IDX    = TASK / 12       (0..5) 0,1=llama31-8b base/ft
#                                         2,3=ministral3-14b base/ft
#                                         4,5=gemma3-4b base/ft
#   STRATEGY_IDX = (TASK % 12) / 4 (0..2) per eval_config.prompt_strategies order
#   SCHEMA_IDX   = (TASK % 4) / 2  (0..1) per eval_config.schema_enforced order
#   CASE_HALF    = TASK % 2        (0..1) case slice i of 2
# Verify this mapping against expand_models() without a GPU:
#   python check_shard_map.py
#
# To resume shards that died, resubmit with the --resume ARGUMENT:
#   sbatch --array=48-71%24 submit_eval_array.sh --resume
# Without it a resubmitted task starts from zero, because run_id embeds a timestamp and so
# names a fresh directory. It must be an argument, not OCPQ_RESUME=1: this cluster does not
# propagate the submitting environment into jobs.

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

# Same stack as the fine-tuning runs, so the adapters are loaded by what wrote them.
# A conda env, not the old $HOME/envs/nlxocpq-venv virtualenv.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# The eval environment is a clone of the fine-tuning env plus scikit-learn and openai,
# which the eval path needs and training does not. Built by make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

# Fail here rather than 40 minutes in: LLM.py imports sklearn and openai unguarded at module
# level, so a missing one kills the job before any model loads.
if [ ! -f "$PROJECT_DIR/check_eval_deps.py" ]; then
  echo "FATAL: $PROJECT_DIR/check_eval_deps.py not found — copy the current NLxOCPQ files to the cluster." >&2
  exit 1
fi
python "$PROJECT_DIR/check_eval_deps.py" --verbose || exit 1

# SLURM does not source ~/.bashrc — load the token explicitly.
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

# Offline by design, as in the training jobs: a multi-day matrix must not be able to die on a hub
# outage or rate limit partway through. Requires the cache to be populated first:
#     bash prefetch_models.sh && VERIFY=1 bash prefetch_models.sh
# Set OCPQ_ALLOW_HUB=1 to disable if a model is genuinely missing from the cache.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi

# Keep ocpq_cli scratch on node-local storage. Correctness does not depend on it (every
# invocation gets its own mkdtemp), but 24 concurrent tasks writing multi-hundred-MB exports
# to a shared filesystem is a throughput and quota problem.
export OCPQ_CLI_SCRATCH_DIR="${OCPQ_CLI_SCRATCH_DIR:-${TMPDIR:-/tmp}/ocpq-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID}"
mkdir -p "$OCPQ_CLI_SCRATCH_DIR"
trap 'rm -rf "$OCPQ_CLI_SCRATCH_DIR"' EXIT

# The full schema block reaches 10,493 estimated tokens (angular_github_commits_ocel.xml). The
# old 3500 default silently dropped sections for that log and for bpic2017, contradicting the
# "every model always receives the complete schema" design.
export OCPQ_SCHEMA_TOKEN_BUDGET="${OCPQ_SCHEMA_TOKEN_BUDGET:-17000}"

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

# Resume, opt-in. A dead shard's checkpoint is useless unless the resubmitted task is pointed at
# the directory it wrote, and run_id embeds a timestamp so the default is always a fresh one.
#
# Requested by ARGUMENT, not environment: this cluster does not propagate the submitting
# environment into jobs (verified in jobs 3015021 and 3015720), so `OCPQ_RESUME=1 sbatch ...`
# would have been silently ignored and a dead 14-hour shard would have restarted from zero —
# exactly what resume exists to prevent. The env var is kept as a fallback.
#
#   sbatch --array=48-71%24 submit_eval_array.sh --resume
OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
for arg in "$@"; do
  case "$arg" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    *) echo "Warning: ignoring unrecognised argument '$arg'" >&2 ;;
  esac
done
echo "Resume requested: ${OCPQ_RESUME_REQUESTED}"

RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  SHARD_GLOB="$PROJECT_DIR/eval_runs/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}_c${CASE_HALF}of2_*"
  # Newest by mtime. Ambiguity is not resolved silently: if nothing matches we stop, because
  # continuing would quietly restart a 14-hour shard from zero.
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: OCPQ_RESUME=1 but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  MATCHES="$(ls -1d $SHARD_GLOB 2>/dev/null | wc -l)"
  echo "Resuming from ${RESUME_DIR} (${MATCHES} candidate(s), newest chosen)"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting shard task=${TASK} (model=${MODEL_IDX} strategy=${STRATEGY_IDX} schema=${SCHEMA_IDX} case-half=${CASE_HALF}/2) at: $(date)"
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/eval_testset.py" \
    --config "$PROJECT_DIR/eval_config.json" \
    --model-idx "${MODEL_IDX}" \
    --strategy-idx "${STRATEGY_IDX}" \
    --schema-idx "${SCHEMA_IDX}" \
    --case-shard "${CASE_HALF}/2" \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "nlxocpq-eval-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Shard ${TASK} exit code: $rc"
exit $rc
