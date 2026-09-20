#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_selfcons
#SBATCH --output=logs/nlxocpq-selfcons-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# 8 case-shards per arm keeps every task under ~6h against this limit. Measured on the
# 2026-08-29 smoke: 12.7-96.6 s/run depending on arm and case, so at 4 shards the slowest
# base-arm task projected 10.8-14.1h and would have been cancelled at the limit. Gold
# executes once per case and is cached in-process, so the 10x repeat multiplier applies to
# generation and prediction execution only, not to ground truth.
#SBATCH --time=14:00:00
#SBATCH --array=0-15%8
#SBATCH --account=thes2358
#SBATCH --export=ALL

# SELF-CONSISTENCY over execution results.
#
# Draws 10 samples per question at temperature 0.7 (configs/eval_config.selfconsistency.json),
# so the modal RESULT SET can be elected afterwards by src/selfconsistency.py. The matrix ran
# greedy with one sample per case, so nothing here has been measured before.
#
# TWO ARMS, each at a configuration it can actually perform in. Same model family, so the only
# variable is fine-tuning:
#   tasks 0-3  llama31-8b-finetuned / allowed-entities-only / schema=on   (greedy F1 .1760, 70.4% exec)
#   tasks 4-7  llama31-8b-base      / chain-of-thought-plus / schema=on   (greedy F1 .1023, 45.0% exec)
#
# The base arm deliberately does NOT use allowed-entities-only. That cell executed 0 of 1352 runs
# across the whole matrix -- there would be nothing to vote over.
#
# Indices are hardcoded rather than derived from TASK arithmetic (as submit_eval_array.sh does)
# because the two arms use DIFFERENT strategies, which no single cross-product expresses without
# also running the two vacuous cells.
#
#   sbatch --array=0-15%8 slurm/submit_selfconsistency.sh
#   sbatch --array=3 slurm/submit_selfconsistency.sh --resume     # one shard, resumed
#
# Then:
#   python src/eval_testset.py --merge-shards eval_runs/selfconsistency/shard_* #       --merge-output eval_runs/selfconsistency/merged
#   python src/selfconsistency.py --run eval_runs/selfconsistency/merged

set -u
# SLURM copies the batch script into its spool directory before executing it, so
# ${BASH_SOURCE[0]} points at /var/spool/slurm/... and NOT at this file in the repo. The reorg
# (019c2f2) replaced the previous `cd "${SLURM_SUBMIT_DIR:?}"` with a BASH_SOURCE lookup that
# had never been run under sbatch; the first job to try it, 3294532, died on all 8 tasks with
# "FATAL: /var/spool/slurm/src/check_eval_deps.py not found".
#
# $SLURM_SUBMIT_DIR (where sbatch was invoked) is therefore the authority under SLURM, with
# BASH_SOURCE kept only for running this script directly from a shell. Each candidate is
# validated against a marker file, so a wrong guess fails HERE rather than as a confusing
# missing-file error somewhere downstream.
_ocpq_resolve_project_dir() {
  local cand
  for cand in "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:-}/.." \
              "$(dirname "${BASH_SOURCE[0]}")" "$(dirname "${BASH_SOURCE[0]}")/.."; do
    [ -n "$cand" ] || continue
    [ -d "$cand" ] || continue
    if [ -f "$cand/src/eval_testset.py" ]; then (cd "$cand" && pwd); return 0; fi
  done
  return 1
}
if ! PROJECT_DIR="$(_ocpq_resolve_project_dir)"; then
  echo "FATAL: could not locate the NLxOCPQ repo root (looked for src/eval_testset.py)." >&2
  echo "       SLURM_SUBMIT_DIR = ${SLURM_SUBMIT_DIR:-<unset>}" >&2
  echo "       BASH_SOURCE[0]   = ${BASH_SOURCE[0]}" >&2
  echo "       Submit from the repo root: sbatch slurm/<script>.sh" >&2
  exit 1
fi
SCRIPT_DIR="$PROJECT_DIR/slurm"
cd "$PROJECT_DIR"

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
# which the eval path needs and training does not. Built by scripts/make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

if [ ! -f "$PROJECT_DIR/src/check_eval_deps.py" ]; then
  echo "FATAL: $PROJECT_DIR/src/check_eval_deps.py not found — copy the current NLxOCPQ files to the cluster." >&2
  exit 1
fi
python "$PROJECT_DIR/src/check_eval_deps.py" --verbose || exit 1

# SLURM does not source ~/.bashrc — load the token explicitly.
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN not set — ensure HF provider runs are disabled or token provided in config."
fi

# Offline by design, as in the training jobs: a multi-day matrix must not be able to die on a hub
# outage or rate limit partway through. Requires the cache to be populated first:
#     bash scripts/prefetch_models.sh && VERIFY=1 bash scripts/prefetch_models.sh
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

TASK="${SLURM_ARRAY_TASK_ID}"

# Case-shards PER ARM. The array must be 0-$((2*SHARDS_PER_ARM-1)).
# 8, not 4: the 10-minute smoke (job 3297883) projected the slowest base-arm task at
# 10.8-14.1h against the 14h limit. Each case still runs exactly once, so splitting finer
# duplicates nothing but the ~2.5min startup -- it does not add GPU work.
SHARDS_PER_ARM=8
ARM=$(( TASK / SHARDS_PER_ARM ))
SHARD=$(( TASK % SHARDS_PER_ARM ))

EXPECTED_MAX=$(( 2 * SHARDS_PER_ARM - 1 ))
if [ "$TASK" -gt "$EXPECTED_MAX" ]; then
  echo "FATAL: task ${TASK} exceeds the ${SHARDS_PER_ARM}-shards-per-arm layout (max ${EXPECTED_MAX})." >&2
  echo "       Submit with --array=0-${EXPECTED_MAX}%8 or update SHARDS_PER_ARM." >&2
  exit 1
fi

if [ "$ARM" -eq 0 ]; then
  MODEL_IDX=1; STRATEGY_IDX=0; ARM_NAME="llama31-8b-finetuned/allowed-entities-only"
else
  MODEL_IDX=0; STRATEGY_IDX=2; ARM_NAME="llama31-8b-base/chain-of-thought-plus"
fi
SCHEMA_IDX=0   # eval_config.schema_enforced is [true, false]; 0 == on

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
for arg in "$@"; do
  case "$arg" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    *) echo "Warning: ignoring unrecognised argument '$arg'" >&2 ;;
  esac
done

RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  SHARD_GLOB="$PROJECT_DIR/eval_runs/selfconsistency/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}_c${SHARD}of${SHARDS_PER_ARM}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: --resume given but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_DIR}"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting self-consistency task=${TASK} arm=${ARM_NAME} (model=${MODEL_IDX} strategy=${STRATEGY_IDX} schema=${SCHEMA_IDX} case-shard=${SHARD}/${SHARDS_PER_ARM}) at: $(date)"
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.selfconsistency.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     --case-shard "${SHARD}/${SHARDS_PER_ARM}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-selfcons-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Self-consistency shard ${TASK} exit code: $rc"
exit $rc
