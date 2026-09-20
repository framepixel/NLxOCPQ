#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_typedabl
#SBATCH --output=logs/nlxocpq-typedabl-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --account=thes2358
#SBATCH --export=ALL

# TYPE-AWARE VALUE FILTERS -- the first NON-ORACLE constraint in this project.
#
# WITHDRAWN PREMISE, NOT RUN (notes 35.4, 35.6). Every type disagreement quoted below sat on an
# attribute the cluster's STALE corpus declared as a string; on the corrected corpus the model is
# type-correct on 8 of 9 attribute filters without this constraint. The header is kept as written,
# as the pre-registration of an experiment that was deliberately not spent.
#
# notes 34.3: the model collapses attribute comparisons into string membership. `Age` is declared
# Integer and build_schema_block says so, yet "older than 40" comes out as String{is_in:["40"]}
# where gold has Integer{min:41}. Across the 2026-09-11 run: Boolean->String 16 times,
# Integer->String 11 times. Gold contradicts a declared type 0 times in 117.
#
# The information was never missing, so this adds a CONSTRAINT, not knowledge -- and the types come
# from the LOG SCHEMA, not gold. That is the difference from notes 32: everything that worked there
# was an oracle. `oracle_conditions` is empty for every rung here, and must stay empty.
#
#   task 0  T0  constrained only                          reproduces N1 ~ 0.2162
#   task 1  T1  + type-aware value filters                the isolated effect
#   task 2  T2  + constant (0,1) root arity               the best available NON-ORACLE stack
#
# T2's arity constant is also non-oracle: (0,1) is the majority class of the SYNTHETIC TRAINING
# corpus (34.6%), so choosing it needs nothing from the testset. See notes 33.4.
#
# PRE-REGISTERED, and honest about the size: this touches ~10 of 169 cases, so expect ~+0.02 F1 --
# comparable to the constant-arity gain and near what a single greedy run resolves at n=135.
#   * TYPE-CONTRADICTION RATE AT T1 MUST BE 0. True by construction if the grammar binds; if it is
#     not, the constraint is not applied and nothing downstream is interpretable. Compliance gate.
#   * T1 > T0 -> the first working non-oracle component, however small. Report the size.
#   * T1 ~ T0 -> the type error was a symptom, not a cause; those cases were wrong anyway.
#   * T1 < T0 -> forcing a type the model did not want pushes it into worse filters elsewhere.
#   * T2 vs T1 -> whether the two non-oracle components add up or overlap.
#
#   sbatch --array=0-2 slurm/submit_typed_ablation.sh

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

# The config carries ONE prompt strategy; the rungs differ by decoding settings, not by strategy.
STRATEGY_IDX=0
MODEL_IDX=1      # llama31-8b-finetuned (verified by src/check_shard_map.py)
SCHEMA_IDX=0     # eval_config.schema_enforced is [true]; validation+repair stays ON in all rungs
                 # so D2 measures constrained decoding, not constrained decoding plus repair removal

case "$TASK" in
  0) RUNG_NAME="T0 baseline / constrained only, budget 4000"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --no-type-aware-filters --no-constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_tok4000" ;;
  1) RUNG_NAME="T1 / + TYPE-AWARE value filters (non-oracle)"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --type-aware-filters --no-constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_typed_tok4000" ;;
  2) RUNG_NAME="T2 / + constant (0,1) root arity -- best NON-ORACLE stack"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --type-aware-filters --constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_typed_arityconst_tok4000" ;;
  *) echo "FATAL: task ${TASK} is not a rung; this array is 0-2." >&2; exit 1 ;;
esac

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
# `--case-shard I/N` is accepted as an ARGUMENT as well as via $OCPQ_CASE_SHARD. The env var needs
# sbatch to propagate the submitting environment, and that silently did not happen three times
# running (2026-09-12, jobs 4041388 and 4042439: no CASE SHARD line, _case_shard null, slice 0
# again). Arguments after the script name are passed through by sbatch directly, so this form
# cannot be lost in propagation:
#     sbatch --array=0-2 slurm/submit_typed_ablation.sh --case-shard 2/11
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    --case-shard)
      shift
      [ "$#" -gt 0 ] || { echo "FATAL: --case-shard needs a value like 2/11" >&2; exit 1; }
      OCPQ_CASE_SHARD="$1" ;;
    --case-shard=*) OCPQ_CASE_SHARD="${1#*=}" ;;
    *) echo "Warning: ignoring unrecognised argument '$1'" >&2 ;;
  esac
  shift
done

RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  # Must include the rung suffix: all three rungs share model/strategy/schema indices, so the
  # bare prefix matches every rung and `ls -t | head -1` would resume the wrong one.
  SHARD_GLOB="$PROJECT_DIR/eval_runs/typed_ablation/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: --resume given but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_DIR}"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

# Targeted smoke: OCPQ_CASE_SHARD=I/N restricts this task to case slice I of N.
#
# The slice is CONTIGUOUS, and the testset's first 16 cases contain ZERO gold attribute
# value-filters -- so a plain 10-minute smoke (which just processes cases in order until the wall
# clock stops it) exercises NONE of what this ablation tests. The 2026-09-12 smoke did exactly
# that: T0 and T1 produced byte-identical predictions on all 14 shared cases, and the
# type-contradiction gate passed 0-of-0. A constraint that never binds is indistinguishable from
# the pre-registered "T1 ~ T0" result, so that gate has to be made non-vacuous BEFORE the full run.
#
#   OCPQ_CASE_SHARD=2/11 sbatch --array=0-2 slurm/submit_typed_ablation.sh
#
# covers cases 32-47: 6 renting_log_high cases (the log carrying the exact Age/isMarried errors of
# notes 34.3) and 7 whose gold uses an attribute value-filter.
#
# --case-shard appends `_c{I}of{N}` to the run directory, so a targeted smoke can never collide
# with, or be merged into, the full run.
# A malformed value must stop the job, not silently fall through to a full slice-0 run -- that is
# the failure this whole block exists to prevent.
CASE_ARGS=()
if [ -n "${OCPQ_CASE_SHARD:-}" ]; then
  case "$OCPQ_CASE_SHARD" in
    [0-9]*/[0-9]*) : ;;
    *) echo "FATAL: --case-shard/OCPQ_CASE_SHARD must look like I/N (got '$OCPQ_CASE_SHARD')." >&2
       exit 1 ;;
  esac
fi
if [ -n "${OCPQ_CASE_SHARD:-}" ]; then
  CASE_ARGS=(--case-shard "$OCPQ_CASE_SHARD")
  echo "CASE SHARD: ${OCPQ_CASE_SHARD} -- targeted SUBSET, not a full run."
fi

echo "Starting typed ablation task=${TASK} rung=${RUNG_NAME} at: $(date)"
echo "Decode args: ${DECODE_ARGS[*]}"
if [ "$TASK" != "0" ]; then
  echo "NOTE: NON-ORACLE. Every constraint here comes from the LOG SCHEMA or the training corpus,"
  echo "      never from gold, so these rungs ARE system performance. oracle_conditions must be []."
  echo "      Type-contradiction rate at T1/T2 must come out 0; if not, the grammar is not binding."
fi
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.typed_ablation.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${DECODE_ARGS[@]}"     "${CASE_ARGS[@]}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-typedabl-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Typed ablation rung ${TASK} exit code: $rc"
exit $rc
