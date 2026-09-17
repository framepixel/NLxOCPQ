#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_nonoracle
#SBATCH --output=logs/nlxocpq-nonoracle-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --account=thes2358
#SBATCH --export=ALL

# THE NON-ORACLE SYSTEM, MEASURED -- Thesis/notes/11-evaluation-readiness.md 41 (results), 33.4, 37.
#
# Every rung here is SYSTEM PERFORMANCE. Nothing reads gold: the structural check reads only the
# generated tree (notes 36.6), and the constant root arity (0 event vars, 1 object var) is the
# majority class of the SYNTHETIC TRAINING corpus (34.6%), so choosing it needs nothing from the
# testset (notes 33.4). `oracle_conditions` must be EMPTY in all four manifests.
#
# A 2x2 over the two components, so each effect and their interaction are separable:
#
#   task 0  B0  constrained only                        == N1 / S0 (F1 0.2354 on 2026-09-13/14)
#   task 1  B1  + structural check and repair           == S1 (0.2500), re-measured IN THIS JOB
#   task 2  B2  + constant (0,1) root arity             the estimate of notes 33.4/38.4, measured
#   task 3  B3  + both                                  the best non-oracle stack this project has
#
# WHY IT EXISTS: the two best non-oracle components have never run together. S1 measured the
# structural check; the constant arity was only ESTIMATED, at 0.2449-0.2713 against N1's 0.2354
# (notes 38.4, re-derived on the corrected corpus). B turns that estimate into a measurement and
# gives the thesis one number for "the system, with no oracle".
#
# PRE-REGISTERED
#   * COMPLIANCE, B1/B3: every schema-valid record carries `structural_report`, and every record with
#     a blocking defect has `repair_attempted`. Otherwise the switch did not reach the pipeline.
#   * COMPLIANCE, B2/B3: every parsed record's root box declares exactly 0 event and 1 object
#     variable. True by construction if the grammar binds; if it is not, nothing below is readable.
#   * THIS IS A MEASUREMENT, NOT A SIGNIFICANCE TEST. Expected effects are +0.01..+0.04 (constant
#     arity, notes 38.4) and +0.0146 (structural check, notes 37.2), against a resolution of about
#     +-0.03 at n=135 (notes 37.7). Report the number and its interval; a null is not "no effect".
#   * THE SPLIT THAT MATTERS for B2/B3: gold's root arity is (0,1) in 107 of 169 cases and something
#     else in 62. Forcing makes the correct query UNREACHABLE in those 62, so report "gain on 107,
#     loss on 62" beside the pooled figure -- notes 33.2 bracketed this; here it is measured.
#   * INTERACTION: (B3 - B2) - (B1 - B0) says whether the two components add or overlap.
#   * B1 vs notes 37's S1: same condition, same flags, different job. A difference between them is
#     greedy generation's run-to-run variation (notes 37.5), and bounds it.
#
# Smoke first -- slice 8/11 holds small logs and 9 of the silent-failure cases the check targets:
#   sbatch --array=0-3 slurm/submit_non_oracle_rungs.sh --case-shard 8/11
# Full run:
#   sbatch --array=0-3 slurm/submit_non_oracle_rungs.sh
#
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
  0) RUNG_NAME="B0 baseline / constrained only (== N1 / S0)"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --no-structural-validation --no-type-aware-filters --no-constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_tok4000" ;;
  1) RUNG_NAME="B1 / + structural check and repair (== S1, re-measured in this job)"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --structural-validation --no-type-aware-filters --no-constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_structval_tok4000" ;;
  2) RUNG_NAME="B2 / + constant (0,1) root arity"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --no-structural-validation --no-type-aware-filters --constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_arityconst_tok4000" ;;
  3) RUNG_NAME="B3 / + constant (0,1) root arity + structural check"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000 --structural-validation --no-type-aware-filters --constant-root-arity --no-force-root-arity)
     RUNG_SUFFIX="_constrained_structval_arityconst_tok4000" ;;
  *) echo "FATAL: task ${TASK} is not a rung; this array is 0-3." >&2; exit 1 ;;
esac

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
# `--case-shard I/N` is accepted as an ARGUMENT as well as via $OCPQ_CASE_SHARD. The env var needs
# sbatch to propagate the submitting environment, and that silently did not happen three times
# running (2026-09-12, jobs 4041388 and 4042439: no CASE SHARD line, _case_shard null, slice 0
# again). Arguments after the script name are passed through by sbatch directly, so this form
# cannot be lost in propagation:
#     sbatch --array=0-3 slurm/submit_non_oracle_rungs.sh --case-shard 8/11
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
  # Must include the rung suffix: all four rungs share model/strategy/schema indices, so the
  # bare prefix matches every rung and `ls -t | head -1` would resume the wrong one.
  SHARD_GLOB="$PROJECT_DIR/eval_runs/non_oracle/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: --resume given but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_DIR}"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

# Targeted smoke: OCPQ_CASE_SHARD=I/N (or --case-shard I/N) restricts this task to case slice I of N.
# The slice is CONTIGUOUS. 8/11 holds 9 of N2's 44 never-evaluated cases and only small logs, so a
# 10-minute smoke exercises the repair instead of passing a compliance check 0-of-0.
#   sbatch --array=0-3 slurm/submit_non_oracle_rungs.sh --case-shard 8/11
# --case-shard appends `_c{I}of{N}` to the run directory, so a smoke can never collide with, or be
# merged into, the full run.
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

echo "Starting non-oracle rungs task=${TASK} rung=${RUNG_NAME} at: $(date)"
echo "Decode args: ${DECODE_ARGS[*]}"
echo "NOTE: NON-ORACLE. Nothing here reads gold -- the structural check sees only the generated"
echo "      tree, and (0,1) is the training corpus's majority arity. Every rung IS system performance."
if [ "$TASK" = "1" ] || [ "$TASK" = "3" ]; then
  echo "      Every schema-valid record must carry structural_report; if not, the switch did not reach the pipeline."
fi
if [ "$TASK" = "2" ] || [ "$TASK" = "3" ]; then
  echo "      Every parsed record's root box must declare exactly 0 event and 1 object variable."
fi
echo "Python: $(which python)"; python --version
# Recorded because greedy generation is not bit-reproducible across jobs (notes 37.5): B1 repeats S1
# exactly, so any difference between them is job-to-job variation, and node and GPU are what make
# that attributable.
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.non_oracle.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${DECODE_ARGS[@]}"     "${CASE_ARGS[@]}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-nonoracle-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Non-oracle rung ${TASK} exit code: $rc"
exit $rc
