#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_examples
#SBATCH --output=logs/nlxocpq-examples-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-11
#SBATCH --account=thes2358
#SBATCH --export=ALL
# EVALUATION 2 WITH EXAMPLE LOOKUP -- THE SCORED RUNGS (notes 45), Qwen3.8 only, on the testset. The two
# NEW cells of the 2x2 (log access x examples); the other two are evaluation 2's A1* and A1'* (notes 44.9):
#
#   rung 0  A1*+S   the tool loop + search_examples, 8 turns / 12 tool calls
#   rung 1  A1'*+S  the same loop without log access (check_tree + search_examples)
#
# Run only if Qwen3.8 PASSED the example-lookup gate (notes 45). Each rung is split into SHARDS contiguous
# case slices: array task t runs rung t / SHARDS on slice t % SHARDS. 12 tasks = 2 rungs x 6 shards.
#     sbatch --array=0-1 slurm/submit_unleashed_examples_rungs.sh --smoke   # 2 rungs, slice 8/11
#     sbatch slurm/submit_unleashed_examples_rungs.sh                       # the full run
set -u
SHARDS=6
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
  exit 1
fi
cd "$PROJECT_DIR"
CONFIG="$PROJECT_DIR/configs/eval_config.unleashed_examples.json"

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

TASK="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=1   # Qwen3.8 only (notes 45)
SMOKE=0
OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --model-idx) shift; [ "$#" -gt 0 ] || { echo "FATAL: --model-idx needs a value (0-3)" >&2; exit 1; }
                 MODEL_IDX="$1" ;;
    --model-idx=*) MODEL_IDX="${1#*=}" ;;
    --smoke) SMOKE=1 ;;
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    *) echo "Warning: ignoring unrecognised argument '$1'" >&2 ;;
  esac
  shift
done
case "$MODEL_IDX" in
  1) ;;
  *) echo "FATAL: pass --model-idx 0-3 (a model of eval_config.unleashed.json); got '${MODEL_IDX}'." >&2
     exit 1 ;;
esac

if [ "$SMOKE" = "1" ]; then
  # The smoke: task t is rung t on the small-log slice 8/11, as evaluation 1's rung smoke.
  RUNG="$TASK"
  CASE_SHARD="8/11"
else
  RUNG=$((TASK / SHARDS))
  CASE_SHARD="$((TASK % SHARDS))/${SHARDS}"
fi
case "$RUNG" in
  0) RUNG_NAME="A1*+S / tool loop + search_examples, reasoning first"
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think2048keep_turns8_tools12_turntok2000_pFBEXLS" ;;
  1) RUNG_NAME="A1'*+S / same loop without log access, + search_examples, reasoning first"
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think2048keep_turns8_turntok2000_pFBEXLS" ;;
  *) echo "FATAL: task ${TASK} maps to rung ${RUNG}; this array is 0-11 (0-1 with --smoke)." >&2; exit 1 ;;
esac
STRATEGY_IDX="$RUNG"   # the config's strategies are in rung order
SCHEMA_IDX=0

if ! RUNTIME="$(python3 "$PROJECT_DIR/src/model_runtime.py" "$CONFIG" "$MODEL_IDX")"; then
  echo "FATAL: no runtime for model index ${MODEL_IDX} in ${CONFIG}" >&2
  exit 1
fi
OCPQ_RUNTIME_ENV="${RUNTIME%%|*}"
export HF_HOME="${RUNTIME#*|}"
export HF_HUB_CACHE="$HF_HOME/hub"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$OCPQ_RUNTIME_ENV"
echo "Model index: ${MODEL_IDX}  rung ${RUNG} (${RUNG_NAME})  case shard ${CASE_SHARD}"
echo "Runtime: conda env ${OCPQ_RUNTIME_ENV}, HF cache ${HF_HUB_CACHE}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"
python "$PROJECT_DIR/src/check_eval_deps.py" --verbose || exit 1
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi
export OCPQ_CLI_SCRATCH_DIR="${OCPQ_CLI_SCRATCH_DIR:-${TMPDIR:-/tmp}/ocpq-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID}"
mkdir -p "$OCPQ_CLI_SCRATCH_DIR"
trap 'rm -rf "$OCPQ_CLI_SCRATCH_DIR"' EXIT
export OCPQ_SCHEMA_TOKEN_BUDGET="${OCPQ_SCHEMA_TOKEN_BUDGET:-17000}"

SHARD_TAG="_c${CASE_SHARD%%/*}of${CASE_SHARD##*/}"
RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  SHARD_GLOB="$PROJECT_DIR/eval_runs/unleashed_examples/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${SHARD_TAG}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  [ -n "$RESUME_DIR" ] || { echo "FATAL: --resume but nothing matches ${SHARD_GLOB}" >&2; exit 1; }
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting example lookup, task ${TASK}, at: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
if [ "$RUNG" = "0" ]; then
  python "$PROJECT_DIR/src/check_probe.py" --out "logs/nlxocpq-examples-probe-check-${SLURM_JOB_ID}-${TASK}.json" || exit 1
fi
set -e
python -u "$PROJECT_DIR/src/eval_testset.py" --config "$CONFIG" --model-idx "${MODEL_IDX}" \
  --strategy-idx "${STRATEGY_IDX}" --schema-idx "${SCHEMA_IDX}" --case-shard "${CASE_SHARD}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "logs/nlxocpq-examples-${SLURM_JOB_ID}-${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Example lookup, task ${TASK}, exit code: $rc"
exit $rc
