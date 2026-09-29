#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_unleashed
#SBATCH --output=logs/nlxocpq-unleashed-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-17
#SBATCH --account=thes2358
#SBATCH --export=ALL
# EVALUATION 2 ("UNLEASHED") -- THE SCORED RUNGS (notes 44), for ONE model of
# configs/eval_config.unleashed.json, on the testset. Every generation reasons first in the model's own
# channel (reasoning_token_budget), then acts under the grammar; vendor sampling, fixed seeds, 2 draws.
#
#   rung 0  A0*   one-shot (B0's prompt), reasoning first
#   rung 1  A1*   the tool loop, 8 turns / 12 tool calls
#   rung 2  A1'*  the same loop without log access (check_tree only)       headline: A1* - A1'*
#
# One rung takes ~10-25 GPU-h per model, over the 20 h wall, so each rung is split into SHARDS
# contiguous case slices: array task t runs rung t / SHARDS on case slice t % SHARDS. 18 tasks = 3 rungs x
# 6 shards. The shards of a rung are merged afterwards (eval_testset.py --merge-shards).
#
# Only a model that PASSED the evaluation-2 gate is run, in the pre-registered order of notes 44
# (0 Ministral-Reasoning, 1 Qwen3.8, 2 Glimmer, 3 Gemma 4):
#     sbatch --array=0-2 slurm/submit_unleashed_rungs.sh --model-idx 0 --smoke   # 3 rungs, slice 8/11
#     sbatch slurm/submit_unleashed_rungs.sh --model-idx 0                        # the full run
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
CONFIG="$PROJECT_DIR/configs/eval_config.unleashed.json"

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

TASK="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX=""
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
  0|1|2|3) ;;
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
  0) RUNG_NAME="A0* / one-shot, reasoning first"
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think1024" ;;
  1) RUNG_NAME="A1* / tool loop, reasoning first"
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think1024_turns8_tools12_turntok2000_pFBEXL" ;;
  2) RUNG_NAME="A1'* / same loop without log access, reasoning first"
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think1024_turns8_turntok2000_pFBEXL" ;;
  *) echo "FATAL: task ${TASK} maps to rung ${RUNG}; this array is 0-17 (0-2 with --smoke)." >&2; exit 1 ;;
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
  SHARD_GLOB="$PROJECT_DIR/eval_runs/unleashed/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${SHARD_TAG}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  [ -n "$RESUME_DIR" ] || { echo "FATAL: --resume but nothing matches ${SHARD_GLOB}" >&2; exit 1; }
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting evaluation 2, task ${TASK}, at: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
if [ "$RUNG" = "1" ]; then
  python "$PROJECT_DIR/src/check_probe.py" --out "logs/nlxocpq-unleashed-probe-check-${SLURM_JOB_ID}-${TASK}.json" || exit 1
fi
set -e
python -u "$PROJECT_DIR/src/eval_testset.py" --config "$CONFIG" --model-idx "${MODEL_IDX}" \
  --strategy-idx "${STRATEGY_IDX}" --schema-idx "${SCHEMA_IDX}" --case-shard "${CASE_SHARD}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "logs/nlxocpq-unleashed-${SLURM_JOB_ID}-${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Evaluation 2, task ${TASK}, exit code: $rc"
exit $rc
