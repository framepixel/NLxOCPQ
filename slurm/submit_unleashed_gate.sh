#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_unleashed_gate
#SBATCH --output=logs/nlxocpq-unleashed-gate-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --account=thes2358
#SBATCH --export=ALL
# EVALUATION 2 ("UNLEASHED") -- THE GATE (notes 44). One task per model of
# configs/eval_config.unleashed_gate.json: A1* (the tool loop, every generation reasoning first in the
# model's own channel) on data/agent_devset.json -- 38 entries, NEVER scored -- one draw per case.
#
#   task 0  ministral3-14b-reasoning   task 2  muse-glimmer-30b-reasoning
#   task 1  qwen3.8-27b-thinking       task 3  gemma4-31b-thinking
#
# READ COMPLIANCE, REASONING AND TIME ONLY, against the pass rule of notes 44. NO F1 FROM THIS JOB IS
# QUOTABLE, and none enters a decision. The reasoning budget (and a model's reasoning effort) may be
# changed on the evidence of this gate only, then frozen.
#
# Prerequisites on a login node: bash scripts/prefetch_pinned_models.sh (all four are pinned), and
# python src/long_prompt_smoke.py for Gemma (memory, notes 43.19).
#     sbatch slurm/submit_unleashed_gate.sh                       # all four
#     sbatch --array=0 slurm/submit_unleashed_gate.sh --case-shard 0/19   # a two-case load smoke
set -u
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
CONFIG="$PROJECT_DIR/configs/eval_config.unleashed_gate.json"

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

TASK="${SLURM_ARRAY_TASK_ID}"
MODEL_IDX="$TASK"
OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
# Arguments, not exported variables: sbatch passes them through (the --case-shard lesson).
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    --case-shard) shift; [ "$#" -gt 0 ] || { echo "FATAL: --case-shard needs I/N" >&2; exit 1; }
                  OCPQ_CASE_SHARD="$1" ;;
    --case-shard=*) OCPQ_CASE_SHARD="${1#*=}" ;;
    *) echo "Warning: ignoring unrecognised argument '$1'" >&2 ;;
  esac
  shift
done

# The model's runtime comes from its config entry (src/model_runtime.py, standard library only).
if ! RUNTIME="$(python3 "$PROJECT_DIR/src/model_runtime.py" "$CONFIG" "$MODEL_IDX")"; then
  echo "FATAL: no runtime for model index ${MODEL_IDX} in ${CONFIG}" >&2
  exit 1
fi
OCPQ_RUNTIME_ENV="${RUNTIME%%|*}"
export HF_HOME="${RUNTIME#*|}"
export HF_HUB_CACHE="$HF_HOME/hub"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$OCPQ_RUNTIME_ENV"
echo "Model index: ${MODEL_IDX}"
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

STRATEGY_IDX=0   # the gate config carries ONE strategy: agentic-log-tools (A1*)
SCHEMA_IDX=0
RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_think1024_turns8_tools12_turntok2000_pFBEXL"
CASE_ARGS=()
if [ -n "${OCPQ_CASE_SHARD:-}" ]; then
  case "$OCPQ_CASE_SHARD" in
    [0-9]*/[0-9]*) CASE_ARGS=(--case-shard "$OCPQ_CASE_SHARD")
                   echo "CASE SHARD: ${OCPQ_CASE_SHARD} -- a subset, not the gate." ;;
    *) echo "FATAL: --case-shard must look like I/N (got '$OCPQ_CASE_SHARD')." >&2; exit 1 ;;
  esac
fi
RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  SHARD_GLOB="$PROJECT_DIR/eval_runs/unleashed_gate/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  [ -n "$RESUME_DIR" ] || { echo "FATAL: --resume but nothing matches ${SHARD_GLOB}" >&2; exit 1; }
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting the evaluation-2 gate, model ${MODEL_IDX}, at: $(date)"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
python "$PROJECT_DIR/src/check_probe.py" --testset "$PROJECT_DIR/data/agent_devset.json" \
  --out "logs/nlxocpq-unleashed-gate-probe-check-${SLURM_JOB_ID}-${TASK}.json" || exit 1
set -e
python -u "$PROJECT_DIR/src/eval_testset.py" --config "$CONFIG" --model-idx "${MODEL_IDX}" \
  --strategy-idx "${STRATEGY_IDX}" --schema-idx "${SCHEMA_IDX}" "${CASE_ARGS[@]}" "${RESUME_ARGS[@]}" \
  2>&1 | tee "logs/nlxocpq-unleashed-gate-${SLURM_JOB_ID}-${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Evaluation-2 gate, model ${MODEL_IDX}, exit code: $rc"
exit $rc
