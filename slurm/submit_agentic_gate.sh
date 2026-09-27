#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_agentgate
#SBATCH --output=logs/nlxocpq-agentgate-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --array=0,1,4,5
#SBATCH --account=thes2358
#SBATCH --export=ALL

# MODEL GATE FOR THE AGENTIC ARM -- TODO "AGENTIC PLAN STAGE 4", item 6; notes 43.
#
# Runs the tool rung A1 (8 turns, 12 tool calls) on data/agent_devset.json -- 38 entries, NEVER
# scored in the thesis -- for the two BASE models, with and without the `thought` field:
#
#   task  model                       thought
#   0/1   llama31-8b-base    (idx 0)  off / on
#   4/5   ministral3-14b-base (idx 2) off / on
#   8/9   muse-glimmer-30b-base (idx 8) off / on   the second arm's gate (notes 43.16), run on its
#                                                  own: sbatch --array=8,9 slurm/submit_agentic_gate.sh
#                                                  It reads compliance and s/turn, and fixes Glimmer's
#                                                  CHAT settings only -- never the frozen protocol.
#
# The agentic arm runs on base instruct models (TODO option D; notes 43.6). The adapters were
# trained to emit exactly ONE query tree, and the 2026-09-23 smoke confirmed it: fine-tuned Llama
# (tasks 2/3) made 0 tool calls in 10 cases and declared a tree on turn 1 every time. Tasks 2/3 and
# 6/7 (the fine-tuned adapters, idx 1 and 3) stay in the table so a fine-tuned check can still be
# run explicitly with --array, but they are not part of the gate.
#
# READ COMPLIANCE AND COST ONLY, from each summary's compliance block and records:
#   malformed-action rate, no-answer rate, budget exhaustion, final_source mix, mean turns and tool
#   calls, probe restarts, thought uptake (odd tasks), and WALL CLOCK PER CASE -- the number A2's
#   budget is sized from (it must fit case_timeout_sec 600, identical on every arm; CLAUDE.md 6).
# NO F1 FROM THIS JOB IS QUOTABLE. Dev entries have no floors, so skill is undefined, and choosing a
# model or a prompt by dev F1 and then reporting it would be tuning on what we report.
# A model that cannot hold the protocol is not measurable in the agentic arm; that is a result.
#
# Prerequisite, on a login node, once: build the probe server (see submit_agentic_rungs.sh).
#   sbatch slurm/submit_agentic_gate.sh
#   sbatch --array=0,1 slurm/submit_agentic_gate.sh --case-shard 0/8   # smoke: Llama base
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

# The conda environment and the HF cache are chosen per MODEL, so they are set up after the
# task has named the model (below; slurm/model_runtime.sh, notes 43.16).


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

STRATEGY_IDX=0   # the gate config carries ONE strategy: agentic-log-tools
SCHEMA_IDX=0
case "$TASK" in
  0|1) MODEL_IDX=0; MODEL_NAME="llama31-8b-base" ;;
  2|3) MODEL_IDX=1; MODEL_NAME="llama31-8b-finetuned" ;;
  4|5) MODEL_IDX=2; MODEL_NAME="ministral3-14b-base" ;;
  6|7) MODEL_IDX=3; MODEL_NAME="ministral3-14b-finetuned" ;;
  # The second arm's gate (notes 43.16): sbatch --array=8,9 slurm/submit_agentic_gate.sh
  8|9) MODEL_IDX=8; MODEL_NAME="muse-glimmer-30b-base" ;;
  *) echo "FATAL: task ${TASK} is not a gate task (0-9; the gate is 0,1,4,5, and 8,9 for Glimmer)." >&2; exit 1 ;;
esac

# Same stack as the fine-tuning runs, so the adapters are loaded by what wrote them.
# A conda env, not the old $HOME/envs/nlxocpq-venv virtualenv.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
# The eval environment is a clone of the fine-tuning env plus scikit-learn and openai,
# which the eval path needs and training does not. Built by scripts/make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
# Index 8 (Muse Glimmer) runs in its own clone on a newer transformers; every other index runs
# exactly where it did before (slurm/model_runtime.sh).
source "$SCRIPT_DIR/model_runtime.sh"
ocpq_model_runtime "$MODEL_IDX" || exit 1
conda activate "$OCPQ_RUNTIME_ENV"
echo "Runtime: conda env ${OCPQ_RUNTIME_ENV}, HF cache ${OCPQ_RUNTIME_HF_HUB_CACHE}"
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
export HF_HOME="$OCPQ_RUNTIME_HF_HOME"
export HF_HUB_CACHE="$OCPQ_RUNTIME_HF_HUB_CACHE"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi
if [ $((TASK % 2)) -eq 0 ]; then
  THOUGHT_ARGS=(--no-agent-thought)
  RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns8_tools12_turntok2000_pFBEXL"
else
  THOUGHT_ARGS=(--agent-thought)
  RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns8_tools12_turntok2000_pFBEXL_thought"
fi
RUNG_NAME="${MODEL_NAME} / A1 ${THOUGHT_ARGS[*]}"
AGENT_ARGS=(--agent-max-turns 8 --agent-max-tool-calls 12 "${THOUGHT_ARGS[@]}")

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
# `--case-shard I/N` is accepted as an ARGUMENT as well as via $OCPQ_CASE_SHARD. The env var needs
# sbatch to propagate the submitting environment, and that silently did not happen three times
# running (2026-09-12, jobs 4041388 and 4042439: no CASE SHARD line, _case_shard null, slice 0
# again). Arguments after the script name are passed through by sbatch directly, so this form
# cannot be lost in propagation:
#     sbatch --array=0-3 slurm/submit_agentic_gate.sh --case-shard 8/11
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
  # Must include the rung suffix: the thought-off and thought-on tasks share model/strategy/schema indices, so the
  # bare prefix matches every rung and `ls -t | head -1` would resume the wrong one.
  SHARD_GLOB="$PROJECT_DIR/eval_runs/agentic_gate/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
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
#   sbatch --array=0-3 slurm/submit_agentic_gate.sh --case-shard 8/11
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

echo "Starting agentic GATE task=${TASK} ${RUNG_NAME} at: $(date)"
echo "NOTE: dev set only. Read compliance and cost; no F1 from this job is quotable."
echo "Python: $(which python)"; python --version
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"

python "$PROJECT_DIR/src/check_probe.py" --testset "$PROJECT_DIR/data/agent_devset.json" --out "logs/nlxocpq-agentgate-probe-check-${SLURM_JOB_ID}-${TASK}.json" || exit 1

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.agentic_gate.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${AGENT_ARGS[@]}"     "${CASE_ARGS[@]}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-agentgate-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Agentic gate task ${TASK} exit code: $rc"
exit $rc
