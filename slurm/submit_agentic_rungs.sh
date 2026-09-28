#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_agentic
#SBATCH --output=logs/nlxocpq-agentic-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --array=0-4
#SBATCH --account=thes2358
#SBATCH --export=ALL

# THE AGENTIC ARM -- Thesis/notes/11-evaluation-readiness.md 43. Plan: TODO, "AGENTIC PLAN STAGE 4".
#
# Every rung is SYSTEM PERFORMANCE. The probe reads the log and the model's own candidate trees and
# cannot reach gold (tests/test_ocpq_probe.py, tests/test_agentic_no_gold.py); no oracle- strategy,
# force_root_arity false. `oracle_conditions` must be EMPTY in all five manifests.
#
#   task 0  A0        one-shot, no tools, no loop       the SAME base model, B0's prompt (notes 43.6)
#   task 1  A1        tool loop, 8 turns / 12 tool calls
#   task 2  A1-prime  the SAME loop, 8 turns, NO tools  compute-matched control (self-refine)
#   task 3  A2        tool loop, 16 turns / 24 tool calls  sized from the Stage 5 dev run
#   task 4  A1+th     A1 with a `thought` field         the reasoning condition (TODO, `thought`)
#
# PRE-REGISTERED
#   * COMPLIANCE FIRST, before any F1, from each summary's compliance block: malformed-action rate,
#     no-answer rate, budget exhaustion, final_source mix, probe restarts / panics / refusals,
#     probe-vs-CLI agreement (anything below 1.0 not explained by CLI truncation is a harness
#     defect), and for task 4 the thought uptake. A rung whose model cannot hold the protocol is
#     not measuring tools, and says so.
#   * ATTRIBUTION: A1 - A1-prime is the tool effect; A1 - A0 mixes tools with extra compute;
#     A2 - A1 is budget alone; A1+th - A1 is reasoning at a fixed budget, reported with mean
#     completion chars beside it because a thought costs tokens every turn.
#   * THE EFFECT THAT COUNTS: paired bootstrap over per-case F1 differences on the same 135
#     discriminating cases (notes 37.7 gives +-0.03 independent resolution; pairing is tighter).
#     Skill is reported beside F1: an agent that drifts to large plausible sets lands near the floor.
#   * MECHANISM: per case, whether each revision changed the POPULATION (the per-node counts in
#     agent.turn_summary) or only the syntax. Mostly-syntactic revisions mean the arm has not
#     touched the compositional failure of notes 40, whatever the F1 says.
#   * ONE CAP FOR EVERY ARM (CLAUDE.md 6): case_timeout_sec 600 in all five rungs.
#
# A2's BUDGET IS SIZED (notes 43.13): the Stage 5 dev run (Ministral 3 14B base, pFBEXL, 38 cases x
# thought off/on) took a median 6.4-7.0 s per turn (p90 ~9 s), longest case 95 s at 8 turns, and
# cases used 7.1-7.2 of 8 turns -- the budget binds, so A2 is informative. 16 turns extrapolate to a
# worst case of ~190 s, inside the 600 s cap every arm shares (CLAUDE.md 6). The decision is a
# committed constant below (A2_SIZED), not an exported variable: an environment variable is not
# reliably propagated by sbatch (the --case-shard lesson, 2026-09-12).
#
# BEFORE THE FIRST JOB, on a login node: build the probe server once (the first release build
# takes minutes and must not happen inside a GPU job):
#     cargo build --release --manifest-path ../OCPQ/backend/web-server/Cargo.toml -p ocpq-web-server
#
# Smoke first (slice 8/11: small logs), then src/smoke_check.py, then the full run -- model index 2
# is Ministral 3 14B base, the model gate's choice (notes 43.11):
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 2 --case-shard 8/11
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 2
#
# THE PROTOCOL IS pFBEXLR (notes 43.17): the frozen Stage 5 protocol plus R, the repeat guard, added
# after Muse Glimmer's gate and run on BOTH models so the two arms share one protocol. The prompts
# are the frozen ones (R adds no text). Runs write to eval_runs/agentic_r; the completed Ministral
# pFBEXL run of notes 43.15 stays in eval_runs/agentic and is not replaced. Both models, in order:
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 2 --case-shard 8/11   # Ministral
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 2
#
# THE SECOND ARM (notes 43.16): model index 8 is Muse Glimmer-30B, the same five rungs and the same
# protocol. It runs in its own conda env with its weights on $HPCWORK (slurm/model_runtime.sh).
# Prerequisites, on a login node: OCPQ_ENV_VARIANT=glimmer bash scripts/make_eval_env.sh, then
# bash scripts/prefetch_glimmer.sh && VERIFY=1 bash scripts/prefetch_glimmer.sh. Its model gate
# (slurm/submit_agentic_gate.sh, tasks 8,9) comes first, then:
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 8 --case-shard 8/11
#   sbatch --array=0-4 slurm/submit_agentic_rungs.sh --model-idx 8
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
# model index is known (below the guard; slurm/model_runtime.sh, notes 43.16).

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

STRATEGY_IDX=0   # set per rung below: 0 one-shot, 1 agentic-log-tools, 2 agentic-self-refine
SCHEMA_IDX=0     # schema_enforced is [true]
# The arm runs on a BASE instruct model (TODO option D; notes 43.6): the adapters were trained to
# emit exactly one tree, and the 2026-09-23 smoke on fine-tuned Llama confirmed it (0 tool calls in
# 10 cases). Which base model is the gate's decision, so -- like A2's budget -- there is no default:
#     sbatch --array=0-2,4 slurm/submit_agentic_rungs.sh --model-idx 0
# ($OCPQ_AGENT_MODEL_IDX also works, but see --case-shard below for why the argument is safer.)
# Indices are the expansion of eval_config.agentic.json: even = base, odd = its fine-tuned adapter.
# Checked after the arguments are parsed.
MODEL_IDX="${OCPQ_AGENT_MODEL_IDX:-}"
# The config sets constrained decoding, the ORDERED and BOUNDED grammar (notes 43.7-43.8) and a
# 4000-token budget for every rung, so every shard directory carries
# `_constrained_ordered_bounded_tok4000`; the loop rungs then add their agent suffix, including
# the per-turn cap `_turntok2000` (config `agent_turn_token_budget`). Exactly
# eval_testset._decoding_suffix followed by eval_testset._agent_suffix;
# tests/test_agentic_rungs.py pins every RUNG_SUFFIX against both.
A2_TURNS=16
A2_TOOL_CALLS=24
A2_SIZED=1   # notes 43.13: sized from the Stage 5 dev run; see the header

case "$TASK" in
  0) RUNG_NAME="A0 / one-shot, no tools (B0's prompt, this base model)"
     STRATEGY_IDX=0
     AGENT_ARGS=()
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000" ;;
  1) RUNG_NAME="A1 / tool loop, 8 turns, 12 tool calls"
     STRATEGY_IDX=1
     AGENT_ARGS=(--agent-max-turns 8 --agent-max-tool-calls 12 --no-agent-thought)
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns8_tools12_turntok2000_pFBEXLR" ;;
  2) RUNG_NAME="A1-prime / same loop, 8 turns, NO tools (compute-matched control)"
     STRATEGY_IDX=2
     AGENT_ARGS=(--agent-max-turns 8 --agent-max-tool-calls 12 --no-agent-thought)
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns8_turntok2000_pFBEXLR" ;;
  3) RUNG_NAME="A2 / tool loop, ${A2_TURNS} turns, ${A2_TOOL_CALLS} tool calls"
     if [ "${A2_SIZED}" != "1" ]; then
       echo "FATAL: A2's budget is not marked as sized (A2_SIZED). Size it from a dev run so A2" >&2
       echo "       fits case_timeout_sec, record the evidence in the notes, and set A2_SIZED=1." >&2
       exit 1
     fi
     STRATEGY_IDX=1
     AGENT_ARGS=(--agent-max-turns "${A2_TURNS}" --agent-max-tool-calls "${A2_TOOL_CALLS}" --no-agent-thought)
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns${A2_TURNS}_tools${A2_TOOL_CALLS}_turntok2000_pFBEXLR" ;;
  4) RUNG_NAME="A1+thought / A1 with a thought field"
     STRATEGY_IDX=1
     AGENT_ARGS=(--agent-max-turns 8 --agent-max-tool-calls 12 --agent-thought)
     RUNG_SUFFIX="_constrained_ordered_bounded_tok4000_turns8_tools12_turntok2000_pFBEXLR_thought" ;;
  *) echo "FATAL: task ${TASK} is not a rung; this array is 0-4." >&2; exit 1 ;;
esac

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
# `--case-shard I/N` is accepted as an ARGUMENT as well as via $OCPQ_CASE_SHARD. The env var needs
# sbatch to propagate the submitting environment, and that silently did not happen three times
# running (2026-09-12, jobs 4041388 and 4042439: no CASE SHARD line, _case_shard null, slice 0
# again). Arguments after the script name are passed through by sbatch directly, so this form
# cannot be lost in propagation:
#     sbatch --array=0-3 slurm/submit_agentic_rungs.sh --model-idx 0 --case-shard 8/11
while [ "$#" -gt 0 ]; do
  case "$1" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    --case-shard)
      shift
      [ "$#" -gt 0 ] || { echo "FATAL: --case-shard needs a value like 2/11" >&2; exit 1; }
      OCPQ_CASE_SHARD="$1" ;;
    --case-shard=*) OCPQ_CASE_SHARD="${1#*=}" ;;
    --model-idx)
      shift
      [ "$#" -gt 0 ] || { echo "FATAL: --model-idx needs a value (0, 2 or 8)" >&2; exit 1; }
      MODEL_IDX="$1" ;;
    --model-idx=*) MODEL_IDX="${1#*=}" ;;
    *) echo "Warning: ignoring unrecognised argument '$1'" >&2 ;;
  esac
  shift
done

case "$MODEL_IDX" in
  0|2|8) ;;
  1|3) echo "FATAL: model index ${MODEL_IDX} is a fine-tuned adapter. The agentic arm runs on a" >&2
       echo "       base model (0 llama31-8b-base, 2 ministral3-14b-base, 8 muse-glimmer-30b-base)." >&2
       exit 1 ;;
  *) echo "FATAL: pass --model-idx with a base model of the arm" >&2
     echo "       (0 llama31-8b-base, 2 ministral3-14b-base, 8 muse-glimmer-30b-base); '${MODEL_IDX}' is not one." >&2
     exit 1 ;;
esac
echo "Model index: ${MODEL_IDX}"

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

RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  # Must include the rung suffix: A1, A2 and A1+thought share model/strategy/schema indices, so the
  # bare prefix matches every rung and `ls -t | head -1` would resume the wrong one.
  SHARD_GLOB="$PROJECT_DIR/eval_runs/agentic_r/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
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
#   sbatch --array=0-3 slurm/submit_agentic_rungs.sh --model-idx 0 --case-shard 8/11
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

echo "Starting agentic rungs task=${TASK} rung=${RUNG_NAME} at: $(date)"
echo "Agent args: ${AGENT_ARGS[*]:-(none)}"
echo "NOTE: SYSTEM PERFORMANCE. The probe reads the log and the model's own trees, never gold."
echo "      Read the summary's compliance block BEFORE any F1."
echo "Python: $(which python)"; python --version
# Greedy generation is not bit-reproducible across jobs (notes 37.5), so node and GPU are recorded
# to make any cross-job difference attributable.
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"

# The tool rungs need the probe server: check it serves every log (the three XML ones included)
# with identity assertions, and record its memory beside the 64 GB limit, before GPU time is spent.
if [ "$STRATEGY_IDX" = "1" ]; then
  python "$PROJECT_DIR/src/check_probe.py" --out "logs/nlxocpq-agentic-probe-check-${SLURM_JOB_ID}-${TASK}.json" || exit 1
fi

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.agentic.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${AGENT_ARGS[@]}"     "${CASE_ARGS[@]}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-agentic-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Agentic rung ${TASK} exit code: $rc"
exit $rc
