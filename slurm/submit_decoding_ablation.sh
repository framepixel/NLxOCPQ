#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_decodeabl
#SBATCH --output=logs/nlxocpq-decodeabl-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# One rung per task, 169 cases each, greedy. D2 adds a per-token grammar mask; the smoke measures
# that overhead and is allowed to stop the experiment if it is worse than ~3x.
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --account=thes2358
#SBATCH --export=ALL

# THE DECODING ABLATION.
#
# `no_result` is 42.7% of fine-tuned discriminating samples and is half of the ONLY pair of
# failure buckets that clears the trivial floor (`no_result` + `arity_mismatch` = 0.8307; notes
# 27-28). This measures how much of it is recoverable, and by which of two distinct mechanisms.
#
# WHAT THIS IS NOT: `schema_enforced` does not constrain generation and never did. On the
# local-gpu backend `response_format` never reaches generate() -- it only appends the sentence
# "Return only valid JSON", identically whether the flag is on or off. What it actually switches
# is `effective_schema_validation`, i.e. whether output is PARSED, VALIDATED and REPAIRED after
# the fact. That repair pass runs on 2043 cases and fails on 1613 (79%). See notes 29.
#
# Constrained decoding is the other thing: at every step the OCPQ grammar says which tokens can
# still lead to a valid BindingBoxTree and the rest are masked to -inf. Invalid output stops being
# something to detect and becomes something that cannot be produced.
#
#   task 0  D0  budget 2000, free         the published baseline, re-run in-job
#   task 1  D1  budget 4000, free         isolates TRUNCATION
#   task 2  D2  budget 4000, constrained  isolates SCHEMA VIOLATION
#
# D1 sits between them deliberately. Failed generations run median 794 completion tokens against
# 357 for successful ones and pile up at the 2000 cap, so truncation and schema violation are two
# separate mechanisms; without D1 they are confounded. D2 inherits D1's budget because constrained
# output may be LONGER -- the fix for one mechanism must not reintroduce the other.
#
# PRE-REGISTERED, against the trivial floor 0.7250 (discriminating), not against D0. The oracle
# caps this experiment at 0.6099 even if it works perfectly, so CLEARING THE FLOOR IS NOT ON THE
# TABLE. The question is how much of the 42.7% is recoverable and by which mechanism.
#   * schema_valid_rate at D2 not ~1.0 -> the constraint is not binding. Nothing downstream is
#     interpretable; fix that before reading anything else. This is the compliance gate.
#   * schema_valid ~1.0 and F1 rises -> syntax was a real cap. Report the size.
#   * schema_valid ~1.0 and F1 flat  -> the queries that failed to parse were also semantically
#     wrong. The notes-25.3 syntax/semantics split again, and the most likely outcome.
#   * F1 FALLS at D2 -> forcing well-formed output pushes the model into valid-but-wrong
#     structures. A real cost of constrained decoding, and nobody has measured it here.
#
#   sbatch --array=0-2 slurm/submit_decoding_ablation.sh
#   sbatch --array=2 slurm/submit_decoding_ablation.sh --resume     # one rung, resumed
#
# Then:
#   python src/eval_testset.py --merge-shards eval_runs/decoding_ablation/shard_* #       --merge-output eval_runs/decoding_ablation/merged

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
  0) RUNG_NAME="D0 baseline / budget 2000, free"
     DECODE_ARGS=(--no-constrained-decoding)
     RUNG_SUFFIX="" ;;
  1) RUNG_NAME="D1 / budget 4000, free -- isolates TRUNCATION"
     DECODE_ARGS=(--no-constrained-decoding --token-budget 4000)
     RUNG_SUFFIX="_tok4000" ;;
  2) RUNG_NAME="D2 / budget 4000, CONSTRAINED -- isolates SCHEMA VIOLATION"
     DECODE_ARGS=(--constrained-decoding --token-budget 4000)
     RUNG_SUFFIX="_constrained_tok4000" ;;
  *) echo "FATAL: task ${TASK} is not a rung; this array is 0-2." >&2; exit 1 ;;
esac

OCPQ_RESUME_REQUESTED="${OCPQ_RESUME:-0}"
for arg in "$@"; do
  case "$arg" in
    --resume) OCPQ_RESUME_REQUESTED=1 ;;
    *) echo "Warning: ignoring unrecognised argument '$arg'" >&2 ;;
  esac
done

RESUME_ARGS=()
if [ "$OCPQ_RESUME_REQUESTED" = "1" ]; then
  # Must include the rung suffix: all three rungs share model/strategy/schema indices, so the
  # bare prefix matches every rung and `ls -t | head -1` would resume the wrong one.
  SHARD_GLOB="$PROJECT_DIR/eval_runs/decoding_ablation/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}${RUNG_SUFFIX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: --resume given but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_DIR}"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting decoding ablation task=${TASK} rung=${RUNG_NAME} at: $(date)"
echo "Decode args: ${DECODE_ARGS[*]}"
if [ "$TASK" = "2" ]; then
  echo "NOTE: constrained decoding is ON. schema_valid_rate must come out ~1.0."
  echo "      If it does not, the grammar is not binding and nothing downstream is readable."
fi
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.decoding_ablation.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${DECODE_ARGS[@]}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-decodeabl-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Decoding ablation rung ${TASK} exit code: $rc"
exit $rc
