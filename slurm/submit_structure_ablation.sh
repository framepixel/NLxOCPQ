#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_structabl
#SBATCH --output=logs/nlxocpq-structabl-%A_%a.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# One condition per task, 169 cases each, greedy. Measured 12.7-96.6 s/run on the 2026-08-29
# smoke, so ~1-5h per task; 14h is the same limit the other eval jobs use and is ample here.
#SBATCH --time=14:00:00
#SBATCH --array=0-2
#SBATCH --account=thes2358
#SBATCH --export=ALL

# THE STRUCTURE ABLATION.
#
# Measures the CEILING of the whole composition-fixing programme -- the training rebalance and
# the decomposition architecture alike -- before either is built. Same logic as the oracle
# ceiling in notes 22: bound an intervention before paying for it.
#
# Each task runs one rung over all 169 cases (configs/eval_config.structure_ablation.json):
#   task 0  schema-allowed-entities-only            S0, today's prompt, run in-job
#   task 1  oracle-skeleton-given                   S1, + gold node count, edges, var arity
#   task 2  oracle-skeleton-and-constraints-given   S2, + gold constraint TYPES
#
# S1 and S2 ARE ORACLE CONDITIONS. They are shown part of the GOLD query, so their scores are
# ceilings and can never be reported as system performance. The hint withholds everything
# semantic -- no entity type names, no filters, no constraint parameters -- so the model must
# still ground the query itself; `tests/test_structure_hint.py` asserts each of those absences.
# The run manifest records `oracle_conditions`, and every record carries the hint it was given.
#
# S0 runs HERE rather than being read from matrix_full: the 2026-08-28 reorg changed the pipeline
# fingerprint, so the baseline has to come from the same code as S1/S2 to be comparable.
#
# PRE-REGISTERED DECISION RULE, against the trivial floor (0.5148 on the discriminating
# stratum), NOT against S0:
#   * S2 still below the floor -> composition is bounded below the floor. Drop both the
#     rebalance and decomposition; move to grounding. Most likely, and most valuable.
#   * S2 clears it, S1 does not -> constraint selection is the bottleneck, not tree shape.
#   * S1 clears it -> tree shape alone is the bottleneck.
#   * S1 ~ S2 ~ S0 -> check HINT COMPLIANCE first; that pattern is also what ignoring the
#     hint looks like, and a null result is uninterpretable without it.
#
#   sbatch --array=0-2 slurm/submit_structure_ablation.sh
#   sbatch --array=1 slurm/submit_structure_ablation.sh --resume     # one rung, resumed
#
# Then:
#   python src/eval_testset.py --merge-shards eval_runs/structure_ablation/shard_* #       --merge-output eval_runs/structure_ablation/merged

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

# One task per rung. Indices are into `prompt_strategies` in
# configs/eval_config.structure_ablation.json, which is ordered S0, S1, S2 for exactly this
# reason -- reordering that list silently relabels every task.
STRATEGY_IDX="$TASK"
MODEL_IDX=1      # llama31-8b-finetuned (verified by src/check_shard_map.py)
SCHEMA_IDX=0     # eval_config.schema_enforced is [true, false]; 0 == on

case "$TASK" in
  0) RUNG_NAME="S0 baseline / schema-allowed-entities-only" ;;
  1) RUNG_NAME="S1 ORACLE / oracle-skeleton-given" ;;
  2) RUNG_NAME="S2 ORACLE / oracle-skeleton-and-constraints-given" ;;
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
  SHARD_GLOB="$PROJECT_DIR/eval_runs/structure_ablation/shard_m${MODEL_IDX}_s${STRATEGY_IDX}_e${SCHEMA_IDX}_*"
  RESUME_DIR="$(ls -1dt $SHARD_GLOB 2>/dev/null | head -n 1)"
  if [ -z "$RESUME_DIR" ] || [ ! -d "$RESUME_DIR" ]; then
    echo "FATAL: --resume given but no run directory matches ${SHARD_GLOB}" >&2
    exit 1
  fi
  echo "Resuming from ${RESUME_DIR}"
  RESUME_ARGS=(--resume-from "$RESUME_DIR")
fi

echo "Starting structure ablation task=${TASK} rung=${RUNG_NAME} (model=${MODEL_IDX} strategy=${STRATEGY_IDX} schema=${SCHEMA_IDX}) at: $(date)"
if [ "$TASK" != "0" ]; then
  echo "NOTE: this is an ORACLE condition -- the model is shown part of the gold query."
  echo "      Its score is a CEILING and must never be reported as system performance."
fi
echo "Python: $(which python)"; python --version

set -e
python -u "$PROJECT_DIR/src/eval_testset.py"     --config "$PROJECT_DIR/configs/eval_config.structure_ablation.json"     --model-idx "${MODEL_IDX}"     --strategy-idx "${STRATEGY_IDX}"     --schema-idx "${SCHEMA_IDX}"     "${RESUME_ARGS[@]}"     2>&1 | tee "logs/nlxocpq-structabl-shard${TASK}.log"
rc=${PIPESTATUS[0]:-0}
echo "Structure ablation rung ${TASK} exit code: $rc"
exit $rc
