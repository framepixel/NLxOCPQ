#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_smoke
#SBATCH --output=logs/nlxocpq-eval-smoke-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# 6h. Measured 2026-08-16: 141 of 144 runs took 4h01m, so a 4h limit missed the finish by three
# runs. That measurement included Gemma 3 4B, which alone accounted for 68% of the wall time --
# despite being the smallest model it ran 5-13x slower per run than Ministral 14B (381 s/run at
# worst against 28 s). Gemma was replaced by Ministral 3 3B on 2026-08-18, whose vocabulary is
# 131,072 against Gemma's 262,208 and which shares the 14B's architecture, so the full smoke
# should now finish well inside this limit. The 6h is kept until a full run re-measures it.
#SBATCH --time=06:00:00
#SBATCH --account=thes2358
# Site default may be --export=NONE; options are passed as arguments regardless, this is belt-and-braces.
#SBATCH --export=ALL

# Pre-launch smoke test for the 36-condition matrix.
#
# Deliberately a single job rather than an array: every condition runs in one process, so all six
# models (three bases + three adapters) are actually loaded. That is the failure mode a sharded
# run would only reveal task by task, hours apart. The array's index arithmetic is checked
# separately and without a GPU by src/check_shard_map.py.
#
# 36 conditions x 4 cases = 144 runs. Options are ARGUMENTS, not environment variables --
# this cluster does not propagate the submitting environment into jobs.
#
#   sbatch slurm/submit_eval_smoke.sh
#   sbatch slurm/submit_eval_smoke.sh --limit 2                               # faster ~1h pass
#   sbatch slurm/submit_eval_smoke.sh --config configs/eval_config.smoke.gemma.json   # subset
#   python src/smoke_check.py eval_runs/smoke/<run_id>
#
# src/smoke_check.py is what makes this a test: the job exiting 0 only means nothing crashed.

set -euo pipefail

# Resolve the repo root from this script's own location, so the job works whether it was
# submitted from the repo root (`sbatch slurm/<this>.sh`) or from inside `slurm/`, and
# regardless of where $SLURM_SUBMIT_DIR points. Everything below is repo-root-relative.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p logs

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

source "$HOME/miniforge3/etc/profile.d/conda.sh"
# The eval environment is a clone of the fine-tuning env plus scikit-learn and openai,
# which the eval path needs and training does not. Built by scripts/make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

# The first thing worth smoke-testing is the environment itself: LLM.py imports sklearn and
# openai unguarded at module level, so a gap here kills every job before a model loads.
if [ ! -f "$PROJECT_DIR/src/check_eval_deps.py" ]; then
  echo "FATAL: $PROJECT_DIR/src/check_eval_deps.py not found — copy the current NLxOCPQ files to the cluster." >&2
  exit 1
fi
python "$PROJECT_DIR/src/check_eval_deps.py" --verbose || exit 1

[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
if [ "${OCPQ_ALLOW_HUB:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1
fi

export OCPQ_CLI_SCRATCH_DIR="${OCPQ_CLI_SCRATCH_DIR:-${TMPDIR:-/tmp}/ocpq-smoke-$SLURM_JOB_ID}"
mkdir -p "$OCPQ_CLI_SCRATCH_DIR"
trap 'rm -rf "$OCPQ_CLI_SCRATCH_DIR"' EXIT

export OCPQ_SCHEMA_TOKEN_BUDGET="${OCPQ_SCHEMA_TOKEN_BUDGET:-17000}"

OCPQ_CLI_DIR="${OCPQ_REPO_DIR:-$(dirname "$PROJECT_DIR")/OCPQ}/backend/ocpq_cli"
if [ -d "$OCPQ_CLI_DIR" ]; then
  echo "Pre-compiling OCPQ Rust CLI..."
  cargo build -q --release --manifest-path "$OCPQ_CLI_DIR/Cargo.toml"
fi

# Options come from ARGUMENTS, not the environment: this cluster does not propagate the
# submitting environment into jobs (verified in jobs 3015021 and 3015720). Env vars are kept as
# a fallback for direct (non-sbatch) invocation.
#
#   sbatch slurm/submit_eval_smoke.sh                                        # full 36-condition smoke
#   sbatch slurm/submit_eval_smoke.sh --config configs/eval_config.smoke.gemma.json  # subset
#   sbatch slurm/submit_eval_smoke.sh --limit 2                              # faster slice
SMOKE_CONFIG_ARG=""
SMOKE_LIMIT_ARG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --config) SMOKE_CONFIG_ARG="$2"; shift 2 ;;
    --limit)  SMOKE_LIMIT_ARG="$2";  shift 2 ;;
    *) echo "Warning: ignoring unrecognised argument '$1'" >&2; shift ;;
  esac
done
[ -n "$SMOKE_CONFIG_ARG" ] && OCPQ_SMOKE_CONFIG="$SMOKE_CONFIG_ARG"
[ -n "$SMOKE_LIMIT_ARG" ] && OCPQ_SMOKE_LIMIT="$SMOKE_LIMIT_ARG"

SMOKE_CONFIG="${OCPQ_SMOKE_CONFIG:-$PROJECT_DIR/configs/eval_config.smoke.json}"
case "$SMOKE_CONFIG" in
  /*) ;;
  *) SMOKE_CONFIG="$PROJECT_DIR/$SMOKE_CONFIG" ;;
esac
if [ ! -f "$SMOKE_CONFIG" ]; then
  echo "FATAL: smoke config not found: $SMOKE_CONFIG" >&2
  exit 1
fi
echo "Smoke config: $SMOKE_CONFIG"

# Apply OCPQ_SMOKE_LIMIT by deriving a config rather than editing the committed one, so a
# tweaked slice can never leak into a later run. Derives from the *resolved* config above.
if [ -n "${OCPQ_SMOKE_LIMIT:-}" ]; then
  SMOKE_SRC="$SMOKE_CONFIG"
  SMOKE_CONFIG="${TMPDIR:-/tmp}/eval_config.smoke.$SLURM_JOB_ID.json"
  python - "$SMOKE_SRC" "$SMOKE_CONFIG" "$OCPQ_SMOKE_LIMIT" <<'PY'
import json, sys
src, dst, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = json.load(open(src, encoding="utf-8"))
cfg["limit"] = limit
json.dump(cfg, open(dst, "w", encoding="utf-8"), indent=2)
print(f"smoke limit overridden to {limit}")
PY
fi

echo "Starting smoke evaluation at: $(date)"
echo "Python: $(which python)"; python --version

python -u "$PROJECT_DIR/src/eval_testset.py" --config "$SMOKE_CONFIG" \
    2>&1 | tee "logs/nlxocpq-eval-smoke.log"
rc=${PIPESTATUS[0]:-0}
echo "smoke eval exit code: $rc"
[ "$rc" -eq 0 ] || exit "$rc"

# The run finishing is necessary, not sufficient. Assert the shape of what it produced.
# Taken from eval_testset's own report rather than by globbing a hardcoded directory, which
# silently picked the wrong (or no) run as soon as a config set a different output_dir.
RUN_DIR="$(sed -n 's/^Evaluation complete\. Outputs saved in //p' logs/nlxocpq-eval-smoke.log | tail -n 1)"
case "$RUN_DIR" in
  /*) ;;                                  # already absolute
  "") ;;                                  # handled below
  *) RUN_DIR="$PROJECT_DIR/$RUN_DIR" ;;   # eval_testset reports a path relative to its cwd
esac
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "FATAL: could not determine the smoke run directory from the evaluation output" >&2
  exit 1
fi
echo "Checking $RUN_DIR"
python "$PROJECT_DIR/src/smoke_check.py" "$RUN_DIR"
