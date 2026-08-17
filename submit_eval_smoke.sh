#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_smoke
#SBATCH --output=nlxocpq-eval-smoke-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# 6h. Measured 2026-08-16: 141 of 144 runs took 4h01m, so a 4h limit missed the finish by three
# runs. Gemma 3 4B dominates -- despite being the smallest model it is 5-13x slower per run than
# Ministral 14B (381 s/run at worst against 28 s), because Gemma 3 requires eager attention and
# cannot use the fused kernels the others get. It alone accounted for 68% of the wall time.
# This does not threaten the matrix: there each array task runs one model, and the worst
# projected shard (gemma3-4b-finetuned x grammar-guided, 85 cases) is ~9h against a 14h limit.
#SBATCH --time=06:00:00
#SBATCH --account=thes2358

# Pre-launch smoke test for the 36-condition matrix.
#
# Deliberately a single job rather than an array: every condition runs in one process, so all six
# models (three bases + three adapters) are actually loaded. That is the failure mode a sharded
# run would only reveal task by task, hours apart. The array's index arithmetic is checked
# separately and without a GPU by check_shard_map.py.
#
# 36 conditions x 4 cases = 144 runs. Override the slice with OCPQ_SMOKE_LIMIT (2 for a faster
# ~1h pass, 4 for the log coverage the seed was chosen to give).
#
#   sbatch submit_eval_smoke.sh
#   python smoke_check.py eval_runs/smoke/<run_id>
#
# smoke_check.py is what makes this a test: the job exiting 0 only means nothing crashed.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

if [ -d "$SLURM_SUBMIT_DIR/NLxOCPQ" ]; then
  PROJECT_DIR="$SLURM_SUBMIT_DIR/NLxOCPQ"
else
  PROJECT_DIR="$SLURM_SUBMIT_DIR"
fi

module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

source "$HOME/miniforge3/etc/profile.d/conda.sh"
# The eval environment is a clone of the fine-tuning env plus scikit-learn and openai,
# which the eval path needs and training does not. Built by make_eval_env.sh, which
# verifies no package shared with training changed version. Falls back to the training
# env only if you have installed those two into it yourself.
conda activate "${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:${LD_LIBRARY_PATH:-}"

# The first thing worth smoke-testing is the environment itself: LLM.py imports sklearn and
# openai unguarded at module level, so a gap here kills every job before a model loads.
if [ ! -f "$PROJECT_DIR/check_eval_deps.py" ]; then
  echo "FATAL: $PROJECT_DIR/check_eval_deps.py not found — copy the current NLxOCPQ files to the cluster." >&2
  exit 1
fi
python "$PROJECT_DIR/check_eval_deps.py" --verbose || exit 1

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

OCPQ_CLI_DIR="$SLURM_SUBMIT_DIR/OCPQ/backend/ocpq_cli"
if [ -d "$OCPQ_CLI_DIR" ]; then
  echo "Pre-compiling OCPQ Rust CLI..."
  cargo build -q --release --manifest-path "$OCPQ_CLI_DIR/Cargo.toml"
fi

# OCPQ_SMOKE_CONFIG selects a subset smoke rather than duplicating this script per model set.
# e.g. OCPQ_SMOKE_CONFIG=$PWD/eval_config.smoke.gemma.json -> 12 conditions instead of 36.
SMOKE_CONFIG="${OCPQ_SMOKE_CONFIG:-$PROJECT_DIR/eval_config.smoke.json}"
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

python -u "$PROJECT_DIR/eval_testset.py" --config "$SMOKE_CONFIG" \
    2>&1 | tee "nlxocpq-eval-smoke.log"
rc=${PIPESTATUS[0]:-0}
echo "smoke eval exit code: $rc"
[ "$rc" -eq 0 ] || exit "$rc"

# The run finishing is necessary, not sufficient. Assert the shape of what it produced.
# Taken from eval_testset's own report rather than by globbing a hardcoded directory, which
# silently picked the wrong (or no) run as soon as a config set a different output_dir.
RUN_DIR="$(sed -n 's/^Evaluation complete\. Outputs saved in //p' nlxocpq-eval-smoke.log | tail -n 1)"
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
python "$PROJECT_DIR/smoke_check.py" "$RUN_DIR"
