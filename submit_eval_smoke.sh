#!/usr/bin/env bash
#SBATCH --job-name=nlxocpq_eval_smoke
#SBATCH --output=nlxocpq-eval-smoke-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
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

# Apply OCPQ_SMOKE_LIMIT by deriving a config rather than editing the committed one, so a
# tweaked slice can never leak into a later run.
SMOKE_CONFIG="$PROJECT_DIR/eval_config.smoke.json"
if [ -n "${OCPQ_SMOKE_LIMIT:-}" ]; then
  SMOKE_CONFIG="${TMPDIR:-/tmp}/eval_config.smoke.$SLURM_JOB_ID.json"
  python - "$PROJECT_DIR/eval_config.smoke.json" "$SMOKE_CONFIG" "$OCPQ_SMOKE_LIMIT" <<'PY'
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
RUN_DIR="$(ls -1dt "$PROJECT_DIR"/eval_runs/smoke/*/ 2>/dev/null | head -n 1)"
if [ -z "$RUN_DIR" ]; then
  echo "FATAL: smoke run produced no run directory under eval_runs/smoke" >&2
  exit 1
fi
echo "Checking $RUN_DIR"
python "$PROJECT_DIR/smoke_check.py" "$RUN_DIR"
