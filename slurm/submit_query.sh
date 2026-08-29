#!/usr/bin/env zsh
#SBATCH --job-name=query_run
#SBATCH --output=logs/query-run-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --time=00:06:00

# set -euo pipefail

# Always run from the folder where this script lives (repo-local, no username path).
# Run from the directory where sbatch was invoked
# Resolve the repo root from this script's own location, so the job works whether it was
# submitted from the repo root (`sbatch slurm/<this>.sh`) or from inside `slurm/`, and
# regardless of where $SLURM_SUBMIT_DIR points. Everything below is repo-root-relative.
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
mkdir -p logs

# CRITICAL: Purge all modules first, then reload only what we need
module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

# Activate the project virtual environment
source "$HOME/envs/nlxocpq-venv/bin/activate"
# export LD_LIBRARY_PATH="$HOME/envs/nlxocpq-venv/lib:${LD_LIBRARY_PATH:-}"
# Keep the venv's runtime libraries first; the Actran directory ships an older
# libstdc++.so.6 that breaks numpy/torch on this cluster.
export LD_LIBRARY_PATH="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib:$HOME/envs/nlxocpq-venv/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$gcc_lib_dir/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"

# Set compilation flags to prevent Intel-optimized symbol dependencies
export CFLAGS="-march=x86-64 -mtune=generic"
export CXXFLAGS="-march=x86-64 -mtune=generic"
export RUSTFLAGS="-C target-cpu=generic"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG

# Run script
echo "Starting diagnostic tests at: $(date)"
echo "Python binary: $(which python)"
python --version

echo "Test 1: Simple print"
python -c "print('hello world')" || echo "CRASHED: simple print"

echo "Test 2: Import sys"
python -c "import sys; print(sys.version)" || echo "CRASHED: import sys"

echo "Test 3: Import numpy"
python -c "import numpy; print(numpy.__version__)" || echo "CRASHED: import numpy"

echo "Test 4: Check CUDA availability"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count())" || echo "CRASHED: torch CUDA check"

echo "Test 5: Run src/cli.py with --help"
python src/cli.py --help > /dev/null 2>&1 && echo "src/cli.py --help: OK" || echo "CRASHED: src/cli.py --help"

echo ""
echo "Starting actual CLI at: $(date)"
set +e
python -u src/cli.py \
  --backend local-gpu \
  --model models/lora_model_ocpq \
  --prompt-strategy schema-allowed-entities-only \
  --ocel ../OCEL_Logs/hospital_log_high.json \
  "For every person, the diagnosis must occur exactly once." \
  > logs/query-cli.stdout 2> logs/query-cli.stderr
cli_status=$?
set -e

echo "CLI exit status: $cli_status"
echo "--- STDOUT (last 100 lines) ---"
tail -n 100 logs/query-cli.stdout
echo "--- STDERR (last 50 lines) ---"
tail -n 50 logs/query-cli.stderr

exit $cli_status
