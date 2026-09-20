#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora
#SBATCH --output=logs/ocpq-lora-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

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

# Activate the project virtual environment.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train


# The PyTorch wheel used by src/train_unsloth.py needs a newer libstdc++.so.6 than
# the default cluster runtime provides. Load the newer GCC toolchain and keep
# its lib directory ahead of the conda environment libraries.
gcc_libstdcxx_path="$(gcc -print-file-name=libstdc++.so.6)"
gcc_libstdcxx_dir="$(dirname "$gcc_libstdcxx_path")"
gcc_libgcc_path="$(gcc -print-file-name=libgcc_s.so.1)"
gcc_libgcc_dir="$(dirname "$gcc_libgcc_path")"
if [[ -z "$gcc_libstdcxx_dir" || "$gcc_libstdcxx_dir" == "." ]]; then
	echo "Unable to resolve libstdc++.so.6 from the loaded GCC module." >&2
	exit 1
fi
if [[ -z "$gcc_libgcc_dir" || "$gcc_libgcc_dir" == "." ]]; then
	echo "Unable to resolve libgcc_s.so.1 from the loaded GCC module." >&2
	exit 1
fi
export LD_LIBRARY_PATH="$gcc_libstdcxx_dir:$gcc_libgcc_dir:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}"

# Set compilation flags to prevent Intel-optimized symbol dependencies
export CFLAGS="-march=x86-64 -mtune=generic"
export CXXFLAGS="-march=x86-64 -mtune=generic"
export RUSTFLAGS="-C target-cpu=generic"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG

# Make sure the environment already contains the training dependencies.
# Install these once outside the job script, not on every submission.

# Fail fast if the env drifted away from the torch/torchao pair this job expects.
# python -c "import importlib.metadata as m, sys; v = m.version('torchao'); print('torchao', v); sys.exit(0 if v == '0.12.0' else 1)"

# Run script
python --version
if [[ "${OCPQ_DEBUG_RUNTIME:-}" == "1" ]]; then
	echo "gcc: $(which gcc)"
	echo "gcc libstdc++: $(gcc -print-file-name=libstdc++.so.6)"
	echo "gcc libgcc_s: $(gcc -print-file-name=libgcc_s.so.1)"
	strings -a "$(gcc -print-file-name=libstdc++.so.6)" | grep -m1 CXXABI_1.3.11 || echo "missing CXXABI_1.3.11"
	echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
	echo "LD_PRELOAD=$LD_PRELOAD"
fi

LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}" \
	OCPQ_FORCE_4BIT=1 OCPQ_QUANTIZED_MODEL="mistralai/Mixtral-8x7B-Instruct-v0.1" \
	python -u src/train_unsloth.py

