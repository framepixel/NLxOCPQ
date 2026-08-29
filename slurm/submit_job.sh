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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
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

