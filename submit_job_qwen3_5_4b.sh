#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora-qwen3-5-4b
#SBATCH --output=ocpq-lora-qwen3-5-4b-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

set -euo pipefail

# Run from the directory where sbatch was invoked
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR not set}"

# CRITICAL: Purge all modules first, then reload only what we need
module purge
module load GCC/13.3.0
module load CUDA/12.4.0
module load Python/3.12.3

unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER
unset PYTHONHOME PYTHONPATH

# Activate the experimental environment for newer models.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train-next

# Use a persistent cache by default to avoid shared FS / temp filesystem issues.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-false}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"
if [[ "${OCPQ_CLEAR_HF_CACHE:-0}" == "1" ]]; then
	rm -rf "$HF_HUB_CACHE" "$HF_HOME/xet" "$TRANSFORMERS_CACHE"
fi
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"

# Reduce download flakiness on HPC.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

# Add the Intel compiler runtime that provides libsvml.so for bitsandbytes/triton.
intel_runtime_lib_dir="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib"
intel_libsvml="$intel_runtime_lib_dir/libsvml.so"
intel_libiomp5="$intel_runtime_lib_dir/libiomp5.so"
intel_libintlc="$intel_runtime_lib_dir/libintlc.so"

# The PyTorch wheel used by train_unsloth.py needs a newer libstdc++.so.6 than
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
for intel_lib in "$intel_libsvml" "$intel_libiomp5" "$intel_libintlc"; do
	if [[ ! -f "$intel_lib" ]]; then
		echo "Missing Intel runtime library: $intel_lib" >&2
		exit 1
	fi
done
export LD_LIBRARY_PATH="$intel_runtime_lib_dir:$gcc_libstdcxx_dir:$gcc_libgcc_dir:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path:$intel_libsvml:$intel_libiomp5${LD_PRELOAD:+:$LD_PRELOAD}"

# Set compilation flags to prevent Intel-optimized symbol dependencies
export CFLAGS="-march=x86-64 -mtune=generic"
export CXXFLAGS="-march=x86-64 -mtune=generic"
export RUSTFLAGS="-C target-cpu=generic"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OCPQ_DATASET_NUM_PROC="${OCPQ_DATASET_NUM_PROC:-1}"
export OCPQ_ENABLE_EVAL="${OCPQ_ENABLE_EVAL:-1}"
export OCPQ_DISABLE_TQDM="${OCPQ_DISABLE_TQDM:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export TQDM_MONITOR_INTERVAL="${TQDM_MONITOR_INTERVAL:-0}"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG

# Model selection (use a 4-bit or full-precision ID as needed).
: "${OCPQ_QUANTIZED_MODEL:=Qwen/Qwen3.5-4B}"
: "${OCPQ_MODEL_LABEL:=qwen3.5-4b}"
: "${OCPQ_CHAT_TEMPLATE:=qwen-2.5}"

python --version

# Preload GCC ABI libraries first, then the Intel runtime needed by bitsandbytes/triton.
LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path:$intel_libsvml:$intel_libiomp5${LD_PRELOAD:+:$LD_PRELOAD}" \
	OCPQ_QUANTIZED_MODEL="$OCPQ_QUANTIZED_MODEL" OCPQ_MODEL_LABEL="$OCPQ_MODEL_LABEL" OCPQ_CHAT_TEMPLATE="$OCPQ_CHAT_TEMPLATE" OCPQ_DATASET_NUM_PROC="$OCPQ_DATASET_NUM_PROC" OCPQ_ENABLE_EVAL="$OCPQ_ENABLE_EVAL" OCPQ_DISABLE_TQDM="$OCPQ_DISABLE_TQDM" TQDM_DISABLE="$TQDM_DISABLE" TQDM_MONITOR_INTERVAL="$TQDM_MONITOR_INTERVAL" \
	python -u train_unsloth.py
