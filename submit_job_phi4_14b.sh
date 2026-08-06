#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora-phi4-14b
#SBATCH --output=ocpq-lora-phi4-14b-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# 16h: the 6h22m reference run trained at max_seq_length=1024, which truncated a large share
# of the corpus. At 4096 nothing is truncated, so mean sequence length (and wall time) rises.
#SBATCH --time=16:00:00

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
# Deliberately NOT setting TRANSFORMERS_CACHE. It is deprecated (transformers warns on every
# run) and, when set, transformers uses it as the hub cache *instead of* HF_HUB_CACHE. That
# split the cache in two: snapshot_download wrote to $HF_HOME/hub while transformers looked in
# $HF_HOME/transformers, so a pre-populated cache was invisible and HF_HUB_OFFLINE=1 failed with
# "No config file found". Unset it explicitly in case it is exported by the login shell.
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-1}"
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-false}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING="${HF_HUB_DISABLE_EXPERIMENTAL_WARNING:-1}"
export HF_HUB_OFFLINE=1
if [[ "${OCPQ_CLEAR_HF_CACHE:-0}" == "1" ]]; then
	rm -rf "$HF_HUB_CACHE" "$HF_HOME/xet" "$HF_HOME/transformers"
fi
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

# Reduce download flakiness on HPC.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

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
## Add Intel runtime libs (libsvml, libiomp5) used by bitsandbytes/triton if present
intel_runtime_lib_dir="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib"
intel_libsvml="$intel_runtime_lib_dir/libsvml.so"
intel_libiomp5="$intel_runtime_lib_dir/libiomp5.so"
intel_libintlc="$intel_runtime_lib_dir/libintlc.so"

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
export OCPQ_ENABLE_TRAINER_EVAL="${OCPQ_ENABLE_TRAINER_EVAL:-1}"
export OCPQ_ENABLE_FINAL_EVAL="${OCPQ_ENABLE_FINAL_EVAL:-1}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"
export OCPQ_TRAIN_BATCH_SIZE="${OCPQ_TRAIN_BATCH_SIZE:-1}"
# Eval batch size is decoupled from the train batch size on purpose. Training is capped at 1
# by memory (gradients + optimizer state); evaluation runs under no_grad, so a larger batch
# fits easily on an H100 and cuts eval wall time several-fold. With eval_steps=250 over ~6750
# steps there are ~27 evaluations over 6000 validation rows, which dominated the previous
# runtimes. This is a pure speed change: eval loss is token-weighted and padding is masked to
# -100, so the value is identical at any batch size.
export OCPQ_EVAL_BATCH_SIZE="${OCPQ_EVAL_BATCH_SIZE:-8}"
export OCPQ_GRADIENT_ACCUMULATION_STEPS="${OCPQ_GRADIENT_ACCUMULATION_STEPS:-8}"
# 4096, not 1024: at 1024 a large share of finetune_ocpq.jsonl rows exceeded the budget and
# SFTTrainer truncates the END of the sequence, i.e. the assistant's JSON target. 4096 also
# covers the longest inference prompt (chain-of-thought-plus, ~4k tokens).
export OCPQ_MAX_SEQ_LENGTH="${OCPQ_MAX_SEQ_LENGTH:-4096}"
# H100 has 93 GB — gradient checkpointing not needed and avoids unsloth Triton kernel bugs.
export OCPQ_GRADIENT_CHECKPOINTING="${OCPQ_GRADIENT_CHECKPOINTING:-false}"

# Resume behavior: prefer the lightweight adapter snapshot if available.
# Default to a FRESH run. The previous default ("adapter") resolves to adapters/<run_label>,
# so a new run would silently load the *existing* adapter's LoRA weights and continue training
# on top of them -- carrying over whatever the previous run learned. That is wrong whenever the
# data or training config changed. To resume a run that was interrupted (e.g. wall-time kill),
# set OCPQ_RESUME_FROM_CHECKPOINT=latest explicitly, which restores full trainer state.
export OCPQ_RESUME_FROM_CHECKPOINT="${OCPQ_RESUME_FROM_CHECKPOINT:-0}"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG

# Model selection.
: "${OCPQ_QUANTIZED_MODEL:=unsloth/phi-4-unsloth-bnb-4bit}"
: "${OCPQ_MODEL_LABEL:=phi4-14b}"
: "${OCPQ_CHAT_TEMPLATE:=auto}"
: "${OCPQ_ATTENTION_IMPLEMENTATION:=eager}"

python --version

LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}" PYTORCH_NVML_BASED_CUDA_CHECK="$PYTORCH_NVML_BASED_CUDA_CHECK" \
	OCPQ_FORCE_4BIT=1 OCPQ_QUANTIZED_MODEL="$OCPQ_QUANTIZED_MODEL" OCPQ_MODEL_LABEL="$OCPQ_MODEL_LABEL" OCPQ_CHAT_TEMPLATE="$OCPQ_CHAT_TEMPLATE" OCPQ_ATTENTION_IMPLEMENTATION="$OCPQ_ATTENTION_IMPLEMENTATION" OCPQ_DATASET_NUM_PROC="$OCPQ_DATASET_NUM_PROC" OCPQ_ENABLE_TRAINER_EVAL="$OCPQ_ENABLE_TRAINER_EVAL" OCPQ_ENABLE_FINAL_EVAL="$OCPQ_ENABLE_FINAL_EVAL" \
	OCPQ_RESUME_FROM_CHECKPOINT="$OCPQ_RESUME_FROM_CHECKPOINT" python -u train_unsloth.py
