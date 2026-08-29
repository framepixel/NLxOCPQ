#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-lora-ministral3-3b
#SBATCH --output=logs/ocpq-lora-ministral3-3b-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
# 12h, matching the Llama 3.1 8B script. The 14B took 8h19m and the 8B 4h01m, so a 3B should
# finish well inside this; the margin absorbs the eval passes, which consumed 54-58% of every
# previous run because unsloth forces eval batch size to 1 internally.
#SBATCH --time=12:00:00
#SBATCH --account=thes2358

set -euo pipefail

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

# Activate the experimental environment for newer models.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${OCPQ_TRAIN_ENV:-nlxocpq-train-v5}"

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
# Offline by design: a 10-hour training run must not be able to fail on a hub outage or a
# rate limit partway through. This REQUIRES the cache to be populated first — Llama 3.1 8B
# and Ministral 3 have never been downloaded on this cluster, and without them the job dies
# with "No config file found" before training starts. Run on a login node beforehand:
#     bash scripts/prefetch_models.sh  &&  VERIFY=1 bash scripts/prefetch_models.sh
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
# Default to a FRESH run. The previous default ("adapter") resolves to models/adapters/<run_label>,
# so a new run would silently load the *existing* adapter's LoRA weights and continue training
# on top of them -- carrying over whatever the previous run learned. That is wrong whenever the
# data or training config changed. To resume a run that was interrupted (e.g. wall-time kill),
# set OCPQ_RESUME_FROM_CHECKPOINT=latest explicitly, which restores full trainer state.
export OCPQ_RESUME_FROM_CHECKPOINT="${OCPQ_RESUME_FROM_CHECKPOINT:-0}"

# Ensure no Intel paths contaminate the build
unset PKG_CONFIG_PATH
unset LIBSQLITE3_SYS_USE_PKG_CONFIG

# Model selection.
# Ministral 3 3B replaces Gemma 3 4B as the small-model arm: 256k context. Multimodal
# (3.4B language + 0.4B vision), so src/train_unsloth.py attaches LoRA to the language tower
# only. CHAT_TEMPLATE must stay `auto` - a named unsloth preset would impose Llama-3
# formatting on a Mistral model.
#
# CRITICAL: this names an unsloth PRE-QUANTIZED id, not `mistralai/Ministral-3-3B-...`.
# With load_in_4bit=True unsloth silently substitutes its own redistribution for a
# full-precision id, so the adapter ends up trained against weights that are not what
# inference loads. That is exactly what happened to Gemma 3, whose script requested
# `google/gemma-3-4b-it`, and it could not be undone: the substituted artifact fails to
# load for inference at all ("FP4 quantization state not initialized", job 3015021).
# Requesting the unsloth id directly makes train and eval agree by construction, as they
# already do for Llama 3.1 and Ministral 14B.
#
# The `-unsloth-bnb-4bit` (dynamic) flavour deliberately matches the 14B rather than the
# plain `-bnb-4bit`: both rungs of the 3B<->14B size ladder must be quantized identically
# for that comparison to isolate scale.
: "${OCPQ_QUANTIZED_MODEL:=unsloth/Ministral-3-3B-Instruct-2512-unsloth-bnb-4bit}"
: "${OCPQ_MODEL_LABEL:=ministral3-3b}"
: "${OCPQ_CHAT_TEMPLATE:=auto}"
: "${OCPQ_ATTENTION_IMPLEMENTATION:=eager}"

python --version

LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}" PYTORCH_NVML_BASED_CUDA_CHECK="$PYTORCH_NVML_BASED_CUDA_CHECK" \
	OCPQ_FORCE_4BIT=1 OCPQ_QUANTIZED_MODEL="$OCPQ_QUANTIZED_MODEL" OCPQ_MODEL_LABEL="$OCPQ_MODEL_LABEL" OCPQ_CHAT_TEMPLATE="$OCPQ_CHAT_TEMPLATE" OCPQ_ATTENTION_IMPLEMENTATION="$OCPQ_ATTENTION_IMPLEMENTATION" OCPQ_DATASET_NUM_PROC="$OCPQ_DATASET_NUM_PROC" OCPQ_ENABLE_TRAINER_EVAL="$OCPQ_ENABLE_TRAINER_EVAL" OCPQ_ENABLE_FINAL_EVAL="$OCPQ_ENABLE_FINAL_EVAL" \
	OCPQ_RESUME_FROM_CHECKPOINT="$OCPQ_RESUME_FROM_CHECKPOINT" python -u src/train_unsloth.py
