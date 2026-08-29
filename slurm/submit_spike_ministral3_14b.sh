#!/usr/bin/env zsh
#SBATCH --job-name=ocpq-spike-ministral3
#SBATCH --output=logs/ocpq-spike-ministral3-%J.out
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
# A spike, not a run: ~200 rows and a few hundred steps. 2h is generous and exists only so a
# hung download or a compile stall cannot sit on the GPU.
#SBATCH --time=02:00:00

# SPIKE — does Ministral 3 14B load, tokenize and fine-tune in this pipeline at all?
#
# Ministral-3-14B-Instruct-2512 has a 256k context, which removes the accumulated-history
# overflow that Phi-4's 16k window is exposed to under chain-of-thought-plus. Before adopting
# it, four things have to be checked, in the order they will bite:
#
#   1. transformers v5 / mistral-common. The working env (`nlxocpq-train-next`) pins
#      transformers 4.57.6 and produced all three current adapters — it must NOT be upgraded
#      (CLAUDE.md 2.2). This script therefore activates a SEPARATE env.
#   2. Chat template. Mistral's is [INST]...[/INST], unlike the other three models. The
#      duplicate-BOS check in src/train_unsloth.py is what catches a mis-applied template.
#   3. Vision encoder. The model is 13.5B language + 0.4B vision. Text-only LoRA must attach
#      to the language tower — this is the Gemma-3 situation that previously caused the
#      adapter-provenance problem.
#   4. Quantization. The instruct release is FP8 while this pipeline trains 4-bit bnb, so the
#      4-bit repo id has to be confirmed rather than assumed.
#
# PASS => swap Phi-4 out. FAIL => keep Phi-4 14B; the SCHEMA_BLOCK_TOKEN_BUDGET in LLM.py is
# what keeps its 16k window sufficient.
#
# Usage:
#   python src/make_spike_slice.py --rows 200        # once, creates finetune_ocpq.spike.jsonl
#   sbatch slurm/submit_spike_ministral3_14b.sh

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

# A SEPARATE environment from nlxocpq-train-next: Ministral 3 needs transformers v5, and
# upgrading the env that trained the superseded adapters would make those results
# unreproducible. Built from requirements/requirements.unsloth-train-v5.txt (formerly named
# requirements.ministral-spike.txt, before the spike passed and the stack became the real
# training environment).
#
# This spike has already passed (2026-08-10, readiness note 6.6); the script is kept so the
# result can be reproduced. The default below is the env the spike actually ran in. If that
# env no longer exists, point OCPQ_SPIKE_ENV at nlxocpq-train-v5 — same stack.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${OCPQ_SPIKE_ENV:-nlxocpq-spike-ministral}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
unset TRANSFORMERS_CACHE
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-1}"
export HF_HUB_DISABLE_SYMLINKS_WARNING="${HF_HUB_DISABLE_SYMLINKS_WARNING:-1}"
# Unlike the training jobs this must be ONLINE the first time — the weights are not cached yet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY CUDA_MPS_ACTIVE_THREAD_PERCENTAGE

# Same libstdc++/Intel runtime handling as the training jobs — the PyTorch wheel needs a newer
# libstdc++ than the default cluster runtime provides.
gcc_libstdcxx_path="$(gcc -print-file-name=libstdc++.so.6)"
gcc_libstdcxx_dir="$(dirname "$gcc_libstdcxx_path")"
gcc_libgcc_path="$(gcc -print-file-name=libgcc_s.so.1)"
gcc_libgcc_dir="$(dirname "$gcc_libgcc_path")"
intel_runtime_lib_dir="/cvmfs/software.hpc.rwth.de/Linux/RH8/x86_64/intel/skylake_avx512/software/intel-compilers/2024.2.0/compiler/2024.2/lib"
export LD_LIBRARY_PATH="$intel_runtime_lib_dir:$gcc_libstdcxx_dir:$gcc_libgcc_dir:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NVML_BASED_CUDA_CHECK="${PYTORCH_NVML_BASED_CUDA_CHECK:-1}"

# Spike-sized training. Small slice, frequent logging so the loss curve is visible early, and
# a fresh run so nothing is warm-started from an existing adapter.
export OCPQ_FINETUNE_DATASET="${OCPQ_FINETUNE_DATASET:-finetune_ocpq.spike.jsonl}"
export OCPQ_TRAIN_BATCH_SIZE="${OCPQ_TRAIN_BATCH_SIZE:-1}"
export OCPQ_EVAL_BATCH_SIZE="${OCPQ_EVAL_BATCH_SIZE:-4}"
export OCPQ_GRADIENT_ACCUMULATION_STEPS="${OCPQ_GRADIENT_ACCUMULATION_STEPS:-4}"
export OCPQ_MAX_SEQ_LENGTH="${OCPQ_MAX_SEQ_LENGTH:-4096}"
export OCPQ_GRADIENT_CHECKPOINTING="${OCPQ_GRADIENT_CHECKPOINTING:-false}"
export OCPQ_LOGGING_STEPS="${OCPQ_LOGGING_STEPS:-5}"
export OCPQ_EVAL_STEPS="${OCPQ_EVAL_STEPS:-25}"
export OCPQ_NUM_TRAIN_EPOCHS="${OCPQ_NUM_TRAIN_EPOCHS:-1}"
export OCPQ_RESUME_FROM_CHECKPOINT=0
# Keep the spike's artifacts away from models/adapters/ so a failed spike cannot be mistaken for a
# usable adapter, and cannot overwrite a real one.
export OCPQ_OUTPUT_DIR="${OCPQ_OUTPUT_DIR:-outputs/spike-ministral3-14b}"
export OCPQ_ADAPTER_OUTPUT_DIR="${OCPQ_ADAPTER_OUTPUT_DIR:-outputs/spike-ministral3-14b/adapter}"
# Show a few formatted examples so the chat template can be eyeballed in the log.
export OCPQ_PREVIEW_EXAMPLES="${OCPQ_PREVIEW_EXAMPLES:-3}"

# Model selection. `auto` keeps the tokenizer's own [INST] template — do NOT substitute a
# named unsloth preset, which would silently impose Llama-3 formatting on a Mistral model.
#
# EXPORT, not `: "${VAR:=...}"`. A bare parameter assignment sets a *shell* variable that the
# python child never sees, so src/train_unsloth.py falls back to its own default
# (unsloth/llama-3-8b-Instruct-bnb-4bit) and trains the wrong model — while this script's own
# echo still prints the intended name. That exact mistake made one spike look like a pass.
export OCPQ_QUANTIZED_MODEL="${OCPQ_QUANTIZED_MODEL:-unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit}"
export OCPQ_MODEL_LABEL="${OCPQ_MODEL_LABEL:-ministral3-14b}"
export OCPQ_CHAT_TEMPLATE="${OCPQ_CHAT_TEMPLATE:-auto}"
export OCPQ_ATTENTION_IMPLEMENTATION="${OCPQ_ATTENTION_IMPLEMENTATION:-eager}"

echo "=== SPIKE: ${OCPQ_QUANTIZED_MODEL} ==="
python --version
python -c "import transformers, sys; print('transformers', transformers.__version__)"
python -c "import unsloth, sys; print('unsloth', getattr(unsloth,'__version__','?'))" || true
python -c "import mistral_common, sys; print('mistral_common', mistral_common.__version__)" || \
	echo "mistral_common NOT installed — tokenization will likely fail"

# Fail fast if the export above did not reach the child process. Two seconds here beats
# discovering from a parameter count at the end that the wrong model trained.
python - <<'PYCHECK'
import os
import sys
model = os.environ.get("OCPQ_QUANTIZED_MODEL", "")
print(f"python sees OCPQ_QUANTIZED_MODEL = {model!r}")
if "ministral" not in model.lower():
    sys.exit("ABORT: the spike model did not reach python; it would train the default instead.")
PYCHECK

LD_PRELOAD="$gcc_libgcc_path:$gcc_libstdcxx_path${LD_PRELOAD:+:$LD_PRELOAD}" \
	OCPQ_FORCE_4BIT=1 python -u src/train_unsloth.py

echo
echo "=== SPIKE GO/NO-GO — check the log above ==="
echo "  0. FIRST: 'Trainable parameters = X of ~14,000,000,000' and a 'Training label:'"
echo "     containing 'ministral'. ~8.07B or a llama-3 label means the wrong model trained"
echo "     and every other check below is meaningless."
echo "  1. model loaded in 4-bit without an unknown-architecture error"
echo "  2. BOS/EOS check reported exactly one leading BOS (no duplicate-BOS warning)"
echo "  3. the previewed examples show [INST] ... [/INST], not <|start_header_id|>/<|eot_id|>"
echo "  4. LoRA attached and trainable params are non-zero"
echo "  5. training loss decreased over the logged steps"
echo "  6. adapter saved under ${OCPQ_ADAPTER_OUTPUT_DIR}"
echo "Any failure => keep Phi-4 14B and rely on SCHEMA_BLOCK_TOKEN_BUDGET."
