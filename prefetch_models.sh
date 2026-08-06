#!/usr/bin/env bash
# Pre-download every checkpoint the fine-tuning jobs need, into the same HuggingFace cache
# the SLURM jobs read.
#
# Why: the submit scripts deliberately throttle downloads (HF_HUB_ENABLE_HF_TRANSFER=0,
# HF_HUB_DOWNLOAD_THREADS=1, HF_HUB_DISABLE_XET=1) to reduce flakiness on the cluster. That
# makes a cold download slow, and on a compute node that time is billed against the job's
# wall clock — a 10-minute smoke test never got past downloading for LLaMA and Phi-4. Run
# this once on a login node so the training jobs start with everything local.
#
# Usage (login node):
#     bash prefetch_models.sh
#
# Then verify the cache is complete by launching a short job with HF_HUB_OFFLINE=1 — if it
# reaches the "[train_unsloth] BOS check OK" line, nothing else needs downloading.

set -euo pipefail

# --- same cache location the submit scripts use ---------------------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE"

# gemma-3 is a gated repo: this must be a token that has accepted its license.
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

# Unlike the training jobs, a login node is not paying GPU time per second, so use the fast
# path here. Override to 0/1 if the cluster's egress is unhappy with parallel connections.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-8}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate nlxocpq-train-next

python - <<'PY'
import os
from huggingface_hub import snapshot_download

# What each submit script asks for, plus the checkpoints unsloth silently substitutes.
#
# With load_in_4bit=True, unsloth swaps a full-precision id for its own pre-quantized
# redistribution. That is why the previous Gemma adapter_config recorded
# `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` even though the job requested
# `google/gemma-3-4b-it`. Both are fetched so the substitution cannot trigger a download
# inside the job. LLaMA and Phi-4 already name unsloth ids, so no substitution applies.
REPOS = [
    ("unsloth/llama-3-8b-Instruct-bnb-4bit", "llama3-8b  (requested)"),
    ("unsloth/phi-4-unsloth-bnb-4bit",       "phi4-14b   (requested)"),
    ("google/gemma-3-4b-it",                 "gemma3-4b  (requested; gated)"),
    ("unsloth/gemma-3-4b-it-unsloth-bnb-4bit", "gemma3-4b  (unsloth 4-bit substitution)"),
]

token = os.environ.get("HF_TOKEN") or None
failures = []

for repo_id, label in REPOS:
    print(f"\n=== {label}: {repo_id} ===", flush=True)
    try:
        path = snapshot_download(repo_id=repo_id, token=token, resume_download=True)
        total = sum(
            os.path.getsize(os.path.join(root, f))
            for root, _dirs, files in os.walk(path)
            for f in files
            if not os.path.islink(os.path.join(root, f))
        )
        print(f"  OK  {path}  ({total / 1024**3:.2f} GiB)")
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        failures.append((repo_id, str(exc)))

print("\n" + "=" * 72)
if failures:
    print("Some downloads failed:")
    for repo_id, err in failures:
        print(f"  - {repo_id}: {err[:160]}")
    print("\nIf a gated repo failed with 401/403, accept its license on huggingface.co")
    print("with the same account as HF_TOKEN, then re-run.")
    raise SystemExit(1)
print("All checkpoints cached under:", os.environ["HF_HUB_CACHE"])
print("The training jobs read this same cache, so they will start without downloading.")
print("\nIMPORTANT: do NOT run the jobs with OCPQ_CLEAR_HF_CACHE=1 — it deletes this cache.")
PY
