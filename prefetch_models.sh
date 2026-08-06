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
# TRANSFORMERS_CACHE must stay unset. It is deprecated, and when set transformers uses it as
# the hub cache *instead of* HF_HUB_CACHE — which previously pointed downloads at
# $HF_HOME/hub while transformers looked in $HF_HOME/transformers.
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HOME" "$HF_HUB_CACHE"

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
# VERIFY=1 resolves everything from the local cache only, exactly as a job with
# HF_HUB_OFFLINE=1 would. Use it to confirm the cache is complete without burning GPU time.
verify_only = os.environ.get("VERIFY", "0").strip().lower() in {"1", "true", "yes", "on"}
if verify_only:
    print("VERIFY=1 -> resolving from local cache only, downloading nothing.\n")
failures = []

for repo_id, label in REPOS:
    print(f"\n=== {label}: {repo_id} ===", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo_id,
            token=token,
            resume_download=not verify_only,
            local_files_only=verify_only,
        )
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
    verb = "could not be resolved from the cache" if verify_only else "failed to download"
    print(f"Some repos {verb}:")
    for repo_id, err in failures:
        print(f"  - {repo_id}: {err[:160]}")
    if verify_only:
        print("\nRe-run without VERIFY=1 to fetch what is missing.")
    else:
        print("\nIf a gated repo failed with 401/403, accept its license on huggingface.co")
        print("with the same account as HF_TOKEN, then re-run.")
    raise SystemExit(1)

print("All checkpoints resolved from:", os.environ["HF_HUB_CACHE"])
if verify_only:
    print("The cache is complete — a job with HF_HUB_OFFLINE=1 will not need the network.")
else:
    print("The training jobs read this same cache, so they will start without downloading.")
    print("Confirm with:  VERIFY=1 bash prefetch_models.sh")
print("\nIMPORTANT: do NOT run the jobs with OCPQ_CLEAR_HF_CACHE=1 — it deletes this cache.")
print("IMPORTANT: leave TRANSFORMERS_CACHE unset; setting it hides this cache from transformers.")
PY
