#!/usr/bin/env bash
# Pre-download Muse Glimmer-30B (the second agentic arm, notes 43.16) into its cache on $HPCWORK.
#
# Why a separate script and cache: the bf16 checkpoint is ~60 GB (13 files, two weight shards) and
# $HOME is capped at 150 GB with ~108 GB in use (2026-09-27), so it cannot share
# ~/.cache/huggingface with the other models. slurm/model_runtime.sh points model index 8 at the
# same $HPCWORK cache, and the jobs run with HF_HUB_OFFLINE=1, so this must complete first.
#
# The repo id and the pinned commit are read from configs/eval_config.agentic.json -- the entry the
# jobs load -- so the download and the run cannot disagree about which weights are meant.
#
# Usage (login node, network needed):
#     bash scripts/prefetch_glimmer.sh              # download (~60 GB)
#     VERIFY=1 bash scripts/prefetch_glimmer.sh     # resolve from the cache only, download nothing
set -euo pipefail

if [ -z "${HPCWORK:-}" ]; then
  echo "FATAL: \$HPCWORK is unset; Glimmer's cache lives there (slurm/model_runtime.sh)." >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$SCRIPT_DIR/../configs/eval_config.agentic.json"

export HF_HOME="$HPCWORK/nlxocpq-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HUB_DOWNLOAD_THREADS="${HF_HUB_DOWNLOAD_THREADS:-8}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

echo "== Cache: $HF_HUB_CACHE"
df -h "$HPCWORK" 2>/dev/null | tail -1 | awk '{print "== $HPCWORK filesystem: " $4 " free of " $2}' || true

source "$HOME/miniforge3/etc/profile.d/conda.sh"
# Any env with huggingface_hub works; this only downloads.
conda activate "${OCPQ_TRAIN_ENV:-nlxocpq-train-v5}"

python - "$CONFIG" <<'PY'
import json
import os
import sys

from huggingface_hub import snapshot_download

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
entry = next((m for m in cfg["models"] if m.get("label") == "muse-glimmer-30b-base"), None)
if entry is None or not entry.get("revision"):
    raise SystemExit("FATAL: no muse-glimmer-30b-base entry with a pinned revision in the config")
verify_only = os.environ.get("VERIFY", "0").strip().lower() in {"1", "true", "yes", "on"}
print(f"== {entry['name']} @ {entry['revision'][:12]}"
      + ("  (VERIFY: cache only)" if verify_only else ""), flush=True)
path = snapshot_download(repo_id=entry["name"], revision=entry["revision"],
                         token=os.environ.get("HF_TOKEN") or None,
                         local_files_only=verify_only)
files = [os.path.join(root, f) for root, _dirs, names in os.walk(path) for f in names]
total = sum(os.path.getsize(f) for f in files)
print(f"   OK  {path}")
print(f"   {len(files)} files, {total / 1024**3:.1f} GiB")
if verify_only:
    print("The cache is complete -- a job with HF_HUB_OFFLINE=1 will not need the network.")
else:
    print("Confirm with:  VERIFY=1 bash scripts/prefetch_glimmer.sh")
PY
