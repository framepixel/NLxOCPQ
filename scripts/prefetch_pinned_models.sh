#!/usr/bin/env bash
# Pre-download the agentic arm's pinned models into their cache on $HPCWORK (notes 43.16, 43.18).
#
# Which models: every entry of the agentic configs that pins a hub commit (`revision`) -- evaluation
# 1's configs/eval_config.agentic.json (Glimmer, Qwen3.8, Gemma 4) and evaluation 2's
# configs/eval_config.unleashed.json (the same three plus Ministral-3-14B-Reasoning) -- or only the
# labels given as arguments. A repo at one commit is fetched once, whichever config names it. Repo
# id and commit are read from the configs the jobs load, so the download and the run cannot disagree
# about which weights are meant.
#
# Why $HPCWORK: each is 55-63 GB in bf16, and $HOME is capped at 150 GB with ~95 GB in use
# (2026-09-28). slurm/model_runtime.sh points these indices at the same cache, and the jobs run with
# HF_HUB_OFFLINE=1, so this must complete first.
#
# Usage (login node, network needed):
#     bash scripts/prefetch_pinned_models.sh                                   # all pinned models
#     bash scripts/prefetch_pinned_models.sh qwen3.8-27b-base gemma4-31b-base  # just these
#     VERIFY=1 bash scripts/prefetch_pinned_models.sh [labels]                 # cache only, download nothing
set -euo pipefail

if [ -z "${HPCWORK:-}" ]; then
  echo "FATAL: \$HPCWORK is unset; the pinned models' cache lives there (slurm/model_runtime.sh)." >&2
  exit 1
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS=("$SCRIPT_DIR/../configs/eval_config.agentic.json" "$SCRIPT_DIR/../configs/eval_config.unleashed.json")

export HF_HOME="$HPCWORK/nlxocpq-hf"
export HF_HUB_CACHE="$HF_HOME/hub"
unset TRANSFORMERS_CACHE
mkdir -p "$HF_HUB_CACHE"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
export HF_HUB_DOWNLOAD_RETRIES="${HF_HUB_DOWNLOAD_RETRIES:-5}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
[ -f "$HOME/.hf_token" ] && export HF_TOKEN="$(cat "$HOME/.hf_token")"

echo "== Cache: $HF_HUB_CACHE"
df -h "$HPCWORK" 2>/dev/null | tail -1 | awk '{print "== $HPCWORK filesystem: " $4 " free of " $2}' || true

source "$HOME/miniforge3/etc/profile.d/conda.sh"
# Any env with huggingface_hub works; this only downloads.
conda activate "${OCPQ_TRAIN_ENV:-nlxocpq-train-v5}"

python - "${#CONFIGS[@]}" "${CONFIGS[@]}" "$@" <<'PY'
import json
import os
import sys

from huggingface_hub import snapshot_download

n_configs = int(sys.argv[1])
config_paths, wanted = sys.argv[2:2 + n_configs], set(sys.argv[2 + n_configs:])
pinned, seen = [], {}
for path in config_paths:
    for m in json.load(open(path, encoding="utf-8"))["models"]:
        if not m.get("revision"):
            continue
        key = (m["name"], m["revision"])
        if key in seen:                       # the same weights under another config's label
            seen[key].setdefault("aliases", []).append(m["label"])
            continue
        seen[key] = dict(m)
        pinned.append(seen[key])
labels_of = lambda m: {m["label"], *m.get("aliases", [])}
unknown = wanted - set().union(*(labels_of(m) for m in pinned))
if unknown:
    raise SystemExit(f"FATAL: no pinned model labelled {sorted(unknown)}; "
                     f"pinned: {[m['label'] for m in pinned]}")
entries = [m for m in pinned if not wanted or labels_of(m) & wanted]
verify_only = os.environ.get("VERIFY", "0").strip().lower() in {"1", "true", "yes", "on"}
failures = []
for entry in entries:
    print(f"\n== {entry['label']}: {entry['name']} @ {entry['revision'][:12]}"
          + ("  (VERIFY: cache only)" if verify_only else ""), flush=True)
    try:
        path = snapshot_download(repo_id=entry["name"], revision=entry["revision"],
                                 token=os.environ.get("HF_TOKEN") or None,
                                 local_files_only=verify_only)
    except Exception as exc:  # noqa: BLE001 - reported per model, the others still run
        print(f"   FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        failures.append(entry["label"])
        continue
    files = [os.path.join(root, f) for root, _dirs, names in os.walk(path) for f in names]
    total = sum(os.path.getsize(f) for f in files)
    print(f"   OK  {path}\n   {len(files)} files, {total / 1024**3:.1f} GiB")
if failures:
    raise SystemExit(f"\nFAILED: {failures}")
print("\nThe cache is complete -- a job with HF_HUB_OFFLINE=1 will not need the network."
      if verify_only else "\nConfirm with:  VERIFY=1 bash scripts/prefetch_pinned_models.sh")
PY
