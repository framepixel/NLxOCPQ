#!/usr/bin/env bash
# Build the evaluation conda environment: a clone of the fine-tuning env plus the two packages
# the eval path needs and training does not.
#
# Why a clone rather than installing into nlxocpq-train-v5 directly: `pip install scikit-learn`
# pulls numpy and scipy constraints, and if the resolver decides to move numpy underneath torch
# and unsloth, the environment that produced the recorded adapters has silently changed. Cloning
# keeps the training env byte-identical to what `requirements.unsloth-train-v5.txt` pins, while
# the eval env still loads adapters on the same torch/transformers/peft build that wrote them.
#
# Run on a login node (needs network), not in a batch job:
#     bash make_eval_env.sh
#
# The script refuses to finish quietly if the install moved any package that training depends on
# — a silently upgraded numpy is exactly the failure this separation exists to prevent.
set -euo pipefail

SRC_ENV="${OCPQ_TRAIN_ENV:-nlxocpq-train-v5}"
DST_ENV="${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
# Only these may be added. Anything else that changes version is a collateral upgrade.
NEW_PACKAGES=(scikit-learn openai)

source "$HOME/miniforge3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$SRC_ENV"; then
  echo "FATAL: source environment '$SRC_ENV' not found." >&2
  exit 1
fi

echo "== Snapshotting $SRC_ENV"
conda activate "$SRC_ENV"
BEFORE="$(mktemp)"
pip freeze --disable-pip-version-check > "$BEFORE"
echo "   $(wc -l < "$BEFORE") packages pinned"
conda deactivate

if conda env list | awk '{print $1}' | grep -qx "$DST_ENV"; then
  if [ "${OCPQ_FORCE_RECREATE:-0}" = "1" ]; then
    echo "== Removing existing $DST_ENV (OCPQ_FORCE_RECREATE=1)"
    conda env remove -n "$DST_ENV" -y
  else
    echo "== $DST_ENV already exists; skipping clone."
    echo "   Set OCPQ_FORCE_RECREATE=1 to rebuild it from scratch."
  fi
fi

if ! conda env list | awk '{print $1}' | grep -qx "$DST_ENV"; then
  echo "== Cloning $SRC_ENV -> $DST_ENV (conda hardlinks what it can; pip packages are copied)"
  conda create --name "$DST_ENV" --clone "$SRC_ENV" -y
fi

echo "== Installing ${NEW_PACKAGES[*]} into $DST_ENV"
conda activate "$DST_ENV"
# Show what the resolver intends before it acts. A dry run that wants to touch torch, numpy or
# transformers is the signal to stop and pin explicitly instead.
pip install --disable-pip-version-check --dry-run "${NEW_PACKAGES[@]}" 2>&1 | tail -20 || true
pip install --disable-pip-version-check "${NEW_PACKAGES[@]}"

echo "== Verifying the eval imports"
python - <<'PY'
import importlib, sys
required = ("torch", "transformers", "peft", "safetensors", "huggingface_hub",
            "numpy", "sklearn", "openai", "pydantic")
missing = [m for m in required if importlib.util.find_spec(m) is None]
if missing:
    print(f"FATAL: still missing {missing}", file=sys.stderr)
    sys.exit(1)
import torch, transformers
print(f"   torch {torch.__version__} | transformers {transformers.__version__}")
print("   all eval imports resolve")
PY

echo "== Checking nothing else moved"
AFTER="$(mktemp)"
pip freeze --disable-pip-version-check > "$AFTER"
python - "$BEFORE" "$AFTER" "${NEW_PACKAGES[@]}" <<'PY'
import sys

def parse(path):
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" in line:
            name, _, ver = line.partition("==")
            out[name.strip().lower().replace("_", "-")] = ver.strip()
    return out

before, after = parse(sys.argv[1]), parse(sys.argv[2])
allowed = {p.lower().replace("_", "-") for p in sys.argv[3:]}

changed = {n: (before[n], after[n]) for n in before if n in after and before[n] != after[n]}
removed = sorted(set(before) - set(after))
added = sorted(set(after) - set(before))

print(f"   added   ({len(added)}): {', '.join(added) or 'none'}")
if changed:
    print(f"   CHANGED ({len(changed)}):")
    for n, (b, a) in sorted(changed.items()):
        print(f"     {n}: {b} -> {a}")
if removed:
    print(f"   REMOVED ({len(removed)}): {', '.join(removed)}")

# Additions are expected: the two packages plus their genuinely new transitive dependencies
# (scipy, joblib, threadpoolctl, httpx and friends). Version *changes* to packages training
# already had are not — that is the drift this environment exists to avoid.
problems = []
if changed:
    problems.append(f"{len(changed)} package(s) changed version")
if removed:
    problems.append(f"{len(removed)} package(s) removed")
if problems:
    print("\nFATAL: " + "; ".join(problems), file=sys.stderr)
    print("The eval env no longer matches the training stack. Pin the offending package to its "
          "training version and re-run, or fall back to installing nothing and guarding the "
          "imports in LLM.py instead.", file=sys.stderr)
    sys.exit(1)
print("   no pre-existing package changed version — eval stack matches training")
PY

pip freeze --disable-pip-version-check > requirements.eval-v5.txt
echo "== Wrote requirements.eval-v5.txt ($(wc -l < requirements.eval-v5.txt) packages)"

ENV_PATH="$(python -c 'import sys, pathlib; print(pathlib.Path(sys.prefix))')"
echo "== Disk: $(du -sh "$ENV_PATH" 2>/dev/null | cut -f1) at $ENV_PATH"
echo
echo "Done. The eval scripts default to OCPQ_EVAL_ENV=$DST_ENV."
