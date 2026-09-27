#!/usr/bin/env bash
# Build the evaluation conda environment: a clone of the fine-tuning env plus the two packages
# the eval path needs and training does not.
#
# Why a clone rather than installing into nlxocpq-train-v5 directly: `pip install scikit-learn`
# pulls numpy and scipy constraints, and if the resolver decides to move numpy underneath torch
# and unsloth, the environment that produced the recorded adapters has silently changed. Cloning
# keeps the training env byte-identical to what `requirements/requirements.unsloth-train-v5.txt` pins, while
# the eval env still loads adapters on the same torch/transformers/peft build that wrote them.
#
# Run on a login node (needs network), not in a batch job:
#     bash scripts/make_eval_env.sh
#
# The script refuses to finish quietly if the install moved any package that training depends on
# — a silently upgraded numpy is exactly the failure this separation exists to prevent.
set -euo pipefail

# OCPQ_ENV_VARIANT=glimmer builds the SECOND arm's environment (notes 43.16) instead:
#     OCPQ_ENV_VARIANT=glimmer bash scripts/make_eval_env.sh
# Muse Glimmer's model type (`muse_glimmer`) exists only from transformers 5.15, and the eval env
# is on 5.5.0 -- the stack every recorded run used. So Glimmer gets a clone of the eval env with
# ONE intended change, transformers, and the check below still refuses any other move (torch
# above all). The completed Ministral arm's environment is not touched.
VARIANT="${OCPQ_ENV_VARIANT:-default}"
case "$VARIANT" in
  default)
    SRC_ENV="${OCPQ_TRAIN_ENV:-nlxocpq-train-v5}"
    DST_ENV="${OCPQ_EVAL_ENV:-nlxocpq-eval-v5}"
    # Only these may be added. Anything else that changes version is a collateral upgrade.
    # xgrammar added 2026-09-02 for constrained decoding. It pulls apache-tvm-ffi and
    # nothing else; verified locally that torch, transformers, pydantic and peft all stay
    # put. The check below is what proves that on the cluster too -- if it reports a moved
    # package, stop rather than working around it: the adapters were written by that stack.
    NEW_PACKAGES=(scikit-learn openai xgrammar)
    MAY_CHANGE=""
    REQUIREMENTS_OUT="requirements.eval-v5.txt" ;;
  glimmer)
    SRC_ENV="nlxocpq-eval-v5"
    DST_ENV="nlxocpq-eval-glimmer"
    NEW_PACKAGES=("transformers==5.17.0")
    # transformers moves these with it; they are part of the same intended change.
    MAY_CHANGE="transformers tokenizers huggingface-hub safetensors"
    REQUIREMENTS_OUT="requirements.eval-glimmer.txt" ;;
  *)
    echo "FATAL: OCPQ_ENV_VARIANT must be 'default' or 'glimmer' (got '$VARIANT')." >&2
    exit 1 ;;
esac

# Resolve relative to this script rather than the caller's cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS_CHECK="$SCRIPT_DIR/../src/check_eval_deps.py"
if [ ! -f "$DEPS_CHECK" ]; then
  echo "FATAL: $DEPS_CHECK not found." >&2
  echo "It is a companion file of this script and must be copied to the cluster too." >&2
  exit 1
fi

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
# Uses the same checker the jobs run, so a green build here means a green pre-flight there.
python "$DEPS_CHECK" --verbose
if [ "$VARIANT" = "glimmer" ]; then
  # The reason this environment exists: can this transformers build Glimmer's architecture? A
  # config-only check -- it needs neither the weights nor the network.
  python -c "from transformers import AutoConfig; AutoConfig.for_model('muse_glimmer'); print('   transformers knows muse_glimmer')"
fi

echo "== Checking nothing else moved"
AFTER="$(mktemp)"
pip freeze --disable-pip-version-check > "$AFTER"
OCPQ_MAY_CHANGE="$MAY_CHANGE" python - "$BEFORE" "$AFTER" "${NEW_PACKAGES[@]}" <<'PY'
import os
import re
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

def norm(name):
    return re.split(r"[=<>!~ ]", name, 1)[0].strip().lower().replace("_", "-")

before, after = parse(sys.argv[1]), parse(sys.argv[2])
allowed = {norm(p) for p in sys.argv[3:]}
# The glimmer variant's ONE intended change and what moves with it. Empty for the default build.
may_change = {norm(p) for p in os.environ.get("OCPQ_MAY_CHANGE", "").split()}

intended = {n: (before[n], after[n]) for n in before
            if n in after and before[n] != after[n] and n in may_change}
if intended:
    print(f"   intended ({len(intended)}):")
    for n, (b, a) in sorted(intended.items()):
        print(f"     {n}: {b} -> {a}")
changed = {n: (before[n], after[n]) for n in before
           if n in after and before[n] != after[n] and n not in may_change}
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
print("   no other pre-existing package changed version — the stack matches its source env")
PY

pip freeze --disable-pip-version-check > "$SCRIPT_DIR/../requirements/$REQUIREMENTS_OUT"
echo "== Wrote requirements/$REQUIREMENTS_OUT ($(wc -l < "$SCRIPT_DIR/../requirements/$REQUIREMENTS_OUT") packages)"

ENV_PATH="$(python -c 'import sys, pathlib; print(pathlib.Path(sys.prefix))')"
echo "== Disk: $(du -sh "$ENV_PATH" 2>/dev/null | cut -f1) at $ENV_PATH"
echo
if [ "$VARIANT" = "glimmer" ]; then
  echo "Done. The agentic scripts run model index 8 in $DST_ENV (slurm/model_runtime.sh)."
else
  echo "Done. The eval scripts default to OCPQ_EVAL_ENV=$DST_ENV."
fi
