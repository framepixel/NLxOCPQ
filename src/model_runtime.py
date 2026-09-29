#!/usr/bin/env python3
"""Print the runtime a config's model runs in: `<conda env>|<HF_HOME>`.

Evaluation 2's submit scripts (notes 44) read each model's `runtime` from its config entry
BEFORE any conda environment is active, so this uses the standard library only and runs on the
cluster's module Python. The choice follows the config the job loads, not an exported variable,
for the reason slurm/model_runtime.sh gives (sbatch has dropped exported variables before).

    python3 src/model_runtime.py configs/eval_config.unleashed.json 1
    -> nlxocpq-eval-glimmer|/hpcwork/<user>/nlxocpq-hf

`hf_cache` is "hpcwork" ($HPCWORK/nlxocpq-hf, the pinned models' cache) or "home"
(~/.cache/huggingface). Exits non-zero, with the reason on stderr, on any problem.
"""
import json
import os
import sys


def runtime(config_path, model_idx):
    cfg = json.load(open(config_path, encoding="utf-8"))
    models = cfg.get("models") or []
    if not 0 <= model_idx < len(models):
        raise SystemExit(f"model index {model_idx} is not in {config_path} (0-{len(models) - 1})")
    entry = models[model_idx]
    spec = entry.get("runtime")
    if not isinstance(spec, dict) or not spec.get("conda_env") or not spec.get("hf_cache"):
        raise SystemExit(f"{entry.get('label')}: no runtime {{conda_env, hf_cache}} in the config")
    if spec["hf_cache"] == "hpcwork":
        base = os.environ.get("HPCWORK")
        if not base:
            raise SystemExit(f"{entry.get('label')} reads its weights from $HPCWORK, which is unset")
        hf_home = base.rstrip("/") + "/nlxocpq-hf"      # a cluster path: always "/"
    elif spec["hf_cache"] == "home":
        hf_home = os.path.expanduser("~").rstrip("/") + "/.cache/huggingface"
    else:
        raise SystemExit(f"{entry.get('label')}: hf_cache must be 'hpcwork' or 'home'")
    return f"{spec['conda_env']}|{hf_home}"


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: model_runtime.py <config.json> <model index>")
    print(runtime(sys.argv[1], int(sys.argv[2])))
