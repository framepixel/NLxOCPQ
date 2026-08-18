#!/usr/bin/env python3
"""Verify submit_eval_array.sh's task->condition arithmetic covers the matrix exactly.

The array splits 36 conditions across 72 tasks with index arithmetic written in bash:

    MODEL_IDX    = TASK / 12
    STRATEGY_IDX = (TASK % 12) / 4
    SCHEMA_IDX   = (TASK % 4) / 2
    CASE_HALF    = TASK % 2

An off-by-one there does not crash: it silently evaluates one condition twice and another not at
all, and the gap only surfaces at merge time after ~40 GPU-hours. The smoke test runs as a single
job and so does not exercise sharding, which leaves this as the one part of the launch path with
no coverage. Pure Python, no GPU, runs anywhere:

    python check_shard_map.py
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    from expand_models import expand_models
except Exception:  # noqa: BLE001 - reported by the check itself
    expand_models = None

TOTAL_TASKS = 72
CASE_HALVES = 2


def shard_map(task):
    """Reproduce the bash arithmetic exactly, including its integer division."""
    return {
        "model_idx": task // 12,
        "strategy_idx": (task % 12) // 4,
        "schema_idx": (task % 4) // 2,
        "case_half": task % CASE_HALVES,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="eval_config.json")
    ap.add_argument("--tasks", type=int, default=TOTAL_TASKS)
    args = ap.parse_args()

    failures = []

    def check(ok, name, detail=""):
        print(f"  {name:<56} {'PASS' if ok else 'FAIL'}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # Fall back to the config sitting next to this script. The default is relative, so
    # `python NLxOCPQ/check_shard_map.py` from $HOME reported "FATAL: eval_config.json not found"
    # -- indistinguishable, moments before launching the matrix, from the config actually being
    # missing. Anchoring on the script's own directory makes the check work from anywhere.
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        beside_script = Path(__file__).resolve().parent / args.config
        if beside_script.is_file():
            cfg_path = beside_script
        else:
            print(f"FATAL: {cfg_path} not found (also looked in {beside_script.parent})")
            return 1
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    if expand_models is None:
        print("FATAL: could not import expand_models")
        return 1
    resolved = expand_models(cfg)
    models = resolved["models"]
    strategies = resolved["prompt_strategies"]
    schemas = resolved["schema_enforced"]
    n_conditions = len(models) * len(strategies) * len(schemas)

    print(f"config: {cfg_path}")
    print(f"  {len(models)} models x {len(strategies)} strategies x {len(schemas)} schema "
          f"settings = {n_conditions} conditions\n")

    print("shape")
    check(args.tasks == n_conditions * CASE_HALVES,
          f"{args.tasks} tasks == {n_conditions} conditions x {CASE_HALVES} halves",
          f"expected {n_conditions * CASE_HALVES}")

    print("\nindex ranges")
    maps = [shard_map(t) for t in range(args.tasks)]
    check(all(0 <= m["model_idx"] < len(models) for m in maps), "model_idx within range")
    check(all(0 <= m["strategy_idx"] < len(strategies) for m in maps),
          "strategy_idx within range")
    check(all(0 <= m["schema_idx"] < len(schemas) for m in maps), "schema_idx within range")
    check(all(0 <= m["case_half"] < CASE_HALVES for m in maps), "case_half within range")

    print("\ncoverage")
    combos = Counter((m["model_idx"], m["strategy_idx"], m["schema_idx"]) for m in maps)
    check(len(combos) == n_conditions, f"all {n_conditions} conditions reached",
          f"got {len(combos)}")
    wrong = {k: v for k, v in combos.items() if v != CASE_HALVES}
    check(not wrong, f"each condition appears exactly {CASE_HALVES}x (once per half)",
          str(dict(list(wrong.items())[:3])))
    per_condition_halves = {}
    for m in maps:
        key = (m["model_idx"], m["strategy_idx"], m["schema_idx"])
        per_condition_halves.setdefault(key, set()).add(m["case_half"])
    check(all(h == {0, 1} for h in per_condition_halves.values()),
          "each condition gets both case halves, not one twice")

    print("\nresolved model order (what MODEL_IDX selects)")
    for i, m in enumerate(models):
        print(f"    {i}  {m['label']:<26} {m['name']}")
    # The bash comment documents base/finetuned alternating in pairs; if that ever stops holding,
    # the comment becomes a trap for whoever next reads the mapping.
    pairs_ok = all(
        ("finetuned" in models[i + 1]["label"]) and ("finetuned" not in models[i]["label"])
        for i in range(0, len(models) - 1, 2))
    check(pairs_ok, "models alternate base, finetuned in pairs (as documented)")

    print()
    if failures:
        print(f"RESULT: FAIL -> {failures}")
        return 1
    print("RESULT: PASS — the array covers every condition exactly once per case half")
    return 0


if __name__ == "__main__":
    sys.exit(main())
