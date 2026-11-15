"""`compute_accounting.py` must count each shard once and keep the two kinds of hours apart.

Pinned here: rescored `_recomputed` copies and excluded names are never counted (they ran on CPU
or belong to another group), a group matching nothing is an error rather than zero hours, and the
case and shard hours come from different sources.

Run from the repo root:  python tests/test_compute_accounting.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import compute_accounting as ca  # noqa: E402
import paths  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


with tempfile.TemporaryDirectory() as root:
    base = Path(root)
    for name, elapsed, durations in [("shard_a", 3600, [10.0, 20.0]),
                                     ("shard_a_recomputed", 999999, [1e6]),
                                     ("shard_b_pFBEXLR", 7200, [30.0])]:
        shard = base / "runs" / name
        shard.mkdir(parents=True)
        (shard / "checkpoint.json").write_text(json.dumps({"elapsed_sec": elapsed}))
        (shard / "predictions.json").write_text(json.dumps(
            [{"total_duration_sec": d, "model_label": "m"} for d in durations]))
    original = paths.REPO_ROOT
    ca.paths.REPO_ROOT = base
    try:
        group = {"id": "g", "label": "G", "globs": ["runs/shard_*"], "exclude": ["pFBEXLR"]}
        check("rescored copies and exclusions are skipped",
              [p.name for p in ca.shard_dirs(group)], ["shard_a"])
        row = ca.account(group)
        check("shard hours come from the checkpoint", row["shard_hours"], 1.0)
        check("case hours come from the records", round(row["case_hours"] * 3600, 6), 30.0)
        try:
            ca.shard_dirs({"id": "none", "globs": ["runs/nothing_*"]})
            check("a group matching nothing raises", False, True)
        except SystemExit:
            check("a group matching nothing raises", True, True)
    finally:
        ca.paths.REPO_ROOT = original

check("percentiles of nothing are undefined", ca.percentiles([])["median"], None)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
