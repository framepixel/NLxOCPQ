#!/usr/bin/env python3
"""Assert that a smoke evaluation run is healthy. Exits non-zero if it is not.

The smoke job exiting 0 only means nothing raised. It does not mean every condition produced
usable output, that the result store was written, or that the metric keys are populated -- and
a matrix that silently loses one of six models to a load failure is worse than one that crashes.
This is the part that turns the run into a test.

Reports every failed check rather than stopping at the first, so one submission diagnoses
everything wrong at once instead of one problem per 1-2 hour cycle.

    python smoke_check.py eval_runs/smoke/<run_id>
"""
import argparse
import json
import sys
from pathlib import Path

EXPECTED_CONDITIONS = 36
REQUIRED_ARTIFACTS = (
    "predictions.json",
    "summary.json",
    "config_resolved.json",
    "bb_breakdown.json",
    "divergence_report.json",
)
# Statuses that mean the pipeline produced a comparable answer. `no_result` is not an error in
# itself -- a valid query can legitimately bind nothing -- but a condition producing only these
# has not demonstrated it works.
HEALTHY_STATUSES = {"success", "no_result"}


class Checker:
    def __init__(self):
        self.failures = []
        self.warnings = []

    def check(self, ok, name, detail=""):
        line = f"  {name:<58} {'PASS' if ok else 'FAIL'}"
        if detail:
            line += f"  {detail}"
        print(line)
        if not ok:
            self.failures.append(name)
        return ok

    def warn(self, ok, name, detail=""):
        if not ok:
            print(f"  {name:<58} WARN  {detail}")
            self.warnings.append(name)
        return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Smoke run directory (eval_runs/smoke/<run_id>)")
    ap.add_argument("--expected-conditions", type=int, default=EXPECTED_CONDITIONS)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    c = Checker()
    print(f"Smoke check: {run_dir}\n")

    print("artifacts")
    for name in REQUIRED_ARTIFACTS:
        c.check((run_dir / name).is_file(), f"{name} written")
    if c.failures:
        # Everything below reads these files; continuing would just cascade.
        print("\nRESULT: FAIL (missing artifacts)")
        return 1

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = json.loads((run_dir / "predictions.json").read_text(encoding="utf-8"))
    resolved = json.loads((run_dir / "config_resolved.json").read_text(encoding="utf-8"))
    limit = int((resolved.get("config") or {}).get("limit") or 0)

    print("\nconditions")
    by_config = summary.get("by_config") or {}
    c.check(len(by_config) == args.expected_conditions,
            f"all {args.expected_conditions} conditions present", f"got {len(by_config)}")
    if limit:
        wrong = {cid: b["metrics"].get("n") for cid, b in by_config.items()
                 if (b.get("metrics") or {}).get("n") != limit}
        # `n` is not in the by_config metrics block; fall back to counting records per config.
        counts = {}
        for r in records:
            counts[r.get("config_id")] = counts.get(r.get("config_id"), 0) + 1
        wrong = {cid: n for cid, n in counts.items() if n != limit}
        c.check(not wrong, f"every condition ran all {limit} cases", str(dict(list(wrong.items())[:3])))
        c.check(len(counts) == args.expected_conditions,
                "every condition produced records", f"{len(counts)} with records")

    print("\nper-condition health")
    dead = []
    for cid, n in sorted(
            ((cid, sum(1 for r in records
                       if r.get("config_id") == cid and r.get("status") in HEALTHY_STATUSES))
             for cid in {r.get("config_id") for r in records}), key=lambda x: x[0]):
        if n == 0:
            dead.append(cid)
    c.check(not dead, "no condition failed every case", f"dead: {dead[:4]}")

    statuses = {}
    for r in records:
        statuses[r.get("status")] = statuses.get(r.get("status"), 0) + 1
    print(f"    status distribution: {dict(sorted(statuses.items()))}")
    healthy = sum(v for k, v in statuses.items() if k in HEALTHY_STATUSES)
    c.warn(healthy >= 0.5 * len(records), "at least half of all runs healthy",
           f"{healthy}/{len(records)}")

    print("\nmodels")
    labels = sorted({r.get("model_label") for r in records if r.get("model_label")})
    c.check(len(labels) == 6, "all six models produced records", f"{labels}")

    print("\nstratum labels")
    unlabelled = [r.get("case_id") for r in records if r.get("discriminating") is None]
    c.check(not unlabelled, "no record missing its stratum label",
            f"{len(unlabelled)} unlabelled")
    no_floor = [r.get("case_id") for r in records if r.get("baseline_f1") is None]
    c.check(not no_floor, "no record missing its trivial floor", f"{len(no_floor)} without")

    print("\nresult store")
    store = summary.get("result_store") or {}
    c.check(bool(store) and store.get("blobs", 0) > 0, "result store written",
            f"{store.get('blobs', 0)} blobs, {store.get('bytes', 0) / 1048576:.2f} MB")
    c.check((run_dir / "result_store").is_dir(), "result_store/ directory exists")

    print("\nmetric keys")
    bad = []
    for cid, block in by_config.items():
        m = block.get("metrics") or {}
        for key in ("result_set_f1_strict", "structural_f1", "mean_baseline_f1",
                    "skill_score", "below_floor_rate", "skill_n", "skill_undefined"):
            if key not in m:
                bad.append(f"{cid}:{key} missing")
        f1 = m.get("result_set_f1_strict")
        if f1 is not None and not (0.0 <= f1 <= 1.0):
            bad.append(f"{cid}:result_set_f1_strict={f1}")
        sk = m.get("skill_score")
        if sk is not None and not (-1.0 <= sk <= 1.0):
            bad.append(f"{cid}:skill_score={sk}")
        n_here = sum(1 for r in records if r.get("config_id") == cid)
        if m.get("skill_n") is not None and m.get("skill_undefined") is not None:
            if m["skill_n"] + m["skill_undefined"] != n_here:
                bad.append(f"{cid}:skill_n+undefined={m['skill_n']}+{m['skill_undefined']}!={n_here}")
    c.check(not bad, "metric keys present and in range", str(bad[:3]))

    print("\nstrata")
    strat = summary.get("by_stratum") or {}
    c.check("discriminating" in strat and "non_discriminating" in strat,
            "by_stratum block present")
    c.check(strat.get("unlabelled", 0) == 0, "by_stratum reports no unlabelled records",
            str(strat.get("unlabelled")))

    print("\nschema completeness")
    # A dropped schema section contradicts the "every model always receives the complete schema"
    # design, and is exactly what the raised token budget is meant to prevent.
    omissions = [r for r in records if (r.get("metrics") or {}).get("schema_omissions")]
    c.check(not omissions, "no schema sections dropped for any log",
            f"{len(omissions)} record(s) with omissions")

    print("\ngold execution")
    sources = {}
    for r in records:
        s = (r.get("ground_truth") or {}).get("source")
        sources[s] = sources.get(s, 0) + 1
    print(f"    ground-truth sources: {dict(sorted(sources.items(), key=lambda x: str(x[0])))}")
    c.warn(None not in sources, "every record records how gold was obtained",
           f"{sources.get(None, 0)} without a source")

    print()
    if c.warnings:
        print(f"{len(c.warnings)} warning(s): {c.warnings}")
    if c.failures:
        print(f"RESULT: FAIL -> {c.failures}")
        return 1
    print("RESULT: PASS — the pipeline is ready for the full matrix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
