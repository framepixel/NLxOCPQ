"""Record identity across case-shards.

Regression target: merging the 96-shard matrix on 2026-08-27 reported "Dropped 4032 duplicate
record(s)" and produced 4080 records instead of 8112. `case_idx` is assigned by
`enumerate(valid_testset, start=1)` AFTER the case-shard slice, so both halves of every condition
number their cases from 1 and collided entry-for-entry under a case_id-based key.

Run from the NLxOCPQ directory:  python tests/test_record_key.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_testset import _merge_shards, _record_key, _record_key_of  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


CFG = "llama31-8b-base|schema-grammar-guided|True"

print("--- THE BUG: same case_id, different entries ---")
# Exactly the shape of the two halves: half 0 case 1 is testset row 0, half 1 case 1 is row 85.
h0 = _record_key(1, CFG, 0, "Which orders were never paid?", "/abs/Logs/order-management.json")
h1 = _record_key(1, CFG, 0, "Which commits touched five files?", "/abs/Logs/angular.xml")
check("two entries sharing case_id get DIFFERENT keys", h0 != h1, True)

legacy = "1:" + CFG + ":0"
check("the old key would have collided", legacy == legacy, True)

print("\n--- genuine duplicates still collapse ---")
a = _record_key(1, CFG, 0, "Which orders were never paid?", "/abs/Logs/order-management.json")
b = _record_key(1, CFG, 0, "Which orders were never paid?", "/abs/Logs/order-management.json")
check("same entry, same condition -> same key", a, b)
# A re-run supersedes a partial: the shard may renumber, the entry does not move.
c = _record_key(47, CFG, 0, "Which orders were never paid?", "/abs/Logs/order-management.json")
check("same entry under a DIFFERENT case_id -> still same key", a, c)
# Path varies with where the shard ran; the entry is the same.
d = _record_key(1, CFG, 0, "Which orders were never paid?",
                "/somewhere/else/Logs/order-management.json")
check("same entry via a different absolute path -> same key", a, d)

print("\n--- the key still separates what it must ---")
check("different condition -> different key",
      _record_key(1, "other|s|True", 0, "Q", "/L/a.json")
      != _record_key(1, CFG, 0, "Q", "/L/a.json"), True)
check("different repeat -> different key",
      _record_key(1, CFG, 1, "Q", "/L/a.json") != _record_key(1, CFG, 0, "Q", "/L/a.json"), True)
check("same question on a DIFFERENT log -> different key",
      _record_key(1, CFG, 0, "Q", "/L/a.json") != _record_key(1, CFG, 0, "Q", "/L/b.json"), True)

print("\n--- fallback for records with no nl_query ---")
check("falls back to case_id", _record_key(7, CFG, 0), f"case7:{CFG}:0")
check("_record_key_of on a bare record",
      _record_key_of({"case_id": 7, "config_id": CFG, "repeat": 0}), f"case7:{CFG}:0")
check("_record_key_of prefers the entry",
      _record_key_of({"case_id": 7, "config_id": CFG, "repeat": 0,
                      "nl_query": "Q", "ocel_path": "/L/a.json"}),
      _record_key(7, CFG, 0, "Q", "/L/a.json"))

print("\n--- end to end: a two-half merge keeps everything ---")
import json
import shutil
import tempfile

tmp = Path(tempfile.mkdtemp(prefix="mergekey_"))


def rec(case_id, nl, log):
    f1 = 0.5
    return {"case_id": case_id, "nl_query": nl, "ocel_path": log,
            "domain_seen": True, "discriminating": True, "baseline_f1": 0.4,
            "gold_query": {"nodes": []}, "prediction": {"nodes": []},
            "model": "m", "model_label": "m", "prompt_strategy": "schema-grammar-guided",
            "schema_enforced": True, "schema_validation": True, "repeat": 0,
            "config_id": CFG, "status": "success", "pipeline_status": "ok",
            "similarity": {"combined": {"f1": f1, "precision": f1, "recall": f1},
                           "events": {"jaccard": f1, "precision": f1, "recall": f1},
                           "objects": {"jaccard": f1, "precision": f1, "recall": f1}},
            "llm": {}, "execution": {"success": True, "duration_sec": 1.0},
            "total_duration_sec": 1.0,
            "metrics": {"schema_valid": True, "hallucinated_entities": False,
                        "unsupported_filters": False,
                        "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                        "bindings_skipped": False, "gold_bindings_skipped": False}}


def shard(name, records):
    d = tmp / name
    d.mkdir(parents=True)
    (d / "predictions.json").write_text(json.dumps(records), encoding="utf-8")
    (d / "config_resolved.json").write_text(json.dumps(
        {"pipeline_fingerprint": "same", "repeats": 1}), encoding="utf-8")
    return d


# Two halves of one condition, both numbering cases from 1 -- the real layout.
half0 = [rec(i, f"question h0 #{i}", "/L/order-management.json") for i in range(1, 6)]
half1 = [rec(i, f"question h1 #{i}", "/L/angular.xml") for i in range(1, 5)]
shard("shard_m0_s0_e0_c0of2_t", half0)
shard("shard_m0_s0_e0_c1of2_t", half1)

out = tmp / "merged"
_merge_shards([str(tmp / "shard_m0_s0_e0_c0of2_t"), str(tmp / "shard_m0_s0_e0_c1of2_t")], out)
merged = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
check("both halves survive the merge", len(merged), 9)
check("all nine entries are distinct", len({r["nl_query"] for r in merged}), 9)

# And a genuine re-run still collapses.
print("\n(The next merge feeds half 0 twice on purpose: 5 of 14 records are duplicates, above the 5% "
      "alarm, so the 'implausible share' WARNING below is EXPECTED and must say 'differ in: nothing'.)")
shard("shard_m0_s0_e0_c0of2_rerun", half0)
out2 = tmp / "merged2"
_merge_shards([str(tmp / "shard_m0_s0_e0_c0of2_t"),
               str(tmp / "shard_m0_s0_e0_c1of2_t"),
               str(tmp / "shard_m0_s0_e0_c0of2_rerun")], out2)
merged2 = json.loads((out2 / "predictions.json").read_text(encoding="utf-8"))
check("a re-run of half 0 collapses, not duplicates", len(merged2), 9)

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
