"""The result-export size guard: refuse to parse what would SIGKILL the shard.

Regression target: shard 60 of job 3087762, OOM-killed twice (64 GB, then 122 GB) parsing a
prediction's result export. A SIGKILL cannot be caught, so the size must be checked before the
read rather than the failure handled after it.

Run from the NLxOCPQ directory:  python test_result_size_guard.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_testset import (  # noqa: E402
    _build_summary,
    _classify_run_status,
    _load_result_ids,
    _oversized_counts,
    _result_file_too_large,
)

failures = []
MB = 1024 ** 2
GB = 1024 ** 3


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


tmp = Path(tempfile.mkdtemp(prefix="sizeguard_"))

# A real, parseable ocpq_cli-shaped result, plus a deliberately fat one.
small = tmp / "small.json"
small.write_text(json.dumps({"evaluationResults": [], "bindingsSkipped": False}), encoding="utf-8")
fat = tmp / "fat.json"
fat.write_bytes(b'{"pad": "' + b"x" * (3 * MB) + b'"}')
fat_size = fat.stat().st_size

print("--- _result_file_too_large ---")
check("under the cap -> None", _result_file_too_large(str(fat), 1), None)
check("over the cap -> the size", _result_file_too_large(str(fat), 0.001), fat_size)
check("cap of 0 disables the guard", _result_file_too_large(str(fat), 0), None)
check("cap of None disables the guard", _result_file_too_large(str(fat), None), None)
check("missing file is not 'too large'", _result_file_too_large(str(tmp / "nope"), 0.001), None)
check("no path at all", _result_file_too_large(None, 1), None)

print("\n--- _load_result_ids returns a triple ---")
ids, skipped, oversized = _load_result_ids(str(small), {"nodes": []}, max_gb=0)
check("small file parses, no oversize flag", oversized, None)
ids, skipped, oversized = _load_result_ids(str(fat), {"nodes": []}, max_gb=0.001)
check("oversized file refused", oversized, fat_size)
check("oversized yields no ids (scored 0 strict)", ids, None)
check("oversized yields no truncation flag", skipped, None)
ids, skipped, oversized = _load_result_ids(str(tmp / "nope"), None, max_gb=1)
check("missing file unchanged", (ids, skipped, oversized), (None, None, None))

# The pre-guard behaviour must be reproducible exactly, because 95 of 96 shards ran that way.
ids_off, _, over_off = _load_result_ids(str(fat), {"nodes": []}, max_gb=0)
check("guard off: the fat file is parsed like before", over_off, None)

print("\n--- _classify_run_status ---")
ok_args = ("ok", None, True, {"success": True}, "f.json", None)
check("without the flag it is success", _classify_run_status(*ok_args), "success")
check("with the flag it is result_too_large",
      _classify_run_status(*ok_args, result_too_large=True), "result_too_large")
# Ordering matters: schema failures are decided before the answer is ever read.
check("schema_invalid still wins over the flag",
      _classify_run_status("ok", None, False, {"success": True}, "f.json", None,
                           result_too_large=True), "schema_invalid")
check("a timeout still wins",
      _classify_run_status("timeout", "exceeded the 600s time limit", False, None, None, None,
                           result_too_large=True), "timeout")

print("\n--- counting and reporting ---")


def rec(case_id, model, *, f1, too_large=False, nbytes=None):
    sim = None if f1 is None else {
        "combined": {"f1": f1, "precision": f1, "recall": f1},
        "events": {"jaccard": f1, "precision": f1, "recall": f1},
        "objects": {"jaccard": f1, "precision": f1, "recall": f1}}
    return {
        "case_id": case_id, "nl_query": f"q{case_id}",
        "ocel_path": "x.json", "domain_seen": True, "discriminating": True,
        "baseline_f1": 0.5, "gold_query": {"nodes": []}, "prediction": {"nodes": []},
        "model": model, "model_label": model,
        "prompt_strategy": "schema-grammar-guided", "schema_enforced": True,
        "schema_validation": True, "repeat": 0,
        "config_id": f"{model}|schema-grammar-guided|True",
        "status": "result_too_large" if too_large else ("success" if f1 is not None
                                                        else "schema_invalid"),
        "pipeline_status": "ok", "similarity": sim, "llm": {},
        "execution": {"success": f1 is not None, "duration_sec": 1.0},
        "total_duration_sec": 1.0,
        "metrics": {"schema_valid": True, "hallucinated_entities": False,
                    "unsupported_filters": False,
                    "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                    "bindings_skipped": None if f1 is None else False,
                    "gold_bindings_skipped": False,
                    "result_too_large": too_large, "result_file_bytes": nbytes},
    }


records = []
for m in ("model-a", "model-b"):
    records += [rec(1, m, f1=0.9), rec(2, m, f1=None, too_large=True, nbytes=12 * GB),
                rec(3, m, f1=None), rec(4, m, f1=0.6)]

c = _oversized_counts(records)
check("counted", c["result_too_large_n"], 2)
check("denominator is every record", c["result_too_large_denominator"], 8)
check("rate", c["result_too_large_rate"], 2 / 8)
check("largest size surfaced", c["result_too_large_max_bytes"], 12 * GB)
check("empty input safe", _oversized_counts([])["result_too_large_rate"], None)
check("record with no metrics safe", _oversized_counts([{}])["result_too_large_n"], 0)
check("status alone is enough",
      _oversized_counts([{"status": "result_too_large"}])["result_too_large_n"], 1)

summary = _build_summary(records, 1)
KEYS = ["result_too_large_n", "result_too_large_denominator", "result_too_large_rate"]
for k in KEYS:
    check(f"by_config carries {k}",
          all(k in v["metrics"] for v in summary["by_config"].values()), True)
    check(f"by_stratum carries {k}", k in summary["by_stratum"]["discriminating"], True)
    check(f"by_domain_seen carries {k}", k in summary["by_domain_seen"]["seen"], True)
check("run-level total", summary["oversized_results"]["result_too_large_n"], 2)

print("\n--- scored 0 strict, never excluded ---")
cfg = summary["by_config"]["model-a|schema-grammar-guided|True"]["metrics"]
check("strict counts it as 0", round(cfg["result_set_f1_strict"], 6), round(1.5 / 4, 6))
check("executed excludes it", round(cfg["result_set_f1_executed"], 6), round(1.5 / 2, 6))

print("\n--- inertness: no oversized records changes nothing ---")
clean = [r for r in records if not (r.get("metrics") or {}).get("result_too_large")]
sc = _build_summary(clean, 1)
check("n is 0", sc["oversized_results"]["result_too_large_n"], 0)
check("max_bytes is None when none fired", sc["oversized_results"]["result_too_large_max_bytes"],
      None)
check("timeout keys still present", "timeout_n" in sc["by_stratum"]["discriminating"], True)
check("truncation keys still present",
      "bindings_skipped_n" in sc["by_stratum"]["discriminating"], True)

import shutil
shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
