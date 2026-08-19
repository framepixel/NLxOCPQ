"""Per-case timeout: status classification, counting, scoring, and dual-path reporting.

The cap itself cannot fire on Windows (no SIGALRM), so this exercises everything downstream of
it plus the guard that makes the no-SIGALRM path a clean no-op. Firing is verified on Linux.

Run from the NLxOCPQ directory, on both platforms:  python test_case_timeout.py
The SIGALRM section is skipped on Windows and is the reason to run this on the cluster too.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_testset import (
    CaseTimeout,
    _build_summary,
    _case_time_limit,
    _classify_run_status,
    _timeout_counts,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}")
        print(f"        want {want!r}")
        failures.append(name)


def rec(case_id, model, *, f1, status="success", pipeline_status="ok", baseline=0.5):
    similarity = None if f1 is None else {
        "combined": {"f1": f1, "precision": f1, "recall": f1},
        "events": {"jaccard": f1, "precision": f1, "recall": f1},
        "objects": {"jaccard": f1, "precision": f1, "recall": f1},
    }
    return {
        "case_id": case_id,
        "nl_query": f"q{case_id}",
        "ocel_path": "Logs/order-management.json",
        "domain_seen": True,
        "discriminating": True,
        "baseline_f1": baseline,
        "gold_query": {"nodes": []},
        "prediction": {"nodes": []},
        "model": model,
        "model_label": model,
        "prompt_strategy": "schema-grammar-guided",
        "schema_enforced": True,
        "schema_validation": True,
        "repeat": 0,
        "config_id": f"{model}|schema-grammar-guided|True",
        "status": status,
        "pipeline_status": pipeline_status,
        "similarity": similarity,
        "llm": {},
        "execution": {"success": f1 is not None, "duration_sec": 1.0},
        "total_duration_sec": 1.0,
        "metrics": {
            "schema_valid": True,
            "hallucinated_entities": False,
            "unsupported_filters": False,
            "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
            "bindings_skipped": None if f1 is None else False,
            "gold_bindings_skipped": False,
        },
    }


def build_records():
    """Per model: clean, timed-out, schema-invalid (no result), clean."""
    out = []
    for model in ("model-a", "model-b"):
        out += [
            rec(1, model, f1=0.90),
            rec(2, model, f1=None, status="timeout", pipeline_status="timeout"),
            rec(3, model, f1=None, status="schema_invalid", pipeline_status="error"),
            rec(4, model, f1=0.60),
        ]
    return out


print("--- _classify_run_status ---")
# The timeout message contains the word "execution", which the substring matches below it would
# otherwise capture. This is the regression the explicit branch exists to prevent.
msg = "case exceeded the 600s time limit (case_timeout_sec) between generation and execution"
check("timeout wins over the 'execution' substring match",
      _classify_run_status("timeout", msg, False, None, None, None), "timeout")
check("a real execution failure is still execution_failed",
      _classify_run_status("error", "failed to execute query", False, None, None, None),
      "execution_failed")
check("schema failure unchanged",
      _classify_run_status("error", "could not be validated against the schema",
                           False, None, None, None), "schema_invalid")
check("unknown error still pipeline_error",
      _classify_run_status("error", "something else entirely", False, None, None, None),
      "pipeline_error")
check("ok path untouched",
      _classify_run_status("ok", None, True, {"success": True}, "f.json", None), "success")

print("\n--- _timeout_counts ---")
counts = _timeout_counts(build_records())
check("timeouts counted", counts["timeout_n"], 2)
check("denominator is EVERY record, not just executed", counts["timeout_denominator"], 8)
check("rate over all records", counts["timeout_rate"], 2 / 8)
check("empty input is safe", _timeout_counts([]),
      {"timeout_n": 0, "timeout_denominator": 0, "timeout_rate": None})
check("record with neither status key is safe",
      _timeout_counts([{}])["timeout_n"], 0)
check("pipeline_status alone is enough",
      _timeout_counts([{"pipeline_status": "timeout"}])["timeout_n"], 1)
check("status alone is enough",
      _timeout_counts([{"status": "timeout"}])["timeout_n"], 1)
check("both set counts once",
      _timeout_counts([{"status": "timeout", "pipeline_status": "timeout"}])["timeout_n"], 1)

print("\n--- keys reach BOTH summary paths ---")
summary = _build_summary(build_records(), 1)
KEYS = ["timeout_n", "timeout_denominator", "timeout_rate"]
for k in KEYS:
    check(f"by_config carries {k}",
          all(k in c["metrics"] for c in summary["by_config"].values()), True)
    check(f"by_stratum carries {k}", k in summary["by_stratum"]["discriminating"], True)
    check(f"by_domain_seen carries {k}", k in summary["by_domain_seen"]["seen"], True)
check("run-level total present", summary["timeouts"]["timeout_n"], 2)
check("by_config rate is per condition",
      {c["metrics"]["timeout_rate"] for c in summary["by_config"].values()}, {1 / 4})

print("\n--- scoring: 0 in strict, absent from executed, denominator intact ---")
cfg = summary["by_config"]["model-a|schema-grammar-guided|True"]["metrics"]
# 4 records: 0.90, timeout->0, schema_invalid->0, 0.60
check("strict scores the timeout as 0", round(cfg["result_set_f1_strict"], 6), round(1.5 / 4, 6))
# executed keeps only the two that produced a result
check("executed excludes it", round(cfg["result_set_f1_executed"], 6), round(1.5 / 2, 6))
check("execution_success_rate untouched by the timeout",
      cfg["execution_success_rate"], 0.5)

print("\n--- inertness when no case times out ---")
clean = [r for r in build_records() if r.get("status") != "timeout"]
s_clean = _build_summary(clean, 1)
check("timeout_n is 0", s_clean["timeouts"]["timeout_n"], 0)
check("rate is 0.0, not None, when records exist", s_clean["timeouts"]["timeout_rate"], 0.0)
check("truncation keys still present alongside",
      all("bindings_skipped_n" in c["metrics"] for c in s_clean["by_config"].values()), True)

print("\n--- _case_time_limit guard ---")
check("disabled when 0", _case_time_limit(0).active, False)
check("disabled when None", _case_time_limit(None).active, False)
import signal as _sig
expected_active = hasattr(_sig, "SIGALRM")
check("active follows SIGALRM availability", _case_time_limit(600).active, expected_active)

ran = False
with _case_time_limit(0):
    ran = True
check("disabled cap is a transparent no-op", ran, True)

ran = False
with _case_time_limit(600):
    ran = True
check("enabled cap does not disturb a fast body", ran, True)

# The alarm must be cleared even when the body raises, or the next case inherits a live timer.
try:
    with _case_time_limit(600):
        raise ValueError("boom")
except ValueError:
    pass
check("CaseTimeout is catchable as Exception", issubclass(CaseTimeout, Exception), True)

if hasattr(_sig, "SIGALRM"):
    print("\n--- SIGALRM actually fires (Linux only) ---")
    import time as _t

    # Fires: a body that sleeps past the deadline must raise CaseTimeout, not hang.
    t0 = _t.time()
    try:
        with _case_time_limit(1):
            _t.sleep(10)
        check("slow body raised", False, True)
    except CaseTimeout as exc:
        elapsed = _t.time() - t0
        check("slow body raised CaseTimeout", True, True)
        check("fired at the deadline, not after the full sleep", elapsed < 5, True)
        check("message names the budget", "1s time limit" in str(exc), True)

    # Cleared: after an exception escapes the body, no stale timer may survive into the next
    # case. A leaked alarm would kill an unrelated case seconds later -- silently, and blamed
    # on the wrong record.
    try:
        with _case_time_limit(1):
            raise ValueError("boom")
    except ValueError:
        pass
    _t.sleep(2.0)
    check("no stale alarm after an exception escaped the body", True, True)

    # Handler restored: the previous SIGALRM handler must be put back.
    sentinel = lambda *a: None
    _sig.signal(_sig.SIGALRM, sentinel)
    with _case_time_limit(5):
        pass
    check("previous SIGALRM handler restored",
          _sig.getsignal(_sig.SIGALRM) is sentinel, True)
    _sig.signal(_sig.SIGALRM, _sig.SIG_DFL)
else:
    print("\nSKIP  SIGALRM firing tests (no SIGALRM on this platform -- run on the cluster)")

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
