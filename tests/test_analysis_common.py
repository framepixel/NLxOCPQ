"""`analysis_common.py` must read records exactly as the evaluator aggregated them.

The analysis scripts cannot import `eval_testset.py` (it pulls in torch), so they read per-record
values and prove the reading with a reproduction guard against `summary.json`. Pinned here:

  * the strict-zero convention: a record without a comparable result scores 0, not "missing";
  * `evaluated` follows `_execution_succeeded`, including its stdout fallback for old records;
  * questions are identified by question text + log, never by `case_id`, which restarts in every
    case shard -- two records with the same `case_id` on different questions stay apart;
  * draws are averaged within a question before anything else;
  * the guard passes on a consistent run and catches a doctored summary.

Run from the repo root:  python tests/test_analysis_common.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import analysis_common as ac  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def rec(question, f1=None, config="c1", repeat=1, discriminating=True, evaluated=True,
        status="success", case_id=1, skill=None, structural=0.5, schema_valid=True,
        stdout=None):
    """One record in the shape eval_testset writes."""
    similarity = None if f1 is None else {"combined": {"f1": f1}}
    return {
        "nl_query": question, "ocel_path": "/somewhere/order-management.json",
        "case_id": case_id, "config_id": config, "repeat": repeat,
        "discriminating": discriminating, "status": status, "similarity": similarity,
        "execution": {"success": status != "schema_invalid", "evaluated": evaluated,
                      "cli_error": None},
        "query_stdout": stdout,
        "metrics": {"skill": skill, "structural": {"f1": structural},
                    "schema_valid": schema_valid},
    }


# --- per-record readers -------------------------------------------------------------------------
check("strict F1 of a non-result is 0", ac.strict_f1(rec("q", f1=None)), 0.0)
check("executed F1 of a non-result is undefined", ac.executed_f1(rec("q", f1=None)), None)
check("strict F1 reads the combined score", ac.strict_f1(rec("q", f1=0.75)), 0.75)
check("not evaluated when the CLI declined", ac.evaluated(rec("q", evaluated=False)), 0.0)
check("not evaluated when execution failed",
      ac.evaluated(rec("q", status="execution_failed")), 0.0)
old = rec("q", evaluated=None, stdout="Exported result in 2ms")
check("old record: exported without evaluating is not evaluated", ac.evaluated(old), 0.0)
old = rec("q", evaluated=None, stdout="Evaluated in 3ms\nExported result in 2ms")
check("old record: the evaluation line counts", ac.evaluated(old), 1.0)
old = rec("q", evaluated=None, stdout="... 120 lines omitted (cap 200) ...")
check("old record: a clipped stdout never proves non-evaluation", ac.evaluated(old), 1.0)

# --- identity and averaging ---------------------------------------------------------------------
records = [
    rec("first question", f1=1.0, case_id=1, repeat=1),
    rec("first question", f1=0.0, case_id=1, repeat=2),
    rec("second question", f1=0.5, case_id=1, repeat=1),  # same case_id, another shard half
    rec("second question", f1=0.5, case_id=1, repeat=2),
    rec("easy question", f1=1.0, case_id=2, discriminating=False),
]
values = ac.case_values(records, "strict_f1")
check("case_id collisions do not merge questions", len(values), 2)
check("draws are averaged per question",
      values[("first question", "order-management.json")], 0.5)
check("the non-discriminating stratum is excluded by default",
      ("easy question", "order-management.json") in values, False)
check("population all keeps it", len(ac.case_values(records, "strict_f1", "all")), 3)
check("a single draw can be selected",
      ac.case_values(records, "strict_f1", draw=2)[("first question", "order-management.json")],
      0.0)
check("a log can be excluded",
      ac.case_values(records, "strict_f1", exclude_logs=["order-management.json"]), {})
try:
    ac.case_values(records + [rec("first question", f1=1.0, repeat=1)], "strict_f1")
    check("a duplicated record raises", False, True)
except ValueError:
    check("a duplicated record raises", True, True)
try:
    ac.case_values(records, "strict_f1", config_ids=["c1", "missing"])
    check("pooling a condition the run lacks raises", False, True)
except ValueError:
    check("pooling a condition the run lacks raises", True, True)
try:
    ac.entry_key({"case_id": 3})
    check("a record without nl_query cannot be identified", False, True)
except ValueError:
    check("a record without nl_query cannot be identified", True, True)

# --- the reproduction guard ---------------------------------------------------------------------
run_records = [rec("a", f1=1.0, skill=1.0), rec("b", f1=None, evaluated=False, skill=-1.0,
                                                 status="execution_failed")]
summary = {
    "by_config": {"c1": {"metrics": {
        "result_set_f1_strict": 0.5, "result_set_f1_executed": 1.0, "skill_score": 0.0,
        "execution_success_rate": 0.5, "structural_f1": 0.5, "schema_valid_rate": 1.0}}},
    "by_stratum": {"discriminating": {"result_set_f1_strict": 0.5, "skill_score": 0.0}},
}


class FakeRun:
    """The attributes `reproduction_errors` reads, without a directory on disk."""

    def __init__(self, records, summary):
        self.path = Path("synthetic")
        self.records = records
        self.summary = summary
        self.config_ids = ["c1"]


check("the guard passes on consistent data",
      ac.reproduction_errors(FakeRun(run_records, summary)), [])
doctored = copy.deepcopy(summary)
doctored["by_config"]["c1"]["metrics"]["result_set_f1_strict"] = 0.51
check("the guard catches a doctored summary",
      len(ac.reproduction_errors(FakeRun(run_records, doctored))), 1)
missing = copy.deepcopy(summary)
del missing["by_config"]["c1"]
check("the guard reports a condition missing from the summary",
      len(ac.reproduction_errors(FakeRun(run_records, missing))), 1)

# --- formatting -------------------------------------------------------------------------------
check("signed numbers use a math minus", ac.fmt(-0.0216, signed=True), "$-$0.0216")
check("undefined renders as a dash", ac.fmt(None), "--")
check("rates render as escaped percentages", ac.fmt_pct(0.885), "88.5\\%")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
