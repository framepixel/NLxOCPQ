"""A query ocpq_cli declined to evaluate must not count as a successful execution.

2026-09-13: `run_evaluate` (OCPQ/backend/ocpq_cli/src/main.rs) serialises
`Result<EvaluateBoxTreeResult, String>` straight into its export and exits 0 either way. When
`evaluate_box_tree` returns `Err` -- "Could not get o1" in 61 of 68 replayed trees, a filter naming
a variable no enclosing box declares -- the export is `{"Err": "Could not get o1"}`, and
`println!("Evaluated in ...")` is never reached
because the `?` returns first. The harness saw exit 0 plus a result file and recorded `success`, and
`execution_success_rate`, computed from the exit status, counted it: 31 of N2's 117 discriminating
"successes" (86.7% reported, 63.7% evaluated), and 400 of matrix_full's 3,100.

Two signals, one per situation:
  * live, while the export still exists: its `Err` message (`_read_cli_error`);
  * afterwards, for `--recompute`: the stored stdout, whose "Evaluated in" line is printed if and
    only if evaluation returned Ok (`_cli_reported_evaluation`).

Run from the repo root:  python tests/test_cli_evaluation_error.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_testset import (  # noqa: E402
    _build_summary,
    _classify_run_status,
    _cli_reported_evaluation,
    _execution_succeeded,
    _read_cli_error,
    _reclassify_unevaluated,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


# Verbatim shapes from the 2026-09-13 arity ablation records.
EVALUATED = ("Imported OCEL 2.0 in 116.51036ms\nEvaluated in 1.208104ms (Size: 2)\n"
             "Exported result in 1.103134ms")
NOT_EVALUATED = "Imported OCEL 2.0 in 93.351043ms\nExported result in 104.811µs"
ERROR_MESSAGE = "Could not get o1"

print("--- _cli_reported_evaluation: what the CLI's own stdout says ---")
check("an evaluation line -> True", _cli_reported_evaluation(EVALUATED), True)
check("exported without evaluating -> False", _cli_reported_evaluation(NOT_EVALUATED), False)
check("no stdout -> unknown, not failed", _cli_reported_evaluation(None), None)
check("empty stdout -> unknown", _cli_reported_evaluation(""), None)
check("a harness warning -> unknown",
      _cli_reported_evaluation("warning: run_ocpq_query returned no result"), None)
check("a panic dump that never exported -> unknown",
      _cli_reported_evaluation("thread '<unnamed>' panicked at structs.rs:784:35"), None)
check("a CLIPPED stdout without the evaluation line -> unknown: the line may be in the cut",
      _cli_reported_evaluation("Imported OCEL 2.0 in 1ms\nthread panicked\n[... 9000 characters / "
                               "~120 lines omitted (cap 8000); ocpq_cli panics once per worker "
                               "thread ...]\nExported result in 2ms"), None)
check("the bindings-skipped notice still counts as evaluated",
      _cli_reported_evaluation("Imported OCEL 2.0 in 1ms\n[!!!] Query yielded too many results. "
                               "Some bindings were skipped.\nEvaluated in 9ms (Size: 3)\n"
                               "Exported result in 2ms"), True)

print("\n--- _read_cli_error: the export's own message ---")
tmp = Path(tempfile.mkdtemp(prefix="clierr_"))
err_file = tmp / "err.json"
err_file.write_text(json.dumps({"Err": ERROR_MESSAGE}), encoding="utf-8")
ok_file = tmp / "ok.json"
ok_file.write_text(json.dumps({"Ok": {"evaluationResults": [{"situations": []}]}}),
                   encoding="utf-8")
check("an Err export yields its message", _read_cli_error(str(err_file)), ERROR_MESSAGE)
check("an Ok export is not an error", _read_cli_error(str(ok_file)), None)
check("a missing file is not an error", _read_cli_error(str(tmp / "nope.json")), None)
check("no path at all", _read_cli_error(None), None)
check("a file over the size bound is never parsed for this",
      _read_cli_error(str(err_file), max_bytes=4), None)
(tmp / "junk.json").write_text("not json", encoding="utf-8")
check("an unreadable file is not an error", _read_cli_error(str(tmp / "junk.json")), None)

print("\n--- _classify_run_status ---")
ok = ("ok", None, True, {"success": True}, "f.json")
check("evaluated -> success", _classify_run_status(*ok, EVALUATED), "success")
check("an Err export -> execution_failed",
      _classify_run_status(*ok, NOT_EVALUATED, cli_error=ERROR_MESSAGE), "execution_failed")
check("stdout proving non-evaluation, message unread -> execution_failed",
      _classify_run_status(*ok, NOT_EVALUATED), "execution_failed")
check("no stdout recorded -> unchanged (success), for every older call site",
      _classify_run_status(*ok, None), "success")
check("result_too_large still decided first",
      _classify_run_status(*ok, NOT_EVALUATED, result_too_large=True), "result_too_large")
check("schema_invalid still decided first",
      _classify_run_status("ok", None, False, {"success": True}, "f.json", NOT_EVALUATED,
                           cli_error=ERROR_MESSAGE), "schema_invalid")
check("timeout still decided first",
      _classify_run_status("timeout", "exceeded the 600s time limit", False, None, None,
                           NOT_EVALUATED), "timeout")

print("\n--- _execution_succeeded ---")
check("exit 0 and evaluated", _execution_succeeded(
    {"status": "success", "execution": {"success": True}, "query_stdout": EVALUATED}), True)
check("exit 0, never evaluated (an old, un-recomputed record)", _execution_succeeded(
    {"status": "success", "execution": {"success": True}, "query_stdout": NOT_EVALUATED}), False)
check("exit 0 with a recorded CLI error", _execution_succeeded(
    {"status": "execution_failed", "execution": {"success": True, "cli_error": ERROR_MESSAGE}}),
    False)
check("non-zero exit", _execution_succeeded({"execution": {"success": False}}), False)
check("no stdout stored -> the exit status stands", _execution_succeeded(
    {"status": "success", "execution": {"success": True}}), True)

print("\n--- _reclassify_unevaluated (the --recompute path) ---")
records = [
    {"status": "success", "execution": {"success": True}, "query_stdout": NOT_EVALUATED},
    {"status": "success", "execution": {"success": True}, "query_stdout": EVALUATED},
    {"status": "schema_invalid", "execution": {"success": False}, "query_stdout": NOT_EVALUATED},
    {"status": "success", "execution": {"success": True}, "query_stdout": None},
]
check("only the never-evaluated success is relabelled", _reclassify_unevaluated(records), 1)
check("...to execution_failed", records[0]["status"], "execution_failed")
check("...recording where it came from", records[0].get("status_reclassified_from"), "success")
check("...and that it was not evaluated", records[0]["execution"].get("evaluated"), False)
check("...with a reason, since the message itself is gone",
      bool(records[0]["execution"].get("cli_error")), True)
check("an evaluated record is untouched", records[1]["status"], "success")
check("a non-success record is untouched", records[2]["status"], "schema_invalid")
check("a record without stdout is untouched", records[3]["status"], "success")
check("idempotent", _reclassify_unevaluated(records), 0)

print("\n--- the rate itself ---")


def rec(case_id, *, f1, stdout):
    sim = None if f1 is None else {
        "combined": {"f1": f1, "precision": f1, "recall": f1},
        "events": {"jaccard": f1, "precision": f1, "recall": f1},
        "objects": {"jaccard": f1, "precision": f1, "recall": f1}}
    return {
        "case_id": case_id, "nl_query": f"q{case_id}", "ocel_path": "x.json",
        "domain_seen": True, "discriminating": True, "baseline_f1": 0.5,
        "gold_query": {"nodes": []}, "prediction": {"nodes": []},
        "model": "m", "model_label": "m", "prompt_strategy": "schema-allowed-entities-only",
        "schema_enforced": True, "schema_validation": True, "repeat": 0, "config_id": "m|c",
        "status": "success", "pipeline_status": "ok", "similarity": sim, "llm": {},
        "execution": {"success": True, "duration_sec": 1.0}, "query_stdout": stdout,
        "total_duration_sec": 1.0,
        "metrics": {"schema_valid": True, "hallucinated_entities": False,
                    "unsupported_filters": False,
                    "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                    "bindings_skipped": False, "gold_bindings_skipped": False,
                    "result_too_large": False, "result_file_bytes": None},
    }


run = [rec(1, f1=0.9, stdout=EVALUATED), rec(2, f1=0.4, stdout=EVALUATED),
       rec(3, f1=None, stdout=NOT_EVALUATED), rec(4, f1=None, stdout=NOT_EVALUATED)]
summary = _build_summary(run, 1)
by_config = summary["by_config"]["m|c"]["metrics"]
stratum = summary["by_stratum"]["discriminating"]
check("by_config: execution_success_rate counts only evaluated runs",
      by_config["execution_success_rate"], 0.5)
check("by_config: the old exit-status rate is kept under its own name",
      by_config["cli_exit_success_rate"], 1.0)
check("by_stratum: execution_success_rate counts only evaluated runs",
      stratum["execution_success_rate"], 0.5)
check("by_stratum: cli_exit_success_rate kept", stratum["cli_exit_success_rate"], 1.0)
check("the primary metric is untouched by the relabelling",
      round(stratum["result_set_f1_strict"], 6), round(1.3 / 4, 6))

import shutil  # noqa: E402
shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> a never-evaluated query is an execution failure, live and on recompute")
