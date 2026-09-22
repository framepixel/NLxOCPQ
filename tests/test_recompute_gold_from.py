"""`--recompute --gold-from`: a corrected gold is scored without re-running a single model.

Records carry their own copy of the gold query, so correcting an entry in `data/testset.json`
(#25, 2026-09-22) changes nothing in a finished run unless the rescore is pointed at the new gold.
Four ways that could go wrong, each checked here:

  * **Scoring against the old gold anyway**, or re-pointing a record whose gold did not change.
  * **Losing the old query.** It is kept on the record, so the correction is auditable.
  * **Hiding the oracle caveat.** A prediction generated from the OLD gold (a structure hint, a
    gold-derived forced arity) described a different query; it must be flagged and counted -- and
    the NON-oracle constant-arity rung, which fills `forced_root_arity` too, must not be.
  * **Re-pointing across logs.** A matching question on a different log is refused.

Run from the repo root:  python tests/test_recompute_gold_from.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import result_store  # noqa: E402
from eval_testset import _recompute_metrics, _regold_records  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


LOG = "..\\Logs\\order-management.json"
OLD_GOLD = {"nodes": [{"Box": [{"newObjectVars": {"0": ["orders"]}}, []]}], "edgeNames": []}
NEW_GOLD = {"nodes": [{"Box": [{"newEventVars": {"0": ["pay order"]}}, []]}], "edgeNames": []}
KEPT_GOLD = {"nodes": [{"Box": [{"newObjectVars": {"0": ["items"]}}, []]}], "edgeNames": []}
PREDICTION = {"nodes": [{"Box": [{"newEventVars": {"0": ["pay order"]},
                                  "newObjectVars": {}}, []]}], "edgeNames": []}


def sits(*rows):
    return (set(), set(), {tuple(r) for r in rows})


OLD_SET = sits(((), (1,), False), ((), (2,), True))      # orders -- the old answer kind
NEW_SET = sits(((10,), (), False), ((11,), (), True))    # payment events -- the new answer kind


def rec(nl, gold, **extra):
    base = {"case_id": 1, "nl_query": nl, "ocel_path": LOG, "gold_query": gold,
            "prediction": PREDICTION, "model": "m", "model_label": "m",
            "prompt_strategy": "schema-grammar-guided", "schema_enforced": True,
            "schema_validation": True, "repeat": 1, "config_id": "m|s|True",
            "status": "success", "pipeline_status": "ok", "domain_seen": True,
            "discriminating": True, "baseline_f1": 0.4,
            "similarity": {"combined": {"f1": 0.0, "precision": 0.0, "recall": 0.0}},
            "llm": {}, "execution": {"success": True, "duration_sec": 1.0},
            "total_duration_sec": 1.0,
            "metrics": {"schema_valid": True, "hallucinated_entities": False,
                        "unsupported_filters": False, "bindings_skipped": False,
                        "gold_bindings_skipped": False}}
    base.update(extra)
    return base


tmp = Path(tempfile.mkdtemp(prefix="regold-"))
run = tmp / "run"
(run / "result_store").mkdir(parents=True)
gold_store = tmp / "audit_store"
records = [
    rec("Retrieve the payment events.", OLD_GOLD),
    rec("Which items exist?", KEPT_GOLD),
    rec("Retrieve the payment events.", OLD_GOLD, config_id="m|oracle|True",
        structure_hint_level="shape", forced_root_arity=None),
    rec("Retrieve the payment events.", OLD_GOLD, config_id="m|const|True",
        forced_root_arity=[0, 0, 1], constant_root_arity=True),
]
(run / "predictions.json").write_text(json.dumps(records), encoding="utf-8")
# The run executed the OLD gold and the prediction; only the audit store has the NEW gold.
result_store.store(run / "result_store", LOG, OLD_GOLD, OLD_SET)
result_store.store(run / "result_store", LOG, KEPT_GOLD, OLD_SET)
result_store.store(run / "result_store", LOG, PREDICTION, NEW_SET)
result_store.store(gold_store, LOG, NEW_GOLD, NEW_SET)
testset = tmp / "testset.json"
testset.write_text(json.dumps([
    {"nl_query": "Retrieve the payment events.", "ocel_path": LOG, "ocpq_query": NEW_GOLD},
    {"nl_query": "Which items exist?", "ocel_path": LOG, "ocpq_query": KEPT_GOLD},
]), encoding="utf-8")

print("--- without --gold-from nothing moves: the records' own gold is scored ---")
out = _recompute_metrics(run, tmp / "plain")
plain = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
check("the prediction scores 0 against the OLD gold (it answers a different kind)",
      plain[0]["similarity"]["combined"]["f1"], 0.0)
check("no regold block in the summary",
      "regold" in json.loads((out / "summary.json").read_text(encoding="utf-8")), False)

print("\n--- with --gold-from the corrected gold is scored, from the audit store ---")
out = _recompute_metrics(run, tmp / "regold", gold_from=testset, gold_store_dir=gold_store)
redone = json.loads((out / "predictions.json").read_text(encoding="utf-8"))
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
check("the re-pointed record scores 1.0 against the NEW gold",
      redone[0]["similarity"]["combined"]["f1"], 1.0)
check("its gold_query is the corrected one", redone[0]["gold_query"], NEW_GOLD)
check("and the superseded query is kept on the record", redone[0]["gold_query_superseded"],
      OLD_GOLD)
check("an entry whose gold did not change is left alone",
      ("gold_query_superseded" in redone[1], redone[1]["gold_query"]), (False, KEPT_GOLD))
check("no gold went missing", summary["recompute_unavailable"]["gold_missing"], 0)
check("the summary says how many records moved", summary["regold"]["records_regolded"], 3)

print("\n--- the oracle caveat is flagged, and only for gold-derived input ---")
check("a structure-hint record is flagged",
      redone[2].get("oracle_input_from_superseded_gold"), True)
check("the constant-arity rung is NOT (its [0,0,1] never came from gold)",
      redone[3].get("oracle_input_from_superseded_gold"), None)
check("the count in the summary", summary["regold"]["oracle_input_from_superseded_gold"], 1)

print("\n--- a matching question on another log is refused ---")
elsewhere = [rec("Retrieve the payment events.", OLD_GOLD, ocel_path="..\\Logs\\other.json")]
try:
    _regold_records(elsewhere, testset)
    check("re-pointing across logs raises", "no raise", "ValueError")
except ValueError:
    check("re-pointing across logs raises", "ValueError", "ValueError")

shutil.rmtree(tmp, ignore_errors=True)
print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> a corrected gold is rescored from the store, auditably, caveats counted")
