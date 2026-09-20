"""`failure_anatomy.py` must classify exactly as notes 23 did, and must never invent a score.

Notes 23's figures were computed ad hoc and could not be reproduced. `src/failure_anatomy.py` makes
them a committed diagnostic, so its bucketing and its structural comparisons are now the thing the
thesis rests on -- and both are easy to get quietly wrong:

  * the notes-23.1 disjoint rule is `precision < 0.05` **AND** `recall < 0.05`; an OR would sweep in
    every over-narrow prediction and inflate the headline class;
  * a record with no comparison (never executed, or its result set too large to store) must be
    `unscored`, NOT disjoint -- counting non-answers as disjoint answers would overstate exactly the
    class notes 23.2 is about;
  * every F1 must come from the record. The one thing this script may not do is compute a metric
    (CLAUDE.md 2.5).

Run from the repo root:  python tests/test_failure_anatomy.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from failure_anatomy import (  # noqa: E402
    _constraint_types,
    _entity_names,
    _filter_types,
    _sat_count,
    _size_filter_fields,
    analyse,
    band,
    bucket,
    gold_distribution,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


_case_counter = [0]


def rec(precision=None, recall=None, f1=None, gold=None, prediction=None, status="success",
        nl_query=None):
    """One record in the shape eval_testset writes. Each gets a distinct question by default."""
    similarity = None
    if f1 is not None:
        similarity = {"combined": {"f1": f1, "precision": precision, "recall": recall},
                      "events": {}, "objects": {}}
    _case_counter[0] += 1
    return {"nl_query": nl_query or f"question {_case_counter[0]}", "status": status,
            "similarity": similarity,
            "gold_query": gold if gold is not None else box_tree(1),
            "prediction": prediction if prediction is not None else box_tree(1),
            "metrics": {"schema_valid": True}, "execution": {"success": status == "success"}}


def box_tree(n_boxes, constraints=None, filters=None, size_filters=None,
             event_types=None, object_types=None):
    """A BindingBoxTree with `n_boxes` boxes; extras go in the FIRST box."""
    nodes = []
    for i in range(n_boxes):
        data = {"newEventVars": {"0": list(event_types)} if (i == 0 and event_types) else {},
                "newObjectVars": {"0": list(object_types)} if (i == 0 and object_types) else {},
                "filters": (filters or []) if i == 0 else [],
                "sizeFilters": (size_filters or []) if i == 0 else [],
                "constraints": (constraints or []) if i == 0 else [],
                "evVarLabels": {}, "obVarLabels": {}, "labels": []}
        nodes.append({"Box": [data, []]})
    return {"nodes": nodes, "edgeNames": []}


print("--- 23.2 bucketing: the rule is AND, and a non-answer is not a disjoint answer ---")
check("perfect match -> exact", bucket(rec(1.0, 1.0, 1.0)), "exact")
check("nothing in common -> disjoint", bucket(rec(0.0, 0.0, 0.0)), "disjoint")
check("both below the threshold -> disjoint", bucket(rec(0.04, 0.02, 0.03)), "disjoint")
check("precision low, recall HIGH -> not disjoint (an AND, not an OR)",
      bucket(rec(0.02, 1.0, 0.04)), "over_broad")
check("recall low, precision HIGH -> over-narrow, not disjoint",
      bucket(rec(1.0, 0.02, 0.04)), "over_narrow")
check("genuine partial overlap", bucket(rec(0.5, 0.5, 0.5)), "partial")
check("no similarity block at all -> unscored", bucket(rec()), "unscored")
check("similarity present but f1 null -> unscored",
      bucket({"similarity": {"combined": {"f1": None}}}), "unscored")
check("threshold 0.0 moves a 0.04/0.02 record out of disjoint",
      bucket(rec(0.04, 0.02, 0.03), threshold=0.0), "partial")
check("...and leaves a true zero disjoint", bucket(rec(0.0, 0.0, 0.0), threshold=0.0), "disjoint")

print("\n--- structure of a tree ---")
tree = box_tree(
    3,
    constraints=[{"type": "SAT", "child_names": ["A"]},
                 {"type": "SAT", "child_names": ["B"]},
                 {"type": "SizeFilter", "filter": {"type": "NumChilds", "min": 1}},
                 {"type": "Filter", "filter": {"type": "TimeBetweenEvents", "min": 0}}],
    filters=[{"type": "O2E", "object": 0, "event": 0},
             {"type": "ObjectAttributeValueFilter", "object": 0}],
    size_filters=[{"type": "NumChilds", "child_name": "A", "min": 1}],
    event_types=["place order"], object_types=["orders", "items"])
check("node count", len(tree["nodes"]), 3)
check("SAT counted twice", _sat_count(tree), 2)
check("a Filter constraint carries its filter's type",
      _constraint_types(tree)["Filter:TimeBetweenEvents"], 1)
check("the SizeFilter CONSTRAINT is counted", _constraint_types(tree)["SizeFilter"], 1)
check("the box-level sizeFilters FIELD is counted separately", _size_filter_fields(tree), 1)
check("box-level filters by type", dict(_filter_types(tree)),
      {"O2E": 1, "ObjectAttributeValueFilter": 1})
events, objects = _entity_names(tree)
check("event activities", events, {"place order"})
check("object types", objects, {"orders", "items"})
check("an unparseable prediction has no entities", _entity_names(None), (set(), set()))
check("...and no constraints", dict(_constraint_types("not a tree")), {})

print("\n--- 23.3 / 23.4: agreement is measured on the disjoint set only ---")
gold = box_tree(2, constraints=[{"type": "SAT", "child_names": ["A"]}],
                event_types=["ship"], object_types=["orders"])
same_entities_wrong_shape = box_tree(1, event_types=["ship"], object_types=["orders"])
wrong_entities = box_tree(2, constraints=[{"type": "SAT", "child_names": ["A"]}],
                          event_types=["pay"], object_types=["items"])
records = [
    rec(0.0, 0.0, 0.0, gold=gold, prediction=same_entities_wrong_shape),   # disjoint, entities agree
    rec(0.0, 0.0, 0.0, gold=gold, prediction=wrong_entities),              # disjoint, entities differ
    rec(1.0, 1.0, 1.0, gold=gold, prediction=gold),                        # exact -> not in 23.3
    rec(gold=gold, prediction=None, status="execution_failed"),            # unscored
]
report = analyse(records)
check("buckets", report["buckets"],
      {"disjoint": 2, "exact": 1, "unscored": 1})
check("executed counts only comparable records", report["executed"], 3)
check("23.3 denominator is the disjoint set", report["entities"]["n"], 2)
check("one of the two disjoint predictions names gold's object types",
      report["entities"]["object_types"], 1)
check("...and its event activities", report["entities"]["event_activities"], 1)
check("node-count agreement on the disjoint set", report["composition"]["node_count"], 1)
check("gold's SAT is counted over the disjoint set only",
      report["composition"]["constructs"]["SAT"]["gold"], 2)
check("the prediction emitted one of them",
      report["composition"]["constructs"]["SAT"]["predicted"], 1)

print("\n--- 23.5 / 23.7 ---")
check("depth buckets keyed by GOLD tree size", sorted(report["depth"]), ["2"])
check("...counting every record, scored or not", report["depth"]["2"]["n"], 4)
check("shape-matched rows", report["shape"]["shape_matched"]["n"], 2)
check("shape-mismatched rows", report["shape"]["shape_mismatched"]["n"], 1)
check("a shape-matched record can still be disjoint",
      report["shape"]["shape_matched"]["still_disjoint"], 1)

print("\n--- the F1 comes from the record, never from this script ---")
# Deliberately inconsistent: precision and recall say 1.0, the stored f1 says 0.5. A script computing
# its own F1 would report 1.0 here.
odd = [rec(1.0, 1.0, 0.5, gold=gold, prediction=gold)]
check("mean F1 is the STORED value", analyse(odd)["shape"]["shape_matched"]["mean_f1"], 0.5)
check("...and exact-match uses it too", analyse(odd)["shape"]["shape_matched"]["exact"], 0)

print("\n--- reporting helpers ---")
check("a band states its numerator and denominator", "(3/4)" in band(3, 4), True)
check("an empty denominator does not divide by zero", band(0, 0), "n/a (n=0)")
check("gold distribution counts two different questions twice",
      gold_distribution([rec(gold=box_tree(2)), rec(gold=box_tree(2))])["n"], 2)
check("...and one question evaluated under many conditions ONCE",
      gold_distribution([rec(gold=box_tree(2), nl_query="same"),
                         rec(gold=box_tree(2), nl_query="same")])["n"], 1)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> notes 23's buckets and structural comparisons are reproducible")
