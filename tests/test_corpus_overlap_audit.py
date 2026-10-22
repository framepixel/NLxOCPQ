"""`corpus_overlap_audit.py` must measure "unseen" the way the thesis claims it.

Pinned here:
  * contamination folds case and whitespace for questions, and compares gold queries as
    canonical JSON, so key order cannot hide a duplicate;
  * type overlap is per log and over the types a query DECLARES (both variable kinds, every box);
  * the name-free shape blanks strings but keeps numbers -- the definition that reproduces §45.2's
    5 of 169 -- so two queries differing only in a type name share a shape, and two differing in a
    bound do not;
  * a block the testset uses and training never shows is reported, and so is one far rarer.

Run from the repo root:  python tests/test_corpus_overlap_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from corpus_overlap_audit import (  # noqa: E402
    block_coverage,
    canonical,
    contamination,
    declared_types,
    fold,
    name_free_shape,
    type_overlap,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def tree(events=None, objects=None, child=None, size_min=None):
    """A one- or two-box query with the given declared types."""
    box = {"newEventVars": {"0": events} if events else {},
           "newObjectVars": {"0": objects} if objects else {},
           "filters": [], "sizeFilters": [], "constraints": []}
    if size_min is not None:
        box["sizeFilters"] = [{"type": "NumChilds", "child_name": "A", "min": size_min,
                               "max": None}]
    nodes = [{"Box": [box, [1] if child else []]}]
    if child:
        nodes.append({"Box": [{"newEventVars": {"1": child}, "newObjectVars": {}, "filters": [],
                               "sizeFilters": [], "constraints": []}, []]})
    return {"nodes": nodes, "edgeNames": [[[0, 1], "A"]] if child else []}


check("folding ignores case and spacing", fold("  Which  ORDERS were paid? "),
      "which orders were paid?")
check("canonical JSON ignores key order", canonical({"b": 1, "a": 2}), canonical({"a": 2, "b": 1}))
check("a non-query has no canonical form", canonical("not a query"), None)

test = [{"nl_query": "Which orders were paid?", "ocel_path": "/x/order-management.json",
         "ocpq_query": tree(objects=["orders"], child=["pay order"])},
        {"nl_query": "Which trucks left?", "ocel_path": "/x/container.json",
         "ocpq_query": tree(objects=["Truck"])}]
check("contamination: folded question and canonical gold both count",
      contamination(test, ["which ORDERS were  paid?"], [tree(objects=["Truck"])]),
      {"nl_exact": 0, "nl_folded": 1, "gold_exact": 1})
check("contamination: nothing shared, nothing counted",
      contamination(test, ["Which invoices were paid?"], [tree(objects=["Invoice"])]),
      {"nl_exact": 0, "nl_folded": 0, "gold_exact": 0})

check("declared types cover both kinds in every box",
      declared_types(tree(objects=["orders"], child=["pay order"])),
      ({"pay order"}, {"orders"}))
rows = {r["log"]: r for r in type_overlap(test, {"Orders", "pay order"})}
check("type overlap is per log: exact", rows["order-management.json"]["exact"], 1)
check("type overlap is per log: any case", rows["order-management.json"]["folded"], 2)
check("a log sharing nothing scores 0", rows["container.json"]["folded"], 0)

check("shapes ignore names",
      name_free_shape(tree(objects=["orders"], size_min=50)),
      name_free_shape(tree(objects=["Truck"], size_min=50)))
check("shapes keep numbers",
      name_free_shape(tree(objects=["orders"], size_min=50))
      == name_free_shape(tree(objects=["orders"], size_min=5)), False)

gold = [tree(objects=["a"], size_min=1), tree(objects=["b"], child=["e"])]
coverage = block_coverage(gold, [tree(objects=["c"])] * 100)
check("a block the training never shows is reported",
      bool(coverage["testset_uses_training_never_shows"]), True)
coverage = block_coverage(gold, gold * 50)
check("identical distributions report nothing", (coverage["testset_uses_training_never_shows"],
                                                 coverage["under_trained"]), ([], []))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
