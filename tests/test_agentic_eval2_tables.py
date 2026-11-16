"""`agentic_eval2_tables.py` must compare trees the way notes 11 §45.9 did.

The coarse structure was calibrated against §45.9 (13--18% borrowed with examples, 0.9% for A1*
against the examples A1*+S was shown); these checks pin it on synthetic trees:

  * names and values do not count, only the shape: children, variable counts, filter kinds;
  * kinds are a multiset (two O2O filters differ from one) and boxes keep their order;
  * the example pool stores trees as Python literals, the records as JSON: both are read;
  * a SAT is a constraint of type SAT in any box;
  * the notes' figures are compared at the decimals the notes print.

Run from the repo root:  python tests/test_agentic_eval2_tables.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_eval2_tables import _reconcile, _tree, coarse_structure, has_sat  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def box(objects, events, filters=(), constraints=()):
    return {"newEventVars": {str(i): [e] for i, e in enumerate(events)},
            "newObjectVars": {str(i): [o] for i, o in enumerate(objects)},
            "filters": [{"type": t} for t in filters], "sizeFilters": [],
            "constraints": [{"type": t} for t in constraints]}


TREE = {"nodes": [{"Box": [box(["order"], [], constraints=["SAT"]), [1]]},
                  {"Box": [box(["item"], ["pack"], filters=["O2O", "E2O"]), []]}]}
RENAMED = {"nodes": [{"Box": [box(["Customer"], [], constraints=["SAT"]), [1]]},
                     {"Box": [box(["Invoice"], ["Send"], filters=["O2O", "E2O"]), []]}]}
TWO_O2O = copy.deepcopy(TREE)
TWO_O2O["nodes"][1]["Box"][0]["filters"].append({"type": "O2O"})
SWAPPED = {"nodes": [TREE["nodes"][1], TREE["nodes"][0]]}

check("names do not change the structure", coarse_structure(TREE), coarse_structure(RENAMED))
check("filter kinds are a multiset", coarse_structure(TREE) == coarse_structure(TWO_O2O), False)
check("box order counts", coarse_structure(TREE) == coarse_structure(SWAPPED), False)
check("a pool tree (Python literal) is read", _tree(repr(TREE)), TREE)
check("a record tree (JSON) is read", _tree(json.dumps(TREE)), TREE)
check("a non-tree is None", _tree("{not a tree"), None)
check("SAT in any box counts", has_sat(TREE), True)
check("no SAT", has_sat({"nodes": [TREE["nodes"][1]]}), False)
check("19.745% is not the notes' 19.8%",
      _reconcile("x", {"r": 217 / 1099}, {"r": 0.198}) != [], True)
check("546.4 tokens is the notes' 546", _reconcile("x", {"t": 546.4}, {"t": 546}), [])
check("probe-CLI pairs compare exactly", _reconcile("x", {"p": (298, 299)}, {"p": (298, 299)}),
      [])

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("\nall checks passed")
