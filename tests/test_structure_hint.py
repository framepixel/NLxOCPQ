"""The oracle structure hint: does it describe gold's tree without leaking gold's meaning?

The ablation only measures what it claims to if the hint carries SHAPE and withholds MEANING. A
leak of entity names or constraint parameters would turn "structure helps" into "we told it the
answer", and the leak would be invisible in the resulting numbers -- they would simply look good.
So every withheld field is asserted absent by name here rather than trusted to the implementation.

The other failure this guards is the quiet one: a malformed gold entry must degrade to the
baseline prompt, not raise, or the ablation loses cases for a reason unrelated to what it studies.

Run from the repo root:  python tests/test_structure_hint.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_testset import (  # noqa: E402
    STRUCTURE_HINT_LEVELS,
    _build_structure_hint,
    _constraint_type_name,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def absent(name, needle, haystack):
    ok = needle.lower() not in (haystack or "").lower()
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        LEAKED {needle!r} into the hint")
        failures.append(name)


# A three-node gold with everything the hint must withhold: entity names, O2E/O2O filters, a
# size filter, a time window with real bounds, and an attribute value.
def box(ev, ob, filters, constraints, children):
    return {"Box": [{"newEventVars": ev, "newObjectVars": ob, "filters": filters,
                     "sizeFilters": [{"type": "SizeFilter", "min": 2, "max": 9}],
                     "constraints": constraints, "evVarLabels": {}, "obVarLabels": {}}, children]}


GOLD = {
    "nodes": [
        box({"0": ["place order"]}, {"0": ["orders"]},
            [{"type": "O2E", "object": 0, "event": 0, "qualifier": "places"}],
            [{"type": "SAT", "child_names": ["A"]}], [1, 2]),
        box({"1": ["package delivered"]}, {"1": ["items"]},
            [{"type": "O2O", "object": 0, "other_object": 1, "qualifier": "contains"}],
            [{"type": "Filter", "filter": {"type": "TimeBetweenEvents", "from_event": 0,
                                           "to_event": 1, "min_seconds": 0,
                                           "max_seconds": 2116800}}], []),
        box({}, {"2": ["employees"]}, [],
            [{"type": "Filter", "filter": {"type": "ObjectAttributeValueFilter",
                                           "attribute": "Region", "value": "Rhineland"}}], []),
    ],
    "edgeNames": [[[0, 1], "A"], [[0, 2], "B"]],
}

print("--- _constraint_type_name ---")
check("plain type", _constraint_type_name({"type": "SAT"}), "SAT")
check("wrapped basic filter",
      _constraint_type_name({"type": "Filter", "filter": {"type": "TimeBetweenEvents"}}),
      "Filter:TimeBetweenEvents")
check("wrapped with no inner type", _constraint_type_name({"type": "Filter"}), "Filter")
check("garbage passes through as None", _constraint_type_name("SAT"), None)

print("\n--- skeleton level: shape is described ---")
skel = _build_structure_hint(GOLD, "skeleton")
check("3 nodes announced", "3 node(s)." in skel, True)
check("node 0 arity", "node 0: declares 1 new event variable(s), 1 new object variable(s)" in skel,
      True)
check("node 2 has no event vars", "node 2: declares 0 new event variable(s)" in skel, True)
check("node 0 children named with edge labels",
      'children: node 1 on edge "A", node 2 on edge "B"' in skel, True)
check("leaf marked", "node 1: declares 1 new event variable(s), 1 new object variable(s)"
      "; no children" in skel, True)
check("edgeNames reproduced", '"A"' in skel and '"B"' in skel, True)

print("\n--- skeleton level: MEANING is withheld ---")
for term in ("place order", "orders", "package delivered", "items", "employees"):
    absent(f"entity name {term!r} withheld", term, skel)
for term in ("O2E", "O2O", "places", "contains"):
    absent(f"filter/qualifier {term!r} withheld", term, skel)
for term in ("2116800", "min_seconds", "max_seconds", "Rhineland", "Region", "sizeFilters"):
    absent(f"parameter {term!r} withheld", term, skel)
absent("constraint types withheld at skeleton level", "SAT", skel)
absent("wrapped constraint types withheld too", "TimeBetweenEvents", skel)

print("\n--- skeleton+constraints level: types appear, parameters still do not ---")
both = _build_structure_hint(GOLD, "skeleton+constraints")
check("SAT named", "SAT" in both, True)
check("wrapped type named", "Filter:TimeBetweenEvents" in both, True)
check("second wrapped type named", "Filter:ObjectAttributeValueFilter" in both, True)
check("count stated", "carries 1 constraint(s) of type" in both, True)
for term in ("2116800", "min_seconds", "max_seconds", "Rhineland", "child_names"):
    absent(f"parameter {term!r} still withheld", term, both)
for term in ("place order", "orders", "items", "employees"):
    absent(f"entity name {term!r} still withheld", term, both)
check("a node with no constraints says so",
      "carries no constraints" in _build_structure_hint(
          {"nodes": [box({}, {}, [], [], [])], "edgeNames": []}, "skeleton+constraints"), True)

print("\n--- the two levels differ ---")
check("constraints level is a superset in content", len(both) > len(skel), True)

print("\n--- inertness and bad input: degrade, never raise ---")
check("unknown level -> None", _build_structure_hint(GOLD, "everything"), None)
check("None level -> None", _build_structure_hint(GOLD, None), None)
check("gold is None -> None", _build_structure_hint(None, "skeleton"), None)
check("gold is a string -> None", _build_structure_hint("{}", "skeleton"), None)
check("no nodes key -> None", _build_structure_hint({"edgeNames": []}, "skeleton"), None)
check("empty nodes -> None", _build_structure_hint({"nodes": []}, "skeleton"), None)
check("nodes present but none are Boxes -> None",
      _build_structure_hint({"nodes": [{"NotABox": []}], "edgeNames": []}, "skeleton"), None)
check("malformed edge entries are skipped, not fatal",
      "1 node(s)." in _build_structure_hint(
          {"nodes": [box({}, {}, [], [], [])], "edgeNames": [["bad"], None, [[0], "X"]]},
          "skeleton"), True)

print("\n--- the level names the pipeline dispatches on ---")
check("exactly two levels", STRUCTURE_HINT_LEVELS, ("skeleton", "skeleton+constraints"))

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
