"""The oracle structure hint: does the scaffold describe gold's tree without leaking gold's meaning?

The ablation only measures what it claims to if the hint carries SHAPE and withholds MEANING. A
leak of entity names or constraint parameters would turn "structure helps" into "we told it the
answer", and the leak would be invisible in the resulting numbers -- they would simply look good.
So every withheld field is asserted absent by name here rather than trusted to the implementation.

The hint is a JSON scaffold to complete, not a prose description. The prose version measured 18%
compliance on the cluster (job 3339751), below the 30% the same model reaches with no hint at all;
see `_build_structure_hint`'s docstring. Two consequences are tested here: the scaffold must be
valid JSON (a malformed one would be worse than useless to a JSON-emitting model), and the
structural fields must survive into it verbatim.

The other failure this guards is the quiet one: a malformed gold entry must degrade to the
baseline prompt, not raise, or the ablation loses cases for a reason unrelated to what it studies.

Run from the repo root:  python tests/test_structure_hint.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_testset import (  # noqa: E402
    STRUCTURE_HINT_LEVELS,
    _build_structure_hint,
    _constraint_stub,
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


def scaffold_of(hint):
    """The JSON body of the hint, without the instruction preamble."""
    return json.loads(hint[hint.index("{"):])


# A three-node gold with everything the hint must withhold: entity names, O2E/O2O filters with
# qualifiers, a size filter with bounds, a time window with real seconds, and an attribute value.
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

print("\n--- _constraint_stub keeps the type, drops the parameters ---")
st = _constraint_stub({"type": "SAT", "child_names": ["A"]})
check("SAT type kept", st["type"], "SAT")
check("child_names dropped -- derivable from edgeNames", "child_names" in st, False)
st2 = _constraint_stub({"type": "Filter", "filter": {"type": "TimeBetweenEvents",
                                                     "min_seconds": 0, "max_seconds": 99}})
check("wrapper preserved", st2["type"], "Filter")
check("inner type kept", st2["filter"]["type"], "TimeBetweenEvents")
check("inner parameters dropped", "max_seconds" in st2["filter"], False)
check("unparseable constraint -> None", _constraint_stub(None), None)

print("\n--- skeleton level: the scaffold is valid JSON with the right shape ---")
skel = _build_structure_hint(GOLD, "skeleton")
sc = scaffold_of(skel)
check("scaffold parses as JSON", isinstance(sc, dict), True)
check("3 nodes", len(sc["nodes"]), 3)
check("node count stated in the instruction", "EXACTLY 3 node(s)" in skel, True)
check("edgeNames reproduced verbatim", sc["edgeNames"], GOLD["edgeNames"])
b0 = sc["nodes"][0]["Box"]
check("node 0 children preserved", b0[1], [1, 2])
check("node 1 is a leaf", sc["nodes"][1]["Box"][1], [])
check("variable KEYS preserved", list(b0[0]["newEventVars"]), ["0"])
check("node 2 binds no event vars", sc["nodes"][2]["Box"][0]["newEventVars"], {})
check("node 2 object key preserved", list(sc["nodes"][2]["Box"][0]["newObjectVars"]), ["2"])

print("\n--- skeleton level: every semantic slot is a fill marker ---")
check("event var value is a placeholder", b0[0]["newEventVars"]["0"], ["<FILL: event type>"])
check("object var value is a placeholder", b0[0]["newObjectVars"]["0"], ["<FILL: object type>"])
check("filters withheld", str(b0[0]["filters"]).startswith("<FILL"), True)
check("sizeFilters withheld", str(b0[0]["sizeFilters"]).startswith("<FILL"), True)
check("constraints withheld at this level", str(b0[0]["constraints"]).startswith("<FILL"), True)

print("\n--- skeleton level: MEANING is withheld ---")
for term in ("place order", "orders", "package delivered", "items", "employees"):
    absent(f"entity name {term!r} withheld", term, skel)
for term in ("O2E", "O2O", "places", "contains"):
    absent(f"filter/qualifier {term!r} withheld", term, skel)
for term in ("2116800", "min_seconds", "max_seconds", "Rhineland", "Region", "child_names"):
    absent(f"parameter {term!r} withheld", term, skel)
# The KEY "sizeFilters" now appears (it is an OCPQ schema field the prompt already names); what
# must not appear is gold's sizeFilter CONTENTS.
absent("gold sizeFilter bound 'min' withheld", '"min"', skel)
absent("gold sizeFilter bound 'max' withheld", '"max"', skel)
absent("constraint types withheld at skeleton level", "SAT", skel)
absent("wrapped constraint types withheld too", "TimeBetweenEvents", skel)

print("\n--- skeleton+constraints: types appear, parameters still do not ---")
both = _build_structure_hint(GOLD, "skeleton+constraints")
scb = scaffold_of(both)
check("scaffold still valid JSON", len(scb["nodes"]), 3)
c0 = scb["nodes"][0]["Box"][0]["constraints"]
check("constraints is now a list", isinstance(c0, list), True)
check("SAT named", c0[0]["type"], "SAT")
c1 = scb["nodes"][1]["Box"][0]["constraints"]
check("wrapped type named", c1[0]["filter"]["type"], "TimeBetweenEvents")
c2 = scb["nodes"][2]["Box"][0]["constraints"]
check("second wrapped type named", c2[0]["filter"]["type"], "ObjectAttributeValueFilter")
for term in ("2116800", "min_seconds", "max_seconds", "Rhineland", "child_names"):
    absent(f"parameter {term!r} still withheld", term, both)
for term in ("place order", "orders", "items", "employees", "O2E", "O2O"):
    absent(f"entity/filter {term!r} still withheld", term, both)
empty = _build_structure_hint({"nodes": [box({}, {}, [], [], [])], "edgeNames": []},
                              "skeleton+constraints")
check("a node with no constraints gets an empty list",
      scaffold_of(empty)["nodes"][0]["Box"][0]["constraints"], [])

print("\n--- the instruction tells the model what is fixed ---")
for phrase in ("Do not add, remove or renumber", "edgeNames", "variable keys",
               'No "<FILL...>" may remain'):
    check(f"instruction mentions {phrase!r}", phrase in skel, True)
check("skeleton level says constraints are not given",
      "constraints are NOT given" in skel, True)
check("constraints level says types are given",
      "TYPE is given" in both, True)

print("\n--- inertness and bad input: degrade, never raise ---")
check("unknown level -> None", _build_structure_hint(GOLD, "everything"), None)
check("None level -> None", _build_structure_hint(GOLD, None), None)
check("gold is None -> None", _build_structure_hint(None, "skeleton"), None)
check("gold is a string -> None", _build_structure_hint("{}", "skeleton"), None)
check("no nodes key -> None", _build_structure_hint({"edgeNames": []}, "skeleton"), None)
check("empty nodes -> None", _build_structure_hint({"nodes": []}, "skeleton"), None)
check("nodes present but none are Boxes -> None",
      _build_structure_hint({"nodes": [{"NotABox": []}], "edgeNames": []}, "skeleton"), None)
odd = _build_structure_hint({"nodes": [{"Box": [{"newObjectVars": {"0": ["x"]}}]}],
                             "edgeNames": []}, "skeleton")
check("a Box with no child list still renders", scaffold_of(odd)["nodes"][0]["Box"][1], [])
absent("...and does not leak its entity", "x", scaffold_of(odd)["nodes"][0]["Box"][0]
       ["newObjectVars"]["0"][0].replace("<FILL: object type>", ""))

print("\n--- the level names the pipeline dispatches on ---")
check("exactly two levels", STRUCTURE_HINT_LEVELS, ("skeleton", "skeleton+constraints"))

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
