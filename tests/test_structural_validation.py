"""The pre-execution structural check: does it flag what ocpq_cli cannot evaluate, and no gold query?

2026-09-13: 88 of the three arity rungs' 507 executions were never evaluated. The tree passed
Pydantic, ocpq_cli's `evaluate_box_tree` returned an `Err`, and `run_evaluate` serialised that
`Result` into the export and exited 0 -- so the harness recorded success and the empty answer scored
0. Replayed through the CLI, 61 of the 68 distinct failing trees reported "Could not get oN": a
filter naming a variable no enclosing box declares.

`ocpq_schema.structural_defects` names such defects without running the query. Kinds are BLOCKING
(the Rust evaluator errors on them) or advisory (it tolerates them), and
`LLM._apply_structural_validation` spends one repair call only on blocking ones when the switch is
on. The split is from the evaluator's source and was checked against the 507 stored predictions:
advisory kinds appeared in 26 trees that evaluated normally.

The gate is two-sided, as for every constraint in this project:
  * GOLD MUST BE CLEAN. A check that flags a correct query triggers a repair that can only damage
    it -- the silent cap on gold this project has already caught twice.
  * THE DEFECTS MUST BE CAUGHT, one synthetic tree per kind, including the scoping rule the Rust
    evaluator applies: ancestors are in scope, siblings are not.

Plus the wiring: inert when off, no repair for advisory-only trees, one call for blocking ones, and
a repair kept only when it removes blocking defects -- and the rung names, pinned against the submit
script as for the other ablations.

Run from the repo root:  python tests/test_structural_validation.py
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ocpq_schema import (  # noqa: E402
    BLOCKING_DEFECT_KINDS,
    BindingBoxTree,
    blocking_defects,
    defect_kind,
    structural_defects,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def kinds(tree):
    return sorted({defect_kind(defect) for defect in structural_defects(tree)})


def box(ev=None, ob=None, filters=None, size=None, constraints=None, children=None):
    return {"Box": [{"newEventVars": ev or {}, "newObjectVars": ob or {},
                     "filters": filters or [], "sizeFilters": size or [],
                     "constraints": constraints or [], "evVarLabels": {}, "obVarLabels": {},
                     "labels": []}, children or []]}


def o2e(obj, ev):
    return {"type": "O2E", "object": obj, "event": ev, "qualifier": None, "filterLabel": None}


TESTSET = json.load(open(REPO / "data" / "testset.json", encoding="utf-8"))

print("--- THE GATE, half one: no gold query is flagged, not even advisorily ---")
flagged = [(i, structural_defects(e["ocpq_query"])) for i, e in enumerate(TESTSET)]
flagged = [(i, d) for i, d in flagged if d]
check(f"all {len(TESTSET)} gold queries have zero structural defects", flagged[:3], [])

# Root binds an order; its child binds a payment event and links it to the root's order.
VALID = {"nodes": [box(ob={"0": ["orders"]}, children=[1],
                       constraints=[{"type": "SAT", "child_names": ["A"]}]),
                   box(ev={"0": ["pay order"]}, filters=[o2e(0, 0)])],
         "edgeNames": [[[0, 1], "A"]]}

print("\n--- the reference tree is valid OCPQ and clean ---")
BindingBoxTree.model_validate(VALID)
check("no defects", structural_defects(VALID), [])
check("an ancestor's variable is in scope (the child reads the root's object 0)", kinds(VALID), [])

print("\n--- the blocking/advisory split follows what the evaluator does ---")
check("blocking kinds",
      sorted(BLOCKING_DEFECT_KINDS),
      ["child_out_of_range", "cycle", "malformed_tree", "multiple_parents", "root_count",
       "unbound_variable"])
check("defect_kind reads the tag", defect_kind("[root_count] 2 node(s) have no parent"),
      "root_count")
check("defect_kind tolerates an untagged string", defect_kind("no tag"), "")

print("\n--- THE GATE, half two: each defect is caught, with the right kind ---")
root_reads_child_event = copy.deepcopy(VALID)
root_reads_child_event["nodes"][0]["Box"][0]["filters"] = [o2e(0, 0)]
check("an O2E reading an event only a CHILD declares -- the dominant 'Could not get' shape",
      kinds(root_reads_child_event), ["unbound_variable"])

undeclared_object = copy.deepcopy(VALID)
undeclared_object["nodes"][1]["Box"][0]["filters"] = [o2e(5, 0)]
check("a filter reading an object nobody declares", kinds(undeclared_object), ["unbound_variable"])

sibling = {"nodes": [box(ob={"0": ["orders"]}, children=[1, 2]),
                     box(ev={"1": ["pay order"]}, filters=[o2e(0, 1)]),
                     box(ev={"2": ["place order"]}, filters=[o2e(0, 1)])],
           "edgeNames": [[[0, 1], "A"], [[0, 2], "B"]]}
BindingBoxTree.model_validate(sibling)
check("a SIBLING's variable is not in scope", kinds(sibling), ["unbound_variable"])
check("...and exactly the sibling reference is reported, not the legitimate one",
      len(structural_defects(sibling)), 1)

tbe = copy.deepcopy(VALID)
tbe["nodes"][1]["Box"][0]["filters"].append(
    {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 7, "min_seconds": None,
     "max_seconds": 60.0})
check("TimeBetweenEvents on an undeclared event is BLOCKING (check_binding errors)", kinds(tbe),
      ["unbound_variable"])

wrapped = copy.deepcopy(VALID)
wrapped["nodes"][0]["Box"][0]["constraints"].append({"type": "Filter", "filter": o2e(0, 3)})
check("a Filter-wrapped constraint reading an undeclared variable", kinds(wrapped),
      ["unbound_variable"])

for label, optional_filter in (
    ("an at-event timepoint", {"type": "ObjectAttributeValueFilter", "object": 0,
                               "attribute_name": "price", "at_time": {"type": "AtEvent", "event": 4},
                               "value_filter": {"type": "Float", "min": 1.0, "max": None}}),
    ("a NotEqual operand", {"type": "NotEqual", "var_1": {"Event": 0}, "var_2": {"Event": 9}}),
):
    tree = copy.deepcopy(VALID)
    tree["nodes"][1]["Box"][0]["filters"].append(optional_filter)
    check(f"{label} naming an undeclared variable is ADVISORY (looked up as optional)",
          kinds(tree), ["unbound_optional_reference"])

out_of_range = copy.deepcopy(VALID)
out_of_range["nodes"][0]["Box"][1] = [1, 5]
check("a CHILDREN-LIST index outside the tree is blocking",
      "child_out_of_range" in kinds(out_of_range), True)

stray_name = copy.deepcopy(VALID)
stray_name["edgeNames"].append([[1, 7], "B"])
check("an edgeNames entry pointing outside the tree is advisory (nobody walks it)",
      kinds(stray_name), ["edge_name_out_of_range"])

no_edge = copy.deepcopy(VALID)
no_edge["edgeNames"] = []
check("a child listed with no edgeNames entry is caught",
      "children_mismatch" in kinds(no_edge), True)

two_roots = copy.deepcopy(VALID)
two_roots["nodes"][0]["Box"][1] = []
two_roots["edgeNames"] = []
two_roots["nodes"][0]["Box"][0]["constraints"] = []
check("a node nobody lists as a child is a second root", "root_count" in kinds(two_roots), True)
check("...and its filter loses the root's object, which is reported too",
      "unbound_variable" in kinds(two_roots), True)

two_parents = {"nodes": [box(ob={"0": ["orders"]}, children=[1, 2]),
                         box(ev={"1": ["pay order"]}, filters=[o2e(0, 1)], children=[2]),
                         box(ev={"2": ["place order"]}, filters=[o2e(0, 2)])],
               "edgeNames": [[[0, 1], "A"], [[0, 2], "B"], [[1, 2], "C"]]}
check("a node with two parents", "multiple_parents" in kinds(two_parents), True)

cyclic = copy.deepcopy(VALID)
cyclic["nodes"][1]["Box"][1] = [0]
cyclic["edgeNames"].append([[1, 0], "B"])
check("a cycle", "cycle" in kinds(cyclic), True)

for label, where in (("SAT", "constraints"), ("NumChilds size filter", "sizeFilters"),
                     ("SizeFilter-wrapped NumChilds", "wrapped")):
    tree = copy.deepcopy(VALID)
    data = tree["nodes"][0]["Box"][0]
    if where == "constraints":
        data["constraints"] = [{"type": "SAT", "child_names": ["B"]}]
    elif where == "sizeFilters":
        data["sizeFilters"] = [{"type": "NumChilds", "child_name": "B", "min": 1, "max": None}]
    else:
        data["constraints"].append({"type": "SizeFilter", "filter": {
            "type": "NumChilds", "child_name": "B", "min": 1, "max": None}})
    check(f"a {label} naming a child that does not exist (advisory)", kinds(tree),
          ["unknown_child_name"])

check("blocking_defects keeps only blocking kinds",
      [defect_kind(d) for d in blocking_defects(structural_defects(two_roots))
       + blocking_defects(structural_defects(stray_name))],
      sorted({defect_kind(d) for d in structural_defects(two_roots)} & BLOCKING_DEFECT_KINDS))

print("\n--- model output is data: garbage is reported, never raised ---")
for garbage in (None, {}, {"nodes": "x"}, {"nodes": [{"NotABox": 1}]},
                {"nodes": [box()], "edgeNames": [["nonsense"]]}):
    try:
        result = structural_defects(garbage)
        check(f"no exception on {str(garbage)[:36]}", isinstance(result, list), True)
    except Exception as exc:  # noqa: BLE001
        check(f"no exception on {str(garbage)[:36]}", repr(exc), "a list")
check("deterministic", structural_defects(sibling) == structural_defects(copy.deepcopy(sibling)),
      True)

# Two real matrix_v2 predictions raised TypeError here (2026-09-22 census): a child name that is a
# list or a dict cannot be a set member. They are malformed trees, and are now reported as such.
unhashable_names = {
    "a list in SAT child_names": {"nodes": [box(constraints=[{"type": "SAT",
                                                              "child_names": [["A"]]}],
                                                children=[1]), box()],
                                  "edgeNames": [[[0, 1], "A"]]},
    "a dict as a NumChilds child_name": {"nodes": [box(size=[{"type": "NumChilds",
                                                              "child_name": {"n": "A"},
                                                              "min": 1, "max": None}],
                                                       children=[1]), box()],
                                         "edgeNames": [[[0, 1], "A"]]},
    "a list as an edge name": {"nodes": [box(children=[1]), box()],
                               "edgeNames": [[[0, 1], ["A"]]]},
}
for label, tree in unhashable_names.items():
    try:
        check(f"{label}: reported as malformed_tree, not raised", "malformed_tree" in kinds(tree),
              True)
    except TypeError as exc:
        check(f"{label}: reported as malformed_tree, not raised", repr(exc), "malformed_tree")
missing_name = {"nodes": [box(constraints=[{"type": "SAT", "child_names": [None]}], children=[1]),
                          box()], "edgeNames": [[[0, 1], "A"]]}
check("a MISSING name keeps its old advisory reading (only the crash inputs changed)",
      kinds(missing_name), ["unknown_child_name"])

print("\n--- the repair wiring: inert when off, blocking-only when on, kept only if it helps ---")
import LLM  # noqa: E402

check("off by default", LLM.get_structural_validation(), False)
calls = []


def fake_fix(model_name, messages, prompt, max_tokens=2000, schema_enforced=True):
    calls.append(prompt)
    return copy.deepcopy(fake_fix.result), "raw"


real_fix = LLM.schema_fix_with_llm
LLM.schema_fix_with_llm = fake_fix
try:
    LLM.set_structural_validation(False)
    given = copy.deepcopy(root_reads_child_event)
    check("off: the tree comes back as given",
          LLM._apply_structural_validation(given, "m", [], query="q") is given, True)
    check("off: no repair call", calls, [])
    check("off: no report", LLM.get_last_structural_report(), None)

    LLM.set_structural_validation(True)
    fake_fix.result = VALID
    LLM._apply_structural_validation(copy.deepcopy(VALID), "m", [], query="q")
    check("on, clean tree: no repair call", calls, [])
    check("on, clean tree: nothing found", LLM.get_last_structural_report()["defects_before"], [])

    advisory_only = copy.deepcopy(VALID)
    advisory_only["nodes"][0]["Box"][0]["constraints"] = [{"type": "SAT", "child_names": ["B"]}]
    given = copy.deepcopy(advisory_only)
    out = LLM._apply_structural_validation(given, "m", [], query="q")
    report = LLM.get_last_structural_report()
    check("on, advisory-only tree: NO repair call -- it evaluates, repairing could only damage it",
          calls, [])
    check("...the tree is kept as given", out is given, True)
    check("...but the advisory defect is recorded", [defect_kind(d) for d in report["defects_before"]],
          ["unknown_child_name"])
    check("...and nothing is blocking", report["blocking_before"], [])

    out = LLM._apply_structural_validation(copy.deepcopy(root_reads_child_event), "m", [],
                                           query="q")
    report = LLM.get_last_structural_report()
    check("on, blocking defect: exactly one repair call", len(calls), 1)
    check("the repair prompt names every blocking defect",
          all(d in calls[0] for d in blocking_defects(structural_defects(root_reads_child_event))),
          True)
    check("the repair prompt carries no gold -- only the tree and its defects",
          "testset" not in calls[0] and "gold" not in calls[0].lower(), True)
    check("a repair that removes the blocking defect is accepted", report["accepted"], True)
    check("...and replaces the tree", structural_defects(out), [])
    check("...with the after-state recorded", report["blocking_after"], [])

    calls.clear()
    fake_fix.result = root_reads_child_event
    given = copy.deepcopy(root_reads_child_event)
    out = LLM._apply_structural_validation(given, "m", [], query="q")
    check("a repair that fixes nothing is rejected", LLM.get_last_structural_report()["accepted"],
          False)
    check("...and the original tree is kept", out is given, True)

    fake_fix.result = None
    out = LLM._apply_structural_validation(given, "m", [], query="q")
    check("a repair that does not validate keeps the original", out is given, True)
    check("...and says why", bool(LLM.get_last_structural_report()["repair_error"]), True)

    def raising_fix(**kwargs):
        raise RuntimeError("CUDA out of memory")

    LLM.schema_fix_with_llm = raising_fix
    out = LLM._apply_structural_validation(given, "m", [], query="q")
    check("a repair that raises keeps the original and never fails the case", out is given, True)

    LLM.schema_fix_with_llm = fake_fix
    calls.clear()
    fake_fix.result = VALID
    LLM.verify_ocpq_json(copy.deepcopy(root_reads_child_event), "m", [], query="q")
    check("verify_ocpq_json routes a valid tree through the check when on", len(calls), 1)

    LLM.set_structural_validation(True)
    check("the per-case setter clears the previous case's report",
          LLM.get_last_structural_report(), None)
finally:
    LLM.schema_fix_with_llm = real_fix
    LLM.set_structural_validation(False)

print("\n--- rung names: one definition for config_id, shard name and submit script ---")
from eval_testset import _decoding_suffix  # noqa: E402

check("every published rung keeps its name",
      [_decoding_suffix(False, False, None), _decoding_suffix(True, False, 4000),
       _decoding_suffix(True, True, 4000), _decoding_suffix(True, False, 4000, True)],
      ["", "_constrained_tok4000", "_constrained_arityforced_tok4000",
       "_constrained_typed_tok4000"])
RUNGS = {
    "S0": _decoding_suffix(True, False, 4000),
    "S1": _decoding_suffix(True, False, 4000, structural_validation=True),
    "S2": _decoding_suffix(True, True, 4000),
    "S3": _decoding_suffix(True, True, 4000, structural_validation=True),
}
check("four distinct rungs", len(set(RUNGS.values())), 4)
check("S0 is N1's condition and S2 is N2's, so both are directly comparable",
      (RUNGS["S0"], RUNGS["S2"]), ("_constrained_tok4000", "_constrained_arityforced_tok4000"))
script = (REPO / "slurm" / "submit_structural_ablation.sh").read_text(encoding="utf-8")
for rung, suffix in RUNGS.items():
    check(f"the submit script declares {rung}'s suffix ({suffix})",
          f'RUNG_SUFFIX="{suffix}"' in script, True)
check("the resume glob looks in this ablation's own directory and uses the suffix",
      "eval_runs/structural_ablation/shard_" in script and "${RUNG_SUFFIX}_*" in script, True)
check("the submit script's mechanism check names the message the CLI actually prints",
      "Could not get" in script, True)
check("the submit script is LF, as bash on the cluster needs", "\r" in script, False)
config = json.load(open(REPO / "configs" / "eval_config.structural_ablation.json", encoding="utf-8"))
check("the ablation writes to its own directory", config.get("output_dir"),
      "eval_runs/structural_ablation")
main_config = json.load(open(REPO / "configs" / "eval_config.json", encoding="utf-8"))
check("the main matrix config does not switch it on",
      bool(main_config.get("structural_validation")), False)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> gold is clean, every defect kind is caught, and only blocking ones are repaired")
