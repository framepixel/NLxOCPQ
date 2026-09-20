"""Forcing the root box's arity in code, and the normalisation that closes the notes-31 residue.

Two changes, tested together because they ship together.

**Forced root arity** is the experiment §25.7 deferred since the structure ablation. Asking the
model to adopt gold's shape did not work — 33% → 30% → 27% across the three rungs of notes 25 —
so this does not ask. The decoding grammar is tightened per case so the root box CAN ONLY declare
gold's number of variables. It is an ORACLE control: the count comes from gold, so any run using
it is a ceiling.

The feasibility gate here caught a real design error before any GPU time. Pinning
`minProperties`/`maxProperties` on `BindingBoxData` directly seems obvious and is wrong: that
definition is shared by every node, so it forces CHILD boxes to the root's arity too. 141 of the
169 gold queries have a child whose arity differs from its root, and the first version of this
gate rejected exactly those 141. The root therefore gets its own `$def`, reached positionally
through `prefixItems`, with `items` left generic so the node COUNT stays free.

**The normalisation** repairs the notes-31 residue: xgrammar's `any_order=True` lets a required
property be skipped when an optional one is present, and 10 of the 11 constrained failures were
one shape — a box carrying optional `labels` and omitting required `filters`. It is deliberately
narrow, and the test that matters most is the one asserting what it REFUSES to fill.

Run from the repo root:  python tests/test_forced_arity.py
"""
import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_testset import _gold_root_arity, root_node_index  # noqa: E402
from LLM import (  # noqa: E402
    _arity_pinned_schema,
    get_forced_root_arity,
    get_last_normalized_fields,
    set_forced_root_arity,
    validate_and_normalize_ocpq_json,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


try:
    import xgrammar as xgr
except ImportError:
    print("FAIL  xgrammar is not installed.")
    sys.exit(1)

accepts = xgr.testing._is_grammar_accept_string
TESTSET = json.load(open(REPO / "data" / "testset.json", encoding="utf-8"))

_cache = {}


def grammar_for(key):
    if key not in _cache:
        _cache[key] = xgr.Grammar.from_json_schema(
            json.dumps(_arity_pinned_schema(*key)), any_order=True, strict_mode=True)
    return _cache[key]


print("--- gold root arity is read correctly ---")
q = TESTSET[0]["ocpq_query"]
check("returns (root_index, n_event, n_object)", len(_gold_root_arity(q)), 3)
check("root index agrees with root_node_index", _gold_root_arity(q)[0], root_node_index(q))
for bad in (None, "{}", {}, {"nodes": []}, {"nodes": [{"NotABox": 1}]}):
    check(f"unreadable gold -> None ({str(bad)[:18]})", _gold_root_arity(bad), None)

print("\n--- THE GATE: every gold query is expressible under its own arity ---")
rejected = []
for i, entry in enumerate(TESTSET):
    gold = entry["ocpq_query"]
    text = json.dumps(gold, separators=(",", ":"))
    if not accepts(grammar_for(_gold_root_arity(gold)), text):
        rejected.append(i)
check(f"all {len(TESTSET)} gold queries accepted under their own arity", len(rejected), 0)
if rejected:
    print(f"        rejected {len(rejected)}: {rejected[:6]}")
print(f"        ({len(_cache)} distinct grammars for {len(TESTSET)} cases)")

print("\n--- ...and rejected under any OTHER arity, or the constraint is not binding ---")
leaked = []
for i, entry in enumerate(TESTSET):
    gold = entry["ocpq_query"]
    root, n_ev, n_ob = _gold_root_arity(gold)
    text = json.dumps(gold, separators=(",", ":"))
    for wrong in ((root, n_ev + 1, n_ob), (root, n_ev, n_ob + 1)):
        if accepts(grammar_for(wrong), text):
            leaked.append((i, wrong))
check("no gold query is accepted under a neighbouring arity", len(leaked), 0)
if leaked:
    print(f"        leaked: {leaked[:5]}")

print("\n--- the pin is on the ROOT only; children keep their own arity ---")
# The error the first version of this gate caught. 141 of 169 gold queries have a child whose
# arity differs from its root, so a grammar that pinned BindingBoxData globally rejected them all.
differing = 0
for entry in TESTSET:
    gold = entry["ocpq_query"]
    nodes = gold.get("nodes") or []
    root = root_node_index(gold)

    def arity_of(i):
        box = (nodes[i] or {}).get("Box")
        data = box[0] if isinstance(box, list) and box else {}
        return len(data.get("newEventVars") or {}), len(data.get("newObjectVars") or {})

    if any(arity_of(i) != arity_of(root) for i in range(len(nodes)) if i != root):
        differing += 1
check("most gold queries have a child of differing arity, and all still pass",
      differing > 100 and not rejected, True)
print(f"        ({differing}/{len(TESTSET)} have such a child)")
schema = _arity_pinned_schema(0, 0, 1)
check("the generic node def survives for non-root positions",
      schema["properties"]["nodes"]["items"], {"$ref": "#/$defs/BindingBoxTreeNodeBox"})
check("node count is NOT pinned -- this forces arity, not tree shape",
      "maxItems" in schema["properties"]["nodes"], False)

print("\n--- the setter, and inertness ---")
set_forced_root_arity(None)
check("None means free", get_forced_root_arity(), None)
set_forced_root_arity((0, 1, 2))
check("stores the triple", get_forced_root_arity(), (0, 1, 2))
set_forced_root_arity(None)

print("\n--- the normalisation fills only what has ONE possible meaning ---")
BOX_MISSING_FILTERS = {
    "nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                        "labels": [], "sizeFilters": [], "constraints": [],
                        "evVarLabels": {}, "obVarLabels": {}}, []]}],
    "edgeNames": [],
}
out = validate_and_normalize_ocpq_json(copy.deepcopy(BOX_MISSING_FILTERS))
check("a box missing `filters` now validates", out is not None, True)
check("...and it says what it filled", get_last_normalized_fields(), ["nodes[0].filters"])
check("the filled value is the empty list", (out or {}).get("nodes", [{}])[0]["Box"][0]["filters"],
      [])

complete = copy.deepcopy(BOX_MISSING_FILTERS)
complete["nodes"][0]["Box"][0]["filters"] = []
validate_and_normalize_ocpq_json(complete)
check("a complete query fills nothing", get_last_normalized_fields(), [])

print("\n--- ...and REFUSES the one that would invert a query's meaning ---")
# `SAT` over an EMPTY child set is satisfied vacuously, so defaulting `child_names` to [] turns a
# compliance query into one reporting 0% violations while looking correct. This is a recorded
# defect in this project, not a hypothetical.
SAT_MISSING_CHILD_NAMES = {
    "nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                        "filters": [], "sizeFilters": [],
                        "constraints": [{"type": "SAT"}],
                        "evVarLabels": {}, "obVarLabels": {}}, []]}],
    "edgeNames": [],
}
check("a SAT missing `child_names` is NOT rescued",
      validate_and_normalize_ocpq_json(copy.deepcopy(SAT_MISSING_CHILD_NAMES)), None)
check("...and nothing was filled", get_last_normalized_fields(), [])

ATTR_MISSING_NAME = {
    "nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                        "filters": [{"type": "ObjectAttributeValueFilter", "object": 0,
                                     "at_time": {"type": "Always"},
                                     "value_filter": {"type": "Integer", "min": 1, "max": None}}],
                        "sizeFilters": [], "constraints": [],
                        "evVarLabels": {}, "obVarLabels": {}}, []]}],
    "edgeNames": [],
}
check("a filter missing `attribute_name` is NOT rescued",
      validate_and_normalize_ocpq_json(copy.deepcopy(ATTR_MISSING_NAME)), None)

print("\n--- every gold query still validates unchanged ---")
touched = []
for i, entry in enumerate(TESTSET):
    if validate_and_normalize_ocpq_json(copy.deepcopy(entry["ocpq_query"])) is None:
        touched.append(("rejected", i))
    elif get_last_normalized_fields():
        touched.append(("modified", i, get_last_normalized_fields()))
check("gold is neither rejected nor modified by the normalisation", touched[:3], [])

print("\n--- the rung suffix is ONE definition: config_id, shard name, submit script ---")
# This drifted twice. First `config_id` carried the decoding rung and the shard name did not, so
# all three decoding rungs landed on `shard_m1_s0_e0_*`. Then the arity rung was added to
# `config_id` and missed in the shard name, so N1 and N2 both became
# `shard_m1_s0_e0_constrained_tok4000_*` -- and `--resume` for N1, which globs that prefix and
# takes the newest, would have silently resumed into N2's ORACLE directory. The submit script
# hard-codes the same strings to build its resume glob, so all three are pinned against each other.
from eval_testset import _decoding_suffix  # noqa: E402

RUNGS = {
    "N0": (False, False, None),
    "N1": (True, False, 4000),
    "N2": (True, True, 4000),
}
suffixes = {k: _decoding_suffix(*v) for k, v in RUNGS.items()}
# The typed ablation adds two NON-ORACLE flags, and its three rungs must be three conditions too.
TYPED = {
    "T0": dict(constrained_decoding=True, force_root_arity=False, generation_token_budget=4000),
    "T1": dict(constrained_decoding=True, force_root_arity=False, generation_token_budget=4000,
               type_aware_filters=True),
    "T2": dict(constrained_decoding=True, force_root_arity=False, generation_token_budget=4000,
               type_aware_filters=True, constant_root_arity=True),
}
typed_suffixes = {k: _decoding_suffix(**v) for k, v in TYPED.items()}
check("the typed rungs get three distinct suffixes", len(set(typed_suffixes.values())), 3)
check("T0 is the same condition as N1, so it is directly comparable",
      typed_suffixes["T0"], suffixes["N1"])
typed_script = (REPO / "slurm" / "submit_typed_ablation.sh").read_text(encoding="utf-8")
for rung, suffix in typed_suffixes.items():
    check(f"the typed submit script declares {rung}'s suffix ({suffix})",
          f'RUNG_SUFFIX="{suffix}"' in typed_script, True)
# The oracle flag and the constant must never both apply: gold would silently win.
check("force_root_arity suppresses the constant, so a rung cannot be secretly an oracle",
      _decoding_suffix(True, True, 4000, False, True), _decoding_suffix(True, True, 4000, False, False))
check("N0 keeps the published, unsuffixed name", suffixes["N0"], "")
check("N1 and N2 are distinguishable", suffixes["N1"] != suffixes["N2"], True)
check("all three suffixes are distinct", len(set(suffixes.values())), 3)
check("the oracle rung is identifiable from its suffix alone",
      "_arityforced" in suffixes["N2"], True)

script = (REPO / "slurm" / "submit_arity_ablation.sh").read_text(encoding="utf-8")
for rung, suffix in suffixes.items():
    declared = f'RUNG_SUFFIX="{suffix}"'
    check(f"submit script declares {rung}'s suffix exactly ({suffix or 'empty'})",
          declared in script, True)

# And the resume glob must use it, or a rung resumes into another rung's directory.
check("the resume glob is built from RUNG_SUFFIX", "${RUNG_SUFFIX}_*" in script, True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> arity is forced on the root only, and the normalisation stays narrow")
