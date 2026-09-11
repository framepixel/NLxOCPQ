"""Type-aware value filters: does the constraint block the model's error without blocking gold?

The model's most repeated semantic error is collapsing attribute comparisons into string
membership. `Age` is declared Integer and `build_schema_block` says so, yet "older than 40" comes
out as `String{is_in: ["40"]}` where gold has `Integer{min: 41}`. Across the 2026-09-11 run:
`Boolean -> String` 16 times, `Integer -> String` 11 times. Gold contradicts a declared type
**0 times in 117**.

The information was never missing, so this adds a constraint rather than knowledge -- and the
types come from the LOG SCHEMA, not gold, which is what makes it the first candidate for a
non-oracle working component. Everything in notes 32 that worked was an oracle.

The gate is two-sided, as in notes 30 and 32:

  * **gold must survive.** A constraint that blocks gold is a silent cap that reads as a result.
    Gold's zero contradictions say it should survive; this asserts it per query, per log.
  * **the observed errors must not.** A constraint that accepts everything would pass the first
    half and be worthless.

Plus the two things that make the encoding safe at all: grouping by declared TYPE rather than by
attribute (bpic2017 has 494 (event type, name) pairs but only 19 distinct names), and the fact
that no attribute name carries conflicting declared types in any log -- without which grouping by
name would be wrong.

Run from the repo root:  python tests/test_type_aware_filters.py
"""
import collections
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import paths  # noqa: E402
from eval_testset import _gold_root_arity, _resolve_ocel_path  # noqa: E402
from LLM import (  # noqa: E402
    VALUE_FILTER_FOR_DECLARED,
    _arity_pinned_schema,
    _type_aware_schema,
    get_type_aware_filters,
    set_type_aware_filters,
)
from LogEvents import extract_ocel_schema  # noqa: E402
from ocpq_schema import OCPQ_JSON_SCHEMA, BindingBoxTree  # noqa: E402

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
LOGS = sorted({_resolve_ocel_path(e["ocel_path"], paths.REPO_ROOT) for e in TESTSET})

_grammars = {}


def grammar_for(log, with_arity=None):
    key = (log, with_arity)
    if key not in _grammars:
        import copy
        base = (_arity_pinned_schema(*with_arity) if with_arity
                else copy.deepcopy(OCPQ_JSON_SCHEMA))
        _grammars[key] = xgr.Grammar.from_json_schema(
            json.dumps(_type_aware_schema(base, log)), any_order=True, strict_mode=True)
    return _grammars[key]


print("--- grouping by attribute NAME is only safe if names do not carry conflicting types ---")
conflicts = []
for log in LOGS:
    sch = extract_ocel_schema(log)
    for kind in ("object_attributes", "event_attributes"):
        by_name = collections.defaultdict(set)
        for attrs in (sch.get(kind) or {}).values():
            for a in attrs:
                by_name[a["name"]].add(str(a.get("type", "")).lower())
        conflicts += [(Path(log).name, n, sorted(t)) for n, t in by_name.items() if len(t) > 1]
check(f"no attribute name has conflicting declared types across {len(LOGS)} evaluation logs",
      conflicts[:3], [])

print("\n--- the grammar compiles for every evaluation log ---")
slowest = (0.0, None)
for log in LOGS:
    t0 = time.time()
    grammar_for(log)
    slowest = max(slowest, (time.time() - t0, Path(log).name))
check("every evaluation log compiles", len(_grammars), len(LOGS))
check("...and none is slow enough to matter at inference", slowest[0] < 5.0, True)
print(f"        (slowest: {slowest[1]} at {slowest[0]:.2f}s)")

print("\n--- THE GATE, half one: gold must survive the constraint ---")
rejected = []
for entry in TESTSET:
    log = _resolve_ocel_path(entry["ocel_path"], paths.REPO_ROOT)
    text = json.dumps(entry["ocpq_query"], separators=(",", ":"))
    if not accepts(grammar_for(log), text):
        rejected.append(entry["nl_query"][:60])
check(f"all {len(TESTSET)} gold queries accepted under their own log's type-aware grammar",
      rejected[:3], [])

print("\n--- THE GATE, half two: the observed errors must not ---")
# renting_log_high declares Age Integer, isMarried Boolean, isCitizen String. These are the exact
# three shapes seen in the run, not invented ones.
RENTING = _resolve_ocel_path("../Logs/renting_log_high.json", paths.REPO_ROOT)


def doc(attr, value_filter):
    return ('{"nodes":[{"Box":[{"newEventVars":{},"newObjectVars":{"0":["Person"]},'
            '"filters":[{"type":"ObjectAttributeValueFilter","object":0,"attribute_name":"'
            + attr + '","at_time":{"type":"Always"},"value_filter":' + value_filter + '}],'
            '"sizeFilters":[],"constraints":[],"evVarLabels":{},"obVarLabels":{}},[]]}],'
            '"edgeNames":[]}')


plain = xgr.Grammar.from_json_schema(BindingBoxTree, any_order=True, strict_mode=True)
CASES = [
    ("gold: Age Integer{min:41}", "Age", '{"type":"Integer","min":41}', True),
    ("the error: Age String{is_in}", "Age", '{"type":"String","is_in":["40"]}', False),
    ("gold: isMarried Boolean", "isMarried", '{"type":"Boolean","is_true":true}', True),
    ("the error: isMarried String", "isMarried", '{"type":"String","is_in":["1"]}', False),
    ("a genuinely String attribute", "isCitizen", '{"type":"String","is_in":["0"]}', True),
    ("Float on an Integer attribute is permitted", "Age", '{"type":"Float","min":41.0}', True),
]
for label, attr, value_filter, want in CASES:
    text = doc(attr, value_filter)
    got = accepts(grammar_for(RENTING), text)
    check(f"{'accepts' if want else 'rejects'} {label}", got, want)
    if not want:
        # The plain grammar must ADMIT it, or the new constraint is not what blocked it.
        check(f"   ...and the plain grammar admits it, so it is this constraint that blocks",
              accepts(plain, text), True)

print("\n--- an unknown attribute name is refused, not silently allowed ---")
check("rejects an attribute the log does not declare",
      accepts(grammar_for(RENTING), doc("NotARealAttribute", '{"type":"String","is_in":["x"]}')),
      False)

print("\n--- the two constraints compose ---")
# Arity pinning and type-awareness rewrite disjoint parts of the schema. A rung may carry both, so
# neither may break the other.
sample = next(e for e in TESTSET
              if _resolve_ocel_path(e["ocel_path"], paths.REPO_ROOT) == RENTING)
arity = _gold_root_arity(sample["ocpq_query"])
combined = grammar_for(RENTING, with_arity=arity)
check("gold survives arity pinning AND type-awareness together",
      accepts(combined, json.dumps(sample["ocpq_query"], separators=(",", ":"))), True)
check("...and the type error is still blocked under the combination",
      accepts(combined, doc("Age", '{"type":"String","is_in":["40"]}')), False)

print("\n--- inertness: with no log set, nothing changes ---")
set_type_aware_filters(None)
check("the setter defaults to off", get_type_aware_filters(), None)
set_type_aware_filters(RENTING)
check("...and stores the log when set", get_type_aware_filters(), RENTING)
set_type_aware_filters(None)
import copy as _copy  # noqa: E402
untouched = _type_aware_schema(_copy.deepcopy(OCPQ_JSON_SCHEMA), "/nonexistent/log.json")
check("an unreadable log leaves the schema alone rather than raising",
      untouched["$defs"]["BindingBoxData"]["properties"]["filters"],
      OCPQ_JSON_SCHEMA["$defs"]["BindingBoxData"]["properties"]["filters"])

print("\n--- the type mapping is the one the analysis assumed ---")
check("Integer admits Integer and Float", set(VALUE_FILTER_FOR_DECLARED["integer"]),
      {"ValueFilterInteger", "ValueFilterFloat"})
check("Boolean admits only Boolean", set(VALUE_FILTER_FOR_DECLARED["boolean"]),
      {"ValueFilterBoolean"})
check("String admits only String", set(VALUE_FILTER_FOR_DECLARED["string"]),
      {"ValueFilterString"})

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> gold survives, the observed errors do not, and the constraints compose")
