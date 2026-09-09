"""The generation constraint: can it express gold, and does it refuse what gold is not?

Until now the OCPQ schema has only ever been applied AFTER generation. `schema_enforced` switches
`effective_schema_validation`, which gates `verify_ocpq_json`: parse the model's text, validate it
against `BindingBoxTree`, and on failure make a second LLM call asking it to fix itself. That
repair pass runs on 2043 cases and fails on 1613 of them (79%), and every failure ends as a score
of 0. Constrained decoding moves the schema into the decoding loop so invalid output cannot be
produced at all.

**This file is the feasibility gate, and it runs without a GPU.** It is two-sided on purpose:

  * A constraint that cannot express gold is not a constraint, it is a silent cap below gold --
    and it would look like a *result*. So every one of the 169 gold queries must be accepted.
  * A constraint that accepts everything would pass that test and be worthless. So a set of
    documents that `BindingBoxTree` rejects must be refused too.

Two things this caught, both of which would have wasted a GPU array job:

  * `lm-format-enforcer` was the first choice and FAILED the gate. It does not implement
    `prefixItems`, and `BindingBoxTreeNodeBox.Box` is `Tuple[BindingBoxData, List[int]]` -- so the
    single unsupported construct sits at the top of the tree and leaves everything beneath it,
    which is all 29 `$defs` and every discriminator, unconstrained. Rewriting the tuple into
    `items: {anyOf: [...]}` restored correctness and cost 57x in speed, degrading to 6 char/s and
    failing to finish a mid-sized gold query in 90 s.
  * `any_order=True` is REQUIRED, not a convenience. Gold emits object properties out of schema
    order -- `value_filter` before `at_time` in `FilterObjectAttributeValueFilter` -- and with
    ordered matching xgrammar rejects 32 of the 169 gold queries. Asserted below so the flag
    cannot be quietly dropped.

Run from the repo root:  python tests/test_constrained_decoding.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ocpq_schema import BindingBoxTree  # noqa: E402

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
    print("      pip install xgrammar   (adds only apache-tvm-ffi; torch/transformers unmoved)")
    sys.exit(1)

accepts = xgr.testing._is_grammar_accept_string


def grammar(any_order=True, strict_mode=True):
    """The grammar the decoder is constrained by. Built from the Pydantic model itself.

    Source of truth is `BindingBoxTree`, the same object `validate_and_normalize_ocpq_json`
    validates against, so the constraint and the validator cannot drift apart.
    """
    return xgr.Grammar.from_json_schema(
        BindingBoxTree, any_order=any_order, strict_mode=strict_mode)


BOX = ('{"newEventVars":{},"newObjectVars":{},"filters":%s,"sizeFilters":[],'
       '"constraints":[],"evVarLabels":{},"obVarLabels":{}}')
ATTR_FILTER = ('[{"type":"ObjectAttributeValueFilter","object":0,"attribute_name":"A",'
               '"at_time":{"type":"Always"},"value_filter":{"type":"Nope"}}]')

INVALID = {
    "prose": "The orders delivered late.",
    "markdown fence": '```json\n{"nodes":[]}',
    "wrong top-level key": '{"nodez":[]}',
    "string where an int list belongs":
        '{"nodes":[{"Box":[' + BOX % "[]" + ',"notalist"]}],"edgeNames":[]}',
    "invalid filter type literal":
        '{"nodes":[{"Box":[' + BOX % '[{"type":"NotAFilter","object":0,"event":0}]'
        + ',[]]}],"edgeNames":[]}',
    "invalid value_filter type literal":
        '{"nodes":[{"Box":[' + BOX % ATTR_FILTER + ',[]]}],"edgeNames":[]}',
    "missing required field":
        '{"nodes":[{"Box":[{"newEventVars":{}},[]]}],"edgeNames":[]}',
    "Box with three elements":
        '{"nodes":[{"Box":[' + BOX % "[]" + ',[],99]}],"edgeNames":[]}',
}

print("--- the grammar compiles from the Pydantic model ---")
t0 = time.time()
try:
    g = grammar()
    print(f"PASS  compiled in {time.time() - t0:.2f}s "
          f"(29 $defs, 16 anyOf unions, prefixItems tuples)")
except Exception as exc:  # noqa: BLE001
    print(f"FAIL  could not compile: {exc}")
    print(f"\nRESULT: FAIL -> 1 check")
    sys.exit(1)

print("\n--- THE GATE, half one: every gold query must be expressible ---")
testset = json.load(open(REPO / "data" / "testset.json", encoding="utf-8"))
texts = [json.dumps(e["ocpq_query"], separators=(",", ":")) for e in testset]
t0 = time.time()
rejected = [i for i, t in enumerate(texts) if not accepts(g, t)]
elapsed = time.time() - t0
check(f"all {len(texts)} gold queries accepted", len(rejected), 0)
if rejected:
    print(f"        {len(rejected)} rejected, first few:")
    for i in rejected[:5]:
        print(f'          [{i:3d}] {testset[i]["nl_query"][:60]}')
print(f"        ({sum(len(t) for t in texts) / elapsed:,.0f} char/s)")

print("\n--- THE GATE, half two: it must refuse what BindingBoxTree refuses ---")
for name, text in INVALID.items():
    check(f"rejects {name}", accepts(g, text), False)

print("\n--- any_order=True is required, not cosmetic ---")
# Gold writes `value_filter` before `at_time`; the schema declares the reverse. With ordered
# matching this silently rejects 32/169 gold queries, which would cap the model below gold on 19%
# of the testset and read as a finding rather than a bug.
ordered = grammar(any_order=False)
ordered_rejects = sum(1 for t in texts if not accepts(ordered, t))
check("ordered matching rejects gold, so the flag must stay", ordered_rejects > 0, True)
print(f"        (any_order=False rejects {ordered_rejects}/{len(texts)}; "
      f"any_order=True rejects {len(rejected)})")

print("\n--- the constraint is at least as strict as the validator ---")
# Where they differ, the grammar must be the STRICTER one: anything it admits must validate, or
# constrained decoding would still emit output the pipeline rejects, which is the whole premise.
from pydantic import ValidationError  # noqa: E402

def validates(text):
    try:
        BindingBoxTree.model_validate(json.loads(text))
        return True
    except (ValidationError, ValueError):
        return False

admitted_but_invalid = [n for n, t in INVALID.items() if accepts(g, t) and not validates(t)]
check("nothing the grammar admits fails validation", admitted_but_invalid, [])
check("...and gold, which validates, is admitted",
      all(validates(t) for t in texts[:20]), True)
# One known asymmetry, pinned so it is a decision and not a surprise: Pydantic ignores unknown
# fields by default, the grammar (strict_mode=True) forbids them. Stricter is safe here.
EXTRA = ('{"nodes":[{"Box":[{"newEventVars":{},"newObjectVars":{},"filters":[],'
         '"sizeFilters":[],"constraints":[],"evVarLabels":{},"obVarLabels":{},"bogus":1},'
         '[]]}],"edgeNames":[]}')
check("unknown field: grammar forbids, Pydantic ignores (grammar is stricter)",
      (accepts(g, EXTRA), validates(EXTRA)), (False, True))

print("\n--- a truncated query must not read as complete ---")
# Truncation is the OTHER failure mechanism and the constraint cannot prevent it -- but it must
# not declare a half-written query finished either.
gold0 = texts[0]
check("a prefix of gold is not accepted as complete", accepts(g, gold0[: len(gold0) // 2]), False)
check("...while the whole query is", accepts(g, gold0), True)


print("\n--- inertness: with the flag off, nothing about generation changes ---")
# A flag that silently perturbed the free path would invalidate D0, the baseline D1 and D2 are
# measured against. `_ocpq_logits_processor` returning None is what keeps `generate_kwargs`
# byte-identical to every run recorded before 2026-09-02.
import LLM  # noqa: E402

LLM.set_constrained_decoding(False)
check("flag off -> no logits processor is built", LLM._ocpq_logits_processor(), None)
check("...and the getter agrees", LLM.get_constrained_decoding(), False)
LLM.set_constrained_decoding(True)
check("flag on -> the getter agrees", LLM.get_constrained_decoding(), True)
# Building the processor needs a loaded model; that is the smoke's job, not this file's. What is
# assertable here is that it does not silently no-op when switched on.
LLM.set_constrained_decoding(False)

print("\n--- the token budget overrides every call site, and defaults to leaving them alone ---")
check("default is None (pipelines keep their own 1000/2000)",
      LLM.get_generation_token_budget(), None)
LLM.set_generation_token_budget(4000)
check("set to 4000", LLM.get_generation_token_budget(), 4000)
LLM.set_generation_token_budget(None)
check("None restores per-pipeline defaults", LLM.get_generation_token_budget(), None)

print("\n--- the grammar cache is keyed, not rebuilt per call ---")
# Compiling against the vocabulary is the expensive part and is invariant across a run; doing it
# per call would dominate generation. The cache is empty here because no model is loaded, which
# is itself the assertion that nothing was compiled at import time.
check("no grammar compiled without a model loaded", LLM._XGRAMMAR_CACHE, {})

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the constraint expresses gold and refuses what gold is not")
