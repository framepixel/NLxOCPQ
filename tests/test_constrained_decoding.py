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
    cannot be quietly dropped. (Qualified 2026-09-25, notes 43.7: that is gold AS WRITTEN. In
    schema order the ordered grammar admits all 169, and it forbids the repeated key any_order
    admits, so the agentic arm sets `constrained_any_order: false`. Asserted at the end.)

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
check("ordered matching rejects gold AS WRITTEN (the default stays; see the end)",
      ordered_rejects > 0, True)
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


print("\n--- the rung must be part of the CONDITION's identity, not just the shard ---")
# The three rungs share model, strategy and schema setting. If the decoding settings do not reach
# `config_id`, all 507 records collapse onto 169 distinct `_record_key`s and `--merge-shards`
# drops two thirds as duplicates -- silently, and choosing survivors by glob order. That is the
# same failure the 96-shard matrix hit on 2026-08-27. The smoke reproduced it: D0 and D2 produced
# byte-identical `config_id`s and 11 of 25 records collided.
import eval_testset as ET  # noqa: E402


def config_id_for(constrained, budget):
    """Mirror of the construction in `run_evaluation`, kept in step by the assertions below."""
    suffix = ""
    if constrained:
        suffix += "_constrained"
    if budget is not None:
        suffix += f"_tok{budget}"
    return ET._slug(f"llama31-8b-finetuned_schema-allowed-entities-only_schema-on{suffix}")


rung_ids = [config_id_for(False, None), config_id_for(False, 4000), config_id_for(True, 4000)]
check("the three rungs get three distinct config_ids", len(set(rung_ids)), 3)
check("D0 keeps the published id unchanged",
      rung_ids[0], "llama31-8b-finetuned_schema-allowed-entities-only_schema-on")
check("D1 is distinguishable", rung_ids[1].endswith("_tok4000"), True)
check("D2 records that it was constrained", "_constrained" in rung_ids[2], True)

# ...and therefore the merge key separates them, which is the property that actually matters.
def key(config_id):
    return ET._record_key(1, config_id, 1, "Which orders were late?", "/logs/order-management.json")

check("distinct merge keys, so nothing is dropped as a duplicate",
      len({key(c) for c in rung_ids}), 3)

print("\n--- KNOWN LIMITATION: any_order lets a required field be skipped ---")
# This is why the full run came out 158/169 schema-valid at D2 rather than 169/169, and 10 of
# those 11 failures are this one shape. xgrammar's `any_order=True` permits a REQUIRED property to
# be omitted when an OPTIONAL one is present: the model emitted `labels` (optional, default []) and
# dropped `filters` (required). Asserted rather than described, so the day it is fixed upstream
# this test fails and tells us the residue can be reclaimed.
#
# It cannot be closed from the schema side, and the alternatives are worse -- all measured:
#   any_order=False                -> rejects 32/169 gold outright
#   every property required        -> rejects gold; 412 of 424 gold boxes omit `labels`
#   drop optional properties       -> rejects 167/169 gold (qualifier, min_seconds, min/max)
# The fix is a normalisation in LLM.py -- an absent list-typed `filters` is `[]` under OCPQ
# semantics -- not a grammar setting. See notes 31.4.
WITH_OPTIONAL_NO_REQUIRED = (
    '{"nodes":[{"Box":[{"newEventVars":{},"newObjectVars":{},"labels":[],"sizeFilters":[],'
    '"constraints":[],"evVarLabels":{},"obVarLabels":{}},[]]}],"edgeNames":[]}')
check("the hole is still open (grammar admits it)", accepts(g, WITH_OPTIONAL_NO_REQUIRED), True)
check("...and Pydantic still rejects it, which is what makes it a hole",
      validates(WITH_OPTIONAL_NO_REQUIRED), False)
check("without the optional field present, the same omission IS caught",
      accepts(g, '{"nodes":[{"Box":[{"newEventVars":{},"newObjectVars":{},"sizeFilters":[],'
                 '"constraints":[],"evVarLabels":{},"obVarLabels":{}},[]]}],"edgeNames":[]}'),
      False)
# The residue is bounded: it needs an optional property, and BindingBoxData has exactly one.
opt = [k for k in BindingBoxTree.model_json_schema()["$defs"]["BindingBoxData"]["properties"]
       if k not in BindingBoxTree.model_json_schema()["$defs"]["BindingBoxData"]["required"]]
check("BindingBoxData has exactly one optional property, so the hole is narrow", opt, ["labels"])


print("\n--- the arity rung is a third condition, not a relabelled second ---")
# Same collision risk as the decoding rungs: N1 and N2 share model, strategy, schema setting AND
# constrained_decoding, differing only in `force_root_arity`. Without that in `config_id` the
# merge would drop one of them as a duplicate.
def config_id_for_arity(constrained, forced, budget):
    suffix = ""
    if constrained:
        suffix += "_constrained"
    if forced:
        suffix += "_arityforced"
    if budget is not None:
        suffix += f"_tok{budget}"
    return ET._slug(f"llama31-8b-finetuned_schema-allowed-entities-only_schema-on{suffix}")

arity_ids = [config_id_for_arity(False, False, None),
             config_id_for_arity(True, False, 4000),
             config_id_for_arity(True, True, 4000)]
check("N0/N1/N2 get three distinct config_ids", len(set(arity_ids)), 3)
check("N0 is still the published baseline id",
      arity_ids[0], "llama31-8b-finetuned_schema-allowed-entities-only_schema-on")
check("the oracle rung is identifiable from its id alone",
      "_arityforced" in arity_ids[2], True)
check("three distinct merge keys", len({key(c) for c in arity_ids}), 3)

print("\n--- the agent's TURN grammar: exactly one well-formed action per turn ---")
# notes 43.5. The loop compiles AgentTurn / AgentTurnWithThought (a union: one variant per tool plus
# a final answer), with the exact flags LLM uses. The gate smoke of 2026-09-23 showed what the
# earlier all-optional AgentAction grammar admitted: a final_tree AND a free-form args.tree in one
# object, running to the 4000-token limit -- the cause of both timeouts. Each claim is measured here.
from ocpq_schema import AgentTurn, AgentTurnWithThought  # noqa: E402

turn_g = xgr.Grammar.from_json_schema(AgentTurn, any_order=True, strict_mode=True)
thought_g = xgr.Grammar.from_json_schema(AgentTurnWithThought, any_order=True, strict_mode=True)
ONE_BOX = ('{"nodes":[{"Box":[{"newEventVars":{},"newObjectVars":{"0":["orders"]},"filters":[],'
           '"sizeFilters":[],"constraints":[],"evVarLabels":{},"obVarLabels":{}},[]]}],'
           '"edgeNames":[]}')
TOOL = '{"tool":"sample","args":{"type":"orders"}}'
THOUGHT_TOOL = '{"thought":"orders first","tool":"sample","args":{"type":"orders"}}'
check("accepts a typed tool call", accepts(turn_g, TOOL), True)
check("accepts a try_query carrying a real tree",
      accepts(turn_g, '{"tool":"try_query","args":{"tree":' + ONE_BOX + '}}'), True)
check("accepts a final answer", accepts(turn_g, '{"final_tree":' + ONE_BOX + '}'), True)
check("accepts optional arguments left out",
      accepts(turn_g, '{"tool":"inspect_situations","args":{}}'), True)
check("REJECTS a final_tree and a tool call in one object (the smoke's runaway shape)",
      accepts(turn_g, '{"final_tree":' + ONE_BOX + ',"tool":"try_query","args":{"tree":'
              + ONE_BOX + '}}'), False)
check("REJECTS free-form arguments (args was Dict[str, Any])",
      accepts(turn_g, '{"tool":"sample","args":{"type":"orders","tree":{"nodes":[]}}}'), False)
check("REJECTS a try_query tree that is not a BindingBoxTree",
      accepts(turn_g, '{"tool":"try_query","args":{"tree":{"nodes":"x"}}}'), False)
check("REJECTS an unknown tool name", accepts(turn_g, '{"tool":"summon","args":{}}'), False)
check("REJECTS a sample without its required type",
      accepts(turn_g, '{"tool":"sample","args":{}}'), False)
check("the plain grammar REJECTS a thought (strict_mode) -- the off condition stays clean",
      accepts(turn_g, THOUGHT_TOOL), False)
check("the thought grammar accepts a thought beside a tool call", accepts(thought_g, THOUGHT_TOOL),
      True)
check("the thought grammar still accepts a turn without one (it is optional)",
      accepts(thought_g, TOOL), True)
check("the thought grammar accepts a thought after the action (any_order)",
      accepts(thought_g, '{"tool":"sample","args":{"type":"o"},"thought":"x"}'), True)

print("\n--- THE ORDERED GRAMMAR: no repeated keys, nothing gold needs is lost (notes 43.7) ---")
# The 2026-09-24 gate smoke on BASE Llama 3.1 8B timed out 10/10 cases: 58 of 70 turns ran to the
# 4000-token limit repeating one key ("filters": [], "filters": [], ...; "labels": [] x hundreds;
# "type": "String" x hundreds). any_order=True admits that; the ordered grammar does not. The
# agentic configs set `constrained_any_order: false`; every other config keeps the default.
DUP_FILTERS = ONE_BOX.replace('"filters":[],', '"filters":[],"filters":[],"filters":[],')
DUP_LABELS = ONE_BOX.replace('"obVarLabels":{}}', '"obVarLabels":{},"labels":[],"labels":[]}')
TBE = ('{"type":"Filter","filter":{"type":"TimeBetweenEvents","from_event":0,"to_event":1,'
       '"min_seconds":0,"max_seconds":0%s}}')
ONE_TBE = ONE_BOX.replace('"constraints":[]', '"constraints":[' + TBE % "" + ']')
DUP_TBE = ONE_BOX.replace('"constraints":[]',
                          '"constraints":[' + TBE % ',"min_seconds":0,"max_seconds":0' + ']')
ordered_tree = grammar(any_order=False)
check("any_order ADMITS a repeated key -- the hole the base model looped in (pinned)",
      [accepts(g, t) for t in (DUP_FILTERS, DUP_LABELS, DUP_TBE)], [True, True, True])
check("the ordered grammar REFUSES every smoke shape",
      [accepts(ordered_tree, t) for t in (DUP_FILTERS, DUP_LABELS, DUP_TBE)], [False, False, False])
check("...and accepts the same trees written once", [accepts(ordered_tree, t) for t in
                                                     (ONE_BOX, ONE_TBE)], [True, True])
# The 32 rejections above are gold AS WRITTEN. A constrained model writes in the grammar's order,
# so what matters is whether every gold TREE is expressible -- i.e. gold re-serialised in schema
# order. If this ever drops below all, the ordered grammar caps the model below gold.
for label, path in (("testset", "data/testset.json"), ("dev set", "data/agent_devset.json")):
    entries = json.load(open(REPO / path, encoding="utf-8"))
    in_order = [json.dumps(BindingBoxTree.model_validate(e["ocpq_query"])
                           .model_dump(by_alias=True, mode="json"), separators=(",", ":"))
                for e in entries]
    check(f"every {label} gold tree, in schema order, is admitted by the ordered grammar",
          [i for i, t in enumerate(in_order) if not accepts(ordered_tree, t)], [])

turn_ord = xgr.Grammar.from_json_schema(AgentTurn, any_order=False, strict_mode=True)
thought_ord = xgr.Grammar.from_json_schema(AgentTurnWithThought, any_order=False,
                                           strict_mode=True)
check("ordered turn grammar: a tool call, a try_query and a final answer are admitted",
      [accepts(turn_ord, TOOL), accepts(turn_ord, '{"tool":"try_query","args":{"tree":'
                                        + ONE_BOX + '}}'),
       accepts(turn_ord, '{"final_tree":' + ONE_BOX + '}')], [True, True, True])
check("ordered turn grammar: a repeated key inside a try_query tree is REFUSED",
      accepts(turn_ord, '{"tool":"try_query","args":{"tree":' + DUP_FILTERS + '}}'), False)
check("ordered turn grammar: a repeated key in the turn itself is REFUSED",
      accepts(turn_ord, '{"tool":"sample","tool":"sample","args":{"type":"orders"}}'), False)
check("ordered thought grammar: the thought comes FIRST, before the action it informs",
      accepts(thought_ord, THOUGHT_TOOL), True)
check("...and not after it (the variants declare `thought` first, ocpq_schema)",
      accepts(thought_ord, '{"tool":"sample","args":{"type":"o"},"thought":"x"}'), False)
check("ordered thought grammar: the thought stays optional", accepts(thought_ord, TOOL), True)

LLM.set_constrained_any_order(None)
check("the setter's default is any_order (every recorded run)", LLM.get_constrained_any_order(),
      True)
LLM.set_constrained_any_order(False)
check("...and it can be switched off", LLM.get_constrained_any_order(), False)
LLM.set_constrained_any_order(None)
_llm_src = (REPO / "src" / "LLM.py").read_text(encoding="utf-8")
check("the decoder compiles with the flag, not a constant",
      "from_json_schema(source, any_order=constrained_any_order, strict_mode=True" in _llm_src,
      True)
check("...and the grammar cache is keyed by it (an any_order grammar is never reused)",
      '"__name__", None), constrained_any_order,' in _llm_src, True)

print("\n--- THE BOUNDED GRAMMAR: no free whitespace, bounded tree lists (notes 43.8) ---")
# The 2026-09-26 smoke (ordered grammar, base Llama 3.1 8B): 8/10 cases still timed out. One turn
# wrote 398 visible characters and ~3900 tokens of blanks; others repeated list items (one empty
# box x60, a child list [0, 0, 0, ...]). Built with LLM's own transform, so what is tested here is
# what the decoder compiles.
bounded_tree_schema = LLM._bounded_schema(BindingBoxTree.model_json_schema())
bounded_turn_schema = LLM._bounded_schema(AgentTurn.model_json_schema())
bounded_thought_schema = LLM._bounded_schema(AgentTurnWithThought.model_json_schema())


def bounded(schema):
    return xgr.Grammar.from_json_schema(json.dumps(schema), any_order=False, strict_mode=True,
                                        any_whitespace=False)


b_tree, b_turn, b_thought = (bounded(bounded_tree_schema), bounded(bounded_turn_schema),
                             bounded(bounded_thought_schema))


def placed_bounds(schema):
    """Each bound read back from the path it must sit at -- a count would pass on Pydantic's own
    tuple `maxItems: 2` without any bound having been placed."""
    d = schema["$defs"]
    tree = d.get("BindingBoxTree", schema)
    box = d["BindingBoxData"]["properties"]
    child_names = {name: dfn["properties"]["child_names"].get("maxItems")
                   for name, dfn in d.items() if "child_names" in dfn.get("properties", {})}
    return {"nodes": tree["properties"]["nodes"].get("maxItems"),
            "edgeNames": tree["properties"]["edgeNames"].get("maxItems"),
            **{k: box[k].get("maxItems") for k in ("filters", "sizeFilters", "constraints",
                                                    "labels")},
            "children": d["BindingBoxTreeNodeBox"]["properties"]["Box"]["prefixItems"][1]
            .get("maxItems"),
            "child_names": set(child_names.values()) if child_names else None}


_expected = dict(LLM.OCPQ_TREE_MAX_ITEMS, child_names={LLM.OCPQ_TREE_MAX_ITEMS["child_names"]})
check("every bound sits at its path in the tree schema", placed_bounds(bounded_tree_schema),
      _expected)
check("...and in the turn union's embedded tree (same definitions)",
      placed_bounds(bounded_turn_schema), _expected)
_before = BindingBoxTree.model_json_schema()
_snapshot = json.dumps(_before, sort_keys=True)
LLM._bounded_schema(_before)
check("the transform leaves its input alone (it returns a copy)",
      json.dumps(_before, sort_keys=True), _snapshot)
for label, path in (("testset", "data/testset.json"), ("dev set", "data/agent_devset.json")):
    trees = [json.dumps(BindingBoxTree.model_validate(e["ocpq_query"])
                        .model_dump(by_alias=True, mode="json"))
             for e in json.load(open(REPO / path, encoding="utf-8"))]
    check(f"every {label} gold tree passes the bounded tree grammar",
          [i for i, t in enumerate(trees) if not accepts(b_tree, t)], [])
    check(f"...as a final answer and as a try_query in the bounded turn grammar",
          [i for i, t in enumerate(trees)
           if not (accepts(b_turn, '{"final_tree": ' + t + '}')
                   and accepts(b_turn, '{"tool": "try_query", "args": {"tree": ' + t + '}}'))],
          [])
SP_BOX = ('{"newEventVars": {}, "newObjectVars": {}, "filters": [], "sizeFilters": [], '
          '"constraints": [], "evVarLabels": {}, "obVarLabels": {}}')
SP_NODE = '{"Box": [' + SP_BOX + ', []]}'


def tree_of(nodes=1, children="[]", box=SP_BOX, gap=""):
    node = '{"Box": [' + box + ', ' + children + ']}'
    return '{"nodes": [' + ", ".join([node] * nodes) + '],' + (gap or " ") + '"edgeNames": []}'


check("a clean tree with standard separators is admitted", accepts(b_tree, tree_of()), True)
check("REFUSES the whitespace runaway (the smoke's blanks)",
      accepts(b_tree, tree_of(gap=" " * 40 + "\n" * 10)), False)
check("...which the unbounded grammar admits (pinned: why the flag exists)",
      accepts(grammar(any_order=False),
              tree_of(gap=" " * 40 + "\n" * 10).replace(", ", ",").replace(": ", ":")
              .replace(",  ", ", ")), True)
check(f"nodes: {LLM.OCPQ_TREE_MAX_ITEMS['nodes']} admitted, one more REFUSED",
      (accepts(b_tree, tree_of(nodes=LLM.OCPQ_TREE_MAX_ITEMS["nodes"])),
       accepts(b_tree, tree_of(nodes=LLM.OCPQ_TREE_MAX_ITEMS["nodes"] + 1))), (True, False))
check("a child list [0, 0, 0, 0, 0] (the smoke's shape) is REFUSED",
      accepts(b_tree, tree_of(children="[0, 0, 0, 0, 0]")), False)
many_filters = SP_BOX.replace('"filters": []', '"filters": [' + ", ".join(
    ['{"type": "O2E", "object": 0, "event": 0}'] * (LLM.OCPQ_TREE_MAX_ITEMS["filters"] + 1)) + "]")
check("a filter list over its bound is REFUSED", accepts(b_tree, tree_of(box=many_filters)), False)
# notes 43.9: OCPQ refuses a variable name that is not a number in quotes; both base models wrote
# {"Event": [...]} and resubmitted the unparseable tree. All 652 gold variable keys are numeric.
for field in ("newEventVars", "newObjectVars"):
    named = SP_BOX.replace(f'"{field}": {{}}', f'"{field}": {{"Event": ["x"]}}')
    numbered = SP_BOX.replace(f'"{field}": {{}}', f'"{field}": {{"12": ["x"]}}')
    check(f"{field}: a word as a variable name is REFUSED, a number admitted",
          (accepts(b_tree, tree_of(box=named)), accepts(b_tree, tree_of(box=numbered))),
          (False, True))
for field in ("evVarLabels", "obVarLabels"):
    named = SP_BOX.replace(f'"{field}": {{}}', f'"{field}": {{"event": "EXCLUDED"}}')
    check(f"{field}: a word as a variable name is REFUSED (the smoke's shape)",
          accepts(b_tree, tree_of(box=named)), False)
# notes 43.10: the key rule must not cost the VALUE rule. The first version used `propertyNames`,
# with which xgrammar 0.2.5 drops the value schema -- the gate then saw {"1": {"type": "X"}} and
# label values outside the enum, and 87% of Llama's turns failed validation. Both directions, for
# every map the rule touches:
BAD_VALUES = {
    "newObjectVars": ('{"1": {"type": "X"}}', "an object where a list of type names belongs"),
    "newEventVars": ('{"1": "X"}', "a bare string where a list of type names belongs"),
    "obVarLabels": ('{"1": "DiagnosticEquipment"}', "a label outside IGNORED/INCLUDED/EXCLUDED"),
    "evVarLabels": ('{"0": ["X"]}', "a list where a label belongs"),
}
for field, (value, what) in BAD_VALUES.items():
    bad = tree_of(box=SP_BOX.replace(f'"{field}": {{}}', f'"{field}": {value}'))
    check(f"{field}: {what} is REFUSED by the bounded grammar",
          accepts(b_tree, bad), False)
    check(f"...and by the unbounded one (the bounded grammar may only be STRICTER)",
          accepts(grammar(any_order=False), bad.replace(", ", ",").replace(": ", ":")), False)
check("a valid label value under a numeric key is admitted",
      accepts(b_tree, tree_of(box=SP_BOX.replace('"obVarLabels": {}',
                                                   '"obVarLabels": {"1": "EXCLUDED"}'))), True)
check("the bounded schema carries the value schema under the key pattern (no propertyNames)",
      ("propertyNames" in json.dumps(bounded_tree_schema),
       all("patternProperties" in bounded_tree_schema["$defs"]["BindingBoxData"]["properties"][k]
           for k in ("newEventVars", "newObjectVars", "evVarLabels", "obVarLabels"))),
      (False, True))
check("...which the unbounded grammar admits (pinned: the schema allows any string)",
      accepts(grammar(any_order=False), tree_of(box=SP_BOX.replace(
          '"newEventVars": {}', '"newEventVars": {"Event": ["x"]}'))), True)
check("the bounded turn grammar refuses the runaway inside a try_query",
      accepts(b_turn, '{"tool": "try_query", "args": {"tree": '
              + tree_of(nodes=LLM.OCPQ_TREE_MAX_ITEMS["nodes"] + 1) + '}}'), False)
check("the bounded thought grammar still takes the thought first",
      accepts(b_thought, '{"thought": "orders first", "tool": "sample", "args": {"type": "o"}}'),
      True)
check("the decoder compiles with the flag (any_whitespace follows it)",
      "any_whitespace=not constrained_bounded_shape" in _llm_src, True)
check("...applies the bounds to whatever source it compiles",
      "source = json.dumps(_bounded_schema(as_dict))" in _llm_src, True)
check("...and caches by it", "constrained_any_order,\n           constrained_bounded_shape)"
      in _llm_src.replace("\r\n", "\n"), True)
check("off by default: every recorded run compiled the unbounded grammar",
      LLM.get_constrained_bounded_shape(), False)

print("\n--- STAGE 5 TURN GRAMMARS, compiled as the agentic configs compile them (notes 43.12) ---")
# check_tree in both loop rungs, and a final-only last turn. Compiled ordered + bounded, exactly as
# `_ocpq_logits_processor` does with the agentic configs' flags.
from ocpq_schema import PROBE_TOOL_NAMES, agent_turn_model  # noqa: E402

GOOD = tree_of()
CHECK = '{"tool": "check_tree", "args": {"tree": ' + GOOD + '}}'
FINAL = '{"final_tree": ' + GOOD + '}'
SAMPLE = '{"tool": "sample", "args": {"type": "orders", "n": 5}}'
for label, tools, thought, final_only, want in (
        ("tool rung + check_tree", PROBE_TOOL_NAMES + ("check_tree",), False, False,
         (True, True, True)),
        ("control rung: check_tree only", ("check_tree",), False, False, (True, True, False)),
        ("control rung + thought", ("check_tree",), True, False, (True, True, False)),
        ("reserved last turn", (), False, True, (False, True, False)),
        ("reserved last turn + thought", (), True, True, (False, True, False))):
    model = agent_turn_model(tools, thought, final_only)
    compiled = bounded(LLM._bounded_schema(model.model_json_schema()))
    check(f"{label} ({model.__name__}): check_tree / final / sample admitted",
          (accepts(compiled, CHECK), accepts(compiled, FINAL), accepts(compiled, SAMPLE)), want)
    if thought:
        check(f"{label}: the thought still comes first",
              accepts(compiled, FINAL.replace('{"final_tree"', '{"thought": "x", "final_tree"')),
              True)
check("check_tree carries a bounded tree too (the runaway shapes are refused inside it)",
      accepts(bounded(LLM._bounded_schema(agent_turn_model(("check_tree",)).model_json_schema())),
              '{"tool": "check_tree", "args": {"tree": '
              + tree_of(nodes=LLM.OCPQ_TREE_MAX_ITEMS["nodes"] + 1) + '}}'), False)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the constraint expresses gold and refuses what gold is not")
