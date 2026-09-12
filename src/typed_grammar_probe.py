#!/usr/bin/env python
"""Where does the type-aware constraint get lost -- the schema, the grammar, or the matcher?

The 2026-09-12 targeted smoke (`--case-shard 2/11`) failed the compliance gate. T2 ran with
`type_aware_filters=True` and still emitted the exact errors the constraint exists to forbid:

    case 11  Age (declared integer)       -> String{is_in:["40"]}
    case 12  isMarried (declared boolean) -> String{is_in:["1"]}
    case 13  Age                          -> String{is_in:["25"]}
    case 14  isMarried                    -> String{is_in:["1"]}

All four are in node 0 -- the ROOT box -- and the pre-registration says a non-zero
type-contradiction rate means the constraint is not applied and nothing downstream is
interpretable.

The schema layer has been cleared OFFLINE, without xgrammar:

  * the composed schema puts `Age` only in `FilterObjectAttributeValueFilter__integer`
    (value_filter: Integer|Float) and `isMarried` only in `__boolean`;
  * `BindingBoxDataRoot` -- the private root def that `_arity_pinned_schema` creates -- points at
    the four typed defs, with zero references left to the untyped one, so composition is correct;
  * `jsonschema` REJECTS all four emitted predictions against that schema and ACCEPTS the same
    document with `Integer{min:41}`.

So the constraint is expressible in JSON Schema and is not being enforced during generation. Every
compile logs this, five times:

    json_schema_converter.cc:829: Warning: oneOf with overlapping or non-provably-disjoint
    branches cannot be represented exactly; falling back to anyOf semantics

`filters` is a `oneOf` over 10 branches sharing `type`/`object`/`at_time`, and the four typed
branches differ only by an `attribute_name` enum and a `value_filter` $ref -- not provably
disjoint. The suspicion is that the relaxation lets `attribute_name` come from one branch and
`value_filter` from another, which is exactly the observed output.

`tests/test_type_aware_filters.py` asserted the opposite and passed, so it does NOT predict
production. This bisects the three layers to find where they diverge:

    A  jsonschema                            -- the schema itself         (expect REJECT)
    B  xgr.testing._is_grammar_accept_string -- what the test used        (test says REJECT)
    C  the compiled matcher                  -- what generation uses      (production says ACCEPT)

Run on the cluster, where xgrammar is installed. No GPU needed for A and B; C needs a tokenizer.

    python src/typed_grammar_probe.py
    python src/typed_grammar_probe.py --tokenizer meta-llama/Llama-3.1-8B-Instruct
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from LLM import _arity_pinned_schema, _type_aware_schema  # noqa: E402
from ocpq_schema import OCPQ_JSON_SCHEMA  # noqa: E402

RENTING = str((paths.REPO_ROOT / ".." / "Logs" / "renting_log_high.json").resolve())

# The exact shape the model emitted at T2, root box, arity (0 event, 1 object).
def document(value_filter, attribute="Age"):
    return {
        "nodes": [{"Box": [{
            "newEventVars": {}, "newObjectVars": {"0": ["Applicant"]},
            "filters": [{"type": "ObjectAttributeValueFilter", "object": 0,
                         "attribute_name": attribute, "at_time": {"type": "Always"},
                         "value_filter": value_filter}],
            "sizeFilters": [], "constraints": [], "labels": [],
            "evVarLabels": {}, "obVarLabels": {}}, []]}],
        "edgeNames": [],
    }

WRONG = document({"type": "String", "is_in": ["40"]})          # must be REJECTED
RIGHT = document({"type": "Integer", "min": 41, "max": None})  # must be ACCEPTED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=None,
                    help="HF id or local path; enables layer C (the compiled matcher).")
    ap.add_argument("--arity", default="0,0,1",
                    help="root_index,n_event,n_object as T2 forced it (default 0,0,1).")
    args = ap.parse_args()
    arity = tuple(int(x) for x in args.arity.split(","))

    schema = _type_aware_schema(_arity_pinned_schema(*arity), RENTING)
    schema_text = json.dumps(schema)
    results = {}

    print("=== A. jsonschema -- is the constraint in the schema at all? ===")
    try:
        import jsonschema
        for label, doc, want_valid in (("wrong type", WRONG, False), ("right type", RIGHT, True)):
            try:
                jsonschema.validate(doc, schema)
                got = True
            except jsonschema.ValidationError:
                got = False
            ok = got == want_valid
            results[f"A/{label}"] = got
            print(f"   {'OK ' if ok else 'BAD'} {label:12s} valid={got} (want {want_valid})")
    except ImportError:
        print("   jsonschema not installed; skipping (it is not a runtime dependency).")

    import xgrammar as xgr
    grammar = xgr.Grammar.from_json_schema(schema_text, any_order=True, strict_mode=True)

    print("\n=== B. xgr.testing._is_grammar_accept_string -- what the TEST used ===")
    accepts = xgr.testing._is_grammar_accept_string
    for label, doc, want_accept in (("wrong type", WRONG, False), ("right type", RIGHT, True)):
        got = accepts(grammar, json.dumps(doc, separators=(",", ":")))
        results[f"B/{label}"] = got
        print(f"   {'OK ' if got == want_accept else 'BAD'} {label:12s} accepted={got} "
              f"(want {want_accept})")

    print("\n=== C. the compiled matcher -- what GENERATION actually uses ===")
    if not args.tokenizer:
        print("   skipped: pass --tokenizer to run this layer.")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        info = xgr.TokenizerInfo.from_huggingface(tok)
        compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
        for label, doc, want_accept in (("wrong type", WRONG, False), ("right type", RIGHT, True)):
            text = json.dumps(doc, separators=(",", ":"))
            matcher = xgr.GrammarMatcher(compiled)
            # Token by token, the way generation drives it -- accepting the whole string at once
            # can take a different path through the library.
            got = all(matcher.accept_token(t) for t in tok.encode(text, add_special_tokens=False))
            results[f"C/{label}"] = got
            print(f"   {'OK ' if got == want_accept else 'BAD'} {label:12s} accepted={got} "
                  f"(want {want_accept})")

    print("\n=== VERDICT ===")
    wrong = {k: v for k, v in results.items() if k.endswith("wrong type")}
    leaking = [k.split("/")[0] for k, v in wrong.items() if v]
    if not leaking:
        print("   No layer accepts the wrong type. The constraint is enforceable, and the")
        print("   production failure is in the WIRING, not the grammar -- check that the logits")
        print("   processor reaches the generate() call on the path T2 took.")
    else:
        print(f"   Layers accepting the WRONG type: {', '.join(leaking)}")
        print("   The first such layer is where the constraint is lost. If B rejects and C")
        print("   accepts, the test is checking something generation never uses, and the")
        print("   correlation between `attribute_name` and `value_filter` is not representable")
        print("   in the compiled grammar -- a redesign, not a bug fix.")


if __name__ == "__main__":
    main()
