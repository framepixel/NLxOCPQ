#!/usr/bin/env python
"""Where does the type-aware constraint get lost -- the schema, the grammar, or the matcher?

CORRECTED 2026-09-15: the smoke described below did NOT fail because the constraint was lost. The
cluster held old copies of the logs, which declared `Age` and `isMarried` as strings, so the typed
grammar correctly allowed `String` there -- the gate was measuring the corpus (notes 35.1, 35.4). The
original description is kept as the record of what this probe was written to investigate.

The 2026-09-12 targeted smoke (`--case-shard 2/11`) appeared to fail the compliance gate. T2 ran with
`type_aware_filters=True` and still emitted the exact errors the constraint exists to forbid:

    case 11  Age (declared integer)       -> String{is_in:["40"]}
    case 12  isMarried (declared boolean) -> String{is_in:["1"]}
    case 13  Age                          -> String{is_in:["25"]}
    case 14  isMarried                    -> String{is_in:["1"]}

All four sit in node 0 -- the ROOT box. The pre-registration says a non-zero type-contradiction
rate means the constraint is not applied and nothing downstream is interpretable.

This bisects the three layers the constraint passes through, on the SAME document:

    A  jsonschema                            -- is it in the schema at all?
    B  xgr.testing._is_grammar_accept_string -- what tests/test_type_aware_filters.py used
    C  the compiled matcher                  -- what generation actually drives

Layer A is the control: a plain JSON Schema validator. If A does not reject the wrong type then
the composed SCHEMA is wrong and B and C are uninterpretable. Everything A depends on is printed
before the verdict -- the log path, whether the typed defs exist, which branch the attribute
landed in -- because the first cluster run of this probe disagreed with the same check run
offline, and an inverted control is a probe bug until proven otherwise.

Run on the cluster, where xgrammar is installed. A and B need no GPU and no tokenizer.

    python src/typed_grammar_probe.py                  # layers A and B
    python src/typed_grammar_probe.py --with-matcher   # adds layer C
"""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from LLM import _arity_pinned_schema, _type_aware_schema  # noqa: E402
from LogEvents import extract_ocel_schema  # noqa: E402

ATTRIBUTE = "Age"           # declared integer in renting_log_high
WRONG_VF = {"type": "String", "is_in": ["40"]}
RIGHT_VF = {"type": "Integer", "min": 41, "max": None}
UNTYPED_REF = '"#/$defs/FilterObjectAttributeValueFilter"'


def synthetic(value_filter, attribute=ATTRIBUTE, object_type="Applicant"):
    """A hand-written document, used only when no real prediction is available. A hand-written
    document can be invalid for reasons unrelated to the constraint under test. (This was once
    blamed for the cluster's inverted control; the cause was the cluster's stale corpus, notes 35.)"""
    return {
        "nodes": [{"Box": [{
            "newEventVars": {}, "newObjectVars": {"0": [object_type]},
            "filters": [{"type": "ObjectAttributeValueFilter", "object": 0,
                         "attribute_name": attribute, "at_time": {"type": "Always"},
                         "value_filter": value_filter}],
            "sizeFilters": [], "constraints": [], "labels": [],
            "evVarLabels": {}, "obVarLabels": {}}, []]}],
        "edgeNames": [],
    }


def _retype(document, attribute, value_filter):
    out = copy.deepcopy(document)
    for node in (out.get("nodes") or []):
        box = node.get("Box") if isinstance(node, dict) else None
        data = box[0] if isinstance(box, list) and box else None
        for flt in ((data or {}).get("filters") or []):
            if isinstance(flt, dict) and flt.get("attribute_name") == attribute:
                flt["value_filter"] = copy.deepcopy(value_filter)
    return out


def from_smoke(run_glob, attribute=ATTRIBUTE):
    """The prediction the model actually emitted, and the same document with the type corrected.
    Faithful in a way a hand-written document is not: it is known to be well-formed, because the
    pipeline already validated and executed it."""
    for pred_file in sorted(paths.REPO_ROOT.glob(run_glob)):
        try:
            records = json.loads(pred_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for record in records:
            prediction = record.get("prediction")
            if not isinstance(prediction, dict):
                continue
            for node in (prediction.get("nodes") or []):
                box = node.get("Box") if isinstance(node, dict) else None
                data = box[0] if isinstance(box, list) and box else None
                for flt in ((data or {}).get("filters") or []):
                    if isinstance(flt, dict) and flt.get("attribute_name") == attribute:
                        return (_retype(prediction, attribute, WRONG_VF),
                                _retype(prediction, attribute, RIGHT_VF),
                                f"{pred_file.parent.name} case {record.get('case_id')}")
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="OCEL log (default ../Logs/renting_log_high.json)")
    ap.add_argument("--arity", default="0,0,1", help="root_index,n_event,n_object (T2 used 0,0,1)")
    ap.add_argument("--with-matcher", action="store_true",
                    help="Also run layer C. Needs the eval tokenizer, resolved from the config.")
    ap.add_argument("--config", default="configs/eval_config.typed_ablation.json")
    ap.add_argument("--model-idx", type=int, default=1, help="As the submit script passes it.")
    ap.add_argument("--smoke-glob",
                    default="eval_runs/typed_ablation_smoke/*c2of11*typed*/predictions.json")
    args = ap.parse_args()
    arity = tuple(int(x) for x in args.arity.split(","))

    log = Path(args.log) if args.log else (paths.REPO_ROOT / ".." / "Logs" / "renting_log_high.json")
    log = log.resolve()

    print("=== what this probe is testing against ===")
    print(f"   log                : {log}")
    print(f"   log exists         : {log.exists()}")
    if not log.exists():
        print("   FATAL: the log is missing, so `_type_aware_schema` silently returns the schema")
        print("          UNCHANGED and every layer below is meaningless. Pass --log.")
        return 1

    declared = {}
    ocel = extract_ocel_schema(str(log))
    for kind in ("object_attributes", "event_attributes"):
        for attrs in (ocel.get(kind) or {}).values():
            for attr in attrs:
                declared[attr["name"]] = str(attr.get("type", "")).lower()
    print(f"   {ATTRIBUTE} declared as   : {declared.get(ATTRIBUTE)!r}")

    schema = _type_aware_schema(_arity_pinned_schema(*arity), str(log))
    typed_defs = sorted(k for k in schema["$defs"] if "AttributeValueFilter__" in k)
    print(f"   typed defs created : {len(typed_defs)} {typed_defs}")
    if not typed_defs:
        print("   FATAL: the rewrite produced no typed branches -- nothing below can reject.")
        return 1
    for name in typed_defs:
        enum = schema["$defs"][name].get("properties", {}).get("attribute_name", {}).get("enum", [])
        if ATTRIBUTE in enum:
            print(f"   {ATTRIBUTE} lives in      : {name}")
    root = schema["$defs"].get("BindingBoxDataRoot")
    if root is not None:
        blob = json.dumps(root["properties"]["filters"])
        print(f"   root filters union : typed refs={blob.count('AttributeValueFilter__')}, "
              f"untyped refs={blob.count(UNTYPED_REF)}")

    wrong, right, provenance = from_smoke(args.smoke_glob)
    if wrong is None:
        print(f"   documents          : SYNTHETIC (nothing matched {args.smoke_glob})")
        wrong, right = synthetic(WRONG_VF), synthetic(RIGHT_VF)
    else:
        print(f"   documents          : REAL, from {provenance}")

    schema_text = json.dumps(schema)
    results = {}

    print("\n=== A. jsonschema -- the control: is the constraint in the schema? ===")
    control_ok = True
    try:
        import jsonschema
        for label, doc, want in (("wrong type", wrong, False), ("right type", right, True)):
            try:
                jsonschema.validate(doc, schema)
                got, why = True, ""
            except jsonschema.ValidationError as exc:
                got, why = False, str(exc.message)[:100]
            results[f"A/{label}"] = got
            print(f"   {'OK ' if got == want else 'BAD'} {label:11s} valid={got!s:5s} "
                  f"(want {want})" + (f"  <- {why}" if why else ""))
        control_ok = (results.get("A/wrong type") is False
                      and results.get("A/right type") is True)
        if not control_ok:
            print("\n   CONTROL BROKEN. Stop here: B and C cannot be read until A rejects the")
            print("   wrong type and accepts the right one. Either the document is malformed for")
            print("   an unrelated reason (see the message above) or the composed schema is not")
            print("   what this probe assumes.")
    except ImportError:
        print("   jsonschema not installed; skipping the control.")

    import xgrammar as xgr
    grammar = xgr.Grammar.from_json_schema(schema_text, any_order=True, strict_mode=True)

    print("\n=== B. xgr.testing._is_grammar_accept_string -- what the TEST used ===")
    for label, doc, want in (("wrong type", wrong, False), ("right type", right, True)):
        got = xgr.testing._is_grammar_accept_string(grammar, json.dumps(doc, separators=(",", ":")))
        results[f"B/{label}"] = got
        print(f"   {'OK ' if got == want else 'BAD'} {label:11s} accepted={got!s:5s} (want {want})")

    print("\n=== C. the compiled matcher -- what GENERATION actually drives ===")
    if not args.with_matcher:
        print("   skipped: pass --with-matcher.")
    else:
        from expand_models import expand_models
        cfg = expand_models(json.loads((paths.REPO_ROOT / args.config).read_text(encoding="utf-8")))
        model_name = cfg["models"][args.model_idx]["name"]
        print(f"   tokenizer          : {model_name} (model-idx {args.model_idx})")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        compiled = xgr.GrammarCompiler(
            xgr.TokenizerInfo.from_huggingface(tok)).compile_grammar(grammar)
        for label, doc, want in (("wrong type", wrong, False), ("right type", right, True)):
            text = json.dumps(doc, separators=(",", ":"))
            matcher = xgr.GrammarMatcher(compiled)
            got = all(matcher.accept_token(t) for t in tok.encode(text, add_special_tokens=False))
            results[f"C/{label}"] = got
            print(f"   {'OK ' if got == want else 'BAD'} {label:11s} accepted={got!s:5s} "
                  f"(want {want})")

    print("\n=== VERDICT ===")
    if not control_ok:
        print("   Layer A did not behave as a control, so nothing below it is interpretable.")
        print("   Fix the probe or the schema first.")
        return 1
    leaking = sorted({k.split("/")[0] for k, v in results.items()
                      if k.endswith("wrong type") and v})
    if not leaking:
        print("   No layer accepts the wrong type. The constraint IS enforceable, so the")
        print("   production failure is WIRING: check that the logits processor reaches the")
        print("   generate() call on T2's path.")
    else:
        print(f"   Layers accepting the WRONG type: {', '.join(leaking)}")
        print("   B rejecting while C accepts means the test checks something generation never")
        print("   uses, and the attribute_name/value_filter correlation is not representable in")
        print("   the compiled grammar -- a redesign, not a bug fix.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
