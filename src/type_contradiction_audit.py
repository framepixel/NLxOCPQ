#!/usr/bin/env python
"""How often does a prediction give an attribute a value-filter its DECLARED type forbids?

This is the compliance gate for the type-aware rungs, and the measurement behind notes 34.3's
claim that the model "collapses attribute comparisons into string membership".

It must run where the run ran. On 2026-09-12 the same audit executed on a laptop against cluster
predictions reported 4 contradictions that did not exist: it resolved each record's log by
BASENAME into the local `Logs/`, and the local and cluster corpora disagreed about what `Age` was
declared to be. So every record is scored against the log it actually names, and the audit refuses
to proceed if that log is missing rather than substituting a similarly-named one:

    python src/type_contradiction_audit.py eval_runs/typed_ablation/shard_*
    python src/type_contradiction_audit.py <run_dir> --by-attribute

`--compare-logs` additionally prints the declared type each log gives every attribute that appears
in a prediction, which is what makes a corpus divergence visible instead of silent.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from LogEvents import extract_ocel_schema  # noqa: E402

# Which value-filter kinds a declared type admits. Read from LLM.py rather than restated, so the
# audit cannot drift from the grammar it is auditing.
def _value_filter_map():
    import ast
    source = (paths.REPO_ROOT / "src" / "LLM.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "VALUE_FILTER_FOR_DECLARED":
            return ast.literal_eval(node.value)
    raise RuntimeError("VALUE_FILTER_FOR_DECLARED not found in src/LLM.py")


VALUE_FILTER_FOR_DECLARED = _value_filter_map()
# "ValueFilterInteger" -> the `type` discriminator a query actually carries.
TYPE_OF_DEF = {f"ValueFilter{k}": k for k in
               ("Integer", "Float", "Boolean", "String", "Time")}

_schema_cache = {}


def declared_types(log_path):
    if log_path not in _schema_cache:
        schema = extract_ocel_schema(log_path)
        mapping = {}
        for kind in ("object_attributes", "event_attributes"):
            for attributes in (schema.get(kind) or {}).values():
                for attribute in attributes:
                    mapping[attribute["name"]] = str(attribute.get("type", "")).lower()
        _schema_cache[log_path] = mapping
    return _schema_cache[log_path]


def attribute_filters(query):
    """Every (attribute_name, value_filter type) in the tree, any depth."""
    out = []
    for node in (query.get("nodes") or []):
        box = node.get("Box") if isinstance(node, dict) else None
        data = box[0] if isinstance(box, list) and box else None
        if not isinstance(data, dict):
            continue
        for flt in (data.get("filters") or []):
            if isinstance(flt, dict) and "attribute_name" in flt:
                value_filter = flt.get("value_filter") or {}
                out.append((flt.get("attribute_name"), value_filter.get("type")))
    return out


def audit(run_dir, by_attribute=False, compare_logs=False):
    predictions = Path(run_dir) / "predictions.json"
    if not predictions.exists():
        print(f"   no predictions.json in {run_dir}")
        return None
    records = json.loads(predictions.read_text(encoding="utf-8"))

    # Whether this is a GATE or a MEASUREMENT depends on the rung. A rung with the constraint OFF
    # is where the error rate is measured (notes 34.3); only a rung with it ON must come out zero.
    # Reported as a failure either way, the baseline's own statistic reads as a broken experiment.
    config_path = Path(run_dir) / "config_resolved.json"
    constrained = False
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        constrained = bool(config.get("type_aware_filters"))

    emitted = contradictions = unknown = 0
    cases_with_filters = set()
    detail = []
    per_attribute = collections.Counter()
    missing_logs = set()
    seen_logs = {}

    for record in records:
        prediction = record.get("prediction")
        if not isinstance(prediction, dict):
            continue
        log = record.get("ocel_path")
        if not log or not Path(log).exists():
            missing_logs.add(str(log))
            continue
        seen_logs.setdefault(str(log), 0)
        mapping = declared_types(str(log))
        for name, emitted_type in attribute_filters(prediction):
            emitted += 1
            cases_with_filters.add(record.get("case_id"))
            seen_logs[str(log)] += 1
            declared = mapping.get(name)
            if declared is None:
                unknown += 1
                detail.append((record.get("case_id"), name, "UNDECLARED", emitted_type))
                continue
            allowed = {TYPE_OF_DEF[d] for d in VALUE_FILTER_FOR_DECLARED.get(declared, ())}
            if emitted_type not in allowed:
                contradictions += 1
                per_attribute[(name, declared, emitted_type)] += 1
                detail.append((record.get("case_id"), name, declared, emitted_type))

    role = "type-aware ON -> GATE" if constrained else "type-aware OFF -> MEASUREMENT"
    print(f"=== {Path(run_dir).name}")
    print(f"   rung                            : {role}")
    if missing_logs:
        print(f"   REFUSING to score {len(missing_logs)} record(s): their log is not on this host.")
        for log in sorted(missing_logs)[:3]:
            print(f"      {log}")
        print("   Run this where the run ran; scoring against a same-named local log is how a")
        print("   corpus divergence turns into a phantom finding.")
    print(f"   attribute value-filters emitted : {emitted} (over {len(cases_with_filters)} cases)")
    print(f"   TYPE CONTRADICTIONS             : {contradictions}")
    print(f"   undeclared attribute names      : {unknown}")
    for row in detail[:12]:
        print(f"      case {row[0]} | {row[1]} declared={row[2]} emitted={row[3]}")
    if by_attribute and per_attribute:
        print("   by attribute:")
        for (name, declared, got), n in per_attribute.most_common():
            print(f"      {name:24s} {declared:8s} -> {got:8s}  x{n}")
    if compare_logs:
        print("   logs used by this run (declared types as THIS host reads them):")
        for log in sorted(seen_logs):
            mapping = declared_types(log)
            hist = collections.Counter(mapping.values())
            print(f"      {Path(log).name:34s} {dict(hist)}")
    # A gate that passes while scoring nothing is worse than no gate: the 2026-09-12 smoke passed
    # 0-of-0 twice because the case slice contained no attribute filters at all.
    return {"contradictions": contradictions, "emitted": emitted,
            "refused": len(missing_logs), "constrained": constrained}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--by-attribute", action="store_true")
    ap.add_argument("--compare-logs", action="store_true",
                    help="Print each log's declared-type histogram, to expose a corpus mismatch.")
    args = ap.parse_args()

    totals = []
    for run_dir in args.run_dirs:
        totals.append(audit(run_dir, args.by_attribute, args.compare_logs))
        print()

    scored = [t for t in totals if t is not None]
    if not scored:
        return
    gated = [t for t in scored if t["constrained"]]
    measured = [t for t in scored if not t["constrained"]]

    if measured:
        emitted_m = sum(t["emitted"] for t in measured)
        bad_m = sum(t["contradictions"] for t in measured)
        print("=== MEASUREMENT (rungs with the constraint OFF) ===")
        if emitted_m:
            print(f"   {bad_m} of {emitted_m} attribute value-filters contradict the declared "
                  f"type ({100.0 * bad_m / emitted_m:.0f}%).")
            print("   This is the notes-34.3 statistic, not a gate. It is only comparable across")
            print("   runs that used the SAME corpus -- see --compare-logs.")
        else:
            print("   no attribute value-filter emitted; nothing to measure.")
        if not gated:
            return
        print()

    if not gated:
        return
    print("=== GATE (rungs with the constraint ON) ===")
    refused = sum(t["refused"] for t in gated)
    emitted = sum(t["emitted"] for t in gated)
    contradictions = sum(t["contradictions"] for t in gated)
    if contradictions:
        print("   FAIL: a type-aware rung must contradict a declared type ZERO times.")
        print("   Either the grammar is not binding, or the run and this host disagree about")
        print("   the corpus -- check --compare-logs before concluding anything.")
    elif refused:
        print(f"   INCONCLUSIVE: {refused} record(s) refused because their log is absent here.")
        print("   Zero contradictions out of a partly-unscored run is not a pass.")
    elif emitted == 0:
        print("   INCONCLUSIVE: no attribute value-filter was emitted at all, so the constraint")
        print("   was never exercised. 0 contradictions out of 0 opportunities proves nothing --")
        print("   pick a case slice that reaches them (--case-shard 2/11 covers 7).")
    else:
        print(f"   PASS: {emitted} attribute value-filter(s) emitted, none contradicting a")
        print("   declared type.")


if __name__ == "__main__":
    main()
