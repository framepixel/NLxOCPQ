#!/usr/bin/env python3
"""What the model produces instead of the right query: notes 23, made reproducible.

Notes 23 ("Where the failure actually is", 2026-08-30) is cited throughout the thesis -- entity
agreement 80.0%, node-count agreement 31.4%, `SAT` emitted at 14% of gold's rate, and the 0.3770
shape-matched ceiling that bounds the whole composition programme. Every one of those figures was
computed ad hoc from `eval_runs/selfconsistency`, at temperature 0.7, on the corpus notes 35 later
found to be stale, and no script in the repo could reproduce them. This is that script.

**It classifies and counts; it does not score.** Every F1, precision and recall it prints is read from
the record exactly as `src/eval_testset.py` wrote it (CLAUDE.md 2.5). The only judgements it makes are
structural: which bucket a prediction falls in, and whether two trees agree on a feature.

DENOMINATORS, and why there are two. Notes 23 measured ONE condition over 1690 samples (10 per case).
A greedy matrix gives 169 records for that condition, of which ~63 are disjoint -- enough for the
shape, not for two-point differences. So every table is reported twice:

  * `--condition ...`  the notes-23 cell (fine-tuned / allowed-entities-only / schema on) and the
    BASE PAIR (base / chain-of-thought-plus / schema on). The base column is not decoration: notes
    23.3's claim survives only because the *unconstrained* base arm also names gold's object types
    most of the time, which is what rules out `allowed-entities-only`'s vocabulary as the explanation.
  * `--pooled`  the same measures over each arm's six conditions (~1014 records). Tighter, but it
    mixes prompt strategies, so it is a different measurement and is labelled as one.

Every percentage prints its n and a 95% band, so a 63-record 80% is never read as a 690-record 80%.

    python src/failure_anatomy.py --run_dir eval_runs/matrix_v2/merged --pooled \\
        --out eval_runs/matrix_v2/failure_anatomy_20260917.json
    python src/failure_anatomy.py --run_dir eval_runs/matrix_v2/merged --training   # adds notes 23.6
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from analyze_divergence import _combined, _node_count  # noqa: E402

# The notes-23.1 threshold: a result set with essentially nothing in common with gold. 23.1 records
# that moving it to 0.0 changes counts by under two points; `--disjoint-threshold` re-checks that.
DISJOINT_THRESHOLD = 0.05
# Notes 23.6's training row, measured 2026-08-30 over all 60,018 rows of data/finetune_ocpq.jsonl.
# The corpus has not changed since (its .meta.json records seed 3407 and the generator hash), so the
# row is reused unless --training re-measures it: parsing 234 MB costs ~a minute for a constant.
TRAINING_ROW_RECORDED = {"source": "notes 23.6 (2026-08-30, not re-measured)",
                         "one_box_share": 0.346, "mean_boxes": 2.06, "sat_share": 0.221,
                         "n": 60018}


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def band(count: int, n: int) -> str:
    """`p% +-b (k/n)` -- the 95% normal-approximation band, so small denominators announce themselves."""
    if not n:
        return "n/a (n=0)"
    p = count / n
    half = 1.96 * math.sqrt(max(p * (1 - p), 0.0) / n)
    return f"{p:6.1%} +-{half:4.1%} ({count}/{n})"


# --------------------------------------------------------------------------- structure of a query
def _boxes(query: Any) -> list:
    """Every box's data dict, in node order. A `Box` node is `[data, children]`."""
    if not isinstance(query, dict):
        return []
    out = []
    for node in query.get("nodes") or []:
        box = node.get("Box") if isinstance(node, dict) else None
        if isinstance(box, list) and box and isinstance(box[0], dict):
            out.append(box[0])
    return out


def _entity_names(query: Any) -> tuple:
    """`(event activities, object types)` named anywhere in the tree, as sets.

    Read from the variable declarations rather than from filters: a box declares what it binds, and
    that is what notes 23.3 compares.
    """
    events, objects = set(), set()
    for box in _boxes(query):
        for key, target in (("newEventVars", events), ("newObjectVars", objects)):
            declared = box.get(key)
            if not isinstance(declared, dict):
                continue
            for value in declared.values():
                if isinstance(value, list):
                    target.update(str(v) for v in value)
                elif value is not None:
                    target.add(str(value))
    return events, objects


def _constraint_types(query: Any) -> Counter:
    """Constraint types over the whole tree. A `Filter` constraint carries its filter's type."""
    counts = Counter()
    for box in _boxes(query):
        for constraint in box.get("constraints") or []:
            if not isinstance(constraint, dict):
                continue
            kind = str(constraint.get("type"))
            if kind == "Filter":
                inner = constraint.get("filter")
                inner_type = inner.get("type") if isinstance(inner, dict) else None
                kind = f"Filter:{inner_type}"
            elif kind == "SizeFilter":
                pass  # the constraint form; the box-level field is counted separately
            counts[kind] += 1
    return counts


def _filter_types(query: Any) -> Counter:
    """Box-level `filters[].type` over the whole tree."""
    counts = Counter()
    for box in _boxes(query):
        for flt in box.get("filters") or []:
            if isinstance(flt, dict):
                counts[str(flt.get("type"))] += 1
    return counts


def _size_filter_fields(query: Any) -> int:
    """How many entries sit in the box-level `sizeFilters` field (notes 23.4's inversion)."""
    return sum(len(box.get("sizeFilters") or []) for box in _boxes(query))


def _sat_count(query: Any) -> int:
    return _constraint_types(query).get("SAT", 0)


# ------------------------------------------------------------------------------------- bucketing
def bucket(record: dict, threshold: float = DISJOINT_THRESHOLD) -> str:
    """Which notes-23.2 bucket this record falls in. Reads the stored comparison, never recomputes it.

    Order matters: a record with no comparison has no overlap to measure, so `unscored` is decided
    before anything else. `exact` is separated from `partial` because notes 23.4's bucket table needs
    it.
    """
    combined = _combined(record)
    if combined is None or combined.get("f1") is None:
        return "unscored"
    f1 = float(combined["f1"])
    precision = float(combined.get("precision") or 0.0)
    recall = float(combined.get("recall") or 0.0)
    if f1 >= 0.9999:
        return "exact"
    # Zero overlap is disjoint at any threshold: with `--disjoint-threshold 0` -- notes 23.1's
    # alternative -- a strict `<` alone would make the class empty rather than stricter.
    if precision <= 0.0 and recall <= 0.0:
        return "disjoint"
    if precision < threshold and recall < threshold:
        return "disjoint"
    if recall >= 0.95:
        return "over_broad"
    if precision >= 0.95:
        return "over_narrow"
    return "partial"


def _agree(gold: Any, prediction: Any, feature) -> Optional[bool]:
    """Do gold and the prediction agree on one structural feature? None when either is unreadable."""
    if not isinstance(gold, dict) or not isinstance(prediction, dict):
        return None
    return feature(gold) == feature(prediction)


# ---------------------------------------------------------------------------------- the analysis
def analyse(records: list, threshold: float = DISJOINT_THRESHOLD) -> dict:
    """Every notes-23 measure over one set of records."""
    buckets = Counter(bucket(r, threshold) for r in records)
    executed = [r for r in records if bucket(r, threshold) != "unscored"]
    disjoint = [r for r in records if bucket(r, threshold) == "disjoint"]
    parsed = [r for r in records if isinstance(r.get("prediction"), dict)
              and _node_count(r.get("prediction")) is not None]

    # 23.3 -- entity grounding, on the disjoint set
    ent = {"object_types": 0, "event_activities": 0, "both": 0, "n": len(disjoint)}
    for rec in disjoint:
        gold_events, gold_objects = _entity_names(rec.get("gold_query"))
        pred_events, pred_objects = _entity_names(rec.get("prediction"))
        same_objects = gold_objects == pred_objects
        same_events = gold_events == pred_events
        ent["object_types"] += same_objects
        ent["event_activities"] += same_events
        ent["both"] += same_objects and same_events

    # 23.4 -- composition, on the same set
    comp = {"node_count": 0, "filter_multiset": 0, "constraint_multiset": 0, "n": len(disjoint),
            "gold_boxes": 0.0, "pred_boxes": 0.0}
    gold_constructs, pred_constructs = Counter(), Counter()
    gold_size_fields = pred_size_fields = 0
    for rec in disjoint:
        gold, pred = rec.get("gold_query"), rec.get("prediction")
        comp["node_count"] += bool(_agree(gold, pred, _node_count))
        comp["filter_multiset"] += bool(_agree(gold, pred, _filter_types))
        comp["constraint_multiset"] += bool(_agree(gold, pred, _constraint_types))
        comp["gold_boxes"] += _node_count(gold) or 0
        comp["pred_boxes"] += _node_count(pred) or 0
        gold_constructs += _constraint_types(gold)
        pred_constructs += _constraint_types(pred)
        gold_size_fields += _size_filter_fields(gold)
        pred_size_fields += _size_filter_fields(pred)
    if disjoint:
        comp["gold_boxes"] /= len(disjoint)
        comp["pred_boxes"] /= len(disjoint)
    comp["constructs"] = {k: {"gold": gold_constructs.get(k, 0), "predicted": pred_constructs.get(k, 0)}
                          for k in sorted(set(gold_constructs) | set(pred_constructs))}
    comp["size_filter_field"] = {"gold": gold_size_fields, "predicted": pred_size_fields}

    # 23.4's bucket table -- node agreement and SAT rate by outcome
    bucket_table = {}
    for name in ("exact", "partial", "disjoint"):
        rows = [r for r in records if bucket(r, threshold) == name]
        if not rows:
            continue
        bucket_table[name] = {
            "n": len(rows),
            "gold_nodes": sum(_node_count(r.get("gold_query")) or 0 for r in rows) / len(rows),
            "pred_nodes": sum(_node_count(r.get("prediction")) or 0 for r in rows) / len(rows),
            "node_match": sum(bool(_agree(r.get("gold_query"), r.get("prediction"), _node_count))
                              for r in rows) / len(rows),
            "gold_sat_per_query": sum(_sat_count(r.get("gold_query")) for r in rows) / len(rows),
            "pred_sat_per_query": sum(_sat_count(r.get("prediction")) for r in rows) / len(rows),
        }

    # 23.5 -- the capability frontier, by gold depth
    depth = {}
    for rec in records:
        nodes = _node_count(rec.get("gold_query")) or 0
        key = "4+" if nodes >= 4 else str(nodes)
        cell = depth.setdefault(key, {"n": 0, "executed": 0, "f1_sum": 0.0, "exact": 0})
        cell["n"] += 1
        combined = _combined(rec)
        f1 = float(combined["f1"]) if combined and combined.get("f1") is not None else 0.0
        cell["executed"] += rec.get("status") == "success"
        cell["f1_sum"] += f1
        cell["exact"] += f1 >= 0.9999

    # 23.6 -- the shape of what the model emits (parsed only)
    emitted = {"n": len(parsed),
               "one_box": sum(1 for r in parsed if _node_count(r["prediction"]) == 1),
               "mean_boxes": (sum(_node_count(r["prediction"]) or 0 for r in parsed) / len(parsed))
               if parsed else 0.0,
               "with_sat": sum(1 for r in parsed if _sat_count(r["prediction"]) > 0)}

    # 23.7 -- structure is necessary, not sufficient
    shape = {}
    for name, want in (("shape_matched", True), ("shape_mismatched", False)):
        rows = [r for r in executed
                if _agree(r.get("gold_query"), r.get("prediction"), _node_count) is want]
        if not rows:
            continue
        f1s = []
        for r in rows:
            combined = _combined(r)
            f1s.append(float(combined["f1"]) if combined and combined.get("f1") is not None else 0.0)
        shape[name] = {
            "n": len(rows),
            "mean_f1": sum(f1s) / len(rows),
            "exact": sum(1 for f in f1s if f >= 0.9999),
            "still_disjoint": sum(1 for r in rows if bucket(r, threshold) == "disjoint"),
        }

    return {"n": len(records), "buckets": dict(buckets), "executed": len(executed),
            "entities": ent, "composition": comp, "bucket_table": bucket_table,
            "depth": depth, "emitted": emitted, "shape": shape}


def gold_distribution(records: list) -> dict:
    """Notes 23.6's gold row, from the distinct gold queries present in the run."""
    seen = {}
    for rec in records:
        seen.setdefault(rec.get("nl_query"), rec.get("gold_query"))
    trees = [t for t in seen.values() if isinstance(t, dict)]
    if not trees:
        return {"n": 0}
    return {"n": len(trees),
            "one_box_share": sum(1 for t in trees if _node_count(t) == 1) / len(trees),
            "mean_boxes": sum(_node_count(t) or 0 for t in trees) / len(trees),
            "sat_share": sum(1 for t in trees if _sat_count(t) > 0) / len(trees),
            "sizes": dict(Counter(_node_count(t) for t in trees))}


def training_distribution(path: Path) -> dict:
    """Notes 23.6's training row, re-measured from data/finetune_ocpq.jsonl (234 MB, ~1 min)."""
    sizes, sat, total = Counter(), 0, 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                target = row["messages"][-1]["content"]
                tree = json.loads(target)
            except Exception:  # noqa: BLE001 - a row we cannot parse is not a tree
                continue
            nodes = _node_count(tree)
            if nodes is None:
                continue
            total += 1
            sizes[nodes] += 1
            sat += _sat_count(tree) > 0
    if not total:
        return {"n": 0}
    return {"source": str(path), "n": total,
            "one_box_share": sizes.get(1, 0) / total,
            "mean_boxes": sum(k * v for k, v in sizes.items()) / total,
            "sat_share": sat / total,
            "sizes": dict(sorted(sizes.items()))}


# -------------------------------------------------------------------------------------- printing
def _print_block(title: str, report: dict) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    n = report["n"]
    print(f"records {n} | executed (comparable) {report['executed']}")
    print("\n23.2  what the predictions are")
    for name in ("exact", "disjoint", "partial", "over_broad", "over_narrow", "unscored"):
        if name in report["buckets"]:
            print(f"    {name:12} {band(report['buckets'][name], n)}")

    ent = report["entities"]
    print(f"\n23.3  entity grounding, on the {ent['n']} DISJOINT predictions")
    for key, label in (("object_types", "identical object types"),
                       ("event_activities", "identical event activities"),
                       ("both", "both identical")):
        print(f"    {label:28} {band(ent[key], ent['n'])}")

    comp = report["composition"]
    print(f"\n23.4  composition, same {comp['n']} predictions")
    for key, label in (("node_count", "same number of nodes"),
                       ("filter_multiset", "identical filter-type multiset"),
                       ("constraint_multiset", "identical constraint-type multiset")):
        print(f"    {label:32} {band(comp[key], comp['n'])}")
    print(f"    mean boxes: gold {comp['gold_boxes']:.2f} vs predicted {comp['pred_boxes']:.2f}")
    print(f"    {'construct':28}{'gold':>8}{'predicted':>11}")
    for name, counts in sorted(comp["constructs"].items(), key=lambda kv: -kv[1]["gold"])[:8]:
        print(f"    {name:28}{counts['gold']:>8}{counts['predicted']:>11}")
    size_field = comp["size_filter_field"]
    print(f"    box-level sizeFilters field: gold {size_field['gold']}, "
          f"predicted {size_field['predicted']}")

    if report["bucket_table"]:
        print("\n23.4  by outcome")
        print(f"    {'bucket':10}{'n':>6}{'gold nodes':>12}{'pred nodes':>12}"
              f"{'node match':>12}{'gold SAT/q':>12}{'pred SAT/q':>12}")
        for name, cell in report["bucket_table"].items():
            print(f"    {name:10}{cell['n']:>6}{cell['gold_nodes']:>12.2f}{cell['pred_nodes']:>12.2f}"
                  f"{cell['node_match']:>11.1%}{cell['gold_sat_per_query']:>12.2f}"
                  f"{cell['pred_sat_per_query']:>12.2f}")

    print("\n23.5  by gold tree size")
    print(f"    {'boxes':8}{'n':>6}{'exec':>9}{'mean F1':>10}{'exact':>9}")
    for key in sorted(report["depth"], key=lambda k: (k == "4+", k)):
        cell = report["depth"][key]
        print(f"    {key:8}{cell['n']:>6}{cell['executed'] / cell['n']:>9.1%}"
              f"{cell['f1_sum'] / cell['n']:>10.4f}{cell['exact'] / cell['n']:>9.1%}")

    emitted = report["emitted"]
    if emitted["n"]:
        print(f"\n23.6  what the model emits ({emitted['n']} parsed)")
        print(f"    1-box share {emitted['one_box'] / emitted['n']:.1%} | "
              f"mean boxes {emitted['mean_boxes']:.2f} | "
              f"with >=1 SAT {emitted['with_sat'] / emitted['n']:.1%}")

    if report["shape"]:
        print("\n23.7  conditioned on getting the tree shape right  (floor 0.7250)")
        print(f"    {'':18}{'n':>6}{'mean F1':>10}{'exact':>9}{'still disjoint':>16}")
        for name, cell in report["shape"].items():
            print(f"    {name:18}{cell['n']:>6}{cell['mean_f1']:>10.4f}"
                  f"{cell['exact'] / cell['n']:>9.1%}{cell['still_disjoint'] / cell['n']:>16.1%}")


def _select(records: list, model: str, strategy: Optional[str], schema: Optional[bool]) -> list:
    out = []
    for rec in records:
        if rec.get("model_label") != model:
            continue
        if strategy and rec.get("prompt_strategy") != strategy:
            continue
        if schema is not None and bool(rec.get("schema_enforced")) != schema:
            continue
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="a finished run or merged directory")
    ap.add_argument("--model", default="llama31-8b-finetuned",
                    help="the primary arm (notes 23's fine-tuned column)")
    ap.add_argument("--strategy", default="schema-allowed-entities-only")
    ap.add_argument("--base-model", default="llama31-8b-base",
                    help="the base pair; notes 23.3's confound check")
    ap.add_argument("--base-strategy", default="schema-chain-of-thought-plus")
    ap.add_argument("--schema-enforced", default="on", choices=["on", "off", "both"])
    ap.add_argument("--pooled", action="store_true",
                    help="also report each arm over ALL its conditions (a different denominator)")
    ap.add_argument("--training", action="store_true",
                    help="re-measure notes 23.6's training row from data/finetune_ocpq.jsonl")
    ap.add_argument("--disjoint-threshold", type=float, default=DISJOINT_THRESHOLD)
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    records = _read_json(run_dir / "predictions.json")
    schema = {"on": True, "off": False, "both": None}[args.schema_enforced]
    print(f"failure anatomy of {run_dir}  ({len(records)} records)")
    print(f"disjoint threshold: precision < {args.disjoint_threshold} AND "
          f"recall < {args.disjoint_threshold}  (notes 23.1)")

    report = {"run_dir": str(run_dir), "disjoint_threshold": args.disjoint_threshold,
              "conditions": {}, "pooled": {}}

    for label, model, strategy in (("primary", args.model, args.strategy),
                                   ("base pair", args.base_model, args.base_strategy)):
        rows = _select(records, model, strategy, schema)
        if not rows:
            print(f"\n(no records for {model} / {strategy}; skipped)")
            continue
        title = f"{label}: {model} / {strategy} / schema={args.schema_enforced}"
        block = analyse(rows, args.disjoint_threshold)
        _print_block(title, block)
        report["conditions"][title] = block

    if args.pooled:
        for model in sorted({r.get("model_label") for r in records if r.get("model_label")}):
            rows = _select(records, model, None, None)
            block = analyse(rows, args.disjoint_threshold)
            _print_block(f"pooled (ALL conditions -- a different denominator): {model}", block)
            report["pooled"][model] = block

    gold = gold_distribution(records)
    training = (training_distribution(paths.DATA / "finetune_ocpq.jsonl") if args.training
                else dict(TRAINING_ROW_RECORDED))
    report["gold_distribution"] = gold
    report["training_distribution"] = training
    print(f"\n{'=' * 78}\n23.6  distributions\n{'=' * 78}")
    print(f"    {'row':28}{'1-box':>9}{'mean boxes':>13}{'>=1 SAT':>10}{'n':>8}")
    print(f"    {'training set':28}{training.get('one_box_share', 0):>9.1%}"
          f"{training.get('mean_boxes', 0):>13.2f}{training.get('sat_share', 0):>10.1%}"
          f"{training.get('n', 0):>8}")
    if not args.training:
        print(f"      ^ {TRAINING_ROW_RECORDED['source']}; pass --training to re-measure")
    print(f"    {'test gold':28}{gold.get('one_box_share', 0):>9.1%}"
          f"{gold.get('mean_boxes', 0):>13.2f}{gold.get('sat_share', 0):>10.1%}{gold.get('n', 0):>8}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
