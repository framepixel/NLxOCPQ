#!/usr/bin/env python3
"""Post-hoc error analysis: what the model actually gets wrong, not just where.

`slice_results_by_bb.py` reports mean `combined_f1` per (condition x building block) and
`bb_cooccurrence.py` reports which blocks appear together. Neither surfaces a concrete query
pair, so "Gemma scores 0.41 on S1" cannot be turned into "Gemma omits the NumChilds size
filter and returns the unfiltered population". This fills that gap.

Nothing new is captured during evaluation: every signal used is already in predictions.json
(`similarity.combined` carries f1/precision/recall/gold_size/pred_size, and `metrics` carries
the grounding flags), so this is extraction over a ~40 MB artifact rather than collection.

This computes **no metrics** — it classifies and groups values `eval_testset.py` already
produced, keeping metric logic in one place (CLAUDE.md 2.5). Its output is diagnostic and
must never be cited as a result.

Usage:
    python analyze_divergence.py --run_dir eval_runs/<run_id>
                                 [--labels testset_bb_labels.json]
                                 [--out divergence_report.json]
                                 [--per-bucket 3]
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

from slice_results_by_bb import BB_ORDER, _bb_category, _build_bb_index, CATEGORY_NAMES


# Failure classes, in precedence order — the first match wins, so each record is counted
# once. Ordered from "never produced an answer" to "produced a nearly right answer".
CLASS_SCHEMA_INVALID = "schema_invalid"
CLASS_EXECUTION_FAILED = "execution_failed"
CLASS_EMPTY_RESULT = "empty_result"
CLASS_WRONG_POPULATION = "wrong_population"
CLASS_OVER_SELECTION = "over_selection"
CLASS_UNDER_SELECTION = "under_selection"
CLASS_PARTIAL = "partial"

CLASS_ORDER = [
    CLASS_SCHEMA_INVALID,
    CLASS_EXECUTION_FAILED,
    CLASS_EMPTY_RESULT,
    CLASS_WRONG_POPULATION,
    CLASS_OVER_SELECTION,
    CLASS_UNDER_SELECTION,
    CLASS_PARTIAL,
]

CLASS_DESCRIPTIONS = {
    CLASS_SCHEMA_INVALID: "did not parse as a BindingBoxTree",
    CLASS_EXECUTION_FAILED: "parsed, but ocpq_cli produced no result",
    CLASS_EMPTY_RESULT: "ran and returned nothing while gold returned something",
    CLASS_WRONG_POPULATION: "ran and answered about something else entirely (f1 = 0)",
    CLASS_OVER_SELECTION: "found everything gold found, plus extra (dropped a restriction)",
    CLASS_UNDER_SELECTION: "everything it found was right, but it missed some (over-constrained)",
    CLASS_PARTIAL: "overlaps gold in both directions",
}


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _combined(record: dict) -> Optional[dict]:
    """The root-situation comparison block, or None when no comparison was possible."""
    similarity = record.get("similarity")
    if not isinstance(similarity, dict):
        return None
    combined = similarity.get("combined")
    return combined if isinstance(combined, dict) else None


def classify(record: dict) -> Optional[str]:
    """Return the failure class for a record, or None when it matched gold exactly.

    Precedence matters: a prediction that failed to parse never reached execution, and one
    that failed to execute has no result sets to compare, so the earlier checks must come
    first or the later ones would read missing data as a semantic difference.
    """
    if not (record.get("metrics") or {}).get("schema_valid"):
        return CLASS_SCHEMA_INVALID

    combined = _combined(record)
    if combined is None or not (record.get("execution") or {}).get("success"):
        return CLASS_EXECUTION_FAILED

    f1 = combined.get("f1")
    if f1 is None:
        return CLASS_EXECUTION_FAILED
    if f1 >= 1.0:
        return None  # exact match on root situations

    gold_size = combined.get("gold_size") or 0
    pred_size = combined.get("pred_size") or 0
    precision = combined.get("precision") or 0.0
    recall = combined.get("recall") or 0.0

    if pred_size == 0 and gold_size > 0:
        return CLASS_EMPTY_RESULT
    if f1 == 0.0:
        return CLASS_WRONG_POPULATION
    if recall >= 1.0:
        return CLASS_OVER_SELECTION
    if precision >= 1.0:
        return CLASS_UNDER_SELECTION
    return CLASS_PARTIAL


def _safe_bb_labels(query: Any) -> list:
    """BB labels for a query, tolerating predictions that parse but are structurally odd."""
    if not isinstance(query, dict):
        return []
    try:
        from tag_testset_bb import detect_bb_labels

        return sorted(detect_bb_labels(query))
    except Exception:  # noqa: BLE001 - a malformed prediction is data, not an error
        return []


def _entity_types(query: Any) -> set:
    """Event and object type names a query references."""
    if not isinstance(query, dict):
        return set()
    try:
        from eval_testset import _extract_query_entities

        events, objects = _extract_query_entities(query)
        return set(events) | set(objects)
    except Exception:  # noqa: BLE001
        return set()


def _node_count(query: Any) -> Optional[int]:
    if not isinstance(query, dict):
        return None
    nodes = query.get("nodes")
    return len(nodes) if isinstance(nodes, list) else None


def diagnose(record: dict) -> dict:
    """Describe *how* a prediction differs from gold, using only cheap structural signals.

    The BB diff is the most informative part and costs nothing: a gold query carrying `S1`
    whose prediction does not is, by construction, a model that dropped the count threshold.
    """
    gold_query = record.get("gold_query")
    prediction = record.get("prediction")

    bb_gold = _safe_bb_labels(gold_query)
    bb_pred = _safe_bb_labels(prediction)
    types_gold = _entity_types(gold_query)
    types_pred = _entity_types(prediction)
    metrics = record.get("metrics") or {}
    combined = _combined(record) or {}

    return {
        "bb_gold": bb_gold,
        "bb_pred": bb_pred,
        "bb_dropped": sorted(set(bb_gold) - set(bb_pred), key=_bb_sort_key),
        "bb_invented": sorted(set(bb_pred) - set(bb_gold), key=_bb_sort_key),
        "types_missing": sorted(types_gold - types_pred),
        "types_extra": sorted(types_pred - types_gold),
        "nodes_gold": _node_count(gold_query),
        "nodes_pred": _node_count(prediction),
        "gold_size": combined.get("gold_size"),
        "pred_size": combined.get("pred_size"),
        "hallucinated_entities": bool(metrics.get("hallucinated_entities")),
        "hallucinated_attributes": metrics.get("hallucinated_attributes") or [],
        "invalid_relations": metrics.get("invalid_relations") or [],
        "unsupported_filters": bool(metrics.get("unsupported_filters")),
        "bindings_skipped": bool(metrics.get("bindings_skipped")),
    }


def _bb_sort_key(bb: str):
    """Sort BB ids by their catalogue order rather than alphabetically (F2 before F10)."""
    return (BB_ORDER.index(bb), bb) if bb in BB_ORDER else (len(BB_ORDER), bb)


def _example(record: dict, failure_class: str, diagnosis: dict) -> dict:
    """A retained example. Only these carry the full query pair."""
    combined = _combined(record) or {}
    return {
        "case_id": record.get("case_id"),
        "config_id": record.get("config_id"),
        "model_label": record.get("model_label"),
        "prompt_strategy": record.get("prompt_strategy"),
        "schema_enforced": record.get("schema_enforced"),
        "domain_seen": record.get("domain_seen"),
        # Stratum + floor, so a retained example can be read in context: on a
        # non-discriminating entry the trivial baseline already scores ~1.0, which changes what
        # a given failure class means.
        "discriminating": record.get("discriminating"),
        "baseline_f1": record.get("baseline_f1"),
        "structural_f1": ((record.get("metrics") or {}).get("structural") or {}).get("f1"),
        # F1 against the floor, in [-1, 1]. A negative value on a retained example says the
        # prediction did worse than a do-nothing query -- which the raw `f1` below can hide
        # entirely when the floor is high.
        "skill": ((record.get("metrics") or {}).get("skill")),
        "class": failure_class,
        "nl_query": record.get("nl_query"),
        "ocel_path": record.get("ocel_path"),
        "f1": combined.get("f1"),
        "precision": combined.get("precision"),
        "recall": combined.get("recall"),
        "diagnosis": diagnosis,
        "error": record.get("error"),
        "gold_query": record.get("gold_query"),
        "prediction": record.get("prediction"),
    }


def build_report(predictions: list, bb_index: dict, per_bucket: int = 3) -> dict:
    """Classify every imperfect prediction and roll the findings up.

    Args:
        predictions: Records from a run's predictions.json.
        bb_index: nl_query -> BB labels, from `slice_results_by_bb._build_bb_index`.
        per_bucket: Examples retained per (class x condition) bucket.
    """
    class_counts = Counter()
    by_condition = defaultdict(Counter)
    buckets = defaultdict(list)
    # BB -> [occurrences in gold, times absent from the prediction]
    bb_occurrences = Counter()
    bb_dropped = Counter()
    dropped_together = Counter()
    total = 0
    matched = 0

    for record in predictions:
        total += 1
        failure_class = classify(record)
        config_id = record.get("config_id") or "unknown"
        if failure_class is None:
            matched += 1
            by_condition[config_id]["exact_match"] += 1
            continue

        diagnosis = diagnose(record)
        class_counts[failure_class] += 1
        by_condition[config_id][failure_class] += 1

        # Only count BB drops where the prediction was structurally readable; an unparseable
        # prediction "drops" every block trivially and would swamp the ranking.
        if failure_class != CLASS_SCHEMA_INVALID and diagnosis["bb_pred"]:
            # Prefer the curated testset labels over re-deriving them from the gold query.
            gold_bbs = bb_index.get(record.get("nl_query", "")) or diagnosis["bb_gold"]
            for bb in gold_bbs:
                bb_occurrences[bb] += 1
                if bb not in diagnosis["bb_pred"]:
                    bb_dropped[bb] += 1
            for bb in diagnosis["bb_dropped"]:
                dropped_together[bb] += 1

        buckets[(failure_class, config_id)].append((record, diagnosis))

    examples = []
    for (failure_class, config_id), items in buckets.items():
        # Deterministic: worst first, then by case_id. Stable across re-runs, no RNG.
        items.sort(key=lambda pair: (
            (_combined(pair[0]) or {}).get("f1") if (_combined(pair[0]) or {}).get("f1") is not None else -1.0,
            pair[0].get("case_id") or 0,
        ))
        for record, diagnosis in items[:per_bucket]:
            examples.append(_example(record, failure_class, diagnosis))
    examples.sort(key=lambda e: (CLASS_ORDER.index(e["class"]), e["config_id"] or "", e["case_id"] or 0))

    imperfect = total - matched
    ranking = []
    for bb in sorted(bb_occurrences, key=_bb_sort_key):
        occurrences = bb_occurrences[bb]
        dropped = bb_dropped[bb]
        ranking.append({
            "bb": bb,
            "category": CATEGORY_NAMES.get(_bb_category(bb), _bb_category(bb)),
            "gold_occurrences_in_diverging_cases": occurrences,
            "dropped": dropped,
            "drop_rate": round(dropped / occurrences, 4) if occurrences else None,
        })
    ranking.sort(key=lambda row: (-(row["drop_rate"] or 0), -row["dropped"]))

    return {
        "totals": {
            "records": total,
            "exact_match": matched,
            "diverging": imperfect,
            "diverging_rate": round(imperfect / total, 4) if total else None,
        },
        "by_class": {
            name: {
                "n": class_counts.get(name, 0),
                "share_of_diverging": (
                    round(class_counts.get(name, 0) / imperfect, 4) if imperfect else None
                ),
                "means": CLASS_DESCRIPTIONS[name],
            }
            for name in CLASS_ORDER
        },
        "dropped_bb_ranking": ranking,
        "by_condition": {cid: dict(counts) for cid, counts in sorted(by_condition.items())},
        "examples_per_bucket": per_bucket,
        "examples": examples,
    }


def _print_summary(report: dict) -> None:
    totals = report["totals"]
    print(f"\nRecords {totals['records']} | exact match {totals['exact_match']} | "
          f"diverging {totals['diverging']} ({(totals['diverging_rate'] or 0):.1%})")

    print("\n--- Failure classes ---")
    print(f"{'class':20s} {'n':>6s} {'share':>8s}  meaning")
    for name in CLASS_ORDER:
        row = report["by_class"][name]
        if not row["n"]:
            continue
        share = f"{row['share_of_diverging']:.1%}" if row["share_of_diverging"] is not None else "-"
        print(f"{name:20s} {row['n']:6d} {share:>8s}  {row['means']}")

    ranking = [r for r in report["dropped_bb_ranking"] if r["dropped"]]
    if ranking:
        print("\n--- Building blocks most often dropped (in diverging cases) ---")
        print(f"{'bb':6s} {'category':13s} {'dropped':>8s} {'of':>6s} {'rate':>7s}")
        for row in ranking[:15]:
            print(f"{row['bb']:6s} {row['category']:13s} {row['dropped']:8d} "
                  f"{row['gold_occurrences_in_diverging_cases']:6d} {(row['drop_rate'] or 0):6.1%}")

    print(f"\nRetained {len(report['examples'])} example(s) "
          f"({report['examples_per_bucket']} per class x condition).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Evaluation run directory.")
    parser.add_argument("--labels", default="data/testset_bb_labels.json")
    parser.add_argument("--out", default="divergence_report.json")
    parser.add_argument("--per-bucket", type=int, default=3, dest="per_bucket",
                        help="Examples kept per (failure class x condition). Default 3.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.json in {run_dir}")

    predictions = _load_json(predictions_path)
    labels_path = Path(args.labels)
    bb_index = _build_bb_index(_load_json(labels_path)) if labels_path.exists() else {}
    if not bb_index:
        print(f"Warning: {labels_path} not found; BB drop ranking falls back to the gold queries.")

    report = build_report(predictions, bb_index, per_bucket=args.per_bucket)
    out_path = Path(args.out)
    if out_path.parent == Path("."):
        out_path = run_dir / out_path
    _write_json(out_path, report)
    _print_summary(report)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
