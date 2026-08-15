#!/usr/bin/env python3
"""Post-hoc per-building-block breakdown of evaluation results.

Joins predictions.json with testset_bb_labels.json on nl_query, then groups
metrics by (config_id, bb_label) and writes bb_breakdown.json.

Usage:
    python slice_results_by_bb.py --run_dir eval_runs/<run_id>
                                   [--labels testset_bb_labels.json]
                                   [--out bb_breakdown.json]
                                   [--min_n 1]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


BB_ORDER = [
    "V1","V2","V3","V4","V5","V6",
    "F1","F2","F3","F4","F5","F6","F7","F8","F9",
    "F10","F11","F12","F13","F14","F15","F16","F17","F18",
    "S1","S2","S3",
    "C1","C2","C3","C4","C5","C6","C7",
    "L1","L2","L3",
    "T1","T2","T3","T4","T5",
]

# BB category (first letter) -> human-readable name. Used for the category-level
# breakdown the discussion section reports against (BB-class performance).
CATEGORY_ORDER = ["V", "F", "S", "C", "L", "T"]
CATEGORY_NAMES = {
    "V": "Variables",
    "F": "Filters",
    "S": "SizeFilters",
    "C": "Constraints",
    "L": "Labels",
    "T": "Tree",
}


def _bb_category(bb: str) -> str:
    """Map a BB id (e.g. 'F11') to its category letter (e.g. 'F')."""
    return bb[0] if bb else ""


def _mean(vals: list) -> float | None:
    """Mean of the present values, or None when there are none.

    Drops None entries, matching `eval_testset._mean`. A record whose prediction failed to
    execute has `similarity: null` and so contributes `combined_f1: None`; summing that
    raised TypeError and took down the whole analysis step — which runs at the end of every
    evaluation, so one non-executing prediction failed the shard.
    """
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _skill_block(recs: list) -> dict:
    """Skill figures for one group of records, read from `metrics.skill`.

    Not recomputed here: `eval_testset` already writes the per-record value, and importing it
    would be circular because it imports `_build_bb_index` from this module.

    Skill is undefined where the trivial floor is 1.0, so `skill_n` reports the denominator --
    a building block made mostly of non-discriminating entries can have a skill score backed by
    very few records, and that must be visible rather than inferred.
    """
    skills = [((r.get("metrics") or {}).get("skill")) for r in recs]
    defined = [s for s in skills if s is not None]
    return {
        # Result F1 renormalised against the floor: 0 means no better than a no-op query.
        "skill_score": _mean(defined),
        # What a clamp at 0 would hide -- how often the model did worse than doing nothing.
        "below_floor_rate": _mean([1 if s < 0 else 0 for s in defined]),
        "skill_n": len(defined),
    }


def _load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_bb_index(labels_data: list) -> dict[str, list[str]]:
    return {entry["nl_query"]: entry["bb_labels"] for entry in labels_data}


def _execution_success(record: dict) -> int:
    exec_stats = record.get("execution") or {}
    return 1 if exec_stats.get("success") else 0


def slice_by_bb(
    predictions: list[dict],
    bb_index: dict[str, list[str]],
    min_n: int = 1,
) -> dict:
    """Group predictions by (config_id, bb_label) and compute per-group metrics.

    Returns:
        {
          config_id: {
            bb_label: {
              "n": int,
              "result_set_f1_strict": float,   # primary, failures count as 0
              "structural_f1": float,          # did it express the right constructs?
              "mean_baseline_f1": float,       # what a no-op query already scores
              "skill_score": float,            # the same F1 renormalised against that floor
              "below_floor_rate": float,       # how often it did worse than a no-op query
              "skill_n": int,                  # denominator: skill is undefined at floor 1.0
              "combined_f1": float,            # deprecated alias
              "schema_valid_rate": float,
              "execution_success_rate": float,
            }
          }
        }
    """
    missing_nl = set()
    # groups[config_id][bb_label] = [record, ...]
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for rec in predictions:
        nl = rec.get("nl_query", "")
        bbs = bb_index.get(nl)
        if bbs is None:
            missing_nl.add(nl)
            continue
        cid = rec["config_id"]
        for bb in bbs:
            groups[cid][bb].append(rec)

    if missing_nl:
        print(f"Warning: {len(missing_nl)} nl_query values not found in BB labels file.")
        for nl in list(missing_nl)[:5]:
            print(f"  missing: {nl[:80]!r}")

    result: dict = {}
    for cid, bb_map in groups.items():
        result[cid] = {}
        for bb, recs in bb_map.items():
            if len(recs) < min_n:
                continue
            result[cid][bb] = {
                "n": len(recs),
                # Result-level, counting failures as 0 (see eval_testset._result_f1_values):
                # the conditional variant alone rewards attempting fewer queries.
                "result_set_f1_strict": _mean([
                    0.0 if ((r.get("similarity") or {}).get("combined") or {}).get("f1") is None
                    else ((r.get("similarity") or {}).get("combined") or {}).get("f1")
                    for r in recs
                ]),
                # Query-level: did the model express this building block correctly? A BB where
                # the result score is high but the structural score is low is the interesting
                # case -- right rows for the wrong query.
                "structural_f1": _mean([
                    ((r.get("metrics") or {}).get("structural") or {}).get("f1") for r in recs
                ]),
                # What a no-op query already scores on these entries.
                "mean_baseline_f1": _mean([r.get("baseline_f1") for r in recs]),
                # Read from the record rather than recomputed: importing eval_testset here
                # would be circular (it imports _build_bb_index from this module).
                **_skill_block(recs),
                # Deprecated alias of result_set_f1_executed.
                "combined_f1": _mean([
                    (r.get("similarity") or {}).get("combined", {}).get("f1")
                    for r in recs
                ]),
                "schema_valid_rate": _mean(
                    [1 if r["metrics"]["schema_valid"] else 0 for r in recs]
                ),
                "execution_success_rate": _mean(
                    [_execution_success(r) for r in recs]
                ),
            }
    return result


def slice_by_category(
    predictions: list[dict],
    bb_index: dict[str, list[str]],
    min_n: int = 1,
) -> dict:
    """Group predictions by (config_id, BB-category) and compute per-group metrics.

    A record is counted once per category it touches, even if it contains several
    BBs from the same category (e.g. F1 + F4 -> one count under 'F'). Returns the
    same metric shape as slice_by_bb, keyed by category letter (V/F/S/C/L/T).
    """
    missing_nl = set()
    groups: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for rec in predictions:
        nl = rec.get("nl_query", "")
        bbs = bb_index.get(nl)
        if bbs is None:
            missing_nl.add(nl)
            continue
        cid = rec["config_id"]
        for cat in {_bb_category(bb) for bb in bbs}:
            groups[cid][cat].append(rec)

    if missing_nl:
        print(f"Warning: {len(missing_nl)} nl_query values not found in BB labels file.")

    result: dict = {}
    for cid, cat_map in groups.items():
        result[cid] = {}
        for cat, recs in cat_map.items():
            if len(recs) < min_n:
                continue
            result[cid][cat] = {
                "n": len(recs),
                "result_set_f1_strict": _mean([
                    0.0 if ((r.get("similarity") or {}).get("combined") or {}).get("f1") is None
                    else ((r.get("similarity") or {}).get("combined") or {}).get("f1")
                    for r in recs
                ]),
                "structural_f1": _mean([
                    ((r.get("metrics") or {}).get("structural") or {}).get("f1") for r in recs
                ]),
                "mean_baseline_f1": _mean([r.get("baseline_f1") for r in recs]),
                **_skill_block(recs),
                "combined_f1": _mean([
                    (r.get("similarity") or {}).get("combined", {}).get("f1")
                    for r in recs
                ]),
                "schema_valid_rate": _mean(
                    [1 if r["metrics"]["schema_valid"] else 0 for r in recs]
                ),
                "execution_success_rate": _mean(
                    [_execution_success(r) for r in recs]
                ),
            }
    return result


def _print_ascii_table(breakdown: dict, primary_metric: str = "result_set_f1_strict") -> None:
    config_ids = sorted(breakdown.keys())
    if not config_ids:
        print("No data.")
        return

    # Collect all BB labels present across all configs
    all_bbs_seen = set()
    for bb_map in breakdown.values():
        all_bbs_seen.update(bb_map.keys())
    bbs_to_show = [bb for bb in BB_ORDER if bb in all_bbs_seen]

    # Column widths — truncate config_id for display
    col_width = 8
    bb_col = 6
    label_header = primary_metric.replace("_rate", "").replace("_", " ")

    # Abbreviate config IDs for column headers
    def short_id(cid: str) -> str:
        parts = cid.split("_")
        abbrevs = []
        for p in parts:
            if p in ("base", "fine", "tuned", "finetuned"):
                abbrevs.append(p[:2])
            elif p.startswith("schema"):
                abbrevs.append("sch")
            elif p in ("on", "off"):
                abbrevs.append(p)
            elif p in ("allowed", "entities", "only"):
                abbrevs.append("ent")
            elif p in ("grammar", "guided"):
                abbrevs.append("grm")
            elif p in ("chain", "of", "thought", "plus"):
                abbrevs.append("cot")
            else:
                abbrevs.append(p[:4])
        return "_".join(abbrevs)[:col_width]

    col_labels = [short_id(cid) for cid in config_ids]

    header = f"{'BB':<{bb_col}}" + "".join(f" {c:>{col_width}}" for c in col_labels)
    print(f"\n--- {label_header.upper()} by Building Block ---")
    print(header)
    print("-" * len(header))

    for bb in bbs_to_show:
        row = f"{bb:<{bb_col}}"
        for cid in config_ids:
            val = (breakdown.get(cid) or {}).get(bb, {}).get(primary_metric)
            if val is None:
                row += f" {'—':>{col_width}}"
            else:
                row += f" {val:>{col_width}.2f}"
        print(row)

    # Also print count column based on first config
    print()
    print(f"{'BB':<{bb_col}} {'n':>{col_width}}  (example count, first config)")
    print("-" * (bb_col + col_width + 30))
    first_cid = config_ids[0] if config_ids else None
    for bb in bbs_to_show:
        n = (breakdown.get(first_cid) or {}).get(bb, {}).get("n", 0) if first_cid else 0
        print(f"{bb:<{bb_col}} {n:>{col_width}}")


def _print_category_table(breakdown: dict, primary_metric: str = "result_set_f1_strict") -> None:
    config_ids = sorted(breakdown.keys())
    if not config_ids:
        print("No category data.")
        return

    cats_seen = set()
    for cat_map in breakdown.values():
        cats_seen.update(cat_map.keys())
    cats_to_show = [c for c in CATEGORY_ORDER if c in cats_seen]

    col_width = 8
    cat_col = 14
    label_header = primary_metric.replace("_rate", "").replace("_", " ")

    def short_id(cid: str) -> str:
        return cid[:col_width]

    col_labels = [short_id(cid) for cid in config_ids]
    header = f"{'Category':<{cat_col}}" + "".join(f" {c:>{col_width}}" for c in col_labels)
    print(f"\n--- {label_header.upper()} by BB Category ---")
    print(header)
    print("-" * len(header))

    for cat in cats_to_show:
        label = f"{cat}:{CATEGORY_NAMES.get(cat, cat)}"
        row = f"{label:<{cat_col}}"
        for cid in config_ids:
            val = (breakdown.get(cid) or {}).get(cat, {}).get(primary_metric)
            if val is None:
                row += f" {'—':>{col_width}}"
            else:
                row += f" {val:>{col_width}.2f}"
        print(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Path to eval_runs/<run_id>/")
    parser.add_argument("--labels", default="testset_bb_labels.json",
                        help="Path to testset_bb_labels.json (default: cwd)")
    parser.add_argument("--out", default="bb_breakdown.json",
                        help="Output filename (written inside --run_dir)")
    parser.add_argument("--min_n", type=int, default=1,
                        help="Minimum examples per (config, BB) cell to include (default: 1)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    predictions_path = run_dir / "predictions.json"
    labels_path = Path(args.labels)
    out_path = run_dir / args.out

    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions.json not found in {run_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"BB labels file not found: {labels_path}\n"
            "Run: python tag_testset_bb.py"
        )

    predictions = _load_json(predictions_path)
    labels_data = _load_json(labels_path)

    print(f"Loaded {len(predictions)} prediction records from {predictions_path}")
    print(f"Loaded {len(labels_data)} BB-label entries from {labels_path}")

    bb_index = _build_bb_index(labels_data)
    breakdown = slice_by_bb(predictions, bb_index, min_n=args.min_n)
    category_breakdown = slice_by_category(predictions, bb_index, min_n=args.min_n)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(breakdown, f, indent=2)
    print(f"\nWrote {args.out} to {out_path}")

    cat_out_path = run_dir / "bb_category_breakdown.json"
    with open(cat_out_path, "w", encoding="utf-8") as f:
        json.dump(category_breakdown, f, indent=2)
    print(f"Wrote bb_category_breakdown.json to {cat_out_path}")

    config_count = len(breakdown)
    bb_counts = defaultdict(int)
    for bb_map in breakdown.values():
        for bb in bb_map:
            bb_counts[bb] += 1
    print(f"Configs: {config_count}  |  BB labels covered: {len(bb_counts)}/{len(BB_ORDER)}")

    _print_ascii_table(breakdown, primary_metric="result_set_f1_strict")
    _print_category_table(category_breakdown, primary_metric="result_set_f1_strict")


if __name__ == "__main__":
    main()
