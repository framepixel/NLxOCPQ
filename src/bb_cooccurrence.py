#!/usr/bin/env python3
"""Empirical co-occurrence of OCPQ building blocks in the testset.

Answers questions like "do F11 and V5 ever appear in the same query?" purely
from the data — no claims about what the OCPQ grammar *permits*, only what the
handcrafted testset actually contains. Builds two symmetric co-occurrence
matrices from testset_bb_labels.json:

  * BB x BB        (42 x 42) — written to JSON in full
  * Category x Category (6 x 6, V/F/S/C/L/T) — also rendered as an ASCII matrix

Each diagonal cell is the number of queries containing that BB / category; each
off-diagonal cell is the number of queries containing BOTH.

Optionally, with --run_dir, joins a run's predictions.json to report mean
combined_f1 on the most common BB-pair co-occurrence patterns, i.e. how well
models do when two building blocks must be combined.

Usage:
    python bb_cooccurrence.py [--labels testset_bb_labels.json]
                              [--out bb_cooccurrence.json]
                              [--run_dir eval_runs/<run_id>] [--top 15]
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from slice_results_by_bb import (
    BB_ORDER,
    CATEGORY_ORDER,
    CATEGORY_NAMES,
    _bb_category,
)


def _load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean(vals: list):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def build_cooccurrence(labels_data: list, keys: list, mapper=lambda x: x) -> dict:
    """Symmetric co-occurrence counts over *keys* (BB ids or category letters).

    mapper maps a raw BB id to the unit of interest (identity for BBs, _bb_category
    for categories). Diagonal = #queries containing the unit; off-diagonal =
    #queries containing both units.
    """
    counts: dict[str, dict[str, int]] = {a: defaultdict(int) for a in keys}
    for entry in labels_data:
        units = sorted({mapper(bb) for bb in entry.get("bb_labels", []) if mapper(bb) in counts})
        for u in units:
            counts[u][u] += 1  # diagonal: presence count
        for a, b in combinations(units, 2):
            counts[a][b] += 1
            counts[b][a] += 1
    # materialise as plain dicts in canonical order
    return {a: {b: counts[a].get(b, 0) for b in keys} for a in keys}


def _print_matrix(matrix: dict, keys: list, title: str, label_fn=lambda k: k) -> None:
    present = [k for k in keys if matrix.get(k, {}).get(k, 0) > 0]
    if not present:
        print(f"\n{title}: no data.")
        return
    cw = 4
    lab_w = max(len(label_fn(k)) for k in present) + 1
    header = " " * lab_w + "".join(f"{k:>{cw}}" for k in present)
    print(f"\n--- {title} ---")
    print(header)
    for a in present:
        row = f"{label_fn(a):<{lab_w}}"
        for b in present:
            v = matrix[a][b]
            row += f"{v:>{cw}}" if v else f"{'.':>{cw}}"
        print(row)
    print("(diagonal = #queries containing it; off-diagonal = #queries containing both)")


def cooccurrence_performance(predictions: list, bb_index: dict, top: int) -> list:
    """Mean combined_f1 per BB-pair, for the most frequently co-occurring pairs."""
    pair_recs: dict[tuple, list] = defaultdict(list)
    for rec in predictions:
        bbs = bb_index.get(rec.get("nl_query", ""))
        if not bbs:
            continue
        for a, b in combinations(sorted(set(bbs)), 2):
            pair_recs[(a, b)].append(rec)
    rows = []
    for (a, b), recs in pair_recs.items():
        # count distinct test cases (nl_query) so frequency isn't inflated by configs
        n_cases = len({r.get("nl_query") for r in recs})
        rows.append({
            "pair": f"{a}+{b}",
            "n_cases": n_cases,
            "n_records": len(recs),
            "combined_f1": _mean([
                (r.get("similarity") or {}).get("combined", {}).get("f1") for r in recs
            ]),
        })
    rows.sort(key=lambda r: (-r["n_cases"], r["pair"]))
    return rows[:top]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="data/testset_bb_labels.json")
    parser.add_argument("--out", default="data/bb_cooccurrence.json")
    parser.add_argument("--run_dir", default="", help="Optional eval run dir to join predictions for per-pair performance.")
    parser.add_argument("--top", type=int, default=15, help="Top-N co-occurring pairs to report performance for.")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    if not labels_path.exists():
        raise FileNotFoundError(f"BB labels file not found: {labels_path}\nRun: python tag_testset_bb.py")
    labels_data = _load_json(labels_path)
    print(f"Loaded {len(labels_data)} BB-label entries from {labels_path}")

    bb_matrix = build_cooccurrence(labels_data, BB_ORDER)
    cat_matrix = build_cooccurrence(labels_data, CATEGORY_ORDER, mapper=_bb_category)

    output = {"bb_cooccurrence": bb_matrix, "category_cooccurrence": cat_matrix}

    if args.run_dir:
        run_dir = Path(args.run_dir)
        predictions_path = run_dir / "predictions.json"
        if predictions_path.exists():
            predictions = _load_json(predictions_path)
            bb_index = {e["nl_query"]: e["bb_labels"] for e in labels_data}
            perf = cooccurrence_performance(predictions, bb_index, args.top)
            output["pair_performance_top"] = perf
            print(f"Joined {len(predictions)} predictions for per-pair performance.")
        else:
            print(f"Warning: {predictions_path} not found; skipping pair performance.")

    # With --run_dir the output carries that run's per-pair performance, so it belongs in
    # the run directory (CLAUDE.md 2.3) rather than the repo root, where the next run would
    # overwrite it. An --out with an explicit directory is always honoured as given.
    out_path = Path(args.out)
    if args.run_dir and out_path.parent == Path("."):
        out_path = Path(args.run_dir) / out_path
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")

    _print_matrix(cat_matrix, CATEGORY_ORDER, "Category co-occurrence",
                  label_fn=lambda c: f"{c}:{CATEGORY_NAMES.get(c, c)}")

    if "pair_performance_top" in output:
        print(f"\n--- Top {args.top} co-occurring BB pairs by combined_f1 ---")
        print(f"{'pair':<12}{'cases':>6}{'f1':>8}")
        for r in output["pair_performance_top"]:
            f1 = f"{r['combined_f1']:.2f}" if r["combined_f1"] is not None else "—"
            print(f"{r['pair']:<12}{r['n_cases']:>6}{f1:>8}")


if __name__ == "__main__":
    main()
