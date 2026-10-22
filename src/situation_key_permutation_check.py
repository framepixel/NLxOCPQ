#!/usr/bin/env python3
"""Read-only check: how much does the situation key's dependence on variable ORDER move F1?

WHY
---
Result F1 compares root situations keyed as (event indices, object indices, violated), with the
indices of each kind ordered by variable index (`eval_testset._binding_entity_indices`). The key
is therefore independent of how variables are *numbered*, but not of the *order* in which two
variables of the same kind are declared: a prediction that declares (customers, employees) where
gold declares (employees, customers) returns the same situations under a different key, and scores
0 (Thesis/script/FLAGS.md TH-21, NS-18).

This script measures that sensitivity on finished runs without re-running anything. For every
record with both a gold and a predicted root result in the run's result store it compares

    base F1   the F1 of the stored situation sets, as the metric scored them, and
    perm F1   the best F1 over all reorderings of the prediction's event variables and of its
              object variables -- one reordering per record, applied to every situation, which is
              exactly what declaring the variables in a different order does.

A reordering that raises F1 marks a record the key penalised for declaration order alone. Records
without a stored result (never executed, truncated, or over the store's size cap) are skipped:
their strict F1 is 0 whatever the key.

Measured 2026-10-12 on every scored run: one question changes, "Which sales employees confirmed
orders for which customers?" -- non-discriminating -- so every discriminating figure is unchanged.

Usage (from the repo root):
    python src/situation_key_permutation_check.py eval_runs/matrix_v2/merged [more runs ...]
"""
import argparse
import gzip
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Sequence, Set, Tuple

import analysis_common as ac
import paths

Situation = Tuple[tuple, tuple, bool]

# Reorderings are enumerated exhaustively; past this many variables of one kind the factorial
# explodes, and no testset query declares that many.
MAX_VARIABLES_PER_KIND = 6


def load_situations(store: Path, digest: Optional[str]) -> Optional[Set[Situation]]:
    """A stored root result as a set of situations; None if absent or truncated."""
    if not digest:
        return None
    path = store / digest[:2] / f"{digest}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        document = json.load(handle)
    if str(document.get("truncated")) in ("True", "true"):
        return None
    situations = document["situations"]
    if isinstance(situations, str):
        situations = json.loads(situations.replace("False", "false").replace("True", "true"))
    return {(tuple(events), tuple(objects), bool(violated))
            for events, objects, violated in situations}


def f1(prediction: Set[Situation], gold: Set[Situation]) -> float:
    """Set F1 with the metric's conventions: two empty sets agree, one empty set scores 0."""
    if not prediction and not gold:
        return 1.0
    if not prediction or not gold:
        return 0.0
    overlap = len(prediction & gold)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(prediction), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def best_reordering_f1(prediction: Set[Situation], gold: Set[Situation]) -> Tuple[float, bool]:
    """(best F1 over variable reorderings, whether any reordering was possible)."""
    best = f1(prediction, gold)
    event_widths = {len(events) for events, _, _ in prediction}
    object_widths = {len(objects) for _, objects, _ in prediction}
    if len(event_widths) != 1 or len(object_widths) != 1:
        return best, False
    n_events, n_objects = event_widths.pop(), object_widths.pop()
    if n_events < 2 and n_objects < 2:
        return best, False
    if max(n_events, n_objects) > MAX_VARIABLES_PER_KIND:
        return best, False
    for event_order in itertools.permutations(range(n_events)):
        for object_order in itertools.permutations(range(n_objects)):
            reordered = {(tuple(events[i] for i in event_order),
                          tuple(objects[i] for i in object_order), violated)
                         for events, objects, violated in prediction}
            best = max(best, f1(reordered, gold))
    return best, True


def check_run(run: ac.Run, store: Path, discriminating_questions: int = 135) -> dict:
    """The sensitivity report for one run, reading result sets from `store`.

    Raises:
        SystemExit: if no record could be scored. A run whose store is missing would otherwise
            report "0 records rise" -- an all-clear measured on nothing.
    """
    scored = exposed = 0
    changed = []
    gain_by_config: Dict[str, float] = defaultdict(float)
    for record in run.records:
        reference = record.get("result_ref") or {}
        gold = load_situations(store, reference.get("gold"))
        prediction = load_situations(store, reference.get("prediction"))
        if gold is None or prediction is None:
            continue
        scored += 1
        base = f1(prediction, gold)
        best, reorderable = best_reordering_f1(prediction, gold)
        exposed += reorderable
        if record.get("discriminating"):
            gain_by_config[record.get("config_id")] += best - base
        if best > base + 1e-12:
            changed.append({"config_id": record.get("config_id"),
                            "question": record.get("nl_query"),
                            "log": ac.entry_key(record)[1],
                            "discriminating": record.get("discriminating"),
                            "repeat": record.get("repeat"),
                            "base_f1": round(base, 6), "best_f1": round(best, 6)})
    if scored == 0:
        raise SystemExit(f"{run.provenance['run']}: no record has both results in {store}; "
                         "pass the store that holds them with --result-store")
    max_gain = max((gain / discriminating_questions for gain in gain_by_config.values()),
                   default=0.0)
    return {
        "run": run.provenance["run"],
        "result_store": ac.repo_relative(store),
        "records_with_both_results": scored,
        "records_with_reorderable_prediction": exposed,
        "records_whose_f1_rises": len(changed),
        "distinct_questions_affected": len({c["question"] for c in changed}),
        "discriminating_records_affected": sum(1 for c in changed if c["discriminating"]),
        "max_mean_discriminating_gain_per_condition": max_gain,
        "changed": changed,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="Run directories with a result_store/.")
    parser.add_argument("--result-store", type=Path, default=None,
                        help="Store to read instead of each run's own result_store/ (a rescored "
                             "copy such as matrix_v2/merged_regold_20260922 has none).")
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    reports, runs = [], []
    for run_dir in args.runs:
        run = ac.load_run(run_dir)
        runs.append(run)
        store = args.result_store or run.path / "result_store"
        if not store.is_absolute():
            store = paths.REPO_ROOT / store
        if not store.is_dir():
            raise SystemExit(f"{run.path}: no result store at {store}; pass --result-store")
        report = check_run(run, store)
        reports.append(report)
        print(f"{report['run']} ({report['records_with_both_results']} records scored): "
              f"{report['records_whose_f1_rises']} record(s) rise, "
              f"{report['distinct_questions_affected']} question(s), "
              f"{report['discriminating_records_affected']} discriminating; largest mean "
              f"discriminating gain per condition "
              f"{report['max_mean_discriminating_gain_per_condition']:.4f}")
        for item in report["changed"][:10]:
            print(f"    {item['config_id']}: {item['base_f1']} -> {item['best_f1']}  "
                  f"{item['question'][:70]}")
    if not args.no_write:
        out = ac.write_json("situation_key_permutation", {"runs": reports}, runs)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
