#!/usr/bin/env python3
"""What the testset is made of, and what its floor means: the Appendix A numbers.

WHY
---
The testset description in notes 11 (§1, §2, §9.3–§9.4, §11.2, §12.2) was verified against the
files but produced by no committed script (Thesis/script/FLAGS.md NS-09, NS-10). This script
regenerates every one of those figures from `data/` and the discriminability audit, and checks them
against `configs/analysis_expected.json`.

WHAT IT REPORTS
---------------
size and strata   entries, discriminating / non-discriminating (from the audit's verdicts), per
                  log, the ≥10-per-log and ≥5-per-block floors, multi-root trees, trees whose
                  root is not node 0 (the case `eval_testset.root_node_index` exists for).
exclusions        `testset_excluded.json` by reason.
provenance        169 = carried over (in the 192- and 150-entry versions) + restored (in the 192
                  only) + later additions (in the 150 only), matched by question against the
                  archived audits of both versions; their union must be 169 + the 44 excluded.
shapes            boxes per gold tree.
blocks            entries per building block (`testset_bb_labels.json`).
floor by kind     compliance (some box carries constraints) vs retrieval. §11.2 describes the
                  split as "root has constraints", but its figures (154 / 15, mean no-op F1
                  0.617 / 0.562) are those of "any box"; the root-only split gives 140 / 29. For
                  compliance the no-op F1 has the closed form 1 − violation rate: its fit is
                  reported as the median absolute error and the share within 0.05.
high floors       discriminating entries whose floor exceeds 0.90, by the no-op F1 (the
                  published 32) and by the trivial floor that skill uses (46).
skill grid        the skill score of synthetic models (§12.2), against both floors: the no-op F1
                  the published grid used, and the trivial floor (§27) that `skill_score` uses
                  today. "Fails 80%" is taken in expectation: each entry scores 0.2 × skill(1) +
                  0.8 × skill(0).

Usage (from the repo root):
    python src/testset_composition.py
"""
import argparse
import json
import os
import statistics
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import analysis_common as ac
import paths

EXPECTED_PATH = paths.CONFIGS / "analysis_expected.json"
AUDIT_PATH = paths.EVAL_RUNS / "testset_discriminability.json"
# The archived audits of the two earlier testset versions; each lists its entries' questions.
ARCHIVE_192 = paths.EVAL_RUNS / "_archive_audit_20260808" / "testset_discriminability.json"
ARCHIVE_150 = paths.EVAL_RUNS / "_archive_audit_150" / "testset_discriminability.json"

SYNTHETIC_F1 = {"horrible": 0.0, "good_085": 0.85, "excellent_097": 0.97}
FIT_TOLERANCE = 0.05
HIGH_FLOOR = 0.90


def _read(path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def skill(f1: float, floor: Optional[float]) -> Optional[float]:
    """The evaluator's skill score (`eval_testset._skill_score`), re-declared and pinned by test."""
    if f1 is None or floor is None or floor >= 1.0:
        return None
    if f1 >= floor:
        return (f1 - floor) / (1.0 - floor)
    return (f1 - floor) / floor


def roots(query: dict) -> List[int]:
    """Indices of the boxes no box lists as a child."""
    nodes = query.get("nodes") or []
    children = {c for node in nodes for c in (node.get("Box") or [None, []])[1]}
    return [i for i in range(len(nodes)) if i not in children]


def has_constraints(query: dict, root_only: bool = False) -> bool:
    """Compliance query: some box (or, with `root_only`, the root box) carries constraints."""
    nodes = query.get("nodes") or []
    if root_only:
        found = roots(query)
        nodes = [nodes[found[0] if found else 0]]
    return any(node["Box"][0].get("constraints") for node in nodes)


def skill_grid(entries: List[dict], floor_key: str) -> Dict[str, Optional[float]]:
    """Mean skill of each synthetic model over entries with a defined skill."""
    grid = {}

    def mean_over(per_entry):
        values = [v for v in (per_entry(e) for e in entries) if v is not None]
        return sum(values) / len(values) if values else None

    for name, f1 in SYNTHETIC_F1.items():
        grid[name] = mean_over(lambda e, f1=f1: skill(f1, e[floor_key]))
    grid["always_trivial"] = mean_over(lambda e: skill(e["baseline_f1"], e[floor_key]))

    def fails(e):
        hit, miss = skill(1.0, e[floor_key]), skill(0.0, e[floor_key])
        return None if hit is None or miss is None else 0.2 * hit + 0.8 * miss
    grid["fails_80_percent"] = mean_over(fails)
    return grid


def compose() -> Dict[str, Any]:
    test = _read(paths.DATA / "testset.json")
    excluded = _read(paths.DATA / "testset_excluded.json")
    labels = _read(paths.DATA / "testset_bb_labels.json")
    audit = _read(AUDIT_PATH)["entries"]
    if [a["nl_query"] for a in audit] != [t["nl_query"] for t in test]:
        raise SystemExit("the discriminability audit is not aligned with data/testset.json")
    non_disc = {a["index"] for a in audit if a["verdict"] == "non_discriminating"}
    disc = [a for a in audit if a["index"] not in non_disc]

    questions = {t["nl_query"] for t in test}
    old_192 = {e["nl_query"] for e in _read(ARCHIVE_192)["entries"]}
    old_150 = {e["nl_query"] for e in _read(ARCHIVE_150)["entries"]}

    per_log = Counter(os.path.basename(t["ocel_path"]) for t in test)
    per_log_disc = Counter(os.path.basename(test[a["index"]]["ocel_path"]) for a in disc)
    blocks = Counter(b for row in labels for b in row["bb_labels"])
    boxes = Counter(len(t["ocpq_query"]["nodes"]) for t in test)
    root_lists = [roots(t["ocpq_query"]) for t in test]

    kinds = {"compliance": [], "retrieval": []}
    for entry, row in zip(test, audit):
        kind = "compliance" if has_constraints(entry["ocpq_query"]) else "retrieval"
        kinds[kind].append(row)
    errors = [abs(row["baseline_f1"] - (1 - row["violation_rate"]))
              for row in kinds["compliance"]
              if row.get("baseline_f1") is not None and row.get("violation_rate") is not None]

    results = {
        "entries": len(test),
        "discriminating": len(disc),
        "non_discriminating": len(non_disc),
        "per_log": {log: {"entries": per_log[log], "discriminating": per_log_disc[log]}
                    for log in sorted(per_log, key=lambda l: (-per_log[l], l))},
        "min_entries_per_log": min(per_log.values()),
        "excluded_by_reason": dict(Counter(e["reason"] for e in excluded).most_common()),
        "provenance": {
            "carried_over": len(questions & old_150 & old_192),
            "restored": len(questions & (old_192 - old_150)),
            "later_additions": len(questions & (old_150 - old_192)),
            "unaccounted": len(questions - old_150 - old_192),
        },
        "union_of_versions": len(old_192 | old_150),
        "boxes_per_tree": dict(sorted(boxes.items())),
        "multi_root_trees": sum(1 for r in root_lists if len(r) != 1),
        "root_not_node_0": sum(1 for r in root_lists if r and r[0] != 0),
        "blocks_present": len(blocks),
        "min_entries_per_block": min(blocks.values()),
        "max_entries_per_block": max(blocks.values()),
        "thinnest_blocks": sorted(b for b, n in blocks.items() if n == min(blocks.values())),
        "entries_per_block": dict(sorted(blocks.items(), key=lambda kv: (kv[0][0],
                                                                         int(kv[0][1:])))),
        "kind_counts": {k: len(v) for k, v in kinds.items()},
        "kind_counts_root_only": {
            "compliance": sum(1 for t in test if has_constraints(t["ocpq_query"], True)),
            "retrieval": sum(1 for t in test if not has_constraints(t["ocpq_query"], True))},
        "mean_noop_floor_by_kind": {k: statistics.mean(r["baseline_f1"] for r in v)
                                    for k, v in kinds.items() if v},
        "compliance_fit_median_error": statistics.median(errors),
        "compliance_fit_within_005": sum(1 for e in errors if e <= FIT_TOLERANCE) / len(errors),
        "discriminating_floor_above_090": sum(1 for a in disc if a["baseline_f1"] > HIGH_FLOOR),
        "discriminating_trivial_floor_above_090": sum(1 for a in disc
                                                      if a["trivial_floor"] > HIGH_FLOOR),
        "mean_noop_f1_discriminating": statistics.mean(a["baseline_f1"] for a in disc),
        "mean_trivial_floor_discriminating": statistics.mean(a["trivial_floor"] for a in disc),
        "skill_grid_noop_floor": skill_grid(disc, "baseline_f1"),
        "skill_grid_trivial_floor": skill_grid(disc, "trivial_floor"),
    }
    return results


def reconcile(results: dict, expected: dict) -> List[str]:
    """Differences from the published figures, compared at the published precision."""
    problems = []

    def same(got, want) -> bool:
        if isinstance(want, dict):
            return isinstance(got, dict) and all(same(got.get(k), v) for k, v in want.items())
        if isinstance(want, float):
            digits = len(repr(want).split(".")[1]) if "." in repr(want) else 0
            return got is not None and round(got, max(digits, 1)) == round(want, max(digits, 1))
        return got == want

    for key, entry in expected.items():
        if key.startswith("_") or key not in results:
            continue
        if not same(results[key], entry["value"]):
            problems.append(f"{key}: computed {results[key]}, notes {entry['value']} "
                            f"({entry['section']})")
    return problems


def tables(results: dict) -> List[tuple]:
    """(name, tabular, caption, label) for Appendix A."""
    log_lines = ["  \\small", "  \\begin{tabular}{@{}lrr@{}}", "    \\toprule",
                 "    Log & Entries & Discriminating \\\\", "    \\midrule"]
    for log, row in results["per_log"].items():
        log_lines.append(f"    \\texttt{{{log.replace('_', chr(92) + '_')}}} & {row['entries']} & "
                         f"{row['discriminating']} \\\\")
    log_lines += ["    \\midrule",
                  f"    Total & {results['entries']} & {results['discriminating']} \\\\",
                  "    \\bottomrule", "  \\end{tabular}"]

    blocks = list(results["entries_per_block"].items())
    half = (len(blocks) + 1) // 2
    block_lines = ["  \\small", "  \\begin{tabular}{@{}lrlr@{}}", "    \\toprule",
                   "    Block & Entries & Block & Entries \\\\", "    \\midrule"]
    for left, right in zip(blocks[:half], blocks[half:] + [("", "")]):
        block_lines.append(f"    {left[0]} & {left[1]} & {right[0]} & {right[1]} \\\\")
    block_lines += ["    \\bottomrule", "  \\end{tabular}"]

    grid_names = [("horrible", "F1 0 everywhere"), ("fails_80_percent", "fails 80\\%, perfect "
                  "otherwise"), ("always_trivial", "always the no-op query"),
                  ("good_085", "F1 0.85 everywhere"), ("excellent_097", "F1 0.97 everywhere")]
    grid_lines = ["  \\small", "  \\begin{tabular}{@{}lrr@{}}", "    \\toprule",
                  "    Synthetic model & \\makecell[r]{Skill against\\\\no-op F1} & "
                  "\\makecell[r]{Skill against\\\\trivial floor} \\\\", "    \\midrule"]
    for key, name in grid_names:
        grid_lines.append(f"    {name} & {ac.fmt(results['skill_grid_noop_floor'][key], signed=True)}"
                          f" & {ac.fmt(results['skill_grid_trivial_floor'][key], signed=True)} \\\\")
    grid_lines += ["    \\bottomrule", "  \\end{tabular}"]
    return [
        ("testset_per_log", "\n".join(log_lines),
         "Testset entries per log, and how many are discriminating.", "tab:testset:logs"),
        ("testset_blocks", "\n".join(block_lines),
         "Testset entries using each building block.", "tab:testset:blocks"),
        ("testset_skill_grid", "\n".join(grid_lines),
         "Mean skill of synthetic models over the 135 discriminating questions, against the "
         "no-op query's F1 and against the trivial floor.", "tab:testset:skill"),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    results = compose()
    problems = reconcile(results, _read(EXPECTED_PATH)["testset_composition"])
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("per_log", "entries_per_block")}, indent=1)[:4000])
    print(f"\n{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if args.no_write:
        return 0
    out = ac.write_json("testset_composition", {"results": results,
                                                "differences_from_notes": problems})
    print(f"\nwrote {out}")
    for name, tabular, caption, label in tables(results):
        print(f"wrote {ac.write_tex(name, tabular, caption, label)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
