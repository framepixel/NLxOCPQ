#!/usr/bin/env python3
"""GPU time and per-case run time behind every reported evaluation (Appendix D).

WHY
---
Notes 11 quotes GPU-hours and run-time percentiles (§39 "94.8 GPU-hours, median 32.7 s per run",
§18.2's cap percentiles, per-rung costs in §41–§45) that no script produced (Thesis/script/
FLAGS.md NS-04, NS-15). There are no SLURM accounting exports in the repo, but every shard wrote a
`checkpoint.json` with its elapsed time and every record its own duration, which is enough.

TWO DIFFERENT HOURS -- always say which
---------------------------------------
case hours    the sum of every record's `total_duration_sec`: generation, execution and scoring
              of the cases themselves. §39's "94.8 GPU-hours" is this figure.
shard hours   the sum of the shards' checkpoint `elapsed_sec`. One shard ran on one H100, so this
              is the GPU wall time of the work, including model loading, the gold queries and the
              summaries -- 113.7 h for the matrix. The GPU time SLURM charged is at least this
              (it adds environment start-up); quote shard hours as "GPU-hours".

Per-case durations are reported as median / p90 / p99 / max per group, and per model for the
matrix. Training time comes from the training logs where they survive (src/training_summary.py).

Usage (from the repo root):
    python src/compute_accounting.py
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import analysis_common as ac
import paths

RUNS_PATH = paths.CONFIGS / "compute_runs.json"


def shard_dirs(group: dict) -> List[Path]:
    """The group's shard directories: glob matches minus rescored copies and exclusions."""
    found = set()
    for pattern in group["globs"]:
        for match in glob.glob(str(paths.REPO_ROOT / pattern)):
            path = Path(match)
            if not path.is_dir() or path.name.endswith("_recomputed"):
                continue
            if any(token in path.name for token in group.get("exclude", ())):
                continue
            found.add(path)
    if not found:
        raise SystemExit(f"group {group['id']}: no shard directory matches {group['globs']}")
    return sorted(found)


def percentiles(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"median": None, "p90": None, "p99": None, "max": None}
    array = np.asarray(values)
    return {"median": float(np.median(array)), "p90": float(np.percentile(array, 90)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max())}


def account(group: dict) -> dict:
    """Shard and case hours, durations, and per-model seconds for one group."""
    shards = shard_dirs(group)
    elapsed, durations = 0.0, []
    per_model: Dict[str, List[float]] = defaultdict(list)
    records = 0
    missing_checkpoint = []
    for shard in shards:
        checkpoint = shard / "checkpoint.json"
        if checkpoint.exists():
            with open(checkpoint, encoding="utf-8") as handle:
                elapsed += float(json.load(handle).get("elapsed_sec") or 0.0)
        else:
            missing_checkpoint.append(shard.name)
        with open(shard / "predictions.json", encoding="utf-8") as handle:
            for record in json.load(handle):
                records += 1
                seconds = record.get("total_duration_sec")
                if seconds is not None:
                    durations.append(float(seconds))
                    per_model[record.get("model_label")].append(float(seconds))
    return {
        "id": group["id"],
        "label": group["label"],
        "corpus": group.get("corpus"),
        "shards": len(shards),
        "shards_without_checkpoint": missing_checkpoint,
        "records": records,
        "case_hours": sum(durations) / 3600,
        "shard_hours": elapsed / 3600,
        "case_seconds": percentiles(durations),
        "seconds_per_record_by_model": {model: float(np.median(values))
                                        for model, values in sorted(per_model.items())},
    }


def reconcile(row: dict, expected: dict) -> List[str]:
    problems = []
    checks = {"case_hours": row["case_hours"], "median_case_seconds": row["case_seconds"]["median"],
              "p90_case_seconds": row["case_seconds"]["p90"],
              "p99_case_seconds": row["case_seconds"]["p99"]}
    for key, got in checks.items():
        if key in expected and round(got, 1) != round(float(expected[key]), 1) and not (
                isinstance(expected[key], int) and round(got) == expected[key]):
            problems.append(f"{key}: computed {got:.1f}, notes {expected[key]} "
                            f"({expected.get('section')})")
    return problems


def tex_table(rows: List[dict]) -> str:
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrrrr@{}}",
             "    \\toprule",
             "    Evaluation & Shards & Records & \\makecell[r]{GPU-hours\\\\(shards)} & "
             "\\makecell[r]{Case\\\\hours} & \\makecell[r]{Median s\\\\per case} & "
             "\\makecell[r]{p90 s\\\\per case} \\\\",
             "    \\midrule"]
    for row in rows:
        label = row["label"] + (" (old corpus)" if row["corpus"] == "old" else "")
        lines.append(f"    {label} & {row['shards']} & {row['records']:,} & "
                     f"{row['shard_hours']:.1f} & {row['case_hours']:.1f} & "
                     f"{row['case_seconds']['median']:.1f} & {row['case_seconds']['p90']:.0f} \\\\"
                     .replace(",", "{,}"))
    total_shard = sum(r["shard_hours"] for r in rows)
    total_case = sum(r["case_hours"] for r in rows)
    lines += ["    \\midrule",
              f"    Total & {sum(r['shards'] for r in rows)} & "
              f"{sum(r['records'] for r in rows):,} & {total_shard:.1f} & {total_case:.1f} & & \\\\"
              .replace(",", "{,}"),
              "    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(RUNS_PATH, encoding="utf-8") as handle:
        groups = json.load(handle)["groups"]
    rows, problems = [], []
    for group in groups:
        row = account(group)
        rows.append(row)
        found = reconcile(row, group.get("expected") or {})
        problems += [f"{group['id']}: {p}" for p in found]
        q = row["case_seconds"]
        print(f"{row['label']:<40} shards {row['shards']:>3} records {row['records']:>5} "
              f"GPU-h {row['shard_hours']:6.1f} case-h {row['case_hours']:6.1f} "
              f"median {q['median']:6.1f} p90 {q['p90']:6.1f} max {q['max']:7.1f}")
    matrix = next(r for r in rows if r["id"] == "matrix")
    print("\nmatrix, median seconds per record by model:")
    for model, seconds in matrix["seconds_per_record_by_model"].items():
        print(f"  {model:<28} {seconds:6.1f}")
    print(f"\ntotal GPU-hours (shards) {sum(r['shard_hours'] for r in rows):.1f}, "
          f"case hours {sum(r['case_hours'] for r in rows):.1f}")
    print(f"{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if args.no_write:
        return 0
    out = ac.write_json("compute_accounting", {"groups": rows, "differences_from_notes": problems})
    table = ac.write_tex("compute_accounting", tex_table(rows),
                         "GPU time of every evaluation. GPU-hours sum the shards' wall time on one "
                         "H100 each, including model loading; case hours sum the per-case time "
                         "alone. The training runs are not included.", "tab:compute")
    print(f"wrote {out}\nwrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
