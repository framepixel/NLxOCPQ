#!/usr/bin/env python3
"""Level tables for the ablation rungs and the agentic rungs: one row per condition.

WHY
---
`paired_stats.py` gives the differences between rungs; the chapter also needs each rung's own
level -- F1, F1 where a result exists, skill, the evaluated share, exact matches, structural F1 --
which the notes quote in §36.1, §37, §41, §43.15, §44.9 and §45.9 and no script produced. The
conditions are the ones `configs/analysis_contrasts.json` already defines, so every rung is read
from exactly the run its paired contrast uses. Each run is checked against its summary.json first.

Every figure is over the 135 discriminating questions; a question's value is the mean over its
draws (evaluation 2 has two), so "exact" is the share of question-draws whose F1 is 1.

Usage (from the repo root):
    python src/rung_tables.py
"""
import argparse
import sys
from typing import Dict, List, Optional, Sequence

import analysis_common as ac
from paired_stats import RunCache, _condition_records, load_spec

# Which conditions form which table, in row order (names are the spec's condition keys).
TABLES = {
    "rungs_ablations": [("Decoding and arity (corrected corpus)", ["N0", "N1", "N2"]),
                        ("Structural ablation", ["S0", "S1", "S2", "S3"]),
                        ("Non-oracle system", ["B0", "B1", "B2", "B3"])],
    "rungs_agentic": [("Evaluation 1 (Ministral 3 14B)", ["A0", "A1p", "A1", "A1t", "A2"]),
                      ("Evaluation 2 (Qwen3.8)", ["Q_A0", "Q_A1p", "Q_A1"]),
                      ("Example lookup", ["Q_A1pS", "Q_A1S", "G_A1pS", "G_A1S"])],
}
CAPTIONS = {
    "rungs_ablations": "Each rung of the decoding, arity, structural and non-oracle ablations "
                       "(fine-tuned Llama 3.1 8B).",
    "rungs_agentic": "Each rung of the agentic evaluations; evaluation~2 values are the mean of "
                     "two draws per question.",
}


def level(records: List[dict]) -> Dict[str, Optional[float]]:
    """Per-question means (over draws) of every metric, then the mean over questions."""
    row = {}
    for metric in ("strict_f1", "executed_f1", "skill", "evaluated", "structural_f1"):
        values = ac.case_values(records, metric)
        row[metric] = ac.mean(values.values())
    disc = [r for r in records if r.get("discriminating") is True]
    row["exact"] = ac.mean(1.0 if ac.strict_f1(r) == 1.0 else 0.0 for r in disc)
    row["questions"] = len(ac.case_values(records, "strict_f1"))
    return row


def tabular(groups: List[tuple]) -> str:
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrrrr@{}}",
             "    \\toprule",
             "    Rung & F1 & \\makecell[r]{F1\\\\executed} & Skill & Evaluated & Exact & "
             "\\makecell[r]{Structural\\\\F1} \\\\",
             "    \\midrule"]
    for index, (title, rows) in enumerate(groups):
        if index:
            lines.append("    \\midrule")
        lines.append(f"    \\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
        for label, row in rows:
            lines.append(f"    {label} & {ac.fmt(row['strict_f1'])} & {ac.fmt(row['executed_f1'])}"
                         f" & {ac.fmt(row['skill'], 3)} & {ac.fmt_pct(row['evaluated'])} & "
                         f"{ac.fmt_pct(row['exact'])} & {ac.fmt(row['structural_f1'], 3)} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    spec = load_spec()
    cache = RunCache()
    payload = {}
    for table, groups in TABLES.items():
        rendered = []
        for title, names in groups:
            rows = []
            for name in names:
                condition = spec["conditions"][name]
                records = _condition_records(condition, cache.get(condition["run"]))
                row = level(records)
                rows.append((condition["label"], row))
                payload.setdefault(table, {})[name] = {"label": condition["label"], **row}
                print(f"{condition['label']:<34} F1 {row['strict_f1']:.4f} exec "
                      f"{row['executed_f1']:.4f} skill {row['skill']:+.3f} evaluated "
                      f"{row['evaluated']:.1%} exact {row['exact']:.1%} struct "
                      f"{row['structural_f1']:.3f}")
            rendered.append((title, rows))
        if not args.no_write:
            runs = [cache.get(spec["conditions"][n]["run"]) for _, names in groups for n in names]
            ac.write_tex(table, tabular(rendered),
                         CAPTIONS[table] + " Over the 135 discriminating questions; skill is "
                         "relative to the trivial floor.", f"tab:{table.replace('_', ':')}",
                         list({r.provenance['run']: r for r in runs}.values()))
    if not args.no_write:
        print(f"wrote {ac.write_json('rung_tables', payload, cache.runs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
