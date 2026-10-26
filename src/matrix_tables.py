#!/usr/bin/env python3
"""The matrix tables of the results chapter, from one merged matrix run.

WHY
---
The §39.2 tables in notes 11 were read off `summary.json` by hand (Thesis/script/FLAGS.md NS-02),
and `summary.json` cannot give them directly: its per-condition metrics are over all 169
questions, while the thesis reports the 135 discriminating ones (IC-08). Worse, the notes mix the
two -- §39.2's arm table is discriminating, but its per-strategy, validation on/off and seen/unseen
figures are over all 169, and on the discriminating questions the strategy ranking changes
(chain-of-thought-plus and grammar-guided swap). So every table here is written once per
population, and the population is in the caption.

WHAT IT WRITES (to `paths.THESIS_TABLES`, one file per table and population)
-------------------------------------------------------------------------------
matrix_arms_<pop>          per arm (model x base/fine-tuned), pooled over its six conditions
matrix_strategy_<pop>      per prompting strategy, pooled over 16 conditions
matrix_validation_<pop>    validation-repair on / off, pooled over 24 conditions
matrix_interaction_<pop>   strict F1 and evaluated rate, arm x strategy (replaces the old-corpus
                           "0 runnable in 1,352" of §21.4 with the matrix_v2 counts)
matrix_outcomes_<pop>      record status per arm (schema-invalid, parse-failed, ...)
matrix_best_<pop>          the five best single conditions

There is no familiar/unfamiliar table: `domain_seen` is the hard-coded list of logs that were
"seen" before the corpus was de-leaked, and `corpus_overlap_audit.py` measures zero exact type
overlap for every log, so the split measures nothing (FLAGS IC-06; the section was dropped).

Every mean is over records, which here equals the mean over questions: each condition holds each
question exactly once (the guard below and `case_values` both refuse duplicates).

Usage (from the repo root):
    python src/matrix_tables.py                                   # the reference run
    python src/matrix_tables.py --run eval_runs/matrix_v2/merged  # e.g. the pre-re-gold merge
"""
import argparse
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Optional, Sequence

import analysis_common as ac

# The reference: matrix_v2 rescored against the corrected gold of entry #25 (notes 11 §43.2).
DEFAULT_RUN = "eval_runs/matrix_v2/merged_regold_20260922"

# Presentation only: the order and names in which arms and strategies appear in the thesis.
MODELS = [
    ("llama31-8b", "Llama 3.1 8B"),
    ("ministral3-14b", "Ministral 3 14B"),
    ("ministral3-3b", "Ministral 3 3B"),
    ("gemma3-4b", "Gemma 3 4B"),
]
VARIANTS = [("base", "base"), ("finetuned", "fine-tuned")]
STRATEGIES = [
    ("schema-allowed-entities-only", "allowed entities only"),
    ("schema-grammar-guided", "grammar-guided"),
    ("schema-chain-of-thought-plus", "chain-of-thought plus"),
]
# The same names broken over two lines, for column headers.
STRATEGY_HEADERS = {
    "schema-allowed-entities-only": r"\makecell{allowed\\entities only}",
    "schema-grammar-guided": r"\makecell{grammar-\\guided}",
    "schema-chain-of-thought-plus": r"\makecell{chain-of-\\thought plus}",
}
STATUSES = [
    ("success", r"\makecell[r]{returned\\a result}"),
    ("execution_failed", r"\makecell[r]{execution\\failed}"),
    ("schema_invalid", r"\makecell[r]{schema-\\invalid}"),
    ("parse_failed", "unparseable"),
]

# Columns of the summary tables: (metric, header, formatter).
COLUMNS = [
    ("strict_f1", "F1", lambda v: ac.fmt(v)),
    ("executed_f1", r"\makecell[r]{F1\\executed}", lambda v: ac.fmt(v)),
    ("skill", "Skill", lambda v: ac.fmt(v, digits=3)),
    ("evaluated", "Evaluated", lambda v: ac.fmt_pct(v)),
    ("structural_f1", r"\makecell[r]{Structural\\F1}", lambda v: ac.fmt(v, digits=3)),
]


def arm_label(model_label: str) -> str:
    """'llama31-8b-finetuned' -> 'Llama 3.1 8B fine-tuned'."""
    for key, name in MODELS:
        for suffix, variant in VARIANTS:
            if model_label == f"{key}-{suffix}":
                return f"{name} {variant}"
    return model_label


def arms() -> List[str]:
    """Model labels in thesis order."""
    return [f"{key}-{suffix}" for key, _ in MODELS for suffix, _ in VARIANTS]


def metric_row(records: List[dict]) -> Dict[str, Optional[float]]:
    """Mean of every column metric over the records, plus the record count."""
    row = {metric: ac.mean(ac.METRICS[metric](r) for r in records) for metric, _, _ in COLUMNS}
    row["records"] = len(records)
    return row


def grouped(records: List[dict], key: Callable[[dict], object]) -> Dict[object, List[dict]]:
    """Records grouped by a key function, keeping first-seen order."""
    groups: Dict[object, List[dict]] = defaultdict(list)
    for record in records:
        groups[key(record)].append(record)
    return groups


def summary_tabular(rows: List[tuple]) -> str:
    """A booktabs tabularx: a wrapping label column, then `COLUMNS`. Rows are (label, row)."""
    head = " & ".join(["", *[header for _, header, _ in COLUMNS]])
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X"
             + "r" * len(COLUMNS) + "@{}}",
             "    \\toprule", f"    {head} \\\\", "    \\midrule"]
    for label, row in rows:
        if label is None:
            lines.append("    \\midrule")
            continue
        cells = [fmt(row[metric]) for metric, _, fmt in COLUMNS]
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def build(run: ac.Run, population: str) -> Dict[str, dict]:
    """Every table for one population, as {name: {"rows": ..., "caption": ..., "tabular": ...}}."""
    records = [r for r in run.records if ac.in_population(r, population)]
    label = ac.POPULATION_LABELS[population]
    tables: Dict[str, dict] = {}

    by_arm = grouped(records, lambda r: r.get("model_label"))
    arm_rows = [(arm_label(a), metric_row(by_arm[a])) for a in arms() if a in by_arm]
    arm_rows += [(None, None), ("pooled", metric_row(records))]
    tables["arms"] = {
        "rows": arm_rows,
        "caption": f"Matrix results per arm on {label}, each arm pooled over its three prompting "
                   f"strategies and validation-repair on and off. Skill is relative to the trivial "
                   f"floor; Evaluated is the share of records that \\texttt{{ocpq\\_cli}} "
                   f"evaluated.",
    }

    by_strategy = grouped(records, lambda r: r.get("prompt_strategy"))
    tables["strategy"] = {
        "rows": [(name, metric_row(by_strategy[key])) for key, name in STRATEGIES
                 if key in by_strategy],
        "caption": f"Matrix results per prompting strategy on {label}, each pooled over the "
                   f"eight arms and validation-repair on and off.",
    }

    by_validation = grouped(records, lambda r: bool(r.get("schema_enforced")))
    tables["validation"] = {
        "rows": [(name, metric_row(by_validation[flag]))
                 for flag, name in ((True, "validation-repair on"),
                                    (False, "validation-repair off"))
                 if flag in by_validation],
        "caption": f"Matrix results with validation-repair on and off on {label}, each pooled "
                   f"over the eight arms and three prompting strategies.",
    }

    for name in ("arms", "strategy", "validation"):
        tables[name]["tabular"] = summary_tabular(tables[name]["rows"])

    tables["interaction"] = interaction_table(records, label)
    tables["outcomes"] = outcome_table(records, label)
    tables["best"] = best_table(records, label)
    return tables


def interaction_table(records: List[dict], label: str) -> dict:
    """Strict F1 and evaluated rate per arm x strategy, pooled over validation on/off."""
    cells = grouped(records, lambda r: (r.get("model_label"), r.get("prompt_strategy")))
    present = [(key, name) for key, name in STRATEGIES
               if any(k[1] == key for k in cells)]
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{STRATEGY_HEADERS[key]}}}" for key, _ in present)
    rules = " ".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(present)))
    lines = ["  \\small", "  \\begin{tabular}{@{}l" + "rr" * len(present) + "@{}}",
             "    \\toprule", f"    & {head} \\\\", f"    {rules}",
             "    Arm" + " & F1 & Evaluated" * len(present) + " \\\\", "    \\midrule"]
    rows = []
    for arm in arms():
        values = []
        for key, _ in present:
            group = cells.get((arm, key), [])
            values.append(metric_row(group) if group else None)
        if not any(values):
            continue
        rows.append((arm, values))
        rendered = " & ".join(f"{ac.fmt(v['strict_f1'])} & {ac.fmt_pct(v['evaluated'])}"
                              if v else "-- & --" for v in values)
        lines.append(f"    {arm_label(arm)} & {rendered} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return {
        "rows": [(arm, [None if v is None else v for v in vals]) for arm, vals in rows],
        "caption": f"Strict result F1 and evaluated share per arm and prompting strategy on "
                   f"{label}, each cell pooled over validation-repair on and off.",
        "tabular": "\n".join(lines),
    }


def outcome_table(records: List[dict], label: str) -> dict:
    """Record status per arm, as shares of the arm's records."""
    by_arm = grouped(records, lambda r: r.get("model_label"))
    statuses = [(key, name) for key, name in STATUSES
                if any(r.get("status") == key for r in records)]
    other = sorted({r.get("status") for r in records} - {key for key, _ in statuses})
    statuses += [(key, key.replace("_", "\\_")) for key in other]
    lines = ["  \\small", "  \\begin{tabular}{@{}l" + "r" * len(statuses) + "@{}}",
             "    \\toprule",
             "    Arm & " + " & ".join(name for _, name in statuses) + " \\\\", "    \\midrule"]
    rows = []
    for arm in arms():
        group = by_arm.get(arm)
        if not group:
            continue
        counts = Counter(r.get("status") for r in group)
        shares = {key: counts.get(key, 0) / len(group) for key, _ in statuses}
        rows.append((arm, {"records": len(group), "counts": dict(counts)}))
        lines.append(f"    {arm_label(arm)} & "
                     + " & ".join(ac.fmt_pct(shares[key]) for key, _ in statuses) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return {
        "rows": rows,
        "caption": f"Outcome of every generation per arm on {label}. `Returned a result' counts "
                   f"every record the engine ran (status \\texttt{{success}}), including trees "
                   f"it declined to evaluate.",
        "tabular": "\n".join(lines),
    }


def best_table(records: List[dict], label: str, top: int = 5) -> dict:
    """The `top` single conditions by strict F1."""
    by_config = grouped(records, lambda r: r.get("config_id"))
    scored = sorted(((metric_row(group), config, group[0]) for config, group in by_config.items()),
                    key=lambda item: -item[0]["strict_f1"])[:top]
    strategy_names = dict(STRATEGIES)
    rows = []
    for row, config, first in scored:
        name = (f"{arm_label(first.get('model_label'))}, "
                f"{strategy_names.get(first.get('prompt_strategy'), first.get('prompt_strategy'))}"
                f", validation {'on' if first.get('schema_enforced') else 'off'}")
        rows.append((name, row))
    return {
        "rows": rows,
        "caption": f"The {top} best single matrix conditions by strict result F1 on {label}.",
        "tabular": summary_tabular(rows),
    }


def serialisable(tables: Dict[str, dict]) -> Dict[str, list]:
    """The JSON payload: each table's rows, without the LaTeX."""
    return {name: [[label, row] for label, row in table["rows"] if label is not None]
            for name, table in tables.items()}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default=DEFAULT_RUN, help="Merged matrix run directory.")
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    run = ac.load_run(args.run)
    ac.assert_reproduces_summary(run)
    payload = {}
    written = []
    for population, short in (("discriminating", "disc"), ("all", "all")):
        tables = build(run, population)
        payload[population] = serialisable(tables)
        print(f"\n== {ac.POPULATION_LABELS[population]} ==")
        for label, row in tables["arms"]["rows"] + [(None, None)] + tables["strategy"]["rows"]:
            if label is not None:
                print(f"  {label:<32} F1 {row['strict_f1']:.4f}  evaluated "
                      f"{row['evaluated']:.1%}  skill {row['skill']:+.4f}  n={row['records']}")
        if args.no_write:
            continue
        for name, table in tables.items():
            written.append(ac.write_tex(f"matrix_{name}_{short}", table["tabular"],
                                        table["caption"], f"tab:matrix:{name}:{short}", [run],
                                        table.get("notes", ())))
    if not args.no_write:
        out = ac.write_json("matrix_tables", payload, [run])
        print(f"\nwrote {out} and {len(written)} tables to {written[0].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
