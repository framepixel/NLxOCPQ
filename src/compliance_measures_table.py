#!/usr/bin/env python3
"""The agent compliance measures and the threshold each model gate sets on them, as one table.

WHY
---
Ch. 3 listed the compliance measures as bullets and the gate thresholds in a separate paragraph,
and the two disagreed (the author asked for one table, 2026-11-10). The thresholds are read from
`configs/agent_gate_criteria.json`, the pre-registered gates that `agent_gate_report.py` applies,
so the table cannot drift from what the gates checked. The definitions follow the evaluator:
`eval_testset._agent_compliance` (what every agentic record reports) and the measure list in
`agent_gate_report.py`'s docstring (what the gates compute).

2026-11-15: the thesis no longer reports compliance figures as results (Section 4.3 cut them), so the
table holds what the gates check and nothing else: the six measures no gate reads (cap hits, mean
reasoning tokens, no answer, budget exhausted, context limit, probe restarts) are out, and so are the
two thought-field footnotes (the thought-field rung is cut; evaluation 1's thresholds are the ones its
gate applied with the thought field off).

Usage (from the repo root):
    python src/compliance_measures_table.py            # writes tables/agent_compliance_measures.tex
    python src/compliance_measures_table.py --no-write # prints the rows only
"""
import argparse
import json
from typing import Callable, Dict, List, Optional, Tuple

import analysis_common as ac
import paths

CRITERIA_PATH = paths.CONFIGS / "agent_gate_criteria.json"
GATES = ("eval1", "eval2", "examples")      # column order
NOT_GATED = "--"

# Criterion keys a row reads, so the test can check that every gated criterion has a row.
# `repeat_applies_to` qualifies `repeat_max` (evaluation 1 gates repeats with the thought off, the
# only evaluation-1 rung the thesis reports, so the cell shows the plain threshold).
QUALIFIERS = {"repeat_applies_to"}


def _pct(value: float) -> str:
    return f"{100 * value:.0f}\\%"


def _at_most_pct(c: dict, key: str) -> Optional[str]:
    return f"$\\leq$\\,{_pct(c[key])}" if key in c else None


def _at_least_pct(c: dict, key: str) -> Optional[str]:
    return f"$\\geq$\\,{_pct(c[key])}" if key in c else None


def _repeat(c: dict) -> Optional[str]:
    return _at_most_pct(c, "repeat_max")


def _p90_turn(c: dict) -> Optional[str]:
    limit = c.get("p90_seconds_per_turn_max")
    if limit is None:
        return None
    if isinstance(limit, dict):       # evaluation 1: thought field off / on; the thesis reports off
        return f"$\\leq$\\,{limit['thought_off']}\\,s"
    return f"$\\leq$\\,{limit}\\,s"


def _p90_case(c: dict) -> Optional[str]:
    limit = c.get("p90_case_seconds_max")
    return None if limit is None else f"$\\leq$\\,{limit:,}\\,s"


def _loop_timeouts(c: dict) -> Optional[str]:
    return "none" if "loop_timeout_min_repeats" in c else None


def _agreement(c: dict) -> Optional[str]:
    rate = c.get("probe_cli_agreement")
    if rate is None:
        return None
    return "all cases" if rate == 1.0 else _at_least_pct(c, "probe_cli_agreement")


def _pipeline(c: dict) -> Optional[str]:
    return "none" if c.get("no_pipeline_errors") else None


# (group, measure, what it counts, criterion keys read, cell function)
Row = Tuple[str, str, str, Tuple[str, ...], Callable[[dict], Optional[str]]]
ROWS: List[Row] = [
    ("Per turn", "Malformed actions",
     "turns whose action does not parse or validate / turns",
     ("malformed_max",), lambda c: _at_most_pct(c, "malformed_max")),
    ("Per turn", "Repeated actions",
     "turns that repeat an earlier action of the case exactly (same tool, same arguments) / turns",
     ("repeat_max",), _repeat),
    ("Per turn", "Forced closes",
     "reasoning steps the harness closed at the 2,048-token budget / reasoning steps",
     ("forced_close_max",), lambda c: _at_most_pct(c, "forced_close_max")),
    ("Per turn", "p90 seconds per turn",
     "90th percentile, over cases, of the model's seconds per turn",
     ("p90_seconds_per_turn_max",), _p90_turn),
    ("Per case", "Final source",
     "where the scored tree comes from: submitted, last evaluated (lever E), last valid, none;"
     " the gates count the submitted share",
     ("declared_min",), lambda c: _at_least_pct(c, "declared_min")),
    ("Per case", "Timeouts",
     "cases past the case cap / cases",
     ("timeout_max",), lambda c: _at_most_pct(c, "timeout_max")),
    ("Per case", "Loop-driven timeouts",
     "timed-out cases that sent one action three or more times",
     ("loop_timeout_min_repeats",), _loop_timeouts),
    ("Per case", "p90 seconds per case",
     "90th percentile of the whole case's seconds",
     ("p90_case_seconds_max",), _p90_case),
    ("Per case", "Example-search uptake",
     "cases with at least one \\texttt{search\\_examples} call / cases",
     ("uptake_min",), lambda c: _at_least_pct(c, "uptake_min")),
    ("Harness", "Probe/CLI agreement",
     "cases where the probe and \\texttt{ocpq\\_cli} count the same root results for the final tree"
     " / cases with both counts",
     ("probe_cli_agreement",), _agreement),
    ("Harness", "Pipeline errors",
     "cases that end on a harness fault instead of a model outcome",
     ("no_pipeline_errors",), _pipeline),
]

def load_criteria(path=CRITERIA_PATH) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        gates = json.load(handle)["gates"]
    return {gate["id"]: gate["criteria"] for gate in gates}


def cells(criteria: Dict[str, dict]) -> List[Tuple[str, str, str, List[str]]]:
    """Every row as (group, measure, definition, one cell per gate)."""
    return [(group, name, what, [fn(criteria[g]) or NOT_GATED for g in GATES])
            for group, name, what, _keys, fn in ROWS]


def unconsumed_criteria(criteria: Dict[str, dict]) -> Dict[str, List[str]]:
    """Gate criteria that no row reads -- must be empty, or the table hides a gate."""
    read = {key for *_rest, keys, _fn in ROWS for key in keys} | QUALIFIERS
    return {g: sorted(set(c) - read) for g, c in criteria.items() if set(c) - read}


def tabular(criteria: Dict[str, dict]) -> str:
    lines = [
        "  \\small",
        "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{2.9cm}"
        ">{\\raggedright\\arraybackslash}Xccc@{}}",
        "    \\toprule",
        "    & & \\multicolumn{3}{c@{}}{Gate threshold} \\\\",
        "    \\cmidrule(l){3-5}",
        "    Measure & What it counts & Eval.~1 & Eval.~2 & \\makecell{Example\\\\lookup} \\\\",
    ]
    group = None
    for row_group, name, what, gate_cells in cells(criteria):
        if row_group != group:
            lines.append("    \\midrule")
            lines.append(f"    \\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{row_group}}}}} \\\\")
            group = row_group
        lines.append(f"    {name} & {what} & " + " & ".join(gate_cells) + " \\\\")
    lines += [
        "    \\bottomrule",
        "  \\end{tabularx}",
    ]
    return "\n".join(lines)


CAPTION = ("The agentic compliance measures and each gate's threshold on the 38 development"
           " questions; -- means not checked by that gate.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-write", action="store_true", help="print the rows only")
    args = parser.parse_args(argv)
    criteria = load_criteria()
    missing = unconsumed_criteria(criteria)
    if missing:
        print(f"ERROR: gate criteria with no row: {missing}")
        return 1
    for group, name, _what, gate_cells in cells(criteria):
        print(f"{group:9} {name:24} " + " | ".join(gate_cells))
    if not args.no_write:
        out = ac.write_tex("agent_compliance_measures", tabular(criteria), CAPTION,
                           "tab:eval:compliance",
                           notes=[f"thresholds: {ac.repo_relative(CRITERIA_PATH)}",
                                  "definitions: eval_testset._agent_compliance and the measure list"
                                  " in agent_gate_report.py"])
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
