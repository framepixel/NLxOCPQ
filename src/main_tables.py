#!/usr/bin/env python3
"""The thesis's main-text results tables: the conference paper's layout, with the rows the
chapter quotes.

WHY
---
Supervisor feedback (2026-11-06): Chapter 4 had 21 tables, the paired comparisons sat in tables of
their own, and captions ran to six lines. The conference paper had already solved this with one
table per research question, each level next to its paired gain and interval
(`src/paper_tables.py`). This script writes the thesis versions of those tables, adding the rows
the chapter's text quotes and the paper does not need: the structure hint and the constant root
arity, the evaluated share before and after each intervention, the example-lookup gain, and the
pooled strategy and validation-repair levels. The detailed tables move to Appendix F unchanged.

Like `paper_tables.py`, it computes nothing. Every number is read from the analysis JSON the
detailed tables are generated from, so a main table, its appendix twin and the paper's table
cannot disagree. Regenerate those first if a run changed (`Thesis/script/tables/README.md`); the
structure-hint levels need `src/rung_tables.py` from 2026-11-06 or later.

WHAT IT WRITES (to `Thesis/script/tables`)
------------------------------------------
main_matrix         tab:matrix:models       per model: base and fine-tuned F1, the paired
                                            fine-tuning gain and CI, evaluated, structural F1 (RQ1)
main_strategy       tab:matrix:strategy     F1 per model, variant and strategy, with the pooled
                                            strategy and validation-repair levels (RQ1)
main_interventions  tab:abl:interventions   every intervention on fine-tuned Llama 3.1 8B: F1 and
                                            evaluated share before and after, paired gain, CI (RQ3)
main_agent          tab:agentic:results     the agents' rungs, the log-access gain of each, and
                                            the example-lookup gain (RQ4)

Usage (from the repo root):
    python src/main_tables.py              # write the four tables
    python src/main_tables.py --no-write   # print the values only
"""
import argparse
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import analysis_common as ac
import paper_tables as pt

# The interventions are the conference paper's four (paper_tables.INTERVENTIONS: row, reference,
# (rung before, rung after), paired contrast), so thesis and paper report the same set. Cut to
# these on 2026-11-13 at the author's request: the structure hint, the token budget, the constant
# arity and the arity predictor are no longer reported. Rungs are rung_tables.json's; the vote's
# levels come from selfconsistency.json and its reference is the matrix cell itself. The thesis
# keeps the evaluated share before and after (RQ3: runnable is not right) where the paper shows
# questions better / worse.
INTERVENTIONS = pt.INTERVENTIONS
VOTE_CONFIG = "llama31-8b-finetuned_schema-allowed-entities-only_schema-on"
MATRIX_CELL = "Llama 3.1 8B fine-tuned, allowed entities only, validation on"

# The paper's agent rows (pt.AGENTS) with names short enough for one line at the thesis's width.
AGENTS = [
    ("Ministral 3 14B\\\\evaluation~1", "A0", "A1p", "A1", "e1_log"),
    ("Qwen3.8 27B\\\\evaluation~2", "Q_A0", "Q_A1p", "Q_A1", "e2_log"),
    ("Qwen3.8 27B\\\\+ example lookup", None, "Q_A1pS", "Q_A1S", "ex_q_log_given_S"),
    ("Muse Glimmer 30B\\\\+ example lookup", None, "G_A1pS", "G_A1S", "ex_g_log_given_S"),
]
# Example lookup was run with and without log access for Qwen3.8 only (Glimmer passed the
# example-lookup gate alone, so it has no cells without examples): A1'*+S - A1'* and A1*+S - A1*.
EXAMPLE_GAINS = [
    ("Loop, no log", "ex_q_without_log"),
    ("Loop, log tools", "ex_q_with_log"),
]


def narrow(tabular: str) -> str:
    """The thesis's text width is 418pt: these tables need a 4pt column gap (default 6pt) to keep
    every row label on one line."""
    return "  \\setlength{\\tabcolsep}{4pt}\n" + tabular


def check_contrasts(by_id: Dict[str, dict]) -> None:
    """Every contrast this script adds must be on the 135 discriminating questions too."""
    used = ([c for *_, c in INTERVENTIONS if c]
            + [c for _, c in EXAMPLE_GAINS])
    for contrast_id in used:
        contrast = by_id[contrast_id]
        if contrast["population"] != "discriminating" or contrast["n"] != 135:
            sys.exit(f"contrast {contrast_id} is not on the 135 discriminating questions")


def group_row(title: str, width: int) -> str:
    """An italic panel heading across the table."""
    return f"\\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{{title}}}}}"


def matrix_table(documents: Dict[str, dict], by_id: Dict[str, dict]) -> Tuple[str, str]:
    """RQ1: the paper's matrix table. Its nine columns are 47pt wider than the thesis's text
    width, so evaluated share and structural F1 each show base -> fine-tuned in one cell."""
    arms = dict((label, row) for label, row in documents["matrix"]["discriminating"]["arms"])
    rows = []
    for label, name, contrast_id, _ in pt.MODELS:
        base, tuned = arms[f"{label} base"], arms[f"{label} fine-tuned"]
        gain = by_id[contrast_id]
        rows.append(" & ".join([
            name, ac.fmt(base["strict_f1"]), ac.fmt(tuned["strict_f1"]),
            ac.fmt(gain["mean"], signed=True), pt.ci(gain["ci"]),
            f"{ac.fmt_pct(base['evaluated'])} $\\to$ {ac.fmt_pct(tuned['evaluated'])}",
            f"{ac.fmt(base['structural_f1'], digits=3)} $\\to$ "
            f"{ac.fmt(tuned['structural_f1'], digits=3)}"]))
        print(f"  {name:<18} {base['strict_f1']:.4f} -> {tuned['strict_f1']:.4f}  "
              f"gain {gain['mean']:+.4f} [{gain['ci'][0]:+.4f}, {gain['ci'][1]:+.4f}]")
    head = [r"& \multicolumn{4}{c}{Strict F1} & Evaluated & \makecell[r]{Structural\\F1} \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-6} \cmidrule(l){7-7}",
            r"Model & Base & Tuned & Gain & 95\% CI & base $\to$ tuned & base $\to$ tuned \\"]
    caption = ("The evaluation matrix per model on the 135 discriminating questions, averaged over "
               "strategies and validation-repair; gain: fine-tuned minus base.")
    return narrow(pt.tabularx(">{\\raggedright\\arraybackslash}Xrrrcrr", head, rows)), caption


def strategy_table(documents: Dict[str, dict]) -> Tuple[str, str]:
    """RQ1: F1 per model, variant and strategy, then the pooled strategy and validation levels."""
    interaction = dict(documents["matrix"]["discriminating"]["interaction"])
    strategy = dict(documents["matrix"]["discriminating"]["strategy"])
    validation = dict(documents["matrix"]["discriminating"]["validation"])
    rows = []
    for _, name, _, key in pt.MODELS:
        cells = []
        for variant in ("base", "finetuned"):
            values = [cell["strict_f1"] for cell in interaction[f"{key}-{variant}"]]
            cells.append([pt.bold(ac.fmt(v), v == max(values)) for v in values])
        rows.append(" & ".join([name, *[c for pair in zip(*cells) for c in pair]]))
    # Pooled levels: matrix_tables.json's strategy rows are in pt.STRATEGIES' column order.
    pooled = [strategy[label]["strict_f1"] for label in
              ("allowed entities only", "grammar-guided", "chain-of-thought plus")]
    on, off = (validation[f"validation-repair {flag}"] for flag in ("on", "off"))
    rows += ["\\midrule", group_row("Averaged over the four models", 7),
             "Both variants & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{ac.fmt(v)}}}"
                                             for v in pooled),
             "Validation-repair on / off & \\multicolumn{6}{c}{"
             f"{ac.fmt(on['strict_f1'])} / {ac.fmt(off['strict_f1'])} (evaluated "
             f"{ac.fmt_pct(on['evaluated'])} / {ac.fmt_pct(off['evaluated'])})}}"]
    for label, value in zip(("allowed entities only", "grammar-guided", "chain-of-thought plus"),
                            pooled):
        print(f"  pooled {label:<24} {value:.4f}")
    print(f"  validation on / off {on['strict_f1']:.4f} / {off['strict_f1']:.4f}")
    head = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{s}}}" for s in pt.STRATEGIES) + r" \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(l){6-7}",
            "Model" + " & Base & Fine-tuned" * 3 + r" \\"]
    caption = ("Strict F1 per prompting strategy on the 135 discriminating questions, averaged over "
               "validation-repair; the best strategy of each model and variant is in bold.")
    return pt.tabularx(">{\\raggedright\\arraybackslash}Xrrrrrr", head, rows), caption


def interventions_table(documents: Dict[str, dict], by_id: Dict[str, dict]) -> Tuple[str, str]:
    """RQ3: each intervention against the condition it changes, F1 and evaluated share."""
    rungs = dict(documents["rungs"]["rungs_ablations"])
    vote = documents["selfcons"]["breakdown"][VOTE_CONFIG]["all_cases"]
    cell = dict(documents["matrix"]["discriminating"]["best"])[MATRIX_CELL]
    rows = []
    for name, reference, levels, contrast_id in INTERVENTIONS:
        if levels is None:
            before, after = cell, {"strict_f1": vote["f1_discriminating"]["majority"],
                                   "evaluated": None}
            gain = vote["paired_vs_greedy"]["majority"]
        else:
            before, after = (rungs[k] for k in levels)
            gain = by_id[contrast_id]
        rows.append(" & ".join([
            name, reference, ac.fmt(before["strict_f1"]), ac.fmt(after["strict_f1"]),
            ac.fmt(gain["mean"], signed=True), pt.ci(gain["ci"]),
            ac.fmt_pct(before["evaluated"]), ac.fmt_pct(after["evaluated"])]))
        print(f"  {name:<40} {before['strict_f1']:.4f} -> {after['strict_f1']:.4f}  "
              f"gain {gain['mean']:+.4f} [{gain['ci'][0]:+.4f}, {gain['ci'][1]:+.4f}]")
    head = [r"& & \multicolumn{4}{c}{Strict F1} & \multicolumn{2}{c}{Evaluated} \\",
            r"\cmidrule(lr){3-6} \cmidrule(l){7-8}",
            r"Intervention & Reference & Before & After & Gain & 95\% CI & Before & After \\"]
    caption = ("Interventions on the output of fine-tuned Llama 3.1 8B (allowed entities only, "
               "validation-repair on) over the 135 discriminating questions; "
               "$^{\\dagger}$oracle.")
    # Two wrapping text columns: with the reference as an `l` column the first one is narrower than
    # the word "constrained" at the thesis's width (overfull by 1.4pt).
    return narrow(pt.tabularx(">{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}Xrrrcrr",
                              head, rows)), caption


def agent_table(documents: Dict[str, dict], by_id: Dict[str, dict]) -> Tuple[str, str]:
    """RQ4: the paper's agent table turned on its side -- one column per agent, one row per
    rung, the gains below with their intervals -- because with the agents as rows their names
    wrap onto three lines at the thesis's width. Then the example-lookup gain of Qwen3.8."""
    rungs = dict(documents["rungs"]["rungs_agentic"])

    def level(key: Optional[str]) -> str:
        if key is None:
            return "--"
        row = rungs[key]
        return f"{ac.fmt(row['strict_f1'])} ({ac.fmt_pct(row['evaluated'], digits=0)})"

    def gain_rows(label: str, gains: List[Optional[dict]]) -> List[str]:
        return [" & ".join([label, *("--" if g is None else ac.fmt(g["mean"], signed=True)
                                     for g in gains)]),
                " & ".join(["\\quad 95\\% CI", *("--" if g is None else pt.ci(g["ci"])
                                                 for g in gains)])]

    # Long labels go into the italic panel headings: the row labels share the line with four
    # wide columns.
    rows = [group_row("F1 (evaluated share)", 5)]
    for index, rung in enumerate(("One-shot", "Loop, no log", "Loop, log tools")):
        rows.append(" & ".join([rung, *(level(agent[1 + index]) for agent in AGENTS)]))
    rows += ["\\midrule", group_row("Gain from log access", 5),
             *gain_rows("Log tools $-$ no log", [by_id[a[4]] for a in AGENTS])]
    # Qwen3.8 is the only agent run with and without examples; the gain sits in its +S column.
    lookup = dict(EXAMPLE_GAINS)
    rows += ["\\midrule", group_row("Gain from example lookup", 5)]
    for label, contrast_id in lookup.items():
        rows += gain_rows(label, [None, None, by_id[contrast_id], None])
    for agent in AGENTS:
        print(f"  {agent[0]:<28} {level(agent[2])} -> {level(agent[3])}  "
              f"log gain {by_id[agent[4]]['mean']:+.4f}")
    for label, contrast_id in lookup.items():
        print(f"  example gain, {label:<20} {by_id[contrast_id]['mean']:+.4f}")
    head = [" & " + " & ".join(f"\\makecell[r]{{{name}}}" for name, *_ in AGENTS) + r" \\"]
    caption = ("The agents on the 135 discriminating questions; evaluation~2 and example-lookup "
               "values are the mean of two draws per question.")
    return narrow(pt.tabularx(">{\\raggedright\\arraybackslash}Xrrrr", head, rows)), caption


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    documents, notes = pt.load_inputs()
    pt.check_reference_run(documents)
    by_id = pt.contrasts(documents)
    check_contrasts(by_id)
    if "H0" not in dict(documents["rungs"]["rungs_ablations"]):
        sys.exit("rung_tables.json has no structure-hint rungs: run src/rung_tables.py first")

    builders = {
        "matrix": ("tab:matrix:models", lambda: matrix_table(documents, by_id)),
        "strategy": ("tab:matrix:strategy", lambda: strategy_table(documents)),
        "interventions": ("tab:abl:interventions", lambda: interventions_table(documents, by_id)),
        "agent": ("tab:agentic:results", lambda: agent_table(documents, by_id)),
    }
    tables: Dict[str, Tuple[str, str, str]] = {}
    for name, (label, build) in builders.items():
        print(f"== {name}")
        tables[name] = (label, *build())
    if args.no_write:
        return 0
    written: List = [ac.write_tex(f"main_{name}", tabular, caption, label, notes=notes)
                     for name, (label, tabular, caption) in tables.items()]
    print(f"\nwrote {len(written)} tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
