#!/usr/bin/env python3
"""The conference paper's tables, cut down from the thesis analysis JSON.

WHY
---
The paper (`../Conference Paper/lncs.tex`) has 13 single-column pages and no room for the
thesis tables: those carry thesis-only columns ("F1 executed", whose own docstring says never to
report it alone), rung codes (N1, B0, A1'*) that the paper never defines, and one paired table
per family of contrasts. The paper needs four tables, each answering one research question in
words a reader of the paper can follow.

Nothing here is computed from a run. Every number is READ from the JSON that the thesis tables
are generated from, so a paper table and its thesis twin cannot disagree:

  eval_runs/analysis/matrix_tables.json       src/matrix_tables.py      (arm and strategy levels)
  eval_runs/analysis/paired_stats.json        src/paired_stats.py       (every paired gain and CI)
  eval_runs/analysis/rung_tables.json         src/rung_tables.py        (ablation and agent levels)
  eval_runs/analysis/selfconsistency.json     src/selfconsistency.py    (majority vote)
  eval_runs/analysis/testset_composition.json src/testset_composition.py (the trivial floor)

Regenerate those first if a run changed; this script then only reformats. Each table header
names the JSON it read and the sha256 prefix of that file.

WHAT IT WRITES (to `--out`, default `../Conference Paper/tables`)
-----------------------------------------------------------------
paper_matrix         per model: base and fine-tuned F1, the paired fine-tuning gain with its 95%
                     CI, evaluated share and structural F1 (RQ1)
paper_strategy       F1 per model, variant and prompting strategy, best strategy in bold (RQ1)
paper_interventions  four interventions on fine-tuned Llama 3.1 8B: F1 before and after, paired
                     gain, CI, questions better / worse (RQ3)
paper_agent          the agent's rungs per model in plain words, and the log-access gain (RQ4)

All four are on the 135 discriminating questions, like the thesis's main tables.

Usage (from the repo root):
    python src/paper_tables.py              # write the four tables
    python src/paper_tables.py --no-write   # print the values only
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import analysis_common as ac
import paths

# The matrix run every input must have read (matrix_tables.py's DEFAULT_RUN).
REFERENCE_RUN = "eval_runs/matrix_v2/merged_regold_20260922"

INPUTS = {
    "matrix": "matrix_tables.json",
    "paired": "paired_stats.json",
    "rungs": "rung_tables.json",
    "selfcons": "selfconsistency.json",
    "testset": "testset_composition.json",
}

# Rows of the matrix tables, smallest model first. (label in matrix_tables.json, paper name,
# paired_stats contrast id of the fine-tuning gain, model_label prefix in the interaction table)
MODELS = [
    ("Ministral 3 3B", "Ministral 3 3B", "ft_m3", "ministral3-3b"),
    ("Gemma 3 4B", "Gemma 3 4B", "ft_gemma", "gemma3-4b"),
    ("Llama 3.1 8B", "Llama 3.1 8B", "ft_llama", "llama31-8b"),
    ("Ministral 3 14B", "Ministral 3 14B", "ft_m14", "ministral3-14b"),
]
# Column order of each interaction row in matrix_tables.json (matrix_tables.STRATEGIES).
STRATEGIES = [
    r"\makecell{Allowed\\entities only}",
    r"\makecell{Grammar-\\guided}",
    r"\makecell{Chain-of-\\thought plus}",
]

# The interventions of RQ3: (paper name, reference in words, source of the before/after levels,
# paired contrast). Levels come from rung_tables.json, except the vote's (selfconsistency.json).
# Constrained decoding is N0 -> N1, whose N0 is the matrix's best cell rerun (0.2030), so it and
# the vote share their reference; the thesis's D0 -> D2 replicate gives the same gain (+0.0324).
INTERVENTIONS = [
    ("Grammar-constrained decoding", "free decoding", ("N0", "N1"), "N0_N1"),
    ("Structural check", "grammar-constrained decoding", ("B0", "B1"), "B0_B1"),
    ("Majority vote of 10 samples", "one greedy query", None, None),
    (r"Gold's root arity forced$^{\dagger}$", "grammar-constrained decoding", ("N1", "N2"), "N1_N2"),
]

# The agent rows: (paper name, one-shot rung, loop-without-log rung, loop-with-log rung,
# contrast id of the log-access gain). Rung keys are rung_tables.json's.
AGENTS = [
    ("Ministral 3 14B (evaluation 1)", "A0", "A1p", "A1", "e1_log"),
    ("Qwen3.8 27B, reasoning (evaluation 2)", "Q_A0", "Q_A1p", "Q_A1", "e2_log"),
    ("Qwen3.8 27B + example lookup", None, "Q_A1pS", "Q_A1S", "ex_q_log_given_S"),
    ("Muse Glimmer 30B + example lookup", None, "G_A1pS", "G_A1S", "ex_g_log_given_S"),
]


def load_inputs() -> Tuple[Dict[str, dict], List[str]]:
    """Read every input JSON and describe it for the table headers.

    Returns:
        The parsed documents by key, and one header note per file (path, sha256 prefix, the
        command that generated it).
    """
    documents, notes = {}, []
    for key, name in INPUTS.items():
        path = paths.ANALYSIS_DIR / name
        with open(path, encoding="utf-8") as handle:
            documents[key] = json.load(handle)
        generated = documents[key].get("generated") or {}
        notes.append(f"read {ac.repo_relative(path)} sha256 {ac.sha256_of(path)[:16]}"
                     + (f" (from: {generated['command']})" if generated.get("command") else ""))
    return documents, notes


def check_reference_run(documents: Dict[str, dict]) -> None:
    """Refuse to mix runs: the matrix numbers must all come from the reference run."""
    matrix_runs = [i["run"] for i in documents["matrix"]["generated"]["inputs"]]
    greedy_run = documents["selfcons"].get("greedy_run")
    if matrix_runs != [REFERENCE_RUN] or greedy_run != REFERENCE_RUN:
        sys.exit(f"inputs do not read {REFERENCE_RUN}: matrix_tables {matrix_runs}, "
                 f"selfconsistency greedy {greedy_run}. Regenerate them first.")


def contrasts(documents: Dict[str, dict]) -> Dict[str, dict]:
    """paired_stats.json's contrasts by id; every one the paper uses must be on the 135
    discriminating questions (the post-hoc subsets, e.g. without the angular log, are not)."""
    by_id = {c["id"]: c for c in documents["paired"]["contrasts"]}
    used = ([m[2] for m in MODELS] + [i[3] for i in INTERVENTIONS if i[3]]
            + [a[4] for a in AGENTS])
    for contrast_id in used:
        contrast = by_id[contrast_id]
        if contrast["population"] != "discriminating" or contrast["n"] != 135:
            sys.exit(f"contrast {contrast_id} is not on the 135 discriminating questions")
    return by_id


def ci(values: Sequence[float]) -> str:
    """A 95% interval cell: `lower, upper`, signed."""
    return f"{ac.fmt(values[0], signed=True)}, {ac.fmt(values[1], signed=True)}"


def bold(text: str, on: bool) -> str:
    """Wrap a cell in \\textbf when `on`."""
    return f"\\textbf{{{text}}}" if on else text


def tabularx(spec: str, head: List[str], rows: List[str]) -> str:
    """A booktabs tabularx over the text width, rows already joined with `&`."""
    lines = ["  \\small", f"  \\begin{{tabularx}}{{\\textwidth}}{{@{{}}{spec}@{{}}}}",
             "    \\toprule", *[f"    {line}" for line in head], "    \\midrule"]
    lines += [f"    {row} \\\\" if row != "\\midrule" else "    \\midrule" for row in rows]
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def matrix_table(documents: Dict[str, dict], by_id: Dict[str, dict],
                 floor: float) -> Tuple[str, str]:
    """RQ1: base vs fine-tuned per model, the paired gain in the same row."""
    arms = dict((label, row) for label, row in documents["matrix"]["discriminating"]["arms"])
    rows = []
    for label, name, contrast_id, _ in MODELS:
        base, tuned = arms[f"{label} base"], arms[f"{label} fine-tuned"]
        gain = by_id[contrast_id]
        rows.append(" & ".join([
            name, ac.fmt(base["strict_f1"]), ac.fmt(tuned["strict_f1"]),
            ac.fmt(gain["mean"], signed=True), ci(gain["ci"]),
            ac.fmt_pct(base["evaluated"]), ac.fmt_pct(tuned["evaluated"]),
            ac.fmt(base["structural_f1"], digits=3), ac.fmt(tuned["structural_f1"], digits=3)]))
        print(f"  {name:<18} {base['strict_f1']:.4f} -> {tuned['strict_f1']:.4f}  "
              f"gain {gain['mean']:+.4f} [{gain['ci'][0]:+.4f}, {gain['ci'][1]:+.4f}]")
    head = [r"& \multicolumn{4}{c}{Strict F1} & \multicolumn{2}{c}{Evaluated} "
            r"& \multicolumn{2}{c}{Structural F1} \\",
            r"\cmidrule(lr){2-5} \cmidrule(lr){6-7} \cmidrule(l){8-9}",
            r"Model & Base & Fine-tuned & Gain & 95\% CI "
            r"& Base & Fine-tuned & Base & Fine-tuned \\"]
    caption = (f"The evaluation matrix per model on the 135 discriminating questions (trivial "
               f"floor {floor:.4f}), each pooled over the three prompting strategies and "
               f"validation-repair on and off. The gain is fine-tuned minus base, paired over "
               f"questions, with its 95\\% bootstrap interval. Evaluated: share of queries "
               f"\\texttt{{ocpq\\_cli}} evaluated.")
    return tabularx(">{\\raggedright\\arraybackslash}Xrrrcrrrr", head, rows), caption


def strategy_table(documents: Dict[str, dict]) -> Tuple[str, str]:
    """RQ1: F1 per model, variant and strategy; the best strategy of each arm in bold."""
    interaction = dict(documents["matrix"]["discriminating"]["interaction"])
    rows, base_never_evaluated = [], True
    for _, name, _, key in MODELS:
        cells = []
        for variant in ("base", "finetuned"):
            values = [cell["strict_f1"] for cell in interaction[f"{key}-{variant}"]]
            cells.append([bold(ac.fmt(v), v == max(values)) for v in values])
            if variant == "base":
                base_never_evaluated &= interaction[f"{key}-base"][0]["evaluated"] == 0.0
        # Columns: per strategy, base then fine-tuned.
        rows.append(" & ".join([name, *[c for pair in zip(*cells) for c in pair]]))
    head = [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{s}}}" for s in STRATEGIES) + r" \\",
            r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(l){6-7}",
            "Model" + " & Base & Fine-tuned" * 3 + r" \\"]
    caption = ("Strict F1 per prompting strategy on the 135 discriminating questions, pooled "
               "over validation-repair on and off; the best strategy of each model and variant "
               "is in bold.")
    if base_never_evaluated:
        caption += " Under allowed entities only, no base-model query is evaluated."
    return tabularx(">{\\raggedright\\arraybackslash}Xrrrrrr", head, rows), caption


def interventions_table(documents: Dict[str, dict], by_id: Dict[str, dict],
                        floor: float) -> Tuple[str, str]:
    """RQ3: each intervention against its reference, all on one configuration."""
    rungs = dict(documents["rungs"]["rungs_ablations"])
    vote = (documents["selfcons"]["breakdown"]
            ["llama31-8b-finetuned_schema-allowed-entities-only_schema-on"]["all_cases"])
    greedy = dict(documents["matrix"]["discriminating"]["best"])[
        "Llama 3.1 8B fine-tuned, allowed entities only, validation on"]["strict_f1"]
    rows = []
    for name, reference, levels, contrast_id in INTERVENTIONS:
        if levels is None:
            before, after = greedy, vote["f1_discriminating"]["majority"]
            gain = vote["paired_vs_greedy"]["majority"]
        else:
            before, after = (rungs[k]["strict_f1"] for k in levels)
            gain = by_id[contrast_id]
        rows.append(" & ".join([name, reference, ac.fmt(before), ac.fmt(after),
                                ac.fmt(gain["mean"], signed=True), ci(gain["ci"]),
                                f"{gain['better']} / {gain['worse']}"]))
        print(f"  {name:<40} {before:.4f} -> {after:.4f}  gain {gain['mean']:+.4f}")
    head = [r"Intervention & Reference & \makecell[r]{F1\\before} & \makecell[r]{F1\\after} "
            r"& Gain & 95\% CI & \makecell[r]{Questions\\better / worse} \\"]
    caption = (f"Interventions on the output of fine-tuned Llama 3.1 8B (allowed entities only, "
               f"validation-repair on), on the 135 discriminating questions (trivial floor "
               f"{floor:.4f}). Each gain is paired over questions against the reference named in "
               f"its row, with its 95\\% bootstrap interval. $^{{\\dagger}}$Oracle: uses gold's "
               f"query, so it bounds what a system could reach.")
    # Both text columns wrap: with "grammar-constrained decoding" as the reference (2026-11-14) an
    # `l` reference column leaves the first column narrower than "Grammar-constrained".
    return tabularx(">{\\raggedright\\arraybackslash}X>{\\raggedright\\arraybackslash}Xrrrcr",
                    head, rows), caption


def agent_table(documents: Dict[str, dict], by_id: Dict[str, dict],
                floor: float) -> Tuple[str, str]:
    """RQ4: the agent's rungs in plain words, with the log-access gain of each model."""
    rungs = dict(documents["rungs"]["rungs_agentic"])

    def cell(key: Optional[str]) -> str:
        if key is None:
            return "--"
        row = rungs[key]
        return f"{ac.fmt(row['strict_f1'])} ({ac.fmt_pct(row['evaluated'], digits=0)})"

    rows = []
    for name, one_shot, no_log, with_log, contrast_id in AGENTS:
        gain = by_id[contrast_id]
        rows.append(" & ".join([name, cell(one_shot), cell(no_log), cell(with_log),
                                ac.fmt(gain["mean"], signed=True), ci(gain["ci"])]))
        print(f"  {name:<40} {cell(no_log)} -> {cell(with_log)}  log gain {gain['mean']:+.4f}")
    head = [r"Agent & One-shot & \makecell[r]{Loop,\\no log} & \makecell[r]{Loop,\\log tools} "
            r"& \makecell[r]{Gain from\\log access} & 95\% CI \\"]
    best = max(rungs.values(), key=lambda r: r["strict_f1"])
    caption = (f"Strict F1 of the agents on the 135 discriminating questions (trivial floor "
               f"{floor:.4f}), with the share of evaluated queries in parentheses. Without log "
               f"access the loop can only check its tree; evaluation~2 values are the mean of "
               f"two draws per question. The example-lookup rows add the lookup to both loops. "
               f"The gain from log access is, within each row (same model, same example "
               f"lookup), the loop with log tools minus the loop without, paired over "
               f"questions, with its 95\\% bootstrap interval. The best system reaches "
               f"{best['strict_f1']:.4f}.")
    return tabularx(">{\\raggedright\\arraybackslash}Xrrrrc", head, rows), caption


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default="../Conference Paper/tables",
                        help="Folder the paper \\inputs its tables from.")
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    documents, notes = load_inputs()
    check_reference_run(documents)
    by_id = contrasts(documents)
    floor = documents["testset"]["results"]["mean_trivial_floor_discriminating"]

    builders = {
        "matrix": lambda: matrix_table(documents, by_id, floor),
        "strategy": lambda: strategy_table(documents),
        "interventions": lambda: interventions_table(documents, by_id, floor),
        "agent": lambda: agent_table(documents, by_id, floor),
    }
    tables = {}
    for name, build in builders.items():
        print(f"== {name}")
        tables[name] = build()
    if args.no_write:
        return 0
    out_dir = Path(args.out)
    written = [ac.write_tex(f"paper_{name}", tabular, caption, f"tab:paper:{name}",
                            notes=notes, out_dir=out_dir)
               for name, (tabular, caption) in tables.items()]
    print(f"\nwrote {len(written)} tables to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
