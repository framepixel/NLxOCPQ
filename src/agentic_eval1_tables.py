#!/usr/bin/env python3
"""Thesis tables for evaluation 1 of the agentic arm (Ch. 4, "Evaluation 1").

`rung_tables.py`, `paired_stats.py`, `agent_mechanism_stats.py` and `agent_gate_report.py` cover
the scored levels, the paired differences, the mechanism and the stronger-model gate. This script
writes the three tables they leave out, each from the runs' own records:

  agent_eval1_rungs       the five rungs as they ran: budget, thought, levers, frozen prompt hash
                          (from each run's config_resolved.json), and the contrast each enters.
  agent_eval1_dev         the development-set gate (notes 11 §43.11) and the lever run (§43.13):
                          Llama 3.1 8B and Ministral 3 14B base without levers, Ministral with the
                          five levers FBEXL; thought off and on. Dev-set F1 is shown in its own
                          panel because it was seen while the levers were chosen (FLAGS TH-05).
  agent_eval1_compliance  the pre-registered compliance block of the scored run (§43.15), with the
                          share on the angular log, and each rung's tool calls per question.

DEFINITIONS (each reproduces the notes' figure; the gate script's are reused where they exist)
  malformed, declared, p90 s/turn   `agent_gate_report.measure`.
  clean draft             a `try_query` tree with no blocking structural defect
                          (`ocpq_schema.blocking_defects`): 1/139, 4/159, 30/35, 40/49.
  final with a blocking defect      the record's prediction, same check: 36/37, 34/38, 9/38, 11/38.
  resubmitted unchanged   a `try_query` whose raw action (thought included) equals the case's
                          previous `try_query`: 38/166, 45/115, 9/139, 0/159.
  no declared answer      the case ended on anything but a declared answer (the notes' "budget
                          exhausted": 87%, 74%, 61%, 79%).
  longest question        the case's LLM seconds (419, 447, 189, 138, 95, 79 s).
  probe--CLI              cases where both counted the final tree and agree / such cases; a
                          disagreement the CLI excuses by skipping bindings is counted apart.

Needs the evaluation environment (`ocpq_schema` imports pydantic). Usage, from the repo root:
    python src/agentic_eval1_tables.py
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_common as ac  # noqa: E402
from agent_gate_report import GateTask, measure  # noqa: E402
from ocpq_schema import blocking_defects, structural_defects  # noqa: E402
from paired_stats import load_spec  # noqa: E402

# The development-set runs of notes 11 §43.11 (no levers) and §43.13 (FBEXL), 38 dev cases each.
_GATE = "eval_runs/agentic_gate/shard_"
_PLAIN = "s0_e0_constrained_ordered_bounded_tok4000_turns8_tools12_turntok2000_"
DEV_RUNS = [
    ("Llama 3.1 8B", "off", _GATE + "m0_" + _PLAIN + "20260926_151216"),
    ("Llama 3.1 8B", "on", _GATE + "m0_" + _PLAIN + "thought_20260926_151216"),
    ("Ministral 3 14B", "off", _GATE + "m2_" + _PLAIN + "20260926_151216"),
    ("Ministral 3 14B", "on", _GATE + "m2_" + _PLAIN + "thought_20260926_151216"),
    ("Ministral, FBEXL", "off", _GATE + "m2_" + _PLAIN + "pFBEXL_20260926_201716"),
    ("Ministral, FBEXL", "on", _GATE + "m2_" + _PLAIN + "pFBEXL_thought_20260926_201716"),
]
# The scored rungs, by their condition names in configs/analysis_contrasts.json.
RUNGS = [("A0", "A0"), ("A1p", "A1$'$"), ("A1", "A1"), ("A1t", "A1+thought"), ("A2", "A2")]
LOOP_RUNGS = RUNGS[1:]
SETUP = {
    "A0": "one-shot (B0's prompt)",
    "A1p": "loop, no log",
    "A1": "loop with log tools",
    "A1t": "A1 with a thought",
    "A2": "A1, double budget",
}
CONTRAST = {
    "A0": "the baseline",
    "A1p": "A1$'$ $-$ A0: the loop",
    "A1": "A1 $-$ A1$'$: log access (headline)",
    "A1t": "A1+thought $-$ A1: the thought",
    "A2": "A2 $-$ A1: the budget",
}
TOOLS = ["sample", "attribute_values", "activity_stats", "inspect_situations", "try_query",
         "check_tree"]
ANGULAR = "angular_github_commits_ocel.xml"


def _action(raw: Optional[str]) -> Optional[dict]:
    """A trajectory step's action as a dict (steps store it JSON-encoded, sometimes twice)."""
    try:
        value = json.loads(raw)
        if isinstance(value, str):
            value = json.loads(value)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _blocking(tree) -> bool:
    return bool(blocking_defects(structural_defects(tree)))


def dev_measures(task: GateTask) -> dict:
    """Every row of the development-set table for one run."""
    base = measure(task, loop_min_repeats=2)
    records = task.records
    agents = [r["agent"] for r in records]
    drafts = clean = resubmitted = thought_turns = steps = 0
    for steps_of_case in task.steps():
        previous = None
        for step in steps_of_case:
            steps += 1
            action = _action(step.get("raw"))
            if action is None:
                continue
            thought_turns += bool(action.get("thought"))
            if action.get("tool") != "try_query":
                continue
            drafts += 1
            resubmitted += step.get("raw") == previous
            previous = step.get("raw")
            clean += not _blocking((action.get("args") or {}).get("tree"))
    finals = [r["prediction"] for r in records if isinstance(r.get("prediction"), dict)]
    per_turn = [r["llm"]["duration_sec"] / r["agent"]["turns"]
                for r in records if r["agent"].get("turns") and r.get("llm")]
    strict = [ac.strict_f1(r) for r in records]
    structural = [v for v in (ac.structural_f1(r) for r in records) if v is not None]
    return {
        "cases": len(records),
        "malformed_rate": base["malformed_rate"],
        "no_answer": sum(1 for r in records if r.get("status") == "agent_no_answer"),
        "not_declared": len(records) - base["declared"],
        "tool_calls_per_case": statistics.mean(int(a.get("tool_calls") or 0) for a in agents),
        "drafts": drafts,
        "resubmitted": resubmitted,
        "thought_turns": thought_turns,
        "steps": steps,
        "median_seconds_per_turn": statistics.median(per_turn),
        "p90_seconds_per_turn": base["p90_seconds_per_turn"],
        "longest_llm_seconds": max(r["llm"]["duration_sec"] for r in records if r.get("llm")),
        "clean_drafts": clean,
        "finals": len(finals),
        "finals_blocking": sum(1 for tree in finals if _blocking(tree)),
        "probe_refused": sum(int(a.get("probe_refused_trees") or 0) for a in agents),
        "cases_with_evaluated_draft": sum(
            1 for a in agents if any(t.get("nodes") for t in a.get("turn_summary") or [])),
        "executed": sum(1 for r in records if r.get("status") == "success"),
        "dev_strict_f1": statistics.mean(strict),
        "dev_structural_f1": statistics.mean(structural) if structural else None,
    }


def compliance(task: GateTask) -> dict:
    """The compliance block and tool use of one scored loop rung, over all 169 questions."""
    base = measure(task, loop_min_repeats=2)
    records = task.records
    angular = [r for r in records if ac.entry_key(r)[1] == ANGULAR]
    thought_turns = steps = 0
    for steps_of_case in task.steps():
        for step in steps_of_case:
            steps += 1
            action = _action(step.get("raw"))
            thought_turns += bool(action and action.get("thought"))
    comparisons = [r["agent"]["probe_vs_cli"] for r in records
                   if isinstance(r["agent"].get("probe_vs_cli"), dict)
                   and r["agent"]["probe_vs_cli"].get("agree") is not None]
    actions = Counter(t.get("action") for r in records
                      for t in r["agent"].get("turn_summary") or [])
    return {
        "cases": len(records),
        "turns": base["turns"],
        "malformed": base["malformed"],
        "malformed_rate": base["malformed_rate"],
        "malformed_angular": sum(int(r["agent"].get("malformed_actions") or 0) for r in angular),
        "no_answer": sum(1 for r in records if r.get("status") == "agent_no_answer"),
        "no_answer_angular": sum(1 for r in angular if r.get("status") == "agent_no_answer"),
        "not_declared": len(records) - base["declared"],
        "timeouts": base["timeouts"],
        "pipeline_errors": base["pipeline_errors"],
        "probe_restarts": sum(int(r["agent"].get("probe_restarts") or 0) for r in records),
        "probe_panics": sum(int(r["agent"].get("probe_evaluator_panics") or 0) for r in records),
        "probe_refused": sum(int(r["agent"].get("probe_refused_trees") or 0) for r in records),
        "probe_cli_checked": len(comparisons),
        "probe_cli_agree": sum(1 for c in comparisons if c.get("agree")),
        "probe_cli_excused": sum(1 for c in comparisons
                                 if not c.get("agree") and c.get("cli_bindings_skipped")),
        "thought_turns": thought_turns,
        "steps": steps,
        "tool_calls_per_case": {tool: actions.get(tool, 0) / len(records) for tool in TOOLS},
    }


def _row(label: str, cells: Sequence[str]) -> str:
    return f"    {label} & " + " & ".join(cells) + " \\\\"


def _of(count: int, total: int) -> str:
    return f"{count}/{total}"


def rungs_table(spec: dict, notes: List[str]) -> Path:
    """The five rungs as their run configs recorded them."""
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}llrll>{\\raggedright\\arraybackslash}X@{}}",
             "    \\toprule",
             "    Rung & Setup & \\makecell[r]{Turns /\\\\tool calls} & Thought & Prompt"
             " & Enters the contrast \\\\",
             "    \\midrule"]
    for name, label in RUNGS:
        run = ac.resolve_run(spec["conditions"][name]["run"])
        config = ac._read_json(run / "config_resolved.json")
        agentic = config.get("agentic") or {}
        budget = (f"{agentic['max_turns']} / {agentic['max_tool_calls']}" if agentic else "--")
        thought = ("on" if agentic.get("thought") else "off") if agentic else "--"
        sha = next(iter((agentic.get("prompt_sha256") or {}).values()), None)
        prompt = f"\\texttt{{{sha[:8]}}}" if sha else "B0's"
        lines.append(_row(label, [SETUP[name], budget, thought, prompt, CONTRAST[name]]))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_eval1_rungs", "\n".join(lines),
        "The rungs of evaluation~1 (Ministral 3 14B base; greedy, grammar-constrained, "
        "validation on), as each run recorded them.",
        "tab:agentic:eval1:rungs", notes=notes)


def dev_table(rows: List[dict], notes: List[str]) -> Path:
    """The development-set gate and lever run, six columns."""
    def cells(fn):
        return [fn(m) for m in rows]

    header_models = ("    & \\multicolumn{2}{c}{Llama 3.1 8B}"
                     " & \\multicolumn{2}{c}{Ministral 3 14B}"
                     " & \\multicolumn{2}{c}{Ministral, levers} \\\\")
    lines = ["  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrrrr@{}}",
             "    \\toprule",
             header_models,
             "    \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(l){6-7}",
             "    Thought & off & on & off & on & off & on \\\\",
             "    \\midrule",
             "    \\multicolumn{7}{@{}l}{\\textit{Following the protocol}} \\\\",
             _row("Malformed turns", cells(lambda m: ac.fmt_pct(m["malformed_rate"]))),
             _row("Questions without an answer", cells(lambda m: str(m["no_answer"]))),
             _row("Questions without a declared answer",
                  cells(lambda m: str(m["not_declared"]))),
             _row("Tool calls per question", cells(lambda m: f"{m['tool_calls_per_case']:.1f}")),
             _row("Drafts resubmitted unchanged",
                  cells(lambda m: _of(m["resubmitted"], m["drafts"]))),
             _row("Turns with a thought",
                  cells(lambda m: ac.fmt_pct(m["thought_turns"] / m["steps"])
                        if m["thought"] == "on" else "--")),
             _row("Seconds per turn, median",
                  cells(lambda m: f"{m['median_seconds_per_turn']:.1f}")),
             _row("Seconds per turn, p90", cells(lambda m: f"{m['p90_seconds_per_turn']:.1f}")),
             _row("Longest question (LLM seconds)",
                  cells(lambda m: f"{m['longest_llm_seconds']:.0f}")),
             "    \\midrule",
             "    \\multicolumn{7}{@{}l}{\\textit{Building trees}} \\\\",
             _row("Clean drafts (no blocking defect)",
                  cells(lambda m: _of(m["clean_drafts"], m["drafts"]))),
             _row("Final trees with a blocking defect",
                  cells(lambda m: _of(m["finals_blocking"], m["finals"]))),
             _row("Drafts the probe refused", cells(lambda m: str(m["probe_refused"]))),
             _row("Questions with an evaluated draft",
                  cells(lambda m: str(m["cases_with_evaluated_draft"]))),
             _row("Final query executed", cells(lambda m: str(m["executed"]))),
             "    \\midrule",
             "    \\multicolumn{7}{@{}l}{\\textit{Development-set F1, seen while the levers were "
             "chosen (not a result)}} \\\\",
             _row("Strict F1", cells(lambda m: f"{m['dev_strict_f1']:.3f}")),
             _row("Structural F1", cells(lambda m: ac.fmt(m["dev_structural_f1"], 3))),
             "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_eval1_dev", "\n".join(lines),
        "Model selection and protocol development on the 38 development questions (rung A1).",
        "tab:agentic:eval1:dev", notes=notes)


def compliance_table(rows: List[dict], notes: List[str]) -> Path:
    """The scored run's compliance block and tool use, one column per loop rung."""
    def cells(fn):
        return [fn(m) for m in rows]

    def probe_cli(m):
        if not m["probe_cli_checked"]:
            return "--"
        text = _of(m["probe_cli_agree"], m["probe_cli_checked"])
        return text + (f" (+{m['probe_cli_excused']})" if m["probe_cli_excused"] else "")

    width = 1 + len(rows)
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X"
             + "r" * len(rows) + "@{}}",
             "    \\toprule",
             "    & " + " & ".join(label for _, label in LOOP_RUNGS) + " \\\\",
             "    \\midrule",
             f"    \\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{Compliance, all 169 "
             f"questions}}}} \\\\",
             _row("Malformed turns", cells(lambda m: f"{m['malformed']} "
                                                    f"({ac.fmt_pct(m['malformed_rate'])})")),
             _row("\\quad on the angular log", cells(lambda m: str(m["malformed_angular"]))),
             _row("Questions without an answer", cells(lambda m: str(m["no_answer"]))),
             _row("\\quad on the angular log", cells(lambda m: str(m["no_answer_angular"]))),
             _row("Questions without a declared answer",
                  cells(lambda m: str(m["not_declared"]))),
             _row("Timeouts", cells(lambda m: str(m["timeouts"]))),
             _row("Harness failures (probe unavailable)",
                  cells(lambda m: str(len(m["pipeline_errors"])))),
             _row("Probe restarts / evaluator panics",
                  cells(lambda m: f"{m['probe_restarts']} / {m['probe_panics']}")),
             _row("Drafts the probe refused", cells(lambda m: str(m["probe_refused"]))),
             _row("Probe and CLI agree", cells(probe_cli)),
             _row("Turns with a thought",
                  cells(lambda m: ac.fmt_pct(m["thought_turns"] / m["steps"])
                        if m["thought_turns"] else "--")),
             "    \\midrule",
             f"    \\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{Tool calls per question}}}} \\\\"]
    for tool in TOOLS:
        lines.append(_row(f"\\texttt{{{tool.replace('_', chr(92) + '_')}}}",
                          cells(lambda m, t=tool: f"{m['tool_calls_per_case'][t]:.2f}")))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_eval1_compliance", "\n".join(lines),
        "Compliance of evaluation~1's scored run (Ministral 3 14B base, all 169 questions), "
        "reported before any score.",
        "tab:agentic:eval1:compliance", notes=notes)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    spec = load_spec()
    dev = []
    for model, thought, run in DEV_RUNS:
        m = dev_measures(GateTask(run))
        dev.append({"model": model, "thought": thought, "run": run, **m})
        print(f"{model:<18} thought {thought:<3} malformed {m['malformed_rate']:.3f} no answer "
              f"{m['no_answer']} clean {m['clean_drafts']}/{m['drafts']} finals blocking "
              f"{m['finals_blocking']}/{m['finals']} resubmitted {m['resubmitted']}/{m['drafts']}"
              f" s/turn {m['median_seconds_per_turn']:.1f}/{m['p90_seconds_per_turn']:.1f}"
              f" longest {m['longest_llm_seconds']:.0f} dev F1 {m['dev_strict_f1']:.3f}")
    scored = {}
    for name, label in LOOP_RUNGS:
        m = compliance(GateTask(spec["conditions"][name]["run"]))
        scored[name] = m
        print(f"{label:<14} malformed {m['malformed']} ({m['malformed_rate']:.3f}, angular "
              f"{m['malformed_angular']}) no answer {m['no_answer']} (angular "
              f"{m['no_answer_angular']}) not declared {m['not_declared']} probe-CLI "
              f"{m['probe_cli_agree']}/{m['probe_cli_checked']} +{m['probe_cli_excused']}"
              f" thought {m['thought_turns']}/{m['steps']}")
    if args.no_write:
        return 0
    dev_notes = [f"input: {run}" for _, _, run in DEV_RUNS]
    dev_notes.append("development set only (notes 11 §43.4): no figure here is a result")
    scored_notes = [f"input: {spec['conditions'][name]['run']}" for name, _ in RUNGS]
    written = [rungs_table(spec, scored_notes), dev_table(dev, dev_notes),
               compliance_table([scored[name] for name, _ in LOOP_RUNGS], scored_notes[1:])]
    written.append(ac.write_json("agentic_eval1", {"dev": dev, "scored": scored}))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
