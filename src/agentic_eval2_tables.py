#!/usr/bin/env python3
"""Thesis tables for evaluation 2 and example lookup (Ch. 4, "Evaluation 2", "Example lookup").

`rung_tables.py`, `paired_stats.py`, `agent_mechanism_stats.py` and `agent_gate_report.py` cover
the levels, the paired differences, the log-use mechanism and the gates. This script writes the
three tables they leave out, each from the runs' own records:

  agent_design               evaluation 1, evaluation 2 and example lookup side by side, every row
                             read from the scored runs' config_resolved.json (notes 11 §44.2, §45.3).
  agent_eval2_compliance     the pre-registered compliance block of the scored runs (§44.4: first,
                             before any score), with tool calls and cost per question-draw, for
                             Qwen's four loop cells and Glimmer's two (§44.9, §45.9).
  agent_examples_mechanism   what the examples changed in the final trees (§45.9 "Mechanism"):
                             SAT, box count, one-box share, structure borrowed from a shown
                             example, and the F1 of question-draws that searched against those
                             that did not.

DEFINITIONS (each reproduces the notes' figure)
  repeats, forced closes, declared, probe--CLI   `agent_gate_report.measure`.
  reasoning tokens per step   mean `reasoning_tokens` over the turn summaries that record one
                              (546 / 783, §44.9).
  tool calls per question-draw   turn summaries by action / records (eval 1's convention).
  cost                        `llm.calls`, `llm.prompt_tokens`, `llm.duration_sec` per record.
  SAT                         a final tree with at least one constraint of type `SAT`.
  boxes, one-box              boxes in the final tree; trees with exactly one box.
  shown an example            the record's trajectory lists at least one example index.
  structure borrowed          the final's coarse structure equals a shown example's. The coarse
                              structure is, per box in node order: the number of children, of
                              event variables and of object variables, and the multisets of
                              filter, size-filter and constraint types (no names, no values).
                              Reproduces §45.9's 13--18% and 0.9%. For a cell without S, the
                              examples are those its +S cell was shown for the same question and
                              draw: the rate a tree reaches by chance.
  searched                    `agent.example_search_calls` > 0.

The mechanism table covers the 135 discriminating questions x 2 draws; the compliance table all
169 x 2, as pre-registered.

Usage, from the repo root:
    python src/agentic_eval2_tables.py
    python src/agentic_eval2_tables.py --no-write   # print and reconcile only
"""
import argparse
import ast
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_common as ac  # noqa: E402
import paths  # noqa: E402
from agent_gate_report import GateTask, _key_of, measure  # noqa: E402
from paired_stats import load_spec  # noqa: E402

# The loop cells, by their condition names in configs/analysis_contrasts.json.
CELLS = [("Q_A1p", "A1$'^*$"), ("Q_A1", "A1$^*$"), ("Q_A1pS", "A1$'^*$+S"), ("Q_A1S", "A1$^*$+S"),
         ("G_A1pS", "A1$'^*$+S"), ("G_A1S", "A1$^*$+S")]
# Which +S cell's shown examples a cell without S is compared with (the chance rate).
CONTROL_OF = {"Q_A1p": "Q_A1pS", "Q_A1": "Q_A1S"}
TOOLS = ["sample", "attribute_values", "activity_stats", "inspect_situations", "try_query",
         "check_tree", "search_examples"]
MODEL_NAMES = {"ministral3-14b-base": "Ministral 3 14B base", "qwen3.8-27b-thinking": "Qwen3.8 27B",
               "muse-glimmer-30b-reasoning": "Muse Glimmer 30B"}
SHORT = {"Ministral 3 14B base": "Ministral", "Qwen3.8 27B": "Qwen", "Muse Glimmer 30B": "Glimmer"}
# Columns of the design table: the representative runs (every shard of a merged run is read and
# must agree on each row) and the rungs and headline of each evaluation.
DESIGN = [
    ("Evaluation~1", ["A1", "A1p"], "A0, A1$'$, A1, A1+thought, A2", "A1 $-$ A1$'$"),
    ("Evaluation~2", ["Q_A1", "Q_A1p"], "A0$^*$, A1$'^*$, A1$^*$", "A1$^*$ $-$ A1$'^*$"),
    ("Example lookup", ["Q_A1S", "Q_A1pS", "G_A1S", "G_A1pS"],
     "A1$'^*$+S, A1$^*$+S (A1$^*$, A1$'^*$ reused)",
     "2$\\times$2: log access $\\times$ examples"),
]

# The notes' figures (§44.9, §45.9); a difference is printed, never adopted.
EXPECTED = {
    "Q_A1": {"repeat_rate": 0.074, "forced_close_rate": 0.078, "reasoning_tokens": 546,
             "probe_cli": (298, 299)},
    # §44.9 prints 19.8%; the records give 217 / 1,099 = 19.745% (FLAGS IC-29).
    "Q_A1p": {"repeat_rate": 0.197, "forced_close_rate": 0.165, "reasoning_tokens": 783},
    "Q_A1S": {"probe_cli": (314, 315)},
    "G_A1S": {"probe_cli": (330, 330)},
}
EXPECTED_MECHANISM = {
    "gold": {"sat": 0.704, "boxes": 2.53, "one_box": 0.074},
    "Q_A1S": {"searched_f1": 0.199, "not_searched_f1": 0.336},
    "Q_A1": {"borrowed": 0.009},
}


def _tree(value) -> Optional[dict]:
    """A query tree from a record or the example pool (JSON, or a Python literal in the pool)."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
    return value if isinstance(value, dict) and isinstance(value.get("nodes"), list) else None


def _boxes(tree: dict) -> List[Tuple[dict, list]]:
    return [tuple(n["Box"]) for n in tree["nodes"] if isinstance(n, dict) and "Box" in n]


def coarse_structure(tree: dict) -> tuple:
    """Per box in node order: children, variable counts, filter / size-filter / constraint kinds."""
    out = []
    for box, children in _boxes(tree):
        def kinds(key):
            return tuple(sorted(str(item.get("type")) for item in box.get(key) or []))
        out.append((len(children or []), len(box.get("newEventVars") or {}),
                    len(box.get("newObjectVars") or {}), kinds("filters"), kinds("sizeFilters"),
                    kinds("constraints")))
    return tuple(out)


def has_sat(tree: dict) -> bool:
    return any(c.get("type") == "SAT" for box, _ in _boxes(tree)
               for c in box.get("constraints") or [])


def shard_configs(run_dir: str) -> List[dict]:
    """The config_resolved.json of a run, or of every shard a merged run was built from."""
    run = ac.resolve_run(run_dir)
    if (run / "config_resolved.json").exists():
        return [ac._read_json(run / "config_resolved.json")]
    stem = run.name.replace("merged_", "shard_", 1)
    shards = sorted(run.parent.glob(f"{stem}_e0_c*of*"))
    if not shards:
        raise SystemExit(f"no shards for {run}")
    return [ac._read_json(s / "config_resolved.json") for s in shards]


def _model_entry(config: dict) -> dict:
    return config["config"]["models"][config["config"]["_model_idx"]]


def _decoding(config: dict) -> Optional[Tuple[float, float, int]]:
    """(temperature, top-p, top-k) of the run's model, None for greedy decoding."""
    if not config.get("temperature"):
        return None
    model = _model_entry(config)
    sampling = model.get("sampling") or {}
    return (model.get("temperature", config["temperature"]), sampling.get("top_p"),
            sampling.get("top_k"))


def _decoding_text(decoding: List[Tuple[str, Optional[tuple]]]) -> str:
    """One cell: greedy, or the shared T and top-p with each model's top-k."""
    if all(d is None for _, d in decoding):
        return "greedy"
    shared = {d[:2] for _, d in decoding}
    if len(shared) != 1:
        raise SystemExit(f"models differ in temperature or top-p: {decoding}")
    temperature, top_p = next(iter(shared))
    top_k = " / ".join(f"{d[2]}" + (f" ({SHORT[label]})" if len(decoding) > 1 else "")
                       for label, d in decoding)
    return f"sampled: $T$~{temperature:.1f}, top-$p$ {top_p}, top-$k$ {top_k}"


def design_values(conditions: Sequence[str], spec: dict) -> Dict[str, str]:
    """Every design row for one evaluation; all shards of all its runs must agree on it."""
    rows: Dict[str, set] = {}
    models, decoding = [], []
    for name in conditions:
        for config in shard_configs(spec["conditions"][name]["run"]):
            agentic = config.get("agentic") or {}
            budget = config.get("reasoning_token_budget")
            strategy = next(iter(agentic.get("prompt_sha256") or {}), None)
            values = {
                "reasoning": (f"$\\le$ {budget:,} tokens, then the constrained action"
                              if budget else "none"),
                "history": ("kept" if config.get("reasoning_history") == "keep" else "--")
                if budget else "--",
                "seeds": ("one per call" if config.get("generation_seed_per_call") else "--"),
                "draws": str(config.get("repeats")),
                "cap": f"{config.get('case_timeout_sec'):,}~s",
                "budget": f"{agentic.get('max_turns')} / {agentic.get('max_tool_calls')}",
                "turn_tokens": f"{agentic.get('turn_token_budget'):,}",
                "levers": agentic.get("levers") or "--",
                "examples": str(agentic.get("examples_per_search") or "--"),
                "grammar": ("ordered, bounded" if config.get("constrained_decoding")
                            and not config.get("constrained_any_order")
                            and config.get("constrained_bounded_shape") else "other"),
                "testset": f"\\texttt{{{config['testset_hash'][:8]}}}",
                "ocpq": f"\\texttt{{{config['ocpq']['rev'][:8]}}}",
                "fingerprint": f"\\texttt{{{config['pipeline_fingerprint'][:8]}}}",
                f"prompt_{strategy}": f"\\texttt{{{agentic['prompt_sha256'][strategy][:8]}}}",
            }
            for key, value in values.items():
                rows.setdefault(key, set()).add(value)
            label = MODEL_NAMES.get(_model_entry(config)["label"], _model_entry(config)["label"])
            if label not in models:
                models.append(label)
                decoding.append((label, _decoding(config)))
    for key, values in rows.items():
        if len(values) != 1:
            raise SystemExit(f"{conditions}: shards disagree on {key}: {sorted(values)}")
    out = {key: next(iter(values)) for key, values in rows.items()}
    out["models"] = ", ".join(models)
    out["decoding"] = _decoding_text(decoding)
    return out


def compliance(task: GateTask) -> dict:
    """Compliance, tool use and cost of one scored loop cell, over all 169 x 2 records."""
    base = measure(task, loop_min_repeats=2)
    records = task.records
    agents = [r["agent"] for r in records]
    summaries = [t for a in agents for t in a.get("turn_summary") or []]
    reasoning = [int(t["reasoning_tokens"]) for t in summaries
                 if t.get("reasoning_tokens") is not None]
    actions = Counter(t.get("action") for t in summaries)
    llm = [r.get("llm") or {} for r in records]
    n = len(records)
    return {
        **{key: base[key] for key in ("cases", "turns", "malformed", "malformed_rate", "repeats",
                                      "repeat_rate", "forced_closes", "forced_close_rate",
                                      "timeouts", "declared", "probe_cli_checked",
                                      "probe_cli_agree", "probe_cli_excused", "pipeline_errors",
                                      "status_counts")},
        "reasoning_tokens": statistics.mean(reasoning) if reasoning else None,
        "turns_per_record": base["turns"] / n,
        "no_answer": sum(1 for r in records if r.get("status") == "agent_no_answer"),
        "context_limit": sum(1 for a in agents if a.get("context_limit_hit")),
        "searched": sum(1 for a in agents if int(a.get("example_search_calls") or 0) > 0),
        "tool_calls_per_record": {tool: actions.get(tool, 0) / n for tool in TOOLS},
        "llm_calls": statistics.mean(int(x.get("calls") or 0) for x in llm),
        "prompt_tokens": statistics.mean(int(x.get("prompt_tokens") or 0) for x in llm),
        "llm_seconds": statistics.mean(float(x.get("duration_sec") or 0) for x in llm),
    }


def shown_examples(task: GateTask) -> Dict[str, List[int]]:
    """Per trajectory key, the pool indices of every example shown to the record."""
    shown = {}
    for key, agent in task.trajectories.items():
        indices = []
        for step in agent.get("trajectory") or []:
            examples = step.get("examples")
            if isinstance(examples, str):
                examples = json.loads(examples)
            indices += [int(index) for index, _ in examples or []]
        shown[key] = indices
    return shown


def mechanism(task: GateTask, shown: Dict[Tuple, List[int]], pool: list) -> dict:
    """The final trees of one cell's discriminating records, and what they share with examples.

    Args:
        task: The cell.
        shown: (entry key, draw) -> pool indices shown (the cell's own, or its +S cell's).
        pool: data/ocpq_dataset.json.
    """
    records = [r for r in task.records if ac.in_population(r, "discriminating")]
    finals = [t for t in (_tree(r.get("prediction")) for r in records) if t]
    borrowed = compared = 0
    pool_structure: Dict[int, tuple] = {}
    for record in records:
        indices = shown.get((ac.entry_key(record), record.get("repeat"))) or []
        if not indices:
            continue
        compared += 1
        final = _tree(record.get("prediction"))
        if final is None:
            continue
        for index in indices:
            if index not in pool_structure:
                pool_structure[index] = coarse_structure(_tree(pool[index]["query"]))
        borrowed += coarse_structure(final) in {pool_structure[i] for i in indices}
    searched = [r for r in records if int(r["agent"].get("example_search_calls") or 0) > 0]
    others = [r for r in records if int(r["agent"].get("example_search_calls") or 0) == 0]
    return {
        "records": len(records),
        "finals": len(finals),
        "sat": sum(has_sat(t) for t in finals),
        "boxes": statistics.mean(len(_boxes(t)) for t in finals),
        "one_box": sum(len(_boxes(t)) == 1 for t in finals),
        "compared": compared,
        "borrowed": borrowed,
        "searched": len(searched),
        "searched_f1": statistics.mean(ac.strict_f1(r) for r in searched) if searched else None,
        "not_searched_f1": (statistics.mean(ac.strict_f1(r) for r in others)
                            if searched and others else None),
    }


def gold_mechanism(task: GateTask) -> dict:
    trees = {}
    for record in task.records:
        if ac.in_population(record, "discriminating"):
            trees[ac.entry_key(record)] = _tree(record["gold_query"])
    trees = [t for t in trees.values() if t]
    return {"questions": len(trees), "sat": sum(has_sat(t) for t in trees),
            "boxes": statistics.mean(len(_boxes(t)) for t in trees),
            "one_box": sum(len(_boxes(t)) == 1 for t in trees)}


def _row(label: str, cells: Sequence[str]) -> str:
    return f"    {label} & " + " & ".join(cells) + " \\\\"


def _pct(count: int, total: int) -> str:
    return f"{count} ({ac.fmt_pct(count / total)})" if total else "--"


def _header(first: str = "") -> List[str]:
    return [f"    {first} & \\multicolumn{{4}}{{c}}{{Qwen3.8 27B}}"
            " & \\multicolumn{2}{c}{Muse Glimmer 30B} \\\\",
            "    \\cmidrule(lr){2-5} \\cmidrule(l){6-7}",
            "    & " + " & ".join(label for _, label in CELLS) + " \\\\"]


def design_table(columns: List[Tuple[str, Dict[str, str], str, str]], notes: List[str]) -> Path:
    """Evaluation 1, evaluation 2 and example lookup, one column each."""
    def row(label, key):
        return _row(label, [values.get(key, "--") for _, values, _, _ in columns])

    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}p{3.3cm}"
             + ">{\\raggedright\\arraybackslash}X" * len(columns) + "@{}}",
             "    \\toprule",
             "    & " + " & ".join(title for title, _, _, _ in columns) + " \\\\",
             "    \\midrule",
             row("Models", "models"),
             row("Before each action", "reasoning"),
             row("Earlier reasoning", "history"),
             row("Decoding", "decoding"),
             row("Seeds", "seeds"),
             row("Draws per question", "draws"),
             row("Cap per question", "cap"),
             "    \\midrule",
             row("Turns / tool calls", "budget"),
             row("Action tokens per turn", "turn_tokens"),
             row("Grammar", "grammar"),
             row("Levers", "levers"),
             row("Examples per search", "examples"),
             row("Prompt, log tools", "prompt_agentic-log-tools"),
             row("Prompt, no log", "prompt_agentic-self-refine"),
             row("Testset", "testset"),
             row("OCPQ revision", "ocpq"),
             row("Pipeline fingerprint", "fingerprint"),
             "    \\midrule",
             _row("Rungs", [rungs for _, _, rungs, _ in columns]),
             _row("Headline", [headline for _, _, _, headline in columns]),
             "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_design", "\n".join(lines),
        "The three agentic evaluations as the resolved configuration of each scored run "
        "recorded them.",
        "tab:agentic:design", notes=notes)


def compliance_table(rows: List[dict], notes: List[str]) -> Path:
    """The scored cells' compliance, tool use and cost, one column per loop cell."""
    def cells(fn):
        return [fn(m) for m in rows]

    def probe_cli(m):
        if not m["probe_cli_checked"]:
            return "--"
        text = f"{m['probe_cli_agree']}/{m['probe_cli_checked']}"
        return text + (f" (+{m['probe_cli_excused']})" if m["probe_cli_excused"] else "")

    lines = ["  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrrrr@{}}",
             "    \\toprule", *_header(), "    \\midrule",
             "    \\multicolumn{7}{@{}l}{\\textit{Compliance, all 169 questions $\\times$ 2 draws}}"
             " \\\\",
             _row("Malformed actions", cells(lambda m: str(m["malformed"]))),
             _row("Repeated actions (of turns)", cells(lambda m: ac.fmt_pct(m["repeat_rate"]))),
             _row("Forced closes (of reasoning steps)",
                  cells(lambda m: ac.fmt_pct(m["forced_close_rate"]))),
             _row("Reasoning tokens per step", cells(lambda m: f"{m['reasoning_tokens']:.0f}")),
             _row("Turns per question-draw", cells(lambda m: f"{m['turns_per_record']:.2f}")),
             _row("Without a declared answer",
                  cells(lambda m: str(m["cases"] - m["declared"]))),
             _row("Timeouts / no answer / context limit",
                  cells(lambda m: f"{m['timeouts']}/{m['no_answer']}/{m['context_limit']}")),
             _row("Probe and CLI agree", cells(probe_cli)),
             _row("Question-draws that searched",
                  cells(lambda m: _pct(m["searched"], m["cases"]) if m["searched"] else "--")),
             "    \\midrule",
             "    \\multicolumn{7}{@{}l}{\\textit{Tool calls per question-draw}} \\\\"]
    for tool in TOOLS:
        lines.append(_row(f"\\texttt{{{tool.replace('_', chr(92) + '_')}}}",
                          cells(lambda m, t=tool: f"{m['tool_calls_per_record'][t]:.2f}")))
    lines += ["    \\midrule",
              "    \\multicolumn{7}{@{}l}{\\textit{Cost per question-draw}} \\\\",
              _row("LLM calls", cells(lambda m: f"{m['llm_calls']:.1f}")),
              _row("Prompt tokens (thousands)",
                   cells(lambda m: f"{m['prompt_tokens'] / 1000:.1f}")),
              _row("LLM seconds", cells(lambda m: f"{m['llm_seconds']:.0f}")),
              "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_eval2_compliance", "\n".join(lines),
        "Compliance of the scored runs of evaluation~2 and example lookup (all 169 questions, "
        "two draws each), reported before any score.",
        "tab:agentic:eval2:compliance", notes=notes)


def mechanism_table(gold: dict, rows: List[dict], notes: List[str]) -> Path:
    """What the final trees look like with and without examples, against the gold trees."""
    def cells(fn, gold_cell="--"):
        return [gold_cell] + [fn(m) for m in rows]

    lines = ["  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrrrrr@{}}",
             "    \\toprule",
             "    & & \\multicolumn{4}{c}{Qwen3.8 27B} & \\multicolumn{2}{c}{Muse Glimmer 30B} \\\\",
             "    \\cmidrule(lr){3-6} \\cmidrule(l){7-8}",
             "    & Gold & " + " & ".join(label for _, label in CELLS) + " \\\\",
             "    \\midrule",
             "    \\multicolumn{8}{@{}l}{\\textit{Final trees}} \\\\",
             _row("With a \\texttt{SAT} constraint",
                  cells(lambda m: ac.fmt_pct(m["sat"] / m["finals"]),
                        ac.fmt_pct(gold["sat"] / gold["questions"]))),
             _row("Nodes, mean", cells(lambda m: f"{m['boxes']:.2f}", f"{gold['boxes']:.2f}")),
             _row("One node", cells(lambda m: ac.fmt_pct(m["one_box"] / m["finals"]),
                                   ac.fmt_pct(gold["one_box"] / gold["questions"]))),
             "    \\midrule",
             "    \\multicolumn{8}{@{}l}{\\textit{Examples}} \\\\",
             _row("Question-draws that searched",
                  cells(lambda m: str(m["searched"]) if m["searched"] else "--")),
             _row("\\quad their F1", cells(lambda m: ac.fmt(m["searched_f1"], 3))),
             _row("\\quad F1 of the others", cells(lambda m: ac.fmt(m["not_searched_f1"], 3))),
             _row("Question-draws shown examples", cells(lambda m: str(m["compared"]))),
             _row("\\quad final structure as one of them",
                  cells(lambda m: ac.fmt_pct(m["borrowed"] / m["compared"]))),
             "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "agent_examples_mechanism", "\n".join(lines),
        "What example lookup changed in the final trees over the 135 discriminating questions "
        "(two draws each; gold: one tree per question).",
        "tab:agentic:examples:mechanism", notes=notes)


def _reconcile(name: str, got: dict, want: dict) -> List[str]:
    """Each figure against the notes', rounded to the decimals the notes print."""
    problems = []
    for key, value in want.items():
        have = got[key]
        if isinstance(value, tuple):
            ok = have == value
        else:
            decimals = len(repr(value).split(".")[1]) if isinstance(value, float) else 0
            ok = round(have, decimals) == value
        if not ok:
            problems.append(f"{name} {key}: notes {value}, records {have}")
    return problems


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    spec = load_spec()
    problems: List[str] = []
    columns = [(title, design_values(names, spec), rungs, headline)
               for title, names, rungs, headline in DESIGN]
    for title, values, _, _ in columns:
        print(f"{title:<16} " + " | ".join(f"{k}={v}" for k, v in sorted(values.items())))

    tasks = {name: GateTask(spec["conditions"][name]["run"]) for name, _ in CELLS}
    scored = {}
    for name, label in CELLS:
        m = compliance(tasks[name])
        scored[name] = m
        print(f"{name:<7} repeats {m['repeat_rate']:.4f} forced {m['forced_close_rate']:.4f} "
              f"reasoning {m['reasoning_tokens']:.1f} turns {m['turns_per_record']:.2f} "
              f"declared {m['declared']}/{m['cases']} timeouts {m['timeouts']} no answer "
              f"{m['no_answer']} context {m['context_limit']} probe-CLI {m['probe_cli_agree']}/"
              f"{m['probe_cli_checked']} +{m['probe_cli_excused']} searched {m['searched']}")
        want = dict(EXPECTED.get(name, {}))
        if "probe_cli" in want:
            m["probe_cli"] = (m["probe_cli_agree"], m["probe_cli_checked"])
        problems += _reconcile(name, m, want)

    pool = ac._read_json(paths.DATA / "ocpq_dataset.json")
    own = {}
    for name, _ in CELLS:
        by_key = shown_examples(tasks[name])
        own[name] = {(ac.entry_key(r), r.get("repeat")): by_key.get(_key_of(r)) or []
                     for r in tasks[name].records}
    gold = gold_mechanism(tasks["Q_A1"])
    problems += _reconcile("gold", {**gold, "sat": gold["sat"] / gold["questions"],
                                    "one_box": gold["one_box"] / gold["questions"]},
                           EXPECTED_MECHANISM["gold"])
    examples = {}
    for name, _ in CELLS:
        m = mechanism(tasks[name], own[CONTROL_OF.get(name, name)], pool)
        examples[name] = m
        print(f"{name:<7} SAT {m['sat']}/{m['finals']} boxes {m['boxes']:.3f} one-box "
              f"{m['one_box']} borrowed {m['borrowed']}/{m['compared']} searched {m['searched']} "
              f"F1 {ac.fmt(m['searched_f1'], 3)} / {ac.fmt(m['not_searched_f1'], 3)}")
        if name in EXPECTED_MECHANISM:
            got = {**m, "borrowed": m["borrowed"] / m["compared"] if m["compared"] else None}
            problems += _reconcile(name, got, EXPECTED_MECHANISM[name])
    for line in problems:
        print(f"DIFFERS FROM THE NOTES: {line}")
    if args.no_write:
        return 1 if problems else 0

    runs = [spec["conditions"][name]["run"] for name, _ in CELLS]
    design_runs = [spec["conditions"][n]["run"] for _, names, _, _ in DESIGN for n in names]
    written = [
        design_table(columns, [f"input: {run} (config_resolved.json of every shard)"
                               for run in design_runs]),
        compliance_table([scored[name] for name, _ in CELLS], [f"input: {run}" for run in runs]),
        mechanism_table(gold, [examples[name] for name, _ in CELLS],
                        [f"input: {run}" for run in runs]
                        + ["input: data/ocpq_dataset.json (the example pool)"]),
        ac.write_json("agentic_eval2", {"design": {t: v for t, v, _, _ in columns},
                                        "compliance": scored, "examples_mechanism": examples,
                                        "gold": gold}),
    ]
    for path in written:
        print(f"wrote {path}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
