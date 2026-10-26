#!/usr/bin/env python3
"""The pre-registered model gates of the agentic evaluations, as code.

WHY
---
Three gates decided which models the agentic evaluations score (notes 11 §43.18/§43.19, §44.4/
§44.8, §45.4/§45.7), and every verdict table was computed by throwaway code (Thesis/script/
FLAGS.md NS-05). The criteria are now data (`configs/agent_gate_criteria.json`, transcribed from
the pre-registrations) and each measure below is computed from the gate runs' own records. Each
definition was calibrated against the published tables before it was fixed, and every figure the
notes give is kept as `expected`, so a disagreement is printed, never adopted silently.

THE MEASURES (per gate task; one task = one model, 38 dev cases, one draw)
------------------------------------------------------------------------
malformed       malformed turns / turns.
repeats         turns repeating an earlier action of the same case / turns. Records that store
                `agent.repeat_turns` are read; older ones (evaluation 1, before the field
                existed) count actions whose raw text equals an earlier one in the trajectory.
                Turns are `agent.turns`, the evaluator's own denominator, which counts a turn a
                timeout cut short. §43.19 divided by trajectory steps instead, so Glimmer's
                published 57.0% is 56.8% here; the verdict is unaffected.
forced closes   reasoning steps the harness closed at the budget / steps that recorded reasoning.
timeouts        cases with status `timeout`; loop-driven if the case also repeated actions at least
                `loop_timeout_min_repeats` times (an action sent three or more times).
p90 s/turn      90th percentile over cases of LLM seconds / turns, numpy method "higher"
                (evaluation 1: the method that reproduces all four published values).
p90 case s      90th percentile of whole-case seconds, numpy's default linear method (eval 2).
declared        cases ending on a declared answer / cases.
probe-CLI       cases where the probe and the CLI agree, or disagree only because the CLI skipped
                bindings (the pre-registered exception) / cases with both counts. A tree the
                probe declined to evaluate (`agree` is None) has no probe count and is not
                checkable, as in the evaluator's compliance block.
pipeline        cases whose status is not a model outcome (e.g. `probe_unavailable`).
uptake          cases with at least one `search_examples` call / cases (example-lookup gate).
repeats identical  every repeated action whose result equals the result of the turn it repeats,
                ignoring the probe's per-call `evalVersion` counter (§44.8's "all 122").

Usage (from the repo root):
    python src/agent_gate_report.py            # all gates; JSON + one LaTeX table per gate
    python src/agent_gate_report.py --no-write # print only
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis_common as ac
import paths

CRITERIA_PATH = paths.CONFIGS / "agent_gate_criteria.json"

# Statuses that are the model's own outcome. Anything else is a harness or infrastructure error,
# which criterion 6 of §44.4 forbids.
MODEL_OUTCOMES = {"success", "execution_failed", "schema_invalid", "parse_failed",
                  "agent_no_answer", "timeout"}

# The probe stamps every result with a server counter that differs on every call.
_EVAL_VERSION = re.compile(r"""["']?evalVersion["']?\s*:\s*\d+,?\s*""")

# --check tolerance for rates and seconds published to one decimal (or whole seconds).
RATE_TOLERANCE = 0.0006
SECONDS_TOLERANCE = 0.6


class GateTask:
    """One gate task: a run's records and, where present, its trajectories."""

    def __init__(self, run_dir: str):
        self.run = ac.load_run(run_dir)
        traj_path = self.run.path / "agent_trajectories.jsonl"
        self.trajectories: Dict[str, dict] = {}
        if traj_path.exists():
            with open(traj_path, encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        item = json.loads(line)
                        self.trajectories[item["record_key"]] = item["agent"]
        self.records = [r for r in self.run.records if isinstance(r.get("agent"), dict)]

    def steps(self) -> List[List[dict]]:
        """Per case, the trajectory steps (empty lists where no trajectory was written)."""
        return [list(agent.get("trajectory") or []) for agent in self.trajectories.values()]


def _action_key(raw: str) -> str:
    """The action a step sent, without its free-text `thought`.

    With the thought field on, every raw step starts with the model's thought, so two identical
    actions with different thoughts would never match. The thought is not the action; the
    comparison drops it (an unparseable step is compared as written).
    """
    try:
        action = json.loads(raw)
    except ValueError:
        return raw
    if isinstance(action, dict):
        action.pop("thought", None)
        return json.dumps(action, sort_keys=True)
    return raw


def _raw_repeats(steps: List[dict]) -> List[Tuple[int, int]]:
    """(earlier index, repeating index) for every step whose action was already sent."""
    last_seen: Dict[str, int] = {}
    pairs = []
    for index, step in enumerate(steps):
        raw = step.get("raw")
        if raw is None:
            continue
        raw = _action_key(raw)
        if raw in last_seen:
            pairs.append((last_seen[raw], index))
        last_seen[raw] = index
    return pairs


def measure(task: GateTask, loop_min_repeats: int) -> dict:
    """Every gate measure for one task (see the module docstring for the definitions)."""
    records = task.records
    agents = [r["agent"] for r in records]
    steps = task.steps()
    turns = sum(int(a.get("turns") or 0) for a in agents)
    stored = all("repeat_turns" in a for a in agents)
    repeats = (sum(int(a.get("repeat_turns") or 0) for a in agents) if stored
               else sum(len(_raw_repeats(s)) for s in steps))
    malformed = sum(int(a.get("malformed_actions") or 0) for a in agents)

    summaries = [t for a in agents for t in (a.get("turn_summary") or [])]
    reasoning_steps = [t for t in summaries if t.get("reasoning_tokens") is not None]
    forced = sum(1 for t in reasoning_steps if t.get("forced_close"))

    def case_repeats(record: dict) -> int:
        if "repeat_turns" in record["agent"]:
            return int(record["agent"].get("repeat_turns") or 0)
        key = _key_of(record)
        return len(_raw_repeats(list((task.trajectories.get(key) or {}).get("trajectory") or [])))

    timeouts = [r for r in records if r.get("status") == "timeout"]
    loop_timeouts = [r for r in timeouts if case_repeats(r) >= loop_min_repeats]

    per_turn = [r["llm"]["duration_sec"] / r["agent"]["turns"]
                for r in records if r["agent"].get("turns") and r.get("llm")]
    durations = [r["total_duration_sec"] for r in records
                 if r.get("total_duration_sec") is not None]

    comparisons = [r["agent"]["probe_vs_cli"] for r in records
                   if isinstance(r["agent"].get("probe_vs_cli"), dict)
                   and r["agent"]["probe_vs_cli"].get("agree") is not None]
    agree = sum(1 for c in comparisons if c.get("agree"))
    excused = sum(1 for c in comparisons if not c.get("agree") and c.get("cli_bindings_skipped"))

    searched = sum(1 for a in agents
                   if any(t.get("action") == "search_examples"
                          for t in a.get("turn_summary") or []))
    n = len(records)
    identical, repeated, differing = _repeats_identical(task)
    return {
        "run": task.run.provenance["run"],
        "cases": n,
        "turns": turns,
        "malformed": malformed,
        "malformed_rate": malformed / turns if turns else None,
        "repeats": repeats,
        "repeat_source": "agent.repeat_turns" if stored else "trajectory raw-action equality",
        "repeat_rate": repeats / turns if turns else None,
        "reasoning_steps": len(reasoning_steps),
        "forced_closes": forced,
        "forced_close_rate": forced / len(reasoning_steps) if reasoning_steps else None,
        "timeouts": len(timeouts),
        "timeout_rate": len(timeouts) / n if n else None,
        "loop_timeouts": len(loop_timeouts),
        "p90_seconds_per_turn": (float(np.percentile(per_turn, 90, method="higher"))
                                 if per_turn else None),
        "p90_case_seconds": float(np.percentile(durations, 90)) if durations else None,
        "declared": sum(1 for a in agents if a.get("final_source") == "declared"),
        "declared_rate": (sum(1 for a in agents if a.get("final_source") == "declared") / n
                          if n else None),
        "probe_cli_checked": len(comparisons),
        "probe_cli_agree": agree,
        "probe_cli_excused": excused,
        "probe_cli_rate": (agree + excused) / len(comparisons) if comparisons else None,
        "pipeline_errors": sorted(r.get("status") for r in records
                                  if r.get("status") not in MODEL_OUTCOMES),
        "uptake": searched / n if n else None,
        "repeats_with_result": repeated,
        "repeats_identical": identical,
        "repeats_differing": differing,
        "status_counts": _counts(r.get("status") for r in records),
    }


def _key_of(record: dict) -> str:
    """The trajectory file's `record_key` for a record (`eval_testset._record_key`)."""
    question, log = ac.entry_key(record)
    ident = hashlib.sha256(f"{question}\x00{log}".encode("utf-8")).hexdigest()[:16]
    return f"{ident}:{record.get('config_id')}:{record.get('repeat')}"


def _counts(values) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _strip_counter(result: str) -> str:
    return _EVAL_VERSION.sub("", result)


def _repeats_identical(task: GateTask) -> Tuple[int, int, List[dict]]:
    """Repeated actions whose result equals the result of the turn they repeat.

    Returns:
        (identical, compared, differing), `differing` naming each repeat whose result changed,
        so that every exception to §44.8's "all repeats are genuine" can be inspected.
    """
    identical = compared = 0
    differing = []
    for key, agent in task.trajectories.items():
        case = list(agent.get("trajectory") or [])
        for earlier, later in _raw_repeats(case):
            first, second = case[earlier].get("result"), case[later].get("result")
            if first is None or second is None:
                continue
            compared += 1
            if _strip_counter(str(first)) == _strip_counter(str(second)):
                identical += 1
            else:
                differing.append({"record_key": key, "turn": case[later].get("turn"),
                                  "repeats_turn": case[earlier].get("turn"),
                                  "action": case[later].get("action")})
    return identical, compared, differing


def verdict(measures: Dict[str, dict], criteria: dict) -> Tuple[bool, List[dict]]:
    """Apply one gate's criteria to a model's task(s).

    Args:
        measures: {task name: measure dict}; evaluation 1 has `thought_off` and `thought_on`,
            evaluation 2 and the example gate a single `gate` task.
        criteria: The gate's criteria block.

    Returns:
        (passes, checks), each check {"criterion", "task", "value", "threshold", "passes"}.
    """
    checks = []

    def add(name: str, task: str, value, threshold, passes: bool) -> None:
        checks.append({"criterion": name, "task": task, "value": value,
                       "threshold": threshold, "passes": bool(passes)})

    for task, m in measures.items():
        add("malformed", task, m["malformed_rate"], criteria["malformed_max"],
            m["malformed_rate"] is not None and m["malformed_rate"] <= criteria["malformed_max"])
        applies = criteria.get("repeat_applies_to")
        if applies is None or applies == task:
            add("repeats", task, m["repeat_rate"], criteria["repeat_max"],
                m["repeat_rate"] is not None and m["repeat_rate"] <= criteria["repeat_max"])
        if "forced_close_max" in criteria:
            rate = m["forced_close_rate"]
            add("forced closes", task, rate, criteria["forced_close_max"],
                rate is None or rate <= criteria["forced_close_max"])
        timeout_ok = m["loop_timeouts"] == 0
        if "timeout_max" in criteria:
            timeout_ok = timeout_ok and m["timeout_rate"] <= criteria["timeout_max"]
        add("timeouts", task, {"timeouts": m["timeouts"], "loop_driven": m["loop_timeouts"]},
            criteria.get("timeout_max"), timeout_ok)
        per_turn = criteria.get("p90_seconds_per_turn_max")
        if per_turn:
            add("p90 s/turn", task, m["p90_seconds_per_turn"], per_turn[task],
                m["p90_seconds_per_turn"] <= per_turn[task])
        if "p90_case_seconds_max" in criteria:
            add("p90 case s", task, m["p90_case_seconds"], criteria["p90_case_seconds_max"],
                m["p90_case_seconds"] <= criteria["p90_case_seconds_max"])
        add("declared", task, m["declared_rate"], criteria["declared_min"],
            m["declared_rate"] >= criteria["declared_min"])
        add("probe-CLI", task, m["probe_cli_rate"], criteria["probe_cli_agreement"],
            m["probe_cli_rate"] is None or m["probe_cli_rate"] >= criteria["probe_cli_agreement"])
        if criteria.get("no_pipeline_errors"):
            add("pipeline errors", task, m["pipeline_errors"], 0, not m["pipeline_errors"])
        if "uptake_min" in criteria:
            add("uptake", task, m["uptake"], criteria["uptake_min"],
                m["uptake"] >= criteria["uptake_min"])
    return all(c["passes"] for c in checks), checks


def reconcile(model: dict, measures: Dict[str, dict], passes: bool) -> List[str]:
    """Differences from the published verdict and figures (first task's measures)."""
    expected = model.get("expected") or {}
    first = next(iter(measures.values()))
    problems = []
    if expected.get("verdict") and expected["verdict"] != ("pass" if passes else "fail"):
        problems.append(f"verdict: computed {'pass' if passes else 'fail'}, "
                        f"notes {expected['verdict']}")
    for key in ("repeat_rate", "forced_close_rate", "uptake"):
        if key in expected and first.get(key) is not None:
            if abs(round(first[key], 2 if key == "uptake" else 3) - expected[key]) > RATE_TOLERANCE:
                problems.append(f"{key}: computed {first[key]:.4f}, notes {expected[key]}")
    for key in ("p90_seconds_per_turn", "p90_case_seconds"):
        if key in expected and abs(first[key] - expected[key]) > SECONDS_TOLERANCE:
            problems.append(f"{key}: computed {first[key]:.1f}, notes {expected[key]}")
    if "probe_cli" in expected:
        got = f"{first['probe_cli_agree']}/{first['probe_cli_checked']}"
        if got != expected["probe_cli"]:
            problems.append(f"probe-CLI: computed {got}, notes {expected['probe_cli']}")
    return problems


# --------------------------------------------------------------------------------------------
# LaTeX.
# --------------------------------------------------------------------------------------------

# Column order and two-line headers of the gate tables (criteria as columns, models as rows).
COLUMN_HEADERS = [
    ("malformed", r"\makecell[r]{Mal-\\formed}"),
    ("repeats", "Repeats"),
    ("forced closes", r"\makecell[r]{Forced\\closes}"),
    ("timeouts", r"\makecell[r]{Time-\\outs}"),
    ("p90 s/turn", r"\makecell[r]{p90 s\\per turn}"),
    ("p90 case s", r"\makecell[r]{p90 s\\per case}"),
    ("declared", r"\makecell[r]{De-\\clared}"),
    ("probe-CLI", r"\makecell[r]{Probe--\\CLI}"),
    ("pipeline errors", r"\makecell[r]{Pipeline\\errors}"),
    ("uptake", "Uptake"),
]


def _cell(check: dict) -> str:
    """One value; a value that fails its criterion is set in bold."""
    value = check["value"]
    name = check["criterion"]
    if name in ("malformed", "repeats", "forced closes", "declared", "uptake"):
        text = ac.fmt_pct(value)
    elif name == "probe-CLI":
        text = "--" if value is None else f"{value:.2f}"
    elif name == "timeouts":
        text = str(value["timeouts"])
        if value["loop_driven"]:
            text += f" ({value['loop_driven']} loop)"
    elif name == "pipeline errors":
        text = str(len(value))
    else:
        text = f"{value:.1f}" if name == "p90 s/turn" else f"{value:.0f}"
    return text if check["passes"] else f"\\textbf{{{text}}}"


def tex_table(gate: dict, results: List[dict]) -> str:
    """Models (and their tasks) as rows, the gate's criteria as columns, the verdict last."""
    used = [(name, header) for name, header in COLUMN_HEADERS
            if any(c["criterion"] == name for r in results for c in r["checks"])]
    multi_task = any(len(r["measures"]) > 1 for r in results)
    # Up to ten criteria plus the verdict: smaller type and tighter columns keep it on the page.
    lines = ["  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X"
             + "r" * len(used) + "l@{}}",
             "    \\toprule",
             "    Model & " + " & ".join(header for _, header in used) + " & Verdict \\\\",
             "    \\midrule"]
    for result in results:
        for position, task in enumerate(result["measures"]):
            label = result["name"] + (f", {task.replace('_', ' ')}" if multi_task else "")
            cells = []
            for name, _ in used:
                match = [c for c in result["checks"]
                         if c["criterion"] == name and c["task"] == task]
                cells.append(_cell(match[0]) if match else "")
            verdict_text = ("passes" if result["passes"] else "fails") if position == 0 else ""
            lines.append(f"    {label} & " + " & ".join(cells) + f" & {verdict_text} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--criteria", type=Path, default=CRITERIA_PATH)
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(args.criteria, encoding="utf-8") as handle:
        spec = json.load(handle)
    payload, runs, differences = {}, [], 0
    for gate in spec["gates"]:
        criteria = gate["criteria"]
        results, gate_runs = [], []
        print(f"\n== {gate['title']} (rule notes {gate['rule_section']}, "
              f"published notes {gate['verdict_section']}) ==")
        for model in gate["models"]:
            tasks = {name: GateTask(run) for name, run in model["runs"].items()}
            gate_runs += [t.run for t in tasks.values()]
            measures = {name: measure(t, criteria["loop_timeout_min_repeats"])
                        for name, t in tasks.items()}
            passes, checks = verdict(measures, criteria)
            problems = reconcile(model, measures, passes)
            differences += bool(problems)
            failed = sorted({c["criterion"] for c in checks if not c["passes"]})
            print(f"  {model['name']:<30} {'passes' if passes else 'fails '} "
                  f"{('on ' + ', '.join(failed)) if failed else ''}")
            for problem in problems:
                print(f"      differs from the notes -- {problem}")
            results.append({"name": model["name"], "passes": passes, "checks": checks,
                            "measures": measures, "differences_from_notes": problems})
        compared = sum(m["repeats_with_result"] for r in results for m in r["measures"].values())
        same = sum(m["repeats_identical"] for r in results for m in r["measures"].values())
        print(f"  repeats returning the repeated turn's result: {same}/{compared}")
        payload[gate["id"]] = {"title": gate["title"], "rule_section": gate["rule_section"],
                               "criteria": criteria, "models": results,
                               "repeats_identical": {"identical": same, "compared": compared}}
        if not args.no_write:
            caption = (f"{gate['title']}: each pre-registered criterion, computed from the gate "
                       f"runs. Values that fail their criterion are in bold.")
            ac.write_tex(f"agent_gate_{gate['id']}", tex_table(gate, results), caption,
                         f"tab:gate:{gate['id']}", gate_runs,
                         notes=[f"criteria: notes 11 section {gate['rule_section']}; published "
                                f"verdicts: section {gate['verdict_section']}"])
        runs += gate_runs
    print(f"\n{differences} model(s) differ from the notes in a verdict or a published figure.")
    if not args.no_write:
        out = ac.write_json("agent_gates", payload, _unique(runs))
        print(f"wrote {out}")
    return 0


def _unique(runs: List[ac.Run]) -> List[ac.Run]:
    seen, unique = set(), []
    for run in runs:
        if run.provenance["run"] not in seen:
            seen.add(run.provenance["run"])
            unique.append(run)
    return unique


if __name__ == "__main__":
    sys.exit(main())
