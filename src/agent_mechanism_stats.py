#!/usr/bin/env python3
"""How the agents used the log: tool calls, empty results and revisions, from the trajectories.

WHY
---
The mechanism paragraphs of notes 11 §43.15 and §44.9 ("892 `try_query`, 141 records saw an empty
result, 99 still ended on one, the next action differs in 90%") were computed by throwaway code
(Thesis/script/FLAGS.md NS-06). This script computes them from each rung's
`agent_trajectories.jsonl` with fixed definitions. Each was calibrated against the published
figures; the one deliberate departure (the root rule) is documented below and reported beside the
notes' values.

DEFINITIONS (over the records of the population, default the 135 discriminating questions x draws)
-------------------------------------------------------------------------------------------------
tool calls          trajectory steps by action.
try_query           calls, calls per record, distinct trees per record, records with at least one.
empty result        a `try_query` whose ROOT box selected no situation. The root is located
                    structurally -- the box no other box lists as a child -- as the metric does
                    (`eval_testset.root_node_index`): it is not always node 0. The notes read
                    node 0, which differs on one A1* tree (485 empty results there, 484 here).
next action differs the step after an empty result sends a different action.
mentions empty      the step after an empty result has reasoning matching the configured keyword
                    pattern; the same share after NON-empty results is the base rate. The notes
                    did not record their keywords; this pattern (configs/agent_mechanism.json)
                    reproduces their 356 mentions exactly, and is the canonical one from now on.
ended empty         records whose last evaluated `try_query` was empty, among those with any.
population change   among records with two or more evaluated `try_query`, those whose root
                    situation count changed between tries.

Usage (from the repo root):
    python src/agent_mechanism_stats.py              # JSON + LaTeX table
    python src/agent_mechanism_stats.py --no-write   # print and reconcile only
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import analysis_common as ac
import paths
from agent_gate_report import _key_of

CONFIG_PATH = paths.CONFIGS / "agent_mechanism.json"


def root_index(raw_action: Optional[str]) -> Optional[int]:
    """Index of the root box of a `try_query` tree, exactly as `eval_testset.root_node_index`.

    The first box that no box lists as a child; 0 when every box is referenced (a cycle) or the
    tree is empty -- the evaluator's own fallback, so an agent's empty result means what the
    metric would mean by it.

    Returns:
        The index, or None when the action itself cannot be parsed.
    """
    try:
        tree = json.loads(raw_action or "")["args"]["tree"]
    except (ValueError, KeyError, TypeError):
        return None
    nodes = tree.get("nodes") if isinstance(tree, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return 0
    referenced = set()
    for node in nodes:
        box = node.get("Box") if isinstance(node, dict) else None
        if isinstance(box, list) and len(box) == 2 and isinstance(box[1], list):
            referenced.update(c for c in box[1] if isinstance(c, int))
    for index in range(len(nodes)):
        if index not in referenced:
            return index
    return 0


def root_situations(step: dict) -> Optional[int]:
    """Root situation count of an evaluated `try_query` step; None if not evaluated."""
    result = step.get("result")
    if not isinstance(result, dict) or not result.get("nodes"):
        return None
    index = root_index(step.get("raw"))
    if index is None:
        return None
    for node in result["nodes"]:
        if node.get("node") == index:
            return int(node.get("situations", 0))
    return None


def load_trajectories(pattern: str) -> Dict[str, dict]:
    """`record_key -> agent` from every agent_trajectories.jsonl the glob matches."""
    found: Dict[str, dict] = {}
    directories = sorted(glob.glob(str(paths.REPO_ROOT / pattern)))
    if not directories:
        raise FileNotFoundError(f"no shard directories match {pattern}")
    for directory in directories:
        path = Path(directory) / "agent_trajectories.jsonl"
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    found[item["record_key"]] = item["agent"]
    return found


def describe(records: List[dict], trajectories: Dict[str, dict], mention: re.Pattern) -> dict:
    """Every mechanism statistic for one rung's records."""
    tools: Counter = Counter()
    try_calls = distinct = with_try = 0
    with_empty = ended_empty = empty_results = differs = 0
    mention_after_empty = mention_after_nonempty = nonempty_results = 0
    two_or_more = changed = 0
    missing = 0
    reasoning_steps = 0
    for record in records:
        agent = trajectories.get(_key_of(record))
        if agent is None:
            missing += 1
            continue
        steps = list(agent.get("trajectory") or [])
        tools.update(str(step.get("action")) for step in steps)
        reasoning_steps += sum(1 for step in steps if step.get("reasoning"))
        tries = [step for step in steps if step.get("action") == "try_query"]
        try_calls += len(tries)
        distinct += len({step.get("raw") for step in tries})
        with_try += bool(tries)
        counts, saw_empty, last_empty = [], False, None
        for index, step in enumerate(steps):
            if step.get("action") != "try_query":
                continue
            situations = root_situations(step)
            if situations is None:
                continue
            counts.append(situations)
            empty = situations == 0
            last_empty = empty
            following = steps[index + 1] if index + 1 < len(steps) else None
            reasoning = (following or {}).get("reasoning") or ""
            if empty:
                saw_empty = True
                empty_results += 1
                if following is not None:
                    differs += following.get("raw") != step.get("raw")
                    mention_after_empty += bool(mention.search(reasoning))
            elif following is not None:
                nonempty_results += 1
                mention_after_nonempty += bool(mention.search(reasoning))
        with_empty += saw_empty
        ended_empty += bool(saw_empty and last_empty)
        if len(counts) >= 2:
            two_or_more += 1
            changed += len(set(counts)) > 1
    n = len(records) - missing
    if not reasoning_steps:
        # A rung without a reasoning channel (evaluation 1) cannot mention anything; its counts
        # are undefined, not zero.
        mention_after_empty = mention_after_nonempty = None
    return {
        "records": n,
        "records_without_trajectory": missing,
        "tool_calls": dict(sorted(tools.items())),
        "try_query": try_calls,
        "try_query_per_record": try_calls / n if n else None,
        "distinct_try_query_per_record": distinct / n if n else None,
        "records_with_try_query": with_try,
        "records_with_empty": with_empty,
        "empty_results": empty_results,
        "next_action_differs": differs,
        "mentions_empty_after_empty": mention_after_empty,
        "mentions_empty_after_nonempty": mention_after_nonempty,
        "nonempty_results_followed": nonempty_results,
        "ended_empty": ended_empty,
        "two_or_more_tries": two_or_more,
        "root_count_changed": changed,
    }


def reconcile(stats: dict, expected: dict) -> List[str]:
    """Published counts that the script does not reproduce.

    A difference listed in `expected.known_differences` is printed with its reason, so the
    reconciliation still shows it, but it is not counted as unexplained.
    """
    problems = []
    known = expected.get("known_differences") or {}
    for key, want in expected.items():
        if key in ("section", "known_differences"):
            continue
        if key in known and stats.get(key) != want:
            problems.append(f"{key}: computed {stats.get(key)}, notes {want} "
                            f"(known: {known[key]})")
            continue
        if key == "tool_calls":
            for tool, count in want.items():
                got = stats["tool_calls"].get(tool, 0)
                if got != count:
                    problems.append(f"{tool} calls: computed {got}, notes {count}")
        elif stats.get(key) != want:
            problems.append(f"{key}: computed {stats.get(key)}, notes {want}")
    return problems


def _share(part: Optional[int], whole: int) -> str:
    return "--" if part is None or not whole else f"{part} ({100 * part / whole:.0f}\\%)"


def tex_table(rows: List[dict]) -> str:
    """Rungs as columns (two-line `column` headers), statistics as rows in a wrapping column."""
    labels = [f"\\makecell[r]{{{r['column'][0]}\\\\{r['column'][1]}}}" for r in rows]
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X"
             + "r" * len(rows) + "@{}}",
             "    \\toprule", "    & " + " & ".join(labels) + " \\\\", "    \\midrule"]

    def add(name: str, cells: List[str]) -> None:
        lines.append(f"    {name} & " + " & ".join(cells) + " \\\\")

    stats = [r["stats"] for r in rows]
    add("records", [str(s["records"]) for s in stats])
    add("\\texttt{try\\_query} calls per record", [ac.fmt(s["try_query_per_record"], 2)
                                                    for s in stats])
    add("distinct trees tried per record", [ac.fmt(s["distinct_try_query_per_record"], 2)
                                             for s in stats])
    add("records that saw an empty result", [str(s["records_with_empty"]) for s in stats])
    add("next action differs after one", [_share(s["next_action_differs"], s["empty_results"])
                                          for s in stats])
    add("reasoning mentions it", [_share(s["mentions_empty_after_empty"], s["empty_results"])
                                  for s in stats])
    add("\\quad base rate after a non-empty result",
        [_share(s["mentions_empty_after_nonempty"], s["nonempty_results_followed"])
         for s in stats])
    add("still ended on an empty result", [_share(s["ended_empty"], s["records_with_empty"])
                                           for s in stats])
    add("root count changed between tries", [_share(s["root_count_changed"],
                                                    s["two_or_more_tries"]) for s in stats])
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    mention = re.compile(config["empty_mention_pattern"], re.IGNORECASE)
    population = config["population"]
    rows, runs, differences = [], [], 0
    for rung in config["rungs"]:
        run = ac.load_run(rung["run"])
        ac.assert_reproduces_summary(run)
        runs.append(run)
        records = [r for r in run.records if ac.in_population(r, population)]
        stats = describe(records, load_trajectories(rung["trajectories"]), mention)
        problems = reconcile(stats, rung.get("expected") or {})
        differences += any("(known:" not in p for p in problems)
        print(f"{rung['label']:<40} records {stats['records']:<4} try_query "
              f"{stats['try_query']:<4} saw empty {stats['records_with_empty']:<4} ended empty "
              f"{stats['ended_empty']}")
        for problem in problems:
            print(f"    differs from the notes -- {problem}")
        rows.append({"id": rung["id"], "label": rung["label"], "column": rung["column"],
                     "stats": stats, "differences_from_notes": problems})
    print(f"\n{differences} rung(s) differ from the notes beyond the known differences.")
    if args.no_write:
        return 0
    caption = (f"How the agents used the log, over {ac.POPULATION_LABELS[population]} (both "
               f"draws in evaluation~2). Evaluation~1 is Ministral 3 14B, evaluation~2 Qwen3.8; "
               f"the last two columns add example lookup (S). An empty result is a "
               f"\\texttt{{try\\_query}} whose root box selected nothing. `Reasoning mentions it' "
               f"is a keyword check; its base rate after non-empty results is given below it.")
    payload = {"population": population,
               "empty_mention_pattern": config["empty_mention_pattern"], "rungs": rows}
    out = ac.write_json("agent_mechanism", payload, runs)
    table = ac.write_tex("agent_mechanism", tex_table(rows), caption, "tab:agent:mechanism", runs)
    print(f"wrote {out}\nwrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
