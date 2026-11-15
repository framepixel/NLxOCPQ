#!/usr/bin/env python3
"""The validation-repair switch never touches generation: the evidence, per run.

WHY
---
`schema_enforced` names one dimension of the matrix, and notes 11 §29 showed it switches only
post-hoc validation and one repair call, never decoding (FLAGS IC-02). Its evidence was computed by
throwaway code (FLAGS NS-12). This script recomputes it on any matrix-shaped run:

identity     pairs of records sharing model x strategy x question x draw, one per arm. Where BOTH
             used exactly one LLM call (no repair ran), the two arms must be the same generation:
             identical prompt tokens and identical completion length.
call counts  pairs whose call counts agree; where they differ, the repair pass is what differs.
repair       from the run's `failure_log.jsonl`: `verify_pre_fix` entries are repair attempts,
             `verify_fix_failed` the attempts that still failed validation.
effect       pooled strict F1 with the switch on and off.

§29 quotes its repair counts (2,043 attempts, 1,613 failures) as the matrix's; they are the
self-consistency run's. On the matrix they are 2,776 and 2,450 (`matrix_full`), and the script
reports each run's own counts beside the identity evidence.

Usage (from the repo root):
    python src/schema_arm_identity.py [runs ...]
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import analysis_common as ac
import paths

EXPECTED_PATH = paths.CONFIGS / "analysis_expected.json"
DEFAULT_RUNS = ["eval_runs/matrix_v2/merged", "eval_runs/matrix_full", "logs/selfcons/merged"]


def pair_key(record: dict) -> tuple:
    return (record.get("model_label"), record.get("prompt_strategy"), ac.entry_key(record),
            record.get("repeat"))


def identity(records: List[dict]) -> dict:
    """Identity evidence between the validation-repair arms of one run."""
    on = {pair_key(r): r for r in records if r.get("schema_enforced")}
    off = {pair_key(r): r for r in records if not r.get("schema_enforced")}
    pairs = [(on[k], off[k]) for k in on if k in off]

    def llm(record: dict) -> dict:
        return record.get("llm") or {}

    single = [(a, b) for a, b in pairs if llm(a).get("calls") == 1 and llm(b).get("calls") == 1]
    return {
        "pairs": len(pairs),
        "single_call_pairs": len(single),
        "identical_prompt_tokens": sum(llm(a).get("prompt_tokens") == llm(b).get("prompt_tokens")
                                       for a, b in single),
        "identical_completion_chars": sum(
            llm(a).get("completion_chars") == llm(b).get("completion_chars") for a, b in single),
        "equal_call_counts": sum(llm(a).get("calls") == llm(b).get("calls") for a, b in pairs),
        "strict_f1_on": ac.mean(ac.strict_f1(a) for a, _ in pairs),
        "strict_f1_off": ac.mean(ac.strict_f1(b) for _, b in pairs),
    }


def repair_counts(run_dir: Path) -> Dict[str, Optional[int]]:
    """Repair attempts and failures from the run's failure log."""
    path = run_dir / "failure_log.jsonl"
    if not path.exists():
        return {"repair_runs": None, "repair_failures": None}
    stages: Counter = Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                stages[json.loads(line).get("stage")] += 1
    return {"repair_runs": stages["verify_pre_fix"], "repair_failures": stages["verify_fix_failed"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(EXPECTED_PATH, encoding="utf-8") as handle:
        expected = json.load(handle)["schema_arm_identity"]
    reports, problems, runs = [], [], []
    for run_dir in args.runs:
        run = ac.load_run(run_dir)
        runs.append(run)
        report = {"run": run.provenance["run"], **identity(run.records),
                  **repair_counts(run.path)}
        if report["repair_runs"]:
            report["repair_failure_rate"] = report["repair_failures"] / report["repair_runs"]
        reports.append(report)
        for key, want in (expected.get(report["run"]) or {}).items():
            if key != "section" and report.get(key) != want:
                problems.append(f"{report['run']}: {key} computed {report.get(key)}, notes {want}")
        print(f"{report['run']}: {report['pairs']} pairs; single-call {report['single_call_pairs']} "
              f"(prompt tokens identical {report['identical_prompt_tokens']}, completion chars "
              f"identical {report['identical_completion_chars']}); equal calls "
              f"{report['equal_call_counts']}; repair {report['repair_runs']} runs / "
              f"{report['repair_failures']} failed; F1 on {report['strict_f1_on']} off "
              f"{report['strict_f1_off']}")
    print(f"\n{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if not args.no_write:
        out = ac.write_json("schema_arm_identity", {"runs": reports,
                                                    "differences_from_notes": problems}, runs)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
