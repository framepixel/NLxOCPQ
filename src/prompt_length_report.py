#!/usr/bin/env python3
"""How long the prompts were: the schema block per log, and the longest prompt per condition.

WHY
---
Notes 11 §13.1 sized the schema-block budget from per-log figures (angular 10,493 tokens, bpic2017
4,846, reasoning_benchmark 2,273) and §6.5/§13.1 argued from them that inference prompts run far
past the adapters' 4,096-token training length (FLAGS NS-14, TH train/inference divergence). No
script produced the figures, and they are not tokenizer counts: `build_schema_block` budgets with
`LLM._estimated_tokens`, characters / 3.6. This script reports both, so the thesis can state which.

per log         the schema block exactly as the pipeline renders it (`LLM.build_schema_block` on
                `LogEvents.extract_ocel_schema`), in characters, in the pipeline's estimated
                tokens, and in real tokens under each fine-tuned model's own tokenizer (the
                `tokenizer.json` saved with its adapter).
per condition   the longest prompt any run actually sent, from each run's `summary.json`
                (`max_prompt_tokens`, counted by the serving model's tokenizer), against the
                4,096-token training length.

Needs the evaluation environment (it imports `LLM`, hence torch and transformers).

Usage (from the repo root):
    python src/prompt_length_report.py
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import analysis_common as ac
import paths

EXPECTED_ESTIMATED_TOKENS = {  # notes 11 §13.1, "full block"
    "angular_github_commits_ocel.xml": 10493,
    "bpic2017-ocel2.json": 4846,
    "reasoning_benchmark.json": 2273,
}
EXPECTED_OTHERS_AT_MOST = 1437
TRAINING_MAX_SEQ_LENGTH = 4096  # train_unsloth's resolved max_seq_length (job logs, notes 04)
RUNS = ["eval_runs/matrix_v2/merged", "eval_runs/unleashed/merged_m1_s0",
        "eval_runs/unleashed/merged_m1_s1", "eval_runs/unleashed/merged_m1_s2"]


def tokenizers_by_model() -> Dict[str, object]:
    """Each fine-tuned model's tokenizer, loaded from the file saved beside its adapter."""
    from tokenizers import Tokenizer

    found = {}
    for adapter in sorted((paths.MODELS / "adapters").iterdir()):
        path = adapter / "tokenizer.json"
        if path.exists():
            found[adapter.name.replace("-fine-tuned-quantized", "")] = Tokenizer.from_file(str(path))
    return found


def schema_blocks(logs: List[str]) -> List[dict]:
    """Each log's schema block, measured three ways."""
    import LLM
    from LogEvents import extract_ocel_schema

    tokenizers = tokenizers_by_model()
    rows = []
    for log in logs:
        block = LLM.build_schema_block(extract_ocel_schema(str(paths.OCEL_LOGS / log)))
        rows.append({
            "log": log,
            "characters": len(block),
            "estimated_tokens": LLM._estimated_tokens(block),
            "omitted_sections": LLM.get_last_schema_omissions(),
            "tokens": {name: len(tok.encode(block, add_special_tokens=False).ids)
                       for name, tok in tokenizers.items()},
        })
    return sorted(rows, key=lambda row: -row["estimated_tokens"])


def longest_prompts(run_dirs: List[str]) -> List[dict]:
    """The longest single prompt and the largest per-case prompt total, per condition.

    A record's `llm.prompt_tokens` is summed over all of its LLM calls, and the summary's
    `max_prompt_tokens` is the maximum of that sum -- a per-case total, not a prompt length, so it
    cannot show a context overflow (an agent case of 8 turns sums 8 prompts). A record that made
    exactly one call does carry one prompt's length, so `longest_single_prompt` is measured on
    those records only, and says how many there were.
    """
    rows = []
    for run_dir in run_dirs:
        run = ac.load_run(run_dir)
        by_config: Dict[str, List[dict]] = {}
        for record in run.records:
            by_config.setdefault(record.get("config_id"), []).append(record)
        for config_id, records in by_config.items():
            single = [r["llm"]["prompt_tokens"] for r in records
                      if (r.get("llm") or {}).get("calls") == 1
                      and r["llm"].get("prompt_tokens") is not None]
            totals = [r["llm"]["prompt_tokens"] for r in records
                      if (r.get("llm") or {}).get("prompt_tokens") is not None]
            rows.append({"run": run.provenance["run"], "config_id": config_id,
                         "model": records[0].get("model_label"),
                         "strategy": records[0].get("prompt_strategy"),
                         "single_call_records": len(single),
                         "longest_single_prompt": max(single) if single else None,
                         "largest_case_prompt_total": max(totals) if totals else None})
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(paths.DATA / "testset.json", encoding="utf-8") as handle:
        logs = sorted({os.path.basename(t["ocel_path"].replace("\\", "/"))
                       for t in json.load(handle)})
    blocks = schema_blocks(logs)
    problems = []
    for row in blocks:
        want = EXPECTED_ESTIMATED_TOKENS.get(row["log"])
        if want is not None and row["estimated_tokens"] != want:
            problems.append(f"{row['log']}: estimated {row['estimated_tokens']}, notes {want}")
        if want is None and row["estimated_tokens"] > EXPECTED_OTHERS_AT_MOST:
            problems.append(f"{row['log']}: estimated {row['estimated_tokens']}, notes at most "
                            f"{EXPECTED_OTHERS_AT_MOST}")
        print(f"{row['log']:<34} {row['characters']:>7} chars {row['estimated_tokens']:>6} est. "
              + " ".join(f"{k} {v}" for k, v in row["tokens"].items())
              + (f"  OMITTED {row['omitted_sections']}" if row["omitted_sections"] else ""))
    prompts = longest_prompts(RUNS)
    by_model: Dict[str, int] = {}
    for row in prompts:
        if row["longest_single_prompt"]:
            by_model[row["model"]] = max(by_model.get(row["model"], 0),
                                         row["longest_single_prompt"])
    print("\nlongest single prompt sent (one-call records), per model; training length 4,096:")
    for model, tokens in sorted(by_model.items(), key=lambda kv: -kv[1]):
        print(f"  {model:<36} {tokens:>7}  ({tokens / TRAINING_MAX_SEQ_LENGTH:.1f}x)")
    print(f"\n{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if args.no_write:
        return 0
    out = ac.write_json("prompt_length_report", {
        "chars_per_estimated_token": 3.6, "training_max_seq_length": TRAINING_MAX_SEQ_LENGTH,
        "schema_blocks": blocks, "longest_prompt_by_model": by_model,
        "longest_prompt_by_condition": prompts, "differences_from_notes": problems})
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
