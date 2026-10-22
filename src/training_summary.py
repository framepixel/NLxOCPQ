#!/usr/bin/env python3
"""The fine-tuning runs: what the adapters are, measured, and how training went, where logged.

WHY
---
The v5 training table in notes 04 was copied by hand from the job logs, and Ministral 3 3B is
missing from it (Thesis/script/FLAGS.md NS-03). Three of the four logs no longer exist. So this
script separates what can still be measured from what can only be cited:

measured from the adapter files (all four)
    LoRA tensors and parameters read from the `adapter_model.safetensors` header -- no torch, the
    header is JSON -- with the adapted layers, the projection modules, and the vision-tower
    tensors that must be zero (the multimodal checkpoints train the language tower only); rank,
    alpha and base checkpoint from `adapter_config.json`; the file size.
parsed from a surviving job log (Ministral 3 3B)
    corpus rows and the train/validation split, trainable and total parameters, steps, training
    time, samples per second, final training loss, best evaluation loss.
read from the last checkpoint's trainer state (all but Gemma 3 4B, whose `outputs/` is gone)
    every evaluation loss (so the best one is measured), the last logged training loss and the
    largest gradient norm. The state predates the run's closing line, so it has no runtime.
read from SLURM accounting (all but Ministral 3 3B, whose time is in its log)
    the job's wall time including model loading (`sacct`, recorded in the config).
transcribed (the jobs whose logs were deleted)
    training time, samples per second and the averaged training loss, as notes 04 recorded
    them; the table marks each with a dagger.

Usage (from the repo root):
    python src/training_summary.py
"""
import argparse
import ast
import json
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import analysis_common as ac
import paths

RUNS_PATH = paths.CONFIGS / "training_runs.json"
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def safetensors_header(path: Path) -> Dict[str, Any]:
    """The tensor table of a safetensors file: name -> {dtype, shape, data_offsets}."""
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header


def inspect_adapter(directory: Path) -> dict:
    """Everything about a LoRA adapter that its files state."""
    weights = directory / "adapter_model.safetensors"
    header = safetensors_header(weights)
    names = list(header)
    layers = {int(m.group(1)) for name in names
              for m in [re.search(r"layers\.(\d+)\.", name)] if m}
    modules = Counter(m.group(1) for name in names
                      for m in [re.search(r"\.(\w+_proj)\.lora_", name)] if m)
    params = 0
    for tensor in header.values():
        count = 1
        for dim in tensor["shape"]:
            count *= dim
        params += count
    with open(directory / "adapter_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    return {
        "lora_tensors": len(names),
        "layers": len(layers),
        "tensors_per_layer_and_module_complete": (
            set(modules) == set(PROJECTIONS)
            and all(count == 2 * len(layers) for count in modules.values())),
        "trainable_params": params,
        "vision_tensors": sum(1 for name in names if "vision" in name),
        "file_mib": round(weights.stat().st_size / 2 ** 20, 1),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "base_model": config.get("base_model_name_or_path"),
    }


def _number(text: str) -> float:
    return float(text.replace(",", ""))


def parse_log(path: Path) -> dict:
    """Training figures from an unsloth job log (`ocpq-lora-*.out`)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found: Dict[str, Any] = {}
    patterns = {
        "corpus_rows": r"Loaded (\d+) supervised rows",
        "train_rows": r"Train rows: (\d+)",
        "validation_rows": r"Validation rows: (\d+)",
        "steps": r"Total steps = ([\d,]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        found[key] = int(_number(match.group(1))) if match else None
    match = re.search(r"Trainable parameters = ([\d,]+) of ([\d,]+)", text)
    if match:
        found["trainable_params_logged"] = int(_number(match.group(1)))
        found["total_params"] = int(_number(match.group(2)))
    evaluations, final = [], None
    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{'") and line.endswith("}")):
            continue
        try:
            entry = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if "eval_loss" in entry:
            evaluations.append(float(entry["eval_loss"]))
        if "train_runtime" in entry:
            final = entry
    found["evaluations"] = len(evaluations)
    found["best_eval_loss"] = min(evaluations) if evaluations else None
    found["last_eval_loss"] = evaluations[-1] if evaluations else None
    if final:
        found["train_runtime_sec"] = float(final["train_runtime"])
        found["wall_hours"] = float(final["train_runtime"]) / 3600
        found["samples_per_second"] = float(final["train_samples_per_second"])
        found["train_loss"] = float(final["train_loss"])
    return found


def parse_trainer_state(path: Path) -> dict:
    """Losses and evaluations from a checkpoint's `trainer_state.json` (HF Trainer).

    The state is saved at each checkpoint, before the run's closing summary line, so it holds
    every logged training loss and evaluation but not the total runtime or the averaged train
    loss.
    """
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
    history = state.get("log_history") or []
    evaluations = [float(e["eval_loss"]) for e in history if "eval_loss" in e]
    losses = [e for e in history if "loss" in e]
    return {
        "global_step": state.get("global_step"),
        "evaluations": len(evaluations),
        "best_eval_loss": min(evaluations) if evaluations else None,
        "last_eval_loss": evaluations[-1] if evaluations else None,
        "last_logged_train_loss": float(losses[-1]["loss"]) if losses else None,
        "max_grad_norm": max((float(e.get("grad_norm", 0.0)) for e in losses), default=None),
    }


def slurm_hours(slurm: Optional[dict]) -> Optional[float]:
    """A `sacct` Elapsed value (`[D-]HH:MM:SS`) in hours."""
    if not slurm or not slurm.get("elapsed"):
        return None
    elapsed = slurm["elapsed"]
    days = 0
    if "-" in elapsed:
        day_text, elapsed = elapsed.split("-", 1)
        days = int(day_text)
    hours, minutes, seconds = (int(part) for part in elapsed.split(":"))
    return days * 24 + hours + minutes / 60 + seconds / 3600


def reconcile(row: dict, expected: dict) -> List[str]:
    problems = []
    for key, want in expected.items():
        got = row.get(key)
        if isinstance(want, float) and got is not None:
            same = abs(got - want) <= 10 ** -(len(str(want).split(".")[1]))
        elif key == "train_runtime_sec" and got is not None:
            same = round(got, -1) == want
        else:
            same = got == want
        if not same:
            problems.append(f"{row['model']}: {key} computed {got}, notes {want}")
    return problems


def tex_table(rows: List[dict]) -> str:
    def cell(row, key, fmt):
        value = row.get(key)
        if value is None:
            value = (row.get("transcribed") or {}).get(key)
            return "--" if value is None else fmt(value) + "$^{\\dagger}$"
        return fmt(value)

    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}X" + "r" * len(rows)
             + "@{}}",
             "    \\toprule", "    & " + " & ".join(r["model"] for r in rows) + " \\\\",
             "    \\midrule"]
    specs = [
        ("SLURM job", lambda r: str(r["job"])),
        ("Adapted layers", lambda r: str(r["layers"])),
        ("LoRA tensors", lambda r: str(r["lora_tensors"])),
        ("Trainable parameters", lambda r: f"\\num{{{r['trainable_params']}}}"),
        ("Adapter size (MiB)", lambda r: f"{r['file_mib']:.1f}"),
        ("Vision-tower tensors", lambda r: str(r["vision_tensors"])),
        ("Job time, SLURM (h)", lambda r: "--" if r.get("job_hours") is None
         else f"{r['job_hours']:.2f}"),
        ("Training time (h)", lambda r: cell(r, "wall_hours", lambda v: f"{v:.2f}")),
        ("Samples per second", lambda r: cell(r, "samples_per_second", lambda v: f"{v:.3f}")),
        ("Final training loss", lambda r: cell(r, "train_loss", lambda v: f"{v:.5f}")),
        ("Best evaluation loss", lambda r: cell(r, "best_eval_loss", lambda v: f"{v:.6f}")),
    ]
    for name, render in specs:
        lines.append(f"    {name} & " + " & ".join(render(r) for r in rows) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    with open(RUNS_PATH, encoding="utf-8") as handle:
        runs = json.load(handle)["runs"]
    rows, problems = [], []
    for run in runs:
        row = {"model": run["model"], "label": run["label"], "job": run["job"]}
        row.update(inspect_adapter(paths.REPO_ROOT / run["adapter"]))
        row["job_hours"] = slurm_hours(run.get("slurm"))
        if run.get("trainer_state"):
            row.update(parse_trainer_state(paths.REPO_ROOT / run["trainer_state"]))
        if run.get("log"):
            row.update(parse_log(paths.REPO_ROOT / run["log"]))
            row["training_source"] = f"job log {run['log']}"
        else:
            row["transcribed"] = run.get("transcribed")
            row["training_source"] = (run.get("transcribed") or {}).get("source",
                                                                        "no log, not transcribed")
        problems += reconcile(row, run.get("expected") or {})
        print(f"{row['model']:<18} layers {row['layers']:>2} tensors {row['lora_tensors']:>3} "
              f"trainable {row['trainable_params']:>11,} {row['file_mib']:>6.1f} MiB vision "
              f"{row['vision_tensors']} complete {row['tensors_per_layer_and_module_complete']} "
              f"r={row['rank']} alpha={row['alpha']} | training: {row['training_source']}")
        rows.append(row)
    logged = [r for r in rows if r.get("corpus_rows")]
    for row in logged:
        print(f"  {row['model']}: {row['corpus_rows']} rows -> {row['train_rows']} train / "
              f"{row['validation_rows']} validation, {row['steps']} steps, "
              f"{row['wall_hours']:.2f} h, loss {row['train_loss']}, best eval {row['best_eval_loss']}")
    print(f"\n{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if args.no_write:
        return 0
    out = ac.write_json("training_summary", {"runs": rows, "differences_from_notes": problems})
    table = ac.write_tex(
        "training_summary", tex_table(rows),
        "The four fine-tuning runs. Adapter figures are measured from the saved adapters, job "
        "times from SLURM accounting (including model loading), evaluation losses from the last "
        "checkpoint's trainer state. Figures marked $\\dagger$ were transcribed from job logs "
        "that no longer exist.",
        "tab:training", notes=["dagger = transcribed in notes 04 on 2026-08-15; the logs of jobs "
                               "2945122, 2960623 and 2960624 were deleted"])
    print(f"wrote {out}\nwrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
