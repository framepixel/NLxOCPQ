#!/usr/bin/env python3
"""Cut a small, group-safe slice of the fine-tuning corpus for a model spike.

A spike answers "does this base model load, tokenize and fine-tune at all" — it is not an
experiment, so it needs minutes of GPU time rather than hours. Slicing must respect the same
grouping `train_unsloth.py` uses: `convert_dataset_for_finetuning.py` emits several rows per
source example sharing one target, so a naive head-N would put variants of the same target on
both sides of the train/validation split and report a memorised loss.

Usage:
    python make_spike_slice.py [--rows 200] [--out finetune_ocpq.spike.jsonl]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/finetune_ocpq.jsonl")
    parser.add_argument("--out", default="data/finetune_ocpq.spike.jsonl")
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3407, help="Canonical seed (CLAUDE.md 2.1).")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")

    # Group by assistant target, exactly as the real split does.
    by_target = defaultdict(list)
    with open(source, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_target[row["messages"][-1]["content"]].append(line)

    groups = sorted(by_target)
    random.Random(args.seed).shuffle(groups)

    kept, used_groups = [], 0
    for target in groups:
        if len(kept) >= args.rows:
            break
        kept.extend(by_target[target])
        used_groups += 1

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as f:
        for line in kept:
            f.write(line + "\n")

    print(f"wrote {out}: {len(kept)} rows from {used_groups} target groups "
          f"(source had {sum(len(v) for v in by_target.values())} rows)")
    print(f"point the spike at it with OCPQ_FINETUNE_DATASET={out}")


if __name__ == "__main__":
    main()
