#!/usr/bin/env python3
"""Can the root arity be predicted from the question? The gate before spending 676 runs on it.

Forcing the root box's variable count to GOLD's took result F1 from 0.2162 to 0.3109 -- the
largest effect any intervention in this project has produced (notes 32). That condition is an
ORACLE. This probes the component that would make it a system: one narrow LLM call per case,
grammar-constrained to two integers, asked what the ANSWER binds.

**This is a gate and it is allowed to stop the direction.** It runs ~15 minutes on one GPU and
generates no queries, against ~8 GPU-hours for the rung run it guards. The same pattern rejected
the wrong constraint library (notes 30) and caught a grammar that could not express 83% of gold
(notes 32.2), both before any cluster time was spent.

WHAT COUNTS AS PASSING, and why pooled accuracy will not tell you:

    constant "always (0,1)"               107/169 = 63.3%
    the fine-tuned model, while generating 74/169 = 43.8%

The model is already 20 points WORSE than a predictor that ignores the question. But beating 63.3%
is not sufficient either, because the value is concentrated where the constant is wrong. Splitting
the oracle run by whether gold's arity is the majority class:

    gold IS (0,1)      n=84   free arity 0.3009 -> forced 0.3644
    gold is NOT (0,1)  n=51   free arity 0.0766 -> forced 0.2228

Nearly all the oracle's gain comes from the 51 minority cases. A predictor scoring 70% by being
"the constant plus noise" captures none of it. So this reports per-class recall and the full
confusion matrix, and the decision rests on the minority rows.

One asymmetry to keep in mind when reading the output: forcing a WRONG arity is worse than not
forcing at all. Free decoding leaves the model its own 43.8% chance; a wrongly-forced grammar makes
the correct query unreachable. That is why `predict_root_arity` returns None rather than guessing,
and why abstentions are counted separately below rather than scored as errors.

Usage:
    python src/arity_predictor_probe.py --config configs/eval_config.json --model-idx 1
    python src/arity_predictor_probe.py --limit 20        # a quick shape check
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import paths
from eval_testset import _gold_root_arity, _resolve_ocel_path, _read_json, _write_json
from LLM import (
    build_schema_block,
    predict_root_arity,
    set_constrained_decoding,
    set_generation_temperature,
    set_local_gpu_client,
)
from LogEvents import extract_ocel_schema

# The majority class, and the bar this has to clear. Not a magic number: it is what a predictor
# that ignores the question entirely achieves on this testset.
MAJORITY = (0, 1)


def _expand_model(config, model_idx):
    """The model entry the evaluation would use, so the probe measures the same thing it does.

    Mirrors `eval_testset.run_evaluation` exactly: `expand_models` takes and returns the whole
    CONFIG, not a models list. Getting that wrong cost a cluster job -- the probe died at the first
    line after model load, which is the most expensive place to discover a two-argument mistake.
    `tests/test_arity_predictor.py` now resolves a real config so it cannot recur silently.
    """
    if config.get("auto_expand_finetuned"):
        from expand_models import expand_models
        config = expand_models(config)
    models = config.get("models", [])
    if model_idx is None or model_idx >= len(models):
        labels = [m.get("label") for m in models]
        raise SystemExit(
            f"--model-idx {model_idx} out of range (0-{len(models) - 1}). Available: {labels}")
    return models[model_idx]


def probe(testset_path, config, model_idx, limit):
    """Ask the predictor for every case and compare with gold."""
    testset = _read_json(testset_path)
    if limit:
        testset = testset[:limit]

    model = _expand_model(config, model_idx)
    print(f"model: {model['label']}  ({model['name']})")
    set_local_gpu_client(
        model_name=model["name"],
        device=config.get("local_device", "cuda"),
        dtype=config.get("local_dtype", "auto"),
        quantization=config.get("local_quantization", "4bit"),
    )
    set_generation_temperature(config.get("temperature", 0.0))
    # The whole point: the answer is grammar-constrained to `RootArity`, so it parses by
    # construction and an unparseable reply cannot be confused with a wrong prediction.
    set_constrained_decoding(True)

    schema_cache = {}
    rows = []
    started = time.time()
    for index, entry in enumerate(testset):
        gold = _gold_root_arity(entry.get("ocpq_query"))
        # Anchored on the REPO ROOT, not on the testset file's directory. The testset
        # lives in `data/`, so anchoring there resolves every log to a nonexistent
        # `NLxOCPQ/Logs/` -- which is exactly what this got wrong first time, and what
        # the comment above `anchor_dir` in eval_testset._load_testset warns about.
        ocel_path = _resolve_ocel_path(entry["ocel_path"], paths.REPO_ROOT)
        if ocel_path not in schema_cache:
            schema_cache[ocel_path] = build_schema_block(extract_ocel_schema(ocel_path))
        t0 = time.time()
        predicted = predict_root_arity(
            query=entry["nl_query"],
            ocel_path=ocel_path,
            model_name=model["name"],
            schema_block=schema_cache[ocel_path],
        )
        rows.append({
            "index": index,
            "nl_query": entry["nl_query"],
            "ocel_path": entry["ocel_path"],
            "gold": list(gold[1:]) if gold else None,
            "predicted": list(predicted) if predicted else None,
            "duration_sec": round(time.time() - t0, 3),
        })
        mark = "ok " if (gold and predicted and tuple(gold[1:]) == predicted) else "MISS"
        print(f"  [{index:3d}] {mark} gold={gold[1:] if gold else None} "
              f"pred={predicted}  {entry['nl_query'][:52]}")
    print(f"\n{len(rows)} cases in {time.time() - started:.0f}s")
    return rows


def report(rows):
    """Per-class recall and the confusion matrix. Pooled accuracy is reported but never alone."""
    scored = [r for r in rows if r["gold"] is not None]
    predicted = [r for r in scored if r["predicted"] is not None]
    abstained = len(scored) - len(predicted)

    correct = sum(1 for r in predicted if r["gold"] == r["predicted"])
    const_correct = sum(1 for r in scored if tuple(r["gold"]) == MAJORITY)

    print(f"\n{'=' * 66}\nARITY PREDICTOR GATE\n{'=' * 66}")
    print(f"  cases with readable gold : {len(scored)}")
    print(f"  abstentions (no parse)   : {abstained}   "
          f"(these are NOT forced; safer than a wrong guess)")
    print(f"\n  predictor accuracy       : {correct}/{len(scored)} = {correct / len(scored):.1%}")
    print(f"  constant {str(MAJORITY):8s}        : {const_correct}/{len(scored)} = "
          f"{const_correct / len(scored):.1%}   <- the bar")
    print(f"  the model while generating: 43.8%   (notes 32.4)")

    by_class = defaultdict(lambda: {"n": 0, "hit": 0})
    for r in scored:
        cell = by_class[tuple(r["gold"])]
        cell["n"] += 1
        if r["predicted"] == r["gold"]:
            cell["hit"] += 1
    print(f"\n  PER-CLASS RECALL -- this is what decides it, not the pooled number above.")
    print(f"  The oracle's gain lives in the non-majority rows (0.0766 -> 0.2228, n=51).")
    print(f"    {'gold arity':14s}{'n':>5}{'recall':>9}")
    minority_hits = minority_n = 0
    for arity, cell in sorted(by_class.items(), key=lambda kv: -kv[1]["n"]):
        tag = "  <- majority" if arity == MAJORITY else ""
        if arity != MAJORITY:
            minority_hits += cell["hit"]
            minority_n += cell["n"]
        print(f"    {str(arity):14s}{cell['n']:>5}{cell['hit'] / cell['n']:>9.1%}{tag}")

    print(f"\n  MINORITY-CLASS recall     : {minority_hits}/{minority_n} = "
          f"{minority_hits / minority_n:.1%}" if minority_n else "")
    print(f"  (a constant scores 0% here by construction)")

    confusion = Counter((tuple(r["gold"]), tuple(r["predicted"]) if r["predicted"] else None)
                        for r in scored)
    print(f"\n  CONFUSION (gold -> predicted), most common first:")
    for (gold, pred), n in confusion.most_common(10):
        mark = "  correct" if gold == pred else ""
        print(f"    {str(gold):10s} -> {str(pred):10s}  {n:>4}{mark}")

    # ACCURACY IS THE WRONG STATISTIC, and the first version of this gate used it. A correct
    # prediction on a minority-arity case is worth roughly 2.3x one on a majority case, because
    # that is the per-case gain forcing bought in notes 32.4:
    #     gold IS  (0,1)   free 0.3009 -> forced 0.3644   = +0.064
    #     gold NOT (0,1)   free 0.0766 -> forced 0.2228   = +0.146
    # RE-DERIVED 2026-09-15 on the corrected corpus (notes 35, 36.1), from
    # eval_runs/arity_ablation/*_20260913_*_recomputed, discriminating stratum:
    #     gold IS  (0,1)   n=84   free 0.3358 -> forced 0.3935   = +0.0577
    #     gold NOT (0,1)   n=51   free 0.0701 -> forced 0.2162   = +0.1462
    #
    # AND THE MODEL WAS STILL WRONG, because it priced an ERROR AT ZERO -- as if forcing a wrong
    # arity merely failed to help. Notes 41 measured what it actually does. B2 pinned every root to
    # (0,1); on the 51 discriminating cases whose gold root is something else, ALL 51 scored 0.0000,
    # against 0.0514 under free decoding. A wrongly forced arity does not forgo a gain, it destroys
    # whatever the free model would have scored, because the correct query becomes unreachable:
    #     gold IS  (0,1)   n=84   free 0.3358 -> forced-right 0.3990   = +0.0632   (notes 41.2)
    #     gold NOT (0,1)   n=51   free 0.0514 -> forced-right 0.1976   = +0.1462   (S2, notes 37)
    #     forced WRONG                          -> ~0, i.e. the free score is LOST
    # The loss constants are those free-decoding means. The majority loss is an inference from the
    # measured minority one (no run has forced a wrong NON-(0,1) arity), and it is the reason the
    # verdict is printed twice: once charged, once at zero loss. A predictor that only wins under
    # the zero-loss reading has not passed anything.
    GAIN_MAJORITY, GAIN_MINORITY = 0.0632, 0.1462
    LOSS_MAJORITY, LOSS_MINORITY = 0.3358, 0.0514
    majority_n = by_class.get(MAJORITY, {"n": 0})["n"]
    majority_hits = by_class.get(MAJORITY, {"hit": 0})["hit"]
    # Abstentions are not forced, so they neither gain nor lose. Only PREDICTED cases are charged.
    majority_forced = sum(1 for r in predicted if tuple(r["gold"]) == MAJORITY)
    minority_forced = len(predicted) - majority_forced
    majority_wrong = majority_forced - majority_hits
    minority_wrong = minority_forced - minority_hits

    gains = majority_hits * GAIN_MAJORITY + minority_hits * GAIN_MINORITY
    losses = majority_wrong * LOSS_MAJORITY + minority_wrong * LOSS_MINORITY
    value = gains - losses
    # The constant forces (0,1) on EVERY case: right on every majority case, wrong on every
    # minority one. It abstains never, so it is charged on all of them.
    const_value = majority_n * GAIN_MAJORITY - minority_n * LOSS_MINORITY

    print(f"\n  EXPECTED VALUE in summed F1 (per-class gains AND losses, notes 41.2; not accuracy):")
    print(f"    predictor : {majority_hits} x {GAIN_MAJORITY} + {minority_hits} x {GAIN_MINORITY}"
          f" - {majority_wrong} x {LOSS_MAJORITY} - {minority_wrong} x {LOSS_MINORITY}"
          f" = {value:+.2f}")
    print(f"    constant  : {majority_n} x {GAIN_MAJORITY} - {minority_n} x {LOSS_MINORITY}"
          f" = {const_value:+.2f}")
    print(f"    SENSITIVITY, wrong forces charged NOTHING (the pre-2026-09-20 model):"
          f" predictor {gains:+.2f} vs constant {majority_n * GAIN_MAJORITY:+.2f}")

    verdict_value = value > const_value
    verdict_minority = bool(minority_n and minority_hits > 0)
    print(f"\n{'=' * 66}")
    print(f"  worth more than the constant : {'YES' if verdict_value else 'NO'}   "
          f"(expected value -- the statistic that matters)")
    print(f"  finds minority arities       : {'YES' if verdict_minority else 'NO'}")
    print(f"  pooled accuracy vs constant  : {correct / len(scored):.1%} vs "
          f"{const_correct / len(scored):.1%}   (reported, NOT the criterion)")
    if verdict_value and verdict_minority:
        print("  GATE: PASS -> the rung run is worth its GPU time.")
    else:
        print("  GATE: FAIL -> this predictor is not worth more than ignoring the question.")
        print("        Fix the prompt and re-probe (10 min); do NOT spend 676 runs on it.")
    print("=" * 66)
    return {
        "n": len(scored), "correct": correct, "abstained": abstained,
        "accuracy": correct / len(scored) if scored else None,
        "constant_accuracy": const_correct / len(scored) if scored else None,
        "minority_recall": (minority_hits / minority_n) if minority_n else None,
        "per_class": {str(k): v for k, v in by_class.items()},
        "expected_value": value,
        "constant_expected_value": const_value,
        "expected_value_zero_loss": gains,
        "constant_expected_value_zero_loss": majority_n * GAIN_MAJORITY,
        "forced_wrong": {"majority": majority_wrong, "minority": minority_wrong},
        "gate_passed": bool(verdict_value and verdict_minority),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/eval_config.json")
    parser.add_argument("--testset", default="data/testset.json")
    parser.add_argument("--model-idx", type=int, default=1,
                        help="Index into the expanded models list; 1 is llama31-8b-finetuned.")
    parser.add_argument("--limit", type=int, default=0, help="Probe only the first N cases.")
    parser.add_argument("--out", default=None,
                        help="Where to write the per-case results "
                             "(default: eval_runs/arity_predictor_probe.json).")
    parser.add_argument("--rescore", default=None, metavar="PROBE_JSON",
                        help="Re-run the GATE over an existing probe's stored rows and exit. No "
                             "model, no GPU: the predictions are already in the file, so a change "
                             "to how a prediction is PRICED (notes 41.2) can be applied to a probe "
                             "that already ran. Writes nothing unless --out is given.")
    args = parser.parse_args()

    if args.rescore:
        stored = _read_json(args.rescore)
        rows = stored["rows"] if isinstance(stored, dict) else stored
        print(f"rescoring {len(rows)} stored prediction(s) from {args.rescore}")
        summary = report(rows)
        if args.out:
            _write_json(Path(args.out), {"summary": summary, "rows": rows,
                                         "rescored_from": str(args.rescore)})
            print(f"\nwrote {args.out}")
        return

    config = _read_json(args.config)
    rows = probe(args.testset, config, args.model_idx, args.limit)
    summary = report(rows)
    out = Path(args.out) if args.out else paths.EVAL_RUNS / "arity_predictor_probe.json"
    _write_json(out, {"summary": summary, "rows": rows})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
