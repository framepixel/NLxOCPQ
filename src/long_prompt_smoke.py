#!/usr/bin/env python3
"""Can a model of a config take a LONG prompt on this GPU? Peak memory per prompt length.

Gemma 4 ran out of memory at ~30k tokens in evaluation 1's gate: the attention step asked for 56.5 GiB
on a 93 GiB H100 (notes 43.19). Evaluation 2 makes every prompt longer -- a reasoning step before each
action -- so each model is checked here BEFORE its gate: loaded exactly as a run loads it (its config
entry, NF4, its `sdpa_kernels`), then given prompts of increasing length and asked for a few tokens.
It prints, per length, OK or OOM, the peak GPU memory and the seconds taken. A diagnostic: no metric,
no testset, no gold.

On a GPU node, in the model's own runtime (slurm/submit_long_prompt_smoke.sh does all of this):
    python src/long_prompt_smoke.py configs/eval_config.unleashed_gate.json 3
    python src/long_prompt_smoke.py configs/eval_config.unleashed_gate.json 3 --lengths 16384 65536
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import LLM  # noqa: E402
from eval_testset import _configure_backend, _resolve_model_config  # noqa: E402

FILLER = ("Which orders contain an item heavier than one kilogram, and were they delivered "
          "within two days of payment? ")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("model_idx", type=int)
    parser.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 49152, 65536])
    parser.add_argument("--new-tokens", type=int, default=8)
    args = parser.parse_args()

    import torch

    config = json.load(open(args.config, encoding="utf-8"))
    global_cfg = {"backend": config.get("backend", "local-gpu"),
                  "local_device": config.get("local_device", "cuda"),
                  "local_dtype": config.get("local_dtype", "auto"),
                  "local_quantization": config.get("local_quantization", "4bit")}
    entry = _resolve_model_config(config["models"][args.model_idx], global_cfg)
    _configure_backend(entry, global_cfg)
    print(f"model {entry['label']}  sdpa_kernels {entry.get('sdpa_kernels')}")
    print(f"loaded in {LLM.preload_local_model():.0f}s")
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"GPU {torch.cuda.get_device_name(0)}, {total:.1f} GiB; weights resident "
          f"{torch.cuda.memory_allocated() / 1024 ** 3:.1f} GiB")

    filler_ids = LLM.local_tokenizer.encode(FILLER, add_special_tokens=False)
    worst_ok = None
    for length in sorted(args.lengths):
        ids = (filler_ids * (length // len(filler_ids) + 1))[:length]
        input_ids = torch.tensor([ids], device=LLM.local_model.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        try:
            with torch.no_grad(), LLM._sdpa_context():
                LLM.local_model.generate(input_ids=input_ids,
                                         attention_mask=torch.ones_like(input_ids),
                                         max_new_tokens=args.new_tokens, do_sample=False,
                                         pad_token_id=LLM.local_tokenizer.pad_token_id
                                         or LLM.local_tokenizer.eos_token_id)
            status = "OK "
            worst_ok = length
        except torch.OutOfMemoryError as exc:
            status = f"OOM ({str(exc).splitlines()[0][:90]})"
        except RuntimeError as exc:  # e.g. no allowed SDPA kernel supports this shape
            status = f"ERROR ({str(exc).splitlines()[0][:90]})"
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"  {length:>6} tokens: {status}  peak {peak:5.1f} GiB  {time.time() - started:6.1f}s",
              flush=True)
        if not status.startswith("OK"):
            break
    print(f"RESULT: longest prompt that fits: {worst_ok or 'none'} tokens")
    return 0 if worst_ok else 1


if __name__ == "__main__":
    sys.exit(main())
