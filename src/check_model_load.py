#!/usr/bin/env python3
"""Load one model (and adapter) and generate once. Minutes, not the 3 hours a smoke run costs.

The 2026-08-15 smoke test spent ~1 GPU-hour discovering that Ministral 3 14B could not be loaded
at all. Verifying a loader change by re-running the whole smoke test is the wrong order: this
exercises exactly the failing path -- config resolution, adapter injection, tokenizer type,
context detection, one real generation -- for a single model.

    python check_model_load.py                       # every model in eval_config.json
    python check_model_load.py --model-idx 2 3       # just Ministral base + fine-tuned
    python check_model_load.py --config eval_config.smoke.json

Exits non-zero if any requested model fails.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval_config.json")
    ap.add_argument("--model-idx", type=int, nargs="*", default=None,
                    help="0-based indices into the expanded model list; default all")
    ap.add_argument("--max-tokens", type=int, default=64,
                    help="Generation is only to prove the forward pass works; keep it short")
    args = ap.parse_args()

    from expand_models import expand_models
    import LLM

    cfg = json.loads((HERE / args.config).read_text(encoding="utf-8"))
    models = expand_models(cfg)["models"]
    wanted = args.model_idx if args.model_idx else list(range(len(models)))

    failures = []
    print(f"Checking {len(wanted)} of {len(models)} model(s) from {args.config}\n")

    for idx in wanted:
        if idx < 0 or idx >= len(models):
            print(f"  [{idx}] out of range (0..{len(models) - 1})")
            failures.append(idx)
            continue
        entry = models[idx]
        label, name = entry["label"], entry["name"]
        print(f"[{idx}] {label}\n     {name}")
        started = time.time()
        try:
            LLM.set_local_gpu_client(
                model_name=name,
                device=cfg.get("local_device", "cuda"),
                dtype=cfg.get("local_dtype", "auto"),
                quantization=cfg.get("local_quantization", "4bit"),
            )
            LLM._ensure_local_gpu_model_loaded()

            tok = LLM.local_tokenizer
            model = LLM.local_model
            print(f"     model class     : {type(model).__name__}")
            print(f"     tokenizer class : {type(tok).__name__}")
            # A processor here would break every downstream call: its first positional argument
            # is `images`, not `text`.
            if not hasattr(tok, "apply_chat_template"):
                raise RuntimeError(f"{type(tok).__name__} has no apply_chat_template "
                                   "-- a processor leaked through instead of a tokenizer")
            print(f"     context limit   : {LLM._model_context_limit()}")
            # Recorded because it drives generation speed and is resolved implicitly: nothing in
            # this project sets `attn_implementation`, so transformers picks it (None -> sdpa,
            # falling back to eager when unsupported). Gemma 3 is 5-13x slower per run than
            # Ministral 14B and it is worth knowing from the artifact, not from assumption,
            # which kernel each model actually ended up on.
            # NOT named `cfg`: that is the JSON config dict driving this loop, and shadowing it
            # here made every model after the first fail with
            # "'LlamaConfig' object has no attribute 'get'".
            model_cfg = getattr(model, "config", None)
            attn = getattr(model_cfg, "_attn_implementation", None)
            text_cfg = getattr(model_cfg, "text_config", None)
            print(f"     attention       : {attn}"
                  + (f" (text_config: {getattr(text_cfg, '_attn_implementation', None)})"
                     if text_cfg is not None else ""))
            # Gemma 2 defaulted attn_logit_softcapping to 50.0, which sdpa cannot express; Gemma 3
            # defaults it to None. Where it IS set, a non-eager kernel silently drops the tanh and
            # you are running a different model, not merely different arithmetic.
            for attr in ("attn_logit_softcapping", "final_logit_softcapping"):
                val = getattr(model_cfg, attr, None)
                if val is None and text_cfg is not None:
                    val = getattr(text_cfg, attr, None)
                if val is not None:
                    print(f"     {attr}: {val}  <- kernel choice is NOT numerically neutral here")
            print(f"     vocab size      : "
                  f"{getattr(model_cfg, 'vocab_size', None) or getattr(text_cfg, 'vocab_size', None)}")

            resp = LLM._call_local_gpu_chat_completion(
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=args.max_tokens,
                response_format=None,
            )
            text = resp.choices[0].message.content if getattr(resp, "choices", None) else str(resp)
            print(f"     generated       : {text.strip()[:60]!r}")
            print(f"     OK in {time.time() - started:.1f}s\n")
        except Exception:
            print(f"     FAILED after {time.time() - started:.1f}s")
            traceback.print_exc()
            print()
            failures.append(idx)

    if failures:
        print(f"RESULT: FAIL -> model indices {failures}")
        return 1
    print(f"RESULT: PASS -> {len(wanted)} model(s) load and generate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
