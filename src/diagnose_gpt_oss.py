#!/usr/bin/env python3
"""Controlled diagnostic for the GPT-OSS-20B degenerate-repetition issue.

A previous attempt to evaluate openai/gpt-oss-20b produced output that repeated
the same string over and over. The supervisor suggested framing this as a
"base vs instruct" finding — but gpt-oss-20b is in fact an instruction-tuned
reasoning model (it expects the 'harmony' chat format), and degenerate
repetition is also the classic failure mode of greedy decoding (temperature 0,
no repetition penalty), which the evaluation config mandates.

This script ISOLATES the cause by running a fixed set of prompts under one
factor changed at a time from the evaluation baseline, then reports a repetition
score per condition. Read the result honestly: whichever single change kills the
repetition is the actual cause.

  baseline      : chat template applied, temperature 0.0, no repetition penalty
  +sampling     : baseline but temperature 0.7
  +rep_penalty  : baseline but repetition_penalty 1.3
  raw_prompt    : baseline but NO chat template (harmony format off)

This is a STANDALONE diagnostic — it is deliberately separate from eval_testset.py
so the 6-model comparability of the main evaluation matrix is never touched.

Usage (on the cluster, inside the inference venv):
    python diagnose_gpt_oss.py [--model openai/gpt-oss-20b]
                               [--max-new-tokens 512] [--n-prompts 3]
                               [--out gpt_oss_diagnostic.json]
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import paths


# Mirrors the 4-bit config used by LLM.py for the local-GPU backend, so the
# diagnostic loads the model the same way the evaluation would.
def _quantization_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _default_prompts(n: int) -> list[str]:
    """A few representative NL->OCPQ prompts drawn from the testset if available."""
    testset_path = paths.DATA / "testset.json"
    if testset_path.exists():
        data = json.loads(testset_path.read_text(encoding="utf-8"))
        prompts = [e["nl_query"] for e in data if e.get("nl_query")][:n]
        if prompts:
            return prompts
    return [
        "List every person who applied for a viewing appointment more than three times.",
        "Find purchase orders that were paid before the goods were received.",
        "Which patients were registered but never discharged?",
    ][:n]


SYSTEM_PROMPT = (
    "You convert natural-language questions about an object-centric event log into "
    "an OCPQ query in JSON. Return only the JSON."
)


def repetition_score(text: str, n: int = 3) -> dict:
    """Quantify degenerate repetition.

    Returns:
      ngram_repeat: 1 - (#distinct n-grams / #n-grams); higher = more repetitive.
      max_run: longest run of one token repeated back-to-back.
      length: token (whitespace) count.
    """
    toks = text.split()
    if len(toks) < n + 1:
        return {"ngram_repeat": 0.0, "max_run": 0, "length": len(toks)}
    ngrams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    distinct = len(set(ngrams))
    ngram_repeat = round(1 - distinct / len(ngrams), 4)
    # longest back-to-back identical-token run
    max_run = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    return {"ngram_repeat": ngram_repeat, "max_run": max_run, "length": len(toks)}


def _generate(model, tokenizer, prompt: str, *, use_template: bool,
              temperature: float, repetition_penalty: float, max_new_tokens: int) -> str:
    if use_template:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    else:
        # Raw prompt, no chat template / harmony formatting.
        input_ids = tokenizer(f"{SYSTEM_PROMPT}\n\n{prompt}\n", return_tensors="pt").input_ids.to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    if temperature and temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
    else:
        gen_kwargs.update(do_sample=False)  # greedy

    with torch.no_grad():
        out = model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)


# (label, use_template, temperature, repetition_penalty) — one factor off baseline each.
CONDITIONS = [
    ("baseline",     True,  0.0, 1.0),
    ("+sampling",    True,  0.7, 1.0),
    ("+rep_penalty", True,  0.0, 1.3),
    ("raw_prompt",   False, 0.0, 1.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-oss-20b")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-prompts", type=int, default=3)
    parser.add_argument("--out", default="gpt_oss_diagnostic.json")
    args = parser.parse_args()

    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(3407)

    print(f"Loading {args.model} (4-bit nf4)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16, quantization_config=_quantization_config(),
    )
    model.eval()

    prompts = _default_prompts(args.n_prompts)
    print(f"Running {len(CONDITIONS)} conditions x {len(prompts)} prompts.\n")

    results = []
    for label, use_template, temp, rep in CONDITIONS:
        for pi, prompt in enumerate(prompts):
            text = _generate(
                model, tokenizer, prompt,
                use_template=use_template, temperature=temp,
                repetition_penalty=rep, max_new_tokens=args.max_new_tokens,
            )
            score = repetition_score(text)
            results.append({
                "condition": label, "prompt_idx": pi,
                "use_template": use_template, "temperature": temp,
                "repetition_penalty": rep,
                **score, "output_preview": text[:300],
            })
            print(f"[{label:<12}] prompt {pi}: ngram_repeat={score['ngram_repeat']:.2f} "
                  f"max_run={score['max_run']} len={score['length']}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"model": args.model, "results": results}, indent=2),
                        encoding="utf-8")
    print(f"\nWrote {out_path}")

    # Per-condition mean repetition, to read the cause off directly.
    print("\n--- mean ngram_repeat by condition (higher = more degenerate) ---")
    for label, *_ in CONDITIONS:
        vals = [r["ngram_repeat"] for r in results if r["condition"] == label]
        mean = round(sum(vals) / len(vals), 3) if vals else float("nan")
        print(f"  {label:<12} {mean}")
    print("\nReading: if 'baseline' is high but a single-factor change drops it, that change\n"
          "is the cause. A drop under '+sampling'/'+rep_penalty' implies greedy-decoding\n"
          "degeneration; a drop under 'raw_prompt' implies a chat-template/harmony issue.\n"
          "None of these would support a 'base vs instruct' explanation.")


if __name__ == "__main__":
    main()
