#!/usr/bin/env python3
"""Materialise Gemma 3's two evaluation arms as merged 16-bit checkpoints.

WHY THIS EXISTS
---------------
`unsloth` substitutes its own pre-quantized redistribution when `load_in_4bit=True`, so Gemma's
LoRA was trained against `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` while evaluation loaded
`google/gemma-3-4b-it` and re-quantized it with bitsandbytes. Gemma is the only arm with that
mismatch -- Llama and Ministral request the unsloth artifact directly, so nothing is displaced --
and it is the arm that scored below its own base.

Re-fine-tuning does not fix this. The LoRA is already trained against the right weights. What
cannot be done is *evaluate* against them: loading that artifact through plain
transformers + bitsandbytes raises "FP4 quantization state not initialized" and then asserts in
`fix_4bit_weight_quant_state_from_module` (job 3015021, 2026-08-16 -- both Gemma arms failed to
emit a single token). The artifact is loadable only by unsloth's own loader, which works here in
the training environment.

So the repair is to merge *here*, where the weights are known good, and hand evaluation a plain
checkpoint. `LLM.py` already detects a merged directory (`model.safetensors`) and loads it with no
adapter injection at all, so no pipeline source file has to change for this.

BOTH ARMS ARE PRODUCED THE SAME WAY
-----------------------------------
The base arm is dequantized through the identical code path, using a freshly initialised LoRA
whose B matrices are zero -- so the merge adds exactly nothing while still traversing the same
dequantize-and-merge machinery. Using `google/gemma-3-4b-it` for the base instead would leave the
two arms differing by *both* the LoRA and the quantization lineage, which confounds the one
quantity the comparison is about. Cost: one extra checkpoint on disk.

RESIDUAL LIMITATION TO DISCLOSE
-------------------------------
Merged weights are `dequant(nf4_unsloth(W)) + BA`, and evaluation re-quantizes to nf4. Gemma
therefore carries a quantization round-trip the other three families do not. Much smaller than
the original mismatch, but not zero.

USAGE (in the training environment, e.g. nlxocpq-train-v5):
    python merge_gemma.py            # write both arms
    python merge_gemma.py --verify   # reload both and prove the merge changed the weights
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The checkpoint the LoRA was actually trained against. Recorded in
# adapters/gemma3-4b-fine-tuned-quantized/ocpq_provenance.json as `trained_against`.
TRAINING_BASE = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
ADAPTER_DIR = HERE / "adapters" / "gemma3-4b-fine-tuned-quantized"
FT_OUT = HERE / "adapters" / "gemma3-4b-merged-16bit"
BASE_OUT = HERE / "adapters" / "gemma3-4b-base-16bit"
MAX_SEQ_LEN = 4096

PROMPT = "Return a JSON object with a single key \"ok\" set to true."


def _load_base():
    """Load the training artifact through unsloth's loader -- the only one that can read it."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=TRAINING_BASE,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )
    return model, tokenizer


def _adapter_lora_kwargs():
    """LoRA hyperparameters copied from the trained adapter.

    The zero-adapter used for the base arm does not need to match numerically -- B is zero either
    way -- but matching keeps the two arms on the same code path rather than merely a similar one.
    """
    cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text(encoding="utf-8"))
    kwargs = {
        "r": cfg.get("r", 16),
        "lora_alpha": cfg.get("lora_alpha", 16),
        "lora_dropout": 0.0,
        "bias": "none",
        "use_gradient_checkpointing": False,
    }
    target = cfg.get("target_modules")
    if target:
        kwargs["target_modules"] = target
    return kwargs


def _assert_dir_is_loadable(path: Path, label: str):
    """`LLM.py` recognises a merged directory only by these filenames -- check before trusting it.

    A directory that saved as an adapter instead would be silently re-routed into the LoRA
    injection path at evaluation time, which is exactly the failure this script exists to remove.
    """
    weights = [n for n in ("model.safetensors", "pytorch_model.bin")
               if (path / n).exists()]
    sharded = list(path.glob("model-*-of-*.safetensors"))
    if not weights and not sharded:
        raise SystemExit(
            f"FATAL: {label} at {path} has no model.safetensors / pytorch_model.bin.\n"
            f"       LLM.py would not recognise it as a merged model. Contents: "
            f"{sorted(p.name for p in path.iterdir())[:12]}"
        )
    if (path / "adapter_config.json").exists():
        raise SystemExit(
            f"FATAL: {label} at {path} still contains adapter_config.json.\n"
            f"       LLM.py checks for the adapter FIRST, so this would be loaded as a LoRA "
            f"directory and the merge would be silently bypassed."
        )
    print(f"  [ok] {label} is a merged checkpoint "
          f"({', '.join(weights) or f'{len(sharded)} shards'})")


def build_finetuned():
    print(f"\n=== fine-tuned arm -> {FT_OUT} ===")
    from peft import PeftModel

    model, tokenizer = _load_base()
    # Load the base explicitly rather than passing ADAPTER_DIR to from_pretrained:
    # `adapter_config.json` was redirected to google/gemma-3-4b-it by
    # train_unsloth._restore_requested_base_model_name, so unsloth would resolve the wrong base.
    model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))

    n_lora = sum(1 for n, _ in model.named_parameters() if "lora_" in n)
    print(f"  adapter attached: {n_lora} LoRA tensors live on the model")
    if n_lora == 0:
        raise SystemExit("FATAL: no LoRA tensors attached; the merge would be a no-op.")

    model.save_pretrained_merged(str(FT_OUT), tokenizer, save_method="merged_16bit")
    _assert_dir_is_loadable(FT_OUT, "fine-tuned arm")


def build_base():
    print(f"\n=== base arm -> {BASE_OUT} ===")
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = _load_base()
    # A freshly initialised LoRA has B = 0, so merging adds exactly zero to every weight. The
    # point is not the adapter, it is that the base arm traverses the same dequantize+merge path
    # as the fine-tuned arm, leaving the LoRA as the only difference between the two checkpoints.
    model = FastLanguageModel.get_peft_model(model, **_adapter_lora_kwargs())

    b_tensors = [p for n, p in model.named_parameters() if "lora_B" in n]
    if not b_tensors:
        raise SystemExit("FATAL: zero-adapter attached no lora_B tensors; cannot verify no-op.")
    nonzero = [float(p.abs().max()) for p in b_tensors]
    if max(nonzero) != 0.0:
        raise SystemExit(
            f"FATAL: lora_B is not zero (max |B| = {max(nonzero)}). The base arm would carry a "
            f"random perturbation instead of being a pure dequantization."
        )
    print(f"  zero-adapter verified: {len(b_tensors)} lora_B tensors, all exactly 0")

    model.save_pretrained_merged(str(BASE_OUT), tokenizer, save_method="merged_16bit")
    _assert_dir_is_loadable(BASE_OUT, "base arm")


def verify():
    """Prove the two checkpoints are actually different models.

    Merged checkpoints emit no `[adapter load] injected N/M` line, so the diagnostic that would
    otherwise catch a silently no-op merge is gone. This replaces it: identical logits mean the
    LoRA never made it in, and the fine-tuned arm is really the base arm wearing its name.
    """
    print("\n=== verify ===")
    import torch
    from transformers import AutoTokenizer

    def _logits(path):
        sys.path.insert(0, str(HERE))
        from LLM import _load_text_generation_model  # the eval-time loader, not unsloth's

        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, trust_remote_code=True)
        mdl = _load_text_generation_model(str(path), device_map="auto",
                                          trust_remote_code=True)
        ids = tok(PROMPT, return_tensors="pt").input_ids.to(mdl.device)
        with torch.no_grad():
            out = mdl(ids).logits[0, -1].float().cpu()
        del mdl
        torch.cuda.empty_cache()
        return out

    a = _logits(BASE_OUT)
    b = _logits(FT_OUT)
    delta = float((a - b).abs().max())
    print(f"  max |logit difference| on a fixed prompt: {delta:.6f}")
    if delta == 0.0:
        raise SystemExit(
            "FATAL: the two checkpoints produce identical logits. The LoRA was not merged in, "
            "and the 'fine-tuned' arm is the base model under another name."
        )
    print("  [ok] the fine-tuned arm differs from the base arm")


def repair_provenance():
    """`adapter_config.json.orig` is byte-identical to `adapter_config.json`.

    The backup was taken after `_restore_requested_base_model_name` had already rewritten the
    field, so it preserves nothing: it claims to record the displaced base name and does not.
    Leaving a file that looks like evidence and is not is worse than having no backup at all.
    """
    orig = ADAPTER_DIR / "adapter_config.json.orig"
    live = ADAPTER_DIR / "adapter_config.json"
    if not orig.exists():
        return
    if orig.read_bytes() != live.read_bytes():
        print(f"\n[provenance] {orig.name} already differs from the live config; left alone.")
        return

    prov = json.loads((ADAPTER_DIR / "ocpq_provenance.json").read_text(encoding="utf-8"))
    trained_against = prov.get("trained_against")
    if not trained_against:
        print(f"\n[provenance] cannot repair {orig.name}: provenance has no `trained_against`.")
        return

    restored = json.loads(live.read_text(encoding="utf-8"))
    restored["base_model_name_or_path"] = trained_against
    restored["_ocpq_note"] = (
        "Reconstructed on 2026-08-19 from ocpq_provenance.json:trained_against. The original "
        ".orig backup was taken after _restore_requested_base_model_name had already run and so "
        "was byte-identical to adapter_config.json, preserving nothing. This file records the "
        "base the LoRA was actually TRAINED against; adapter_config.json records the redirect."
    )
    orig.write_text(json.dumps(restored, indent=2) + "\n", encoding="utf-8")
    print(f"\n[provenance] rewrote {orig.name}: base_model_name_or_path -> {trained_against}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="Only reload the two finished checkpoints and compare their logits.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing output directories.")
    args = ap.parse_args()

    if args.verify:
        verify()
        return 0

    for path in (FT_OUT, BASE_OUT):
        if path.exists():
            if not args.force:
                raise SystemExit(
                    f"FATAL: {path} already exists. Pass --force to overwrite, or move it aside.\n"
                    f"       Refusing silently to overwrite a checkpoint an evaluation may cite."
                )
            print(f"[force] removing existing {path}")
            shutil.rmtree(path)

    if not ADAPTER_DIR.is_dir():
        raise SystemExit(f"FATAL: adapter not found at {ADAPTER_DIR}")

    build_finetuned()
    build_base()
    repair_provenance()

    print("\nDone. Next:")
    print(f"  1. python {Path(__file__).name} --verify")
    print(f"  2. point eval_config.json at {BASE_OUT.name} / {FT_OUT.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
