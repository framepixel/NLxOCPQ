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
# Slice of the next-token logit vector stored beside each checkpoint as a reference. Gemma 3's
# vocabulary is ~262k, so the whole vector is not worth persisting; a fixed prefix plus the argmax
# is enough to tell a faithful reload from a mis-mapped one.
REF_LOGIT_SLICE = 512
REF_FILE = "ocpq_reference_logits.json"


def _next_token_logits(model, tokenizer):
    """Logits for the token after PROMPT, from whatever model object is passed."""
    import torch

    ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids).logits[0, -1].float().cpu()
    return out


def _write_reference(out: Path, logits):
    """Capture the in-memory model's behaviour BEFORE saving, so the reload can be checked.

    Without this there is nothing to compare a reloaded checkpoint against: the base-vs-finetuned
    difference proves the LoRA was merged, but not that the weights survived serialisation with
    their names intact.
    """
    top = logits.topk(10)
    (out / REF_FILE).write_text(json.dumps({
        "prompt": PROMPT,
        "argmax": int(logits.argmax()),
        "top10_ids": [int(i) for i in top.indices],
        "logit_prefix": [round(float(v), 5) for v in logits[:REF_LOGIT_SLICE]],
    }, indent=2) + chr(10), encoding="utf-8")


def _save_merged(model, tokenizer, out: Path, label: str) -> str:
    """Write a merged 16-bit checkpoint, around a transformers 5.5.0 / Gemma 3 save bug.

    `save_pretrained` reverses the checkpoint->runtime weight conversion applied at load time.
    On transformers 5.5.0 at least one Gemma 3 conversion has no reverse implemented, so the call
    dies with a bare `NotImplementedError` out of `core_model_loading.reverse_op` -- *after* the
    merge itself has succeeded. Losing a completed merge to a serialisation detail is not
    acceptable, so the reversal is disabled and the result is checked instead of trusted.

    Disabling it writes the state dict in RUNTIME key order rather than the original checkpoint's.
    Whether reloading maps that back to the same weights is exactly the thing not to assume, which
    is why reference logits are captured here and `--verify` compares against them.
    """
    try:
        model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
        return "save_pretrained"
    except NotImplementedError:
        import transformers.modeling_utils as mu

        original = getattr(mu, "revert_weight_conversion", None)
        if original is None:
            raise SystemExit(
                "FATAL: save_pretrained raised NotImplementedError but "
                "transformers.modeling_utils.revert_weight_conversion is absent, so the known "
                "workaround does not apply. This is a different failure -- do not work around it."
            )
        print(f"  [warn] {label}: transformers could not reverse Gemma 3's weight conversion "
              f"(known 5.5.0 issue). Retrying with the reversal disabled; the result is "
              f"verified against reference logits rather than trusted.")
        if out.exists():
            shutil.rmtree(out)
        mu.revert_weight_conversion = lambda model_to_save, state_dict, *a, **k: state_dict
        try:
            model.save_pretrained_merged(str(out), tokenizer, save_method="merged_16bit")
        finally:
            mu.revert_weight_conversion = original
        return "save_pretrained+no_revert"


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
    """Check the checkpoint against the ACTUAL loader, not against a copy of its rules.

    An earlier version of this function kept its own list of acceptable filenames. It drifted:
    it accepted sharded safetensors, `LLM.py` did not, and all 48 Gemma smoke conditions failed
    with "not recognized as a merged model or a PEFT adapter" -- after these checkpoints had
    passed the tokenizer check, the reload check and the separation check. Two copies of one rule
    is one copy too many, so the predicate now lives in `LLM.py` and is imported from there.
    """
    sys.path.insert(0, str(HERE))
    from LLM import is_merged_model_dir

    if not is_merged_model_dir(str(path)):
        raise SystemExit(
            f"FATAL: {label} at {path} is not loadable by LLM.py as a merged model.\n"
            f"       Contents: {sorted(p.name for p in path.iterdir())[:12]}\n"
            f"       Evaluation would fail on every case for this arm."
        )
    if (path / "adapter_config.json").exists():
        raise SystemExit(
            f"FATAL: {label} at {path} still contains adapter_config.json.\n"
            f"       LLM.py checks for the adapter FIRST, so this would be loaded as a LoRA "
            f"directory and the merge would be silently bypassed."
        )
    _assert_weights_are_merged(path, label)
    files = sorted(p.name for p in path.glob("*.safetensors"))
    print(f"  [ok] {label} is loadable by LLM.py "
          f"({len(files)} safetensors file(s): {', '.join(files[:3])})")


def _assert_weights_are_merged(path: Path, label: str):
    """Reject a checkpoint whose tensor NAMES say it is still a PEFT wrapper.

    The filename check above is not sufficient, and this is not hypothetical: on 2026-08-19 an
    arm built with `PeftModel.from_pretrained` fell through unsloth's save dispatch to a plain
    `save_pretrained` on the wrapper. It wrote a perfectly well-formed `model.safetensors`
    containing `lora_A.default.weight` and `base_layer.weight` keys, passed every check there was,
    and only failed two steps later when transformers reported every real weight MISSING and
    randomly initialised them.

    Reading the key names costs milliseconds and catches it at the moment it happens.
    """
    from safetensors import safe_open

    files = sorted(path.glob("*.safetensors"))
    if not files:
        return  # .bin checkpoints are not produced by this script; nothing to inspect.

    offenders = []
    total = 0
    for f in files:
        # A truncated or half-written shard fails here with an opaque rust error. Saying so
        # plainly matters: the alternative is a checkpoint that looks present on disk and only
        # explodes hours later inside an evaluation shard.
        try:
            handle = safe_open(str(f), framework="pt")
        except Exception as exc:
            raise SystemExit(
                f"FATAL: {label}: {f.name} is not readable as safetensors ({exc}). "
                f"The checkpoint is corrupt or incompletely written -- check free disk space "
                f"and re-run the merge with --force."
            ) from exc
        with handle:
            for key in handle.keys():
                total += 1
                if "lora_" in key or ".base_layer." in key:
                    offenders.append(key)

    if offenders:
        sample = "\n         ".join(offenders[:3])
        raise SystemExit(
            f"FATAL: {label} at {path} was NOT merged.\n"
            f"       {len(offenders)} of {total} tensors still carry PEFT wrapper names, e.g.\n"
            f"         {sample}\n"
            f"       This is a raw adapter state dict saved under a merged model's filename.\n"
            f"       Loading it would report every real weight MISSING and randomly initialise\n"
            f"       them. Delete this directory; do not evaluate from it."
        )
    print(f"  [ok] {label}: {total} tensors, none carrying PEFT wrapper names")


def build_finetuned():
    """Attach the trained LoRA the way unsloth recognises, then merge.

    `PeftModel.from_pretrained` was tried first and is WRONG here. It produces a plain PEFT
    object, which unsloth's save dispatch does not recognise as one of its own, so
    `save_pretrained_merged` silently falls through to `unsloth_generic_save` -> plain
    `save_pretrained` on the wrapper. The result looks like a success -- it even prints "Saving
    full fine-tuned model" and writes a `model.safetensors` -- but the state dict contains
    `lora_A.default.weight` / `base_layer.weight` keys instead of merged weights, and reloading it
    reports every real weight MISSING and randomly initialises them.

    Building the adapter with `FastLanguageModel.get_peft_model` and injecting the trained tensors
    keeps the model on unsloth's own path, so the real "Merging weights into 16bit" step runs.
    This mirrors what `LLM.py` already does at evaluation time, where it injects 476/476 with 0
    unexpected -- the same numbers to expect here.
    """
    print(f"\n=== fine-tuned arm -> {FT_OUT} ===")
    from unsloth import FastLanguageModel
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file as st_load

    model, tokenizer = _load_base()
    # Shape the adapter from the trained config rather than loading the adapter directory:
    # `adapter_config.json` was redirected to google/gemma-3-4b-it by
    # train_unsloth._restore_requested_base_model_name, so anything resolving the base from that
    # file would load the wrong weights.
    model = FastLanguageModel.get_peft_model(model, **_adapter_lora_kwargs())

    sf = ADAPTER_DIR / "adapter_model.safetensors"
    binf = ADAPTER_DIR / "adapter_model.bin"
    device = str(next(iter(model.parameters())).device)
    if sf.exists():
        raw = st_load(str(sf), device=device)
    elif binf.exists():
        import torch
        raw = torch.load(str(binf), map_location=device)
    else:
        raise SystemExit(f"FATAL: no adapter weights in {ADAPTER_DIR}")

    # Keep only the LoRA deltas; unsloth embeds quantized base layers alongside them.
    lora_sd = {k: v for k, v in raw.items() if "lora_" in k}
    result = set_peft_model_state_dict(model, lora_sd, adapter_name="default")
    unexpected = list(getattr(result, "unexpected_keys", []) or [])
    loaded = len(lora_sd) - len(unexpected)
    print(f"  adapter injected: {loaded}/{len(lora_sd)} LoRA tensors "
          f"({len(unexpected)} unexpected)")
    if unexpected:
        raise SystemExit(
            f"FATAL: adapter only partially applied: {loaded}/{len(lora_sd)} matched, "
            f"{len(unexpected)} did not, e.g. {', '.join(unexpected[:3])}.\n"
            f"       Merging now would bake a half-fine-tuned model into the checkpoint."
        )
    if loaded == 0:
        raise SystemExit("FATAL: no LoRA tensors matched; the merge would be a no-op.")

    # Non-zero B is what makes this arm differ from the base arm. If the trained B matrices were
    # all zero the adapter would be vacuous, and the two checkpoints would be identical.
    b_max = max((float(p.detach().abs().max()) for n, p in model.named_parameters()
                 if "lora_B" in n), default=0.0)
    print(f"  max |lora_B| = {b_max:.6f} (must be > 0, else the adapter is vacuous)")
    if b_max == 0.0:
        raise SystemExit("FATAL: every lora_B is zero; the trained adapter carries no delta.")

    reference = _next_token_logits(model, tokenizer)
    how = _save_merged(model, tokenizer, FT_OUT, "fine-tuned arm")
    _assert_dir_is_loadable(FT_OUT, "fine-tuned arm")
    _write_reference(FT_OUT, reference)
    print(f"  saved via {how}; reference logits written to {REF_FILE}")


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
    nonzero = [float(p.detach().abs().max()) for p in b_tensors]
    if max(nonzero) != 0.0:
        raise SystemExit(
            f"FATAL: lora_B is not zero (max |B| = {max(nonzero)}). The base arm would carry a "
            f"random perturbation instead of being a pure dequantization."
        )
    print(f"  zero-adapter verified: {len(b_tensors)} lora_B tensors, all exactly 0")

    reference = _next_token_logits(model, tokenizer)
    how = _save_merged(model, tokenizer, BASE_OUT, "base arm")
    _assert_dir_is_loadable(BASE_OUT, "base arm")
    _write_reference(BASE_OUT, reference)
    print(f"  saved via {how}; reference logits written to {REF_FILE}")


def _check_tokenizers():
    """Confirm each merged checkpoint tokenizes identically to the checkpoint it came from.

    transformers 5.5.0 warns on load that these tokenizers carry "an incorrect regex pattern"
    and that this "will lead to incorrect tokenization", citing a Mistral discussion. The warning
    does not appear anywhere in the wave-1 logs, so it arrived with the merged checkpoints --
    unsloth rewrites `tokenizer_config.json` and re-plants the sentencepiece asset when it saves.

    Whether it matters is an empirical question, not a judgement call: if the merged tokenizer
    produces the same ids as the training base for the text this evaluation actually sends, the
    warning is cosmetic. If it does not, every Gemma prompt is subtly wrong and the arm is void.
    Tokenizers are cheap to load, so this is checked rather than argued about.
    """
    from transformers import AutoTokenizer

    # Deliberately weighted towards what the pipeline really sends: JSON punctuation, nested
    # braces, quoted keys, newlines, and OCEL event names with spaces and underscores -- exactly
    # the places a regex difference in the pre-tokenizer would show up.
    samples = [
        PROMPT,
        '{"nodes": [{"id": "0", "type": "Event", "name": "O_Create Offer"}]}',
        '{\n  "filters": [{"type": "O2E", "object": "Offer"}],\n  "constraints": []\n}',
        "W_Shortened completion  / A_Cancelled / item out of stock",
        "Find all orders where payment reminder occurs within 3 days (259200s).",
        "```json\n{\"a\": 1}\n```",
    ]

    ref = AutoTokenizer.from_pretrained(TRAINING_BASE, use_fast=True, trust_remote_code=True)
    mismatches = []
    for label, path in (("base", BASE_OUT), ("fine-tuned", FT_OUT)):
        tok = AutoTokenizer.from_pretrained(str(path), use_fast=True, trust_remote_code=True)
        bad = 0
        for text in samples:
            if tok(text).input_ids != ref(text).input_ids:
                bad += 1
                if bad == 1:
                    mismatches.append((label, text, ref(text).input_ids, tok(text).input_ids))
        print(f"      {label:<11} {'PASS' if bad == 0 else 'FAIL'}  "
              f"{len(samples) - bad}/{len(samples)} samples tokenize identically to "
              f"{TRAINING_BASE}")

    if mismatches:
        label, text, want, got = mismatches[0]
        raise SystemExit(
            f"FATAL: the {label} checkpoint's tokenizer does not match the training base.\n"
            f"       transformers' 'incorrect regex pattern' warning is real for this model.\n"
            f"       sample: {text!r}\n"
            f"       base ids: {want[:16]}\n"
            f"       ckpt ids: {got[:16]}\n"
            f"       Every prompt sent to this arm would be tokenized differently from the text\n"
            f"       the LoRA was trained on. Do NOT evaluate from these checkpoints."
        )


def verify():
    """Two independent checks on the finished checkpoints.

    1. FAITHFUL RELOAD. Because transformers 5.5.0 cannot reverse Gemma 3's weight conversion,
       the checkpoint may have been written with that reversal disabled -- in runtime key order
       rather than the original checkpoint's. Reloading is therefore not something to assume
       works. Each arm is compared against the reference logits captured from the in-memory model
       immediately before it was saved. A name mis-mapping moves logits enormously; the honest
       4-bit-merge vs 16-bit-reload difference is small, hence a tolerance rather than equality.

    2. THE ARMS DIFFER. A merged checkpoint emits no `[adapter load] injected N/M` line, so the
       diagnostic that would otherwise catch a silently no-op merge is gone. Identical logits
       across the two arms mean the LoRA never made it in and the "fine-tuned" arm is the base
       model wearing its name.
    """
    print("\n=== verify ===")
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(HERE))

    def _logits(path):
        """Load exactly the way evaluation does, then take one forward pass.

        Deliberately routed through `set_local_gpu_client` + `_ensure_local_gpu_model_loaded`
        rather than calling `_load_text_generation_model` directly. The direct call skips the
        directory-recognition dispatch -- the merged-model vs LoRA-adapter branch -- which is
        precisely where the 2026-08-19 checkpoints failed. That version of this function loaded
        them happily and reported PASS, while every one of the 48 Gemma smoke conditions died on
        "not recognized as a merged model or a PEFT adapter".

        A verification that enters through a different door than production is not a
        verification of production.
        """
        import LLM

        LLM.set_local_gpu_client(model_name=str(path), device="cuda", dtype="auto",
                                 quantization="4bit")
        LLM._ensure_local_gpu_model_loaded()
        mdl, tok = LLM.local_model, LLM.local_tokenizer
        if mdl is None or tok is None:
            raise SystemExit(f"FATAL: {path} loaded to None through the evaluation path.")

        ids = tok(PROMPT, return_tensors="pt").input_ids.to(mdl.device)
        with torch.no_grad():
            out = mdl(ids).logits[0, -1].float().cpu()

        # Release before the next arm: two 4B models plus activations is avoidable.
        LLM.local_model = None
        LLM.local_tokenizer = None
        LLM.local_model_key = None
        del mdl
        torch.cuda.empty_cache()
        return out


    # One arm at a time; _logits releases each before the next.
    reloaded = {}
    for label, path in (("base", BASE_OUT), ("fine-tuned", FT_OUT)):
        if not path.is_dir():
            raise SystemExit(f"FATAL: {path} does not exist -- run the merge first.")
        reloaded[label] = _logits(path)

    print("\n  [1] faithful reload")
    ok = True
    for label, path in (("base", BASE_OUT), ("fine-tuned", FT_OUT)):
        ref_path = path / REF_FILE
        if not ref_path.is_file():
            print(f"      {label:<11} SKIP  no {REF_FILE} (checkpoint predates this check)")
            continue
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        got = reloaded[label]
        prefix = torch.tensor(ref["logit_prefix"])
        drift = float((got[:len(prefix)] - prefix).abs().max())
        same_argmax = int(got.argmax()) == int(ref["argmax"])
        # 4-bit inference vs a 16-bit reload of the same weights lands well inside a couple of
        # logits. A wrong key mapping is not a near miss -- it is a different model.
        good = drift < 5.0 and same_argmax
        ok = ok and good
        print(f"      {label:<11} {'PASS' if good else 'FAIL'}  "
              f"max|drift|={drift:.4f}  argmax {'matches' if same_argmax else 'DIFFERS'} "
              f"({int(got.argmax())} vs {ref['argmax']})")
    if not ok:
        raise SystemExit(
            "FATAL: a reloaded checkpoint does not reproduce the model it was saved from.\n"
            "       The state dict was written with transformers' weight-conversion reversal\n"
            "       disabled, and reloading has evidently not mapped it back. Do NOT evaluate\n"
            "       from these checkpoints."
        )

    print("\n  [2] the two arms differ")
    delta = float((reloaded["base"] - reloaded["fine-tuned"]).abs().max())
    print(f"      max |logit difference| between arms: {delta:.6f}")
    if delta == 0.0:
        raise SystemExit(
            "FATAL: the two checkpoints produce identical logits. The LoRA was not merged in, "
            "and the 'fine-tuned' arm is the base model under another name."
        )
    print("      PASS  the fine-tuned arm is a different model from the base arm")
    print("\n[ok] both checkpoints are faithful and distinct.")


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
