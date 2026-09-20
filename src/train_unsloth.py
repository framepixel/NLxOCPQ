import atexit
import json
import os

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TQDM_MONITOR_INTERVAL", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
for _env_name in (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_MPS_PIPE_DIRECTORY",
    "CUDA_MPS_LOG_DIRECTORY",
    "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
):
    os.environ.pop(_env_name, None)

import re
import signal
import sys
import time
from pathlib import Path

import paths


model = None
tokenizer = None


def _import_training_stack():
    try:
        from unsloth import FastLanguageModel
        import torch
        from transformers import TrainingArguments, TrainerCallback
        from trl import SFTTrainer
        from datasets import load_dataset
        from unsloth.chat_templates import get_chat_template

        return FastLanguageModel, torch, TrainingArguments, TrainerCallback, SFTTrainer, load_dataset, get_chat_template
    except ImportError as exc:
        message = str(exc)
        if "CXXABI_1.3.11" in message or "libstdc++.so.6" in message:
            print(
                "Failed to import torch/unsloth because the active libstdc++ runtime is too old for the installed PyTorch wheel.\n"
                "On this cluster, load a newer GCC runtime or prepend a newer libstdc++.so.6 to LD_LIBRARY_PATH before running train_unsloth.py.\n"
                "Example:\n"
                "  module load GCC/13.3.0\n"
                "  export LD_LIBRARY_PATH=\"/path/to/newer/gcc/lib:$LD_LIBRARY_PATH\"\n"
                "If you are using conda, reinstalling/updating libstdcxx-ng in the environment can also resolve the mismatch.",
                file=sys.stderr,
            )
        raise


FastLanguageModel, torch, TrainingArguments, TrainerCallback, SFTTrainer, load_dataset, get_chat_template = _import_training_stack()

# Unsloth's exec()-compiled _fast_inner_training_loop changes the class identity
# of trl.trainer.sft_config.SFTConfig in sys.modules as a side effect of its
# patching. This makes torch.save(self.args) fail with PicklingError during
# checkpoint saves. Patch torch.save to skip training_args.bin silently — it is
# metadata-only and not needed to load or use the saved LoRA adapter. Model
# weights are always saved via safetensors and are unaffected.
import pickle as _pickle

_orig_torch_save = torch.save

def _torch_save_patched(obj, f, *args, **kwargs):
    try:
        _orig_torch_save(obj, f, *args, **kwargs)
    except _pickle.PicklingError as _e:
        if "not the same object" in str(_e):
            print(f"[train_unsloth] Skipping {os.path.basename(str(f))} — SFTConfig identity mismatch (known unsloth/trl compat issue)")
        else:
            raise

torch.save = _torch_save_patched


def _slugify(value):
    value = str(value).strip().lower()
    value = value.replace("/", "-")
    value = re.sub(r"[^a-z0-9.-]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "model"


def _read_cuda_resources():
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device_count": 0,
            "device_name": None,
            "total_gb": None,
            "free_gb": None,
            "bf16_supported": False,
        }

    device_index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device_index)
    total_gb = properties.total_memory / (1024 ** 3)
    free_gb = None
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / (1024 ** 3)
    except Exception:
        free_gb = None

    return {
        "cuda_available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(device_index),
        "total_gb": total_gb,
        "free_gb": free_gb,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _choose_training_mode():
    resources = _read_cuda_resources()
    if not resources["cuda_available"]:
        raise RuntimeError("CUDA is required for this training script.")

    force_4bit = os.getenv("OCPQ_FORCE_4BIT", "").strip().lower() in {"1", "true", "yes", "on"}
    force_full_precision = os.getenv("OCPQ_FORCE_FULL_PRECISION", "").strip().lower() in {"1", "true", "yes", "on"}
    min_full_precision_gb = float(os.getenv("OCPQ_MIN_FULL_PRECISION_GB", "32"))

    if force_4bit and force_full_precision:
        raise ValueError("Set only one of OCPQ_FORCE_4BIT or OCPQ_FORCE_FULL_PRECISION.")

    if force_4bit:
        use_4bit = True
    elif force_full_precision:
        use_4bit = False
    elif resources["free_gb"] is not None:
        use_4bit = resources["free_gb"] < min_full_precision_gb
    else:
        use_4bit = resources["total_gb"] < min_full_precision_gb

    quantized_model_name = os.getenv(
        "OCPQ_QUANTIZED_MODEL",
        "unsloth/llama-3-8b-Instruct-bnb-4bit",
    )
    full_precision_model_name = os.getenv(
        "OCPQ_FULL_PRECISION_MODEL",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    )
    selected_model_name = quantized_model_name if use_4bit else full_precision_model_name

    base_label_source = os.getenv("OCPQ_MODEL_LABEL", selected_model_name)
    model_label = _slugify(base_label_source.split("/")[-1])
    precision_label = "quantized" if use_4bit else "full-precision"
    run_label = f"{model_label}-fine-tuned-{precision_label}"

    dtype = None if use_4bit else (torch.bfloat16 if resources["bf16_supported"] else torch.float16)

    return {
        "resources": resources,
        "use_4bit": use_4bit,
        "dtype": dtype,
        "model_name": selected_model_name,
        "precision_label": precision_label,
        "run_label": run_label,
    }

# 1. Configuration
_gc_env = os.getenv("OCPQ_GRADIENT_CHECKPOINTING", "unsloth")
gradient_checkpointing_mode = False if _gc_env == "false" else (True if _gc_env == "true" else _gc_env)

# 4096 covers the whole fine-tuning corpus (longest row ~1.6k tokens) with headroom for the
# inference distribution, whose longest prompts (chain-of-thought-plus) reach ~4k tokens.
# Do not lower this: SFTTrainer truncates the END of the sequence, which is the assistant's
# JSON target, so an undersized budget silently trains the model to emit incomplete JSON.
max_seq_length = 4096
random_seed = int(os.getenv("OCPQ_RANDOM_SEED", "3407"))


def _seed_everything(seed):
    """Seed every RNG the run touches, before anything stochastic happens.

    The component-level seeds (LoRA `random_state`, the dataset split, and
    `TrainingArguments.seed`) already cover the parts that matter most, but HF only calls
    `set_seed` once the Trainer is constructed. Seeding here removes any dependence on
    that ordering and matches the project-wide reproducibility rule.
    """
    import random as _random

    _random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


_seed_everything(random_seed)

validation_split = float(os.getenv("OCPQ_VALIDATION_SPLIT", "0.1"))
learning_rate = float(os.getenv("OCPQ_LEARNING_RATE", "2e-4"))
num_train_epochs = float(os.getenv("OCPQ_NUM_TRAIN_EPOCHS", "1"))
logging_steps = int(os.getenv("OCPQ_LOGGING_STEPS", "100"))
eval_steps = int(os.getenv("OCPQ_EVAL_STEPS", "250"))
save_steps = int(os.getenv("OCPQ_SAVE_STEPS", str(eval_steps)))
finetune_dataset_path = os.getenv("OCPQ_FINETUNE_DATASET", "data/finetune_ocpq.jsonl")
dataset_num_proc = int(os.getenv("OCPQ_DATASET_NUM_PROC", "1"))
disable_tqdm = os.getenv("OCPQ_DISABLE_TQDM", "0").strip().lower() in {"1", "true", "yes", "on"}
enable_trainer_eval = os.getenv("OCPQ_ENABLE_TRAINER_EVAL", "1").strip().lower() in {"1", "true", "yes", "on"}
enable_final_eval = os.getenv("OCPQ_ENABLE_FINAL_EVAL", "1").strip().lower() in {"1", "true", "yes", "on"}
train_batch_size = int(os.getenv("OCPQ_TRAIN_BATCH_SIZE", "2"))
eval_batch_size = int(os.getenv("OCPQ_EVAL_BATCH_SIZE", str(train_batch_size)))
gradient_accumulation_steps = int(os.getenv("OCPQ_GRADIENT_ACCUMULATION_STEPS", "4"))
max_seq_length = int(os.getenv("OCPQ_MAX_SEQ_LENGTH", str(max_seq_length)))

training_mode = _choose_training_mode()
resources = training_mode["resources"]
dtype = training_mode["dtype"]
load_in_4bit = training_mode["use_4bit"]
selected_model_name = training_mode["model_name"]
precision_label = training_mode["precision_label"]
run_label = training_mode["run_label"]
attention_implementation = os.getenv("OCPQ_ATTENTION_IMPLEMENTATION")

output_dir = os.getenv("OCPQ_OUTPUT_DIR") or os.path.join("outputs", run_label)
adapter_output_dir = os.getenv("OCPQ_ADAPTER_OUTPUT_DIR") or str(paths.MODELS / "adapters" / run_label)
recovery_output_dir = os.getenv("OCPQ_RECOVERY_DIR") or os.path.join(output_dir, "recovery")
recovery_snapshot_dir = os.path.join(recovery_output_dir, "latest-adapter")
recovery_manifest_path = os.path.join(recovery_output_dir, "recovery-state.json")
latest_checkpoint_pointer_path = os.path.join(recovery_output_dir, "latest-checkpoint.txt")


def _ensure_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _latest_checkpoint_path(output_directory):
    output_path = Path(output_directory)
    if not output_path.exists():
        return None

    checkpoints = []
    for child in output_path.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda item: item[0])
    return str(checkpoints[-1][1])


def _has_trainer_state(checkpoint_path):
    """Whether a directory is a full HF trainer checkpoint rather than a bare adapter snapshot.

    `adapter_config.json` is the wrong discriminator: *every* PEFT directory has one, including
    `checkpoint-*` produced by the trainer. Testing it therefore classified real trainer
    checkpoints as adapter snapshots, so `resume_from_checkpoint` was never passed to
    `trainer.train()` and the optimizer state, LR-scheduler position and step count were
    silently discarded on every resume.

    `trainer_state.json` is written only by `Trainer._save_checkpoint`, alongside `optimizer.pt`
    and `scheduler.pt`, so it is present exactly when a real resume is possible.

    (`training_args.bin` would be the other candidate, but the `torch.save` monkey-patch above
    deliberately skips it to work around the unsloth/trl SFTConfig pickling issue, so it is
    absent from otherwise-complete checkpoints.)
    """
    return bool(checkpoint_path) and (Path(checkpoint_path) / "trainer_state.json").is_file()


# Opt-in for the weights-only path, which restores adapter weights without optimizer or
# scheduler state. Off by default: it restarts the LR schedule over converged weights.
allow_weights_only_resume = os.getenv(
    "OCPQ_ALLOW_WEIGHTS_ONLY_RESUME", ""
).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_resume_checkpoint(output_directory):
    resume_setting = os.getenv("OCPQ_RESUME_FROM_CHECKPOINT", "latest").strip()
    if not resume_setting:
        return None

    normalized = resume_setting.lower()
    if normalized in {"0", "false", "no", "off", "none"}:
        return None
    if normalized in {"1", "true", "yes", "on", "latest", "auto"}:
        return _latest_checkpoint_path(output_directory)
    if normalized in {"adapter", "recovery", "latest-adapter"}:
        for candidate in (recovery_snapshot_dir, adapter_output_dir):
            candidate_path = Path(candidate)
            if (candidate_path / "adapter_config.json").exists():
                return str(candidate_path)
        return None
    return resume_setting


_recovery_snapshot_in_progress = False
_train_completed = False


def _persist_recovery_snapshot(reason, checkpoint_dir=None, global_step=None, exception_text=None):
    global _recovery_snapshot_in_progress

    if _recovery_snapshot_in_progress:
        return

    if model is None or tokenizer is None:
        print(
            "Skipping recovery snapshot because model/tokenizer are not initialized yet.",
            file=sys.stderr,
        )
        return

    _recovery_snapshot_in_progress = True
    try:
        _ensure_directory(recovery_output_dir)
        _ensure_directory(recovery_snapshot_dir)

        model.save_pretrained(recovery_snapshot_dir)
        tokenizer.save_pretrained(recovery_snapshot_dir)

        manifest = {
            "reason": reason,
            "timestamp": time.time(),
            "selected_model_name": selected_model_name,
            "run_label": run_label,
            "output_dir": output_dir,
            "adapter_output_dir": adapter_output_dir,
            "recovery_snapshot_dir": recovery_snapshot_dir,
            "checkpoint_dir": checkpoint_dir,
            "global_step": global_step,
            "exception_text": exception_text,
        }
        with open(recovery_manifest_path, "w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2, sort_keys=True)

        if checkpoint_dir:
            with open(latest_checkpoint_pointer_path, "w", encoding="utf-8") as pointer_file:
                pointer_file.write(str(checkpoint_dir))
                pointer_file.write("\n")

        print(f"Recovery snapshot saved to '{recovery_snapshot_dir}' ({reason}).")
    except Exception as snapshot_exc:
        print(f"Warning: failed to save recovery snapshot: {snapshot_exc}", file=sys.stderr)
    finally:
        _recovery_snapshot_in_progress = False


def _install_termination_handlers():
    signal_names = {}
    if hasattr(signal, "SIGINT"):
        signal_names[getattr(signal, "SIGINT")] = "SIGINT"
    if hasattr(signal, "SIGTERM"):
        signal_names[getattr(signal, "SIGTERM")] = "SIGTERM"

    def _handle_termination(signum, frame):
        signal_name = signal_names.get(signum, f"signal {signum}")
        _persist_recovery_snapshot(
            reason=f"terminated-by-{signal_name}",
            checkpoint_dir=_latest_checkpoint_path(output_dir),
        )
        raise SystemExit(128 + signum)

    for signal_name in ("SIGINT", "SIGTERM"):
        if hasattr(signal, signal_name):
            try:
                signal.signal(getattr(signal, signal_name), _handle_termination)
            except Exception:
                pass


def _save_on_exit():
    if not _train_completed:
        _persist_recovery_snapshot(
            reason="atexit",
            checkpoint_dir=_latest_checkpoint_path(output_dir),
        )


atexit.register(_save_on_exit)
_install_termination_handlers()

print(
    "CUDA resources detected: "
    f"device={resources['device_name']}, "
    f"total_gb={resources['total_gb']:.1f}, "
    f"free_gb={(resources['free_gb'] if resources['free_gb'] is not None else 'unknown')}, "
    f"mode={precision_label}"
)

# 2. Load Model & Tokenizer using the selected precision mode
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = selected_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    attn_implementation = attention_implementation,
)

# 2b. Did unsloth substitute the checkpoint?
# With load_in_4bit=True it silently swaps a full-precision id for its own pre-quantized
# redistribution, and the LoRA is then trained against weights that are NOT what inference will
# load. Gemma 3 was trained that way for a full cycle before anyone noticed, and it could not be
# repaired afterwards: the substituted artifact cannot be loaded for inference at all ("FP4
# quantization state not initialized", job 3015021).
#
# Reported here, at load time, rather than only at save time — this is the last moment where
# aborting costs minutes instead of hours. The fix is always the same: request the unsloth
# pre-quantized id directly, as the Llama 3.1, Ministral 14B and Ministral 3B scripts do.
_loaded_base = (
    getattr(getattr(model, "config", None), "_name_or_path", None)
    or getattr(getattr(model, "config", None), "name_or_path", None)
)
if _loaded_base and selected_model_name and _loaded_base != selected_model_name:
    print("=" * 78)
    print("WARNING: unsloth substituted the requested checkpoint.")
    print(f"  requested : {selected_model_name}")
    print(f"  loaded    : {_loaded_base}")
    print("  The adapter will be trained against the loaded checkpoint. Unless inference loads")
    print("  that same id, the LoRA will be applied to different weights. Request the unsloth")
    print("  pre-quantized id directly to avoid this.")
    print("=" * 78, flush=True)
else:
    print(f"[model load] base checkpoint confirmed, no substitution: {selected_model_name}",
          flush=True)

# 3. Add LoRA Adapters
# Multimodal checkpoints (Ministral 3, Gemma 3) carry a vision tower alongside the language
# model. This is a text-only task, so say so explicitly: otherwise unsloth applies its
# finetune_(vision|language) filters against our target_modules and warns that "adapters
# attach only where both select", leaving it unclear which tower was actually trained.
# Passed only for vision models, and behind a TypeError fallback, because older unsloth
# builds and text-only models do not accept these keywords.
_is_multimodal = (
    hasattr(tokenizer, "image_processor")
    or getattr(getattr(model, "config", None), "vision_config", None) is not None
)
_lora_kwargs = dict(
    r = 16, # Rank of the adapter (16 is a good standard)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = gradient_checkpointing_mode,
    random_state = random_seed,
    use_rslora = False,
    loftq_config = None,
)
if _is_multimodal:
    print("[train_unsloth] multimodal checkpoint detected — attaching LoRA to the language "
          "tower only (finetune_vision_layers=False)")
try:
    model = FastLanguageModel.get_peft_model(
        model,
        **_lora_kwargs,
        **({"finetune_vision_layers": False, "finetune_language_layers": True}
           if _is_multimodal else {}),
    )
except TypeError:
    # This unsloth build does not expose the tower filters; fall back to the plain call.
    model = FastLanguageModel.get_peft_model(model, **_lora_kwargs)

resume_checkpoint = _resolve_resume_checkpoint(output_dir)
resume_has_trainer_state = _has_trainer_state(resume_checkpoint)

if resume_checkpoint and resume_has_trainer_state:
    # A full trainer checkpoint. Do NOT hand-load the adapter here: `trainer.train()` is given
    # this path below and restores weights, optimizer state, LR-scheduler position and step
    # count together. Loading the weights here as well would be redundant at best.
    print(
        f"[train_unsloth] full trainer checkpoint at {resume_checkpoint}; "
        "weights, optimizer and LR schedule will be restored by the trainer."
    )
elif resume_checkpoint:
    # Weights-only: an adapter snapshot with no optimizer or scheduler state. Training would
    # restart at step 0, re-running warmup and the full lr=2e-4 schedule over already-converged
    # weights. That is what collapsed the GPT-OSS-20B adapter into repeating a single token, so
    # it requires an explicit opt-in rather than a log line nobody reads.
    if not allow_weights_only_resume:
        raise SystemExit(
            f"Refusing to resume from '{resume_checkpoint}': it has no trainer_state.json, so "
            "only the adapter weights can be restored. Training would restart at step 0 and "
            "re-apply warmup plus the full learning-rate schedule to converged weights, which "
            "destroys the adapter.\n"
            "  - To continue an interrupted run, point at a checkpoint-* directory "
            "(OCPQ_RESUME_FROM_CHECKPOINT=latest).\n"
            "  - To start over, set OCPQ_RESUME_FROM_CHECKPOINT=0.\n"
            "  - If you really do want weights-only warm-starting, set "
            "OCPQ_ALLOW_WEIGHTS_ONLY_RESUME=1."
        )
    print(
        f"[train_unsloth] WARNING: restoring adapter weights only from {resume_checkpoint} "
        "(no trainer state). The LR schedule restarts from step 0 over converged weights — "
        "this can destroy the adapter. Proceeding because OCPQ_ALLOW_WEIGHTS_ONLY_RESUME is set."
    )
    try:
        model.delete_adapter("default")
    except Exception:
        pass
    model.load_adapter(
        resume_checkpoint,
        adapter_name="default",
        is_trainable=True,
        low_cpu_mem_usage=True,
    )
    model.set_adapter("default")

# 4. Format Dataset
# OCPQ_CHAT_TEMPLATE controls how the tokenizer formats chat turns.
# Use "auto" to trust the model's own built-in chat_template (required for models
# like Phi-4 whose template is not one of unsloth's named presets).
_chat_template_key = os.getenv("OCPQ_CHAT_TEMPLATE", "llama-3")
if _chat_template_key == "auto":
    # The HuggingFace tokenizer already has the correct Jinja template embedded;
    # calling get_chat_template would overwrite it with a wrong preset.
    pass
else:
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = _chat_template_key,
        mapping = {"role" : "role", "content" : "content", "user": "user", "assistant": "assistant"},
    )

# Set by _resolve_duplicate_bos() below, and honoured by formatting_prompts_func.
strip_leading_bos_from_text = False


def formatting_prompts_func(examples):
    # Applies the model's chat template to our converted JSONL rows.
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        for convo in convos
    ]
    if strip_leading_bos_from_text:
        bos = tokenizer.bos_token
        texts = [t[len(bos):] if t.startswith(bos) else t for t in texts]
    return {"text": texts}

def _grouped_train_validation_split(dataset, test_fraction, seed):
    """Split rows into train/validation, keeping same-target rows on the same side.

    `convert_dataset_for_finetuning.py` emits several rows per source example (a base
    prompt, a reasoning-augmented variant, a contrastive variant, sometimes a rewrite),
    and every one of them carries the *identical* assistant target. Splitting at row
    level therefore puts variants of one example on both sides: measured on the current
    corpus, 99.2% of validation rows had their exact target JSON present in train, which
    makes eval_loss a memorisation score rather than a held-out one. That matters twice
    over, because `load_best_model_at_end` selects the reported checkpoint by eval_loss.

    Grouping by the assistant target keeps every variant of an example together. Examples
    that happen to share an identical gold query are grouped too, which is also correct —
    they are the same query semantically.

    Args:
        dataset: The raw `datasets.Dataset` of chat rows.
        test_fraction: Fraction of *groups* to hold out.
        seed: Seed for the group shuffle.

    Returns:
        `(train_indices, validation_indices)`.
    """
    import random as _random

    group_of_row = []
    groups = {}
    for messages in dataset["messages"]:
        target = messages[-1]["content"]
        if target not in groups:
            groups[target] = len(groups)
        group_of_row.append(groups[target])

    group_ids = list(range(len(groups)))
    _random.Random(seed).shuffle(group_ids)
    holdout = set(group_ids[: int(len(group_ids) * test_fraction)])

    train_indices, validation_indices = [], []
    for row_index, group_id in enumerate(group_of_row):
        (validation_indices if group_id in holdout else train_indices).append(row_index)
    return train_indices, validation_indices


def _has_duplicate_leading_bos(sample_text):
    """Whether tokenizing already-templated text yields two BOS tokens in a row.

    `text=` is passed by keyword, never positionally. For a multimodal checkpoint
    (Ministral 3, Gemma 3) the loader returns a *processor*, not a tokenizer, and a
    processor's first positional parameter is `images` — so a positional call hands the
    prompt string to PIL and dies with "cannot identify image file". `text=` is the correct
    keyword for both a plain tokenizer and a processor.
    """
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        return False
    ids = tokenizer(text=sample_text, add_special_tokens=True)["input_ids"]
    return len(ids) >= 2 and ids[0] == bos_id and ids[1] == bos_id


def _resolve_duplicate_bos():
    """Decide how to keep exactly one BOS, before the dataset is formatted.

    `apply_chat_template` emits the model's BOS inside the text, and the trainer then
    tokenizes that text with `add_special_tokens=True`, whose post-processor can prepend
    a second one. Observed on Llama-3; Phi-4 and Gemma-3 are unaffected. It is repaired
    per model rather than by forcing `add_special_tokens=False` globally, which would
    strip the *only* BOS from a model whose template does not emit one.

    Two remedies are tried in order:

    1. Clear `tokenizer.add_bos_token`. Idiomatic, but on Llama-3 under unsloth the extra
       BOS comes from the fast tokenizer's post-processor and the attribute is already
       absent/false, so this does nothing — hence the fallback.
    2. Strip the leading BOS from the formatted text, leaving the tokenizer to contribute
       the single one. Independent of tokenizer internals and trl version.

    Sets the module-level `strip_leading_bos_from_text` flag used by
    `formatting_prompts_func`.
    """
    global strip_leading_bos_from_text

    probe_messages = raw_dataset[0]["messages"]
    sample = tokenizer.apply_chat_template(
        probe_messages, tokenize=False, add_generation_prompt=False
    )

    if not _has_duplicate_leading_bos(sample):
        print("[train_unsloth] BOS check OK (no duplicate leading BOS).")
        return

    # Remedy 1: the tokenizer flag, if this tokenizer honours it.
    if hasattr(tokenizer, "add_bos_token"):
        try:
            tokenizer.add_bos_token = False
            if not _has_duplicate_leading_bos(sample):
                print(
                    "[train_unsloth] BOS FIXED (tokenizer.add_bos_token = False): the "
                    "tokenizer was adding a second BOS on top of the template's."
                )
                return
        except Exception as exc:
            print(f"[train_unsloth] could not set tokenizer.add_bos_token: {exc}")

    # Remedy 2: drop the template's BOS from the text instead.
    bos = tokenizer.bos_token
    if bos and sample.startswith(bos):
        strip_leading_bos_from_text = True
        if not _has_duplicate_leading_bos(sample[len(bos):]):
            print(
                f"[train_unsloth] BOS FIXED (stripping leading {bos!r} from the formatted "
                "text): the tokenizer adds one via add_special_tokens, so the template's "
                "copy is removed, leaving exactly one."
            )
            return
        strip_leading_bos_from_text = False

    print(
        "[train_unsloth] WARNING: duplicate BOS could not be repaired. Training would see "
        "a doubled BOS, which differs from what inference produces via apply_chat_template."
    )


# Split before formatting so train/validation stay disjoint at the *example* level, not
# merely the row level (see _grouped_train_validation_split).
raw_dataset = load_dataset("json", data_files=finetune_dataset_path, split="train")
train_indices, validation_indices = _grouped_train_validation_split(
    raw_dataset, validation_split, random_seed
)

# Decide the BOS handling before formatting, since the remedy may change the text itself.
_resolve_duplicate_bos()

train_dataset = raw_dataset.select(train_indices).map(formatting_prompts_func, batched=True)
eval_dataset = raw_dataset.select(validation_indices).map(formatting_prompts_func, batched=True)

print(f"Loaded {len(raw_dataset)} supervised rows from {finetune_dataset_path}")
print(f"Train rows: {len(train_dataset)}")
print(f"Validation rows: {len(eval_dataset)} "
      f"({100.0 * len(eval_dataset) / max(len(raw_dataset), 1):.1f}% of rows, "
      "split by target so no variant of a validation example appears in train)")

def _report_formatting_sanity(sample_text):
    """Verify the formatted text after the BOS decision has been applied.

    Re-checks BOS on the real formatted output, and confirms the text ends with the
    template's end-of-turn token — without it the model never learns to stop, which for
    JSON output means unterminated objects.
    """
    if _has_duplicate_leading_bos(sample_text):
        print("[train_unsloth] WARNING: formatted text still tokenizes to a doubled BOS.")
    else:
        print("[train_unsloth] BOS verified: exactly one leading BOS after tokenization.")

    tail = sample_text.rstrip()
    eos_token = tokenizer.eos_token
    known_terminators = [t for t in (eos_token, "<|eot_id|>", "<end_of_turn>", "<|im_end|>") if t]
    if any(tail.endswith(t) for t in known_terminators):
        print("[train_unsloth] EOS check OK (formatted text ends with an end-of-turn token).")
    else:
        print(
            "[train_unsloth] WARNING: formatted text does not end with a known end-of-turn "
            f"token (eos_token={eos_token!r}); the model may never learn to stop. "
            f"Tail: {tail[-60:]!r}"
        )


if len(train_dataset) > 0:
    _report_formatting_sanity(train_dataset["text"][0])

_preview_n = int(os.getenv("OCPQ_PREVIEW_EXAMPLES", "0"))
if _preview_n > 0:
    print(f"\n=== Formatted example preview (OCPQ_PREVIEW_EXAMPLES={_preview_n}) ===")
    for i, text in enumerate(train_dataset["text"][:_preview_n]):
        print(f"\n--- Example {i} ---\n{text}")
    print("=== End of preview ===")

print(
    "[train_unsloth] resolved config: "
    f"max_seq_length={max_seq_length} "
    f"train_batch={train_batch_size} eval_batch={eval_batch_size} "
    f"grad_accum={gradient_accumulation_steps} "
    f"eval_steps={eval_steps} save_steps={save_steps} seed={random_seed}"
)

# 5. Configure the Trainer
# For a multimodal checkpoint the loader returns a processor, but by this point the dataset
# is already formatted into a plain "text" column, so the trainer only has to *tokenize*.
# Hand it the inner text tokenizer: a processor treats its first positional argument as
# `images`, which is what broke the BOS check, and TRL tokenizes internally where we cannot
# add a `text=` keyword. Text-only models are unaffected — they have no `.tokenizer`.
_trainer_tokenizer = getattr(tokenizer, "tokenizer", None) if _is_multimodal else None
if _trainer_tokenizer is not None:
    print("[train_unsloth] passing the processor's inner text tokenizer to SFTTrainer")
else:
    _trainer_tokenizer = tokenizer

trainer = SFTTrainer(
    model = model,
    tokenizer = _trainer_tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset if enable_trainer_eval else None,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = dataset_num_proc,
    packing = False, # Can make training faster, but takes more memory
    args = TrainingArguments(
        per_device_train_batch_size = train_batch_size,
        per_device_eval_batch_size = eval_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        warmup_steps = 10,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = logging_steps,
        eval_strategy = "steps" if enable_trainer_eval else "no",
        eval_steps = eval_steps,
        save_strategy = "steps",
        save_steps = save_steps,
        save_total_limit = 2,
        load_best_model_at_end = enable_trainer_eval,
        metric_for_best_model = "eval_loss" if enable_trainer_eval else None,
        greater_is_better = False,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = random_seed,
        output_dir = output_dir,
        report_to = "none",
        disable_tqdm = disable_tqdm,
        torch_compile = False,
    ),
)

# Unsloth patches the trainer after construction and can override what was requested, so
# report what the trainer will actually use. The first runs at 4096 showed eval running at
# one sample per step (eval_samples_per_second == eval_steps_per_second) despite
# per_device_eval_batch_size=8 being passed, and eval accounted for roughly half of total
# wall time — so it is worth seeing the effective value rather than the requested one.
print(
    "[train_unsloth] trainer effective: "
    f"per_device_train_batch_size={trainer.args.per_device_train_batch_size} "
    f"per_device_eval_batch_size={trainer.args.per_device_eval_batch_size} "
    f"gradient_accumulation_steps={trainer.args.gradient_accumulation_steps}"
)
if trainer.args.per_device_eval_batch_size != eval_batch_size:
    print(
        "[train_unsloth] NOTE: requested eval batch "
        f"{eval_batch_size} but the trainer will use "
        f"{trainer.args.per_device_eval_batch_size}; evaluation will be slower than planned."
    )


def _report_trainer_input_ids():
    """Report the leading token IDs the model will actually train on.

    `_resolve_duplicate_bos()` inspects `tokenizer`, but TRL tokenizes with
    `_trainer_tokenizer`, and for a multimodal checkpoint those are different objects (the
    processor vs. its inner text tokenizer). Unsloth also performs its own de-duplication
    ("We found double BOS tokens - we shall remove one automatically") *after* the trainer
    is built. With three mechanisms acting on the same sequence, the only trustworthy
    statement about how many BOS tokens the model sees is a reading of the trainer's own
    tokenized dataset — one BOS is correct, two means the repairs did not take, and zero
    means two of them fired on the same sequence.
    """
    try:
        row = trainer.train_dataset[0]
        ids = row.get("input_ids") if hasattr(row, "get") else None
        if not ids:
            print("[train_unsloth] trainer dataset exposes no input_ids; skipping BOS audit.")
            return

        bos_id = getattr(_trainer_tokenizer, "bos_token_id", None)
        leading = sum(1 for token in ids[:4] if token == bos_id) if bos_id is not None else 0
        decoded = _trainer_tokenizer.decode(ids[:8])
        print(f"[train_unsloth] trainer input_ids[:8] = {list(ids[:8])}")
        print(f"[train_unsloth] decoded             = {decoded!r}")

        if bos_id is None:
            print("[train_unsloth] tokenizer exposes no bos_token_id; BOS count not checked.")
        elif leading == 1:
            print("[train_unsloth] BOS AUDIT OK: the model trains on exactly one leading BOS.")
        elif leading == 0:
            print(
                "[train_unsloth] BOS AUDIT WARNING: no leading BOS in the trainer's input_ids. "
                "Two de-duplication mechanisms may have fired on the same sequence."
            )
        else:
            print(
                f"[train_unsloth] BOS AUDIT WARNING: {leading} leading BOS tokens in the "
                "trainer's input_ids; the repairs did not take effect on this path."
            )
    except Exception as exc:  # noqa: BLE001 - a diagnostic must never abort training
        print(f"[train_unsloth] could not audit trainer input_ids: {exc}")


_report_trainer_input_ids()


class RecoveryCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _persist_recovery_snapshot(
            reason="trainer-save",
            checkpoint_dir=checkpoint_dir,
            global_step=state.global_step,
        )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        _persist_recovery_snapshot(
            reason="train-end",
            checkpoint_dir=checkpoint_dir,
            global_step=state.global_step,
        )
        return control


trainer.add_callback(RecoveryCallback())

def _manual_eval_loss(model, tokenizer, texts, batch_size=2):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0

    for start_index in range(0, len(texts), batch_size):
        batch_texts = texts[start_index:start_index + batch_size]
        # Keyword `text=`, for the same reason as _has_duplicate_leading_bos: a multimodal
        # checkpoint yields a processor whose first positional argument is `images`.
        batch = tokenizer(
            text=batch_texts,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        batch = {key: value.to(device) for key, value in batch.items()}
        labels = batch["input_ids"].clone()
        padding_mask = batch["attention_mask"] == 0
        labels[padding_mask] = -100

        with torch.no_grad():
            outputs = model(**batch, labels=labels)

        valid_tokens = int((labels != -100).sum().item())
        total_loss += float(outputs.loss.item()) * valid_tokens
        total_tokens += valid_tokens

    return {"eval_loss": total_loss / max(total_tokens, 1)}

# 6. Start Training
print("Starting Unsloth Training...")
# Pass the checkpoint to the trainer only when it carries trainer state; see _has_trainer_state
# for why `adapter_config.json` was the wrong test. The weights-only case has already been
# gated (and loaded) above, so here it deliberately resolves to None.
resume_from_checkpoint = resume_checkpoint if resume_has_trainer_state else None
if resume_from_checkpoint:
    print(f"Resuming from checkpoint (weights + optimizer + LR schedule): {resume_from_checkpoint}")
elif resume_checkpoint:
    print("Continuing from adapter weights only; the LR schedule restarts from step 0.")

try:
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    _train_completed = True
except Exception as train_exc:
    _persist_recovery_snapshot(
        reason="train-exception",
        checkpoint_dir=_latest_checkpoint_path(output_dir),
        exception_text=repr(train_exc),
    )
    raise

if enable_final_eval:
    try:
        if enable_trainer_eval:
            eval_metrics = trainer.evaluate()
        else:
            # Honour the configured eval batch size rather than the function default;
            # this path scores the whole validation split, so batch size dominates its
            # wall time. The loss is token-weighted with padding masked to -100, so the
            # value is unchanged by batching.
            eval_metrics = _manual_eval_loss(
                model, tokenizer, eval_dataset["text"], batch_size=eval_batch_size
            )
        print("Final validation metrics:", eval_metrics)
    except Exception as eval_exc:
        _persist_recovery_snapshot(
            reason="eval-exception",
            checkpoint_dir=_latest_checkpoint_path(output_dir),
            exception_text=repr(eval_exc),
        )
        raise

def _restore_requested_base_model_name(adapter_dir, requested_model_name):
    """Point the saved adapter at the checkpoint the job requested, and record what it displaced.

    When `load_in_4bit=True`, unsloth silently swaps the requested checkpoint for its own
    pre-quantized redistribution, and that substituted name is what lands in
    `adapter_config.json`. `LLM.py` reads that field to decide which base to load.

    So this redirect creates a real train/eval divergence: the LoRA is trained against unsloth's
    artifact and evaluated against the requested one. That is not the preferred outcome — but for
    Gemma 3 it is the only workable one. Pointing inference at the substituted artifact was tried
    on 2026-08-16 (job 3015021) and **both Gemma arms failed to generate a single token**:
    bitsandbytes raises "FP4 quantization state not initialized" and then asserts inside
    `fix_4bit_weight_quant_state_from_module`. Llama and Ministral load from their unsloth
    artifacts without trouble, so this is a defect in that particular redistribution.

    The divergence therefore stays, and what changed is that it is no longer *silent*: it is
    logged at save time and written to `ocpq_provenance.json` beside the adapter. It went
    unnoticed for a full evaluation cycle precisely because nothing recorded it.

    The LoRA weights are unaffected — only which base they are injected into.

    Args:
        adapter_dir: Directory `save_pretrained` just wrote.
        requested_model_name: The checkpoint this run asked for.
    """
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not requested_model_name or not os.path.exists(config_path):
        return

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
    except Exception as exc:
        print(f"Warning: could not read {config_path} to fix base model name: {exc}")
        return

    saved_name = adapter_config.get("base_model_name_or_path")
    if saved_name == requested_model_name:
        return

    adapter_config["base_model_name_or_path"] = requested_model_name
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not write {config_path} to fix base model name: {exc}")
        return

    # The substitution used to happen silently, which is how the Gemma train/eval divergence went
    # unnoticed. Record it in a sidecar rather than in adapter_config.json, which PEFT parses as a
    # LoraConfig and warns about on every load for unknown keys.
    provenance_path = os.path.join(adapter_dir, "ocpq_provenance.json")
    try:
        with open(provenance_path, "w", encoding="utf-8") as f:
            json.dump({
                "trained_against": saved_name,
                "evaluated_against": requested_model_name,
                "requested_by_training_job": requested_model_name,
                "why_they_differ": (
                    "unsloth substituted its own pre-quantized redistribution when "
                    "load_in_4bit=True. base_model_name_or_path is redirected to the requested "
                    "checkpoint because inference must load something that works; see "
                    "11-evaluation-readiness.md for whether the substituted artifact is usable "
                    "for this model."
                ),
            }, f, indent=2)
    except Exception as exc:
        print(f"Warning: could not write {provenance_path}: {exc}")

    print(f"[adapter save] NOTE: unsloth trained against '{saved_name}' but "
          f"base_model_name_or_path was set to the requested '{requested_model_name}'. "
          f"Inference will apply this LoRA to differently-quantized weights — recorded in "
          f"{os.path.basename(provenance_path)}.")

    print(
        "[adapter save] base_model_name_or_path "
        f"'{saved_name}' -> '{requested_model_name}' (unsloth substituted its pre-quantized "
        "redistribution; recording the checkpoint this run requested)"
    )


# 7. Save the Fine-Tuned LoRA Adapter
model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
_restore_requested_base_model_name(adapter_output_dir, selected_model_name)
print(f"Finished! Saved LoRA adapter to '{adapter_output_dir}'")
print(f"Training label: {run_label}")
