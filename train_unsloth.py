import os
import re
import sys


def _import_training_stack():
    try:
        from unsloth import FastLanguageModel
        import torch
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from datasets import load_dataset
        from unsloth.chat_templates import get_chat_template

        return FastLanguageModel, torch, TrainingArguments, SFTTrainer, load_dataset, get_chat_template
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


FastLanguageModel, torch, TrainingArguments, SFTTrainer, load_dataset, get_chat_template = _import_training_stack()


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
max_seq_length = 2048 # Good size for JSON generation
random_seed = int(os.getenv("OCPQ_RANDOM_SEED", "3407"))
validation_split = float(os.getenv("OCPQ_VALIDATION_SPLIT", "0.1"))
learning_rate = float(os.getenv("OCPQ_LEARNING_RATE", "2e-4"))
num_train_epochs = float(os.getenv("OCPQ_NUM_TRAIN_EPOCHS", "1"))
logging_steps = int(os.getenv("OCPQ_LOGGING_STEPS", "100"))
eval_steps = int(os.getenv("OCPQ_EVAL_STEPS", "250"))
save_steps = int(os.getenv("OCPQ_SAVE_STEPS", str(eval_steps)))
finetune_dataset_path = os.getenv("OCPQ_FINETUNE_DATASET", "finetune_ocpq.jsonl")
dataset_num_proc = int(os.getenv("OCPQ_DATASET_NUM_PROC", "1"))
enable_eval = os.getenv("OCPQ_ENABLE_EVAL", "1").strip().lower() in {"1", "true", "yes", "on"}

training_mode = _choose_training_mode()
resources = training_mode["resources"]
dtype = training_mode["dtype"]
load_in_4bit = training_mode["use_4bit"]
selected_model_name = training_mode["model_name"]
precision_label = training_mode["precision_label"]
run_label = training_mode["run_label"]
attention_implementation = os.getenv("OCPQ_ATTENTION_IMPLEMENTATION")
if not attention_implementation and "gpt-oss" in selected_model_name.lower():
    attention_implementation = "eager"

output_dir = os.getenv("OCPQ_OUTPUT_DIR") or os.path.join("outputs", run_label)
adapter_output_dir = os.getenv("OCPQ_ADAPTER_OUTPUT_DIR") or os.path.join("adapters", run_label)

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

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank of the adapter (16 is a good standard)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # CRITICAL for 11GB GPU (slashes VRAM by 30%)
    random_state = random_seed,
    use_rslora = False,
    loftq_config = None,
)

# 4. Format Dataset
# Unsloth applies the exact Llama-3 special tokens (<|start_header_id|>user<|end_header_id|>)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = os.getenv("OCPQ_CHAT_TEMPLATE", "llama-3"),
    mapping = {"role" : "role", "content" : "content", "user": "user", "assistant": "assistant"},
)

def formatting_prompts_func(examples):
    # Applies the Llama 3 chat template to our converted JSONL strings
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# Load and split the fine-tuning corpus before formatting so train/validation remain disjoint.
raw_dataset = load_dataset("json", data_files=finetune_dataset_path, split="train")
dataset_split = raw_dataset.train_test_split(test_size=validation_split, seed=random_seed, shuffle=True)
train_dataset = dataset_split["train"].map(formatting_prompts_func, batched=True)
eval_dataset = dataset_split["test"].map(formatting_prompts_func, batched=True)

print(f"Loaded {len(raw_dataset)} supervised rows from {finetune_dataset_path}")
print(f"Train rows: {len(train_dataset)}")
print(f"Validation rows: {len(eval_dataset)}")

# 5. Configure the Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset if enable_eval else None,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = dataset_num_proc,
    packing = False, # Can make training faster, but takes more memory
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Small batch size to avoid OOM
        per_device_eval_batch_size = 2,
        gradient_accumulation_steps = 4, # Simulate a batch size of 8
        warmup_steps = 10,
        num_train_epochs = num_train_epochs,
        learning_rate = learning_rate,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = logging_steps,
        eval_strategy = "steps" if enable_eval else "no",
        eval_steps = eval_steps,
        save_strategy = "steps",
        save_steps = save_steps,
        save_total_limit = 2,
        load_best_model_at_end = enable_eval,
        metric_for_best_model = "eval_loss" if enable_eval else None,
        greater_is_better = False,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = random_seed,
        output_dir = output_dir,
        report_to = "none",
    ),
)

# 6. Start Training
print("Starting Unsloth Training...")
trainer_stats = trainer.train()
if enable_eval:
    eval_metrics = trainer.evaluate()
    print("Final validation metrics:", eval_metrics)

# 7. Save the Fine-Tuned LoRA Adapter
model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
print(f"Finished! Saved LoRA adapter to '{adapter_output_dir}'")
print(f"Training label: {run_label}")
