from unsloth import FastLanguageModel
import os
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Configuration
max_seq_length = 2048 # Good size for JSON generation
dtype = None # Auto detection
load_in_4bit = True # CRITICAL for fitting in an 11GB GPU
random_seed = int(os.getenv("OCPQ_RANDOM_SEED", "3407"))
validation_split = float(os.getenv("OCPQ_VALIDATION_SPLIT", "0.1"))
learning_rate = float(os.getenv("OCPQ_LEARNING_RATE", "2e-4"))
num_train_epochs = float(os.getenv("OCPQ_NUM_TRAIN_EPOCHS", "1"))
logging_steps = int(os.getenv("OCPQ_LOGGING_STEPS", "100"))
eval_steps = int(os.getenv("OCPQ_EVAL_STEPS", "250"))
save_steps = int(os.getenv("OCPQ_SAVE_STEPS", str(eval_steps)))
output_dir = os.getenv("OCPQ_OUTPUT_DIR", "outputs")
finetune_dataset_path = os.getenv("OCPQ_FINETUNE_DATASET", "finetune_ocpq.jsonl")
adapter_output_dir = os.getenv("OCPQ_ADAPTER_OUTPUT_DIR", "lora_model_ocpq")

# 2. Load Model & Tokenizer in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", # Pre-quantized to save RAM
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters (This makes training possible on 11GB)
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
    chat_template = "llama-3",
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
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
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
        evaluation_strategy = "steps",
        eval_steps = eval_steps,
        save_strategy = "steps",
        save_steps = save_steps,
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
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
eval_metrics = trainer.evaluate()
print("Final validation metrics:", eval_metrics)

# 7. Save the Fine-Tuned LoRA Adapter
model.save_pretrained(adapter_output_dir)
tokenizer.save_pretrained(adapter_output_dir)
print(f"Finished! Saved LoRA adapter to '{adapter_output_dir}'")
