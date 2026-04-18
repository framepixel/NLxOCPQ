from unsloth import FastLanguageModel
import torch
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

# 1. Configuration
max_seq_length = 2048 # Good size for JSON generation
dtype = None # Auto detection
load_in_4bit = True # CRITICAL for fitting in an 11GB GPU

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
    random_state = 3407,
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

# Load our 10,000 JSONL examples
dataset = load_dataset("json", data_files="finetune_ocpq.jsonl", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True,)

# 5. Configure the Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training faster, but takes more memory
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Small batch size to avoid OOM
        gradient_accumulation_steps = 4, # Simulate a batch size of 8
        warmup_steps = 10,
        num_train_epochs = 1, # 1 epoch over 10,000 rows is great for JSON formatting
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 100,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Start Training
print("Starting Unsloth Training...")
trainer_stats = trainer.train()

# 7. Save the Fine-Tuned LoRA Adapter
model.save_pretrained("lora_model_ocpq")
tokenizer.save_pretrained("lora_model_ocpq")
print("Finished! Saved LoRA adapter to 'lora_model_ocpq'")
