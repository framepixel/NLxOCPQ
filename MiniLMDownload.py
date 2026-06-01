import os

from transformers import AutoModel, AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
local_dir = "./local-all-MiniLM-L6-v2"

print(f"Downloading {model_name}...")

# Download the tokenizer and encoder directly so we avoid the crashing wrapper.
tokenizer = AutoTokenizer.from_pretrained(model_name)
temp_model = AutoModel.from_pretrained(model_name)

print(f"Saving model locally to {local_dir}...")
os.makedirs(local_dir, exist_ok=True)
tokenizer.save_pretrained(local_dir)
temp_model.save_pretrained(local_dir)
print("Successfully downloaded and saved!")
