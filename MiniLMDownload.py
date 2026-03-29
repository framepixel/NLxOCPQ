from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"   # Hugging Face model ID for SentenceTransformers
local_dir = "./local-all-MiniLM-L6-v2"

print(f"Downloading {model_name}...")
# This will download the model to the default cache and load it
temp_model = SentenceTransformer(model_name)

print(f"Saving model locally to {local_dir}...")
# This saves the model in a format that can be loaded completely offline
temp_model.save(local_dir)
print("Successfully downloaded and saved!")
