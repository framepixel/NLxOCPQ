#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import json
import os

import openai
import torch
from transformers import AutoModel, AutoTokenizer

import paths


# In[2]:


# Retrieve the API key from environment variables
api_key = os.getenv("HF_TOKEN")

# Set the OpenAI API key
openai.api_key = api_key


# In[3]:


# Load the dataset
with open(paths.DATA / 'ocpq_dataset.json', 'r') as f:
    dataset = json.load(f)

print("Loading the model directly from our local downloaded directory...")
local_dir = str(paths.MODELS / 'local-all-MiniLM-L6-v2')
model_name = 'sentence-transformers/all-MiniLM-L6-v2'

source_dir = local_dir if os.path.isdir(local_dir) else model_name
tokenizer = AutoTokenizer.from_pretrained(source_dir)
model = AutoModel.from_pretrained(source_dir)
print("Model loaded successfully!")


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# In[4]:


def get_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = mean_pooling(model_output, inputs['attention_mask'])
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy()

# For a small prototype, let's embed all dataset questions:
# extract just the natural language questions
nl_queries = [item['nl'] for item in dataset]

print(f"Encoding {len(nl_queries)} examples. Computing embeddings...")
batch_size = 32
embeddings = []

for start in range(0, len(nl_queries), batch_size):
    batch = nl_queries[start:start + batch_size]
    inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs)
    batch_embeddings = mean_pooling(model_output, inputs['attention_mask'])
    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
    embeddings.extend(batch_embeddings.cpu().numpy())

print("Finished compute.")


# In[ ]:


# Save embeddings to a file using numpy
np.save(paths.DATA / 'embeddings.npy', np.array(embeddings))

# Record which corpus these embeddings describe. LLM.load_resources refuses to use them if
# `ocpq_dataset.json` has changed since: the corpus can be regenerated at the same row count,
# in which case the old array still has a matching shape and every retrieval would silently
# return the example that was similar in the OLD corpus, at that index in the NEW one.
import hashlib

_digest = hashlib.sha256()
with open(paths.DATA / 'ocpq_dataset.json', 'rb') as _f:
    for _chunk in iter(lambda: _f.read(1 << 20), b''):
        _digest.update(_chunk)
with open(paths.DATA / 'embeddings.meta.json', 'w', encoding='utf-8') as _f:
    json.dump({'corpus_sha256': _digest.hexdigest(),
               'rows': int(np.array(embeddings).shape[0]),
               'source': 'ocpq_dataset.json'}, _f, indent=2)
print(f"Wrote embeddings.meta.json (corpus sha256 {_digest.hexdigest()[:12]}...)")

# Alternatively, to save as a JSON file:
# with open('embeddings.json', 'w') as f:
#     json.dump(embeddings, f)

