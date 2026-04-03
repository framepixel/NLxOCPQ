#!/usr/bin/env python
# coding: utf-8

# In[1]:


import openai

import json
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from huggingface_hub import InferenceClient

# In[2]:


# Load the dataset
with open('ocpq_dataset.json', 'r') as f:
    dataset = json.load(f)

# Load the generated embeddings
embeddings = np.load('embeddings.npy')

# Load the local embedding model so we can embed user queries on the fly
model = SentenceTransformer('./local-all-MiniLM-L6-v2') 

def get_embedding(text):
    return model.encode(text)

# Initialize Hugging Face Inference Client for generation
hf_client = InferenceClient(api_key=os.getenv("HF_TOKEN"))

# In[3]:


def retrieve_examples(query, dataset, embeddings, top_k=3):
    """
    Retrieve the top-k most similar examples from the dataset based on cosine similarity of embeddings.
    """
    query_emb = get_embedding(query)
    
    # Calculate cosine similarity between the query and all dataset questions
    similarities = cosine_similarity([query_emb], embeddings)[0]
    
    # Get the indices of the top-k most similar questions
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    examples = []
    for idx in top_indices:
        examples.append({
            "nl": dataset[idx]["nl"],
            "nl_clean_1": dataset[idx]["nl_clean_1"] if "nl_clean_1" in dataset[idx] else None,
            "nl_clean_2": dataset[idx]["nl_clean_2"] if "nl_clean_2" in dataset[idx] else None,
            "query": dataset[idx]["query"]
        })
    
    return examples

# In[4]:


retrieve_examples("For any person, the diagnosis must be completed within 3 days after the registration at ER.", dataset, embeddings)

# In[5]:


from LogEvents import events_and_objects_from_ocel

event_names, object_names = events_and_objects_from_ocel("../hospital_log_high.json")

# In[6]:


import re
import ast

def parse_llm_json(content):
    # Extract json chunk
    match = re.search(r'\{.*\}', content, re.DOTALL)
    clean_content = match.group(0) if match else content.strip()
    
    # Auto-heal truncated JSON
    if not clean_content.endswith("}"):
        clean_content += "\n}"
        
    try:
        return json.loads(clean_content)
    except json.JSONDecodeError:
        try:
            # Fallback: convert python-like string (with single quotes) to Python dict
            py_content = clean_content.replace("null", "None").replace("true", "True").replace("false", "False")
            return ast.literal_eval(py_content)
        except Exception as e:
            print(f"Failed to decode JSON. Error: {e}\nRaw output:\n{content}")
            return None

# In[7]:

def structure_extraction(user_query: str):
    prompt = f"""
You are extracting the structure of a process query.

Given a natural language query, identify:

1. Events involved
2. Objects involved
3. Relationships between events (before, after, etc.)
4. Cardinality constraints (e.g., exactly once, at least once)
5. Whether nesting is needed (yes/no)

Return ONLY JSON in this format:

{{
  "events": [...],
  "objects": [...],
  "relations": [...],
  "constraints": [...],
  "requires_nesting": true/false
}}

Query:
{user_query}
"""
    # Upgrading to a more powerful model for better schema adherence
    response = hf_client.chat_completion(
        model="meta-llama/Meta-Llama-3-70B-Instruct", 
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

def query_tree_generation(structure):
    prompt = f"""
You are designing an OCPQ query tree.

Given this extracted structure:

{structure}

Decide:

- Number of nodes
- What each node represents
- Parent-child relationships

Return ONLY JSON:

{{
  "nodes": [
    {{
      "id": 0,
      "description": "...",
      "events": [...],
      "objects": [...]
    }}
  ],
  "edges": [
    {{"from": 0, "to": 1, "label": "A"}}
  ]
}}
"""
    # Upgrading to a more powerful model for better schema adherence
    response = hf_client.chat_completion(
        model="meta-llama/Meta-Llama-3-70B-Instruct", 
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)


def nl_to_ocpq_rag(query, dataset, embeddings, json_tree, query_types=None):
    if query_types is None:
        query_types = ["nl"]
        
    # Retrieve relevant examples
    examples = retrieve_examples(query, dataset, embeddings)
    
    # Construct a prompt with few-shot examples
    example_prompts = []
    for ex in examples:
        for qt in query_types:
            if qt in ex:
                example_prompts.append(
                    f"Question: {ex[qt]}\nJSON: {json.dumps(ex['query'], indent=2)}"
                )
    few_shot_string = "\n\n".join(example_prompts)
    
    prompt = f"""
You are generating OCPQ JSON.

CRITICAL SCHEMA REQUIREMENT:
Output ONLY valid JSON matching the exact schema shown in the examples.
The root object should minimally contain an array of "nodes".
Each node in the "nodes" array MUST be an object with a single key "Box" (and optionally "[", "]" keys depending on the syntax).
The "Box" value MUST always be an array with exactly TWO elements:
1. The node data dictionary (containing newEventVars, newObjectVars, filters, sizeFilters, constraints, evVarLabels, obVarLabels).
2. An array of child node indices (e.g., [], [1]).

Here is the logical plan (query tree) for the query.
LOGICAL PLAN (DO NOT USE THIS AS THE OUTPUT SCHEMA, JUST USE IT TO UNDERSTAND THE LOGIC AND HIERARCHY):
{json_tree}

Use these examples for the EXACT output schema:
{few_shot_string}

Now generate the final OCPQ JSON.
ONLY output valid JSON.
"""

    # Upgrading to a more powerful model for better schema adherence
    response = hf_client.chat_completion(
        model="meta-llama/Meta-Llama-3-70B-Instruct", 
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

def nl_to_ocpq_pipeline(query, dataset, embeddings, output_file="test_query.json", ocel_path="../hospital_log_high.json"):
    structure = structure_extraction(query)
    print("Extracted Structure:", json.dumps(structure, indent=2))
    json_tree = query_tree_generation(structure)
    print("Generated Query Tree:", json.dumps(json_tree, indent=2))
    ocpq_json = nl_to_ocpq_rag(query, dataset, embeddings, json_tree)
    print("Final OCPQ JSON:", json.dumps(ocpq_json, indent=2))
    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)
        
    try:
        from QueryExecutor import run_ocpq_query
        run_ocpq_query(ocel_path, output_file)
    except Exception as e:
        print(f"Error running query automatically: {e}")
        
    return ocpq_json

# In[8]:


result = nl_to_ocpq_pipeline("For any person, the diagnosis must be completed within 3 days after the registration at ER.", dataset, embeddings)
# print(json.dumps(result, indent=2))

# In[9]:


exit()

# In[ ]:


result = nl_to_ocpq_rag("write me a poem about shipping order", dataset, embeddings)
print(json.dumps(result, indent=2))

# In[ ]:


result = nl_to_ocpq_rag("no payment reminder should be sent out after payment has been made", dataset, embeddings)
print(json.dumps(result, indent=2))

# In[ ]:


def nl_to_ocpq(query):

    prompt = f"""
Translate natural language process queries into OCPQ JSON.
Output ONLY valid JSON. You MUST use double quotes for all keys and strings.

CRITICAL SCHEMA REQUIREMENT:
The "Box" key MUST always contain an array with exactly TWO elements:
1. The node data dictionary (containing newEventVars, newObjectVars, filters, etc.)
2. An array of child node indices (e.g., [], [1], or [1, 2])
Example: "Box": [ {{ "newEventVars":... }}, [] ]

Question: {query}
JSON:
"""

    response = hf_client.chat_completion(
        model="meta-llama/Meta-Llama-3-70B-Instruct",
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        response_format={"type": "json_object"} 
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)
