#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import os
import time

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from huggingface_hub import InferenceClient

from LogEvents import events_and_objects_from_ocel

# In[2]:

def load_resources():
    # Load the dataset
    with open('ocpq_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Load the generated embeddings
    embeddings = np.load('embeddings.npy')

    # Load the local embedding model so we can embed user queries on the fly
    model = SentenceTransformer('./local-all-MiniLM-L6-v2') 
    
    return dataset, embeddings, model

dataset, embeddings, model = load_resources()

def get_embedding(text):
    return model.encode(text)

hf_client = None

def set_hf_client(token):
    global hf_client
    hf_client = InferenceClient(api_key=token)

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

# Commented out the stray retrieval call that automatically executes on import
# retrieve_examples("For any person, the diagnosis must be completed within 3 days after the registration at ER.", dataset, embeddings)

# In[5]:

# from LogEvents import events_and_objects_from_ocel
# event_names, object_names = events_and_objects_from_ocel("../hospital_log_high.json")

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

def call_hf_with_retry(model, messages, max_tokens, response_format, max_retries=5):
    for attempt in range(max_retries):
        try:
            return hf_client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format
            )
        except Exception as e:
            if "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited (429). Retrying in {wait_time} seconds (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
            else:
                raise e

def structure_extraction(user_query: str, log_file):
    # Extract event and object names from the OCEL log to provide as context for the LLM
    event_names, object_names = events_and_objects_from_ocel(log_file)

    prompt = f"""
You are extracting the structure of a process query.

Given a natural language query, identify:

1. Events involved
2. Objects involved
3. Relationships between events (before, after, etc.)
4. Cardinality constraints (e.g., exactly once, at least once)
5. Whether nesting is needed (yes/no)

The events and objects must be chosen from the following lists extracted from the log:
Events: {event_names}
Objects: {object_names}

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
    response = call_hf_with_retry(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
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
    response = call_hf_with_retry(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
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

CONSTRAINT SCHEMA: 
The constraints in your JSON MUST ONLY use the following variants:
"O2E", "O2O", "TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter", "ObjectAttributeValueFilter", "BasicFilterCEL".
Do NOT use variants like "TimeAfterEvent" or anything else not in this list.

Here is the logical plan (query tree) for the query.
LOGICAL PLAN (DO NOT USE THIS AS THE OUTPUT SCHEMA, JUST USE IT TO UNDERSTAND THE LOGIC AND HIERARCHY):
{json_tree}

Use these examples for the EXACT output schema:
{few_shot_string}

Now generate the final OCPQ JSON.
ONLY output valid JSON.
"""

    # Upgrading to a more powerful model for better schema adherence
    response = call_hf_with_retry(
        model="meta-llama/Meta-Llama-3-8B-Instruct", 
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
    print("Starting NL to OCPQ pipeline...")
    structure = structure_extraction(query, ocel_path)
    print("Structure extraction completed.")
    print("Tree generation in progress...")
    time.sleep(3)
    json_tree = query_tree_generation(structure)
    print("Tree generation completed.")
    print("Generating final OCPQ JSON...")
    time.sleep(3)
    ocpq_json = nl_to_ocpq_rag(query, dataset, embeddings, json_tree)
    print("OCPQ JSON generation completed.")
    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)
        
    query_result = None
    query_result_file_path = None
    try:
        from QueryExecutor import run_ocpq_query
        query_result, query_result_file_path = run_ocpq_query(ocel_path, output_file)
    except Exception as e:
        query_result = f"Error running query automatically: {e}"
        print(query_result)
        
    return structure, json_tree, ocpq_json, query_result, query_result_file_path
