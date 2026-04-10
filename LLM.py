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

from ocpq_schema import OCPQ_JSON_SCHEMA

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
hf_model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
hf_provider = os.getenv("HF_PROVIDER")

def set_hf_client(token, provider=None):
    global hf_client
    selected_provider = provider or hf_provider
    if selected_provider:
        hf_client = InferenceClient(api_key=token, provider=selected_provider)
    else:
        hf_client = InferenceClient(api_key=token)

def set_hf_model(model_name):
    global hf_model
    hf_model = model_name

def set_hf_provider(provider_name):
    global hf_provider
    hf_provider = provider_name

# In[3]:

def validate_query(q):
    try:
        assert "nodes" in q
        for node in q["nodes"]:
            box = node["Box"]
            assert isinstance(box, list) and len(box) == 2
            assert "newEventVars" in box[0]
        return True
    except:
        return False

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
    # Target json code blocks specifically
    match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
    if not match:
        match = re.search(r'\{.*\}', content, re.DOTALL)
        
    clean_content = match.group(1) if match.lastindex else (match.group(0) if match else content.strip())
    
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
    if hf_client is None:
        raise RuntimeError(
            "HF client is not initialized. Call set_hf_client(token, provider=...) first."
        )

    for attempt in range(max_retries):
        try:
            return hf_client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                response_format=response_format
            )
        except Exception as e:
            err_text = str(e)
            unsupported_model = (
                "model_not_supported" in err_text
                or "not supported by any provider" in err_text
                or "not a chat model" in err_text
            )

            if unsupported_model:
                raise RuntimeError(
                    "Selected model/provider is not usable for chat_completion. "
                    f"model='{model}', provider='{hf_provider or 'auto'}'. "
                    "Pick a chat-capable model that is available for your enabled provider, "
                    "or set a different provider in set_hf_client(token, provider=...) / HF_PROVIDER."
                ) from e

            if "429" in str(e):
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"Rate limited (429). Retrying in {wait_time} seconds (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    raise e
            else:
                raise e

# Step 1: Extract structure from NL query
def structure_extraction(user_query: str, log_file, model_name=None):
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
    selected_model = model_name or hf_model
    response = call_hf_with_retry(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 1.5: Formalize the extracted structure into a more precise logical representation 
# that can be used for tree generation and final OCPQ JSON generation. 
# This is an optional intermediate step that can help improve the quality 
# of the final output by providing a clearer representation of the query logic.
def structure_formalization(structure, model_name=None):
    prompt = f"""
You are formalizing a process query.

Given this extracted structure:

{structure}

Convert it into a precise logical representation using simple rules:

- Use variables like e1, e2 for events
- Use variables like o1 for objects
- Express relations explicitly:
  - before(e1, e2)
  - exists(e1)
  - count(e1) = 1
- Link events to objects:
  - relates(e1, o1)

Return ONLY JSON:

{{
  "variables": {{
    "events": {{...}},
    "objects": {{...}}
  }},
  "logic": [
    ...
  ]
}}
"""
    selected_model = model_name or hf_model
    response = call_hf_with_retry(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 2: Generate a query tree (logical plan) based on the extracted structure
def query_tree_generation(structure, model_name=None):
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
    selected_model = model_name or hf_model
    response = call_hf_with_retry(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 3: Generate the final OCPQ JSON using RAG with retrieved examples for better grounding and adherence to schema
def nl_to_ocpq_rag(query, dataset, embeddings, json_tree, structure, query_types=None, model_name=None):
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
The constraints in your JSON MUST ONLY use the following top-level variants:
"Filter", "SizeFilter", "SAT", "ANY", "NOT", "OR", "AND".
Basic filter logic like "TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter", "ObjectAttributeValueFilter", "BasicFilterCEL" must be wrapped inside a {{ "type": "Filter", "filter": {{ ... }} }} constraint.
Do NOT place "O2E" or "O2O" directly in constraints. They belong in the "filters" array of the node.
If you use "NumChilds" or other size filters in the "sizeFilters" array, place them directly (e.g., {{ "type": "NumChilds", ... }}). Do NOT wrap them in a "SizeFilter" object unless you are placing them inside the "constraints" array.

Here is the logical plan (query tree) for the query.
LOGICAL PLAN (DO NOT USE THIS AS THE OUTPUT SCHEMA, JUST USE IT TO UNDERSTAND THE LOGIC AND HIERARCHY):
{json_tree}

EXTRACTED STRUCTURE (USE THESE EXACT EVENTS AND OBJECTS):
{json.dumps(structure, indent=2)}

WARNING: The examples below are from different domains. DO NOT copy their events (like "submit application") or objects (like "candidates"). You MUST map the logic from the examples to the events and objects in the EXTRACTED STRUCTURE above!

Use these examples for the EXACT output schema:
{few_shot_string}

Now generate the final OCPQ JSON for the exact user query: '{query}'
ONLY output valid JSON.
"""

    selected_model = model_name or hf_model
    response = call_hf_with_retry(
        model=selected_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 4: Verify the generated OCPQ JSON and ask the LLM to fix it if it's invalid or doesn't follow the schema. 
# This is a safety net to improve the quality of the output before executing the query.
def verify_ocpq_json(ocpq_json):
    is_valid = validate_query(ocpq_json)
    if not is_valid:
        print("Generated OCPQ JSON is invalid. Asking LLM to fix it...")
        prompt = f"""
The following JSON is invalid or incomplete.

Fix it so that:
- It follows OCPQ schema
- It correctly represents the query

Return ONLY fixed JSON.

JSON:
{ocpq_json}
"""
        response = call_hf_with_retry(
            model=hf_model,
            messages=[
                {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return parse_llm_json(content)
    else:
        print("Generated OCPQ JSON is valid.")
        return ocpq_json

# Step 5: (Optional) Perform semantic validation of the generated OCPQ JSON against the original query to ensure it captures the intent correctly.
# This can be done by asking the LLM to explain the generated JSON and how it maps to the original query, and to identify any discrepancies or missing elements.
def semantic_validation(ocpq_json, query):
    prompt = f"""
You are verifying an OCPQ query.

User query:
{query}

Generated OCPQ JSON:
{ocpq_json}

Explain whether the query correctly captures the intent.

If NOT:
- explain the mistake
- output a corrected JSON

Return JSON:
{{
  "is_correct": true/false,
  "issues": [...],
  "fixed_query": {{...}}
}}
"""
    response = call_hf_with_retry(
        model=hf_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 6: (Optional) If the query execution fails due to errors in the OCPQ JSON, ask the LLM to fix the JSON based on the error message and try executing again. 
# This creates a feedback loop that can help refine the query iteratively until it works correctly.
def fix_query(ocpq_json, error_message):
    prompt = f"""
The following OCPQ JSON query caused an error when executed:
{ocpq_json}

The error message was:
{error_message}

Please fix the query so that it is valid and can be executed without errors.

Return ONLY the fixed JSON.
"""
    response = call_hf_with_retry(
        model=hf_model,
        messages=[
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    return parse_llm_json(content)

# Step 1: Use a single continuous chat context with Least-to-Most Prompting
def nl_to_ocpq_pipeline(query, dataset, embeddings, output_file="test_query.json", ocel_path="../hospital_log_high.json", model_name=None):
    print("Starting Least-to-Most NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")
    
    # Extract event and object names from the OCEL log
    event_names, object_names = events_and_objects_from_ocel(ocel_path)

    # Initialize the chat messages for context
    messages = [
        {"role": "system", "content": "You are a process mining expert that converts natural language into Object-Centric Process Queries (OCPQ). You must think step by step and respond concisely according to the prompts."}
    ]

    # Sub-problem 1: Extract entities mapping
    prompt_1 = f"Given the user query, map the terms to the exact events and objects involved.\n\nCRITICAL: You MUST translate colloquial terms (e.g. 'person', 'diagnosis', 'registration') directly to the EXACT valid names provided below. DO NOT use names that aren't in these lists.\n\nValid events: {event_names}\nValid objects: {object_names}\n\nQuery: {query}\n\nOutput a clear mapping of the elements found in the query to the valid events and objects. Also identify any constraints (like 'completed within TIME', 'exactly once', 'at least once'). Do not output JSON."
    messages.append({"role": "user", "content": prompt_1})
    
    response_1 = call_hf_with_retry(model=selected_model, messages=messages, max_tokens=1000, response_format=None)
    structure_text = response_1.choices[0].message.content
    messages.append({"role": "assistant", "content": structure_text})
    print("Sub-problem 1 completed: Entities extracted.")
    time.sleep(3)

    # Sub-problem 2: Plan tree topology
    prompt_2 = "Based on the mapped elements, plan the logical tree structure for the OCPQ query.\n\nOCPQ PATTERN CHEAT SHEET:\n- 'completed within TIME': Requires 2 nodes. Parent node has Event A, Object A, and a constraint specifying 'SAT' on the child path. Child node has Event B and a 'TimeBetweenEvents' filter (max_seconds).\n- 'exactly once': Single node with Event A, Object A, and a SizeFilter on the child path (min:1, max:1).\n- 'at least once': Parent node with Event A, child node with Event B.\n- 'must not occur': Single node with a 'NOT' constraint on a child path.\n\nDescribe the exact tree structure (number of nodes, what goes in them, parent-child relationships, and exact filters/constraints) in plain text. Do not output JSON."
    messages.append({"role": "user", "content": prompt_2})
    
    response_2 = call_hf_with_retry(model=selected_model, messages=messages, max_tokens=1000, response_format=None)
    tree_text = response_2.choices[0].message.content
    messages.append({"role": "assistant", "content": tree_text})
    print("Sub-problem 2 completed: Tree structured planned.")
    time.sleep(3)

    # Sub-problem 3: Generate Final JSON
    # Retrieve relevant examples that have reasoning
    examples = retrieve_examples(query, dataset, embeddings)
    example_prompts = []
    for ex in examples:
        qt = "nl"
        if qt in ex:
            reasoning = ex.get('reasoning', 'Step 1: Identify elements. Step 2: Establish logic. Step 3: Implement JSON.')
            example_prompts.append(
                f"Question: {ex[qt]}\nReasoning: {reasoning}\nJSON:\n{json.dumps(ex['query'], indent=2)}"
            )
    few_shot_string = "\n\n".join(example_prompts)

    prompt_3 = f"Now, based on your previous reasoning, output the final OCPQ JSON matching the Exact Schema.\n\nUse these examples as references for the output syntax. DO NOT copy their logic unless it applies to this specific user query:\n{few_shot_string}\n\nOutput strictly and ONLY valid JSON matching the exact schema."
    messages.append({"role": "user", "content": prompt_3})
    messages[0] = {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided schema."} 
    
    response_3 = call_hf_with_retry(
        model=selected_model, 
        messages=messages, 
        max_tokens=2000, 
        response_format={
            "type": "json_schema", 
            "json_schema": {"name": "ocpq_schema", "schema": OCPQ_JSON_SCHEMA, "strict": True}
        }
    )
    content = response_3.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    messages.append({"role": "assistant", "content": content})
    print("Sub-problem 3 completed: OCPQ JSON generated.")
    time.sleep(3)

    ocpq_json = verify_ocpq_json(ocpq_json)
    
    # Execution validation unchanged
    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_stderr = None
    query_result_file_path = None
    
    try:
        from QueryExecutor import run_ocpq_query
        query_result, query_stderr, query_result_file_path = run_ocpq_query(ocel_path, output_file)
    except Exception as e:
        query_result = f"Error running query automatically: {e}"
        print(query_result)
    
    # Auto-fix using the chat history context block
    # Only try to fix if an actual execution error (stderr from a crash/panic) occurred
    if query_stderr and not query_stderr.strip().startswith("warning:"):
        print("Query execution error output:\n", query_stderr)
        
        prompt_4 = f"The generated JSON query caused this execution error:\n{query_stderr}\n\nPlease fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
        messages.append({"role": "user", "content": prompt_4})
        
        fix_response = call_hf_with_retry(model=selected_model, messages=messages, max_tokens=2000, response_format=None)
        fixed_content = fix_response.choices[0].message.content
        fixed_json = parse_llm_json(fixed_content)

        if fixed_json != ocpq_json:
            print("Attempting to re-run fixed query...")
            with open(output_file, "w") as f:
                json.dump(fixed_json, f, indent=2)
            try:
                query_result, query_stderr, query_result_file_path = run_ocpq_query(ocel_path, output_file)
            except Exception as e:
                query_result = f"Error running fixed query automatically: {e}"
                print(query_result)
            ocpq_json = fixed_json

    return structure_text, tree_text, ocpq_json, query_result, query_result_file_path
