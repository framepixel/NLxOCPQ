#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import numpy as np
import os
import time
import traceback
from types import SimpleNamespace

from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModel, AutoTokenizer

from openai import OpenAI
from huggingface_hub import InferenceClient
from pydantic import ValidationError

from ocpq_schema import OCPQ_JSON_SCHEMA, BindingBoxTree, VERIFICATION_JSON_SCHEMA, EXTRACTED_STRUCTURE_SCHEMA, QUERY_TREE_SCHEMA

from LogEvents import events_and_objects_from_ocel, extract_ocel_schema

# In[2]:

def load_resources():
    # Load the dataset
    with open('ocpq_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Load the generated embeddings
    embeddings = np.load('embeddings.npy')

    # Load the local embedding model so we can embed user queries on the fly.
    embedding_error = None
    tokenizer = None
    model = None
    try:
        local_dir = './local-all-MiniLM-L6-v2'
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        source_dir = local_dir if os.path.isdir(local_dir) else model_name
        tokenizer = AutoTokenizer.from_pretrained(source_dir)
        model = AutoModel.from_pretrained(source_dir)
    except Exception as exc:
        embedding_error = str(exc)
        print(
            "Warning: the embedding model could not be loaded. "
            "Falling back to non-embedding example retrieval. "
            f"Reason: {embedding_error}"
        )
    
    return dataset, embeddings, tokenizer, model, embedding_error

dataset, embeddings, tokenizer, model, embedding_load_error = load_resources()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    if model is None or tokenizer is None:
        raise RuntimeError(
            "Embedding model is unavailable. "
            f"embedding load error: {embedding_load_error}"
        )
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**inputs)
    embedding = mean_pooling(model_output, inputs['attention_mask'])
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.squeeze(0).cpu().numpy()

hf_client = None
openai_client = None
hf_model = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
hf_provider = os.getenv("HF_PROVIDER")
hf_base_url = os.getenv("HF_BASE_URL")
hf_backend = os.getenv("HF_BACKEND", "huggingface")
hf_token = os.getenv("HF_TOKEN")
failure_log_path = os.getenv("OCPQ_FAILURE_LOG_PATH", "ocpq_failure_log.jsonl")
few_shot_enabled = True
# Number of retrieved examples per few-shot prompt. 3 is the value every recorded run used;
# it is exposed so the "do more examples help?" comparison can be run without touching call sites.
DEFAULT_FEW_SHOT_K = 3
few_shot_k = DEFAULT_FEW_SHOT_K
generation_temperature = None

LLM_CALL_STATS = []
QUERY_EXEC_STATS = []

local_model = None
local_tokenizer = None
local_model_key = None
local_gpu_device = "cuda"
local_gpu_dtype = "auto"
local_gpu_quantization = "4bit"

def set_hf_client(
    token=None,
    provider=None,
    model_name=None,
    base_url=None,
    backend_mode=None,
    local_device="cuda",
    local_dtype="auto",
    local_quantization="4bit",
):
    global hf_client
    global openai_client
    global hf_model
    global hf_provider
    global hf_base_url
    global hf_backend
    global hf_token
    global local_model
    global local_tokenizer
    global local_model_key
    global local_gpu_device
    global local_gpu_dtype
    global local_gpu_quantization

    selected_provider = provider or hf_provider
    selected_model = model_name or hf_model
    selected_base_url = base_url or hf_base_url
    selected_backend = (backend_mode or "").strip().lower() if backend_mode else None

    hf_model = selected_model
    hf_provider = selected_provider
    hf_base_url = selected_base_url
    hf_token = token or hf_token

    if selected_backend == "local-gpu":
        hf_backend = "local-gpu"
        hf_client = None
        openai_client = None
        hf_provider = None
        hf_base_url = None
        new_device = local_device or "cuda"
        new_dtype = local_dtype or "auto"
        new_quantization = (local_quantization or "4bit").strip().lower()
        local_gpu_device = new_device
        local_gpu_dtype = new_dtype
        local_gpu_quantization = new_quantization
        # Only evict the cached model when the configuration actually changes.
        # Without this guard, calling set_local_gpu_client() in every eval loop
        # iteration resets local_model to None each case, forcing a full reload.
        new_key = (hf_model, hf_token, new_device, new_dtype, new_quantization)
        if local_model_key != new_key:
            local_model = None
            local_tokenizer = None
            local_model_key = None
        return

    if selected_base_url:
        hf_backend = "custom-endpoint"
        openai_client = OpenAI(api_key=token or "EMPTY", base_url=selected_base_url)
        hf_client = None
        return

    openai_client = None
    if selected_provider:
        hf_backend = "hf-provider"
        hf_client = InferenceClient(api_key=token or None, provider=selected_provider)
    else:
        hf_backend = "hf-default"
        hf_client = InferenceClient(api_key=token or None)


def set_few_shot_enabled(enabled=True):
    global few_shot_enabled
    few_shot_enabled = bool(enabled)


def set_few_shot_k(value=None):
    """Set how many examples few-shot retrieval returns.

    Args:
        value: Number of examples, or None to restore the default of 3.
    """
    global few_shot_k
    if value is None:
        few_shot_k = DEFAULT_FEW_SHOT_K
        return few_shot_k
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"few_shot_k must be >= 0, got {parsed}")
    few_shot_k = parsed
    return few_shot_k


def get_few_shot_k():
    return few_shot_k


def set_failure_log_path(path=None):
    """Point the failure log at a specific file, or restore the default.

    The default is relative to the working directory, and `submit_eval_array.sh` `cd`s once
    to `$SLURM_SUBMIT_DIR` and never leaves it — so all 24 concurrently running array tasks
    would append to one file. The payloads embed full prompt messages and exceed the write
    buffer, so records get split across `write()` calls and interleave, corrupting exactly
    the log you would want to read afterwards. `eval_testset.py` therefore points this at
    its own run directory, giving every file a single writer.

    Args:
        path: Destination file, or None to restore `OCPQ_FAILURE_LOG_PATH` / the default.

    Returns:
        The path now in effect.
    """
    global failure_log_path
    if path is None:
        failure_log_path = os.getenv("OCPQ_FAILURE_LOG_PATH", "ocpq_failure_log.jsonl")
    else:
        failure_log_path = str(path)
    return failure_log_path


def get_failure_log_path():
    return failure_log_path


def set_generation_temperature(value):
    global generation_temperature
    if value is None:
        generation_temperature = None
        return
    generation_temperature = float(value)


def get_generation_temperature():
    return generation_temperature


def reset_llm_call_stats():
    LLM_CALL_STATS.clear()


def get_llm_call_stats():
    return list(LLM_CALL_STATS)


def reset_query_exec_stats():
    QUERY_EXEC_STATS.clear()


def get_query_exec_stats():
    return list(QUERY_EXEC_STATS)


def get_few_shot_enabled():
    return few_shot_enabled


def get_hf_backend_description():
    if hf_backend == "local-gpu":
        quant = f", {local_gpu_quantization}" if local_gpu_quantization else ""
        return f"Local GPU ({hf_model}{quant})"
    if hf_backend == "custom-endpoint":
        return f"Custom endpoint ({hf_base_url or 'unset'})"
    if hf_backend == "hf-provider":
        return f"HF provider ({hf_provider or 'auto'})"
    return "HF default"


def set_local_gpu_client(model_name=None, token=None, device="cuda", dtype="auto", quantization="4bit"):
    set_hf_client(
        token=token,
        model_name=model_name,
        backend_mode="local-gpu",
        local_device=device,
        local_dtype=dtype,
        local_quantization=quantization,
    )

def set_hf_model(model_name):
    global hf_model
    hf_model = model_name

def set_hf_provider(provider_name):
    global hf_provider
    hf_provider = provider_name


PROMPT_STRATEGIES = [
    "schema-grammar-guided",
    "schema-allowed-entities-only",
    "schema-few-shot",
    "schema-chain-of-thought-plus",
    "schema-least-to-most",
]
DEFAULT_PROMPT_STRATEGY = os.getenv("OCPQ_PROMPT_STRATEGY", "schema-grammar-guided")


# --- Schema-richness levels (grounding ablation) -------------------------------------
# How much of the OCEL schema is placed in the prompt. This is an independent variable:
# it measures how much of the NL->OCPQ difficulty is *grounding* versus reasoning/syntax.
#
#   entities            event + object type names only (historic behaviour)
#   entities_attributes + declared object attribute names/types. This is the format the
#                       fine-tuning prompts were built with, so it also repairs a
#                       train/inference mismatch: every training prompt carried an
#                       OBJECT ATTRIBUTES block, but inference previously sent none,
#                       leaving the model to invent names like `is_Married`.
#   full                + the type-level E2O and O2O relation graphs, so the model can
#                       see which events touch which objects and route through an
#                       intermediary when no direct relation exists.
SCHEMA_LEVELS = ["entities", "entities_attributes", "full"]
DEFAULT_SCHEMA_LEVEL = os.getenv("OCPQ_SCHEMA_LEVEL", "full")
schema_level = DEFAULT_SCHEMA_LEVEL

# OCEL declares attribute types lowercase ("integer"); OCPQ value_filter discriminators
# and the fine-tuning prompts use capitalised names ("Integer"). Normalise so the prompt
# shows the model exactly the token it must emit in `value_filter.type`.
_ATTR_TYPE_DISPLAY = {
    "integer": "Integer",
    "float": "Float",
    "double": "Float",
    "boolean": "Boolean",
    "bool": "Boolean",
    "string": "String",
    "time": "Time",
    "date": "Time",
}

# Upper bound on the rendered schema block, in estimated tokens. Deliberately ONE value for
# every model: giving a 256k-context model more schema than an 8k one would mean the models
# see different prompts, which confounds the model-vs-model comparison the thesis rests on.
# Sized for the smallest context in the model set, leaving room for the rules block, the
# few-shot examples, and the reserved generation budget. Nothing else caps the prompt —
# the local backend tokenizes without `max_length` — so this is the only guard.
SCHEMA_BLOCK_TOKEN_BUDGET = int(os.getenv("OCPQ_SCHEMA_TOKEN_BUDGET", "3500"))

# Schema text is name-heavy, so it tokenizes denser than prose. Used only to keep the block
# under budget, never to report a metric, so an approximation is sufficient and avoids
# loading a tokenizer on a function called once per evaluation case.
_CHARS_PER_TOKEN = 3.6

# Records why a section was left out, for the run manifest. Reset per render.
_last_schema_omissions: list = []

# Context-overflow warnings are emitted once per (model, rough size) so a systematic
# overflow is reported without flooding 5400 runs' worth of logs.
_context_warned: set = set()


def _estimated_tokens(text):
    return int(len(text) / _CHARS_PER_TOKEN)


def get_last_schema_omissions():
    """Sections dropped from the most recent `build_schema_block` call, if any."""
    return list(_last_schema_omissions)


def _render_attribute(attr):
    """Render one attribute as the fine-tuning prompts do.

    `convert_dataset_for_finetuning.py` emits `name (Type)` and, for low-cardinality string
    attributes, `name (Type: v1, v2)`. Evaluation previously emitted only the former, so a
    model taught to read the literal `"0"` off `isCitizen (String: 0, 1)` was left guessing
    between `"0"`, `"false"` and `"No"`. Matching the training format closes that gap.
    """
    display = _ATTR_TYPE_DISPLAY.get(str(attr.get("type", "")).lower(), "String")
    values = attr.get("values")
    if values:
        return f"{attr['name']} ({display}: {', '.join(str(v) for v in values)})"
    return f"{attr['name']} ({display})"


def _render_attribute_section(header, types, attributes_by_type):
    """Render one `X ATTRIBUTES:` section, or [] when nothing has attributes."""
    lines = []
    for type_name in types:
        attrs = (attributes_by_type or {}).get(type_name) or []
        rendered = ", ".join(_render_attribute(a) for a in attrs if a.get("name"))
        if rendered:
            lines.append(f"  {type_name}: {rendered}")
    return ([header] + lines) if lines else []


def normalize_schema_level(level):
    """Return a valid schema level, falling back to the default."""
    if not level:
        return DEFAULT_SCHEMA_LEVEL
    normalized = str(level).strip().lower().replace("-", "_")
    aliases = {
        "entities": "entities",
        "entities_only": "entities",
        "entities_attributes": "entities_attributes",
        "attributes": "entities_attributes",
        "full": "full",
        "relations": "full",
    }
    return aliases.get(normalized, DEFAULT_SCHEMA_LEVEL)


def set_schema_level(level):
    """Set the active schema-richness level for subsequent pipeline calls."""
    global schema_level
    schema_level = normalize_schema_level(level)
    return schema_level


def build_schema_block(schema, level=None):
    """Render the OCEL schema for the prompt at the requested richness level.

    The ALLOWED EVENTS / ALLOWED OBJECTS / OBJECT ATTRIBUTES sections reproduce the
    format used in the fine-tuning prompts — including the `(String: v1, v2)` value
    enumeration — so the fine-tuned models see at inference what they saw in training.

    The block is capped at `SCHEMA_BLOCK_TOKEN_BUDGET`; sections that do not fit are
    dropped whole and reported by `get_last_schema_omissions()`. Two deliberate
    differences from the training prompt remain and are documented in
    `Thesis/notes/11-evaluation-readiness.md`: the relation graphs and the event-attribute
    section appear here but in no training prompt, because `ocpq_dataset.json` carries
    neither.

    Args:
        schema: Output of LogEvents.extract_ocel_schema().
        level: One of SCHEMA_LEVELS; defaults to the module-level `schema_level`.

    Returns:
        The schema text block to embed in a prompt.
    """
    global _last_schema_omissions
    _last_schema_omissions = []
    level = normalize_schema_level(level or schema_level)

    event_types = schema.get("event_types", [])
    object_types = schema.get("object_types", [])
    lines = [
        f"ALLOWED EVENTS: {json.dumps(event_types)}",
        f"ALLOWED OBJECTS: {json.dumps(object_types)}",
    ]

    if level == "entities":
        return "\n".join(lines)

    def _add(section, name):
        """Append a section unless it would push the block past the budget.

        Sections are dropped whole rather than truncated: half a relation list or half an
        attribute list would read as a complete one, and the model would confidently treat
        the missing entries as nonexistent. An omission is recorded instead.
        """
        if not section:
            return
        candidate = "\n".join(lines + section)
        if _estimated_tokens(candidate) > SCHEMA_BLOCK_TOKEN_BUDGET:
            _last_schema_omissions.append({
                "section": name,
                "estimated_tokens": _estimated_tokens("\n".join(section)),
                "budget": SCHEMA_BLOCK_TOKEN_BUDGET,
            })
            return
        lines.extend(section)

    _add(_render_attribute_section("OBJECT ATTRIBUTES:", object_types,
                                   schema.get("object_attributes")), "object_attributes")

    if level == "entities_attributes":
        return "\n".join(lines)

    # level == "full": add the relation graphs.
    def _render_relations(triples):
        rendered = []
        for source, target, qualifier in triples:
            suffix = f" ({qualifier})" if qualifier else ""
            rendered.append(f"  {source} -> {target}{suffix}")
        return rendered

    e2o = _render_relations(schema.get("e2o", []) or [])
    if e2o:
        _add(["EVENT-TO-OBJECT RELATIONS (only these event/object pairs are linked):"] + e2o,
             "e2o_relations")

    o2o = _render_relations(schema.get("o2o", []) or [])
    if o2o:
        _add(["OBJECT-TO-OBJECT RELATIONS (only these object pairs are linked):"] + o2o,
             "o2o_relations")
    lines.append(
        "Use ONLY the relations listed above. If two entities are not directly related, "
        "connect them through an intermediary that is."
    )

    # Event attributes come last because they are the least often needed and by far the
    # largest: bpic2017 declares 494 of them (~15 000 tokens, more than Llama-3-8B's entire
    # context) and angular 277. Budget-gated, so those two logs simply go without.
    _add(_render_attribute_section("EVENT ATTRIBUTES:", event_types,
                                   schema.get("event_attributes")), "event_attributes")

    return "\n".join(lines)


def normalize_prompt_strategy(prompt_strategy):
    if not prompt_strategy:
        return DEFAULT_PROMPT_STRATEGY
    normalized = prompt_strategy.strip().lower()
    aliases = {
        "grammar-guided": "schema-grammar-guided",
        "schema-grammar-guided": "schema-grammar-guided",
        "allowed-entities-only": "schema-allowed-entities-only",
        "allowed-only": "schema-allowed-entities-only",
        "schema-allowed-entities-only": "schema-allowed-entities-only",
        "few-shot": "schema-few-shot",
        "schema-few-shot": "schema-few-shot",
        "chain-of-thought-plus": "schema-chain-of-thought-plus",
        "schema-chain-of-thought-plus": "schema-chain-of-thought-plus",
        "schema-chain-of-though-plus": "schema-chain-of-thought-plus",
        "least-to-most": "schema-least-to-most",
        "schema-least-to-most": "schema-least-to-most",
    }
    return aliases.get(normalized, normalized)

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
    top_indices = None
    try:
        query_emb = get_embedding(query)

        # Calculate cosine similarity between the query and all dataset questions
        similarities = cosine_similarity([query_emb], embeddings)[0]

        # Get the indices of the top-k most similar questions
        top_indices = np.argsort(similarities)[-top_k:][::-1]
    except Exception as exc:
        print(
            "Warning: using deterministic fallback examples because embedding retrieval failed. "
            f"Reason: {exc}"
        )
        top_indices = np.arange(min(top_k, len(dataset)))
    
    examples = []
    for idx in top_indices:
        examples.append({
            "nl": dataset[idx]["nl"],
            "nl_clean_1": dataset[idx].get("nl_clean_1"),
            "nl_clean_2": dataset[idx].get("nl_clean_2"),
            "query": dataset[idx]["query"],
            "reasoning": dataset[idx].get("reasoning")
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
    if not isinstance(content, str):
        return None
    # Target json code blocks specifically
    match = re.search(r'```(?:json)?\n(.*?)\n```', content, re.DOTALL)
    if not match:
        match = re.search(r'\{.*\}', content, re.DOTALL)

    if match:
        if match.lastindex:
            clean_content = match.group(1)
        else:
            clean_content = match.group(0)
    else:
        clean_content = content.strip()

    if not clean_content:
        return None
    
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


def get_validation_error_text(ocpq_json):
    try:
        candidate = _normalize_constraints_shape(ocpq_json)
        BindingBoxTree.model_validate(candidate)
        return None
    except ValidationError as e:
        return str(e)
    except Exception as e:
        return str(e)


def append_failure_log(
    stage,
    query,
    messages,
    raw_output,
    parsed_json,
    normalized_json,
    allowed_events=None,
    allowed_objects=None,
    validation_error=None,
    extra=None,
):
    payload = {
        "stage": stage,
        "query": query,
        "prompt_messages": messages,
        "raw_output": raw_output,
        "parsed_json": parsed_json,
        "normalized_json": normalized_json,
        "allowed_events": sorted(allowed_events or []),
        "allowed_objects": sorted(allowed_objects or []),
        "validation_error": validation_error,
        "extra": extra or {},
    }

    try:
        with open(failure_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"Warning: failed to write failure log to {failure_log_path}: {exc}")


def _dtype_from_name(torch_module, dtype_name):
    normalized = (dtype_name or "auto").strip().lower()
    if normalized in ("auto", ""):
        return None
    if normalized in ("float16", "fp16", "half"):
        return torch_module.float16
    if normalized in ("bfloat16", "bf16"):
        return torch_module.bfloat16
    if normalized in ("float32", "fp32"):
        return torch_module.float32
    raise ValueError(
        f"Unsupported local dtype '{dtype_name}'. Use one of: auto, float16, bfloat16, float32"
    )


def _build_local_prompt(messages):
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role.upper()}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _bitsandbytes_quantization_config(torch_module, quantization_mode, resolved_dtype):
    mode = (quantization_mode or "").strip().lower()
    if mode in ("", "none", "fp16", "float16", "bfloat16", "bf16", "auto"):
        return None

    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "4-bit local inference requires the 'bitsandbytes' package."
        ) from exc

    if mode in ("4bit", "4-bit", "bnb4", "bnb-4bit", "nf4"):
        compute_dtype = resolved_dtype or torch_module.bfloat16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    raise ValueError(
        f"Unsupported local quantization mode '{quantization_mode}'. Use 4bit or none."
    )


def _ensure_local_gpu_model_loaded():
    global local_model
    global local_tokenizer
    global local_model_key

    model_key = (hf_model, hf_token, local_gpu_device, local_gpu_dtype, local_gpu_quantization)
    if local_model is not None and local_tokenizer is not None and local_model_key == model_key:
        return

    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "Local GPU backend requires PyTorch with CUDA support."
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Local GPU backend selected but no CUDA GPU is available. "
            "Switch to HF Inference Providers or run on a machine with CUDA."
        )

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "transformers is required for local GPU backend but could not be imported."
        ) from exc

    auth_kwargs = {"token": hf_token} if hf_token else {}
    # Support three cases for the `--model` argument when using local GPU:
    # 1) A HF hub id or local model dir with full pretrained weights
    # 2) A local directory that contains a merged/converted model (pytorch_model.bin, etc.)
    # 3) A LoRA adapter directory (PEFT) such as lora_model_ocpq* containing adapter_config.json
    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    resolved_dtype = _dtype_from_name(torch, local_gpu_dtype)
    if resolved_dtype is not None:
        model_kwargs["torch_dtype"] = resolved_dtype
    quantization_config = _bitsandbytes_quantization_config(torch, local_gpu_quantization, resolved_dtype)
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    # If the provided model string refers to an existing local directory, attempt adapter-aware loading
    if os.path.isdir(hf_model):
        # Case: PEFT/LoRA adapter directory
        adapter_config_path = os.path.join(hf_model, "adapter_config.json")
        adapter_safetensors = os.path.join(hf_model, "adapter_model.safetensors")
        merged_weights = any(
            os.path.exists(os.path.join(hf_model, fname))
            for fname in ("pytorch_model.bin", "pytorch_model-00001-of-00002.bin", "model.safetensors")
        )

        if os.path.exists(adapter_config_path) or os.path.exists(adapter_safetensors):
            try:
                from peft import PeftModel
            except Exception as exc:
                raise RuntimeError(
                    "Loading a LoRA adapter requires the 'peft' package. Install it (pip install peft) to use adapter directories."
                ) from exc

            # Read base model name from adapter_config if available
            base_model_name = None
            try:
                with open(adapter_config_path, "r", encoding="utf-8") as f:
                    ac = json.load(f)
                    base_model_name = ac.get("base_model_name_or_path") or ac.get("base_model_name")
            except Exception:
                base_model_name = None

            if not base_model_name:
                raise RuntimeError(
                    "Adapter directory detected but base model name not found in adapter_config.json. "
                    "Include 'base_model_name_or_path' in adapter_config.json or specify the base model explicitly."
                )

            # NOTE: adapters trained before 2026-08-06 record unsloth's pre-quantized
            # redistribution here instead of the checkpoint the training run requested,
            # because unsloth substitutes it when load_in_4bit=True. For Gemma-3 that
            # redistribution ships a vision-tower layer with a missing quant_state ("FP4
            # quantization state not initialized"), so it used to be redirected to
            # `google/gemma-3-4b-it` at load time. `train_unsloth.py` now writes the
            # requested checkpoint into adapter_config.json at save time, so no per-model
            # redirect is needed. Re-train, or fix `base_model_name_or_path` by hand, if you
            # hit that error with an older adapter.

            # Load the base model with 4-bit quantization so its Params4bit layers are
            # properly initialised. The adapter's LoRA A/B weights are then injected
            # manually below, bypassing PeftModel.from_pretrained entirely.
            # Background: unsloth saves adapters in a format that embeds quantized base-
            # model layers alongside the LoRA deltas. PeftModel.from_pretrained restores
            # those quantized layers via bitsandbytes but with quant_state=None (version
            # mismatch between training and inference bitsandbytes), causing every forward
            # pass to raise a bare AssertionError with empty message. Loading only the
            # LoRA A/B tensors (keys containing "lora_") sidesteps the issue entirely.
            base = AutoModelForCausalLM.from_pretrained(base_model_name, **auth_kwargs, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True, **auth_kwargs)

            # Compatibility shim: get_peft_model / load_state_dict inspect Params4bit-
            # specific attributes (compress_statistics, quant_type, …) on the freshly
            # created plain nn.Parameter LoRA matrices, raising AttributeError. Adding
            # class-level defaults makes those accesses return a benign value. Real
            # Params4bit base layers shadow these with instance attributes, unaffected.
            _bnb_compat = {
                "compress_statistics": None,
                "quant_type": "nf4",
                "quant_state": None,
                "blocksize": 64,
                "quant_storage": None,
            }
            _added_compat_attrs = [a for a in _bnb_compat if a not in torch.nn.Parameter.__dict__]
            for _a in _added_compat_attrs:
                setattr(torch.nn.Parameter, _a, _bnb_compat[_a])
            try:
                from peft import PeftConfig, get_peft_model
                try:
                    from peft import set_peft_model_state_dict
                except ImportError:
                    from peft.utils import set_peft_model_state_dict
                from safetensors.torch import load_file as _st_load

                _peft_cfg = PeftConfig.from_pretrained(hf_model)
                model = get_peft_model(base, _peft_cfg)

                _adapter_sf = os.path.join(hf_model, "adapter_model.safetensors")
                _adapter_bin = os.path.join(hf_model, "adapter_model.bin")
                if os.path.exists(_adapter_sf):
                    _raw = _st_load(_adapter_sf, device=str(next(iter(base.parameters())).device))
                elif os.path.exists(_adapter_bin):
                    _raw = torch.load(_adapter_bin, map_location=next(iter(base.parameters())).device)
                else:
                    raise FileNotFoundError("No adapter_model.safetensors or adapter_model.bin found.")

                # Keep only LoRA delta weights; discard any quantized base-layer tensors
                # that unsloth may have embedded in the checkpoint. set_peft_model_state_dict
                # handles the key remapping (including the "default" adapter-name insertion)
                # between the saved checkpoint and the live get_peft_model() structure.
                _lora_sd = {k: v for k, v in _raw.items() if "lora_" in k}
                _result = set_peft_model_state_dict(model, _lora_sd, adapter_name="default")
                _unexpected = list(getattr(_result, "unexpected_keys", []) or [])
                _loaded = len(_lora_sd) - len(_unexpected)
                print(f"[adapter load] injected {_loaded}/{len(_lora_sd)} LoRA tensors "
                      f"({len(_unexpected)} unexpected)")
                if _loaded <= 0:
                    raise RuntimeError("No LoRA tensors matched the model's parameter names.")
            finally:
                for _a in _added_compat_attrs:
                    if _a in torch.nn.Parameter.__dict__:
                        try:
                            delattr(torch.nn.Parameter, _a)
                        except Exception:
                            pass

        elif merged_weights:
            # Directory appears to contain merged weights; load directly from the directory
            tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True, trust_remote_code=True, **auth_kwargs)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(hf_model, **auth_kwargs, **model_kwargs)

        else:
            raise RuntimeError(
                f"Model directory '{hf_model}' not recognized as a merged model or a PEFT adapter."
            )

    else:
        # Normal HF hub id or remote model
        tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True, trust_remote_code=True, **auth_kwargs)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(hf_model, **auth_kwargs, **model_kwargs)

    if not any(param.is_cuda for param in model.parameters()):
        raise RuntimeError(
            "Local backend loaded the model but no parameters are placed on CUDA. "
            "Check your local GPU configuration."
        )

    local_model = model
    local_tokenizer = tokenizer
    if hasattr(local_model, "generation_config") and local_model.generation_config is not None:
        try:
            local_model.generation_config.max_length = None
        except Exception:
            pass
    local_model_key = model_key


def _model_context_limit():
    """The loaded model's context window, read from its own config.

    Preferred over a hardcoded table: the model set changes, and a stale table would
    silently mis-report the very thing this guard exists to catch.
    """
    config = getattr(local_model, "config", None)
    for attr in ("max_position_embeddings", "max_sequence_length", "n_positions"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    text_config = getattr(config, "text_config", None)  # multimodal models nest it
    value = getattr(text_config, "max_position_embeddings", None)
    return value if isinstance(value, int) and value > 0 else None


def _check_context_budget(prompt_tokens, max_tokens):
    """Warn when a prompt plus its generation budget will not fit the context window.

    Nothing in this path truncates: the tokenizer is called without `max_length` and
    `generation_config.max_length` is set to None, so an oversized prompt is not clipped —
    it degrades or errors during generation. Without this the failure would surface only as
    unexplained garbage output in the results.
    """
    limit = _model_context_limit()
    if not limit:
        return
    needed = prompt_tokens + int(max_tokens or 0)
    if needed <= limit:
        return
    key = (local_model_key, needed // 512)
    if key in _context_warned:
        return
    _context_warned.add(key)
    print(
        f"WARNING: prompt ({prompt_tokens} tokens) + generation budget ({max_tokens}) "
        f"= {needed} exceeds the model's {limit}-token context. Output will be truncated "
        f"or degraded. Lower OCPQ_SCHEMA_TOKEN_BUDGET (currently "
        f"{SCHEMA_BLOCK_TOKEN_BUDGET}) or reduce few_shot_k."
    )


def _call_local_gpu_chat_completion(messages, max_tokens, response_format):
    import torch

    _ensure_local_gpu_model_loaded()

    normalized_messages = list(messages)
    if response_format is not None:
        json_instruction = "Return only valid JSON. Do not include markdown fences or commentary."
        if normalized_messages and normalized_messages[0]["role"] == "system":
            # Merge into the existing system message to avoid consecutive system turns,
            # which Gemma's chat template rejects with a role-alternation error.
            normalized_messages = [
                {"role": "system", "content": normalized_messages[0]["content"] + "\n" + json_instruction}
            ] + normalized_messages[1:]
        else:
            normalized_messages = [{"role": "system", "content": json_instruction}] + normalized_messages

    if hasattr(local_tokenizer, "apply_chat_template"):
        input_ids = local_tokenizer.apply_chat_template(
            normalized_messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        input_ids = input_ids.to(local_model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        prompt = _build_local_prompt(normalized_messages)
        model_inputs = local_tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(local_model.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(local_model.device)

    prompt_tokens = int(input_ids.shape[-1])
    _check_context_budget(prompt_tokens, max_tokens)

    do_sample = False
    temperature = generation_temperature
    if temperature is not None and temperature > 0:
        do_sample = True

    generate_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "pad_token_id": local_tokenizer.pad_token_id or local_tokenizer.eos_token_id,
        "eos_token_id": local_tokenizer.eos_token_id,
    }
    if temperature is not None and temperature > 0:
        generate_kwargs["temperature"] = float(temperature)
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    with torch.no_grad():
        output_ids = local_model.generate(**generate_kwargs)

    generated_ids = output_ids[0][input_ids.shape[-1]:]
    content = local_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    completion_tokens = int(generated_ids.shape[-1])
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage,
    )


def _extract_usage_from_response(response):
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return None

    if total_tokens is None:
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _record_llm_call(response, duration_sec, model_name=None, response_format=None):
    usage = _extract_usage_from_response(response)
    content = ""
    try:
        content = response.choices[0].message.content or ""
    except Exception:
        pass
    entry = {
        "model": model_name,
        "backend": hf_backend,
        "duration_sec": round(float(duration_sec), 4),
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "total_tokens": usage.get("total_tokens") if usage else None,
        "completion_chars": len(content),
    }
    if isinstance(response_format, dict):
        entry["response_format"] = response_format.get("type")
    LLM_CALL_STATS.append(entry)


def _build_response_format(schema_name, schema, enforce_schema=True):
    if not enforce_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }

# In[7]:

def call_hf_with_retry(model, messages, max_tokens, response_format, max_retries=5):
    if hf_backend == "local-gpu":
        current_max_tokens = max_tokens
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                response = _call_local_gpu_chat_completion(
                    messages=messages,
                    max_tokens=current_max_tokens,
                    response_format=response_format,
                )
                _record_llm_call(
                    response=response,
                    duration_sec=time.time() - start_time,
                    model_name=model,
                    response_format=response_format,
                )
                return response
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if is_oom and attempt < max_retries - 1:
                    current_max_tokens = max(256, int(current_max_tokens * 0.75))
                    print(
                        "Local GPU OOM during generation. "
                        f"Retrying with max_tokens={current_max_tokens} (attempt {attempt + 1}/{max_retries})."
                    )
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                raise

    if hf_backend == "custom-endpoint":
        if openai_client is None:
            raise RuntimeError(
                "OpenAI client is not initialized. Call set_hf_client(token, base_url=...) first."
            )

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "response_format": response_format,
                }
                if generation_temperature is not None:
                    kwargs["temperature"] = float(generation_temperature)

                start_time = time.time()
                response = openai_client.chat.completions.create(**kwargs)
                _record_llm_call(
                    response=response,
                    duration_sec=time.time() - start_time,
                    model_name=model,
                    response_format=response_format,
                )
                return response
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

    if hf_client is None:
        raise RuntimeError(
            "HF client is not initialized. Call set_hf_client(token, provider=...) first."
        )

    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "response_format": response_format,
            }
            if generation_temperature is not None:
                kwargs["temperature"] = float(generation_temperature)

            start_time = time.time()
            response = hf_client.chat_completion(**kwargs)
            _record_llm_call(
                response=response,
                duration_sec=time.time() - start_time,
                model_name=model,
                response_format=response_format,
            )
            return response
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


FILTER_TYPES = {
    "O2E",
    "O2O",
    "TimeBetweenEvents",
    "NotEqual",
    "EventAttributeValueFilter",
    "ObjectAttributeValueFilter",
    "BasicFilterCEL",
}

SIZE_FILTER_TYPES = {
    "NumChilds",
}

TIME_UNIT_SECONDS = {
    "day": 86400,
    "days": 86400,
    "hour": 3600,
    "hours": 3600,
    "minute": 60,
    "minutes": 60,
    "min": 60,
    "mins": 60,
    "second": 1,
    "seconds": 1,
    "sec": 1,
    "secs": 1,
}


def _normalize_constraints_shape(ocpq_json):
    """
    Normalize common LLM shape mistakes before strict schema validation.

    Common issue: filter objects are placed directly in `constraints` instead of
    being wrapped as {"type": "Filter", "filter": {...}}.
    """
    if not isinstance(ocpq_json, dict):
        return ocpq_json

    nodes = ocpq_json.get("nodes")
    if not isinstance(nodes, list):
        return ocpq_json

    for node in nodes:
        if not isinstance(node, dict):
            continue
        box = node.get("Box")
        if not isinstance(box, list) or len(box) != 2:
            continue
        node_data = box[0]
        if not isinstance(node_data, dict):
            continue

        constraints = node_data.get("constraints")
        if not isinstance(constraints, list):
            continue

        normalized_constraints = []
        for c in constraints:
            if not isinstance(c, dict):
                normalized_constraints.append(c)
                continue

            ctype = c.get("type")
            if ctype in FILTER_TYPES:
                normalized_constraints.append({"type": "Filter", "filter": c})
            elif ctype in SIZE_FILTER_TYPES:
                normalized_constraints.append({"type": "SizeFilter", "filter": c})
            else:
                normalized_constraints.append(c)

        node_data["constraints"] = normalized_constraints

    return ocpq_json


def _extract_allowed_names(items, default_key):
    names = set()
    if not isinstance(items, list):
        return names

    for item in items:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict):
            candidate = item.get(default_key)
            if isinstance(candidate, str):
                names.add(candidate)
    return names


def _extract_allowed_entities_from_messages(messages):
    allowed_events = set()
    allowed_objects = set()

    if len(messages) <= 2:
        return allowed_events, allowed_objects

    try:
        sp1_content = messages[2].get("content", "{}")
        extracted_structure = json.loads(sp1_content)
        allowed_events = _extract_allowed_names(extracted_structure.get("events", []), "event")
        allowed_objects = _extract_allowed_names(extracted_structure.get("objects", []), "object")
    except Exception:
        pass

    return allowed_events, allowed_objects


def _within_seconds_from_query(query):
    if not isinstance(query, str):
        return None

    match = re.search(
        r"within\s+(\d+(?:\.\d+)?)\s*(day|days|hour|hours|minute|minutes|min|mins|second|seconds|sec|secs)\b",
        query,
        re.IGNORECASE,
    )
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).lower()
    multiplier = TIME_UNIT_SECONDS.get(unit)
    if multiplier is None:
        return None
    return float(value * multiplier)


def _normalize_time_between_events_seconds(ocpq_json, query):
    target_max_seconds = _within_seconds_from_query(query)
    if target_max_seconds is None or not isinstance(ocpq_json, dict):
        return ocpq_json

    nodes = ocpq_json.get("nodes")
    if not isinstance(nodes, list):
        return ocpq_json

    for node in nodes:
        if not isinstance(node, dict):
            continue
        box = node.get("Box")
        if not isinstance(box, list) or len(box) != 2:
            continue
        node_data = box[0]
        if not isinstance(node_data, dict):
            continue

        filters = node_data.get("filters", [])
        if isinstance(filters, list):
            for f in filters:
                if isinstance(f, dict) and f.get("type") == "TimeBetweenEvents":
                    f["max_seconds"] = target_max_seconds

        constraints = node_data.get("constraints", [])
        if isinstance(constraints, list):
            for c in constraints:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "Filter" and isinstance(c.get("filter"), dict):
                    wrapped = c["filter"]
                    if wrapped.get("type") == "TimeBetweenEvents":
                        wrapped["max_seconds"] = target_max_seconds

    return ocpq_json


def run_ocpq_query_with_logging(ocel_path, output_file, query, messages, stage):
    start_time = time.time()
    exec_success = False
    exec_stdout = None
    exec_stderr = None
    exec_result_file = None
    try:
        from QueryExecutor import run_ocpq_query
        result = run_ocpq_query(ocel_path, output_file)
        if not result:
            warning = "warning: run_ocpq_query returned no result"
            allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)
            append_failure_log(
                stage=f"{stage}_query_exec_empty",
                query=query,
                messages=messages,
                raw_output=None,
                parsed_json=None,
                normalized_json=None,
                allowed_events=allowed_events,
                allowed_objects=allowed_objects,
                validation_error=warning,
                extra={"warning": warning},
            )
            exec_stdout = warning
            exec_stderr = warning
            return warning, warning, None
        exec_stdout, exec_stderr, exec_result_file = result
        exec_success = bool(exec_result_file and not exec_stderr)
        return result
    except Exception as exc:
        trace = traceback.format_exc()
        allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)
        append_failure_log(
            stage=f"{stage}_query_exec_failed",
            query=query,
            messages=messages,
            raw_output=None,
            parsed_json=None,
            normalized_json=None,
            allowed_events=allowed_events,
            allowed_objects=allowed_objects,
            validation_error=str(exc),
            extra={"traceback": trace},
        )
        exec_stdout = f"Error running query automatically: {exc}"
        exec_stderr = None
        return exec_stdout, None, None
    finally:
        QUERY_EXEC_STATS.append(
            {
                "stage": stage,
                "duration_sec": round(time.time() - start_time, 4),
                "success": exec_success,
                "stdout": exec_stdout,
                "stderr": exec_stderr,
                "result_file": exec_result_file,
            }
        )

def validate_and_normalize_ocpq_json(ocpq_json):
    """
    Strictly validate against the Pydantic BindingBoxTree model.
    Returns a normalized JSON dict (aliases applied) when valid.
    """
    try:
        ocpq_json = _normalize_constraints_shape(ocpq_json)
        validated = BindingBoxTree.model_validate(ocpq_json)
        return validated.model_dump(by_alias=True)
    except ValidationError:
        return None


def schema_fix_with_llm(model_name, messages, prompt, max_tokens=2000, schema_enforced=True):
    """
    Run a schema-constrained repair pass so the output must match OCPQ schema.
    """
    fix_messages = list(messages)
    fix_messages.append({"role": "user", "content": prompt})
    fix_response = call_hf_with_retry(
        model=model_name,
        messages=fix_messages,
        max_tokens=max_tokens,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    fixed_content = fix_response.choices[0].message.content or ""
    fixed_json = parse_llm_json(fixed_content)
    fixed_json = validate_and_normalize_ocpq_json(fixed_json)
    return fixed_json, fixed_content

# Verify the generated OCPQ JSON and ask the LLM to fix it if it's invalid or doesn't follow the schema. 
# This is a safety net to improve the quality of the output before executing the query.
def verify_ocpq_json(ocpq_json, model_name, messages, query=None, schema_enforced=True):
    ocpq_json = _normalize_time_between_events_seconds(ocpq_json, query)
    normalized = validate_and_normalize_ocpq_json(ocpq_json)
    
    is_connected = True
    has_empty_nodes = False
    has_hallucinated_entities = False
    missing_o2e = False
    
    # Extract allowed entities from SP1 if available
    allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)

    if normalized and "nodes" in normalized:
        num_nodes = len(normalized["nodes"])
        if num_nodes > 1:
            referenced_children = set()
            for node in normalized["nodes"]:
                box = node.get("Box", [])
                if len(box) == 2 and isinstance(box[1], list):
                    referenced_children.update(box[1])
            for i in range(1, num_nodes):
                if i not in referenced_children:
                    is_connected = False
                    break
        
        # Check for empty nodes (no events and no objects)
        for node in normalized["nodes"]:
            box = node.get("Box", [])
            if len(box) > 0 and isinstance(box[0], dict):
                node_data = box[0]
                new_events = node_data.get("newEventVars", {})
                new_objects = node_data.get("newObjectVars", {})
                constraints = node_data.get("constraints", [])
                
                if not new_events and not new_objects and not constraints:
                     has_empty_nodes = True
                
                # Check for Cartesian product explosion (missing O2E)
                if new_events and new_objects:
                    filters = node_data.get("filters", [])
                    has_o2e = any(f.get("type") == "O2E" for f in filters)
                    if not has_o2e:
                        missing_o2e = True

                # Check for hallucinated entities (if we have allowed sets)
                if allowed_events:
                    for ev_idx, ev_list in new_events.items():
                        for ev in ev_list:
                            if isinstance(ev, str) and ev not in allowed_events:
                                has_hallucinated_entities = True
                if allowed_objects:
                    for obj_idx, obj_list in new_objects.items():
                        for ob in obj_list:
                            if isinstance(ob, str) and ob not in allowed_objects:
                                has_hallucinated_entities = True

    validation_error = None if normalized is not None else get_validation_error_text(ocpq_json)

    if normalized is None or not is_connected or has_empty_nodes or has_hallucinated_entities or missing_o2e:
        append_failure_log(
            stage="verify_pre_fix",
            query=query,
            messages=messages,
            raw_output=None,
            parsed_json=ocpq_json,
            normalized_json=normalized,
            allowed_events=allowed_events,
            allowed_objects=allowed_objects,
            validation_error=validation_error,
            extra={
                "is_connected": is_connected,
                "has_empty_nodes": has_empty_nodes,
                "has_hallucinated_entities": has_hallucinated_entities,
                "missing_o2e": missing_o2e,
            },
        )
        prompt = f"""
The following JSON is invalid, incomplete, contains disconnected nodes, contains completely empty nodes, misses O2E filters, or has hallucinated entities (events/objects not provided).

Fix it so that:
- It follows OCPQ schema
- It correctly represents the query
- ALL nodes are properly connected in a single tree. Every node (except 0) must be referenced in the child indices array of a parent node.
- Critical filters (O2E, TimeBetweenEvents) are placed in connected child nodes to prevent unbounded Cartesian searches.
- No nodes are empty. Every node must contain at least an event, an object, or a constraint. DO NOT create empty boxes!
- ONLY use the following allowed entities natively mapped from the query. DO NOT use hallucinated terms like Person or Diagnosis!
- If a node has BOTH events and objects, it MUST have an `O2E` filter connecting them. Otherwise, it will cause an infinite Cartesian explosion!
- If the query says "within X days/hours/minutes", convert it to seconds for `max_seconds` (e.g. 3 days = 259200).

Allowed Events: {allowed_events}
Allowed Objects: {allowed_objects}

Return ONLY fixed JSON.

JSON:
{ocpq_json}
"""
        fixed_json, _fixed_content = schema_fix_with_llm(
            model_name=model_name,
            messages=messages,
            prompt=prompt,
            schema_enforced=schema_enforced,
        )
        fixed_json = _normalize_time_between_events_seconds(fixed_json, query)
        fixed_json = validate_and_normalize_ocpq_json(fixed_json)
        if fixed_json is None:
            append_failure_log(
                stage="verify_fix_failed",
                query=query,
                messages=messages,
                raw_output=None,
                parsed_json=ocpq_json,
                normalized_json=fixed_json,
                allowed_events=allowed_events,
                allowed_objects=allowed_objects,
                validation_error=get_validation_error_text(ocpq_json),
                extra={"repair_prompt": prompt},
            )
        return fixed_json
    else:
        return normalized


def run_subproblem_1_extract_entities(query, schema_block, messages, model_name):
    # Only ask for relation grounding when the schema block actually contains a relation
    # graph (schema_level == "full"); at lower schema levels this section is absent and
    # asking for it would just invite hallucinated relations.
    relations_available = (
        "EVENT-TO-OBJECT RELATIONS" in schema_block or "OBJECT-TO-OBJECT RELATIONS" in schema_block
    )
    relations_instruction = ""
    if relations_available:
        relations_instruction = (
            "\nAlso copy which relations from the EVENT-TO-OBJECT RELATIONS and OBJECT-TO-OBJECT RELATIONS "
            "lists above connect the events/objects you just mapped. Copy each relation VERBATIM as shown "
            "(e.g. \"EventType -> ObjectType\") into 'object_event_links' (event-to-object) and "
            "'object_object_links' (object-to-object). Do NOT invent relations that are not listed. If an "
            "object you need is not directly related to any identified event, include the relation chain "
            "through an intermediary object that IS listed, so the chain can be followed later.\n"
        )
    prompt_1 = (
        "Given the user query, map the terms to the exact events and objects involved.\n\n"
        "CRITICAL: You MUST translate colloquial terms (e.g. 'person', 'diagnosis', 'registration') directly "
        "to the EXACT names provided below. DO NOT use names that aren't in these lists.\n\n"
        f"{schema_block}\n\n"
        f"Query: {query}\n\n"
        "Output a clear mapping of the elements found in the query to the valid events and objects. "
        "Also identify any constraints (like 'completed within TIME', 'exactly once', 'at least once').\n"
        f"{relations_instruction}"
        "Return ONLY JSON matching the schema."
    )
    messages.append({"role": "user", "content": prompt_1})

    response_1 = call_hf_with_retry(
        model=model_name,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extracted_structure", "schema": EXTRACTED_STRUCTURE_SCHEMA, "strict": True},
        }
    )
    structure_text = response_1.choices[0].message.content
    messages.append({"role": "assistant", "content": structure_text})
    print("Sub-problem 1 completed: Entities extracted.")
    return structure_text


def run_subproblem_2_plan_tree(messages, model_name):
    # Re-inject extracted structure immediately before generating the tree
    extracted_structure = "{}"
    if len(messages) > 2:
        extracted_structure = messages[2].get("content", "{}")

    prompt_2 = (
        """Based on the mapped elements, plan the logical tree structure for the OCPQ query.

        TREE FORMAT RULES:
        - 'events' and 'objects' on each node list ONLY the events/objects newly introduced AT THAT NODE, not 
        entities already introduced by an ancestor node (a child implicitly inherits its ancestors' bindings).
        - 'description' is the ONLY place to record filters and constraints for that node -- state the EXACT 
        type name and key parameters (e.g. "O2E: event 0 -> object 0", "TimeBetweenEvents:
        max_seconds=259200", "SizeFilter/NumChilds: child A, min=1", "NOT: child A"). Do not leave
        filters/constraints unstated in a node that needs them.
        - 'id' values start at 0 and increase by node order. Every edge's 'from'/'to' MUST reference an 'id'
        that exists in 'nodes' -- never invent or skip ids.

        EXAMPLE (format only -- your actual events/objects/filters must come from the mapped elements below):
        {
          "nodes": [
            {"id": 0, "description": "root: holds Object A, no filters", "events": [], "objects": ["Object A"]},
            {"id": 1, "description": "O2E: event 0 -> object 0", "events": ["Event A"], "objects": []}
          ],
          "edges": [
            {"from": 0, "to": 1, "label": "A"}
          ]
        }
        OCPQ PATTERN CHEAT SHEET (typical shapes, not fixed node counts -- add nodes if extra constraints require it):
        - 'completed within TIME': Typically 2 nodes. Parent node has Event A and Object A. Child node has
        Event B, an 'O2E' filter linking Event B to Object A, and a 'TimeBetweenEvents' filter (max_seconds)
        bounding Event A -> Event B.
        - Occurrence-count constraints ('at least once', 'more than once', 'exactly once', 'must not occur more
        than N times'): Root node has Object A and BOTH a 'SAT' constraint AND a 'SizeFilter'/'NumChilds'
        constraint pointing at the SAME child path (set min/max per the phrasing, e.g. min:1 for 'at least once',
        min:1,max:1 for 'exactly once'). Connected child node has Event A and an 'O2E' filter linking
        Event A to Object A.
        - 'must not occur' / 'must not X' / 'should not X': Requires AT LEAST 2 nodes -- a 'NOT' constraint
        always references a child path by name, so the forbidden pattern must live in a real child node. The
        parent node holds the triggering event/object; the 'NOT' constraint's child_names point at the child
        node(s) describing what must NOT happen.
        - Simple, single-condition queries that need no nesting: a single root node with NO children is valid
        -- put the events, objects, and filters/constraints directly in that one node.

        CRITICAL TO PREVENT SLOW QUERIES (OOM CRASHES):
        1. DO NOT create orphaned/disconnected nodes. EVERY node except the root must be reachable via some
        parent's child index array.
        2. DO NOT create EMPTY nodes that don't contain any events or objects. Every node should have a clear purpose in the query plan.
        3. Filters like `TimeBetweenEvents`, `O2E`, and `O2O` may live in the root node itself for simple,
        non-nested queries, or in any node reachable from the root -- just never in a node that is unreachable.
        4. Connect events to their objects using `O2E` filters to avoid unbounded Cartesian product searches.

        Describe the exact tree structure (number of nodes, what goes in them, parent-child relationships, 
        and exact filters/constraints).

        CRITICAL: ONLY use the events and objects mapped here. DO NOT INVENT NEW ONES!

        {extracted_structure}

        Return ONLY JSON matching the schema."""
    ).replace("{extracted_structure}", extracted_structure)
    messages.append({"role": "user", "content": prompt_2})

    response_2 = call_hf_with_retry(
        model=model_name,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_tree", "schema": QUERY_TREE_SCHEMA, "strict": True},
        }
    )
    tree_text = response_2.choices[0].message.content
    messages.append({"role": "assistant", "content": tree_text})
    print("Sub-problem 2 completed: Tree structure planned.")
    return tree_text


def build_few_shot_examples(query, dataset, embeddings):
    if not few_shot_enabled or few_shot_k <= 0:
        return ""
    examples = retrieve_examples(query, dataset, embeddings, top_k=few_shot_k)
    example_prompts = []
    for ex in examples:
        qt = "nl"
        if qt in ex:
            reasoning = ex.get(
                "reasoning",
                "Step 1: Identify elements. Step 2: Establish logic. Step 3: Implement JSON.",
            )
            example_prompts.append(
                f"Question: {ex[qt]}\nReasoning: {reasoning}\nJSON:\n{json.dumps(ex['query'], indent=2)}"
            )
    return "\n\n".join(example_prompts)


def run_subproblem_3_generate_json(query, dataset, embeddings, messages, model_name, schema_enforced=True):
    few_shot_string = build_few_shot_examples(query, dataset, embeddings)
    
    # Re-inject extracted structure and logical tree right before the end
    extracted_structure = "{}"
    logical_tree = "{}"
    
    if len(messages) > 2:
        extracted_structure = messages[2].get("content", "{}")
    if len(messages) > 4:
        logical_tree = messages[4].get("content", "{}")

    example_block = ""
    if few_shot_string:
        example_block = (
            "Use these examples as references for the output syntax.\n"
            f"EXAMPLES:\n{few_shot_string}\n\n"
        )

    prompt_3 = (
        "Now, based on your previous reasoning, output the final OCPQ JSON matching the Exact Schema.\n\n"
        f"{example_block}"
        "CRITICAL FINAL CONSTRAINTS:\n"
        "1. DO NOT copy events or objects from the examples above.\n"
        "2. You MUST strictly use ONLY the allowed events and objects mapped in your first subproblem.\n"
        f"ALLOWED ENTITIES: {extracted_structure}\n\n"
        "3. You must follow the logical tree plan from your second subproblem.\n"
        f"LOGICAL PLAN: {logical_tree}\n\n"
        "4. Every 'O2E' and 'O2O' filter you emit MUST correspond to a relation actually present in "
        "ALLOWED ENTITIES ('object_event_links'/'object_object_links', when populated) or implied by the "
        "LOGICAL PLAN's node descriptions. Do NOT invent an event-object or object-object link that isn't "
        "grounded there.\n\n"
        "Output strictly and ONLY valid JSON matching the exact schema based entirely on these allowed entities and plan."
    )
    
    # Heuristic Context Pruning: Sever the long chat flow. Inject ONLY the distilled outputs natively into SP3 prompt.
    messages_sp3 = [
        {
            "role": "system",
            "content": "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided schema.",
        },
        {"role": "user", "content": prompt_3}
    ]

    response_3 = call_hf_with_retry(
        model=model_name,
        messages=messages_sp3,
        max_tokens=2000,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    content = response_3.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    if ocpq_json is None:
        allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)
        append_failure_log(
            stage="grammar_guided_parse_failed",
            query=query,
            messages=messages_sp3,
            raw_output=content,
            parsed_json=None,
            normalized_json=None,
            allowed_events=allowed_events,
            allowed_objects=allowed_objects,
            validation_error="Could not parse model output as JSON",
        )
    # SP3 uses a fresh messages_sp3 for its LLM call, so do NOT append the response
    # to the shared `messages` list. Doing so would leave two consecutive assistant turns
    # ([assistant(SP2), assistant(SP3)]), which Gemma's chat template rejects with a
    # role-alternation error in the subsequent verify/fix calls.
    print("Sub-problem 3 completed: OCPQ JSON generated.")
    return ocpq_json

# Use a single continuous chat context with Grammar-Guided Prompting
def _run_pipeline_grammar_guided(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    print("Starting Grammar-Guided NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")
    
    # Extract event and object names from the OCEL log
    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)

    # Initialize the chat messages for context
    messages = [
        {"role": "system", "content": "You are a process mining expert that converts natural language into Object-Centric Process Queries (OCPQ). You must think step by step and respond concisely according to the prompts."}
    ]

    # Sub-problem 1: Extract entities mapping
    structure_text = run_subproblem_1_extract_entities(
        query=query,
        schema_block=schema_block,
        messages=messages,
        model_name=selected_model,
    )

    # Sub-problem 2: Plan tree topology
    tree_text = run_subproblem_2_plan_tree(messages=messages, model_name=selected_model)

    # Sub-problem 3: Generate final JSON
    ocpq_json = run_subproblem_3_generate_json(
        query=query,
        dataset=dataset,
        embeddings=embeddings,
        messages=messages,
        model_name=selected_model,
        schema_enforced=schema_enforced,
    )

    if schema_validation:
        ocpq_json = verify_ocpq_json(
            ocpq_json,
            selected_model,
            messages,
            query=query,
            schema_enforced=schema_enforced,
        )
        if ocpq_json is None:
            raise ValueError("Generated OCPQ JSON could not be validated against the schema.")

        # Full schema gate before writing/running the query.
        ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
        if ocpq_json is None:
            raise ValueError("Refusing to execute: OCPQ JSON failed strict Pydantic validation.")
    elif ocpq_json is None:
        raise ValueError("Generated OCPQ JSON could not be parsed.")

    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_result_file_path = None
    query_stderr = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path,
            output_file=output_file,
            query=query,
            messages=messages,
            stage="grammar_guided",
        )

        # Auto-fix using the chat history context block
        # Only try to fix if an actual execution error (stderr from a crash/panic) occurred
        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_4 = f"The generated JSON query caused this execution error:\n{query_stderr}\n\nPlease fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
            fixed_json, fixed_content = schema_fix_with_llm(
                model_name=selected_model,
                messages=messages,
                prompt=prompt_4,
                schema_enforced=schema_enforced,
            )

            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path,
                    output_file=output_file,
                    query=query,
                    messages=messages,
                    stage="grammar_guided_fix",
                )
                ocpq_json = fixed_json
                messages.append({"role": "user", "content": prompt_4})
                messages.append({"role": "assistant", "content": fixed_content})

    return structure_text, tree_text, ocpq_json, query_result, query_result_file_path


def _run_pipeline_few_shot(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    print("Starting Few-Shot NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)

    messages = [
        {
            "role": "system",
            "content": "You are a process mining expert that converts natural language into Object-Centric Process Queries (OCPQ). You must think step by step and respond concisely according to the prompts.",
        }
    ]

    few_shot_string = build_few_shot_examples(query, dataset, embeddings)
    example_block = f"\nEXAMPLES:\n{few_shot_string}\n" if few_shot_string else ""
    prompt_intro = "Translate natural language process queries into OCPQ JSON based on the examples."
    if not few_shot_string:
        prompt_intro = "Translate natural language process queries into OCPQ JSON."

    prompt_3 = f"""
{prompt_intro}

Output ONLY valid JSON.

CRITICAL: You MUST translate colloquial terms (e.g. 'person', 'diagnosis', 'registration') directly to the EXACT names provided below.
DO NOT use names that aren't in these lists.
{schema_block}
{example_block}
Now translate this query:
Question: {query}
JSON:
"""
    messages.append({"role": "user", "content": prompt_3})
    messages[0] = {
        "role": "system",
        "content": "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided schema.",
    }

    response_3 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=2000,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    content = response_3.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    if ocpq_json is None:
        append_failure_log(
            stage="few_shot_parse_failed",
            query=query,
            messages=messages,
            raw_output=content,
            parsed_json=None,
            normalized_json=None,
            allowed_events=event_names,
            allowed_objects=object_names,
            validation_error="Could not parse model output as JSON",
        )
    messages.append({"role": "assistant", "content": content})
    print("Sub-problem 3 completed: OCPQ JSON generated.")

    if schema_validation:
        normalized = validate_and_normalize_ocpq_json(ocpq_json)
        if normalized is not None:
            ocpq_json = normalized
        else:
            # Branch-parity behavior: run a lightweight repair request when output is invalid.
            prompt = f"""
The following JSON is invalid or incomplete.

Fix it so that:
- It follows OCPQ schema
- It correctly represents the query

Return ONLY fixed JSON.

JSON:
{ocpq_json}
"""
            repair = call_hf_with_retry(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, using double quotes. Strictly follow the OCPQ schema.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
            )
            ocpq_json = parse_llm_json(repair.choices[0].message.content)
            ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
            if ocpq_json is None:
                raise ValueError("Generated OCPQ JSON could not be validated against the schema.")
    elif ocpq_json is None:
        raise ValueError("Generated OCPQ JSON could not be parsed.")

    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_result_file_path = None
    query_stderr = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path,
            output_file=output_file,
            query=query,
            messages=messages,
            stage="few_shot",
        )

        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_4 = (
                f"The generated JSON query caused this execution error:\n{query_stderr}\n\n"
                "Please fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
            )
            messages.append({"role": "user", "content": prompt_4})

            fix_response = call_hf_with_retry(
                model=selected_model,
                messages=messages,
                max_tokens=2000,
                response_format=None,
            )
            fixed_json = parse_llm_json(fix_response.choices[0].message.content)
            fixed_json = validate_and_normalize_ocpq_json(fixed_json)

            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path,
                    output_file=output_file,
                    query=query,
                    messages=messages,
                    stage="few_shot_fix",
                )
                ocpq_json = fixed_json

    return None, None, ocpq_json, query_result, query_result_file_path


def _run_pipeline_allowed_entities_only(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    print("Starting Allowed-Entities-Only NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a JSON-only API. You must output raw, valid OCPQ JSON exactly matching the schema. "
                "Never invent entities or unsupported filter types."
            ),
        }
    ]

    prompt = f"""
Convert this natural language query to OCPQ JSON.

Output ONLY valid JSON.

{schema_block}

Rules:
- The root object MUST contain both "nodes" and "edgeNames".
- "edgeNames" is an array of [[parent_index, child_index], "label"] pairs, one per parent-child
  link used in "nodes" (e.g. [[0, 1], "A"]). Do NOT omit it.
- Basic filters ("TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter",
  "ObjectAttributeValueFilter", "BasicFilterCEL") must be wrapped as
  {{"type": "Filter", "filter": {{...}}}} when placed in "constraints".
- "O2E" and "O2O" belong in the node's "filters" array, never directly in "constraints".
- Use ONLY event and object names from the allowed lists.
- Do NOT invent unsupported filter types such as "ExactlyOnce".
- Represent "exactly once" using supported schema structure, not a custom operator.
- If a node contains both events and objects, include the required O2E filter(s).
- Keep all nodes connected in a single tree and avoid empty nodes.

Query:
{query}
"""
    messages.append({"role": "user", "content": prompt})

    response = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=2000,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    content = response.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    if ocpq_json is None:
        append_failure_log(
            stage="allowed_entities_only_parse_failed",
            query=query,
            messages=messages,
            raw_output=content,
            parsed_json=None,
            normalized_json=None,
            allowed_events=event_names,
            allowed_objects=object_names,
            validation_error="Could not parse model output as JSON",
        )
    messages.append({"role": "assistant", "content": content})
    print("Direct generation completed: OCPQ JSON generated.")

    # Reuse the same message position the verifier expects for allowed-entity extraction.
    extracted_structure_stub = json.dumps(
        {
            "events": [{"event": event_name} for event_name in event_names],
            "objects": [{"object": object_name} for object_name in object_names],
        }
    )
    verifier_messages = [messages[0], messages[1], {"role": "assistant", "content": extracted_structure_stub}]

    if schema_validation:
        ocpq_json = verify_ocpq_json(
            ocpq_json,
            selected_model,
            verifier_messages,
            query=query,
            schema_enforced=schema_enforced,
        )
        if ocpq_json is None:
            raise ValueError("Generated OCPQ JSON could not be validated against the schema.")

        ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
        if ocpq_json is None:
            raise ValueError("Refusing to execute: OCPQ JSON failed strict Pydantic validation.")
    elif ocpq_json is None:
        raise ValueError("Generated OCPQ JSON could not be parsed.")

    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_result_file_path = None
    query_stderr = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path,
            output_file=output_file,
            query=query,
            messages=messages,
            stage="allowed_entities_only",
        )

        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_fix = (
                f"The generated JSON query caused this execution error:\n{query_stderr}\n\n"
                "Please fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
            )
            fixed_json, fixed_content = schema_fix_with_llm(
                model_name=selected_model,
                messages=messages,
                prompt=prompt_fix,
                schema_enforced=schema_enforced,
            )

            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path,
                    output_file=output_file,
                    query=query,
                    messages=messages,
                    stage="allowed_entities_only_fix",
                )
                ocpq_json = fixed_json
                messages.append({"role": "user", "content": prompt_fix})
                messages.append({"role": "assistant", "content": fixed_content})

    return None, None, ocpq_json, query_result, query_result_file_path


def _run_pipeline_chain_of_thought_plus(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    print("Starting Chain-of-Thought-Plus NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)
    messages = [
        {
            "role": "system",
            "content": "You are a process mining expert that converts natural language into Object-Centric Process Queries (OCPQ). You must think step by step and respond concisely according to the prompts.",
        }
    ]

    prompt_1 = f"""
You are extracting the structure of a process query.

Given a natural language query, identify:

1. Events involved
2. Objects involved
3. Relationships between events (before, after, etc.)
4. Cardinality constraints (e.g., exactly once, at least once)
5. Whether nesting is needed (yes/no)

The events and objects must be chosen from the schema extracted from the log:
{schema_block}

Return ONLY JSON in this format:

{{
  "events": [...],
  "objects": [...],
  "relations": [...],
  "constraints": [...],
  "requires_nesting": true/false
}}

Query:
{query}
"""
    messages.append({"role": "user", "content": prompt_1})
    response_1 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extracted_structure", "schema": EXTRACTED_STRUCTURE_SCHEMA, "strict": True},
        },
    )
    structure_text = response_1.choices[0].message.content
    messages.append({"role": "assistant", "content": structure_text})
    print("Sub-problem 1 completed: Entities extracted.")

    prompt_2 = f"""
You are designing an OCPQ query tree.

Given this extracted structure:

{structure_text}

Decide:

- Number of nodes
- What each node represents
- Parent-child relationships

The example below shows the FORMAT ONLY, with 2 nodes and 1 edge. Your actual
tree may need more or fewer nodes/edges depending on the query — include ALL
nodes and edges you decided on above, not just this many.

Return ONLY JSON:

{{
  "nodes": [
    {{
      "id": 0,
      "description": "...",
      "events": [...],
      "objects": [...]
    }},
    {{
      "id": 1,
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
    messages.append({"role": "user", "content": prompt_2})
    response_2 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_tree", "schema": QUERY_TREE_SCHEMA, "strict": True},
        },
    )
    tree_text = response_2.choices[0].message.content
    messages.append({"role": "assistant", "content": tree_text})
    print("Sub-problem 2 completed: Tree structure planned.")

    few_shot_string = build_few_shot_examples(query, dataset, embeddings)
    prompt_3 = f"""
You are generating OCPQ JSON.

CRITICAL SCHEMA REQUIREMENT:
Output ONLY valid JSON matching the exact schema shown in the examples.
The root object MUST contain exactly two top-level keys: "nodes" and "edgeNames".
Each node in the "nodes" array MUST be an object with a single key "Box".
The "Box" value MUST always be an array with exactly TWO elements:
1. The node data dictionary (containing newEventVars, newObjectVars, filters, sizeFilters, constraints, evVarLabels, obVarLabels).
2. An array of child node indices (e.g., [], [1]).
"edgeNames" MUST be an array of [[parent_index, child_index], "label"] pairs, one per parent-child link used in "nodes" (e.g., [[0, 1], "A"]). Do NOT omit this field.

CONSTRAINT SCHEMA:
The constraints in your JSON MUST ONLY use the following top-level variants:
"Filter", "SizeFilter", "SAT", "ANY", "NOT", "OR", "AND".
Basic filter logic like "TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter", "ObjectAttributeValueFilter", "BasicFilterCEL" must be wrapped inside a {{ "type": "Filter", "filter": {{ ... }} }} constraint.
Do NOT place "O2E" or "O2O" directly in constraints. They belong in the "filters" array of the node.
If you use "NumChilds" or other size filters in the "sizeFilters" array, place them directly (e.g., {{ "type": "NumChilds", ... }}). Do NOT wrap them in a "SizeFilter" object unless you are placing them inside the "constraints" array.

Here is the logical plan (query tree) for the query.
LOGICAL PLAN (DO NOT USE THIS AS THE OUTPUT SCHEMA, JUST USE IT TO UNDERSTAND THE LOGIC AND HIERARCHY):
{tree_text}

EXTRACTED STRUCTURE (USE THESE EXACT EVENTS AND OBJECTS):
{json.dumps(structure_text, indent=2)}

WARNING: The examples below are from different domains. DO NOT copy their events (like "submit application") or objects (like "candidates"). You MUST map the logic from the examples to the events and objects in the EXTRACTED STRUCTURE above!

Use these examples for the EXACT output schema:
{few_shot_string}

Now generate the final OCPQ JSON for the exact user query: '{query}'
ONLY output valid JSON.
"""
    messages.append({"role": "user", "content": prompt_3})
    messages[0] = {
        "role": "system",
        "content": "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided schema.",
    }

    response_3 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=2000,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    content = response_3.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    if ocpq_json is None:
        allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)
        append_failure_log(
            stage="chain_of_thought_parse_failed",
            query=query,
            messages=messages,
            raw_output=content,
            parsed_json=None,
            normalized_json=None,
            allowed_events=allowed_events,
            allowed_objects=allowed_objects,
            validation_error="Could not parse model output as JSON",
        )
    messages.append({"role": "assistant", "content": content})
    print("Sub-problem 3 completed: OCPQ JSON generated.")

    if schema_validation:
        ocpq_json = verify_ocpq_json(
            ocpq_json,
            selected_model,
            messages,
            query=query,
            schema_enforced=schema_enforced,
        )
        if ocpq_json is None:
            raise ValueError("Generated OCPQ JSON could not be validated against the schema.")

        ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
        if ocpq_json is None:
            raise ValueError("Refusing to execute: OCPQ JSON failed strict Pydantic validation.")
    elif ocpq_json is None:
        raise ValueError("Generated OCPQ JSON could not be parsed.")

    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_result_file_path = None
    query_stderr = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path,
            output_file=output_file,
            query=query,
            messages=messages,
            stage="chain_of_thought_plus",
        )

        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_4 = (
                f"The generated JSON query caused this execution error:\n{query_stderr}\n\n"
                "Please fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
            )
            fixed_json, fixed_content = schema_fix_with_llm(
                model_name=selected_model,
                messages=messages,
                prompt=prompt_4,
                schema_enforced=schema_enforced,
            )

            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path,
                    output_file=output_file,
                    query=query,
                    messages=messages,
                    stage="chain_of_thought_plus_fix",
                )
                ocpq_json = fixed_json
                messages.append({"role": "user", "content": prompt_4})
                messages.append({"role": "assistant", "content": fixed_content})

    return structure_text, tree_text, ocpq_json, query_result, query_result_file_path


def _run_pipeline_least_to_most(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    print("Starting Least-to-Most NL to OCPQ pipeline...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)
    messages = [
        {
            "role": "system",
            "content": "You are a process mining expert that converts natural language into Object-Centric Process Queries (OCPQ). You must think step by step and respond concisely according to the prompts.",
        }
    ]

    prompt_1 = f"""
You are extracting the structure of a process query.

Given a natural language query, identify:

1. Events involved
2. Objects involved
3. Relationships between events (before, after, etc.)
4. Cardinality constraints (e.g., exactly once, at least once)
5. Whether nesting is needed (yes/no)

The events and objects must be chosen from the schema extracted from the log:
{schema_block}

Return ONLY JSON in this format:

{{
  "events": [...],
  "objects": [...],
  "relations": [...],
  "constraints": [...],
  "requires_nesting": true/false
}}

Query:
{query}
"""
    messages.append({"role": "user", "content": prompt_1})
    response_1 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "extracted_structure", "schema": EXTRACTED_STRUCTURE_SCHEMA, "strict": True},
        },
    )
    structure_text = response_1.choices[0].message.content
    messages.append({"role": "assistant", "content": structure_text})
    print("Sub-problem 1 completed: Entities extracted.")

    prompt_2 = f"""
You are formalizing a process query.

Given this extracted structure:

{structure_text}

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
    messages.append({"role": "user", "content": prompt_2})
    response_2 = call_hf_with_retry(model=selected_model, messages=messages, max_tokens=1000, response_format=None)
    formal_tree = response_2.choices[0].message.content
    messages.append({"role": "assistant", "content": formal_tree})
    print("Sub-problem 2 completed: Tree structure planned.")

    prompt_3 = f"""
You are designing an OCPQ query tree.

Given this extracted structure:

{formal_tree}

Decide:

- Number of nodes
- What each node represents
- Parent-child relationships

The example below shows the FORMAT ONLY, with 2 nodes and 1 edge. Your actual
tree may need more or fewer nodes/edges depending on the query — include ALL
nodes and edges you decided on above, not just this many.

Return ONLY JSON:

{{
  "nodes": [
    {{
      "id": 0,
      "description": "...",
      "events": [...],
      "objects": [...]
    }},
    {{
      "id": 1,
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
    messages.append({"role": "user", "content": prompt_3})
    response_3 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_tree", "schema": QUERY_TREE_SCHEMA, "strict": True},
        },
    )
    tree_text = response_3.choices[0].message.content
    messages.append({"role": "assistant", "content": tree_text})
    print("Sub-problem 3 completed: Query tree designed.")

    few_shot_string = build_few_shot_examples(query, dataset, embeddings)
    prompt_4 = f"""
You are generating OCPQ JSON.

CRITICAL SCHEMA REQUIREMENT:
Output ONLY valid JSON matching the exact schema shown in the examples.
The root object MUST contain exactly two top-level keys: "nodes" and "edgeNames".
Each node in the "nodes" array MUST be an object with a single key "Box".
The "Box" value MUST always be an array with exactly TWO elements:
1. The node data dictionary (containing newEventVars, newObjectVars, filters, sizeFilters, constraints, evVarLabels, obVarLabels).
2. An array of child node indices (e.g., [], [1]).
"edgeNames" MUST be an array of [[parent_index, child_index], "label"] pairs, one per parent-child link used in "nodes" (e.g., [[0, 1], "A"]). Do NOT omit this field.

CONSTRAINT SCHEMA:
The constraints in your JSON MUST ONLY use the following top-level variants:
"Filter", "SizeFilter", "SAT", "ANY", "NOT", "OR", "AND".
Basic filter logic like "TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter", "ObjectAttributeValueFilter", "BasicFilterCEL" must be wrapped inside a {{ "type": "Filter", "filter": {{ ... }} }} constraint.
Do NOT place "O2E" or "O2O" directly in constraints. They belong in the "filters" array of the node.
If you use "NumChilds" or other size filters in the "sizeFilters" array, place them directly (e.g., {{ "type": "NumChilds", ... }}). Do NOT wrap them in a "SizeFilter" object unless you are placing them inside the "constraints" array.

Here is the logical plan (query tree) for the query.
LOGICAL PLAN (DO NOT USE THIS AS THE OUTPUT SCHEMA, JUST USE IT TO UNDERSTAND THE LOGIC AND HIERARCHY):
{tree_text}

EXTRACTED STRUCTURE (USE THESE EXACT EVENTS AND OBJECTS):
{json.dumps(structure_text, indent=2)}

WARNING: The examples below are from different domains. DO NOT copy their events (like "submit application") or objects (like "candidates"). You MUST map the logic from the examples to the events and objects in the EXTRACTED STRUCTURE above!

Use these examples for the EXACT output schema:
{few_shot_string}

Now generate the final OCPQ JSON for the exact user query: '{query}'
ONLY output valid JSON.
"""
    messages.append({"role": "user", "content": prompt_4})
    messages[0] = {
        "role": "system",
        "content": "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided schema.",
    }

    response_4 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=2000,
        response_format=_build_response_format("ocpq_schema", OCPQ_JSON_SCHEMA, schema_enforced),
    )
    content = response_4.choices[0].message.content
    ocpq_json = parse_llm_json(content)
    if ocpq_json is None:
        allowed_events, allowed_objects = _extract_allowed_entities_from_messages(messages)
        append_failure_log(
            stage="least_to_most_parse_failed",
            query=query,
            messages=messages,
            raw_output=content,
            parsed_json=None,
            normalized_json=None,
            allowed_events=allowed_events,
            allowed_objects=allowed_objects,
            validation_error="Could not parse model output as JSON",
        )
    messages.append({"role": "assistant", "content": content})
    print("Sub-problem 4 completed: OCPQ JSON generated.")

    if schema_validation:
        ocpq_json = verify_ocpq_json(
            ocpq_json,
            selected_model,
            messages,
            query=query,
            schema_enforced=schema_enforced,
        )
        if ocpq_json is None:
            raise ValueError("Generated OCPQ JSON could not be validated against the schema.")

        ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
        if ocpq_json is None:
            raise ValueError("Refusing to execute: OCPQ JSON failed strict Pydantic validation.")
    elif ocpq_json is None:
        raise ValueError("Generated OCPQ JSON could not be parsed.")

    prompt_5 = f"""
You are verifying an OCPQ query for correctness and performance.

User query:
{query}

Generated OCPQ JSON:
{ocpq_json}

Please refer to the reasoning steps in the following dataset examples to understand how query bounds and constraints are properly established:
{few_shot_string}

CRITICAL CHECKS TO PREVENT OOM (QUERY EXPLOSION):
1. Are all events explicitly bound to objects using `O2E` filters in the `filters` array? (Unbound events cause Cartesian explosions).
2. Are child paths properly bounded by `TimeBetweenEvents`, `SAT`, `ANY`, or size constraints to prevent infinite graph search?
3. Does the structure match the logical reasoning patterns demonstrated in the examples?

Explain whether the query correctly captures the intent AND safely limits the search space.

If NOT or if it risks a query explosion:
- explain the mistake in the issues list
- output a corrected JSON

Return JSON:
{{
  "is_correct": true/false,
  "issues": ["Issue 1...", "Issue 2..."],
  "fixed_query": {{...}}
}}
"""
    messages.append({"role": "user", "content": prompt_5})

    response_5 = call_hf_with_retry(
        model=selected_model,
        messages=messages,
        max_tokens=1000,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "verification_schema", "schema": VERIFICATION_JSON_SCHEMA, "strict": True},
        }
    )
    verification_text = response_5.choices[0].message.content or ""
    messages.append({"role": "assistant", "content": verification_text})
    print("Sub-problem 5 completed: Query tree verified.")

    verification_json = parse_llm_json(verification_text)
    if not isinstance(verification_json, dict):
        print("Verification response was not valid JSON; defaulting to generated query.")
        verification_json = {"is_correct": True, "issues": ["Invalid verifier output"], "fixed_query": ocpq_json}

    if verification_json.get("is_correct", False):
        with open(output_file, "w") as f:
            json.dump(ocpq_json, f, indent=2)
    else:
        print("LLM indicated the query might be incorrect. Check the verification response for details:\n", verification_json)
        with open(output_file, "w") as f:
            json.dump(verification_json.get("fixed_query", ocpq_json), f, indent=2)

    query_result = None
    query_result_file_path = None
    query_stderr = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path,
            output_file=output_file,
            query=query,
            messages=messages,
            stage="least_to_most",
        )

        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_6 = (
                f"The generated JSON query caused this execution error:\n{query_stderr}\n\n"
                "Please fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a markdown code block."
            )
            fixed_json, fixed_content = schema_fix_with_llm(
                model_name=selected_model,
                messages=messages,
                prompt=prompt_6,
                schema_enforced=schema_enforced,
            )

            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path,
                    output_file=output_file,
                    query=query,
                    messages=messages,
                    stage="least_to_most_fix",
                )
                ocpq_json = fixed_json
                messages.append({"role": "user", "content": prompt_6})
                messages.append({"role": "assistant", "content": fixed_content})

    return structure_text, verification_json, ocpq_json, query_result, query_result_file_path


def nl_to_ocpq_pipeline(
    query,
    dataset,
    embeddings,
    output_file="test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    prompt_strategy=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
):
    strategy = normalize_prompt_strategy(prompt_strategy)
    if strategy == "schema-allowed-entities-only":
        return _run_pipeline_allowed_entities_only(
            query,
            dataset,
            embeddings,
            output_file,
            ocel_path,
            model_name,
            schema_enforced=schema_enforced,
            schema_validation=schema_validation,
            execute_query=execute_query,
        )
    if strategy == "schema-few-shot":
        return _run_pipeline_few_shot(
            query,
            dataset,
            embeddings,
            output_file,
            ocel_path,
            model_name,
            schema_enforced=schema_enforced,
            schema_validation=schema_validation,
            execute_query=execute_query,
        )
    if strategy == "schema-chain-of-thought-plus":
        return _run_pipeline_chain_of_thought_plus(
            query,
            dataset,
            embeddings,
            output_file,
            ocel_path,
            model_name,
            schema_enforced=schema_enforced,
            schema_validation=schema_validation,
            execute_query=execute_query,
        )
    if strategy == "schema-least-to-most":
        return _run_pipeline_least_to_most(
            query,
            dataset,
            embeddings,
            output_file,
            ocel_path,
            model_name,
            schema_enforced=schema_enforced,
            schema_validation=schema_validation,
            execute_query=execute_query,
        )
    if strategy == "schema-grammar-guided":
        return _run_pipeline_grammar_guided(
            query,
            dataset,
            embeddings,
            output_file,
            ocel_path,
            model_name,
            schema_enforced=schema_enforced,
            schema_validation=schema_validation,
            execute_query=execute_query,
        )

    raise ValueError(
        f"Unsupported prompt strategy: {prompt_strategy}. Supported: {', '.join(PROMPT_STRATEGIES)}"
    )


def generate_query_result_summary(result_data, max_rows_per_node=100):
    """
    Generate a human-readable summary from the query result JSON.
    
    Args:
        result_data: The parsed JSON result from the query executor (from query_result.json)
        max_rows_per_node: Maximum situations to include per node in preview
    
    Returns:
        dict with keys: metrics, node_overview, situations_preview
    """
    summary = {
        "metrics": {},
        "node_overview": [],
        "situations_preview": {},
        "status": "error"
    }
    
    if "Err" in result_data:
        summary["error"] = result_data.get("Err", "Unknown error")
        return summary
    
    if "Ok" not in result_data:
        summary["error"] = "Unexpected result shape"
        return summary
    
    ok_data = result_data.get("Ok", {})
    eval_data = ok_data.get("evaluationResults", [])
    object_ids = ok_data.get("objectIds", [])
    event_ids = ok_data.get("eventIds", [])
    bindings_skipped = ok_data.get("bindingsSkipped", False)
    
    if not isinstance(eval_data, list):
        summary["error"] = "No evaluation results in response"
        return summary
    
    # Calculate metrics
    total_situations = 0
    total_violations = 0
    for node in eval_data:
        count = node.get("situationCount", 0)
        violated = node.get("situationViolatedCount", 0)
        total_situations += count
        total_violations += violated
    
    summary["metrics"] = {
        "nodes": len(eval_data),
        "total_situations": total_situations,
        "total_violations": total_violations,
        "satisfaction_rate": f"{(100.0 * (total_situations - total_violations) / total_situations):.1f}%" if total_situations else "N/A",
        "bindings_skipped": bindings_skipped,
    }
    
    # Node overview table
    for idx, node in enumerate(eval_data):
        count = node.get("situationCount", 0)
        violated = node.get("situationViolatedCount", 0)
        summary["node_overview"].append({
            "node": idx,
            "situation_count": count,
            "violated_count": violated,
            "satisfaction_rate": f"{(100.0 * (count - violated) / count):.1f}%" if count else "N/A",
        })
    
    # Situations preview for each node
    def _format_violation(violation):
        if violation is None:
            return "Satisfied"
        if isinstance(violation, str):
            return f"Violated: {violation}"
        if isinstance(violation, dict) and len(violation) == 1:
            k, v = next(iter(violation.items()))
            return f"Violated: {k} ({v})"
        return f"Violated: {violation}"
    
    for idx, node in enumerate(eval_data):
        node_data = node
        situations = node_data.get("situations", []) if isinstance(node_data, dict) else []
        rows = []
        
        for i, situation in enumerate(situations[:max_rows_per_node], start=1):
            if not isinstance(situation, list) or len(situation) != 2:
                continue
            binding, violation = situation
            event_map = binding.get("eventMap", []) if isinstance(binding, dict) else []
            object_map = binding.get("objectMap", []) if isinstance(binding, dict) else []
            label_map = binding.get("labelMap", []) if isinstance(binding, dict) else []
            
            event_parts = []
            for ev_var, ev_idx in event_map:
                ev_name = event_ids[ev_idx] if isinstance(ev_idx, int) and 0 <= ev_idx < len(event_ids) else f"index:{ev_idx}"
                event_parts.append(f"e{ev_var + 1}={ev_name}")
            
            object_parts = []
            for ob_var, ob_idx in object_map:
                ob_name = object_ids[ob_idx] if isinstance(ob_idx, int) and 0 <= ob_idx < len(object_ids) else f"index:{ob_idx}"
                object_parts.append(f"o{ob_var + 1}={ob_name}")
            
            label_parts = []
            for item in label_map:
                if isinstance(item, list) and len(item) == 2:
                    label_parts.append(f"{item[0]}={item[1]}")
            
            rows.append({
                "situation": i,
                "status": _format_violation(violation),
                "events": ", ".join(event_parts) if event_parts else "-",
                "objects": ", ".join(object_parts) if object_parts else "-",
                "labels": ", ".join(label_parts) if label_parts else "-",
            })
        
        summary["situations_preview"][f"node_{idx}"] = {
            "total_situations": len(situations),
            "preview_rows": rows,
            "preview_limit": max_rows_per_node,
        }
    
    summary["status"] = "ok"
    return summary
