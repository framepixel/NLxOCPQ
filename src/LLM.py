#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import typing
import hashlib
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

import paths
from ocpq_schema import AGENT_TOOL_CALLS, AGENT_TURN_SCHEMA, AGENT_TURN_WITH_THOUGHT_SCHEMA, AgentTurn, AgentTurnWithThought  # noqa: E402
from ocpq_schema import CheckTreeCall, DEFECT_HINTS, PROBE_TOOL_NAMES, agent_turn_model, blocking_defects, defect_kind, structural_defects as _tree_structural_defects  # noqa: E402
from ocpq_schema import OCPQ_JSON_SCHEMA, ROOT_ARITY_SCHEMA, AGENT_ACTION_SCHEMA, AgentAction, AGENT_ACTION_WITH_THOUGHT_SCHEMA, AgentActionWithThought, BindingBoxTree, RootArity, VERIFICATION_JSON_SCHEMA, EXTRACTED_STRUCTURE_SCHEMA, QUERY_TREE_SCHEMA

from LogEvents import events_and_objects_from_ocel, extract_ocel_schema
# The agentic arm's tool surface (notes 43). Imported as a module, not as names, so the tool
# registry the prompt is generated from and the dispatcher the probe runs can never be two lists.
import ocpq_probe

# In[2]:

def _corpus_fingerprint(path=None):
    path = paths.DATA / 'ocpq_dataset.json' if path is None else path
    """Content hash of the retrieval corpus, used to detect stale embeddings."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_resources():
    # Load the dataset
    with open(paths.DATA / 'ocpq_dataset.json', 'r') as f:
        dataset = json.load(f)

    # Load the generated embeddings
    embeddings = np.load(paths.DATA / 'embeddings.npy')

    embedding_error = None

    # Guard against embeddings generated from a DIFFERENT corpus. Row counts alone cannot
    # catch this: `ocpq_dataset.json` was regenerated at the same 20 000 rows, so the old
    # (20000, 384) array still lines up and every retrieval would silently return the
    # example that was similar in the OLD corpus, at that index in the NEW one. Compare a
    # content hash written by Embeddings.py instead, and degrade to the documented
    # non-embedding fallback rather than retrieving nonsense.
    meta_path = paths.DATA / 'embeddings.meta.json'
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if meta.get('corpus_sha256') != _corpus_fingerprint():
                embedding_error = (
                    'embeddings.npy was built from a different ocpq_dataset.json '
                    '(corpus hash mismatch) - re-run Embeddings.py'
                )
        except Exception as exc:  # noqa: BLE001 - a bad meta file must not be fatal
            embedding_error = f'could not verify embeddings freshness: {exc}'
    else:
        embedding_error = (
            'embeddings.meta.json is missing, so embeddings.npy cannot be checked against '
            'ocpq_dataset.json - re-run Embeddings.py to create it'
        )
    if embedding_error:
        print(
            'Warning: embedding retrieval disabled. '
            'Falling back to non-embedding example retrieval. '
            f'Reason: {embedding_error}'
        )
        return dataset, embeddings, None, None, embedding_error

    # Load the local embedding model so we can embed user queries on the fly.
    tokenizer = None
    model = None
    try:
        local_dir = str(paths.MODELS / 'local-all-MiniLM-L6-v2')
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
# Consulted only when sampling is active. See `set_generation_seed`.
generation_seed = None
# When True, generation is constrained to the OCPQ grammar so invalid output cannot be produced.
# Default False: every run recorded before 2026-09-02 generated freely, and the flag must not
# change what those runs meant. See `set_constrained_decoding`.
constrained_decoding = False
# xgrammar's `any_order` for that grammar. True is what every recorded constrained run used. False
# (the agentic arm, notes 43.7) makes property order generation order and forbids a key twice,
# which `any_order=True` silently admits. See `set_constrained_any_order`.
constrained_any_order = True
# Tightens that grammar further (the agentic arm, notes 43.8): no free whitespace between JSON
# tokens, and every list in a tree bounded at twice the largest gold list. Off by default, so every
# recorded run is unchanged. See `set_constrained_bounded_shape`.
constrained_bounded_shape = False
# Overrides the per-call `max_tokens` the pipelines pass. None keeps their own values.
generation_token_budget = None
# ORACLE control: `(root_index, n_event_vars, n_object_vars)` taken from gold, or None. Tightens
# the constrained-decoding grammar so the root box must declare exactly that many variables.
# See `set_forced_root_arity` -- any run using it produces a ceiling, never system performance.
forced_root_arity = None
# NON-ORACLE per-case constraint: the log whose declared attribute types the value filters must
# respect, or None. Unlike `forced_root_arity` this comes from the log schema, not gold.
type_aware_filters_log = None
# NON-ORACLE per-case switch: before execution, check the generated tree for defects ocpq_cli
# cannot evaluate (unbound variables, inconsistent edges, unknown child names) and spend ONE
# repair call on them. Default False, so every run recorded before 2026-09-14 is unchanged.
# See `set_structural_validation`.
structural_validation = False
# What that check found and did on the most recent verified tree, or None. Per case.
_last_structural_report = None
# Per-call override: constrain generation to a DIFFERENT Pydantic model than the OCPQ tree. Used
# by the arity predictor, which asks a narrow question and wants a two-integer answer, not a query.
# None keeps the OCPQ grammar. See `set_constrained_schema`.
constrained_schema_override = None
# The agentic arm (notes 43). `agent_probe` is the log-inspection surface -- an `ocpq_probe.OcpqProbe`
# owned by `eval_testset`, or None for the compute-matched control that runs the same loop with no
# tools. `agent_max_turns` caps model turns per case and IS the difference between the A1 and A2
# rungs. `_last_agent_trajectory` is per case, read back like `_last_structural_report`.
agent_probe = None
agent_max_turns = 8
# Whether agentic turns may carry a short `thought` (notes 43, the reasoning condition). Off by
# default, which is exactly the protocol every run before 2026-09-23 used.
agent_thought = False
# Per-TURN generation cap for the agent loop, or None to use the run's budget (notes 43.8). Applied
# where generation happens (`_effective_max_tokens`), because the run-wide budget overrides every
# caller's `max_tokens` there and would silently swallow a cap passed from the loop.
agent_turn_token_budget = None
# True only while an agent turn is generating, so the per-turn cap cannot reach any other call --
# the verifier, the post-execution repair, or the one-shot rung.
_agent_turn_active = False
# The Stage 5 protocol levers (notes 43.12). Each is off by default, so every run before them --
# the model gate included -- is reproduced exactly. See `set_agent_protocol`.
AGENT_PROTOCOL_DEFAULTS = {
    "structural_feedback": False,     # F: structure report on try_query; the check_tree tool
    "bounce_blocking_final": False,   # B: a final tree with blocking defects is returned ONCE
    "finalise_evaluated": False,      # E: fall back to the last tree the probe EVALUATED first
    "worked_example": False,          # X: how a tree links boxes, and one worked example
    "reserve_last_turn": False,       # L: the last turn admits a final answer only
    "cap_feedback": False,            # C: a turn cut off at the per-turn cap is told so (43.15)
    "repeat_guard": False,            # R: an action identical to an earlier one is not re-run (43.17)
}
agent_protocol = dict(AGENT_PROTOCOL_DEFAULTS)
_last_agent_trajectory = None

LLM_CALL_STATS = []
QUERY_EXEC_STATS = []

local_model = None
local_tokenizer = None
local_model_key = None
local_gpu_device = "cuda"
local_gpu_dtype = "auto"
local_gpu_quantization = "4bit"
# Per-model chat settings (notes 43.16), set from a model's config entry. All three are empty for
# every model before Muse Glimmer, which leaves the local path byte-identical to what every recorded
# run used (tests/test_model_chat_options.py).
local_model_revision = None          # a hub commit, so every job loads the same weights
local_chat_template_kwargs = {}      # extra variables for the chat template
local_generation_prefix = None       # text appended after the generation prompt

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
        new_key = _local_model_key(hf_model, hf_token, new_device, new_dtype, new_quantization)
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


def set_generation_seed(value):
    """Seed the next sampled generation, or None to leave the RNG alone.

    Sampling was previously unseeded: `torch.manual_seed` is never called anywhere in this
    project, and the `seed` key in eval_config only feeds `_load_testset`'s shuffle. That is
    fine for a greedy run -- `do_sample` is False, so the RNG is not consulted and the output is
    deterministic regardless. It stops being fine the moment temperature > 0, which is what the
    self-consistency experiment needs: the samples would differ (good) but the run would not be
    reproducible (not good enough for a thesis result).

    Set per (case, condition, repeat) rather than once per process, so a sample's identity does
    not depend on how many cases ran before it -- otherwise a shard boundary would silently
    change the draw, and the 4-way case-shard split would not reproduce a single-job run.

    None leaves the global RNG untouched, reproducing the pre-2026-08-29 behaviour exactly.
    """
    global generation_seed
    generation_seed = None if value is None else int(value)


def set_constrained_decoding(value):
    """Constrain generation to the OCPQ grammar, so invalid output cannot be produced.

    Until now the schema has only ever been applied *after* generation: `schema_enforced` gates
    `verify_ocpq_json`, which parses the model's text, validates it against `BindingBoxTree`, and
    on failure spends a second LLM call asking the model to fix itself. That repair pass runs on
    2043 cases and fails on 1613 of them (79%), each ending as a score of 0. `response_format`
    never reached `generate()` on the local backend at all -- it only ever appended the sentence
    "Return only valid JSON", identically in both settings.

    This is the other thing entirely: at every decoding step the grammar says which tokens could
    still lead to a valid `BindingBoxTree`, and the rest are masked to -inf. Detection becomes
    prevention.

    Off by default, and deliberately so: it changes what the model can emit, so a run that had it
    on is not comparable to one that did not. `Thesis/notes/11-evaluation-readiness.md` 29-30.
    """
    global constrained_decoding
    constrained_decoding = bool(value)


def get_constrained_decoding():
    return constrained_decoding


def set_constrained_any_order(value):
    """Compile the decoding grammar with xgrammar's `any_order` on (default) or off.

    `any_order=True` lets object properties come in any order, which was adopted because gold
    writes some out of schema order (tests/test_constrained_decoding.py). It also admits the SAME
    key any number of times -- `"filters": [], "filters": [], ...` -- and the 2026-09-24 gate smoke
    on base Llama 3.1 8B fell into exactly that loop on 58 of 70 turns, each to the 4000-token
    limit, timing out all 10 cases. The ordered grammar forbids a repeated key and still expresses
    every gold tree once it is written in schema order (169/169 testset, 38/38 dev set), which is
    the only order a constrained model can write anyway. Notes 43.7.

    None means the default (True), so a config that does not mention it is unchanged.
    """
    global constrained_any_order
    constrained_any_order = True if value is None else bool(value)


def get_constrained_any_order():
    return constrained_any_order


# The largest list of each kind in any gold tree (testset 169 + dev set 38, measured 2026-09-26),
# doubled. Doubling keeps the bound far from anything gold needs -- every gold tree is admitted,
# asserted in tests/test_constrained_decoding.py -- while a list that repeats its items forever (the
# 2026-09-26 smoke: one empty box x60, a child list `[0, 0, 0, ...]`) is cut off. Lists of type
# names per variable are deliberately NOT bounded: gold binds up to 67 event types to one variable,
# and the per-turn token cap bounds those instead.
OCPQ_TREE_MAX_ITEMS = {
    "nodes": 16,         # gold max 8
    "edgeNames": 16,     # gold max 7
    "filters": 14,       # gold max 7
    "constraints": 8,    # gold max 4
    "sizeFilters": 2,    # gold max 1
    "labels": 2,         # gold max 1
    "children": 4,       # gold max 2 (the List[int] half of a Box)
    "child_names": 4,    # gold max 2 (SAT/NOT/OR/AND/ANY)
}
# What a variable name must look like in a box's variable and label maps -- see `_bounded_schema`.
OCPQ_VARIABLE_KEY_PATTERN = "^[0-9]+$"


def set_constrained_bounded_shape(value):
    """Tighten the decoding grammar: no free whitespace, and bounded lists (see OCPQ_TREE_MAX_ITEMS).

    Two more holes the ordered grammar (notes 43.7) left open, both measured in the 2026-09-26 gate
    smoke on base Llama 3.1 8B: JSON admits unlimited whitespace between tokens, and one case wrote
    398 visible characters and ~3900 tokens of blanks; and a list may repeat its items without
    end. `any_whitespace=False` keeps the standard `", "` / `": "` separators and nothing else;
    `maxItems` bounds each tree list. Every gold tree still passes both. Notes 43.8.
    """
    global constrained_bounded_shape
    constrained_bounded_shape = bool(value)


def get_constrained_bounded_shape():
    return constrained_bounded_shape


def _bounded_schema(schema):
    """`schema` (a JSON-schema dict) with OCPQ_TREE_MAX_ITEMS applied to every BindingBoxTree in it.

    Works on any schema that embeds the tree -- `BindingBoxTree` itself and the agent's turn union,
    whose `$defs` carry the same definitions -- and leaves one without them unchanged. A copy.
    """
    schema = copy.deepcopy(schema)
    defs = schema.get("$defs", {})
    tree = defs.get("BindingBoxTree", schema if "nodes" in schema.get("properties", {}) else None)
    if tree is not None:
        tree["properties"]["nodes"]["maxItems"] = OCPQ_TREE_MAX_ITEMS["nodes"]
        tree["properties"]["edgeNames"]["maxItems"] = OCPQ_TREE_MAX_ITEMS["edgeNames"]
    box_data = defs.get("BindingBoxData", {}).get("properties", {})
    for key in ("filters", "sizeFilters", "constraints", "labels"):
        if key in box_data:
            box_data[key]["maxItems"] = OCPQ_TREE_MAX_ITEMS[key]
    node_box = defs.get("BindingBoxTreeNodeBox", {}).get("properties", {}).get("Box")
    if node_box is not None:
        node_box["prefixItems"][1]["maxItems"] = OCPQ_TREE_MAX_ITEMS["children"]
    for definition in defs.values():
        child_names = definition.get("properties", {}).get("child_names")
        if child_names is not None:
            child_names["maxItems"] = OCPQ_TREE_MAX_ITEMS["child_names"]
    # Variable names are numbers in quotes: OCPQ refuses anything else ("expected key to be a
    # number in quotes"; ocpq_cli panics on it). The schema admits any string, and in the
    # 2026-09-26 smoke both base models named variables "Event"/"Object" and resubmitted the same
    # unparseable tree. All 652 variable keys in gold are numeric. Notes 43.9.
    #
    # `patternProperties` + `additionalProperties: false`, NOT `propertyNames`: xgrammar 0.2.5
    # honours `propertyNames` on the keys but then DROPS the value schema, so every value became
    # free JSON -- `{"1": {"type": "X"}}` where a list of type names belongs, label values outside
    # IGNORED/INCLUDED/EXCLUDED. That shipped in 9b11b8d and wrecked the full model gate (87% / 46%
    # malformed turns, notes 43.10). This form enforces the key AND keeps the value schema;
    # tests/test_constrained_decoding.py checks both directions.
    for key in ("newEventVars", "newObjectVars", "evVarLabels", "obVarLabels"):
        if key in box_data and "additionalProperties" in box_data[key]:
            value_schema = box_data[key].pop("additionalProperties")
            box_data[key]["patternProperties"] = {OCPQ_VARIABLE_KEY_PATTERN: value_schema}
            box_data[key]["additionalProperties"] = False
    return schema


def set_constrained_schema(model):
    """Constrain the next generation to `model` instead of the OCPQ tree, or None to restore it.

    The grammar machinery is per-call state exactly as `set_generation_seed` and
    `set_forced_root_arity` are, and for the same reason: which schema applies depends on what is
    being asked, so it must not be process-global.

    Only consulted when constrained decoding is on. Callers are expected to clear it in a
    `finally`, because a leaked override would silently constrain the NEXT query generation to a
    two-integer object and fail every case after it.
    """
    global constrained_schema_override
    constrained_schema_override = model


def get_constrained_schema():
    return constrained_schema_override


def set_forced_root_arity(value):
    """Pin how many event/object variables the ROOT box may declare, or None to leave it free.

    **This is an ORACLE control.** The arity comes from the gold query, so any condition using it
    produces a ceiling and can never be reported as system performance. `eval_testset` owns gold
    and records `forced_root_arity` on every record, exactly as it does for `structure_hint`.

    Set per (case, condition, repeat) rather than once per process, for the same reason
    `set_generation_seed` is: the constraint depends on the case, so a shard boundary must not be
    able to change which grammar a case was generated under.

    Only takes effect when constrained decoding is on -- it is a tightening of that grammar, not a
    separate mechanism. `None` restores the plain schema grammar.

    Args:
        value: `(n_event_vars, n_object_vars)` plus the root's index, as
            `(root_index, n_event_vars, n_object_vars)`, or None.
    """
    global forced_root_arity
    forced_root_arity = None if value is None else tuple(int(v) for v in value)


def get_forced_root_arity():
    return forced_root_arity


def set_type_aware_filters(ocel_path):
    """Constrain attribute filters to the declared types of `ocel_path`, or None to leave free.

    **Not an oracle**, and the distinction matters for how any result is reported: the types come
    from the log schema, which a deployment has, where `set_forced_root_arity` takes its number
    from the gold query. A run using only this remains system performance.

    Per case for the same reason the others are -- the constraint depends on which log the case
    uses -- though the compiled grammar caches on the log, so it holds at most one entry per log.
    """
    global type_aware_filters_log
    type_aware_filters_log = ocel_path or None


def get_type_aware_filters():
    return type_aware_filters_log


def set_structural_validation(enabled):
    """Switch the pre-execution structural check, and its single repair call, on or off.

    **Not an oracle.** The check reads only the generated tree, and the repair prompt carries only
    that tree and the defects found in it -- nothing from gold. A rung using it is system
    performance.

    Set per case like the other switches, and it clears the per-case report, so a case can never
    report what the previous case's check found.
    """
    global structural_validation, _last_structural_report
    structural_validation = bool(enabled)
    _last_structural_report = None


def get_structural_validation():
    return structural_validation


def get_last_structural_report():
    """What the structural check found and did on the most recent verified tree; None when off."""
    return _last_structural_report


def set_agent_probe(probe):
    """Hand the agentic strategies their log-inspection surface, or None for no tools.

    **Not an oracle.** `ocpq_probe` reads the log and the model's own candidate trees; it cannot
    reach gold, which `tests/test_ocpq_probe.py` asserts on the module source. The probe is created
    and owned by `eval_testset` (it owns a server process), for the same reason gold-derived
    controls are set there: this module never manages run-level resources.

    None is what makes the compute-matched control possible -- `agentic-self-refine` is the same
    loop, the same turn budget and the same prompt with the tool surface withheld, so a gain over
    it is attributable to log access rather than to extra generations (notes 43).
    """
    global agent_probe
    agent_probe = probe


def get_agent_probe():
    return agent_probe


def set_agent_max_turns(value):
    """Cap how many model turns one agentic case may take. This IS the rung.

    A1 and A2 differ in nothing else, so the number is a measured variable and belongs in the
    config rather than here (CLAUDE.md 4.2). It also has to stay inside `case_timeout_sec`, which
    CLAUDE.md 6 requires to be identical on every arm being compared.
    """
    global agent_max_turns
    agent_max_turns = max(1, int(value))


def get_agent_max_turns():
    return agent_max_turns


def set_agent_thought(value):
    """Let agentic turns carry a `thought` -- a CONDITION, crossed on the tool rung only.

    Switches both the action schema (`AgentActionWithThought`, which the grammar needs because
    `strict_mode` forbids unknown keys) and one line of the protocol rules, so the rungs differ in
    exactly those two things. See `ocpq_schema.AgentActionWithThought` for why a protocol field and
    not native reasoning.
    """
    global agent_thought
    agent_thought = bool(value)


def get_agent_thought():
    return agent_thought


def set_agent_protocol(**levers):
    """Set the Stage 5 protocol levers (notes 43.12); anything not named goes back to its default.

    Replacing rather than merging, so a case can never inherit a lever from a previous condition.
    An unknown name raises: a misspelt lever would silently run the wrong condition.
    """
    global agent_protocol
    unknown = set(levers) - set(AGENT_PROTOCOL_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown agent protocol lever(s): {sorted(unknown)}")
    agent_protocol = {name: bool(levers.get(name, default))
                      for name, default in AGENT_PROTOCOL_DEFAULTS.items()}


def get_agent_protocol():
    return dict(agent_protocol)


def set_agent_turn_token_budget(value):
    """Cap each agent TURN at `value` tokens, or None to use the run's generation budget.

    A turn that loops (a list repeating its items, a string that never closes) runs to the cap, and
    at the run's 4000 tokens that is ~95 s -- six such turns spend a case's 600 s (notes 43.7-43.8).
    The cap bounds the cost of a runaway, not the length of an answer: it is sized so every gold
    tree fits with margin, which `tests/test_agent_turn_budget.py` checks with the evaluated
    models' own tokenizers. The one-shot rung never enters the loop and keeps the run budget.
    """
    global agent_turn_token_budget
    agent_turn_token_budget = None if value is None else int(value)


def get_agent_turn_token_budget():
    return agent_turn_token_budget


def _effective_max_tokens(max_tokens):
    """The `max_tokens` generation actually uses: the run budget overrides the caller's value, and
    inside an agent turn the per-turn cap overrides both. One function, so the local backend and
    the tests read the same rule."""
    if generation_token_budget is not None:
        max_tokens = generation_token_budget
    if _agent_turn_active and agent_turn_token_budget is not None:
        max_tokens = agent_turn_token_budget
    return max_tokens


def get_last_agent_trajectory():
    """Every turn of the most recent agentic case, or None when the last case was not agentic.

    Read by `eval_testset` immediately after the pipeline returns, exactly as
    `get_last_structural_report` is, and recorded on the case record: the trajectory is the
    mechanism evidence the arm exists to produce, not decoration.
    """
    return _last_agent_trajectory


def reset_last_agent_trajectory():
    """Forget the previous case's trajectory. Called by `eval_testset` before every agentic case.

    Without it a case that dies before the loop starts (a probe that failed to load, an exception
    in schema extraction) would read back the PREVIOUS case's trajectory and record it as its own.
    """
    global _last_agent_trajectory
    _last_agent_trajectory = None


def set_generation_token_budget(value):
    """Override every pipeline's `max_tokens`, or None to leave their own values alone.

    The pipelines hard-code 1000/2000 across eight call sites. Truncation is a *measured* failure
    mode distinct from schema violation -- failed generations run median 794 completion tokens
    against 357 for successful ones and pile up at the 2000 cap -- so the budget has to be
    movable to separate the two. Applied at the single point where generation happens rather than
    at the call sites, so no pipeline signature changes and nothing can bypass it.
    """
    global generation_token_budget
    generation_token_budget = None if value is None else int(value)


def get_generation_token_budget():
    return generation_token_budget


def get_generation_seed():
    return generation_seed


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


def set_local_model_options(revision=None, chat_template_kwargs=None, generation_prefix=None):
    """Set the per-model chat settings for the local backend (notes 43.16); None clears each one.

    Called for every model before it is selected, so a model without the settings never inherits
    another model's. Set BEFORE `set_local_gpu_client`: the revision is part of the cached model's
    key, and a key computed without it would keep serving the wrong weights.
    """
    global local_model_revision, local_chat_template_kwargs, local_generation_prefix
    local_model_revision = revision or None
    local_chat_template_kwargs = dict(chat_template_kwargs or {})
    local_generation_prefix = generation_prefix or None


def preload_local_model():
    """Load (and quantize) the selected local model now; seconds taken, 0 if it was loaded already.

    Loading is otherwise lazy -- on the first generation -- which puts it inside the first case's
    time limit. For a pre-quantized 8 GB checkpoint that is seconds; for Muse Glimmer's 60 GB of bf16
    read from $HPCWORK and quantized to NF4 it is minutes, charged to whichever case comes first
    (notes 43.17). Called by `eval_testset` only when the config sets `preload_model`. None when the
    backend is not local.
    """
    if hf_backend != "local-gpu":
        return None
    started = time.time()
    _ensure_local_gpu_model_loaded()
    return round(time.time() - started, 3)


def _local_model_key(model, token, device, dtype, quantization):
    """What identifies the loaded local model. One function for both places that compare it."""
    return (model, token, device, dtype, quantization, local_model_revision)


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
    # ORACLE conditions, for the structure ablation only. Both receive part of the GOLD query as
    # a `structure_hint` built by eval_testset, so their scores are ceilings and can never be
    # reported as system performance. The `oracle-` prefix is load-bearing: it is what makes that
    # unmissable in a config, a log line, a run directory name and a config_id.
    "oracle-skeleton-given",
    "oracle-skeleton-and-constraints-given",
    # The agentic arm (notes 43). NON-ORACLE, both of them: the loop reads the log and its own
    # candidate trees, never gold. `-self-refine` is the compute-matched control -- the same loop
    # and the same turn budget with the tool surface withheld -- so that a gain for `-log-tools`
    # is attributable to log access rather than to spending more generations.
    "agentic-log-tools",
    "agentic-self-refine",
]
ORACLE_PROMPT_STRATEGIES = ("oracle-skeleton-given", "oracle-skeleton-and-constraints-given")
AGENTIC_PROMPT_STRATEGIES = ("agentic-log-tools", "agentic-self-refine")
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
# every model: giving a 256k-context model more schema than a smaller one would mean the models
# see different prompts, which confounds the model-vs-model comparison the thesis rests on.
# Nothing else caps the prompt — the local backend tokenizes without `max_length` — so this is
# the only guard.
#
# Caps the schema SECTION, not the whole prompt. The largest rendered block across the eleven
# evaluation logs is 10,493 tokens (angular_github_commits_ocel.xml; bpic2017 is 4,846, every
# other log is under 1,500), so any ceiling from ~11k upward renders identical prompts. 17000 is
# headroom above that measurement, not a whole-prompt figure.
#
# The previous 3500 was sized for Llama 3 8B's 8k context and silently dropped sections for
# angular and bpic2017, contradicting the design stated in eval_testset.py — that every model
# always receives the complete schema, richness not being a design variable. With Llama 3 and
# Phi-4 replaced, the smallest context in the set is Gemma 3 4B at 128k (Llama 3.1 128k,
# Ministral 3 256k), so the block no longer needs capping below its natural size.
SCHEMA_BLOCK_TOKEN_BUDGET = int(os.getenv("OCPQ_SCHEMA_TOKEN_BUDGET", "17000"))

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
    # largest: bpic2017 declares 494 of them and angular 277. Ordered last so that if the budget
    # ever does bind, this is the section that goes rather than the entity or relation lists.
    # At the current 17000 budget nothing is dropped for any of the eleven evaluation logs.
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
        "skeleton-given": "oracle-skeleton-given",
        "oracle-skeleton-given": "oracle-skeleton-given",
        "skeleton-and-constraints-given": "oracle-skeleton-and-constraints-given",
        "oracle-skeleton-and-constraints-given": "oracle-skeleton-and-constraints-given",
        "agentic": "agentic-log-tools",
        "agentic-log-tools": "agentic-log-tools",
        "log-tools": "agentic-log-tools",
        "agentic-self-refine": "agentic-self-refine",
        "self-refine": "agentic-self-refine",
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
            parsed = ast.literal_eval(py_content)
        except Exception as e:
            print(f"Failed to decode JSON. Error: {e}\nRaw output:\n{content}")
            return None

        # `literal_eval` accepts Python literals that JSON has no concept of. The one that bites
        # is `...`, which models emit constantly as an abbreviation placeholder: it evaluates to
        # `Ellipsis`, so the result *looks* like a parsed dict and only fails later, deep inside
        # `json.dumps`, as "Object of type ellipsis is not JSON serializable". That surfaced as an
        # unhandled `pipeline_error` -- our defect -- rather than the `parse_failed` it really is,
        # and it hit 2 of the first 29 runs of the 2026-08-16 smoke test.
        #
        # A round-trip through `json.dumps` catches the whole family (ellipsis, sets, bytes,
        # complex) rather than special-casing the one we happened to see.
        try:
            json.dumps(parsed)
        except (TypeError, ValueError) as exc:
            print(f"Parsed output is not JSON-serialisable ({exc}); treating as unparseable.\n"
                  f"Raw output:\n{content}")
            return None
        return parsed


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


def is_merged_model_dir(path) -> bool:
    """Does this directory hold full model weights, as opposed to a LoRA adapter?

    Public because it is a contract, not a detail: anything that PRODUCES a checkpoint for this
    pipeline has to agree with what LOADS one. `merge_gemma.py` imports it for exactly that
    reason. On 2026-08-19 the two disagreed -- the merge script accepted sharded safetensors that
    this check did not list -- and all 48 Gemma smoke conditions failed with "not recognized as a
    merged model or a PEFT adapter" after the checkpoints had passed every verification step.

    Sharded safetensors (`model-00001-of-00002.safetensors` + `model.safetensors.index.json`) is
    the ordinary output format for any model past a few GB; `save_pretrained` shards by default.
    Omitting it meant this pipeline could only load models small enough to fit in one file, which
    was never the intent.
    """
    import glob
    import os

    single = (
        "model.safetensors",
        "pytorch_model.bin",
        "pytorch_model-00001-of-00002.bin",
        # Index files are the canonical marker of a sharded checkpoint.
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    if any(os.path.exists(os.path.join(path, name)) for name in single):
        return True
    # Shards without an index: unusual, still loadable by transformers.
    return bool(
        glob.glob(os.path.join(path, "model-*-of-*.safetensors"))
        or glob.glob(os.path.join(path, "pytorch_model-*-of-*.bin"))
    )


def _load_text_generation_model(model_name, **kwargs):
    """Load a checkpoint for text generation under whichever auto class it is registered for.

    `AutoModelForCausalLM` is not universal. Multimodal checkpoints may be registered only under
    `AutoModelForImageTextToText`, and which ones varies by transformers release: on 5.5.0
    Gemma 3 loads fine as a causal LM while Ministral 3 raises

        Unrecognized configuration class Mistral3Config for this kind of AutoModel:
        AutoModelForCausalLM

    That killed all 48 Ministral runs of the 2026-08-15 smoke test -- a third of the matrix
    producing nothing. Because the split does not follow architecture families, this reacts to
    the actual error rather than encoding a mapping of which models are multimodal, which would
    drift with every transformers upgrade.

    The image-text-to-text wrapper generates from `input_ids` alone; no pixel inputs are needed.
    """
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise exc
        print(
            f"[model load] {model_name} is not registered for AutoModelForCausalLM; "
            "loading via AutoModelForImageTextToText (multimodal checkpoint)."
        )
        try:
            return AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
        except ValueError as exc2:
            # Muse Glimmer's card loads it with AutoModelForMultimodalLM (transformers 5.x), and
            # it may be registered only there (notes 43.16). Reached only when BOTH classes
            # above reject the config, so no model that loaded before takes this path.
            if "Unrecognized configuration class" not in str(exc2):
                raise
            try:
                from transformers import AutoModelForMultimodalLM
            except ImportError:
                raise exc2
            print(f"[model load] {model_name} is not registered for AutoModelForImageTextToText "
                  "either; loading via AutoModelForMultimodalLM.")
            return AutoModelForMultimodalLM.from_pretrained(model_name, **kwargs)


def _as_tokenizer(tokenizer_or_processor):
    """Return the tokenizer, unwrapping a processor if one came back.

    `AutoTokenizer` on a multimodal repo can yield a processor whose first positional argument is
    `images`, not `text` -- the failure the Ministral training spike hit. Unwrapping keeps every
    downstream call (`apply_chat_template`, `__call__`, `pad_token`) operating on a tokenizer.
    """
    inner = getattr(tokenizer_or_processor, "tokenizer", None)
    if inner is not None and hasattr(inner, "apply_chat_template"):
        print(f"[model load] unwrapped {type(tokenizer_or_processor).__name__} -> "
              f"{type(inner).__name__}")
        return inner
    return tokenizer_or_processor


def _ensure_local_gpu_model_loaded():
    global local_model
    global local_tokenizer
    global local_model_key

    model_key = _local_model_key(hf_model, hf_token, local_gpu_device, local_gpu_dtype,
                                 local_gpu_quantization)
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
        merged_weights = is_merged_model_dir(hf_model)

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

            # This field decides which weights the LoRA is injected into, and for Gemma 3 it does
            # NOT name the checkpoint the adapter was trained against. unsloth substitutes its own
            # pre-quantized redistribution when load_in_4bit=True, then
            # `train_unsloth._restore_requested_base_model_name` redirects this field back to the
            # requested checkpoint.
            #
            # That divergence is deliberate and unavoidable, not an oversight. Pointing Gemma at
            # the artifact it was actually trained on was tried on 2026-08-16 (job 3015021) and
            # both its arms failed to generate a token: bitsandbytes raises "FP4 quantization
            # state not initialized" and asserts inside fix_4bit_weight_quant_state_from_module.
            # The redistribution is unusable for inference. Llama and Ministral are unaffected —
            # their requested checkpoint already IS the unsloth artifact, so nothing is redirected
            # and train and eval agree.
            #
            # Consequence to disclose: Gemma's LoRA is applied to weights quantized by bnb rather
            # than by unsloth. Both are nf4 over the same checkpoint and should be close, but they
            # are not provably identical. See models/adapters/*/ocpq_provenance.json.

            # Load the base model with 4-bit quantization so its Params4bit layers are
            # properly initialised. The adapter's LoRA A/B weights are then injected
            # manually below, bypassing PeftModel.from_pretrained entirely.
            # Background: unsloth saves adapters in a format that embeds quantized base-
            # model layers alongside the LoRA deltas. PeftModel.from_pretrained restores
            # those quantized layers via bitsandbytes but with quant_state=None (version
            # mismatch between training and inference bitsandbytes), causing every forward
            # pass to raise a bare AssertionError with empty message. Loading only the
            # LoRA A/B tensors (keys containing "lora_") sidesteps the issue entirely.
            base = _load_text_generation_model(base_model_name, **auth_kwargs, **model_kwargs)
            tokenizer = _as_tokenizer(AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True, **auth_kwargs))

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
                # Any unmatched key means the fine-tuning is only partly applied, which produces
                # a silently degraded model rather than an error -- the worst outcome in an
                # evaluation. Both working adapters report 0 unexpected, so nothing legitimate is
                # being rejected here. A multimodal wrapper nests the decoder (e.g. under
                # `model.language_model.*`), so a name shift shows up as unexpected keys; the
                # sample below is what a prefix mapping would be written from.
                if _unexpected:
                    _sample = ", ".join(_unexpected[:3])
                    raise RuntimeError(
                        f"Adapter only partially applied: {_loaded}/{len(_lora_sd)} tensors "
                        f"matched, {len(_unexpected)} did not. Unmatched keys look like: "
                        f"{_sample}. Loading would give a half-fine-tuned model."
                    )
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
            tokenizer = _as_tokenizer(AutoTokenizer.from_pretrained(hf_model, use_fast=True, trust_remote_code=True, **auth_kwargs))
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = _load_text_generation_model(hf_model, **auth_kwargs, **model_kwargs)

        else:
            raise RuntimeError(
                f"Model directory '{hf_model}' not recognized as a merged model or a PEFT adapter."
            )

    else:
        # Normal HF hub id or remote model. A pinned revision (notes 43.16) reaches the tokenizer
        # and the weights alike; without one the kwargs are exactly what every earlier run passed.
        rev_kwargs = {"revision": local_model_revision} if local_model_revision else {}
        tokenizer = _as_tokenizer(AutoTokenizer.from_pretrained(hf_model, use_fast=True, trust_remote_code=True, **auth_kwargs, **rev_kwargs))
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model = _load_text_generation_model(hf_model, **auth_kwargs, **rev_kwargs, **model_kwargs)

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


class PromptTooLongError(RuntimeError):
    """The prompt alone exceeds the model's context, so no generation budget can rescue it."""


def _check_context_budget(prompt_tokens, max_tokens):
    """Guard the context window: raise when hopeless, warn when merely tight.

    Nothing in this path truncates: the tokenizer is called without `max_length` and
    `generation_config.max_length` is set to None, so an oversized prompt is not clipped —
    it degrades or errors during generation. Without this the failure would surface only as
    unexplained garbage output in the results.

    Two distinct cases, and conflating them is what made the 2026-08-15 smoke test spend five
    OOM retries on a single case:

    * **Prompt alone over the limit** — unrecoverable. `call_hf_with_retry` responds to OOM by
      shrinking `max_tokens`, which cannot help when the *prompt* is 1.2M tokens against a 131k
      context. Raising here turns a five-attempt death spiral (ending in a 1,444 GiB allocation
      request) into one fast, correctly-classified failure.
    * **Prompt plus generation budget over the limit** — recoverable by generating less, which
      is exactly what the retry loop does. Warn and let it proceed.
    """
    limit = _model_context_limit()
    if not limit:
        return
    if prompt_tokens > limit:
        raise PromptTooLongError(
            f"Prompt is {prompt_tokens} tokens against a {limit}-token context: no generation "
            "budget can make this fit. Usually runaway conversation history rather than the "
            f"schema block (currently capped at {SCHEMA_BLOCK_TOKEN_BUDGET} tokens)."
        )
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


# Compiled grammar + tokenizer info, keyed by tokenizer identity. Compiling is cheap (~0.04 s)
# but building the token-vocabulary mask is not, and both are invariant across every call in a
# run, so doing either per call would dominate generation time.
_XGRAMMAR_CACHE = {}


# Which `value_filter` variants are admissible for an attribute of each declared type. Integer and
# Float are mutually permitted because gold uses Float bounds on integer-ish attributes and the
# distinction is not one the log's own typing makes reliably; Time permits String because OCEL
# serialises timestamps as strings.
VALUE_FILTER_FOR_DECLARED = {
    "integer": ("ValueFilterInteger", "ValueFilterFloat"),
    "float": ("ValueFilterFloat", "ValueFilterInteger"),
    "boolean": ("ValueFilterBoolean",),
    "string": ("ValueFilterString",),
    "time": ("ValueFilterTime", "ValueFilterString"),
}
_ATTRIBUTE_FILTERS = {
    "FilterObjectAttributeValueFilter": "object_attributes",
    "FilterEventAttributeValueFilter": "event_attributes",
}


def _type_aware_schema(schema, ocel_path):
    """Constrain each attribute filter's `value_filter` to the attribute's DECLARED type.

    Built for notes 34.3's reading that the model collapses comparisons into string membership
    (`Age` declared Integer, "older than 40" emitted as `String{is_in: ["40"]}`). **That reading
    was withdrawn in notes 35.4**: every such disagreement sat on an attribute the cluster's stale
    corpus declared as a string, so the prompt really did say String there. On the corrected
    corpus the model is type-correct on 8 of 9 attribute filters without this constraint, and the
    typed ablation was not run (notes 35.6). Off by default; kept as tested machinery.

    **This is not an oracle.** The types come from the log schema, which any deployment has. That is
    what separates it from the arity forcing of notes 32.

    Grouped by declared TYPE rather than by attribute: one branch per type (at most five), each
    carrying an enum of the names that have it. Branching per attribute would need one branch per
    name, and bpic2017 has 494 (event type, name) pairs. Grouping by name is safe because no
    attribute name carries conflicting declared types in any of the 25 logs -- checked, not assumed.

    Returns the schema unchanged when the log declares no typed attributes, so a log without them
    costs nothing and cannot fail.
    """
    try:
        log_schema = extract_ocel_schema(ocel_path)
    except Exception as exc:  # noqa: BLE001 - a constraint we cannot build is one we skip
        print(f"   type-aware filters unavailable for {ocel_path}: {exc}")
        return schema

    defs = schema["$defs"]
    replacements = {}
    for filter_def, attribute_key in _ATTRIBUTE_FILTERS.items():
        if filter_def not in defs:
            continue
        by_type = {}
        for attributes in (log_schema.get(attribute_key) or {}).values():
            for attribute in attributes:
                declared = str(attribute.get("type", "")).lower()
                if declared in VALUE_FILTER_FOR_DECLARED:
                    by_type.setdefault(declared, set()).add(attribute["name"])
        if not by_type:
            continue
        branches = []
        for declared, names in sorted(by_type.items()):
            branch = copy.deepcopy(defs[filter_def])
            branch["properties"]["attribute_name"] = {"enum": sorted(names)}
            branch["properties"]["value_filter"] = {
                "anyOf": [{"$ref": f"#/$defs/{v}"} for v in VALUE_FILTER_FOR_DECLARED[declared]]}
            name = f"{filter_def}__{declared}"
            defs[name] = branch
            branches.append({"$ref": f"#/$defs/{name}"})
        replacements[f"#/$defs/{filter_def}"] = branches

    if not replacements:
        return schema

    def rewrite_union(node):
        """Swap each attribute-filter ref for its typed branches, wherever the union appears."""
        if isinstance(node, dict):
            for key in ("anyOf", "oneOf"):
                variants = node.get(key)
                if isinstance(variants, list) and any(
                        v.get("$ref") in replacements for v in variants if isinstance(v, dict)):
                    rebuilt = []
                    for variant in variants:
                        ref = variant.get("$ref") if isinstance(variant, dict) else None
                        rebuilt.extend(replacements[ref] if ref in replacements else [variant])
                    node[key] = rebuilt
                    # The discriminator maps a `type` literal to ONE ref, and every branch here
                    # keeps the same literal. Leaving it would be a mapping that no longer
                    # describes the union -- drop it rather than let it drift.
                    node.pop("discriminator", None)
            for value in node.values():
                rewrite_union(value)
        elif isinstance(node, list):
            for value in node:
                rewrite_union(value)

    rewrite_union(schema)
    return schema


def _arity_pinned_schema(root_index, n_event_vars, n_object_vars):
    """The OCPQ schema with the ROOT box's variable counts pinned, every other node left free.

    Pinning `BindingBoxData` directly does not work and the feasibility gate caught it: that
    definition is shared by every node, so pinning it forces child boxes to the root's arity too
    and rejects **141 of the 169** gold queries, since a child usually binds something different
    from its parent. The root is therefore given its own `$def` and reached positionally through
    `prefixItems`, with `items` left generic so the node COUNT stays free -- this experiment forces
    root arity, not tree shape.

    `root_index` is not assumed to be 0. It is 0 for 166 of the 169 gold queries and 1 or 2 for the
    rest, and a valid tree may list a child before its parent.

    Verified over the whole testset: gold is accepted under its own arity 169/169 and rejected
    under a neighbouring arity 169/169 (`tests/test_forced_arity.py`).
    """
    schema = copy.deepcopy(OCPQ_JSON_SCHEMA)
    defs = schema["$defs"]

    pinned = copy.deepcopy(defs["BindingBoxData"])
    for field, count in (("newEventVars", n_event_vars), ("newObjectVars", n_object_vars)):
        pinned["properties"][field] = dict(
            pinned["properties"][field], minProperties=count, maxProperties=count)
    defs["BindingBoxDataRoot"] = pinned

    root_node = copy.deepcopy(defs["BindingBoxTreeNodeBox"])
    root_node["properties"]["Box"]["prefixItems"][0] = {"$ref": "#/$defs/BindingBoxDataRoot"}
    defs["BindingBoxTreeNodeBoxRoot"] = root_node

    generic = {"$ref": "#/$defs/BindingBoxTreeNodeBox"}
    schema["properties"]["nodes"] = {
        "type": "array",
        "prefixItems": [generic] * root_index + [{"$ref": "#/$defs/BindingBoxTreeNodeBoxRoot"}],
        "items": generic,
        "minItems": root_index + 1,
    }
    return schema


def _ocpq_logits_processor():
    """A logits processor that masks any token which cannot continue a valid `BindingBoxTree`.

    Built from the Pydantic model itself, so the grammar constraining generation and the validator
    in `validate_and_normalize_ocpq_json` are one definition and cannot drift apart.

    Two settings are load-bearing and were established by `tests/test_constrained_decoding.py`:

    * `any_order=True` -- gold writes object properties out of schema order (`value_filter` before
      `at_time` in `FilterObjectAttributeValueFilter`). With ordered matching the grammar rejects
      **32 of the 169 gold queries**, which would cap the model below gold on 19% of the testset
      and read as a finding rather than a bug. That counts gold as WRITTEN; in schema order the
      ordered grammar admits all 169. And `any_order=True` admits a repeated key, which base
      models loop on. So it is the default, not a constant: `set_constrained_any_order(False)`
      compiles the ordered grammar instead (the agentic arm, notes 43.7).
    * `strict_mode=True` -- forbids unknown fields. Pydantic ignores them, so here the grammar is
      the stricter of the two, which is the safe direction: everything it admits validates.

    Returns None when constrained decoding is off or xgrammar is unavailable, so the caller adds
    nothing to `generate()` and the free-decoding path stays byte-identical.
    """
    if not constrained_decoding:
        return None
    try:
        import xgrammar as xgr
    except ImportError as exc:  # pragma: no cover - preflight catches this
        raise RuntimeError(
            "constrained decoding was requested but xgrammar is not installed. "
            "Add it to the eval environment (scripts/make_eval_env.sh); "
            "src/check_eval_deps.py checks for it."
        ) from exc

    # `forced_root_arity` is per case, so it is part of the key. Gold has only 6 distinct root
    # arities across 13 (root_index, arity) combinations in the whole testset, so this cache stays
    # tiny even though the grammar is nominally per-case.
    key = (id(local_tokenizer), id(local_model), forced_root_arity, type_aware_filters_log,
           getattr(constrained_schema_override, "__name__", None), constrained_any_order,
           constrained_bounded_shape)
    cached = _XGRAMMAR_CACHE.get(key)
    if cached is None:
        # The model's vocabulary is often LARGER than the tokenizer's, because checkpoints pad the
        # embedding matrix to a round size. The mask has to match the logits row, so its width
        # comes from the model config, not from the tokenizer -- get this wrong and the mask is
        # silently misaligned or `generate` raises on a shape mismatch.
        vocab_size = getattr(getattr(local_model, "config", None), "vocab_size", None)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            local_tokenizer, vocab_size=vocab_size)
        if constrained_schema_override is not None:
            # A narrow question (the arity predictor), not a query. The query-shaped constraints
            # are meaningless here and are deliberately ignored rather than combined.
            source = constrained_schema_override
        elif forced_root_arity is None and type_aware_filters_log is None:
            source = BindingBoxTree
        else:
            # The two transforms compose: both rewrite the same schema and touch disjoint parts of
            # it -- arity pins the root box's variable counts, type-awareness rewrites the
            # attribute-filter union. Applied in sequence so a rung can carry either or both.
            built = (_arity_pinned_schema(*forced_root_arity) if forced_root_arity
                     else copy.deepcopy(OCPQ_JSON_SCHEMA))
            if type_aware_filters_log is not None:
                built = _type_aware_schema(built, type_aware_filters_log)
            source = json.dumps(built)
        if constrained_bounded_shape:
            # Applied LAST, to whatever the source is, so it composes with every transform above.
            as_dict = json.loads(source) if isinstance(source, str) else source.model_json_schema()
            source = json.dumps(_bounded_schema(as_dict))
        cached = xgr.GrammarCompiler(tokenizer_info).compile_grammar(
            xgr.Grammar.from_json_schema(source, any_order=constrained_any_order, strict_mode=True,
                                         any_whitespace=not constrained_bounded_shape)
        )
        _XGRAMMAR_CACHE[key] = cached
    # The matcher carries per-generation state, so it is built fresh per call; the expensive
    # part -- compiling the grammar against the vocabulary -- is what the cache holds.
    return xgr.contrib.hf.LogitsProcessor(cached)


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
            **local_chat_template_kwargs,
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        if local_generation_prefix:
            # Opens the answer channel of a model whose generation prompt stops before it (Muse
            # Glimmer: `<|start|>assistant`, then ` to=user<|message|>` or a reasoning channel).
            # Constrained decoding would otherwise force `{` where the model expects a channel
            # header. The prefix is the text the template itself writes for a plain assistant
            # turn, so the model's current turn looks exactly like its history (notes 43.16).
            prefix_ids = local_tokenizer.encode(local_generation_prefix, add_special_tokens=False)
            input_ids = torch.cat(
                [input_ids, torch.tensor([prefix_ids], dtype=input_ids.dtype)], dim=-1)
        input_ids = input_ids.to(local_model.device)
        attention_mask = torch.ones_like(input_ids)
    else:
        prompt = _build_local_prompt(normalized_messages)
        model_inputs = local_tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(local_model.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(local_model.device)

    # Applied here rather than at the eight call sites that hard-code 1000/2000, so no pipeline
    # signature changes and nothing can bypass it. None leaves each caller's own value alone. The
    # agent's per-turn cap is applied at the same point, for the same reason.
    max_tokens = _effective_max_tokens(max_tokens)

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

    # Added ONLY when constrained decoding is on. With it off, `generate_kwargs` is byte-identical
    # to what every recorded run used -- asserted by tests/test_constrained_decoding.py, because a
    # flag that silently perturbs the free path would invalidate the baseline it is compared to.
    _grammar_processor = _ocpq_logits_processor()
    if _grammar_processor is not None:
        generate_kwargs["logits_processor"] = [_grammar_processor]

    # Seeded immediately before the call, not once per process: the draw must depend only on
    # (case, condition, repeat) so a shard boundary cannot change it. No-op when sampling is off,
    # where the RNG is not consulted at all.
    if do_sample and generation_seed is not None:
        torch.manual_seed(generation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(generation_seed)

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


# The question the arity predictor asks. Kept at module level because it IS the experiment: if the
# predictor underperforms a constant, the prompt is the first thing to look at, and it must be
# quotable in the write-up without digging through a call stack.
ROOT_ARITY_PROMPT = (
    # PROMPT v3. Structurally identical to v1 -- same length, same three POSITIVE examples, no
    # procedure list, no negative example -- and differing from it only in the example nouns.
    #
    # v2 changed two things at once and regressed badly (expected value 7.58 -> 5.82, below the
    # constant's 6.85): it swapped the vocabulary AND added a rule list plus an explicit
    # "wrong: n_event_vars=1, n_object_vars=1" counter-example. The error that counter-example
    # named went from 28 cases to 46, and abstentions tripled -- showing a wrong answer primed it.
    # v3 keeps the vocabulary fix, which is required for hygiene, and drops everything else, so
    # the comparison against v1 measures the vocabulary change alone.
    #
    # The nouns are a support-desk domain that appears in NO evaluation log; writing these in the
    # vocabulary of one (orders, customers, deliveries) would tune the prompt on the evaluation
    # set. The observed distribution of gold arities is likewise absent: it is a label statistic
    # of the testset. Both are asserted in tests/test_arity_predictor.py.
    "An OCPQ query is a tree of binding boxes. The ROOT box holds the query's ANSWER -- the rows "
    "the user gets back. Child boxes exist only to express conditions ABOUT those rows, and the "
    "variables they declare are NOT part of the answer.\n\n"
    "Decide what the ANSWER is about, not what the question mentions. Examples of the "
    "distinction:\n"
    "  \"Which tickets were escalated twice?\"           -> a set of tickets: 0 event variables, "
    "1 object variable. The escalation is a CONDITION, in a child.\n"
    "  \"Which escalations happened out of hours?\"      -> a set of events: 1 event variable, "
    "0 object variables.\n"
    "  \"Which agents handled a ticket reopened later?\" -> still 1 object variable; both the "
    "reopening and the ticket are conditions.\n\n"
    "Count ONLY the variables the root box itself declares.\n\n"
    "{schema_block}\n\n"
    "Question: {query}\n\n"
    "Answer with n_event_vars and n_object_vars for the ROOT box only."
)


def predict_root_arity(query, ocel_path, model_name, schema_block=None):
    """Predict `(n_event_vars, n_object_vars)` for the root box, from the question alone.

    The non-oracle half of notes 32. Forcing the root arity to GOLD's value moved result F1 from
    0.2162 to 0.3109, the largest effect in the project; this is the component that has to supply
    that number without gold.

    Answered through the same constrained decoder as everything else, against `RootArity`, so the
    reply is a parseable two-integer object by construction rather than prose to be regex'd. That
    matters more than usual here: a malformed answer means no prediction, and no prediction is
    strictly better than a wrong one -- forcing the wrong arity makes the correct query
    UNREACHABLE, whereas not forcing leaves the model its own 43.8% chance.

    Returns:
        `(n_event_vars, n_object_vars)`, or None when the model gave nothing usable. The caller is
        expected to treat None as "do not force".
    """
    if schema_block is None:
        schema_block = build_schema_block(extract_ocel_schema(ocel_path))
    messages = [
        {"role": "system", "content": "You are a process mining expert. Answer with JSON only."},
        {"role": "user", "content": ROOT_ARITY_PROMPT.format(
            schema_block=schema_block, query=query)},
    ]
    previous = get_constrained_schema()
    set_constrained_schema(RootArity)
    try:
        response = call_hf_with_retry(
            model=model_name,
            messages=messages,
            max_tokens=64,
            response_format=_build_response_format("root_arity", ROOT_ARITY_SCHEMA, True),
        )
    except Exception as exc:  # noqa: BLE001 - a failed prediction must not fail the case
        print(f"   arity prediction failed: {exc}")
        return None
    finally:
        # In a finally because a leaked override would constrain the NEXT query generation to a
        # two-integer object and fail every case after this one.
        set_constrained_schema(previous)

    parsed = parse_llm_json(response.choices[0].message.content or "")
    try:
        arity = RootArity.model_validate(parsed)
    except Exception:  # noqa: BLE001
        return None
    return arity.n_event_vars, arity.n_object_vars


def _build_response_format(schema_name, schema, enforce_schema=True):
    if not enforce_schema:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {"name": schema_name, "schema": schema, "strict": True},
    }

# In[7]:

# Upper bound on a single message's content, in characters. A well-formed OCPQ query serialises
# to 1-3 KB and the largest gold query in the testset is under 3 KB, so this never binds on a
# valid exchange. It exists for the pathological case: the chain-of-thought-plus pipeline and the
# repair loops append the model's own output back into the history, and one 2026-08-15 smoke case
# compounded that into a 1,245,363-token prompt.
MESSAGE_CONTENT_CHAR_CAP = int(os.getenv("OCPQ_MESSAGE_CHAR_CAP", "20000"))


def _clamp_message_history(messages):
    """Truncate over-long message contents, marking what was dropped.

    Applied at the single point where messages are dispatched to a model rather than at the ~10
    `messages.append` sites, so no future append can bypass it. Returns a new list; the caller's
    history is left intact, so nothing downstream sees a silently rewritten transcript.
    """
    clamped = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str) and len(content) > MESSAGE_CONTENT_CHAR_CAP:
            omitted = len(content) - MESSAGE_CONTENT_CHAR_CAP
            message = dict(message, content=(
                content[:MESSAGE_CONTENT_CHAR_CAP]
                + f"\n[... {omitted} characters omitted: message exceeded the "
                  f"{MESSAGE_CONTENT_CHAR_CAP}-character cap ...]"
            ))
        clamped.append(message)
    return clamped


def call_hf_with_retry(model, messages, max_tokens, response_format, max_retries=5):
    messages = _clamp_message_history(messages)
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
            except PromptTooLongError:
                # Never retried: the retry shrinks max_tokens, and the prompt is what does not
                # fit. Retrying cost five attempts and a 1,444 GiB allocation request once.
                raise
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

# Container fields of `BindingBoxData` whose absence has exactly ONE reading: nothing to apply.
# Filling them is recovering the model's obvious intent, not guessing at it.
_BOX_CONTAINER_DEFAULTS = {
    "filters": list,
    "sizeFilters": list,
    "constraints": list,
    "evVarLabels": dict,
    "obVarLabels": dict,
}
# Names that must NEVER be defaulted, kept as a list rather than left implicit because the reason
# is not obvious from the schema. `child_names` is the dangerous one: an empty child set satisfies
# `SAT` VACUOUSLY, so filling it with [] turns a compliance query into one that reports 0%
# violations while looking correct -- the exact defect Thesis/notes and the OCPQ notes already
# record. `newEventVars`/`newObjectVars` are excluded because declaring no variables is a
# meaningful statement rather than a neutral default, and 0 of the 1722 recorded validation
# failures needed them.
_NEVER_DEFAULTED = ("child_names", "newEventVars", "newObjectVars")

# What the last `validate_and_normalize_ocpq_json` call filled in, so the caller can record it.
# A normalisation is the harness doing work the model did not do; the number has to be visible.
_last_normalized_fields = []


def get_last_normalized_fields():
    """Fields filled by the most recent `validate_and_normalize_ocpq_json` call."""
    return list(_last_normalized_fields)


def _fill_missing_box_containers(ocpq_json):
    """Fill absent `BindingBoxData` container fields with their empty value.

    Narrow on purpose. Constrained decoding leaves a residue that is not the model's fault:
    xgrammar's `any_order=True` lets a REQUIRED property be skipped when an optional one is
    present, and in the 2026-09-10 run 10 of the 11 constrained failures were exactly one shape --
    a box carrying optional `labels` and omitting required `filters`. An absent `filters` is `[]`
    under OCPQ semantics, so recovering it costs nothing and loses nothing.

    Measured scope, from 1722 recorded validation failures across every run: 10 are fixable here,
    655 involve a field with no neutral default (`attribute_name`, `value_filter`, `object`,
    `is_in`, ...) and stay failures, and 1057 fail for other reasons entirely. This is a repair for
    a decoder artefact, not a general leniency -- see `_NEVER_DEFAULTED`.

    Returns the (possibly modified) document; the fields filled are recorded in
    `_last_normalized_fields`.
    """
    if not isinstance(ocpq_json, dict):
        return ocpq_json
    nodes = ocpq_json.get("nodes")
    if not isinstance(nodes, list):
        return ocpq_json

    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        box = node.get("Box")
        if not isinstance(box, list) or not box or not isinstance(box[0], dict):
            continue
        node_data = box[0]
        for field, factory in _BOX_CONTAINER_DEFAULTS.items():
            if field not in node_data:
                node_data[field] = factory()
                _last_normalized_fields.append(f"nodes[{index}].{field}")
    return ocpq_json


def validate_and_normalize_ocpq_json(ocpq_json):
    """
    Strictly validate against the Pydantic BindingBoxTree model.
    Returns a normalized JSON dict (aliases applied) when valid.
    """
    global _last_normalized_fields
    _last_normalized_fields = []
    try:
        ocpq_json = _normalize_constraints_shape(ocpq_json)
        ocpq_json = _fill_missing_box_containers(ocpq_json)
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

STRUCTURAL_REPAIR_PROMPT = """The OCPQ JSON below is valid against the schema, but the query engine cannot evaluate it as written. These defects stop it from being evaluated:
{defects}

Fix exactly these defects:
- A filter may only use an event or object variable that is declared (newEventVars / newObjectVars) in its own node or in one of that node's ancestors. Variables of sibling or child nodes are not visible. Declare the variable where it is used, or move the filter into a node where it is in scope.
- Every child index in a node's children array must refer to an existing node, each node has exactly one parent, and the tree has exactly one root.
- Change nothing that is not needed to fix these defects.

Return ONLY the fixed JSON.

JSON:
{tree}
"""


def _apply_structural_validation(ocpq_json, model_name, messages, query=None,
                                 schema_enforced=True):
    """The pre-execution structural check, on a tree that has already passed schema validation.

    Pydantic accepts trees that ocpq_cli cannot evaluate, and the CLI does not fail loudly on them:
    it writes `{"Err": ...}` to its export and exits 0. On 2026-09-13 that was 88 of the arity
    rungs' 507 executions; replayed, 61 of 68 distinct failing trees said "Could not get oN".

    Only BLOCKING defects -- the kinds the evaluator errors on, `ocpq_schema.BLOCKING_DEFECT_KINDS`
    -- trigger the repair, and only they are shown to the model. Advisory defects are recorded but
    never repaired: a tree carrying only those evaluates, and repairing it could only damage it.
    The repair is ONE call under whatever decoding constraints are active, and is kept only if it
    leaves strictly fewer blocking defects; one that fixes nothing, does not validate, or raises
    leaves the original tree untouched.

    Inert when the switch is off: the tree is returned as given and no report is written.
    """
    global _last_structural_report
    if not structural_validation or ocpq_json is None:
        return ocpq_json
    from ocpq_schema import blocking_defects, structural_defects

    defects = structural_defects(ocpq_json)
    blocking = blocking_defects(defects)
    report = {
        "defects_before": defects,
        "blocking_before": blocking,
        "repair_attempted": False,
        "defects_after": None,
        "blocking_after": None,
        "accepted": False,
        "repair_error": None,
    }
    _last_structural_report = report
    if not blocking:
        return ocpq_json

    report["repair_attempted"] = True
    prompt = STRUCTURAL_REPAIR_PROMPT.format(
        defects="\n".join(f"- {defect}" for defect in blocking),
        tree=json.dumps(ocpq_json),
    )
    try:
        fixed_json, _fixed_content = schema_fix_with_llm(
            model_name=model_name,
            messages=messages,
            prompt=prompt,
            schema_enforced=schema_enforced,
        )
    except Exception as exc:  # noqa: BLE001 - a failed repair keeps the tree; it never fails a case
        report["repair_error"] = f"{type(exc).__name__}: {exc}"
        return ocpq_json
    if fixed_json is not None:
        # Same post-processing as the verifier's own repair, so the two repairs are comparable.
        fixed_json = validate_and_normalize_ocpq_json(
            _normalize_time_between_events_seconds(fixed_json, query))
    if fixed_json is None:
        report["repair_error"] = "repair output did not validate against the schema"
        return ocpq_json

    remaining = structural_defects(fixed_json)
    report["defects_after"] = remaining
    report["blocking_after"] = blocking_defects(remaining)
    if len(report["blocking_after"]) < len(blocking):
        report["accepted"] = True
        return fixed_json
    return ocpq_json

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
        return _apply_structural_validation(
            fixed_json, model_name, messages, query=query, schema_enforced=schema_enforced)
    else:
        return _apply_structural_validation(
            normalized, model_name, messages, query=query, schema_enforced=schema_enforced)


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
    output_file="configs/test_query.json",
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
    output_file="configs/test_query.json",
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
    output_file="configs/test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
    structure_hint=None,
    stage_prefix="allowed_entities_only",
):
    """Single-shot generation from the schema, optionally given gold's tree shape.

    `structure_hint` serves the ORACLE conditions of the structure ablation and is built by
    `eval_testset._build_structure_hint`, which owns the gold query; nothing gold-derived is read
    here. Left at None the prompt and every failure-log stage name are byte-identical to what this
    function produced before the parameter existed, which `tests/test_structure_hint_prompt.py`
    asserts -- the ablation's S0 baseline is this same code path, so any drift would silently
    invalidate the comparison it exists to make.

    Parameterised rather than copied: the verifier stub, the execution-error repair pass and the
    failure logging below are subtle enough that a second copy would drift out of step with this
    one, and the two conditions would then differ by more than the hint.
    """
    print(f"Starting Allowed-Entities-Only NL to OCPQ pipeline"
          f"{' (ORACLE: structure given)' if structure_hint else ''}...")
    selected_model = model_name or hf_model
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)
    hint_block = f"\n{structure_hint}\n" if structure_hint else ""

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
{hint_block}
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
            stage=f"{stage_prefix}_parse_failed",
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
            stage=stage_prefix,
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
                    stage=f"{stage_prefix}_fix",
                )
                ocpq_json = fixed_json
                messages.append({"role": "user", "content": prompt_fix})
                messages.append({"role": "assistant", "content": fixed_content})

    return None, None, ocpq_json, query_result, query_result_file_path


def _run_pipeline_chain_of_thought_plus(
    query,
    dataset,
    embeddings,
    output_file="configs/test_query.json",
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
    output_file="configs/test_query.json",
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


# --- THE AGENTIC ARM (notes 43) -------------------------------------------------------------
#
# Every earlier intervention repaired a query BEFORE it ran and left what it answers unchanged
# (notes 31, 32, 37, 41 -- six times). This loop is the first that can act on what a candidate
# query SELECTS: the model runs a tree against the log through `ocpq_probe`, sees how many
# situations each box binds, and revises. The final tree is then executed and scored on exactly
# the path every other strategy uses, so the arm is comparable rather than a parallel evaluation.

# CLAUDE.md 2.3 wants raw model output recorded before post-processing, and the trajectory is where
# an agentic turn's raw output lives. Capped because a 12-turn case at A2's budget would otherwise
# add tens of KB per record to predictions.json.
_AGENT_RAW_CHAR_CAP = int(os.getenv("OCPQ_AGENT_RAW_CHAR_CAP", "8000"))

AGENT_PROTOCOL_RULES = (
    "Reply with ONE JSON object per turn and nothing else. Two forms are allowed:\n"
    "  {\"tool\": \"<name>\", \"args\": {...}}   to look something up in the log\n"
    "  {\"final_tree\": {...}}                    to answer with the finished OCPQ query\n"
    "Never put both in one reply. Never write prose around the JSON."
)
# Appended to the rules ONLY under `agent_thought`, so the reasoning condition differs from the
# plain one in this line and in the action schema, and in nothing else. Asks for brevity because a
# thought costs tokens every turn against the same turn budget and case_timeout_sec (notes 43).
AGENT_THOUGHT_RULE = (
    "You may START either form with \"thought\": \"...\" -- one or two sentences on what you "
    "expect and why -- before the action itself."
)


# The one-shot prompt's tree rules, verbatim (`_run_pipeline_allowed_entities_only`, rendered). The
# agent prompt first shipped without them, and in the 2026-09-23 gate smoke the model wrote a
# `filters` key in 0 of 12 turns (B0: O2E filters in 158/169 predictions), so A1 - A0 would have
# measured the missing rules as well as the tools. A copy rather than a shared reference because
# B0's prompt is a completed, pinned condition and is not refactored mid-experiment;
# `tests/test_agent_action_protocol.py` asserts this string equals B0's rendered rules exactly.
OCPQ_TREE_RULES = (
    "- The root object MUST contain both \"nodes\" and \"edgeNames\".\n"
    "- \"edgeNames\" is an array of [[parent_index, child_index], \"label\"] pairs, one per parent-child\n"
    "  link used in \"nodes\" (e.g. [[0, 1], \"A\"]). Do NOT omit it.\n"
    "- Basic filters (\"TimeBetweenEvents\", \"NotEqual\", \"EventAttributeValueFilter\",\n"
    "  \"ObjectAttributeValueFilter\", \"BasicFilterCEL\") must be wrapped as\n"
    "  {\"type\": \"Filter\", \"filter\": {...}} when placed in \"constraints\".\n"
    "- \"O2E\" and \"O2O\" belong in the node's \"filters\" array, never directly in \"constraints\".\n"
    "- Use ONLY event and object names from the allowed lists.\n"
    "- Do NOT invent unsupported filter types such as \"ExactlyOnce\".\n"
    "- Represent \"exactly once\" using supported schema structure, not a custom operator.\n"
    "- If a node contains both events and objects, include the required O2E filter(s).\n"
    "- Keep all nodes connected in a single tree and avoid empty nodes."
)


# X (notes 43.12). The model gate's 298 try_query trees: 178 list a node as its OWN child, 205 have
# child lists that disagree with edgeNames, 168 have the wrong number of roots -- box content is
# mostly right, the LINKING is not understood. Rules about the representation plus ONE worked
# example in an invented domain: `tests/test_agent_action_protocol.py` checks it validates, has no
# structural defect, and shares no type name with any corpus log and no tree with any gold.
AGENT_TREE_STRUCTURE_RULES = (
    "How a tree is linked (this is where most mistakes happen):\n"
    "- \"nodes\" is a list; node 0 is the ROOT.\n"
    "- Each node is {\"Box\": [box data, child list]}. The child list holds the indices of OTHER "
    "nodes that are this node's children -- never the node's own index.\n"
    "- Every node except node 0 appears in exactly one child list; node 0 appears in none.\n"
    "- \"edgeNames\" has exactly one entry [[parent, child], \"name\"] for each child link, and "
    "nothing else. SAT, NOT and NumChilds refer to a child by that name.\n"
    "- A filter may read the variables declared in its own box or in any ancestor box, and no "
    "others."
)
AGENT_WORKED_EXAMPLE_QUESTION = "Every book must be shelved at least once."
AGENT_WORKED_EXAMPLE = {
    "nodes": [
        {"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["book"]}, "filters": [],
                  "sizeFilters": [],
                  "constraints": [{"type": "SizeFilter", "filter": {
                      "type": "NumChilds", "child_name": "A", "min": 1, "max": None}}],
                  "evVarLabels": {}, "obVarLabels": {}}, [1]]},
        {"Box": [{"newEventVars": {"0": ["shelve book"]}, "newObjectVars": {},
                  "filters": [{"type": "O2E", "object": 0, "event": 0, "qualifier": None}],
                  "sizeFilters": [], "constraints": [], "evVarLabels": {}, "obVarLabels": {}},
                 []]},
    ],
    "edgeNames": [[[0, 1], "A"]],
}
AGENT_CHECK_TREE_SPEC = {
    "name": "check_tree",
    "args": {"tree": "a complete BindingBoxTree JSON object"},
    "description": ("Check how a candidate tree is linked, WITHOUT running it and without the log: "
                    "every structural defect (child lists, edgeNames, the root, variables in "
                    "scope), each with how to fix it."),
}
AGENT_LAST_TURN_NOTE = "This is your LAST turn: answer now with {\"final_tree\": {...}}."
# C (notes 43.15). Without it, a turn the harness cut at the cap is answered "do not truncate the
# object" -- blaming the model for a cut it did not make, and not saying the object was too LONG.
# On the angular log that was nearly every malformed turn of the scored run: ~70 event types listed
# in one variable, or a dozen full commit messages in one `is_in`.
AGENT_CAP_FEEDBACK = ("it reached the {cap}-token limit for one reply before its JSON object "
                      "closed, so it was cut off. The object is too long, not malformed: write a "
                      "much shorter one. Bind the one type each variable needs instead of listing "
                      "every type, and filter on a few values, not a long list copied from a "
                      "sample")


# R (notes 43.17). Muse Glimmer's model gate: without it, 172 of 302 turns repeated an earlier
# action of the same case exactly -- one try_query sent seven times -- and a 4.7k-character
# check_tree repeated six times timed a case out. Greedy decoding cannot leave that loop by
# itself: the repeat's result is the same, so the next context invites the same action again.
AGENT_REPEAT_NOTE = ("You already ran exactly this action in turn {turn}, and its result, shown "
                     "after that turn, has not changed. Running it again tells you nothing new. "
                     "Change the tree or the arguments to test something different, or answer "
                     "with {{\"final_tree\": {{...}}}}.")


def _turn_hit_cap(response, cap):
    """Did this turn stop at the token cap instead of finishing its object?

    An endpoint says so (`finish_reason == "length"`); the local backend reports only how many
    tokens it generated, and it stops at exactly `cap` when the cap is what stopped it.
    """
    choice = response.choices[0]
    if getattr(choice, "finish_reason", None) == "length":
        return True
    used = getattr(getattr(response, "usage", None), "completion_tokens", None)
    return used is not None and cap is not None and used >= cap
_STRUCTURE_REPORT_CAP = 8


def _agent_structure_block():
    """The X block: the linking rules, then the worked example on an invented log."""
    return (f"{AGENT_TREE_STRUCTURE_RULES}\n\n"
            f"Example, on a different log whose types are not in yours. Question: "
            f"\"{AGENT_WORKED_EXAMPLE_QUESTION}\" The root binds each book; the child binds that "
            f"book's shelve events; NumChilds min 1 makes a book with none a violation.\n"
            f"{json.dumps(AGENT_WORKED_EXAMPLE)}")


def _structure_report(tree):
    """Every structural defect of `tree`, blocking kinds first, each with its fix (F, notes 43.12).

    Static: reads only the model's own tree, never the log and never gold. Capped, with the number
    left out stated, so a badly broken tree cannot flood the transcript.
    """
    defects = _tree_structural_defects(tree)
    blocking = blocking_defects(defects)
    ordered = blocking + [d for d in defects if d not in blocking]
    report = [f"{d} -> fix: {DEFECT_HINTS[defect_kind(d)]}" if defect_kind(d) in DEFECT_HINTS
              else d for d in ordered[:_STRUCTURE_REPORT_CAP]]
    if len(ordered) > _STRUCTURE_REPORT_CAP:
        report.append(f"... and {len(ordered) - _STRUCTURE_REPORT_CAP} more")
    return report


def _tree_blocking_count(tree):
    """Blocking structural defects in a tree the agent wrote -- recorded on every turn that carries
    one, whatever the levers, so the structure rate is comparable across conditions."""
    return len(blocking_defects(_tree_structural_defects(tree)))


def _check_tree_result(tree):
    report = _structure_report(tree)
    if report:
        return {"structure": report}
    return {"structure": [], "note": "no structural defect found -- this checks how the tree is "
                                     "linked, not whether it answers the question"}


def _agent_protocol_rules():
    """The protocol rules for the current condition: the plain rules, plus the thought line."""
    return AGENT_PROTOCOL_RULES + ("\n" + AGENT_THOUGHT_RULE if agent_thought else "")


def _agent_action_model():
    """The PARSED-turn model for the current condition. The grammar is compiled from the stricter
    `AgentTurn` / `AgentTurnWithThought` union instead (see `_run_pipeline_agentic`)."""
    return AgentActionWithThought if agent_thought else AgentAction


class AgentNoFinalAnswer(ValueError):
    """The loop ended with no usable tree: the budget ran out, or every turn was malformed.

    Its own exception type because `eval_testset._classify_run_status` matches on message text, and
    a case that never reached execution must NOT be filed as `execution_failed` -- that would move
    `execution_success_rate`, which CLAUDE.md 6 requires to mean "the query ran".
    """


def _parse_agent_action(content, action_model=AgentAction):
    """Parse one protocol turn. Returns `(action, None)` or `(None, reason)`.

    `action_model` is the condition's schema (`_agent_action_model`). With the plain `AgentAction`
    a stray `thought` key -- possible only without constrained decoding -- is ignored by Pydantic,
    so it can never leak into the plain condition as anything but raw text.

    Deliberately NOT `parse_llm_json`, which is right for queries and wrong for a control channel:
    it takes a greedy `{.*}` span, so prose containing braces swallows everything between the first
    and last one, and it appends a closing `}` to truncated output -- which can turn a cut-off tool
    call into a DIFFERENT, valid action. A control channel that silently invents an action is worse
    than one that refuses, because the refusal is measurable (`agent_malformed_actions`) and the
    invention is not.

    `raw_decode` also stops at the end of the first complete object, so trailing prose after a
    valid action is tolerated rather than fatal.
    """
    if not content or not content.strip():
        return None, "empty reply"
    text = content.strip()
    if text.startswith("```"):                      # a fenced block is common and harmless
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text[3:]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    start = text.find("{")
    if start < 0:
        return None, "no JSON object in the reply"
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON ({exc.msg}); do not truncate the object"
    if not isinstance(parsed, dict):
        return None, "the reply must be a JSON object"
    has_tool, has_tree = bool(parsed.get("tool")), parsed.get("final_tree") is not None
    if has_tool and has_tree:
        return None, "exactly one of `tool` or `final_tree`, not both"
    if not has_tool and not has_tree:
        return None, "the object must carry either `tool` or `final_tree`"

    # Validated against the ONE variant the turn names, not the whole union: a union error lists
    # every branch's complaint, which tells the model nothing. And every tree first gets the repair
    # every other arm gets before validation (`_repair_agent_tree`) -- without it, trees the
    # one-shot pipeline accepts were rejected here, 0 of 10 gate cases declared an answer.
    del _last_agent_parse_normalized[:]
    if has_tree:
        tree, normalized = _repair_agent_tree(parsed.get("final_tree"))
        try:
            fields = {"final_tree": BindingBoxTree.model_validate(tree)}
        except Exception as exc:  # noqa: BLE001 - the model's mistake, reported back to it
            return None, f"`final_tree` is not a valid BindingBoxTree: {str(exc)[:400]}"
    else:
        tool = parsed.get("tool")
        variant = _AGENT_TOOL_CALL_BY_NAME.get(tool)
        if variant is None:
            return None, (f"unknown tool {tool!r}; the tools are "
                          f"{sorted(_AGENT_TOOL_CALL_BY_NAME)}")
        args = parsed.get("args")
        normalized = []
        if (tool in ("try_query", "check_tree") and isinstance(args, dict)
                and isinstance(args.get("tree"), dict)):
            repaired, normalized = _repair_agent_tree(args["tree"])
            args = dict(args, tree=repaired)
        try:
            call = variant.model_validate({"tool": tool, "args": {} if args is None else args})
        except Exception as exc:  # noqa: BLE001 - the model's mistake, reported back to it
            return None, f"malformed {tool} call: {str(exc)[:300]}"
        fields = {"tool": tool, "args": call.args.model_dump(by_alias=True)}
    if action_model is AgentActionWithThought and parsed.get("thought") is not None:
        fields["thought"] = str(parsed["thought"])
    _last_agent_parse_normalized.extend(normalized)
    return action_model(**fields), None


# Container fields the repair filled on the most recent parsed turn. Recorded per turn: the harness
# doing work the model did not do has to be visible (`normalized_fields` on every other arm).
_last_agent_parse_normalized = []

_AGENT_TOOL_CALL_BY_NAME = {
    typing.get_args(call.model_fields["tool"].annotation)[0]: call
    for call in AGENT_TOOL_CALLS + (CheckTreeCall,)}


def _repair_agent_tree(tree):
    """The one-shot pipeline's narrow container repair, applied to an agent's tree. Returns
    `(tree, fields_filled)`.

    `validate_and_normalize_ocpq_json` runs `_fill_missing_box_containers` before strict validation
    on every other arm: xgrammar's `any_order=True` lets a required container (`filters`, ...) be
    skipped, and an absent container is empty under OCPQ semantics. Applied to a COPY, and the
    global normalisation record is left as it was, so the case's own record is not disturbed.
    """
    if not isinstance(tree, dict):
        return tree, []
    before = len(_last_normalized_fields)
    repaired = _fill_missing_box_containers(copy.deepcopy(tree))
    filled = list(_last_normalized_fields[before:])
    del _last_normalized_fields[before:]
    return repaired, filled


def _agent_tool_block(probe):
    """The tool list the model is shown, generated from the probe's own registry.

    Generated rather than written out, because a tool described in the prompt but absent from the
    dispatcher is a silent dead end -- the model spends its budget calling something that can only
    ever answer "unknown tool".
    """
    def line(spec):
        arguments = ", ".join(f"{name} ({description})" for name, description in spec["args"].items())
        return f"- {spec['name']}({arguments}): {spec['description']}"

    structural = agent_protocol["structural_feedback"]
    if probe is None:
        if structural:
            # The control keeps the STATIC check: A1 - A1' then isolates log access (notes 43.12).
            return ("You have NO log tools in this condition. One tool you may call:\n"
                    + line(AGENT_CHECK_TREE_SPEC))
        return ("You have NO tools in this condition. Think it through and answer with "
                "{\"final_tree\": {...}}.")
    lines = ["Tools you may call:"] + [line(spec) for spec in ocpq_probe.TOOL_SPECS]
    if structural:
        lines.append(line(AGENT_CHECK_TREE_SPEC))
    return "\n".join(lines)


AGENT_SYSTEM_PROMPT = ("You are a JSON-only API that writes OCPQ queries. You answer with exactly one "
                       "JSON object per turn and never with prose.")


def _agent_task_prompt(schema_block, query, probe):
    """The first user message of an agentic case, for the current condition.

    With every Stage 5 lever off this renders exactly what the loop sent before them, so the model
    gate stays reproducible. `agent_prompt_sha256` hashes it with placeholders in place of the log
    schema and the question -- the part a Stage 5 freeze fixes.
    """
    levers = agent_protocol
    rules = ("Rules for every tree you write (in try_query and in final_tree):\n"
             f"{OCPQ_TREE_RULES}")
    if levers["worked_example"]:
        rules += "\n\n" + _agent_structure_block()
    extra = ""
    if levers["structural_feedback"]:
        # A check costs a turn: the tool rung is told try_query already reports the same defects,
        # so it need not spend one on a tree it is about to run.
        extra += ("\n- Check your tree with check_tree before you answer, and fix every defect it "
                  "reports." if probe is None else
                  "\n- try_query reports every structural defect beside the counts; check_tree "
                  "reports them without running the tree. Fix every defect before you answer.")
    if levers["reserve_last_turn"]:
        extra += "\n- Your last turn must be a final answer; you will be told when it comes."
    return f"""
Write an OCPQ query answering the question below, using the log to check your work.

{schema_block}

{_agent_tool_block(probe)}

{_agent_protocol_rules()}

{rules}

How to work:
- A query is a tree of binding boxes. The ROOT box holds the answer rows; child boxes express
  conditions about them.
- Before answering, run your candidate with try_query if you have it: a box that binds 0
  situations is a box that selects nothing, and the counts tell you which one collapsed.
- A result of 0 situations is usually wrong. So is a result that selects nearly everything.{extra}
- When you are confident, answer with {{"final_tree": {{...}}}}.

Question:
{query}
"""


def agent_prompt_sha256(with_tools):
    """sha256 of the condition's static prompt: system message + task with placeholders.

    Recorded in the manifest (notes 43.12), so "the prompt was frozen before the scored runs" is a
    value that can be compared, not a claim. Depends on the thought flag and the levers, which is
    the point: a different condition must hash differently.
    """
    marker = object() if with_tools else None
    text = AGENT_SYSTEM_PROMPT + "\n" + _agent_task_prompt("<LOG SCHEMA>", "<QUESTION>", marker)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_pipeline_agentic(
    query,
    dataset,
    embeddings,
    output_file="configs/test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
    structure_hint=None,
    stage_prefix="agentic",
):
    """Let the model probe the log for up to `agent_max_turns` turns, then score its answer.

    `structure_hint` is accepted and DROPPED, and `nl_to_ocpq_pipeline` does not forward it here in
    the first place. Two layers because it carries gold's tree shape and this arm is system
    performance: `tests/test_structure_hint_prompt.py` pins that only the oracle branch forwards
    gold, and `tests/test_agent_action_protocol.py` pins that a hint passed directly reaches no
    message and no trajectory entry.

    With `agent_probe` set this is the tool rung; with it None the same loop runs with the tool
    surface withheld, which is the compute-matched control (A1'). Everything after the loop --
    verification, normalisation, execution and the single stderr repair -- is the code every other
    strategy runs, so the rungs differ in the loop and nothing else.

    Raises:
        AgentNoFinalAnswer: no usable tree was produced. Never raised when any turn yielded a valid
            tree: the best one so far is finalised instead, and `final_source` records which it was.
    """
    global _last_agent_trajectory, _agent_turn_active
    probe = agent_probe
    selected_model = model_name or hf_model
    print(f"Starting agentic NL to OCPQ pipeline "
          f"({'with tools' if probe is not None else 'NO tools, compute-matched control'}, "
          f"max {agent_max_turns} turns)...")
    print(f"Using model: {selected_model}")

    schema = extract_ocel_schema(ocel_path)
    event_names, object_names = schema["event_types"], schema["object_types"]
    schema_block = build_schema_block(schema)

    messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
    task = _agent_task_prompt(schema_block, query, probe)
    messages.append({"role": "user", "content": task})

    trajectory = []
    malformed = 0
    final_tree = None
    final_source = "none"
    last_valid_tree = None
    last_evaluated_tree = None
    levers = dict(agent_protocol)
    # What this condition may call: the probe's tools when there is a probe, and the static
    # check_tree under F in BOTH loop rungs (notes 43.12).
    available_tools = (PROBE_TOOL_NAMES if probe is not None else ()) + (
        ("check_tree",) if levers["structural_feedback"] else ())
    final_bounced = False
    check_tree_calls = 0
    cap_hits = 0
    # R: every tool action of this case by (tool, arguments), with the turn it was first made in.
    seen_actions = {}
    repeat_turns = 0
    turns_used = 0

    # Published BEFORE the loop and updated every turn, so a case the per-case cap kills mid-loop
    # still leaves the turns it took. Those are exactly the cases whose mechanism matters most, and
    # assigning the record only after the loop recorded nothing for them. `complete` says which.
    _last_agent_trajectory = {
        "turns": 0,
        "max_turns": agent_max_turns,
        "malformed_actions": 0,
        "tool_calls": 0,
        "final_source": "none",
        "complete": False,
        "trajectory": trajectory,
        "tool_trajectory": [] if probe is None else probe.trajectory,
    }

    # The condition's schema, fixed for the whole case: the grammar is compiled from it and the
    # parser validates against it, so the two cannot disagree mid-case.
    action_model = _agent_action_model()
    # The GRAMMAR is the strict union (one variant per tool + a final answer); the parsed turn is
    # the flat `action_model`. See ocpq_schema, "THE STRICT TURN GRAMMAR".
    thought_on = action_model is AgentActionWithThought
    _last_agent_trajectory["thought_enabled"] = agent_thought
    _last_agent_trajectory["protocol"] = levers

    def grammar_for(turn):
        """The turn grammar. Without levers: exactly the pre-Stage-5 grammar (the no-tools control
        keeps the full union, whose tool calls it is told it cannot make)."""
        final_only = levers["reserve_last_turn"] and turn == agent_max_turns - 1
        if not final_only and (probe is not None or levers["structural_feedback"]):
            return agent_turn_model(available_tools, thought_on, False)
        if final_only:
            return agent_turn_model((), thought_on, True)
        return AgentTurnWithThought if thought_on else AgentTurn

    previous_schema = get_constrained_schema()
    turn_cap = agent_turn_token_budget or get_generation_token_budget() or 2000
    for turn in range(agent_max_turns):
        turns_used = turn + 1
        turn_model = grammar_for(turn)
        action_schema = turn_model.model_json_schema()
        if (levers["reserve_last_turn"] and turn == agent_max_turns - 1
                and messages[-1]["role"] == "user"):
            # L: said in the message the model is answering, so the roles still alternate.
            messages[-1] = dict(messages[-1], content=messages[-1]["content"] + "\n\n"
                                + AGENT_LAST_TURN_NOTE)
        _last_agent_trajectory["turns"] = turns_used
        _last_agent_trajectory["malformed_actions"] = malformed
        _last_agent_trajectory["cap_hits"] = cap_hits
        _last_agent_trajectory["repeat_turns"] = repeat_turns
        _last_agent_trajectory["tool_calls"] = 0 if probe is None else probe.tool_calls
        try:
            # Constrain to the PROTOCOL, not to a tree. Without this the xgrammar processor masks
            # every turn to a BindingBoxTree (it is attached on every local generation), and the
            # model could not emit a tool call at all -- which would read as "small models cannot
            # do agentic work" when it is a grammar bug.
            set_constrained_schema(turn_model)
            _agent_turn_active = True
            response = call_hf_with_retry(
                model=selected_model,
                messages=messages,
                # The endpoint backend honours this directly; the local one applies the same rule
                # in `_effective_max_tokens`.
                max_tokens=turn_cap,
                # A union cannot be expressed in OpenAI strict mode (no root-level anyOf, and a
                # free-form `args` cannot satisfy additionalProperties:false), so endpoint backends
                # get plain JSON mode and the schema is enforced on the FINAL tree instead.
                response_format=_build_response_format("agent_action", action_schema, False),
            )
        except PromptTooLongError as exc:
            # Never retried by design, and the transcript only grows -- so stop and finalise with
            # whatever the model has already produced rather than losing the case.
            trajectory.append({"turn": turns_used, "error": f"prompt too long: {exc}"})
            break
        finally:
            set_constrained_schema(previous_schema)
            _agent_turn_active = False

        content = response.choices[0].message.content or ""
        action, reason = _parse_agent_action(content, action_model)
        entry = {"turn": turns_used, "raw": content[:_AGENT_RAW_CHAR_CAP]}
        if action is not None and _last_agent_parse_normalized:
            entry["normalized"] = list(_last_agent_parse_normalized)
        thought = getattr(action, "thought", None) if action is not None else None
        if thought:
            # Mechanism evidence: the text goes to the trajectory JSONL, its length to the record.
            entry["thought"] = str(thought)[:_AGENT_RAW_CHAR_CAP]

        if action is None:
            malformed += 1
            # Recorded whatever the levers, so every condition's cap hits can be counted.
            if _turn_hit_cap(response, turn_cap):
                cap_hits += 1
                entry["hit_cap"] = True
                if levers["cap_feedback"]:
                    reason = AGENT_CAP_FEEDBACK.format(cap=turn_cap)
            entry["malformed"] = reason
            trajectory.append(entry)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             f"That reply could not be used: {reason}. {_agent_protocol_rules()}"})
            continue

        if action.final_tree is not None:
            declared = action.final_tree.model_dump(by_alias=True)
            entry["action"] = "final_tree"
            entry["tree_blocking"] = _tree_blocking_count(declared)
            if (levers["bounce_blocking_final"] and not final_bounced and entry["tree_blocking"]
                    and turn < agent_max_turns - 1):
                # B (notes 43.12): returned ONCE with its defects while a turn remains. Static --
                # the same check a structural-validation rung applies -- so both loop rungs get it.
                final_bounced = True
                entry["bounced"] = True
                trajectory.append(entry)
                report = _structure_report(declared)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content":
                                 "Your final_tree cannot be evaluated as written:\n- "
                                 + "\n- ".join(report)
                                 + "\nFix these and answer again with {\"final_tree\": {...}}."})
                continue
            final_tree = declared
            trajectory.append(entry)
            final_source = "declared"
            break

        entry["action"] = action.tool
        # R (notes 43.17): the same tool with the same arguments as an earlier turn of this case.
        # Recorded whatever the levers, so every condition's repeat rate can be counted; the
        # thought is not part of the action, so a new thought on an old action is still a repeat.
        action_key = (action.tool, json.dumps(action.args or {}, sort_keys=True, default=str))
        first_turn = seen_actions.setdefault(action_key, turns_used)
        if first_turn != turns_used:
            repeat_turns += 1
            entry["repeat_of"] = first_turn
            if levers["repeat_guard"]:
                # Not re-run: no probe call, no tool budget, no check_tree count -- the turn is
                # spent, and the model is told why nothing new came back.
                if action.tool in ("check_tree", "try_query"):
                    entry["tree_blocking"] = _tree_blocking_count((action.args or {}).get("tree"))
                note = AGENT_REPEAT_NOTE.format(turn=first_turn)
                entry["guarded"] = True
                entry["result"] = {"repeat_of": first_turn, "note": note}
                trajectory.append(entry)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": note})
                continue
        if action.tool == "check_tree" and "check_tree" in available_tools:
            tree = (action.args or {}).get("tree")
            entry["tree_blocking"] = _tree_blocking_count(tree)
            check_tree_calls += 1
            result = _check_tree_result(tree)
            entry["result"] = result
            trajectory.append(entry)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": json.dumps(result, default=str)})
            continue
        if probe is not None and action.tool not in available_tools:
            # Reachable only without constrained decoding: the grammar offers available tools only.
            entry["result"] = {"error": f"{action.tool} is not a tool in this condition"}
            trajectory.append(entry)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": json.dumps(entry["result"])})
            continue
        if probe is None:
            # The control rung: the model asked for a tool it was told it does not have.
            entry["result"] = {"error": "no tools are available in this condition"}
            trajectory.append(entry)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "You have no tools here. Answer with {\"final_tree\": {...}}."})
            continue

        # A candidate passed to try_query is a tree the model committed to; keep the last one that
        # validates, so a budget-exhausted case still has an answer to score (`last_valid_probe`).
        candidate = None
        if action.tool == "try_query" and isinstance((action.args or {}).get("tree"), dict):
            entry["tree_blocking"] = _tree_blocking_count((action.args or {}).get("tree"))
            candidate = validate_and_normalize_ocpq_json((action.args or {}).get("tree"))
            if candidate is not None:
                last_valid_tree = candidate

        try:
            result = probe.call(action.tool, action.args or {})
        except ocpq_probe.ProbeBudgetExhausted as exc:
            entry["result"] = {"error": str(exc)}
            trajectory.append(entry)
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content":
                             "Your tool budget is spent. Answer now with {\"final_tree\": {...}}."})
            continue
        if action.tool == "try_query" and isinstance(result, dict):
            if "nodes" in result and candidate is not None:
                # E: a tree the probe returned COUNTS for -- not refused, not an error.
                last_evaluated_tree = candidate
            if levers["structural_feedback"] and "defects" not in result:
                # F: what the counts or the server's error cannot say. A refused tree already
                # carries its defects.
                result = dict(result, structure=_structure_report((action.args or {}).get("tree")))
        entry["result"] = result
        trajectory.append(entry)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": json.dumps(result, default=str)})

    if final_tree is None and levers["finalise_evaluated"] and last_evaluated_tree is not None:
        # E, AMENDING the pre-registered rule below (notes 43.12): the last tree the probe actually
        # evaluated comes first. The plain rule finalised refused cyclic trees, which then crashed
        # ocpq_cli with a stack overflow.
        final_tree = last_evaluated_tree
        final_source = "last_evaluated_probe"
    if final_tree is None and last_valid_tree is not None:
        # PRE-REGISTERED (notes 43): finalise with the best tree the model actually produced.
        # Without this the tool rungs would lose cases to protocol compliance that the one-shot
        # rung wins by construction, and the comparison would measure formatting, not capability.
        final_tree = last_valid_tree
        final_source = "last_valid_probe"

    _last_agent_trajectory.update({
        "turns": turns_used,
        "malformed_actions": malformed,
        "tool_calls": 0 if probe is None else probe.tool_calls,
        "final_source": final_source,
        "complete": True,
        "tool_trajectory": [] if probe is None else list(probe.trajectory),
        "final_bounced": final_bounced,
        "check_tree_calls": check_tree_calls,
        "cap_hits": cap_hits,
        "repeat_turns": repeat_turns,
    })

    if final_tree is None:
        raise AgentNoFinalAnswer(
            f"the agent produced no usable query in {turns_used} turn(s) "
            f"({malformed} malformed)")

    ocpq_json = final_tree
    if schema_validation:
        ocpq_json = verify_ocpq_json(
            ocpq_json, selected_model,
            [messages[0], messages[1], {"role": "assistant", "content": json.dumps(
                {"events": [{"event": name} for name in event_names],
                 "objects": [{"object": name} for name in object_names]})}],
            query=query, schema_enforced=schema_enforced)
        if ocpq_json is None:
            raise ValueError("Generated OCPQ JSON could not be validated against the schema.")
        ocpq_json = validate_and_normalize_ocpq_json(ocpq_json)
        if ocpq_json is None:
            raise ValueError("Refusing to execute: OCPQ JSON failed strict Pydantic validation.")

    with open(output_file, "w") as f:
        json.dump(ocpq_json, f, indent=2)

    query_result = None
    query_result_file_path = None
    if execute_query:
        query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
            ocel_path=ocel_path, output_file=output_file, query=query, messages=messages,
            stage=stage_prefix)

        # The same single stderr repair every other strategy gets. Kept so that FINALISATION is
        # identical across all four rungs and only the loop differs; it is counted in the record.
        if schema_validation and query_stderr and not query_stderr.strip().startswith("warning:"):
            prompt_fix = (
                f"The generated JSON query caused this execution error:\n{query_stderr}\n\n"
                "Please fix the OCPQ JSON to resolve this error. Output ONLY the fixed JSON in a "
                "markdown code block.")
            # The repair sees what the one-shot repair sees -- the task, then the query as the
            # model's answer -- NOT the loop transcript. That transcript ends on a USER turn (a
            # tool result, or the task itself), so appending the fix prompt made two user turns in
            # a row: Mistral's chat template refuses that outright (the 2026-09-26 gate smoke, 5 of
            # 10 Ministral cases ended as pipeline_error), and Llama's accepts it silently. This
            # shape alternates under every template and keeps finalisation identical across rungs.
            repair_messages = [messages[0], messages[1],
                               {"role": "assistant", "content": json.dumps(ocpq_json)}]
            fixed_json, fixed_content = schema_fix_with_llm(
                model_name=selected_model, messages=repair_messages, prompt=prompt_fix,
                schema_enforced=schema_enforced)
            if fixed_json is not None and fixed_json != ocpq_json:
                print("Attempting to re-run fixed query...")
                with open(output_file, "w") as f:
                    json.dump(fixed_json, f, indent=2)
                query_result, query_stderr, query_result_file_path = run_ocpq_query_with_logging(
                    ocel_path=ocel_path, output_file=output_file, query=query, messages=messages,
                    stage=f"{stage_prefix}_fix")
                ocpq_json = fixed_json
                _last_agent_trajectory["post_execution_repair"] = True

    return None, None, ocpq_json, query_result, query_result_file_path


def nl_to_ocpq_pipeline(
    query,
    dataset,
    embeddings,
    output_file="configs/test_query.json",
    ocel_path="../hospital_log_high.json",
    model_name=None,
    prompt_strategy=None,
    schema_enforced=True,
    schema_validation=True,
    execute_query=True,
    structure_hint=None,
):
    """Dispatch to a prompt strategy.

    `structure_hint` is only meaningful for the two `oracle-` strategies, which receive part of the
    GOLD query and therefore produce ceilings rather than system performance. It is ignored by
    every other strategy, so passing it cannot silently alter an ordinary condition.
    """
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
    if strategy in ORACLE_PROMPT_STRATEGIES:
        # Same code path as the baseline above, so the ablation's three rungs differ by the hint
        # and by nothing else. A missing hint (malformed gold) degrades to the baseline prompt
        # rather than failing the case; `structure_hint_level` on the record says which happened.
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
            structure_hint=structure_hint,
            stage_prefix=strategy.replace("-", "_"),
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
    if strategy in AGENTIC_PROMPT_STRATEGIES:
        # Both agentic rungs are the same loop. Which one runs is decided by whether a probe was
        # handed over (`set_agent_probe`), so the control cannot accidentally acquire tools: the
        # eval config withholds the surface rather than the prompt asking the model not to use it.
        probe = agent_probe if strategy == "agentic-log-tools" else None
        previous_probe = agent_probe
        set_agent_probe(probe)
        try:
            return _run_pipeline_agentic(
                query,
                dataset,
                embeddings,
                output_file,
                ocel_path,
                model_name,
                schema_enforced=schema_enforced,
                schema_validation=schema_validation,
                execute_query=execute_query,
                # `structure_hint` is deliberately NOT forwarded. It carries gold's tree shape, and
                # this arm is system performance, so the dispatcher does not hand it over at all --
                # a stronger guarantee than the pipeline dropping it, and it keeps the invariant
                # `tests/test_structure_hint_prompt.py` pins: exactly one branch forwards gold.
                stage_prefix=("agentic_tools" if probe is not None else "agentic_self_refine"),
            )
        finally:
            set_agent_probe(previous_probe)

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
