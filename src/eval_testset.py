#!/usr/bin/env python3
import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import signal
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import paths
import result_store
from LogEvents import extract_ocel_schema
from ocpq_schema import BindingBoxTree
from tag_testset_bb import detect_bb_labels
from QueryExecutor import discard_result_file, run_ocpq_query
try:
    from expand_models import expand_models
except Exception:
    expand_models = None
from LLM import (
    dataset as retrieval_dataset,
    embeddings,
    nl_to_ocpq_pipeline,
    get_last_schema_omissions,
    set_failure_log_path,
    set_few_shot_enabled,
    set_few_shot_k,
    set_generation_temperature,
    set_generation_seed,
    set_constrained_decoding,
    set_generation_token_budget,
    set_forced_root_arity,
    get_last_normalized_fields,
    set_schema_level,
    set_hf_client,
    set_local_gpu_client,
    reset_llm_call_stats,
    get_llm_call_stats,
    reset_query_exec_stats,
    get_query_exec_stats,
    get_hf_backend_description,
)


UNSUPPORTED_FILTER_TYPES = {"ExactlyOnce"}
DEFAULT_PROMPT_STRATEGIES = [
    "schema-allowed-entities-only",
    "schema-grammar-guided",
    "schema-chain-of-thought-plus",
]
EVALUATION_PHASES = [
    "generation",
    "schema_check",
    "execution",
    "similarity",
    "summary",
]
RUN_PHASE_TRACE = ["generation", "schema_check", "execution", "similarity"]


class CaseTimeout(Exception):
    """A single case exceeded `case_timeout_sec` somewhere between generation and execution."""


class _case_time_limit:
    """Wall-clock cap on one case, spanning generation through query execution.

    `OCPQ_CLI_TIMEOUT_SEC` (QueryExecutor.py) already caps the backend call, and
    `max_new_tokens` bounds generation, but model loading and the schema-fix retry loop are
    bounded by nothing. This closes that gap around the whole `nl_to_ocpq_pipeline` call.

    Keep the cap ABOVE `OCPQ_CLI_TIMEOUT_SEC`. The CLI limit then fires first for a pure
    execution hang and the record is classified `execution_failed`, which says more than a
    generic `timeout` would; this cap is what catches everything the CLI limit cannot see.

    SIGALRM, not a watchdog thread, because only the main thread can interrupt the running
    generation loop -- and a thread that cannot interrupt would merely observe the hang. The
    signal is delivered between bytecodes, which is enough here: `generate()` is a Python loop
    over forward passes. A single long C call (safetensors load) overshoots the deadline rather
    than being cut short, so the cap is an upper bound on *scheduling*, not a hard guarantee.

    No-ops where SIGALRM is unavailable (Windows) or off the main thread, so local development
    degrades to "no cap" instead of crashing. Evaluation runs on Linux, where it is always live.
    """

    def __init__(self, seconds):
        self.seconds = int(seconds) if seconds else 0
        self._previous = None

    @property
    def active(self):
        if self.seconds <= 0 or not hasattr(signal, "SIGALRM"):
            return False
        # setitimer/alarm raise if called off the main thread.
        try:
            import threading
            return threading.current_thread() is threading.main_thread()
        except Exception:
            return False

    def __enter__(self):
        if not self.active:
            return self

        def _fire(signum, frame):
            raise CaseTimeout(
                f"case exceeded the {self.seconds}s time limit "
                f"(case_timeout_sec) between generation and execution"
            )

        self._previous = signal.signal(signal.SIGALRM, _fire)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self.active:
            return False
        signal.alarm(0)
        if self._previous is not None:
            signal.signal(signal.SIGALRM, self._previous)
        return False

# Evaluation logs whose event/object vocabulary was present in the pre-de-leak training
# domains (container_logistics, hiring, hospital, lending, renting in DatasetGenerator).
# Cases on these logs are tagged domain_seen=True so results can be split seen vs unseen:
# a fine-tuning gain that survives on unseen logs is real skill, not memorised vocabulary.
# After de-leaking + retraining this set is expected to make no difference; the split is
# what proves that and contrasts against the older leaky adapters. Kept as a static set
# (matched on log basename) so no OCEL log has to be re-parsed at evaluation time.
_PRE_DELEAK_SEEN_LOGS = frozenset({
    "container_logistics.json",
    "03_hiring.xml",
    "04_hospital.xml",
    "lending_log_high.json",
    "renting_log_high.json",
})


def _is_domain_seen(ocel_path):
    """True if the log's vocabulary was in the pre-de-leak training data.

    Testset paths are stored Windows-style (``..\\Logs\\x.json``); normalise both
    separators so the basename resolves correctly on Linux (the cluster) too.
    """
    basename = str(ocel_path).replace("\\", "/").rsplit("/", 1)[-1]
    return basename in _PRE_DELEAK_SEEN_LOGS


# Discriminability stratum, keyed by nl_query -- the same join key slice_results_by_bb.py and
# analyze_divergence.py use, and duplicate-free by construction in the testset.
_STRATUM_PATH = paths.EVAL_RUNS / "testset_discriminability.json"
_NON_DISCRIMINATING = {"non_discriminating", "empty_result",
                       "degenerate_same_as_baseline", "degenerate_empty"}  # incl. old names


def _load_stratum_index():
    """nl_query -> (is_discriminating, baseline_f1, variant_floors), from the audit.

    Roughly a fifth of the testset consists of compliance rules its log fully satisfies. A
    result-only metric hands full marks to any query on those, so results must be reported per
    stratum and against `baseline_f1` -- the score a no-op query already achieves. Absent the
    audit the run proceeds unlabelled rather than failing: this is reporting metadata, not a
    precondition for producing predictions.

    `variant_floors` is the same floor computed under each entry of `SIMILARITY_VARIANTS`, so a
    relaxed metric is never read against the strict floor. An audit file written before those
    floors existed yields `{}`, and every variant skill is then undefined rather than wrong --
    comparing a relaxed F1 to a strict floor is the apples-to-oranges error the two-sided skill
    score exists to prevent.
    """
    if not _STRATUM_PATH.exists():
        print(f"Warning: {_STRATUM_PATH.name} not found; records will carry no stratum label. "
              "Run audit_testset_discriminability.py to enable stratified reporting.")
        return {}
    try:
        entries = _read_json(str(_STRATUM_PATH)).get("entries") or []
    except Exception as exc:  # noqa: BLE001 - never fail a run over reporting metadata
        print(f"Warning: could not read {_STRATUM_PATH.name}: {exc}")
        return {}
    # `baseline_f1` stays the NO-OP strategy's score: it is the audit's determinism anchor and the
    # figure older notes quote. The floor skill is measured against is the max over degenerate
    # strategies, carried in `baseline_f1_variants["strict"]` and copied onto each record as
    # `trivial_floor`.
    index = {
        r["nl_query"]: (r["verdict"] not in _NON_DISCRIMINATING, r.get("baseline_f1"),
                        r.get("baseline_f1_variants") or {},
                        r.get("baseline_f1_variants_detail") or {})
        for r in entries if r.get("nl_query")
    }
    if index and not any(floors for _d, _b, floors, _detail in index.values()):
        print(f"Warning: {_STRATUM_PATH.name} carries no per-variant floors; "
              "skill_variants will be undefined. Re-run audit_testset_discriminability.py.")
    return index


# What `stratum_index.get(nl_query, ...)` falls back to when an entry is absent from the audit.
# Named rather than repeated inline: the tuple gained a third element and four call sites had to
# agree on its width.
_STRATUM_MISS = (None, None, {}, {})


def _gold_root_arity(gold_query):
    """`(root_index, n_event_vars, n_object_vars)` for gold's root box, or None if unreadable.

    ORACLE information: this comes from the gold query and is used to tighten the decoding grammar,
    so any condition consuming it produces a ceiling. It lives here rather than in `LLM.py` for the
    same reason `_build_structure_hint` does -- gold never crosses into the pipeline module.

    `root_index` is carried because the root is not always node 0 (it is 0 for 166 of the 169
    testset entries, 1 or 2 for the rest) and the grammar pins the root POSITIONALLY.
    """
    if not isinstance(gold_query, dict):
        return None
    nodes = gold_query.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return None
    index = root_node_index(gold_query)
    if index >= len(nodes):
        return None
    box = (nodes[index] or {}).get("Box") if isinstance(nodes[index], dict) else None
    if not isinstance(box, list) or not box or not isinstance(box[0], dict):
        return None
    data = box[0]
    return (index,
            len(data.get("newEventVars") or {}),
            len(data.get("newObjectVars") or {}))


def _structural_metrics(gold_query, prediction):
    """Set-level agreement between the building blocks gold uses and those the prediction uses.

    This is the query-level counterpart to the result-level comparison: it asks whether the
    model expressed the right *constructs*, which a result comparison cannot see. It uses the
    same 42-label vocabulary the thesis reports against, and is invariant to node numbering and
    tree shape for the same reason the result metric is.

    A prediction that does not parse yields no labels and therefore F1 0 -- the correct reading,
    since it expressed nothing.
    """
    def labels(query):
        if not isinstance(query, dict):
            return set()
        try:
            return set(detect_bb_labels(query))
        except Exception:  # noqa: BLE001 - a malformed prediction is data, not an error
            return set()

    return _set_metrics(labels(gold_query), labels(prediction))


# Oracle conditions. These read the GOLD query and hand part of it to the model, so their scores
# are ceilings and never system performance -- the strategy names, the manifest's
# `oracle_conditions` list and the per-record `structure_hint` all say so, because a number this
# shape is exactly the kind that gets quoted out of context later.
STRUCTURE_HINT_LEVELS = ("skeleton", "skeleton+constraints")

# Which strategies read gold, and how much of its tree they are told. Any strategy absent from
# this map gets no hint, so an ordinary condition cannot accidentally be handed one.
ORACLE_STRUCTURE_LEVELS = {
    "oracle-skeleton-given": "skeleton",
    "oracle-skeleton-and-constraints-given": "skeleton+constraints",
}


def _constraint_type_name(constraint):
    """`SAT` / `SizeFilter` / `Filter:TimeBetweenEvents` -- the type, never its parameters."""
    if not isinstance(constraint, dict):
        return None
    kind = constraint.get("type")
    if kind == "Filter":
        inner = constraint.get("filter")
        inner_kind = inner.get("type") if isinstance(inner, dict) else None
        return f"Filter:{inner_kind}" if inner_kind else "Filter"
    return kind


def _constraint_stub(constraint):
    """A constraint reduced to its TYPE, with every parameter replaced by a fill marker.

    `SAT`'s `child_names` is deliberately left as a fill rather than reproduced: it is derivable
    from `edgeNames`, which the scaffold already gives, so withholding it keeps the promise that
    this level reveals constraint *types* and nothing else.
    """
    name = _constraint_type_name(constraint)
    if name is None:
        return None
    if name.startswith("Filter:"):
        return {"type": "Filter",
                "filter": {"type": name.split(":", 1)[1], "<FILL>": "<FILL: remaining fields>"}}
    return {"type": name, "<FILL>": "<FILL: remaining fields>"}


def _build_structure_hint(gold_query, level):
    """Gold's TREE as a JSON scaffold to complete, withholding everything semantic. Or None.

    The measured failure is compositional, not lexical: among the fine-tuned queries that executed
    and returned a wholly disjoint result set, 80.0% already name exactly the gold object types
    while only 31.4% have the gold node count and 18.0% the gold constraint types (notes 23.3,
    23.4). So the hint carries shape and withholds meaning -- the model must still supply every
    entity name, every filter and every constraint parameter itself. Handing over entity names
    would conflate a solved component with the failing one.

    WHY A JSON SCAFFOLD RATHER THAN PROSE. The first version described the tree in sentences
    ("node 0: declares 1 new event variable(s)..."). The 2026-08-30 smoke (job 3339751) measured
    18% compliance against that format -- BELOW the 30% node agreement the same model reaches on
    the same questions with no hint at all, with 7 of 11 predictions collapsing to a single box
    after being told 2, 3 or 4. Prompt-token deltas confirmed the hint was in the prompt, so it
    was being read and not acted on: a paragraph describing a tree is out of distribution for a
    model fine-tuned to emit OCPQ JSON from a schema and a question. A scaffold is in
    distribution, and it turns "obey this instruction" into "fill these blanks".

    What is deliberately NOT emitted, at either level: `newEventVars`/`newObjectVars` *values*,
    `filters`, `sizeFilters` contents, and constraint parameters (`min_seconds`, `max_seconds`,
    attribute values, qualifiers, `child_names`). Field NAMES appear, but the OCPQ schema in the
    prompt already names every one of them, so they carry nothing gold-specific.
    `tests/test_structure_hint.py` asserts each of these absences.

    A malformed or missing gold returns None rather than raising: the ablation must degrade to the
    baseline prompt on a bad entry, not fail the case.
    """
    if level not in STRUCTURE_HINT_LEVELS or not isinstance(gold_query, dict):
        return None
    raw_nodes = gold_query.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        return None

    scaffold_nodes = []
    for node in raw_nodes:
        box = (node or {}).get("Box") if isinstance(node, dict) else None
        if not box:
            continue
        body = box[0] if isinstance(box[0], dict) else {}
        children = box[1] if len(box) > 1 and isinstance(box[1], list) else []
        # The variable KEYS are structure -- how many variables the node binds, and under which
        # indices the filters will have to reference them. Their VALUES are the answer.
        filled = {
            "newEventVars": {k: ["<FILL: event type>"] for k in (body.get("newEventVars") or {})},
            "newObjectVars": {k: ["<FILL: object type>"]
                              for k in (body.get("newObjectVars") or {})},
            "filters": "<FILL>",
            "sizeFilters": "<FILL>",
        }
        if level == "skeleton+constraints":
            stubs = [s for s in (_constraint_stub(c) for c in body.get("constraints") or []) if s]
            filled["constraints"] = stubs
        else:
            filled["constraints"] = "<FILL>"
        filled["evVarLabels"] = {}
        filled["obVarLabels"] = {}
        scaffold_nodes.append({"Box": [filled, children]})

    if not scaffold_nodes:
        return None

    scaffold = json.dumps(
        {"nodes": scaffold_nodes, "edgeNames": gold_query.get("edgeNames") or []},
        indent=2,
    )
    constraint_line = (
        "Each constraint's TYPE is given; supply its remaining fields."
        if level == "skeleton+constraints"
        else "The constraints are NOT given; derive them from the question."
    )
    return (
        f"Complete this OCPQ query. The tree structure below is FIXED.\n"
        f"- Keep EXACTLY {len(scaffold_nodes)} node(s). Do not add, remove or renumber them.\n"
        f"- Keep every \"edgeNames\" entry and every child list exactly as given.\n"
        # The 2026-08-31 smoke measured 0% arity agreement even on the 36% of cases whose node
        # count was right, so the model was adding variables to nodes the scaffold left empty --
        # plausibly reading "{}" as a blank to fill rather than as "binds nothing".
        f"- Keep each node's variable keys exactly as given. A node showing "
        f"\"newEventVars\": {{}} or \"newObjectVars\": {{}} binds NO variable of that kind -- "
        f"leave it empty, do not add one.\n"
        f"- Replace every \"<FILL...>\" with the correct value for this question. "
        f"No \"<FILL...>\" may remain in your output.\n"
        f"- {constraint_line}\n\n"
        f"{scaffold}"
    )

def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    path = Path(path)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    return value.strip("_") or "item"


def _config_fingerprint(config):
    safe = dict(config or {})
    safe.pop("resume_from", None)
    safe.pop("dry_run", None)
    return hashlib.sha256(json.dumps(safe, sort_keys=True, default=str).encode("utf-8")).hexdigest()


# Source files whose contents determine what a run produces. Hashed into every run manifest so
# shards evaluated at different times can be proven to have come from the same pipeline.
_PIPELINE_SOURCE_FILES = (
    "eval_testset.py", "LLM.py", "QueryExecutor.py", "ocpq_schema.py",
    "LogEvents.py", "result_store.py", "expand_models.py", "tag_testset_bb.py",
    "paths.py",
)


def _pipeline_fingerprint():
    """Hash of the evaluation source, so code drift between shards is detectable.

    `config_fingerprint` covers configuration and `testset_hash` covers the data, but nothing
    covered the code. That matters when the matrix is launched in waves -- running Llama and
    Ministral now and Gemma next week is scientifically sound *only* if the pipeline is identical
    for both, and an edit to a prompt strategy in between would otherwise leave no trace. Recorded
    per run and compared at merge time.
    """
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in sorted(_PIPELINE_SOURCE_FILES):
        path = here / name
        digest.update(name.encode("utf-8"))
        digest.update(_sha256_file(path).encode("utf-8") if path.is_file() else b"MISSING")
    return digest.hexdigest()


def _record_key(case_id, config_id, repeat, nl_query=None, ocel_path=None):
    """Identity of one (entry, condition, repeat) triple, for dedup and resume.

    MUST NOT be built from `case_id` alone. `case_idx` is assigned by
    `enumerate(valid_testset, start=1)` *after* the case-shard slice, so every case-half restarts
    at 1: case 1 of `c0of2` is testset row 0, case 1 of `c1of2` is row 85. Different entries,
    identical key.

    That collision is silent and total. Merging the 96-shard matrix on 2026-08-27 reported
    "Dropped 4032 duplicate record(s)" and produced 4080 records instead of 8112 -- exactly half
    the run discarded, with the survivor of each pair chosen by glob order. The in-file comment
    above the slice claimed "contiguous slices keep case_idx stable"; contiguous slices are
    precisely what makes it restart.

    So identity comes from the entry itself. `nl_query` is unique across the testset (verified:
    169/169), and the log's basename is included because two logs could in principle carry the
    same question, while the full path varies with where a shard ran.

    `case_id` remains the fallback for records predating this field or lacking `nl_query`; within
    a single shard it is unique, so resume is correct either way.
    """
    if nl_query:
        entry = f"{nl_query}\x00{os.path.basename(ocel_path or '')}"
        ident = hashlib.sha256(entry.encode("utf-8")).hexdigest()[:16]
    else:
        ident = f"case{case_id}"
    return f"{ident}:{config_id}:{repeat}"


def _record_key_of(record):
    """`_record_key` for an already-written record."""
    return _record_key(
        record.get("case_id"), record.get("config_id"), record.get("repeat"),
        record.get("nl_query"), record.get("ocel_path"),
    )


def _load_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    return _read_json(path)


def _resolve_ocel_path(raw_path, anchor_dir):
    if not raw_path:
        return raw_path
    normalized = str(raw_path).replace("\\", os.sep).replace("/", os.sep)
    candidate = Path(normalized)
    if candidate.is_absolute():
        return str(candidate)
    return str((anchor_dir / candidate).resolve())


def _log_phase(phase, message):
    print(f"[{phase}] {message}")


def _declared_type_names(node_data: Any, key: str) -> set:
    """Type names declared by one `newEventVars` / `newObjectVars` entry.

    OCPQ serialises these as an object mapping a variable index to a list of type names, and
    gold queries always have that shape. A *prediction* is free to emit anything that parses
    as JSON, so the shape must be treated as data: a wrong shape is a wrong answer that scores
    0, never an exception.

    Both the dict form and a bare list of type names are read, because a model that emitted
    `"newObjectVars": ["orders"]` has still referenced `orders` and should be charged for it
    by `hallucinated_entity_rate`. Anything else contributes nothing.

    Defect this fixes (2026-08-18, job 3039375): `.values()` was called on whatever the model
    put here. A prediction whose `newObjectVars` was a list raised AttributeError from inside
    the scoring block and killed the evaluation at run 48 of 48. `--resume` would have replayed
    the same case and died the same way, so no re-run could have recovered it.

    Args:
        node_data: The `Box[0]` payload of one node.
        key: Either `"newEventVars"` or `"newObjectVars"`.

    Returns:
        The set of type-name strings declared under that key.
    """
    if not isinstance(node_data, dict):
        return set()
    raw = node_data.get(key)
    if isinstance(raw, dict):
        groups = list(raw.values())
    elif isinstance(raw, list):
        groups = raw
    else:
        return set()

    names = set()
    for group in groups:
        if isinstance(group, str):
            names.add(group)
        elif isinstance(group, list):
            names.update(v for v in group if isinstance(v, str))
    return names


def _extract_query_entities(ocpq_json):
    events = set()
    objects = set()
    if not isinstance(ocpq_json, dict):
        return events, objects

    for node in ocpq_json.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        box = node.get("Box", [])
        if not isinstance(box, list) or not box:
            continue
        node_data = box[0]
        if not isinstance(node_data, dict):
            continue
        events.update(_declared_type_names(node_data, "newEventVars"))
        objects.update(_declared_type_names(node_data, "newObjectVars"))
    return events, objects


def _extract_query_attribute_names(ocpq_json):
    """Attribute names referenced by any attribute-value filter in the query."""
    names = set()
    if not isinstance(ocpq_json, dict):
        return names
    for match in re.finditer(r'"attribute_name"\s*:\s*"([^"]+)"', json.dumps(ocpq_json)):
        names.add(match.group(1))
    return names


def _schema_grounding_errors(ocpq_json, ocel_path):
    """Count references to attributes / relations that do not exist in the log's schema.

    These are *post-hoc metrics only*. They are deliberately NOT fed back into the
    repair pass: doing so would hand the model schema information that the lower
    ablation levels intentionally withhold, contaminating the grounding ablation.

    Returns:
        (bad_attributes, bad_relations) as sets. Empty when the schema cannot be read.
    """
    try:
        schema = extract_ocel_schema(str(ocel_path))
    except Exception:
        return set(), set()

    known_attributes = {
        a.get("name")
        for attrs in (schema.get("object_attributes") or {}).values()
        for a in attrs
        if a.get("name")
    }
    known_attributes |= {
        a.get("name")
        for attrs in (schema.get("event_attributes") or {}).values()
        for a in attrs
        if a.get("name")
    }
    used_attributes = _extract_query_attribute_names(ocpq_json)
    # Only judge attribute names when the log actually declares some.
    bad_attributes = (used_attributes - known_attributes) if known_attributes else set()

    # Relations: the query names object/event types; check the pairs it links exist.
    known_e2o = {(e, o) for e, o, _q in (schema.get("e2o") or [])}
    known_o2o = {(a, b) for a, b, _q in (schema.get("o2o") or [])}
    bad_relations = set()
    if isinstance(ocpq_json, dict) and (known_e2o or known_o2o):
        for node in ocpq_json.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            box = node.get("Box", [])
            if not isinstance(box, list) or not box or not isinstance(box[0], dict):
                continue
            data = box[0]
            # Same tolerance as `_extract_query_entities`: `or {}` guarded against null but
            # not against a list, which is truthy and has no `.values()`.
            ev_types = _declared_type_names(data, "newEventVars")
            ob_types = _declared_type_names(data, "newObjectVars")
            for f in (data.get("filters") or []):
                if not isinstance(f, dict):
                    continue
                if f.get("type") == "O2E" and known_e2o:
                    for e in ev_types:
                        for o in ob_types:
                            if (e, o) not in known_e2o:
                                bad_relations.add(f"O2E:{e}->{o}")
                elif f.get("type") == "O2O" and known_o2o:
                    pairs = [(a, b) for a in ob_types for b in ob_types if a != b]
                    for a, b in pairs:
                        if (a, b) not in known_o2o and (b, a) not in known_o2o:
                            bad_relations.add(f"O2O:{a}->{b}")
    return bad_attributes, bad_relations


def _contains_unsupported_filter(ocpq_json):
    try:
        blob = json.dumps(ocpq_json)
    except Exception:
        return False
    return any(token in blob for token in UNSUPPORTED_FILTER_TYPES)


def _binding_entity_indices(entity_map: Any) -> tuple:
    """Return the entity indices of one variable map, ordered by variable index.

    OCPQ serialises `eventMap` / `objectMap` as arrays of `[variable_index,
    entity_index]` pairs (`event_map: Vec<(EventVariable, EventIndex)>` in
    `shared/src/binding_box/structs.rs`). Only the entity index carries meaning for
    result comparison: which variable slot an entity happened to land in is an
    arbitrary choice of the query author, so it must not affect the score.

    Args:
        entity_map: The raw `eventMap` or `objectMap` value from a binding.

    Returns:
        Entity indices ordered by their variable index, as a hashable tuple.
    """
    if not isinstance(entity_map, list):
        return ()
    pairs = [p for p in entity_map if isinstance(p, (list, tuple)) and len(p) == 2]
    ordered = sorted(pairs, key=lambda pair: pair[0])
    return tuple(entity_index for _variable_index, entity_index in ordered)


def root_node_index(query: Any) -> int:
    """Return the index of the tree's root box: the node that is nobody's child.

    `evaluationResults` is index-aligned with `nodes`, but the root is **not**
    guaranteed to be index 0 — a valid tree can list a child first and the parent
    after, and both the OCPQ UI and a generating model may do so. Assuming index 0
    silently scores a child box instead of the answer, so the root is located
    structurally instead.

    Args:
        query: The OCPQ query (BindingBoxTree shape), or None.

    Returns:
        The root node's index; 0 when the query is unavailable or malformed, or when
        the structure is cyclic and no node qualifies.
    """
    nodes = (query or {}).get("nodes") if isinstance(query, dict) else None
    if not isinstance(nodes, list) or not nodes:
        return 0

    referenced = set()
    for node in nodes:
        box = node.get("Box") if isinstance(node, dict) else None
        if isinstance(box, list) and len(box) == 2 and isinstance(box[1], list):
            referenced.update(c for c in box[1] if isinstance(c, int))

    for index in range(len(nodes)):
        if index not in referenced:
            return index
    return 0


def _extract_result_ids(result_data: Any, query: Any = None) -> Optional[tuple]:
    """Extract the comparable result sets from an ocpq_cli result document.

    Only the **root** binding box is considered. In OCPQ the root box holds the query's
    answer; child boxes exist to express constraints over it. Unioning every node would
    score intermediate scaffolding as though it were the result, and would penalise a
    correct query merely for using a different tree shape.

    The root is located via `root_node_index(query)` rather than assumed to be index 0,
    because a valid tree may list a child before its parent. When `query` is omitted the
    function falls back to index 0, which is correct for the overwhelming majority of
    trees but cannot be relied on in general.

    Each root situation is keyed as a canonical tuple of its event indices, its object
    indices, and whether the binding violated a constraint. The violation flag is
    required for compliance queries: a gold query may bind an entire population with
    only a subset violating, and without the flag any query binding that population
    would score a perfect F1.

    There is deliberately **no** fallback to the document's top-level `eventIds` /
    `objectIds`: those list every entity in the log rather than any query result, so
    using them turns an empty answer into a log-sized one. An empty result stays empty.

    Args:
        result_data: Parsed ocpq_cli result JSON.
        query: The query that produced it, used to locate the root box.

    Returns:
        A `(event_ids, object_ids, situations)` triple of sets, or None when the
        document carries no evaluation results at all.
    """
    ok_data = result_data.get("Ok") if isinstance(result_data, dict) else None
    if not isinstance(ok_data, dict):
        return None

    eval_data = ok_data.get("evaluationResults")
    if not isinstance(eval_data, list) or not eval_data:
        return None

    root_index = root_node_index(query)
    if root_index >= len(eval_data):
        root_index = 0
    root = eval_data[root_index]
    situations = root.get("situations") if isinstance(root, dict) else None
    if not isinstance(situations, list):
        situations = []

    event_ids = set()
    object_ids = set()
    situation_keys = set()

    for situation in situations:
        if not isinstance(situation, list) or len(situation) != 2:
            continue
        binding, violation = situation
        if not isinstance(binding, dict):
            continue

        events = _binding_entity_indices(binding.get("eventMap"))
        objects = _binding_entity_indices(binding.get("objectMap"))

        event_ids.update(events)
        object_ids.update(objects)
        situation_keys.add((events, objects, violation is not None))

    return event_ids, object_ids, situation_keys


def _extract_bindings_skipped(result_data: Any) -> Optional[bool]:
    """Return the CLI's `bindingsSkipped` flag, or None when absent.

    ocpq_cli sets this when a query yields so many bindings that some are dropped. Any
    comparison against a truncated result is meaningless, so affected cases must stay
    identifiable after the run.
    """
    ok_data = result_data.get("Ok") if isinstance(result_data, dict) else None
    if not isinstance(ok_data, dict):
        return None
    skipped = ok_data.get("bindingsSkipped")
    return bool(skipped) if isinstance(skipped, bool) else None


def _result_file_too_large(result_path: Optional[str], max_gb: float) -> Optional[int]:
    """Size of `result_path` if it exceeds `max_gb`, else None.

    Parsing an ocpq_cli export costs a measured ~11.7x its size in resident memory, and
    `_read_json` below loads the whole thing at once. There is no way to recover from getting
    this wrong: the kernel sends SIGKILL, which no `except` can catch, and the whole shard dies
    mid-run. Shard 60 of job 3087762 was killed twice this way -- once at 64 GB and again at
    122 GB -- on queries that expand to a single root binding whose child set runs to gigabytes.

    So the size is checked BEFORE the read rather than the failure handled after it.
    """
    if not max_gb or max_gb <= 0 or not result_path:
        return None
    try:
        size = os.path.getsize(result_path)
    except OSError:
        return None
    return size if size > max_gb * (1024 ** 3) else None


def _load_result_ids(result_path: Optional[str], query: Any = None,
                     max_gb: float = 0) -> tuple:
    """Load result sets and the truncation flag from a result file.

    Args:
        result_path: Path to an ocpq_cli result file, or None.
        query: The query that produced the file, used to locate its root box.
        max_gb: Refuse to parse a file larger than this many GB. 0 disables the check,
            which reproduces the pre-2026-08-26 behaviour exactly.

    Returns:
        An `(ids, bindings_skipped, oversized_bytes)` triple. `ids` is the triple returned by
        `_extract_result_ids`, or None when the file is missing/unreadable/empty/oversized.
        `oversized_bytes` is the file size when the guard refused it, else None.
    """
    if not result_path or not os.path.exists(result_path):
        return None, None, None
    # Logged before the read, unconditionally for anything big. Three shards were lost to an
    # OOM whose file size nobody could state afterwards -- the scratch directory is removed by
    # the submit script's EXIT trap, so a killed run leaves no evidence at all. One line here
    # makes the next failure diagnosable instead of another guess at the multiplier.
    try:
        _size = os.path.getsize(result_path)
    except OSError:
        _size = 0
    if _size > 256 * 1024 * 1024:
        _log_phase("result_size", f"{_size / (1024 ** 3):.2f} GB export "
                                  f"(cap {max_gb or 'disabled'} GB, "
                                  f"projected ~{_size * 11.7 / (1024 ** 3):.0f} GB to parse)")
    oversized = _result_file_too_large(result_path, max_gb)
    if oversized is not None:
        return None, None, oversized
    try:
        data = _read_json(result_path)
    except Exception:
        return None, None, None
    return _extract_result_ids(data, query), _extract_bindings_skipped(data), None


def _set_metrics(gold_set, pred_set):
    if gold_set is None or pred_set is None:
        return None
    gold_set = set(gold_set)
    pred_set = set(pred_set)
    intersection = gold_set & pred_set
    union = gold_set | pred_set
    precision = len(intersection) / len(pred_set) if pred_set else (1.0 if not gold_set else 0.0)
    recall = len(intersection) / len(gold_set) if gold_set else (1.0 if not pred_set else 0.0)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    jaccard = len(intersection) / len(union) if union else 1.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "jaccard": round(jaccard, 6),
        "gold_size": len(gold_set),
        "pred_size": len(pred_set),
        "intersection_size": len(intersection),
    }


def _variant_strict(gold_situations, pred_situations):
    """The primary comparison: whole situations, unprojected. Identical to `combined`."""
    return set(gold_situations), set(pred_situations)


def _variant_population(gold_situations, pred_situations):
    """Drop the violation flag: did the query select the right rows?

    A compliance query binds a whole population and flags a subset. This half asks only
    whether the population was right, which is the question a retrieval query asks anyway.
    """
    return ({(events, objects) for events, objects, _violated in gold_situations},
            {(events, objects) for events, objects, _violated in pred_situations})


def _variant_violation(gold_situations, pred_situations):
    """Flagged situations only: did the query catch the right violations?

    Returns None — never an empty comparison — when gold flags nothing. Twenty-two of the
    135 discriminating entries are compliance rules their log fully satisfies, and scoring
    those 0 would measure the testset rather than the model, while scoring them 1 would hand
    full marks to a query that flags nothing. Undefined is the only honest reading, so the
    mean carries its own denominator (`violation_n`) rather than this stratum's silence.
    """
    gold_flagged = {(events, objects) for events, objects, violated in gold_situations if violated}
    if not gold_flagged:
        return None
    return (gold_flagged,
            {(events, objects) for events, objects, violated in pred_situations if violated})


def _variant_kind_projected(gold_situations, pred_situations):
    """Blank the entity kinds gold never binds: did it identify the right things?

    Diagnostic for a measured artefact of the strict key. A situation is keyed by its event
    tuple *and* its object tuple, so a prediction that binds one extra event variable changes
    every key and scores exactly 0 even when its objects are right. On the self-consistency
    store that is 152 samples -- ~18% of executed discriminating samples -- whose mean
    objects-only F1 is 0.5203 and whose `combined` F1 is 0.0000, with only 4 of them actually
    selecting the wrong objects.

    **Never promote this to primary.** A query binding an extra variable really does return a
    different table; this quantifies how much of the measured failure is that, nothing more.
    Where gold binds both kinds (35 of 135 discriminating entries) it is exactly `strict`.
    """
    keep_events = any(events for events, _objects, _violated in gold_situations)
    keep_objects = any(objects for _events, objects, _violated in gold_situations)
    if not keep_events and not keep_objects:
        # Gold binds no entities at all, so there is nothing to project onto: blanking both
        # components would collapse every situation of every query onto one key and hand out
        # a free 1.0. Fall back to the unprojected comparison.
        return _variant_strict(gold_situations, pred_situations)

    def project(situation):
        events, objects, violated = situation
        return (events if keep_events else (), objects if keep_objects else (), violated)

    return ({project(s) for s in gold_situations}, {project(s) for s in pred_situations})


# Decompositions of the result comparison, each a `(gold_situations, pred_situations) ->
# (gold_keys, pred_keys) | None` projection. `_compute_similarity` scores every entry here and
# `audit_testset_discriminability.py` computes the no-op baseline's floor through the same
# table, so a variant and its floor are produced by one definition rather than two that can
# drift. Adding a variant here gives it a metric and a matched floor at once.
#
# `strict` is the primary metric and must stay first and unprojected: `combined` is derived
# from it, six analysis modules read `combined`, and every published number depends on it.
SIMILARITY_VARIANTS = {
    "strict": _variant_strict,
    "population": _variant_population,
    "violation": _variant_violation,
    "kind_projected": _variant_kind_projected,
}


def _variant_metrics(gold_situations, pred_situations):
    """`{variant_name: _set_metrics(...) | None}` for every entry in `SIMILARITY_VARIANTS`."""
    scored = {}
    for name, project in SIMILARITY_VARIANTS.items():
        projected = project(gold_situations, pred_situations)
        scored[name] = None if projected is None else _set_metrics(*projected)
    return scored


def _compute_similarity(gold_ids, pred_ids):
    """Score a prediction's root-node result against the gold query's.

    `events` and `objects` report the flat root-node entity sets, kept for per-kind
    breakdowns. `combined` — the per-record value behind `result_set_f1_strict` /
    `result_set_f1_executed` — compares whole situations, so it preserves which entities
    co-occur within one binding and whether that binding violated a constraint.

    `variants` carries the decompositions of that same comparison (`SIMILARITY_VARIANTS`).
    They are reported *beside* `combined`, never instead of it: `variants["strict"]` is
    `combined` by construction, and the assertion in the tests keeps it that way.
    """
    if gold_ids is None or pred_ids is None:
        return None
    gold_events, gold_objects, gold_situations = gold_ids
    pred_events, pred_objects, pred_situations = pred_ids
    return {
        "events": _set_metrics(gold_events, pred_events),
        "objects": _set_metrics(gold_objects, pred_objects),
        "combined": _set_metrics(gold_situations, pred_situations),
        "variants": _variant_metrics(gold_situations, pred_situations),
    }


def _skill_score(f1, baseline_f1):
    """Result F1 renormalised against what a no-op query already scores on the same entry.

    Reported *beside* `result_set_f1_strict`, never instead of it. The raw F1 has a blind spot
    the restored testset makes concrete: the mean trivial floor is 0.6124, so a model emitting
    executable but do-nothing queries scores 0.5148 on the discriminating stratum -- more than a
    model that fails 80% of its queries -- while doing no work at all. Skill puts that strategy
    at exactly 0.

    Each side is normalised by the distance to the bound actually being approached::

        (f1 - floor) / (1 - floor)   if f1 >= floor    # headroom above trivial
        (f1 - floor) / floor          if f1 <  floor    # distance down toward nothing

    The single-denominator form `(f1 - floor) / (1 - floor)` was measured and rejected: on the
    five entries whose floor exceeds 0.99 the denominator collapses and one wrong answer
    contributes up to -6097, so a model at F1 0.85 everywhere scores -9.06. That magnitude
    measures how little headroom the *item* has, not how wrong the *model* is. Normalising the
    lower branch by `floor` -- the room there is to fall -- removes the amplification: a query
    fetching the wrong entity type scores exactly -1 on every item, whatever its floor.

    The result is bounded in [-1, 1], continuous at `f1 == floor` (both branches give 0) and
    monotone in `f1`.

    Args:
        f1: Root situation F1 for this record. Callers pass 0.0 rather than None for a
            prediction that did not parse or execute, matching `result_set_f1_strict`; a metric
            that dropped those would reward attempting fewer queries.
        baseline_f1: The entry's trivial floor, from the discriminability audit.

    Returns:
        The skill score, or None when it is undefined -- either the record carries no floor
        (stratum join miss) or the floor is 1.0, which is exactly the 34 non-discriminating
        entries, where a no-op query already scores full marks and there is nothing to beat.
    """
    if f1 is None or baseline_f1 is None:
        return None
    if baseline_f1 >= 1.0:
        return None
    if f1 >= baseline_f1:
        return (f1 - baseline_f1) / (1.0 - baseline_f1)
    # `f1 >= 0` always, so `f1 < baseline_f1` implies `baseline_f1 > 0`: no division by zero.
    return (f1 - baseline_f1) / baseline_f1


def _record_floor(record):
    """The floor skill is measured against: the best trivial strategy, not a chosen one.

    `trivial_floor` is the max over degenerate strategies (`_degenerate_floors` in the audit).
    `baseline_f1` -- the no-op baseline alone -- is the fallback for a record written before that
    distinction existed, so an old run keeps the skill it was published with instead of crashing
    or silently switching basis.
    """
    floor = record.get("trivial_floor")
    return record.get("baseline_f1") if floor is None else floor


def _record_skill(record, strict=True):
    """Skill for one prediction record, applying the strict-F1 convention by default."""
    f1 = ((record.get("similarity") or {}).get("combined") or {}).get("f1")
    if f1 is None:
        if not strict:
            return None
        f1 = 0.0
    return _skill_score(f1, _record_floor(record))


def _record_variant_f1(record, variant, strict=True):
    """One variant's F1 for a record, or None when the variant is undefined on this entry.

    The strict convention applies to *unavailable results*, not to *undefined variants*. A
    prediction that never executed scores 0 -- otherwise attempting less would score better.
    But `violation` on an entry gold never flags has no value to substitute: 0 would blame the
    model for the log's compliance and 1 would reward a query that flags nothing, so it stays
    None and drops out of the mean along with its denominator.
    """
    similarity = record.get("similarity")
    if similarity is not None:
        scored = (similarity.get("variants") or {}).get(variant)
        return None if scored is None else scored.get("f1")
    if not strict:
        return None
    # No comparable result. Whether a variant is defined is a property of the *entry*, not of
    # what the model produced, so it cannot be read off the missing similarity block. The audit
    # is the authority: it scores the no-op baseline through the same projections, so a floor
    # present-but-None marks the variant undefined on this entry (`violation` where gold flags
    # nothing). An absent floors dict means the audit predates them -- then the strict
    # substitution still applies and only the skill goes undefined.
    floors = record.get("baseline_f1_variants") or {}
    if floors and variant in floors and floors[variant] is None:
        return None
    return 0.0


def _record_skill_variants(record, strict=True):
    """`{variant: skill}` for one record, each against its own matched floor.

    Absent per-variant floors (an audit file predating them) every value is None. That is the
    intended degradation: a relaxed F1 read against the strict floor would look like skill the
    model does not have.
    """
    floors = record.get("baseline_f1_variants") or {}
    return {
        name: _skill_score(_record_variant_f1(record, name, strict=strict), floors.get(name))
        for name in SIMILARITY_VARIANTS
    }


def _truncation_counts(records) -> dict:
    """How often `ocpq_cli` truncated a result, with the denominator stated explicitly.

    The backend cancels evaluation once a query's expansion exceeds its internal limit
    (`150_000_000` in `shared/src/binding_box/structs.rs`), prints `Too much to handle!` and
    exports a *partial* result set. Comparing against a partial set is not a neutral loss: the
    queries that trigger it are the ones that over-select wildly, so truncation preferentially
    discards false positives and **inflates** the score. One such record scored 0.881 in job
    3039375 while returning a fraction of what its query actually selects.

    The flag was recorded per record but never aggregated, so the rate could only be had by
    grepping `predictions.json`. It is reported rather than acted on: whether these records
    should be scored 0 is a separate decision (`--truncation-policy`), and one worth making from
    a measured rate rather than in advance.

    `gold_bindings_skipped_n` is the number that decides it. Gold queries are handcrafted and
    their largest result is 53,399 root situations, so truncation there is unlikely -- but it is
    not ruled out, and if it happens the entry is unscoreable for *every* model rather than
    evidence about any one of them.

    Args:
        records: Prediction records, each optionally carrying `metrics.bindings_skipped` and
            `metrics.gold_bindings_skipped`.

    Returns:
        The four summary keys. `bindings_skipped_rate` is None when nothing executed.
    """
    pred_truncated = 0
    gold_truncated = 0
    # Denominator: records that produced a parseable result. `_load_result_ids` returns a None
    # flag when there was no file to read, and those records cannot be truncated -- counting
    # them would dilute the rate with schema failures, which are already their own metric.
    with_result = 0
    for record in records:
        metrics = record.get("metrics") or {}
        pred_flag = metrics.get("bindings_skipped")
        if pred_flag is not None:
            with_result += 1
            if pred_flag:
                pred_truncated += 1
        if metrics.get("gold_bindings_skipped"):
            gold_truncated += 1

    return {
        "bindings_skipped_n": pred_truncated,
        "bindings_skipped_denominator": with_result,
        "bindings_skipped_rate": (pred_truncated / with_result) if with_result else None,
        "gold_bindings_skipped_n": gold_truncated,
    }


def _timeout_counts(records) -> dict:
    """How many cases hit the per-case wall-clock cap, over every attempted record.

    Reported next to every headline metric because a timeout is not a neutral omission: the
    record scores 0 in `result_set_f1_strict` (via `similarity: None`), so a model that times
    out often is penalised, and the reader needs the rate to tell "answered wrongly" from "ran
    out of budget". Where a truncation rate is conditional on having produced a result, this
    denominator is *every* record -- a case that timed out during model loading never reached
    generation, and excluding it would hide exactly the failure the cap exists to measure.

    Interpretable only against the `case_timeout_sec` recorded in the same run's manifest, and
    only across arms that shared it; that is why the cap is applied to every model rather than
    to the one it was introduced for.

    Args:
        records: Prediction records, each optionally carrying `status` / `pipeline_status`.

    Returns:
        `timeout_n`, the denominator, and the rate (None when there are no records).
    """
    total = 0
    timed_out = 0
    for record in records:
        total += 1
        if record.get("pipeline_status") == "timeout" or record.get("status") == "timeout":
            timed_out += 1

    return {
        "timeout_n": timed_out,
        "timeout_denominator": total,
        "timeout_rate": (timed_out / total) if total else None,
    }


def _oversized_counts(records) -> dict:
    """Predictions whose result export was too large to load, over every attempted record.

    Reported beside `timeout_n` and for the same reason: the record scores 0 in
    `result_set_f1_strict`, so a model that emits intractable queries is penalised, and the
    reader needs the rate to tell "answered wrongly" from "produced an answer nothing can read".

    Interpretable only against the `max_result_file_gb` recorded in the same run's manifest.
    A run with the guard disabled reports 0 here and can still be killed outright instead --
    which is what happened to shard 60 of job 3087762, twice.
    """
    total = 0
    oversized = 0
    largest = 0
    for record in records:
        total += 1
        metrics = record.get("metrics") or {}
        if metrics.get("result_too_large") or record.get("status") == "result_too_large":
            oversized += 1
            largest = max(largest, int(metrics.get("result_file_bytes") or 0))

    return {
        "result_too_large_n": oversized,
        "result_too_large_denominator": total,
        "result_too_large_rate": (oversized / total) if total else None,
        "result_too_large_max_bytes": largest or None,
    }


def _summarize_llm_stats(stats):
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
    total_chars = 0
    total_duration = 0.0
    has_token_data = False

    for entry in stats:
        total_duration += float(entry.get("duration_sec") or 0.0)
        prompt_tokens = entry.get("prompt_tokens")
        completion_tokens = entry.get("completion_tokens")
        total_tokens_entry = entry.get("total_tokens")
        if prompt_tokens is not None or completion_tokens is not None or total_tokens_entry is not None:
            has_token_data = True
        total_prompt += int(prompt_tokens or 0)
        total_completion += int(completion_tokens or 0)
        total_tokens += int(total_tokens_entry or 0)
        total_chars += int(entry.get("completion_chars") or 0)

    return {
        "calls": len(stats),
        "duration_sec": round(total_duration, 4),
        "prompt_tokens": total_prompt if has_token_data else None,
        "completion_tokens": total_completion if has_token_data else None,
        "total_tokens": total_tokens if has_token_data else None,
        "completion_chars": total_chars,
    }


def _summarize_exec_stats(stats):
    total_duration = sum(float(entry.get("duration_sec") or 0.0) for entry in stats)
    success = any(entry.get("success") for entry in stats)
    last = stats[-1] if stats else {}
    return {
        "duration_sec": round(total_duration, 4),
        "success": bool(success),
        "result_file": last.get("result_file"),
        "stderr": last.get("stderr"),
    }


def _mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _std(values):
    values = [v for v in values if v is not None]
    if len(values) <= 1:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _configure_backend(model_cfg, global_cfg):
    backend = model_cfg.get("backend") or global_cfg.get("backend")
    hf_provider = model_cfg.get("hf_provider") or global_cfg.get("hf_provider")
    hf_token = model_cfg.get("hf_token") or global_cfg.get("hf_token")
    local_device = model_cfg.get("local_device") or global_cfg.get("local_device")
    local_dtype = model_cfg.get("local_dtype") or global_cfg.get("local_dtype")
    local_quantization = model_cfg.get("local_quantization") or global_cfg.get("local_quantization") or "4bit"

    if backend == "local-gpu":
        set_local_gpu_client(
            model_name=model_cfg["name"],
            token=hf_token or None,
            device=local_device,
            dtype=local_dtype,
            quantization=local_quantization,
        )
    else:
        set_hf_client(
            token=hf_token or None,
            provider=hf_provider or None,
            model_name=model_cfg["name"],
        )


def _resolve_model_config(model_cfg, global_cfg):
    merged = dict(global_cfg)
    merged.update({k: v for k, v in model_cfg.items() if v is not None})
    merged["label"] = merged.get("label") or merged.get("name")
    return merged


def _load_testset(path, limit, seed, shuffle):
    rows = _read_json(path)
    # `ocel_path` in the testset is repo-root-relative by contract (entries store
    # `..\Logs\x.json`, pointing at the sibling log collection). Anchor on the repo root,
    # NOT on the testset file's own directory -- the testset lives in `data/` now, and
    # anchoring there would resolve every log to a nonexistent `NLxOCPQ/Logs/`.
    anchor_dir = paths.REPO_ROOT
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    for row in rows:
        if isinstance(row, dict) and row.get("ocel_path"):
            row["ocel_path"] = _resolve_ocel_path(row["ocel_path"], anchor_dir)
    return rows


def _classify_run_status(status, error, schema_valid, exec_stats, query_result_file,
                         query_stdout, result_too_large=False):
    if status != "ok":
        lowered = (error or "").lower()
        # Before the substring matches below: a timeout message mentions "execution", so
        # without this it would be classified `execution_failed` and become indistinguishable
        # from a query the backend actually rejected. The distinction is the whole point of
        # the cap -- "ran out of budget" and "produced a broken query" are different failures.
        if status == "timeout":
            return "timeout"
        if "validated against the schema" in lowered or "strict pydantic" in lowered:
            return "schema_invalid"
        if "parsed" in lowered or "json" in lowered and "parse" in lowered:
            return "parse_failed"
        if "execute" in lowered or "execution" in lowered:
            return "execution_failed"
        return "pipeline_error"

    if not schema_valid:
        return "schema_invalid"

    # Checked before the success path below: the CLI exited cleanly and wrote a result
    # file, so every signal here says "success" -- and the answer was never read. Left
    # unlabelled it would be counted as a correct execution that happened to score 0.
    if result_too_large:
        return "result_too_large"

    exec_success = bool((exec_stats or {}).get("success"))
    if exec_success and query_result_file:
        return "success"

    stderr = (exec_stats or {}).get("stderr") or query_stdout or ""
    if stderr and str(stderr).strip().startswith("warning:"):
        return "no_result"
    if stderr:
        return "execution_failed"
    if query_result_file:
        return "success"
    return "no_result"


def _compute_total_runs(num_cases, num_models, num_strategies, num_schema_settings, repeats):
    return num_cases * num_models * num_strategies * num_schema_settings * repeats


def _format_run_progress(case_idx, total_cases, strategy_idx, total_strategies, schema_idx, total_schema, repeat_idx, total_repeats, run_idx, total_runs):
    return (
        f"case {case_idx}/{total_cases} | "
        f"strategy {strategy_idx}/{total_strategies} | "
        f"schema {schema_idx}/{total_schema} | "
        f"repeat {repeat_idx}/{total_repeats} | "
        f"run {run_idx}/{total_runs}"
    )


def _format_duration(seconds):
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _estimate_remaining_eta(start_time, completed_runs, total_runs):
    if completed_runs <= 0:
        return None
    elapsed = time.time() - start_time
    avg_per_run = elapsed / completed_runs
    return avg_per_run * max(total_runs - completed_runs, 0)


def _hash_query_key(ocel_path, ocpq_query):
    payload = json.dumps(ocpq_query, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{ocel_path}::{digest}"


def run_evaluation(config_path_or_config):
    config = config_path_or_config if isinstance(config_path_or_config, dict) else _read_json(config_path_or_config)
    if config.get("auto_expand_finetuned"):
        if expand_models is None:
            raise RuntimeError(
                "auto_expand_finetuned is enabled, but expand_models.py could not be imported."
            )
        config = expand_models(config)

    testset_path = config.get("testset_path", "data/testset.json")
    output_dir = Path(config.get("output_dir", "eval_runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    testset_abs = Path(testset_path)
    if not testset_abs.exists():
        raise FileNotFoundError(f"Testset not found: {testset_abs}")

    model_idx_filter = config.get("_model_idx")
    strategy_idx_filter = config.get("_strategy_idx")
    schema_idx_filter = config.get("_schema_idx")
    case_shard = config.get("_case_shard")  # (index, total), 0-based index
    if model_idx_filter is not None:
        shard_tag = f"shard_m{model_idx_filter}"
        if strategy_idx_filter is not None:
            shard_tag += f"_s{strategy_idx_filter}"
        if schema_idx_filter is not None:
            shard_tag += f"_e{schema_idx_filter}"
        if case_shard is not None:
            shard_tag += f"_c{case_shard[0]}of{case_shard[1]}"
        # Decoding rungs differ only in these two settings -- same model, strategy and schema
        # index -- so without them every rung produces the same `shard_m1_s0_e0_*` name. Two
        # consequences, both silent: nothing on disk says which rung a directory holds, and
        # `--resume` (which globs that prefix and takes the newest) would resume one rung into
        # another's directory. Suffixed only when non-default, so existing shard names are unchanged.
        if config.get("constrained_decoding"):
            shard_tag += "_constrained"
        if config.get("generation_token_budget") is not None:
            shard_tag += f"_tok{int(config['generation_token_budget'])}"
        run_id = f"{shard_tag}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        run_id = time.strftime("eval_run_%Y%m%d_%H%M%S")
    resume_from = config.get("resume_from")
    if resume_from:
        resume_path = Path(resume_from)
        run_dir = resume_path.parent if resume_path.is_file() else resume_path
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
    else:
        run_dir = output_dir / run_id
        # Created further down, only once we know this is not a dry run — otherwise every
        # `--dry-run` litters eval_runs/ with an empty directory that looks like a real run.

    global_cfg = {
        "backend": config.get("backend", "local-gpu"),
        "hf_provider": config.get("hf_provider", ""),
        "hf_token": config.get("hf_token", os.getenv("HF_TOKEN", "")),
        "local_device": config.get("local_device", "cuda"),
        "local_dtype": config.get("local_dtype", "auto"),
        "local_quantization": config.get("local_quantization", "4bit"),
        "temperature": config.get("temperature", 0.0),
        # None falls back to LLM.DEFAULT_FEW_SHOT_K (3). Carried here explicitly because
        # `global_cfg` is built from a fixed key list, so a `few_shot_k` set in the config
        # would otherwise never reach `set_few_shot_k` and would be silently ignored.
        "few_shot_k": config.get("few_shot_k"),
        # Raw ocpq_cli result files are large (mean 6.6 MB, tail past 700 MB) and a full
        # matrix produces ~5550 of them. Everything the metric needs is extracted the moment
        # they are read, so they are discarded by default; set true to keep them for debugging.
        "keep_result_files": bool(config.get("keep_result_files", False)),
        # Persist the extracted root result sets so a metric can be re-derived later without
        # re-running the matrix (40-100 GPU-hours). Two orders of magnitude smaller than the
        # raw files above: ~100-500 MB gzipped for the whole matrix. On by default, because the
        # one run where it is forgotten is the run whose metric turns out to be wrong.
        "result_store": bool(config.get("result_store", True)),
        "result_store_max_situations": int(
            config.get("result_store_max_situations", result_store.DEFAULT_MAX_SITUATIONS)
        ),
    }

    models = config.get("models") or []
    if not models:
        raise ValueError("Config must include a non-empty 'models' list.")

    if model_idx_filter is not None:
        if model_idx_filter < 0 or model_idx_filter >= len(models):
            raise ValueError(
                f"--model-idx {model_idx_filter} is out of range "
                f"(0–{len(models) - 1} for this config)."
            )
        models = [models[model_idx_filter]]

    prompt_strategies = config.get("prompt_strategies") or DEFAULT_PROMPT_STRATEGIES
    schema_enforced_values = config.get("schema_enforced", [True])

    # Optional finer sharding: restrict this job to a single prompt strategy and/or
    # schema setting so the full conditions matrix can be split across more Slurm
    # array tasks (model x level x strategy x schema), each finishing well within the
    # wall-time limit. Results are recombined afterwards via --merge-shards.
    if strategy_idx_filter is not None:
        if strategy_idx_filter < 0 or strategy_idx_filter >= len(prompt_strategies):
            raise ValueError(
                f"--strategy-idx {strategy_idx_filter} is out of range "
                f"(0–{len(prompt_strategies) - 1} for this config)."
            )
        prompt_strategies = [prompt_strategies[strategy_idx_filter]]
    if schema_idx_filter is not None:
        if schema_idx_filter < 0 or schema_idx_filter >= len(schema_enforced_values):
            raise ValueError(
                f"--schema-idx {schema_idx_filter} is out of range "
                f"(0–{len(schema_enforced_values) - 1} for this config)."
            )
        schema_enforced_values = [schema_enforced_values[schema_idx_filter]]
    keep_result_files = global_cfg["keep_result_files"]
    result_store_enabled = global_cfg["result_store"]
    result_store_max_situations = global_cfg["result_store_max_situations"]
    schema_validation = bool(config.get("schema_validation", True))
    schema_validation_when_no_schema = bool(config.get("schema_validation_when_no_schema", False))
    # Constrain generation to the OCPQ grammar so invalid output cannot be produced, as opposed to
    # `schema_enforced`, which only decides whether the output is CHECKED and repaired afterwards.
    # Both default to today's behaviour so an existing config reproduces exactly.
    constrained_decoding = bool(config.get("constrained_decoding", False))
    # ORACLE: pin the root box's variable counts to gold's. Only meaningful with constrained
    # decoding -- it tightens that grammar rather than being a separate mechanism.
    force_root_arity = bool(config.get("force_root_arity", False))
    _budget = config.get("generation_token_budget")
    generation_token_budget = None if _budget is None else int(_budget)
    repeats = int(config.get("repeats", 1))
    limit = int(config.get("limit", 0))
    seed = int(config.get("seed", 3407))
    shuffle = bool(config.get("shuffle", False))
    dry_run = bool(config.get("dry_run", False))
    # Wall-clock budget for one case, generation through execution. None/0 disables the cap,
    # which is the default so that omitting the key reproduces the pre-cap behaviour exactly.
    # A cap that binds must be applied to every arm or the timeout counts are not comparable
    # across models: an uncapped arm is not "faster", it was allowed to finish.
    case_timeout_sec = config.get("case_timeout_sec")
    # Refuse to parse a prediction's result export larger than this many GB. Parsing costs
    # ~11.7x the file size in RAM and `_read_json` loads it whole, so an over-selecting
    # query can SIGKILL the shard -- uncatchable, and it takes the whole run down. 0/None
    # disables the guard and reproduces the pre-2026-08-26 behaviour exactly.
    max_result_file_gb = float(config.get("max_result_file_gb") or 0)
    # Base for the per-(case, condition, repeat) generation seed. Only consulted when sampling is
    # active; a greedy run does not touch the RNG, so this is inert at temperature 0. None/0
    # leaves generation unseeded, which is the pre-2026-08-29 behaviour.
    generation_seed_base = config.get("generation_seed_base")
    generation_seed_base = int(generation_seed_base) if generation_seed_base else 0
    case_timeout_sec = int(case_timeout_sec) if case_timeout_sec else 0

    testset = _load_testset(testset_path, limit, seed, shuffle)
    valid_testset = [
        row for row in testset
        if row.get("nl_query", "") and row.get("ocel_path") and row.get("ocpq_query")
    ]
    testset_hash = _sha256_file(testset_abs)

    # Optional case-range sharding. With 3 schema levels the matrix reaches 108 conditions;
    # a full 140-case shard would push the slowest model past the wall-time limit, so cases
    # can additionally be split across jobs. NOTE: `case_idx` is assigned AFTER this slice,
    # so it restarts at 1 in every shard and is NOT globally unique -- see `_record_key`,
    # which is why record identity comes from `nl_query` rather than from `case_id`.
    if case_shard is not None:
        shard_index, shard_total = case_shard
        if shard_total < 1 or not (0 <= shard_index < shard_total):
            raise ValueError(f"--case-shard must be i/n with 0 <= i < n (got {shard_index}/{shard_total}).")
        chunk = math.ceil(len(valid_testset) / shard_total)
        start = shard_index * chunk
        valid_testset = valid_testset[start:start + chunk]
        _log_phase(
            "generation",
            f"case shard {shard_index + 1}/{shard_total}: {len(valid_testset)} case(s) "
            f"(offset {start})",
        )
    total_runs = _compute_total_runs(
        len(valid_testset), len(models), len(prompt_strategies), len(schema_enforced_values),
        repeats,
    )
    config_fingerprint = _config_fingerprint(config)

    if dry_run:
        print(f"Dry run: loaded {len(valid_testset)} valid test cases from {testset_abs}.")
        print(
            f"Dry run: {len(models)} models x {len(prompt_strategies)} prompt strategies "
            f"x {len(schema_enforced_values)} schema settings x {repeats} repeats "
            f"= {total_runs} planned runs."
        )
        print(f"Dry run: schema enforcement values = {schema_enforced_values}")
        print(f"Dry run: schema_validation_when_no_schema = {schema_validation_when_no_schema}")
        print(f"Dry run: constrained_decoding = {constrained_decoding}"
              f"  (grammar-masked generation; NOT the same as schema_enforced, which only "
              f"validates and repairs afterwards)")
        print(f"Dry run: force_root_arity = {force_root_arity}"
              f"{'   *** ORACLE CONDITION -- ceiling, never system performance ***' if force_root_arity else ''}")
        print(f"Dry run: generation_token_budget = "
              f"{generation_token_budget if generation_token_budget is not None else 'per-pipeline default (1000/2000)'}")
        return None

    if not resume_from:
        run_dir.mkdir(parents=True, exist_ok=False)

    # Keep the raw-output failure log inside this run rather than at the shared default,
    # which every concurrent array task would otherwise append to at once.
    set_failure_log_path(run_dir / "failure_log.jsonl")

    checkpoint_path = None
    checkpoint_state = {}
    if resume_from:
        resume_path = Path(resume_from)
        if resume_path.is_file():
            checkpoint_path = resume_path
            run_dir = resume_path.parent
        else:
            run_dir = resume_path
            checkpoint_path = run_dir / "checkpoint.json"
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume directory not found: {run_dir}")
        existing_manifest = _load_json_if_exists(run_dir / "config_resolved.json")
        if existing_manifest and existing_manifest.get("run_id"):
            run_id = existing_manifest["run_id"]
        if existing_manifest and existing_manifest.get("config_fingerprint") != config_fingerprint:
            raise ValueError("Resume config does not match the existing run directory.")
        checkpoint_state = _load_json_if_exists(checkpoint_path) or {}
        existing_records = _load_json_if_exists(run_dir / "predictions.json") or checkpoint_state.get("records") or []
    else:
        checkpoint_state = {}
        existing_records = []

    manifest = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "testset_path": str(testset_abs),
        "testset_hash": testset_hash,
        "config_fingerprint": config_fingerprint,
        # Identifies the evaluation source that produced this shard. Shards merged into one
        # result set must agree on it, or they are not comparable.
        "pipeline_fingerprint": _pipeline_fingerprint(),
        "resume_from": str(resume_from) if resume_from else None,
        "records": len(valid_testset),
        "loaded_records": len(testset),
        "skipped_records": len(testset) - len(valid_testset),
        "evaluation_phases": EVALUATION_PHASES,
        "prompt_strategies": prompt_strategies,
        "schema_enforced": schema_enforced_values,
        "schema_validation": schema_validation,
        "schema_validation_when_no_schema": schema_validation_when_no_schema,
        # Recorded in the manifest, and therefore in `config_fingerprint`, because a constrained
        # run is NOT comparable to a free one: the model could not have emitted the same tokens.
        # Two shards that disagree on these must not merge silently.
        "constrained_decoding": constrained_decoding,
        "generation_token_budget": generation_token_budget,
        "force_root_arity": force_root_arity,
        "repeats": repeats,
        # Stated per run, not only in the config, because a timeout rate is meaningless without
        # the budget it was measured against. Also enters `config_fingerprint`, so two shards
        # run under different caps cannot be merged without the difference being visible.
        "case_timeout_sec": case_timeout_sec or None,
        "max_result_file_gb": max_result_file_gb or None,
        # Sampling provenance. A run whose repeats differ is only reproducible if this is set,
        # so it belongs beside the record of what the run was configured to do.
        "generation_seed_base": generation_seed_base or None,
        # Conditions in THIS run that were shown part of the gold query. Their scores are
        # ceilings, not system performance. Stated at the top level of the manifest so a
        # downstream tool or a reader opening the run cold cannot miss it -- an oracle number
        # quoted out of context is the most damaging mistake this run can produce. Empty for
        # every ordinary run.
        # `force_root_arity` takes the root's variable counts from GOLD, so a run using it is a
        # ceiling and can never be reported as system performance -- same standing as the
        # oracle- prompt strategies.
        "oracle_conditions": (["force_root_arity"] if force_root_arity else []) + sorted(
            s for s in prompt_strategies if s in ORACLE_STRUCTURE_LEVELS
        ),
        "temperature": global_cfg.get("temperature"),
        "ocpq_cli_timeout_sec": int(os.getenv("OCPQ_CLI_TIMEOUT_SEC", "300")),
        "config": config,
    }
    _write_json(run_dir / "config_resolved.json", manifest)

    # Loaded once: labelling every record keeps predictions.json self-contained, so no
    # downstream tool has to re-derive the stratum or re-run the audit to report against it.
    stratum_index = _load_stratum_index()
    _log_phase("generation", f"stratum labels loaded for {len(stratum_index)} testset entries.")

    # One store per run directory rather than one shared across the matrix: shards run as
    # separate array tasks, and a per-run store keeps them from writing into the same tree.
    # `_merge_shards` unions them afterwards, which is safe because the filenames are content
    # addresses.
    result_store_dir = run_dir / "result_store"
    if result_store_enabled:
        _log_phase("generation", f"result store enabled at {result_store_dir}")

    ground_truth_cache = {}
    records = list(existing_records)
    completed_keys = {
        _record_key_of(rec)
        for rec in records
        if rec.get("case_id") is not None and rec.get("config_id") and rec.get("repeat") is not None
    }
    total_cases = len(valid_testset)
    completed_runs = len(records)
    elapsed_before_resume = float(checkpoint_state.get("elapsed_sec") or 0.0)
    evaluation_start_time = time.time() - elapsed_before_resume

    _log_phase("generation", f"Starting evaluation run {run_id} with {total_cases} valid test cases.")
    run_counter = completed_runs

    for case_idx, row in enumerate(valid_testset, start=1):
        nl_query = row.get("nl_query", "")
        ocel_path = row.get("ocel_path")
        gold_query = row.get("ocpq_query")
        if not ocel_path or not gold_query:
            continue

        case_key = _hash_query_key(ocel_path, gold_query)
        if case_key not in ground_truth_cache and result_store_enabled:
            # Resume path. `ground_truth_cache` lives only in this process and this block sits
            # above the `completed_keys` skip, so without this a resumed shard re-executes every
            # gold query before reaching any new work -- ~15 min per shard at the audit's
            # measured 10.8 s mean, and worse on the cluster where OCEL import dominates.
            #
            # Reusing the stored result is *more* reproducible than re-executing, not less:
            # MAX_NUM_BINDINGS truncation happens after a parallel try_reduce, so a capped query
            # is not guaranteed to reproduce its own binding set. A truncated or missing blob
            # yields None and falls through to a real execution.
            try:
                cached_ids, cached_skipped = result_store.load_with_meta(
                    result_store_dir, ocel_path, gold_query)
            except Exception as exc:
                cached_ids, cached_skipped = None, None
                _log_phase("execution", f"result store read failed for case {case_idx}: {exc}")
            if cached_ids is not None:
                ground_truth_cache[case_key] = {
                    "result_path": None,
                    # Null rather than invented: no query ran, so there is no duration or output.
                    "duration_sec": None,
                    "stdout": None,
                    "stderr": None,
                    "ids": cached_ids,
                    "bindings_skipped": cached_skipped,
                    "blob": result_store.blob_id(ocel_path, gold_query),
                    "source": "result_store",
                }
                _log_phase("execution", f"case {case_idx:03d} | gold reused from result store")

        if case_key not in ground_truth_cache:
            gt_dir = run_dir / "ground_truth"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_query_path = gt_dir / f"ground_truth_{case_idx}.json"
            _write_json(gt_query_path, gold_query)

            gt_start = time.time()
            gt_stdout, gt_stderr, gt_result_path = run_ocpq_query(ocel_path, str(gt_query_path))
            gt_duration = time.time() - gt_start
            gt_ids, gt_bindings_skipped, _gt_oversized = _load_result_ids(
                gt_result_path, gold_query, max_gb=0)
            # Content-addressed, so gold is written once per entry rather than once per
            # condition. Must happen before the file is discarded below.
            gt_blob = None
            if result_store_enabled:
                try:
                    gt_blob = result_store.store(
                        result_store_dir, ocel_path, gold_query, gt_ids,
                        gt_bindings_skipped, result_store_max_situations,
                    )
                except Exception as exc:
                    _log_phase("execution", f"result store (gold) failed for case {case_idx}: {exc}")
            # `ids` is cached and reused across every condition, so the raw file is never
            # read again. Null the path once discarded rather than recording a dangling one.
            if not keep_result_files:
                discard_result_file(gt_result_path)
                gt_result_path = None

            ground_truth_cache[case_key] = {
                "result_path": gt_result_path,
                "duration_sec": round(gt_duration, 4),
                "stdout": gt_stdout,
                "stderr": gt_stderr,
                "ids": gt_ids,
                "bindings_skipped": gt_bindings_skipped,
                "blob": gt_blob,
                "source": "executed",
            }

        gt_record = ground_truth_cache[case_key]

        for model_idx, model_cfg in enumerate(models, start=1):
            resolved_model = _resolve_model_config(model_cfg, global_cfg)
            _configure_backend(resolved_model, global_cfg)
            set_few_shot_enabled(True)
            # Number of retrieved examples. Absent from the config it stays at 3, the value
            # every recorded run used; set `few_shot_k` to run the more-examples comparison.
            set_few_shot_k(global_cfg.get("few_shot_k"))
            # Per supervisor guidance the model always receives the complete schema
            # (entities + attributes + relations); schema richness is not a design variable.
            set_schema_level("full")
            set_generation_temperature(resolved_model.get("temperature", global_cfg.get("temperature")))
            # Both default to today's behaviour, so a config that does not mention them produces
            # byte-identical generation to every run recorded before 2026-09-02.
            set_constrained_decoding(constrained_decoding)
            set_generation_token_budget(generation_token_budget)

            for strategy_idx, prompt_strategy in enumerate(prompt_strategies, start=1):
                for schema_idx, schema_enforced in enumerate(schema_enforced_values, start=1):
                    effective_schema_validation = schema_validation
                    if not schema_enforced and not schema_validation_when_no_schema:
                        effective_schema_validation = False

                    # The decoding settings are part of the CONDITION, not of the shard, so they
                    # belong in `config_id`. Without them the decoding ablation's three rungs are
                    # one condition: they share model, strategy and schema setting, so all 507
                    # records collapse to 169 distinct `_record_key`s and `--merge-shards` drops
                    # two thirds as duplicates -- the exact failure the 96-shard merge hit on
                    # 2026-08-27, and silent again here. It would also pool all three rungs into a
                    # single `by_config` row, which is the comparison the experiment exists to make.
                    #
                    # Suffixed only when non-default, so every condition recorded before
                    # 2026-09-02 keeps its published `config_id` and nothing downstream re-slices.
                    decoding_suffix = ""
                    if constrained_decoding:
                        decoding_suffix += "_constrained"
                    if force_root_arity:
                        decoding_suffix += "_arityforced"
                    if generation_token_budget is not None:
                        decoding_suffix += f"_tok{generation_token_budget}"
                    config_id = _slug(
                        f"{resolved_model['label']}_{prompt_strategy}"
                        f"_schema-{'on' if schema_enforced else 'off'}{decoding_suffix}"
                    )

                    for repeat_idx in range(1, repeats + 1):
                        current_key = _record_key(case_idx, config_id, repeat_idx,
                                                  nl_query, ocel_path)
                        if current_key in completed_keys:
                            continue

                        # Derived from the record's own identity rather than a counter, so the
                        # draw is stable across re-runs, distinct across repeats, and unchanged
                        # by where the case-shard boundaries fall. A counter would make sample 7
                        # of a 4-way split differ from sample 7 of a single job.
                        set_generation_seed(
                            None if not generation_seed_base
                            else int(hashlib.sha256(
                                f"{generation_seed_base}:{current_key}".encode("utf-8")
                            ).hexdigest()[:8], 16)
                        )

                        run_counter += 1
                        reset_llm_call_stats()
                        reset_query_exec_stats()

                        output_dir_runs = run_dir / "generated" / f"case_{case_idx:03d}"
                        output_dir_runs.mkdir(parents=True, exist_ok=True)
                        output_file = output_dir_runs / f"{config_id}_rep{repeat_idx}.json"

                        start_time = time.time()
                        pipeline_status = "ok"
                        error = None
                        trace = None
                        ocpq_json = None
                        query_stdout = None
                        query_result_file = None

                        _log_phase(
                            "generation",
                            f"{_format_run_progress(case_idx, total_cases, strategy_idx, len(prompt_strategies), schema_idx, len(schema_enforced_values), repeat_idx, repeats, run_counter, total_runs)} | "
                            f"model {model_idx}/{len(models)}:{resolved_model['label']} | "
                            f"{prompt_strategy} | schema={'on' if schema_enforced else 'off'} | eta={_format_duration(_estimate_remaining_eta(evaluation_start_time, run_counter - 1, total_runs))}",
                        )

                        # ORACLE conditions only: hand the model part of the GOLD tree. None for
                        # every ordinary strategy, so the hint cannot leak into a real condition.
                        structure_hint_level = ORACLE_STRUCTURE_LEVELS.get(prompt_strategy)
                        structure_hint = _build_structure_hint(gold_query, structure_hint_level)

                        # ORACLE: pin the root box's variable counts to gold's. Set per case, the
                        # way `set_generation_seed` is, because the constraint depends on the case
                        # -- a shard boundary must not change which grammar a case ran under.
                        # None (flag off, or gold unreadable) leaves the grammar untightened, and
                        # the record says which so those cases can be excluded.
                        case_root_arity = (_gold_root_arity(gold_query)
                                           if force_root_arity else None)
                        set_forced_root_arity(case_root_arity)
                        normalized_fields = []

                        try:
                            with _case_time_limit(case_timeout_sec):
                                _, _, ocpq_json, query_stdout, query_result_file = nl_to_ocpq_pipeline(
                                    query=nl_query,
                                    dataset=retrieval_dataset,
                                    embeddings=embeddings,
                                    output_file=str(output_file),
                                    ocel_path=ocel_path,
                                    model_name=resolved_model["name"],
                                    prompt_strategy=prompt_strategy,
                                    schema_enforced=schema_enforced,
                                    schema_validation=effective_schema_validation,
                                    execute_query=True,
                                    structure_hint=structure_hint,
                                )
                            # Read straight after the pipeline, before anything else can reset it:
                            # `validate_and_normalize_ocpq_json` clears the record on every call,
                            # so a later validation elsewhere would wipe what this case needed.
                            normalized_fields = get_last_normalized_fields()
                        # Ordered before the general handler below, which would otherwise absorb
                        # the timeout into `pipeline_status = "error"` and lose the distinction.
                        except CaseTimeout as exc:
                            pipeline_status = "timeout"
                            error = str(exc)
                            trace = None  # A deadline is not a defect; a traceback says nothing.
                            _log_phase("timeout", f"case {case_idx:03d} | {exc}")
                        except Exception as exc:
                            pipeline_status = "error"
                            error = str(exc)
                            trace = traceback.format_exc()
                            _log_phase("error", f"case {case_idx:03d} | {exc}")

                        total_duration = time.time() - start_time
                        llm_stats = _summarize_llm_stats(get_llm_call_stats())
                        exec_stats = _summarize_exec_stats(get_query_exec_stats())

                        # Everything below derives metrics from model-authored JSON, and a
                        # prediction is data rather than a control path: whatever the model
                        # emitted must be scorable, or scored 0, but never fatal. The generation
                        # call above is already guarded this way; this block was not, and on
                        # 2026-08-18 (job 3039375) a `newObjectVars` emitted as a list took the
                        # whole evaluation down at run 48 of 48. That shape is now handled in
                        # `_declared_type_names` — this guard exists for the shapes nobody has
                        # thought of yet, which a 6,084-run matrix is far likelier to meet than
                        # a 48-run smoke.
                        #
                        # A failure is recorded on the record as `scoring_error`, never silently
                        # averaged: `similarity` is None, so the strict metric treats it as 0 by
                        # the same convention as an unparseable prediction, and the field makes
                        # the distinction visible when reporting.
                        scoring_error = None
                        oversized_bytes = None
                        try:
                            pred_ids, pred_bindings_skipped, oversized_bytes = _load_result_ids(
                                query_result_file, ocpq_json, max_gb=max_result_file_gb
                            )
                            similarity = _compute_similarity(gt_record.get("ids"), pred_ids)
                            pred_events, pred_objects = _extract_query_entities(ocpq_json)
                            gold_events, gold_objects = _extract_query_entities(gold_query)
                            hallucinated = bool(
                                pred_events - gold_events or pred_objects - gold_objects
                            )
                            unsupported = _contains_unsupported_filter(ocpq_json)
                            # Grounding metrics: did the model invent an attribute name (the
                            # `isMarried` -> `is_Married` failure) or link two entity types that
                            # the log never relates? Post-hoc only; never fed back to the model.
                            bad_attributes, bad_relations = _schema_grounding_errors(
                                ocpq_json, ocel_path
                            )
                        except Exception as exc:  # noqa: BLE001 - malformed prediction, not a bug
                            scoring_error = f"{type(exc).__name__}: {exc}"
                            pred_ids, pred_bindings_skipped = None, None
                            similarity = None
                            hallucinated = False
                            unsupported = False
                            bad_attributes, bad_relations = set(), set()
                            _log_phase(
                                "error",
                                f"case {case_idx:03d} | scoring failed, record kept: {scoring_error}",
                            )

                        schema_valid = False
                        if isinstance(ocpq_json, dict):
                            try:
                                BindingBoxTree.model_validate(ocpq_json)
                                schema_valid = True
                            except Exception:
                                schema_valid = False
                        final_status = _classify_run_status(
                            pipeline_status,
                            error,
                            schema_valid,
                            exec_stats,
                            query_result_file,
                            query_stdout,
                            result_too_large=oversized_bytes is not None,
                        )

                        _log_phase("schema_check", f"case {case_idx:03d} | status={final_status}")
                        _log_phase(
                            "execution",
                            f"case {case_idx:03d} | success={bool((exec_stats or {}).get('success'))} "
                            f"| result_file={'yes' if query_result_file else 'no'}",
                        )
                        _log_phase(
                            "similarity",
                            f"case {case_idx:03d} | combined_f1="
                            f"{((similarity or {}).get('combined') or {}).get('f1')}",
                        )

                        # Persist the extracted result set before the raw file goes away. Placed
                        # after `_classify_run_status` for the same reason the discard is: that
                        # function reads `query_result_file`, so nothing may run between it and
                        # the status it produces. A store failure must never fail a completed
                        # generation, hence the guard.
                        pred_blob = None
                        if result_store_enabled and pred_ids:
                            try:
                                pred_blob = result_store.store(
                                    result_store_dir, ocel_path, ocpq_json, pred_ids,
                                    pred_bindings_skipped, result_store_max_situations,
                                )
                            except Exception as exc:
                                _log_phase("execution", f"result store failed for case {case_idx}: {exc}")

                        # Release the raw results only now: `_classify_run_status` above reads
                        # `query_result_file` as its "did this produce a result" signal, so
                        # discarding earlier made every successful run look execution_failed.
                        # Sweep every execution this run performed, not just the one whose path
                        # was returned — the auto-fix retries in LLM.py execute the query more
                        # than once and only the last path comes back.
                        if not keep_result_files:
                            for exec_entry in get_query_exec_stats():
                                discard_result_file(exec_entry.get("result_file"))
                            discard_result_file(query_result_file)
                            query_result_file = None
                            # exec_stats was summarised above and captured the path; clear it
                            # so predictions.json never points at a file that no longer exists.
                            exec_stats["result_file"] = None

                        # Strict convention, matching `result_set_f1_strict`: a prediction that
                        # did not parse or did not execute counts as F1 0, so skill -1. Scoring
                        # it as undefined would reward attempting fewer queries.
                        _strict_f1 = ((similarity or {}).get("combined") or {}).get("f1")
                        _is_disc, _floor, _variant_floors, _floor_detail = (
                            stratum_index.get(nl_query, _STRATUM_MISS))
                        # The max over degenerate strategies, falling back to the no-op score for
                        # an audit file that predates it.
                        _trivial_floor = _variant_floors.get("strict")
                        if _trivial_floor is None:
                            _trivial_floor = _floor
                        skill_value = _skill_score(
                            0.0 if _strict_f1 is None else _strict_f1, _trivial_floor)

                        record = {
                            "case_id": case_idx,
                            "nl_query": nl_query,
                            "ocel_path": ocel_path,
                            "domain_seen": _is_domain_seen(ocel_path),
                            # Reporting stratum. `discriminating` is False for compliance rules
                            # the log fully satisfies, where a result-only comparison awards full
                            # marks to any query; `baseline_f1` is what a no-op query already
                            # scores on this entry, so a model number can be read against a floor.
                            "discriminating": _is_disc,
                            # The no-op strategy's score, kept for provenance and because older
                            # notes quote it. NOT what skill is measured against.
                            "baseline_f1": _floor,
                            # The operative floor: best of the degenerate strategies.
                            "trivial_floor": _trivial_floor,
                            # The same floor under each entry of SIMILARITY_VARIANTS, so a
                            # relaxed F1 is never read against the strict floor. A value present
                            # but None marks a variant undefined on this entry.
                            "baseline_f1_variants": _variant_floors,
                            "baseline_f1_variants_detail": _floor_detail,
                            "gold_query": gold_query,
                            "model": resolved_model["name"],
                            "model_label": resolved_model["label"],
                            "backend": resolved_model.get("backend", global_cfg.get("backend")),
                            "backend_description": get_hf_backend_description(),
                            "prompt_strategy": prompt_strategy,
                            # Non-null ONLY for the oracle ablation conditions, and then it names
                            # exactly how much of the gold tree the model was given. Recorded in
                            # full rather than as a flag so a reader can audit what was handed
                            # over instead of trusting this comment. A null level with an oracle
                            # strategy means gold was malformed and the run fell back to the
                            # baseline prompt -- those cases must be excluded when reporting.
                            "structure_hint_level": structure_hint_level,
                            "structure_hint": structure_hint,
                            "schema_enforced": schema_enforced,
                            "schema_validation": effective_schema_validation,
                            # Per record, not only in the manifest: after `--merge-shards` the
                            # manifest is gone and a reader still has to be able to tell a
                            # constrained prediction from a free one.
                            "constrained_decoding": constrained_decoding,
                            "generation_token_budget": generation_token_budget,
                            # The arity actually handed to the decoder, or None where the flag was
                            # off or gold was unreadable. Recorded per record so a reader can audit
                            # what was given rather than trust the flag.
                            "forced_root_arity": case_root_arity,
                            # What the harness filled in that the model did not produce. A
                            # normalisation flatters the model; the count has to be visible.
                            "normalized_fields": normalized_fields,
                            "repeat": repeat_idx,
                            "config_id": config_id,
                            "status": final_status,
                            "pipeline_status": pipeline_status,
                            "error": error,
                            # Set only when metric derivation itself failed on this record, as
                            # opposed to `error`, which reports a failure of the generation
                            # pipeline. Non-null means the metrics here are defaults, not
                            # measurements; count these when reporting rather than averaging them.
                            "scoring_error": scoring_error,
                            "traceback": trace,
                            "evaluation_phase": "run_complete",
                            "evaluation_phase_trace": RUN_PHASE_TRACE,
                            "prediction": ocpq_json,
                            # Blob ids in `<run_dir>/result_store`, so a record points straight
                            # at its stored results and `--recompute` needs no re-derivation.
                            # Both are content addresses of (log, query), so `gold` repeats
                            # across every condition for a case and `prediction` collapses
                            # whenever two conditions emitted the same query.
                            "result_ref": {
                                "gold": gt_record.get("blob"),
                                "prediction": pred_blob,
                            },
                            "query_stdout": query_stdout,
                            "query_result_file": query_result_file,
                            "llm": llm_stats,
                            "execution": exec_stats,
                            "total_duration_sec": round(total_duration, 4),
                            "metrics": {
                                "schema_valid": schema_valid,
                                "hallucinated_entities": hallucinated,
                                "unsupported_filters": unsupported,
                                "hallucinated_attributes": sorted(bad_attributes),
                                "invalid_relations": sorted(bad_relations),
                                # Schema sections dropped for exceeding SCHEMA_BLOCK_TOKEN_BUDGET.
                                # `build_schema_block` has always recorded these but nothing read
                                # them, so the budget silently truncating the two largest logs was
                                # invisible in the results. Must stay empty: the design is that
                                # every model always receives the complete schema.
                                "schema_omissions": get_last_schema_omissions(),
                                # ocpq_cli drops bindings when a query yields too many;
                                # a comparison against a truncated result is meaningless,
                                # so keep it identifiable after the run.
                                "bindings_skipped": pred_bindings_skipped,
                                # Set when the export was too large to load at all. The
                                # query executed; its answer is unusable. Scored 0 strict
                                # via `similarity: None`, never excluded -- dropping these
                                # would remove exactly the model's worst over-selecting
                                # queries, the selection bias rejected for truncation.
                                "result_too_large": oversized_bytes is not None,
                                "result_file_bytes": oversized_bytes,
                                "gold_bindings_skipped": gt_record.get("bindings_skipped"),
                                # Query-level agreement: did the model express the right
                                # constructs? A result comparison cannot see this, and on the
                                # non-discriminating stratum it is the only informative signal.
                                "structural": _structural_metrics(gold_query, ocpq_json),
                                # How far above the trivial floor this prediction got, in
                                # [-1, 1]. Reported beside the raw F1, not instead of it: a
                                # do-nothing query scores 0 here but ~0.61 on raw F1.
                                "skill": skill_value,
                            },
                            "similarity": similarity,
                            "ground_truth": {
                                # "executed" or "result_store" -- on a resume the gold result is
                                # read back rather than re-run, so a null duration here is the
                                # reuse path, not a missing measurement.
                                "source": gt_record.get("source"),
                                "result_path": gt_record.get("result_path"),
                                "duration_sec": gt_record.get("duration_sec"),
                                "stdout": gt_record.get("stdout"),
                                "stderr": gt_record.get("stderr"),
                                "bindings_skipped": gt_record.get("bindings_skipped"),
                            },
                        }
                        # After the literal, not inside it: this reads `similarity` and
                        # `baseline_f1_variants` off the finished record, so the projections and
                        # their matched floors are joined in exactly one place rather than
                        # re-derived here and again in `--recompute`.
                        record["metrics"]["skill_variants"] = _record_skill_variants(record)

                        records.append(record)
                        completed_keys.add(current_key)
                        completed_runs = len(records)
                        checkpoint_payload = {
                            "run_id": run_id,
                            "config_fingerprint": config_fingerprint,
                            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "elapsed_sec": round(time.time() - evaluation_start_time, 4),
                            "completed_runs": completed_runs,
                            "total_runs": total_runs,
                            "records": records,
                        }
                        _write_json(run_dir / "predictions.json", records)
                        _write_json(run_dir / "checkpoint.json", checkpoint_payload)

    _write_json(run_dir / "predictions.json", records)
    _write_json(
        run_dir / "checkpoint.json",
        {
            "run_id": run_id,
            "config_fingerprint": config_fingerprint,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_sec": round(time.time() - evaluation_start_time, 4),
            "completed_runs": len(records),
            "total_runs": total_runs,
            "records": records,
        },
    )

    summary = _build_summary(records, repeats)
    summary["evaluation_phases"] = EVALUATION_PHASES
    summary["evaluation_phase_trace"] = EVALUATION_PHASES
    if result_store_enabled:
        store_stats = result_store.stats(result_store_dir)
        summary["result_store"] = dict(store_stats, dir=str(result_store_dir))
        _log_phase(
            "summary",
            f"result store: {store_stats['blobs']} blobs, "
            f"{store_stats['bytes'] / 1048576:.1f} MB",
        )
    _log_phase("summary", "Writing summary.json and completing the run.")
    _write_json(run_dir / "summary.json", summary)
    _run_bb_analysis(run_dir)
    _run_divergence_analysis(run_dir)

    return run_dir


def _merge_failure_logs(shard_dirs, output_dir: Path) -> None:
    """Concatenate the shards' raw-output failure logs into the merged run.

    Each shard writes its own `failure_log.jsonl` (one writer per file), so merging is a
    plain concatenation. Keeps the merged directory as self-contained as its predictions.
    """
    lines = []
    for shard_dir in shard_dirs:
        log_path = Path(shard_dir) / "failure_log.jsonl"
        if not log_path.exists():
            continue
        with open(log_path, encoding="utf-8") as f:
            lines.extend(line for line in f if line.strip())
    if not lines:
        return
    with open(output_dir / "failure_log.jsonl", "w", encoding="utf-8") as f:
        f.writelines(line if line.endswith("\n") else line + "\n" for line in lines)
    print(f"Merged {len(lines)} failure-log entries from {len(shard_dirs)} shard(s).")


def _run_divergence_analysis(run_dir: Path) -> None:
    """Write the error-analysis report for this run.

    Purely derived from `predictions.json` — it classifies how each imperfect prediction
    differs from gold and keeps a capped set of concrete query pairs. Diagnostic only:
    nothing here feeds `summary.json`. Failure to produce it must never fail a run that
    otherwise completed, hence the broad guard.
    """
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        return
    labels_path = paths.DATA / "testset_bb_labels.json"
    try:
        from analyze_divergence import build_report
        from slice_results_by_bb import _build_bb_index

        bb_index = _build_bb_index(_read_json(labels_path)) if labels_path.exists() else {}
        report = build_report(_read_json(predictions_path), bb_index)
        _write_json(run_dir / "divergence_report.json", report)
        totals = report["totals"]
        _log_phase(
            "divergence_analysis",
            f"wrote divergence_report.json ({totals['diverging']}/{totals['records']} diverging, "
            f"{len(report['examples'])} example(s) kept)",
        )
    except Exception as exc:  # noqa: BLE001 - analysis must not fail a completed run
        _log_phase("divergence_analysis", f"skipped: {exc}")


def _run_bb_analysis(run_dir: Path) -> None:
    """Write per-BB and per-category breakdowns if the BB labels file is present.

    Guarded: this runs after `summary.json` is written, so a failure here must not fail an
    evaluation whose results are already on disk and already complete.
    """
    labels_path = paths.DATA / "testset_bb_labels.json"
    predictions_path = run_dir / "predictions.json"
    if not labels_path.exists() or not predictions_path.exists():
        return
    try:
        from slice_results_by_bb import _build_bb_index, slice_by_bb, slice_by_category
        predictions = _read_json(predictions_path)
        labels_data = _read_json(labels_path)
        bb_index = _build_bb_index(labels_data)
        breakdown = slice_by_bb(predictions, bb_index)
        category_breakdown = slice_by_category(predictions, bb_index)
        _write_json(run_dir / "bb_breakdown.json", breakdown)
        _write_json(run_dir / "bb_category_breakdown.json", category_breakdown)
        _log_phase(
            "bb_analysis",
            f"wrote bb_breakdown.json + bb_category_breakdown.json ({len(breakdown)} config(s))",
        )
    except Exception as exc:  # noqa: BLE001 - analysis must not fail a completed run
        _log_phase("bb_analysis", f"skipped: {exc}")


def _build_summary(records, repeats):
    by_config = defaultdict(list)
    for rec in records:
        by_config[rec["config_id"]].append(rec)

    def _result_f1_values(recs):
        """(strict, executed) result-F1 lists.

        `strict` scores a record that produced no comparable result as 0 -- schema-invalid or
        non-executing counts as a failure to answer. `executed` keeps only the records that did
        produce one. Reporting both is deliberate: `_mean` drops None, so the executed figure
        alone is silently conditional on execution success and a model that attempts fewer
        queries scores higher for it. The two together separate "could not produce a runnable
        query" from "produced the wrong answer".
        """
        strict, executed = [], []
        for r in recs:
            f1 = ((r.get("similarity") or {}).get("combined") or {}).get("f1")
            strict.append(0.0 if f1 is None else f1)
            if f1 is not None:
                executed.append(f1)
        return strict, executed

    def _skill_values(recs):
        """(strict, executed, defined_count) skill lists.

        Skill is undefined where the floor is 1.0 -- precisely the non-discriminating entries,
        where a no-op query already scores full marks and there is nothing to beat. Those are
        excluded rather than counted as 0, so the count is returned alongside: `_mean` drops
        None, and a mean whose denominator is invisible is the defect that made `combined_f1`
        unreadable.
        """
        strict, executed = [], []
        for r in recs:
            s = _record_skill(r, strict=True)
            if s is not None:
                strict.append(s)
            e = _record_skill(r, strict=False)
            if e is not None:
                executed.append(e)
        return strict, executed

    def _variant_bundle(recs):
        """Per-variant F1, floor and skill, each with its own denominator.

        Every figure here carries `n`: the variants do not share a denominator, because
        `violation` is undefined wherever gold flags nothing (22 of the 135 discriminating
        entries). A mean whose denominator is invisible is the defect that made `combined_f1`
        unreadable, and it would be worse here -- the same table would mix two populations.

        `strict` reproduces `result_set_f1_strict` and `skill_score` exactly. That is the
        regression check, not a redundancy: if it ever diverges, the projection table has
        drifted from the primary metric.
        """
        bundle = {}
        for name in SIMILARITY_VARIANTS:
            f1_vals, skill_vals = [], []
            for r in recs:
                f1 = _record_variant_f1(r, name, strict=True)
                if f1 is not None:
                    f1_vals.append(f1)
                s = _skill_score(f1, (r.get("baseline_f1_variants") or {}).get(name))
                if s is not None:
                    skill_vals.append(s)
            # `baseline_f1_variants` holds the MAX over degenerate strategies -- the operative
            # floor. `_detail` keeps the components so a reader can see which strategy set it;
            # for `violation` the answer is always flag-everything, since the no-op baseline
            # strips the constraints and therefore flags nobody.
            floors = [(r.get("baseline_f1_variants") or {}).get(name) for r in recs]
            noop = [((r.get("baseline_f1_variants_detail") or {}).get(name) or {}).get("noop")
                    for r in recs]
            bundle[name] = {
                "result_set_f1_strict": _mean(f1_vals),
                "n": len(f1_vals),
                "undefined": len(recs) - len(f1_vals),
                "mean_trivial_floor": _mean(floors),
                "mean_baseline_f1_noop": _mean(noop),
                "skill_score": _mean(skill_vals),
                "skill_n": len(skill_vals),
                "below_floor_rate": _mean([1 if s < 0 else 0 for s in skill_vals]),
            }
        return bundle

    def _metric_bundle(recs):
        strict_vals, executed_vals = _result_f1_values(recs)
        executed_f1 = _mean(executed_vals)
        structural = [((r.get("metrics") or {}).get("structural") or {}) for r in recs]
        skill_strict, skill_executed = _skill_values(recs)
        return {
            "n": len(recs),
            # Primary: every record counted, failures included.
            "result_set_f1_strict": _mean(strict_vals),
            # Conditional on producing a comparable result. Same value as the legacy keys below.
            "result_set_f1_executed": executed_f1,
            # Query-level: did it express the right building blocks? Defined whenever the
            # prediction parses, 0 when it does not, so it needs no strict/executed split.
            "structural_f1": _mean([s.get("f1") for s in structural]),
            "structural_precision": _mean([s.get("precision") for s in structural]),
            "structural_recall": _mean([s.get("recall") for s in structural]),
            # What a no-op query already scores on these items -- the floor a model must beat.
            # The NO-OP strategy's mean. Kept at that meaning so the 0.6124 / 0.5148 figures in
            # the notes stay true; it is no longer what skill is measured against.
            "mean_baseline_f1": _mean([r.get("baseline_f1") for r in recs]),
            # The operative floor: best of the degenerate strategies, per entry, then averaged.
            # This is the denominator behind `skill_score`. See notes §27.
            "mean_trivial_floor": _mean([_record_floor(r) for r in recs]),
            # The same result comparison renormalised against that floor, in [-1, 1]: 0 means
            # "no better than a do-nothing query", -1 "returned nothing correct", 1 "perfect".
            # Reported *beside* result_set_f1_strict, never instead of it -- the pair is what
            # separates a model that is right from one that is merely safe.
            "skill_score": _mean(skill_strict),
            # Parity with result_set_f1_executed, and misleading for the same reason: failures
            # are excluded rather than penalised. Never report alone.
            "skill_score_executed": _mean(skill_executed),
            # What a clamp at 0 would have hidden: how often the model did worse than trivial.
            "below_floor_rate": _mean([1 if s < 0 else 0 for s in skill_strict]),
            # Denominators, so the skill figures above are never silently conditional.
            "skill_n": len(skill_strict),
            "skill_undefined": len(recs) - len(skill_strict),
            # Decompositions of the same result comparison, each against its own matched floor.
            # `variants.strict` reproduces the three figures above exactly; the others are
            # secondary and must never be quoted as the headline. See SIMILARITY_VARIANTS.
            "variants": _variant_bundle(recs),
            # Deprecated aliases of result_set_f1_executed, kept so existing notes stay true.
            # `semantic_accuracy` in particular overstates what is measured: this is result-set
            # agreement, not semantic correctness.
            "combined_f1": executed_f1,
            "schema_valid_rate": _mean([1 if r["metrics"]["schema_valid"] else 0 for r in recs]),
            "execution_success_rate": _mean(
                [1 if (r.get("execution") or {}).get("success") else 0 for r in recs]
            ),
            "hallucinated_attribute_rate": _mean(
                [1 if r["metrics"].get("hallucinated_attributes") else 0 for r in recs]
            ),
            "invalid_relation_rate": _mean(
                [1 if r["metrics"].get("invalid_relations") else 0 for r in recs]
            ),
            # How often the backend gave up and returned a partial result. Not a failure rate:
            # these records are scored as measured, and that score is inflated. See
            # `_truncation_counts`.
            **_truncation_counts(recs),
            # How often the per-case cap fired. Scored 0, never excluded -- see
            # `_timeout_counts`. Must be added to the by_config block below as well.
            **_timeout_counts(recs),
            **_oversized_counts(recs),
        }

    # Leakage split: seen vs unseen domains, overall and per model. If the fine-tuning gain
    # holds on unseen logs it is a real skill; if it only shows on seen logs it was memorised
    # vocabulary. After de-leaking + retraining, seen and unseen should be comparable.
    seen_recs = [r for r in records if r.get("domain_seen")]
    unseen_recs = [r for r in records if not r.get("domain_seen")]
    by_domain_seen = {
        "seen": _metric_bundle(seen_recs) if seen_recs else None,
        "unseen": _metric_bundle(unseen_recs) if unseen_recs else None,
        "by_model": {},
    }
    for model_label in sorted({r.get("model_label") for r in records}):
        m_recs = [r for r in records if r.get("model_label") == model_label]
        by_domain_seen["by_model"][model_label] = {
            "seen": _metric_bundle([r for r in m_recs if r.get("domain_seen")]),
            "unseen": _metric_bundle([r for r in m_recs if not r.get("domain_seen")]),
        }

    # Discriminability stratum. Roughly a fifth of the testset are compliance rules their log
    # fully satisfies: correct gold queries that flag nobody, where a result-only comparison
    # awards full marks to any query (their `mean_baseline_f1` sits near 1.0). Pooling them with
    # the informative entries inflates the headline and compresses the differences between
    # models. Both strata are reported and the overall figure is kept -- dropping a stratum from
    # the headline would reintroduce at reporting time the selection bias that removing those
    # entries from the testset caused in the first place.
    disc_recs = [r for r in records if r.get("discriminating") is True]
    nondisc_recs = [r for r in records if r.get("discriminating") is False]
    by_stratum = {
        "discriminating": _metric_bundle(disc_recs) if disc_recs else None,
        "non_discriminating": _metric_bundle(nondisc_recs) if nondisc_recs else None,
        "unlabelled": len([r for r in records if r.get("discriminating") is None]),
        "by_model": {},
    }
    for model_label in sorted({r.get("model_label") for r in records}):
        m_recs = [r for r in records if r.get("model_label") == model_label]
        by_stratum["by_model"][model_label] = {
            "discriminating": _metric_bundle([r for r in m_recs
                                              if r.get("discriminating") is True]),
            "non_discriminating": _metric_bundle([r for r in m_recs
                                                  if r.get("discriminating") is False]),
        }

    summary = {
        "total_records": len(records),
        "by_config": {},
        "by_domain_seen": by_domain_seen,
        "by_stratum": by_stratum,
        "evaluation_phase": "summary",
        # Run-wide timeout total. The per-condition and per-stratum breakdowns are where the
        # comparison lives; this is the number that says at a glance whether the cap bound at
        # all, without having to open by_config.
        "timeouts": _timeout_counts(records),
        "oversized_results": _oversized_counts(records),
    }

    for config_id, recs in by_config.items():
        schema_valid_rate = _mean([1 if r["metrics"]["schema_valid"] else 0 for r in recs])
        hallucinated_rate = _mean([1 if r["metrics"]["hallucinated_entities"] else 0 for r in recs])
        unsupported_rate = _mean([1 if r["metrics"]["unsupported_filters"] else 0 for r in recs])
        # Grounding: fraction of queries inventing an attribute name / an unlinked relation.
        hallucinated_attribute_rate = _mean(
            [1 if r["metrics"].get("hallucinated_attributes") else 0 for r in recs]
        )
        invalid_relation_rate = _mean(
            [1 if r["metrics"].get("invalid_relations") else 0 for r in recs]
        )
        exec_success_rate = _mean([1 if (r.get("execution") or {}).get("success") else 0 for r in recs])

        prompt_tokens = _mean([r["llm"].get("prompt_tokens") for r in recs])
        # Peak, not just the mean: nothing in the pipeline truncates the prompt, so an
        # overflow on one large-schema log is invisible in an average but is exactly the
        # case that produces degraded output. Reported so it can be checked, not guessed.
        _prompt_values = [r["llm"].get("prompt_tokens") for r in recs
                          if r["llm"].get("prompt_tokens") is not None]
        max_prompt_tokens = max(_prompt_values) if _prompt_values else None
        completion_tokens = _mean([r["llm"].get("completion_tokens") for r in recs])
        total_tokens = _mean([r["llm"].get("total_tokens") for r in recs])
        avg_completion_chars = _mean([r["llm"].get("completion_chars") for r in recs])
        avg_llm_calls = _mean([r["llm"].get("calls") for r in recs])
        llm_duration = _mean([r["llm"].get("duration_sec") for r in recs])
        exec_duration = _mean([r["execution"].get("duration_sec") for r in recs])
        total_duration = _mean([r.get("total_duration_sec") for r in recs])

        event_jaccard = _mean([
            (r.get("similarity") or {}).get("events", {}).get("jaccard")
            for r in recs
            if r.get("similarity")
        ])
        event_precision = _mean([
            (r.get("similarity") or {}).get("events", {}).get("precision")
            for r in recs
            if r.get("similarity")
        ])
        event_recall = _mean([
            (r.get("similarity") or {}).get("events", {}).get("recall")
            for r in recs
            if r.get("similarity")
        ])
        object_jaccard = _mean([
            (r.get("similarity") or {}).get("objects", {}).get("jaccard")
            for r in recs
            if r.get("similarity")
        ])
        object_precision = _mean([
            (r.get("similarity") or {}).get("objects", {}).get("precision")
            for r in recs
            if r.get("similarity")
        ])
        object_recall = _mean([
            (r.get("similarity") or {}).get("objects", {}).get("recall")
            for r in recs
            if r.get("similarity")
        ])
        combined_f1 = _mean([
            (r.get("similarity") or {}).get("combined", {}).get("f1")
            for r in recs
            if r.get("similarity")
        ])
        combined_precision = _mean([
            (r.get("similarity") or {}).get("combined", {}).get("precision")
            for r in recs
            if r.get("similarity")
        ])
        combined_recall = _mean([
            (r.get("similarity") or {}).get("combined", {}).get("recall")
            for r in recs
            if r.get("similarity")
        ])

        # Strict counterpart: `_mean` drops None, so the values above are computed only over
        # records that produced a comparable result. Substituting 0 for the rest is what makes
        # the primary metric self-contained and immune to scoring well by attempting less.
        strict_vals, _executed_vals = _result_f1_values(recs)
        result_set_f1_strict = _mean(strict_vals)
        _structural = [((r.get("metrics") or {}).get("structural") or {}) for r in recs]
        structural_f1 = _mean([s.get("f1") for s in _structural])
        structural_precision = _mean([s.get("precision") for s in _structural])
        structural_recall = _mean([s.get("recall") for s in _structural])
        mean_baseline_f1 = _mean([r.get("baseline_f1") for r in recs])
        mean_trivial_floor = _mean([_record_floor(r) for r in recs])
        # `_metric_bundle` above computes the same figures for the stratum/domain splits; this
        # per-condition block is a separate code path, so anything added there must be added
        # here too or it is silently missing from by_config.
        _skill_strict, _skill_executed = _skill_values(recs)

        summary["by_config"][config_id] = {
            "meta": {
                "model_label": recs[0].get("model_label"),
                "model": recs[0].get("model"),
                "prompt_strategy": recs[0].get("prompt_strategy"),
                "schema_enforced": recs[0].get("schema_enforced"),
                "schema_validation": recs[0].get("schema_validation"),
                "repeats": repeats,
            },
            "metrics": {
                "schema_valid_rate": schema_valid_rate,
                "hallucinated_entity_rate": hallucinated_rate,
                "hallucinated_attribute_rate": hallucinated_attribute_rate,
                "invalid_relation_rate": invalid_relation_rate,
                "unsupported_filter_rate": unsupported_rate,
                "execution_success_rate": exec_success_rate,
                "event_jaccard": event_jaccard,
                "event_precision": event_precision,
                "event_recall": event_recall,
                "object_jaccard": object_jaccard,
                "object_precision": object_precision,
                "object_recall": object_recall,
                # PRIMARY. Every record counted: a prediction that did not parse or did not
                # execute scores 0 rather than being dropped from the mean.
                "result_set_f1_strict": result_set_f1_strict,
                # Conditional on producing a comparable result. Read together with
                # execution_success_rate, never alone -- on its own it rewards attempting fewer
                # queries, since failures are excluded rather than penalised.
                "result_set_f1_executed": combined_f1,
                # Query-level agreement over the 42 building blocks: did the model express the
                # right constructs? This is the only informative signal on the entries whose
                # log fully complies, where result comparison gives everyone full marks.
                "structural_f1": structural_f1,
                "structural_precision": structural_precision,
                "structural_recall": structural_recall,
                # The floor: what a no-op query already scores on these items.
                # The no-op strategy alone; kept at that meaning for the notes. The floor skill is
                # measured against is `mean_trivial_floor` -- best of the degenerate strategies.
                "mean_baseline_f1": mean_baseline_f1,
                "mean_trivial_floor": mean_trivial_floor,
                # Result F1 renormalised against that floor, in [-1, 1]. 0 = no better than a
                # do-nothing query, -1 = nothing correct, 1 = perfect. Read next to
                # result_set_f1_strict: a model emitting trivial queries scores ~0.61 there and
                # exactly 0 here, which is the distinction the raw F1 cannot make.
                "skill_score": _mean(_skill_strict),
                # Parity with result_set_f1_executed; misleading alone for the same reason.
                "skill_score_executed": _mean(_skill_executed),
                "below_floor_rate": _mean([1 if s < 0 else 0 for s in _skill_strict]),
                "skill_n": len(_skill_strict),
                "skill_undefined": len(recs) - len(_skill_strict),
                # Mirrors the `variants` block in `_metric_bundle` above. The two aggregation
                # paths are separate by construction, so this must be added in both or it is
                # silently missing from by_config -- the defect the comment below warns about.
                "variants": _variant_bundle(recs),
                # Deprecated aliases of result_set_f1_executed. `semantic_accuracy` overstates
                # what is measured -- this is result-set agreement, not semantic correctness.
                "combined_f1": combined_f1,
                "semantic_accuracy": combined_f1,
                "combined_precision": combined_precision,
                "combined_recall": combined_recall,
                "avg_completion_chars": avg_completion_chars,
                "max_prompt_tokens": max_prompt_tokens,
                "avg_llm_calls": avg_llm_calls,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "llm_duration_sec": llm_duration,
                "exec_duration_sec": exec_duration,
                "total_duration_sec": total_duration,
                # Same four keys `_metric_bundle` adds to by_stratum / by_domain_seen. Added in
                # both places on purpose: this block is the separate code path the comment above
                # `_skill_strict` warns about.
                **_truncation_counts(recs),
                **_timeout_counts(recs),
                **_oversized_counts(recs),
            },
        }

    return summary


TRUNCATION_POLICIES = ("as-measured", "strict-zero")


def _apply_truncation_policy(records, policy: str) -> dict:
    """Apply a truncation policy to already-scored records, in place.

    `as-measured` changes nothing: a partial result is compared as it stands. That is the default
    and the reported primary, but it is not neutral -- see `_truncation_counts` for why it
    inflates. This function exists so the alternative can be produced from the same run.

    `strict-zero` splits on *who* was truncated, because that changes what the truncation is
    evidence of:

    * **Gold truncated** -> the entry is unscoreable for every model, so it is excluded for all
      of them. Excluding on a model-independent criterion introduces no selection bias.
    * **Prediction only** -> scored 0. The model emitted a query that cannot be evaluated, which
      is the same reading `_result_f1_values` already applies to one that does not parse or does
      not execute.

    Deliberately absent: excluding on the prediction side. That is the selection bias CLAUDE.md
    warns about with the non-discriminating entries -- a model whose worst queries are quietly
    removed outscores one that fails honestly.

    Zeroed records keep their original scores under `similarity_as_measured`, so the transform is
    auditable rather than lossy, and `execution.success` is left alone: the query *did* execute,
    and the answer is unusable rather than absent.

    Args:
        records: Prediction records, mutated in place.
        policy: One of `TRUNCATION_POLICIES`.

    Returns:
        A report of what was done, for the summary.

    Raises:
        ValueError: If `policy` is not recognised.
    """
    if policy not in TRUNCATION_POLICIES:
        raise ValueError(f"Unknown truncation policy {policy!r}; expected one of {TRUNCATION_POLICIES}")

    report = {
        "truncation_policy": policy,
        "excluded_gold_truncated": 0,
        "zeroed_prediction_truncated": 0,
    }
    if policy == "as-measured":
        return report

    # Entry-level, resolved before any record is touched: gold is executed once per entry and
    # reused from the store, so every record of an entry must be treated identically even if one
    # of them carries a stale flag.
    gold_truncated_entries = {
        (r.get("ocel_path"), r.get("nl_query"))
        for r in records
        if (r.get("metrics") or {}).get("gold_bindings_skipped")
    }

    for rec in records:
        metrics = rec.setdefault("metrics", {})
        if (rec.get("ocel_path"), rec.get("nl_query")) in gold_truncated_entries:
            rec["excluded_reason"] = "gold_truncated"
            report["excluded_gold_truncated"] += 1
            continue
        if not metrics.get("bindings_skipped"):
            continue

        similarity = rec.get("similarity")
        if isinstance(similarity, dict) and isinstance(similarity.get("combined"), dict):
            rec["similarity_as_measured"] = copy.deepcopy(similarity)
            for key in ("f1", "precision", "recall"):
                if key in similarity["combined"]:
                    similarity["combined"][key] = 0.0
            # The variants are projections of the same truncated result, so they are withdrawn
            # with it. Leaving them scored would make `strict-zero` inconsistent with itself and
            # give the relaxed metrics a sensitivity analysis the primary one does not have.
            for scored in (similarity.get("variants") or {}).values():
                if isinstance(scored, dict):
                    for key in ("f1", "precision", "recall"):
                        if key in scored:
                            scored[key] = 0.0
            # Skill reads the combined F1, so it has to follow rather than keep a value derived
            # from the score that was just withdrawn.
            metrics["skill"] = _record_skill(rec)
            metrics["skill_variants"] = _record_skill_variants(rec)
            report["zeroed_prediction_truncated"] += 1

    return report


def _report_variants(run_dir):
    """Print the result-metric decomposition for a finished run, with its own caveats.

    Each variant is read against its OWN floor. That is the whole point and it is easy to get
    wrong: a relaxed F1 next to the strict floor 0.5148 looks like progress and is arithmetic
    nonsense, which is the error the structure ablation's paired tests exist to prevent.

    The floors do not behave alike, and the report says so inline rather than leaving a reader to
    infer it. The no-op baseline is gold with its constraints, size filters and children stripped,
    so:
      * `population` -- dropping the violation flag collapses gold's flagged and unflagged rows
        onto that baseline, and its floor goes to ~0.9. A relaxation that raises the floor further
        than the model is a trap, not a rescue.
      * `kind_projected` -- the baseline binds the same variables as gold, so the projection is a
        no-op on it and the floor does not move. Any gain here is real headroom.
      * `violation` -- the baseline strips the constraints, so it flags nobody and its floor is 0
        by construction. The no-op query is simply not a comparator for this half; the skill number
        it produces is the raw F1 wearing a disguise. Reported, and not to be quoted as skill.
    """
    run_dir = Path(run_dir)
    summary = _read_json(str(run_dir / "summary.json"))

    print(f"\nResult-metric decomposition: {run_dir}")
    print("`strict` is the primary metric; the rest are secondary and none replaces it.\n")

    def _table(title, block):
        # `by_stratum` mixes metric bundles with two containers -- `unlabelled` is a count and
        # `by_model` is a dict of bundles -- so identify a bundle by its primary metric rather
        # than by position.
        if not isinstance(block, dict) or "result_set_f1_strict" not in block:
            return
        variants = block.get("variants")
        if not variants:
            print(f"{title}\n  (no variants block -- rescore with --recompute)\n")
            return
        print(title)
        print(f"  {'variant':16s} {'F1':>9s} {'F1_n':>6s} {'floor':>9s} {'(no-op)':>9s} "
              f"{'skill':>9s} {'skill_n':>8s} {'<floor':>7s}")
        strict_skill_n = (variants.get("strict") or {}).get("skill_n")
        for name, row in variants.items():
            def _f(key, width=9):
                value = row.get(key)
                return (f"{value:>{width}.4f}" if isinstance(value, (int, float))
                        else f"{'--':>{width}}")
            # A variant whose skill is averaged over fewer records than `strict` is measured on a
            # DIFFERENT subpopulation and its skill is not comparable to strict's. `population`
            # does this hard: the no-op baseline selects gold's population exactly on most
            # compliance entries, so its floor is 1.0 there and skill is undefined -- 102 of the
            # 135 discriminating entries, leaving a quarter of the records and the easier ones.
            mark = "" if row.get("skill_n") == strict_skill_n else "  <- other denominator"
            print(f"  {name:16s} {_f('result_set_f1_strict')} {row.get('n', 0):>6d} "
                  f"{_f('mean_trivial_floor')} {_f('mean_baseline_f1_noop')} "
                  f"{_f('skill_score')} {row.get('skill_n', 0):>8d} "
                  f"{_f('below_floor_rate', 7)}{mark}")
        strict = variants.get("strict", {}).get("result_set_f1_strict")
        primary = (block or {}).get("result_set_f1_strict")
        if strict == primary:
            print(f"  [OK] variants.strict {strict} vs result_set_f1_strict {primary}\n")
        elif all(row.get("skill_n", 0) == 0 for row in variants.values()):
            # Every skill undefined means no record carries a variants block at all: the run was
            # merged from shards written before the projections existed. That is a stale input,
            # not drift, and the fix is a rescore rather than an investigation.
            print(f"  [STALE] this run predates the variants -- rescore it with "
                  f"`--recompute` first; the numbers above are not meaningful\n")
        else:
            print(f"  [MISMATCH] variants.strict {strict} vs result_set_f1_strict {primary}"
                  f"  <-- the projection table has drifted from the metric\n")

    for stratum, block in (summary.get("by_stratum") or {}).items():
        _table(f"stratum: {stratum}", block)
    for model, block in ((summary.get("by_stratum") or {}).get("by_model") or {}).items():
        _table(f"model: {model}", block)
    for config_id, block in (summary.get("by_config") or {}).items():
        _table(f"config: {config_id}", (block or {}).get("metrics"))

    print("`floor` is the best of the degenerate strategies -- no-op baseline and flag-everything")
    print("-- and is what skill divides by. `(no-op)` is the no-op component alone, shown because")
    print("it is the figure notes §22-§26 quote. Where they differ, flag-everything is the higher")
    print("one and the published skill was too generous. See notes §27.\n")
    print("Reading the floors (they do NOT move together):")
    print("  population     the no-op baseline gains the flagged rows too, so its floor is ~0.9")
    print("                 and 102/135 discriminating entries reach exactly 1.0 -- skill is")
    print("                 undefined there, so its skill_n is a quarter of strict's.")
    print("  kind_projected the baseline binds gold's variables, so the projection is a no-op on")
    print("                 it. A gain here is genuine headroom the strict key was hiding.")
    print("  violation      the no-op floor is 0 by construction; flag-everything sets the real")
    print("                 one at ~0.49. Read against that, the model is well below trivial.")


def _recompute_metrics(run_dir, output_dir=None, store_dir=None,
                       truncation_policy: str = "as-measured"):
    """Rebuild every metric of a finished run from its stored result sets. No GPU, no ocpq_cli.

    The point of the result store: a matrix costs 40-100 GPU-hours, so a metric that turns out
    to be wrong -- as `combined_f1` did twice -- must be fixable without re-running it. Edit
    `_compute_similarity`, `_structural_metrics` or `_metric_bundle`, then run this.

    Everything downstream of the model is re-derived: the result similarity from the stored
    situation sets, the structural metric from the record's own queries, and the stratum labels
    from the discriminability audit. Everything upstream -- the generated query, timings, LLM
    stats -- is carried over untouched.

    Non-destructive by default. Overwriting a finished run's `predictions.json` in place is how
    the discriminability audit destroyed completed work three times; the default writes to a
    sibling directory instead.

    Args:
        run_dir: A finished run (or merged) directory containing `predictions.json`.
        output_dir: Where to write. Defaults to `<run_dir>_recomputed`.
        store_dir: Result store to read. Defaults to `<run_dir>/result_store`.
        truncation_policy: How to treat results the backend truncated. Defaults to
            `as-measured`, which changes nothing; see `_apply_truncation_policy`.
    """
    run_dir = Path(run_dir)
    predictions_path = run_dir / "predictions.json"
    if not predictions_path.exists():
        raise FileNotFoundError(f"No predictions.json in {run_dir}")

    store_dir = Path(store_dir) if store_dir else run_dir / "result_store"
    if not store_dir.is_dir():
        raise FileNotFoundError(
            f"No result store at {store_dir}. A run recorded before the store existed, or with "
            "`result_store: false`, can only be rescored by re-executing its queries."
        )

    output_dir = Path(output_dir) if output_dir else run_dir.parent / f"{run_dir.name}_recomputed"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_json(predictions_path)
    stratum_index = _load_stratum_index()

    unavailable = {"gold_missing": 0, "prediction_missing": 0}
    changed = 0
    for rec in records:
        ocel_path = rec.get("ocel_path")
        gold_query = rec.get("gold_query")
        prediction = rec.get("prediction")

        # `load_with_meta`, not `load`: the store carries the CLI's truncation flag alongside the
        # situations, and `--truncation-policy` must act on the stored truth rather than on a
        # record field carried over from whatever the original run believed.
        gold_ids, gold_skipped = (
            result_store.load_with_meta(store_dir, ocel_path, gold_query)
            if gold_query else (None, None)
        )
        # A prediction that never parsed has nothing to look up; that is a genuine null, not a
        # missing blob, so it must not be counted as unavailable.
        pred_ids, pred_skipped = (
            result_store.load_with_meta(store_dir, ocel_path, prediction)
            if prediction else (None, None)
        )

        before = json.dumps(rec.get("similarity"), sort_keys=True)
        if gold_ids is None:
            unavailable["gold_missing"] += 1
            rec["similarity"] = None
        else:
            if prediction and pred_ids is None and rec.get("similarity") is not None:
                # The run scored this record, so its blob should exist. Truncated or absent
                # means the recomputed number would be silently wrong -- report it, never guess.
                unavailable["prediction_missing"] += 1
            rec["similarity"] = _compute_similarity(gold_ids, pred_ids)
        if json.dumps(rec.get("similarity"), sort_keys=True) != before:
            changed += 1

        rec.setdefault("metrics", {})["structural"] = _structural_metrics(gold_query, prediction)
        # Refreshed from the store for the same reason it is read from there: a run recorded
        # before the flag was stored keeps whatever it had, and a stored None does not overwrite
        # a recorded True.
        if pred_skipped is not None:
            rec["metrics"]["bindings_skipped"] = pred_skipped
        if gold_skipped is not None:
            rec["metrics"]["gold_bindings_skipped"] = gold_skipped
        is_disc, baseline_f1, variant_floors, floor_detail = stratum_index.get(
            rec.get("nl_query"), _STRATUM_MISS)
        rec["discriminating"] = is_disc
        rec["baseline_f1"] = baseline_f1
        rec["baseline_f1_variants"] = variant_floors
        rec["baseline_f1_variants_detail"] = floor_detail
        rec["trivial_floor"] = (variant_floors.get("strict")
                                if variant_floors.get("strict") is not None else baseline_f1)
        # Derived last: they read the similarity and floors set just above, so a run recorded
        # before the skill score or the variants existed gains both here without re-executing
        # anything. This is the whole point of the store.
        rec["metrics"]["skill"] = _record_skill(rec)
        rec["metrics"]["skill_variants"] = _record_skill_variants(rec)

    truncation = _apply_truncation_policy(records, truncation_policy)

    repeats = 1
    manifest = _load_json_if_exists(run_dir / "config_resolved.json")
    if manifest:
        repeats = (manifest.get("config") or {}).get("repeats", 1)

    # Excluded records stay in predictions.json -- nothing is thrown away, it is only kept out of
    # the aggregation, so the exclusion can be inspected and reversed.
    scored_records = [r for r in records if not r.get("excluded_reason")]

    _write_json(output_dir / "predictions.json", records)
    summary = _build_summary(scored_records, repeats)
    summary["evaluation_phases"] = EVALUATION_PHASES
    summary["recomputed_from"] = str(run_dir)
    summary["recompute_unavailable"] = unavailable
    summary["truncation"] = truncation
    summary["result_store"] = dict(result_store.stats(store_dir), dir=str(store_dir))
    _write_json(output_dir / "summary.json", summary)
    _run_bb_analysis(output_dir)
    _run_divergence_analysis(output_dir)

    total_unavailable = unavailable["gold_missing"] + unavailable["prediction_missing"]
    print(
        f"Recomputed {len(records)} record(s) from {run_dir} into {output_dir}\n"
        f"  truncation policy: {truncation['truncation_policy']} "
        f"(zeroed {truncation['zeroed_prediction_truncated']}, "
        f"excluded {truncation['excluded_gold_truncated']})\n"
        f"  similarity changed for {changed} record(s)\n"
        f"  unavailable: {total_unavailable} "
        f"(gold {unavailable['gold_missing']}, prediction {unavailable['prediction_missing']})"
    )
    return output_dir


def _merge_shards(shard_dirs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    repeats = 1
    fingerprints = defaultdict(list)
    for shard_dir in shard_dirs:
        shard_path = Path(shard_dir)
        predictions_path = shard_path / "predictions.json"
        if not predictions_path.exists():
            print(f"Warning: no predictions.json in {shard_dir}, skipping.")
            continue
        shard_records = _read_json(predictions_path)
        all_records.extend(shard_records)
        cfg_path = shard_path / "config_resolved.json"
        if cfg_path.exists():
            manifest = _read_json(cfg_path)
            repeats = manifest.get("config", {}).get("repeats", repeats)
            fingerprints[manifest.get("pipeline_fingerprint")].append(shard_path.name)

    if not all_records:
        print("No records found across shards. Nothing to merge.")
        return

    # Shards that had to be re-run leave both the partial and the complete directory behind,
    # and `--merge-shards eval_runs/shard_*` picks up both. Without this, the same
    # (entry, condition, repeat) is averaged twice and combined_f1 silently shifts. Later
    # occurrences win, so a rerun supersedes the partial it replaced.
    deduped = {}
    collisions = []
    for rec in all_records:
        key = _record_key_of(rec)
        prior = deduped.get(key)
        if prior is not None and len(collisions) < 5:
            collisions.append((key, prior, rec))
        deduped[key] = rec

    dropped = len(all_records) - len(deduped)
    if dropped:
        rate = dropped / len(all_records)
        print(f"Dropped {dropped} duplicate record(s) across shards "
              f"({rate:.1%} of {len(all_records)}).")
        # A handful of duplicates is a re-run superseding a partial, which is the point of the
        # dedup. A large fraction is not: on 2026-08-27 this line read "Dropped 4032" -- half
        # the matrix -- because `case_id` restarts at 1 in every case-shard, so the two halves
        # of each condition collided entry-for-entry. It printed once, in the middle of other
        # output, and the run continued. It must never be that quiet again.
        if rate > 0.05:
            print("\n*** WARNING: an implausible share of records collided ***")
            print("    Duplicates should come only from a re-run superseding a partial shard.")
            print("    A large fraction means distinct records are producing the same key.")
            for key, prior, rec in collisions:
                differing = [
                    f for f in ("nl_query", "ocel_path", "case_id", "config_id", "repeat")
                    if prior.get(f) != rec.get(f)
                ]
                print(f"      key {key}")
                print(f"        kept    : case {rec.get('case_id')} | "
                      f"{str(rec.get('nl_query'))[:56]}")
                print(f"        dropped : case {prior.get('case_id')} | "
                      f"{str(prior.get('nl_query'))[:56]}")
                print(f"        differ in: {differing or 'nothing -- a genuine duplicate'}")
            print("    If they differ in nl_query, these are DIFFERENT test cases and the merge")
            print("    is discarding real data. Do not use the result.\n")
    all_records = list(deduped.values())

    # Shards launched in separate waves -- e.g. Llama and Ministral now, Gemma after a decision --
    # are only comparable if the pipeline was identical for all of them. Recorded in the summary
    # rather than merely printed, so a mixed merge cannot be lost from the record.
    summary_fingerprints = {fp: sorted(v) for fp, v in fingerprints.items()}
    if len(fingerprints) > 1:
        print("\n*** WARNING: shards were produced by DIFFERENT pipeline versions ***")
        for fp, shards in summary_fingerprints.items():
            print(f"  {str(fp)[:16]}: {len(shards)} shard(s) e.g. {shards[:3]}")
        print("  These results are not strictly comparable. Re-run the affected shards with a "
              "single pipeline version, or state the difference explicitly.\n")

    _write_json(output_dir / "predictions.json", all_records)
    summary = _build_summary(all_records, repeats)
    summary["evaluation_phases"] = EVALUATION_PHASES
    summary["pipeline_fingerprints"] = summary_fingerprints
    summary["pipeline_consistent"] = len(fingerprints) <= 1

    # Union every shard's result store into the merged directory, so the merged run can be
    # rescored with `--recompute` on its own. Safe without conflict resolution: the filename is
    # a content address, so an id present in two shards names the same (log, query) pair.
    store_sources = [Path(d) / "result_store" for d in shard_dirs]
    if any(p.is_dir() for p in store_sources):
        merged_store = output_dir / "result_store"
        copied = result_store.union_into(merged_store, store_sources)
        stats = result_store.stats(merged_store)
        summary["result_store"] = dict(stats, dir=str(merged_store))
        print(
            f"Merged result store: {stats['blobs']} blobs "
            f"({copied} copied, {stats['bytes'] / 1048576:.1f} MB)"
        )

    _write_json(output_dir / "summary.json", summary)
    _run_bb_analysis(output_dir)
    _run_divergence_analysis(output_dir)
    _merge_failure_logs(shard_dirs, output_dir)
    print(f"Merged {len(all_records)} records from {len(shard_dirs)} shard(s) into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLxOCPQ against a handcrafted testset.")
    parser.add_argument("--config", help="Path to evaluation config JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run counts and exit without running evaluations.")
    parser.add_argument("--resume-from", default="", help="Resume an interrupted run from an existing run directory or checkpoint.json.")
    parser.add_argument("--model-idx", type=int, default=None,
        help="0-based index into the expanded models list. Restricts this job to that model only (for Slurm array parallelism).")
    parser.add_argument("--strategy-idx", type=int, default=None,
        help="0-based index into prompt_strategies. Restricts this job to that strategy only (finer Slurm sharding).")
    parser.add_argument("--constrained-decoding", dest="constrained_decoding",
        action="store_true", default=None,
        help="Constrain generation to the OCPQ grammar so invalid output cannot be produced. "
             "This is NOT --schema-idx: schema_enforced only decides whether output is validated "
             "and repaired AFTERWARDS. Overrides the config key of the same name.")
    parser.add_argument("--no-constrained-decoding", dest="constrained_decoding",
        action="store_false",
        help="Force free generation even if the config enables constrained decoding.")
    parser.add_argument("--force-root-arity", dest="force_root_arity",
        action="store_true", default=None,
        help="ORACLE. Pin the root box's variable counts to GOLD's, by tightening the "
             "constrained-decoding grammar per case. Any run using this produces a CEILING and "
             "can never be reported as system performance. Requires --constrained-decoding.")
    parser.add_argument("--no-force-root-arity", dest="force_root_arity", action="store_false",
        help="Force free root arity even if the config enables it.")
    parser.add_argument("--token-budget", type=int, default=None,
        help="Override every pipeline's max_tokens (they hard-code 1000/2000). Truncation is a "
             "failure mode distinct from schema violation, so the budget has to be movable to "
             "tell them apart.")
    parser.add_argument("--schema-idx", type=int, default=None,
        help="0-based index into schema_enforced. Restricts this job to that schema setting only (finer Slurm sharding).")
    parser.add_argument("--case-shard", default=None, metavar="I/N",
        help="Evaluate only case slice I of N (0-based), e.g. 0/2. Splits the testset across jobs.")
    parser.add_argument("--merge-shards", nargs="+", metavar="DIR",
        help="Merge predictions.json files from multiple shard run-dirs into a combined summary.")
    parser.add_argument("--merge-output", metavar="DIR", default="eval_runs/merged",
        help="Output directory for merged results (default: eval_runs/merged).")
    parser.add_argument("--recompute", metavar="DIR",
        help="Rebuild all metrics of a finished run from its stored result sets. No GPU and no "
             "ocpq_cli: use this after changing a metric instead of re-running the matrix.")
    parser.add_argument("--recompute-output", metavar="DIR", default=None,
        help="Where --recompute writes (default: <DIR>_recomputed). The source run is never "
             "modified.")
    parser.add_argument("--result-store", metavar="DIR", default=None,
        help="Result store for --recompute (default: <DIR>/result_store).")
    parser.add_argument("--variants", metavar="DIR",
        help="Print the result-metric decomposition (strict / population / violation / "
             "kind_projected) for a finished run, each against its own floor. Reads summary.json; "
             "rescore with --recompute first if the run predates the variants.")
    parser.add_argument("--truncation-policy", choices=TRUNCATION_POLICIES, default="as-measured",
        help="How --recompute treats results ocpq_cli truncated ('Too much to handle!'). "
             "'as-measured' (default) compares the partial result as it stands, which flatters "
             "over-selecting queries. 'strict-zero' scores a truncated prediction 0 and excludes "
             "entries whose GOLD was truncated, for every model. Run both and report the "
             "difference; it is a sensitivity analysis, not a correction.")
    args = parser.parse_args()

    if args.merge_shards:
        _merge_shards(args.merge_shards, args.merge_output)
        return

    if args.recompute:
        _recompute_metrics(args.recompute, args.recompute_output, args.result_store,
                           truncation_policy=args.truncation_policy)
        return

    if args.variants:
        _report_variants(args.variants)
        return

    if not args.config:
        parser.error(
            "--config is required unless --merge-shards, --recompute or --variants is used.")

    def _apply_shard_filters(cfg):
        if args.model_idx is not None:
            cfg["_model_idx"] = args.model_idx
        if args.strategy_idx is not None:
            cfg["_strategy_idx"] = args.strategy_idx
        if args.schema_idx is not None:
            cfg["_schema_idx"] = args.schema_idx
        # Written into the config itself, not a `_`-prefixed shard key, so both land in the run
        # manifest and therefore in `config_fingerprint` -- a constrained shard and a free one
        # must not merge silently.
        if args.constrained_decoding is not None:
            cfg["constrained_decoding"] = args.constrained_decoding
        if args.token_budget is not None:
            cfg["generation_token_budget"] = args.token_budget
        if args.force_root_arity is not None:
            cfg["force_root_arity"] = args.force_root_arity
        if args.case_shard:
            try:
                index_str, total_str = str(args.case_shard).split("/")
                cfg["_case_shard"] = (int(index_str), int(total_str))
            except ValueError:
                parser.error(f"--case-shard must look like I/N (got {args.case_shard!r}).")

    if args.dry_run:
        config = _read_json(args.config)
        config["dry_run"] = True
        _apply_shard_filters(config)
        run_evaluation(config)
        print("Dry run complete.")
        return

    config = _read_json(args.config)
    _apply_shard_filters(config)
    if args.resume_from:
        config["resume_from"] = args.resume_from
    run_dir = run_evaluation(config)
    print(f"Evaluation complete. Outputs saved in {run_dir}")


if __name__ == "__main__":
    main()
