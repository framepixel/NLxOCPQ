#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import os
import random
import re
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from ocpq_schema import BindingBoxTree
from QueryExecutor import run_ocpq_query
try:
    from expand_models import expand_models
except Exception:
    expand_models = None
from LLM import (
    dataset as retrieval_dataset,
    embeddings,
    nl_to_ocpq_pipeline,
    set_few_shot_enabled,
    set_generation_temperature,
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
    "schema-grammar-guided",
    "schema-allowed-entities-only",
    "schema-few-shot",
    "schema-chain-of-thought-plus",
    "schema-least-to-most",
]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(value):
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
    return value.strip("_") or "item"


def _extract_query_entities(ocpq_json):
    events = set()
    objects = set()
    if not isinstance(ocpq_json, dict):
        return events, objects

    for node in ocpq_json.get("nodes", []):
        box = node.get("Box", [])
        if not isinstance(box, list) or not box:
            continue
        node_data = box[0]
        if not isinstance(node_data, dict):
            continue
        for values in node_data.get("newEventVars", {}).values():
            if isinstance(values, list):
                events.update(v for v in values if isinstance(v, str))
        for values in node_data.get("newObjectVars", {}).values():
            if isinstance(values, list):
                objects.update(v for v in values if isinstance(v, str))
    return events, objects


def _contains_unsupported_filter(ocpq_json):
    try:
        blob = json.dumps(ocpq_json)
    except Exception:
        return False
    return any(token in blob for token in UNSUPPORTED_FILTER_TYPES)


def _extract_result_ids(result_data):
    event_ids = set()
    object_ids = set()
    if not isinstance(result_data, dict):
        return event_ids, object_ids

    ok_data = result_data.get("Ok") if isinstance(result_data, dict) else None
    if not isinstance(ok_data, dict):
        return event_ids, object_ids

    eval_data = ok_data.get("evaluationResults", [])
    if isinstance(eval_data, list):
        for node in eval_data:
            situations = node.get("situations", []) if isinstance(node, dict) else []
            if not isinstance(situations, list):
                continue
            for situation in situations:
                if not isinstance(situation, list) or len(situation) != 2:
                    continue
                binding = situation[0]
                if not isinstance(binding, dict):
                    continue
                for event_id in binding.get("eventMap", []):
                    event_ids.add(str(event_id))
                for object_id in binding.get("objectMap", []):
                    object_ids.add(str(object_id))

    if not event_ids:
        for event_id in ok_data.get("eventIds", []) or []:
            event_ids.add(str(event_id))
    if not object_ids:
        for object_id in ok_data.get("objectIds", []) or []:
            object_ids.add(str(object_id))

    return event_ids, object_ids


def _load_result_ids(result_path):
    if not result_path or not os.path.exists(result_path):
        return None
    try:
        data = _read_json(result_path)
    except Exception:
        return None
    return _extract_result_ids(data)


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


def _compute_similarity(gold_ids, pred_ids):
    if gold_ids is None or pred_ids is None:
        return None
    gold_events, gold_objects = gold_ids
    pred_events, pred_objects = pred_ids
    event_metrics = _set_metrics(gold_events, pred_events)
    object_metrics = _set_metrics(gold_objects, pred_objects)
    combined_metrics = _set_metrics(gold_events | gold_objects, pred_events | pred_objects)
    return {
        "events": event_metrics,
        "objects": object_metrics,
        "combined": combined_metrics,
    }


def _summarize_llm_stats(stats):
    total_prompt = 0
    total_completion = 0
    total_tokens = 0
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

    return {
        "calls": len(stats),
        "duration_sec": round(total_duration, 4),
        "prompt_tokens": total_prompt if has_token_data else None,
        "completion_tokens": total_completion if has_token_data else None,
        "total_tokens": total_tokens if has_token_data else None,
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

    if backend == "local-gpu":
        set_local_gpu_client(
            model_name=model_cfg["name"],
            token=hf_token or None,
            device=local_device,
            dtype=local_dtype,
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
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if limit and limit > 0:
        rows = rows[:limit]
    return rows


def _hash_query_key(ocel_path, ocpq_query):
    payload = json.dumps(ocpq_query, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{ocel_path}::{digest}"


def run_evaluation(config_path):
    config = _read_json(config_path)
    if config.get("auto_expand_finetuned"):
        if expand_models is None:
            raise RuntimeError(
                "auto_expand_finetuned is enabled, but expand_models.py could not be imported."
            )
        config = expand_models(config)

    testset_path = config.get("testset_path", "testset.json")
    output_dir = Path(config.get("output_dir", "eval_runs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    testset_abs = Path(testset_path)
    if not testset_abs.exists():
        raise FileNotFoundError(f"Testset not found: {testset_abs}")

    run_id = time.strftime("eval_run_%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    global_cfg = {
        "backend": config.get("backend", "local-gpu"),
        "hf_provider": config.get("hf_provider", ""),
        "hf_token": config.get("hf_token", os.getenv("HF_TOKEN", "")),
        "local_device": config.get("local_device", "cuda"),
        "local_dtype": config.get("local_dtype", "auto"),
        "temperature": config.get("temperature", 0.0),
    }

    models = config.get("models") or []
    if not models:
        raise ValueError("Config must include a non-empty 'models' list.")

    prompt_strategies = config.get("prompt_strategies") or DEFAULT_PROMPT_STRATEGIES
    schema_enforced_values = config.get("schema_enforced", [True])
    schema_validation = bool(config.get("schema_validation", True))
    schema_validation_when_no_schema = bool(config.get("schema_validation_when_no_schema", False))
    repeats = int(config.get("repeats", 1))
    limit = int(config.get("limit", 0))
    seed = int(config.get("seed", 3407))
    shuffle = bool(config.get("shuffle", False))

    testset = _load_testset(testset_path, limit, seed, shuffle)
    testset_hash = _sha256_file(testset_abs)

    manifest = {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "testset_path": str(testset_abs),
        "testset_hash": testset_hash,
        "records": len(testset),
        "prompt_strategies": prompt_strategies,
        "schema_enforced": schema_enforced_values,
        "schema_validation": schema_validation,
        "schema_validation_when_no_schema": schema_validation_when_no_schema,
        "repeats": repeats,
        "config": config,
    }
    _write_json(run_dir / "config_resolved.json", manifest)

    ground_truth_cache = {}
    records = []

    for case_idx, row in enumerate(testset, start=1):
        nl_query = row.get("nl_query", "")
        ocel_path = row.get("ocel_path")
        gold_query = row.get("ocpq_query")
        if not ocel_path or not gold_query:
            continue

        case_key = _hash_query_key(ocel_path, gold_query)
        if case_key not in ground_truth_cache:
            gt_dir = run_dir / "ground_truth"
            gt_dir.mkdir(parents=True, exist_ok=True)
            gt_query_path = gt_dir / f"ground_truth_{case_idx}.json"
            _write_json(gt_query_path, gold_query)

            gt_start = time.time()
            gt_stdout, gt_stderr, gt_result_path = run_ocpq_query(ocel_path, str(gt_query_path))
            gt_duration = time.time() - gt_start
            gt_ids = _load_result_ids(gt_result_path)

            ground_truth_cache[case_key] = {
                "result_path": gt_result_path,
                "duration_sec": round(gt_duration, 4),
                "stdout": gt_stdout,
                "stderr": gt_stderr,
                "ids": gt_ids,
            }

        gt_record = ground_truth_cache[case_key]

        for model_cfg in models:
            resolved_model = _resolve_model_config(model_cfg, global_cfg)
            _configure_backend(resolved_model, global_cfg)
            set_few_shot_enabled(True)
            set_generation_temperature(resolved_model.get("temperature", global_cfg.get("temperature")))

            for prompt_strategy in prompt_strategies:
                for schema_enforced in schema_enforced_values:
                    effective_schema_validation = schema_validation
                    if not schema_enforced and not schema_validation_when_no_schema:
                        effective_schema_validation = False

                    config_id = _slug(
                        f"{resolved_model['label']}_{prompt_strategy}_schema-{'on' if schema_enforced else 'off'}"
                    )

                    for repeat_idx in range(1, repeats + 1):
                        reset_llm_call_stats()
                        reset_query_exec_stats()

                        output_dir_runs = run_dir / "generated" / f"case_{case_idx:03d}"
                        output_dir_runs.mkdir(parents=True, exist_ok=True)
                        output_file = output_dir_runs / f"{config_id}_rep{repeat_idx}.json"

                        start_time = time.time()
                        status = "ok"
                        error = None
                        trace = None
                        ocpq_json = None
                        query_stdout = None
                        query_result_file = None

                        try:
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
                            )
                        except Exception as exc:
                            status = "error"
                            error = str(exc)
                            trace = traceback.format_exc()

                        total_duration = time.time() - start_time
                        llm_stats = _summarize_llm_stats(get_llm_call_stats())
                        exec_stats = _summarize_exec_stats(get_query_exec_stats())

                        pred_ids = _load_result_ids(query_result_file)
                        similarity = _compute_similarity(gt_record.get("ids"), pred_ids)

                        schema_valid = False
                        if isinstance(ocpq_json, dict):
                            try:
                                BindingBoxTree.model_validate(ocpq_json)
                                schema_valid = True
                            except Exception:
                                schema_valid = False

                        pred_events, pred_objects = _extract_query_entities(ocpq_json)
                        gold_events, gold_objects = _extract_query_entities(gold_query)
                        hallucinated = bool(pred_events - gold_events or pred_objects - gold_objects)
                        unsupported = _contains_unsupported_filter(ocpq_json)
                        semantic_match = ocpq_json == gold_query

                        record = {
                            "case_id": case_idx,
                            "nl_query": nl_query,
                            "ocel_path": ocel_path,
                            "gold_query": gold_query,
                            "model": resolved_model["name"],
                            "model_label": resolved_model["label"],
                            "backend": resolved_model.get("backend", global_cfg.get("backend")),
                            "backend_description": get_hf_backend_description(),
                            "prompt_strategy": prompt_strategy,
                            "schema_enforced": schema_enforced,
                            "schema_validation": effective_schema_validation,
                            "repeat": repeat_idx,
                            "config_id": config_id,
                            "status": status,
                            "error": error,
                            "traceback": trace,
                            "prediction": ocpq_json,
                            "query_stdout": query_stdout,
                            "query_result_file": query_result_file,
                            "llm": llm_stats,
                            "execution": exec_stats,
                            "total_duration_sec": round(total_duration, 4),
                            "metrics": {
                                "schema_valid": schema_valid,
                                "hallucinated_entities": hallucinated,
                                "unsupported_filters": unsupported,
                                "semantic_match": semantic_match,
                            },
                            "similarity": similarity,
                            "ground_truth": {
                                "result_path": gt_record.get("result_path"),
                                "duration_sec": gt_record.get("duration_sec"),
                                "stdout": gt_record.get("stdout"),
                                "stderr": gt_record.get("stderr"),
                            },
                        }

                        records.append(record)

    _write_json(run_dir / "predictions.json", records)

    summary = _build_summary(records, repeats)
    _write_json(run_dir / "summary.json", summary)

    return run_dir


def _build_summary(records, repeats):
    by_config = defaultdict(list)
    for rec in records:
        by_config[rec["config_id"]].append(rec)

    consistency_by_config = defaultdict(list)
    by_case_config = defaultdict(list)
    for rec in records:
        by_case_config[(rec["case_id"], rec["config_id"])].append(rec)

    for (case_id, config_id), recs in by_case_config.items():
        json_keys = []
        f1_values = []
        exec_success = []
        for rec in recs:
            pred = rec.get("prediction")
            if pred is None:
                json_keys.append("null")
            else:
                json_keys.append(json.dumps(pred, sort_keys=True, separators=(",", ":")))
            combined = (rec.get("similarity") or {}).get("combined")
            f1_values.append(combined.get("f1") if combined else None)
            exec_success.append(bool((rec.get("execution") or {}).get("success")))

        counts = Counter(json_keys)
        mode_rate = max(counts.values()) / max(len(json_keys), 1)
        consistency_by_config[config_id].append(
            {
                "case_id": case_id,
                "json_mode_rate": mode_rate,
                "result_f1_std": _std([v for v in f1_values if v is not None]),
                "execution_success_rate": sum(exec_success) / max(len(exec_success), 1),
            }
        )

    summary = {
        "total_records": len(records),
        "by_config": {},
    }

    for config_id, recs in by_config.items():
        schema_valid_rate = _mean([1 if r["metrics"]["schema_valid"] else 0 for r in recs])
        semantic_match_rate = _mean([1 if r["metrics"]["semantic_match"] else 0 for r in recs])
        hallucinated_rate = _mean([1 if r["metrics"]["hallucinated_entities"] else 0 for r in recs])
        unsupported_rate = _mean([1 if r["metrics"]["unsupported_filters"] else 0 for r in recs])
        exec_success_rate = _mean([1 if (r.get("execution") or {}).get("success") else 0 for r in recs])

        prompt_tokens = _mean([r["llm"].get("prompt_tokens") for r in recs])
        completion_tokens = _mean([r["llm"].get("completion_tokens") for r in recs])
        total_tokens = _mean([r["llm"].get("total_tokens") for r in recs])
        llm_duration = _mean([r["llm"].get("duration_sec") for r in recs])
        exec_duration = _mean([r["execution"].get("duration_sec") for r in recs])
        total_duration = _mean([r.get("total_duration_sec") for r in recs])

        event_jaccard = _mean([
            (r.get("similarity") or {}).get("events", {}).get("jaccard")
            for r in recs
            if r.get("similarity")
        ])
        object_jaccard = _mean([
            (r.get("similarity") or {}).get("objects", {}).get("jaccard")
            for r in recs
            if r.get("similarity")
        ])
        combined_f1 = _mean([
            (r.get("similarity") or {}).get("combined", {}).get("f1")
            for r in recs
            if r.get("similarity")
        ])

        consistency = consistency_by_config.get(config_id, [])
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
                "semantic_match_rate": semantic_match_rate,
                "hallucinated_entity_rate": hallucinated_rate,
                "unsupported_filter_rate": unsupported_rate,
                "execution_success_rate": exec_success_rate,
                "event_jaccard": event_jaccard,
                "object_jaccard": object_jaccard,
                "combined_f1": combined_f1,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "llm_duration_sec": llm_duration,
                "exec_duration_sec": exec_duration,
                "total_duration_sec": total_duration,
            },
            "consistency": {
                "json_mode_rate": _mean([c["json_mode_rate"] for c in consistency]) if consistency else None,
                "result_f1_std": _mean([c["result_f1_std"] for c in consistency]) if consistency else None,
                "execution_success_rate": _mean([c["execution_success_rate"] for c in consistency]) if consistency else None,
            },
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate NLxOCPQ against a handcrafted testset.")
    parser.add_argument("--config", required=True, help="Path to evaluation config JSON.")
    args = parser.parse_args()

    run_dir = run_evaluation(args.config)
    print(f"Evaluation complete. Outputs saved in {run_dir}")


if __name__ == "__main__":
    main()
