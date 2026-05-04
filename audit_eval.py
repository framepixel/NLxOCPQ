import argparse
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

from ocpq_schema import BindingBoxTree


UNSUPPORTED_FILTER_TYPES = {"ExactlyOnce"}
PROMPT_STRATEGIES = [
    "schema-grammar-guided",
    "schema-allowed-entities-only",
    "schema-few-shot",
    "schema-chain-of-thought-plus",
    "schema-least-to-most",
]
DEFAULT_ABLATION_STRATEGIES = [
    "schema-allowed-entities-only",
    "schema-few-shot",
    "schema-least-to-most",
]


def classify_pattern(text):
    text = (text or "").lower()
    tags = []
    if "exactly once" in text:
        tags.append("exactly_once")
    if "within" in text and "after" in text:
        tags.append("within_after")
    if "for any" in text:
        tags.append("for_any")
    if "same" in text:
        tags.append("same_object")
    if not tags:
        tags.append("other")
    return tags


def extract_query_entities(ocpq_json):
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


def contains_unsupported_filter(ocpq_json):
    blob = json.dumps(ocpq_json)
    return any(token in blob for token in UNSUPPORTED_FILTER_TYPES)


def select_eval_rows(dataset_path, size, seed, shuffle):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rows = list(dataset)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(rows)
    if size and size > 0:
        rows = rows[:size]
    return rows


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_value(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return str(value)


def _configure_backend(backend, model, hf_provider, hf_token, local_device, local_dtype):
    from LLM import set_hf_client, set_local_gpu_client

    if backend == "local-gpu":
        set_local_gpu_client(model_name=model, token=hf_token or None, device=local_device, dtype=local_dtype)
    else:
        set_hf_client(token=hf_token or None, provider=hf_provider or None, model_name=model)


def _compute_metrics(prediction, expected_query, allowed_events, allowed_objects, query_stdout, query_result_file):
    schema_valid = False
    if isinstance(prediction, dict):
        try:
            BindingBoxTree.model_validate(prediction)
            schema_valid = True
        except Exception:
            pass

    pred_events, pred_objects = extract_query_entities(prediction)
    allowed_events = set(allowed_events or [])
    allowed_objects = set(allowed_objects or [])
    hallucinated = bool(pred_events - allowed_events or pred_objects - allowed_objects)
    unsupported = contains_unsupported_filter(prediction)
    semantic_match = prediction == expected_query

    execution_success = False
    if query_result_file and os.path.exists(query_result_file):
        execution_success = True
    if isinstance(query_stdout, str) and query_stdout.lower().startswith("error running query automatically"):
        execution_success = False

    return {
        "schema_valid": schema_valid,
        "hallucinated_entities": hallucinated,
        "unsupported_filters": unsupported,
        "semantic_match": semantic_match,
        "execution_success": execution_success,
    }


def run_predictions(
    eval_rows,
    output_path,
    summary_path,
    ocel_path,
    backend,
    model,
    hf_provider,
    hf_token,
    prompt_strategy,
    few_shot_mode,
    local_device,
    local_dtype,
):
    from LLM import dataset as retrieval_dataset
    from LLM import embeddings
    from LLM import nl_to_ocpq_pipeline, set_few_shot_enabled

    _configure_backend(backend, model, hf_provider, hf_token, local_device, local_dtype)
    set_few_shot_enabled(few_shot_mode == "on")

    predictions = []
    for idx, row in enumerate(eval_rows, start=1):
        nl = row.get("nl", "")
        pattern_tags = row.get("pattern_tags") or classify_pattern(nl)
        allowed_events = row.get("events") or []
        allowed_objects = row.get("objects") or []
        expected_query = row.get("query")

        record = {
            "index": idx,
            "nl": nl,
            "pattern_tags": pattern_tags,
            "events": allowed_events,
            "objects": allowed_objects,
            "prompt_strategy": prompt_strategy,
            "few_shot": few_shot_mode,
        }

        start_time = time.time()
        try:
            structure, tree, ocpq_json, query_stdout, query_result_file_path = nl_to_ocpq_pipeline(
                query=nl,
                dataset=retrieval_dataset,
                embeddings=embeddings,
                output_file="test_query.json",
                ocel_path=ocel_path,
                model_name=model,
                prompt_strategy=prompt_strategy,
            )
            record.update(
                {
                    "status": "ok",
                    "prediction": ocpq_json,
                    "query_stdout": _normalize_value(query_stdout),
                    "query_result_file": _normalize_value(query_result_file_path),
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "prediction": None,
                    "query_stdout": None,
                    "query_result_file": None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )

        record["duration_sec"] = round(time.time() - start_time, 3)

        metrics = _compute_metrics(
            record.get("prediction"),
            expected_query,
            allowed_events,
            allowed_objects,
            record.get("query_stdout"),
            record.get("query_result_file"),
        )
        record.update(metrics)
        predictions.append(record)

    summary = summarize_predictions(predictions)
    _write_json(Path(output_path), predictions)
    _write_json(Path(summary_path), summary)
    return summary


def summarize_predictions(predictions):
    counters = Counter()
    by_pattern = defaultdict(Counter)

    for pred in predictions:
        counters["total"] += 1
        if pred.get("status") != "ok":
            counters["errors"] += 1

        for metric in [
            "schema_valid",
            "hallucinated_entities",
            "unsupported_filters",
            "semantic_match",
            "execution_success",
        ]:
            if pred.get(metric):
                counters[metric] += 1

        for tag in pred.get("pattern_tags") or []:
            by_pattern[tag]["total"] += 1
            if pred.get("status") != "ok":
                by_pattern[tag]["errors"] += 1
            for metric in [
                "schema_valid",
                "hallucinated_entities",
                "unsupported_filters",
                "semantic_match",
                "execution_success",
            ]:
                if pred.get(metric):
                    by_pattern[tag][metric] += 1

    total = max(counters["total"], 1)
    summary = {
        "total": counters["total"],
        "errors": counters["errors"],
        "schema_valid_rate": counters["schema_valid"] / total,
        "hallucinated_entity_rate": counters["hallucinated_entities"] / total,
        "unsupported_filter_rate": counters["unsupported_filters"] / total,
        "semantic_match_rate": counters["semantic_match"] / total,
        "execution_success_rate": counters["execution_success"] / total,
        "by_pattern": {},
    }

    for tag in sorted(by_pattern):
        tag_total = max(by_pattern[tag]["total"], 1)
        summary["by_pattern"][tag] = {
            "total": by_pattern[tag]["total"],
            "errors": by_pattern[tag]["errors"],
            "schema_valid_rate": by_pattern[tag]["schema_valid"] / tag_total,
            "hallucinated_entity_rate": by_pattern[tag]["hallucinated_entities"] / tag_total,
            "unsupported_filter_rate": by_pattern[tag]["unsupported_filters"] / tag_total,
            "semantic_match_rate": by_pattern[tag]["semantic_match"] / tag_total,
            "execution_success_rate": by_pattern[tag]["execution_success"] / tag_total,
        }

    return summary


def build_audit_set(dataset_path, output_path, size, seed):
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rng = random.Random(seed)
    buckets = defaultdict(list)
    for item in dataset:
        for tag in classify_pattern(item.get("nl")):
            buckets[tag].append(item)

    tags = sorted(buckets.keys())
    if not tags:
        raise ValueError("No patterns found in dataset.")

    per_tag = max(1, size // len(tags))
    selected = []
    seen = set()
    for tag in tags:
        candidates = buckets[tag][:]
        rng.shuffle(candidates)
        for item in candidates:
            key = item.get("nl")
            if key in seen:
                continue
            selected.append({"pattern_tags": classify_pattern(item.get("nl")), **item})
            seen.add(key)
            if len([row for row in selected if tag in row["pattern_tags"]]) >= per_tag:
                break

    remaining = [item for item in dataset if item.get("nl") not in seen]
    rng.shuffle(remaining)
    for item in remaining:
        if len(selected) >= size:
            break
        selected.append({"pattern_tags": classify_pattern(item.get("nl")), **item})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(selected[:size], f, indent=2)

    print(f"Wrote {min(size, len(selected))} audit examples to {output_path}")


def score_predictions(gold_path, predictions_path):
    with open(gold_path, "r", encoding="utf-8") as f:
        gold = json.load(f)
    with open(predictions_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gold_by_nl = {row["nl"]: row for row in gold}
    counters = Counter()
    by_pattern = defaultdict(Counter)

    for pred in predictions:
        nl = pred.get("nl")
        gold_row = gold_by_nl.get(nl)
        if not gold_row:
            continue

        counters["total"] += 1
        pattern_tags = pred.get("pattern_tags") or classify_pattern(nl)
        predicted_query = pred.get("prediction")
        expected_query = gold_row.get("query")
        allowed_events = set(gold_row.get("events") or list(extract_query_entities(expected_query)[0]))
        allowed_objects = set(gold_row.get("objects") or list(extract_query_entities(expected_query)[1]))

        schema_valid = False
        try:
            BindingBoxTree.model_validate(predicted_query)
            schema_valid = True
            counters["schema_valid"] += 1
        except Exception:
            pass

        pred_events, pred_objects = extract_query_entities(predicted_query)
        hallucinated = bool(pred_events - allowed_events or pred_objects - allowed_objects)
        unsupported = contains_unsupported_filter(predicted_query)
        semantic_match = predicted_query == expected_query

        if hallucinated:
            counters["hallucinated_entities"] += 1
        if unsupported:
            counters["unsupported_filters"] += 1
        if semantic_match:
            counters["semantic_match"] += 1

        for tag in pattern_tags:
            by_pattern[tag]["total"] += 1
            if schema_valid:
                by_pattern[tag]["schema_valid"] += 1
            if hallucinated:
                by_pattern[tag]["hallucinated_entities"] += 1
            if unsupported:
                by_pattern[tag]["unsupported_filters"] += 1
            if semantic_match:
                by_pattern[tag]["semantic_match"] += 1

    total = max(counters["total"], 1)
    print("Overall")
    print(f"  total: {counters['total']}")
    print(f"  schema validity: {counters['schema_valid'] / total:.3f}")
    print(f"  entity hallucination rate: {counters['hallucinated_entities'] / total:.3f}")
    print(f"  unsupported operator rate: {counters['unsupported_filters'] / total:.3f}")
    print(f"  semantic correctness proxy (exact JSON match): {counters['semantic_match'] / total:.3f}")

    print("\nBy pattern")
    for tag in sorted(by_pattern):
        tag_total = max(by_pattern[tag]["total"], 1)
        print(f"  {tag}:")
        print(f"    total: {by_pattern[tag]['total']}")
        print(f"    schema validity: {by_pattern[tag]['schema_valid'] / tag_total:.3f}")
        print(f"    entity hallucination rate: {by_pattern[tag]['hallucinated_entities'] / tag_total:.3f}")
        print(f"    unsupported operator rate: {by_pattern[tag]['unsupported_filters'] / tag_total:.3f}")
        print(f"    semantic correctness proxy: {by_pattern[tag]['semantic_match'] / tag_total:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Build, run, or score an OCPQ audit set.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--dataset", default="ocpq_dataset.json")
    build_parser.add_argument("--output", default="ocpq_audit_set.json")
    build_parser.add_argument("--size", type=int, default=75)
    build_parser.add_argument("--seed", type=int, default=3407)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--gold", default="ocpq_audit_set.json")
    score_parser.add_argument("--predictions", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--dataset", default="ocpq_audit_set.json")
    run_parser.add_argument("--output", default="ocpq_predictions.json")
    run_parser.add_argument("--summary", default="")
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--seed", type=int, default=3407)
    run_parser.add_argument("--shuffle", action="store_true")
    run_parser.add_argument("--ocel", required=True)
    run_parser.add_argument("--backend", choices=["local-gpu", "hf-provider"], required=True)
    run_parser.add_argument("--model", required=True)
    run_parser.add_argument("--hf-provider", default="")
    run_parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    run_parser.add_argument("--prompt-strategy", choices=PROMPT_STRATEGIES, default="schema-grammar-guided")
    run_parser.add_argument("--few-shot", choices=["on", "off"], default="on")
    run_parser.add_argument("--local-device", default="cuda")
    run_parser.add_argument("--local-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")

    ablate_parser = subparsers.add_parser("ablate")
    ablate_parser.add_argument("--dataset", default="ocpq_audit_set.json")
    ablate_parser.add_argument("--output-dir", default="ablation_runs")
    ablate_parser.add_argument("--strategies", default=",".join(DEFAULT_ABLATION_STRATEGIES))
    ablate_parser.add_argument("--few-shot", choices=["on", "off", "both"], default="both")
    ablate_parser.add_argument("--limit", type=int, default=0)
    ablate_parser.add_argument("--seed", type=int, default=3407)
    ablate_parser.add_argument("--shuffle", action="store_true")
    ablate_parser.add_argument("--ocel", required=True)
    ablate_parser.add_argument("--backend", choices=["local-gpu", "hf-provider"], required=True)
    ablate_parser.add_argument("--model", required=True)
    ablate_parser.add_argument("--hf-provider", default="")
    ablate_parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN", ""))
    ablate_parser.add_argument("--local-device", default="cuda")
    ablate_parser.add_argument("--local-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")

    args = parser.parse_args()
    if args.command == "build":
        build_audit_set(args.dataset, args.output, args.size, args.seed)
        return

    if args.command == "score":
        score_predictions(args.gold, args.predictions)
        return

    if args.command == "run":
        eval_rows = select_eval_rows(args.dataset, args.limit, args.seed, args.shuffle)
        output_path = Path(args.output)
        summary_path = Path(args.summary) if args.summary else output_path.with_name(output_path.stem + "_summary.json")
        summary = run_predictions(
            eval_rows=eval_rows,
            output_path=output_path,
            summary_path=summary_path,
            ocel_path=args.ocel,
            backend=args.backend,
            model=args.model,
            hf_provider=args.hf_provider,
            hf_token=args.hf_token,
            prompt_strategy=args.prompt_strategy,
            few_shot_mode=args.few_shot,
            local_device=args.local_device,
            local_dtype=args.local_dtype,
        )
        print(f"Wrote predictions to {output_path}")
        print(f"Wrote summary to {summary_path}")
        print(json.dumps(summary, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = select_eval_rows(args.dataset, args.limit, args.seed, args.shuffle)

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    invalid = [s for s in strategies if s not in PROMPT_STRATEGIES]
    if invalid:
        raise ValueError(f"Unknown strategies: {invalid}")

    few_shot_modes = ["on", "off"] if args.few_shot == "both" else [args.few_shot]
    ablation_summary = {"runs": []}

    for strategy in strategies:
        for few_shot_mode in few_shot_modes:
            name = f"{strategy}_fewshot-{few_shot_mode}"
            predictions_path = output_dir / f"predictions_{name}.json"
            summary_path = output_dir / f"summary_{name}.json"
            summary = run_predictions(
                eval_rows=eval_rows,
                output_path=predictions_path,
                summary_path=summary_path,
                ocel_path=args.ocel,
                backend=args.backend,
                model=args.model,
                hf_provider=args.hf_provider,
                hf_token=args.hf_token,
                prompt_strategy=strategy,
                few_shot_mode=few_shot_mode,
                local_device=args.local_device,
                local_dtype=args.local_dtype,
            )
            ablation_summary["runs"].append(
                {
                    "strategy": strategy,
                    "few_shot": few_shot_mode,
                    "predictions": str(predictions_path),
                    "summary": str(summary_path),
                    "metrics": summary,
                }
            )

    ablation_path = output_dir / "ablation_summary.json"
    _write_json(ablation_path, ablation_summary)
    print(f"Wrote ablation summary to {ablation_path}")


if __name__ == "__main__":
    main()
