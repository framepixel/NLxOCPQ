import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from ocpq_schema import BindingBoxTree


UNSUPPORTED_FILTER_TYPES = {"ExactlyOnce"}


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
    parser = argparse.ArgumentParser(description="Build or score an OCPQ audit set.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--dataset", default="ocpq_dataset.json")
    build_parser.add_argument("--output", default="ocpq_audit_set.json")
    build_parser.add_argument("--size", type=int, default=75)
    build_parser.add_argument("--seed", type=int, default=3407)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--gold", default="ocpq_audit_set.json")
    score_parser.add_argument("--predictions", required=True)

    args = parser.parse_args()
    if args.command == "build":
        build_audit_set(args.dataset, args.output, args.size, args.seed)
    else:
        score_predictions(args.gold, args.predictions)


if __name__ == "__main__":
    main()
