import json
import os


SYSTEM_PROMPT = (
    "You are a JSON-only API that converts natural language process constraints into OCPQ JSON. "
    "Return raw valid JSON only. Follow the provided OCPQ schema exactly. "
    "Never invent entities, events, objects, node names, or unsupported filter types. "
    "Use only schema-supported operators and structures."
)


def _collect_query_entities(ocpq_json):
    used_events = []
    used_objects = []

    if not isinstance(ocpq_json, dict):
        return used_events, used_objects

    for node in ocpq_json.get("nodes", []):
        if not isinstance(node, dict):
            continue
        box = node.get("Box", [])
        if not isinstance(box, list) or not box:
            continue

        node_data = box[0]
        if not isinstance(node_data, dict):
            continue

        for values in node_data.get("newEventVars", {}).values():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value not in used_events:
                        used_events.append(value)

        for values in node_data.get("newObjectVars", {}).values():
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value not in used_objects:
                        used_objects.append(value)

    return used_events, used_objects


def _resolve_allowed_entities(item):
    allowed_events = item.get("events") or []
    allowed_objects = item.get("objects") or []

    if not allowed_events or not allowed_objects:
        query_events, query_objects = _collect_query_entities(item.get("query"))
        if not allowed_events:
            allowed_events = query_events
        if not allowed_objects:
            allowed_objects = query_objects

    return allowed_events, allowed_objects


def _make_user_prompt(query_nl, allowed_events, allowed_objects, include_reasoning=False,
                      object_attributes=None, relations=None):
    prompt = f"""Convert this natural language query to an OCPQ JSON graph.

ALLOWED EVENTS: {json.dumps(allowed_events, ensure_ascii=True)}
ALLOWED OBJECTS: {json.dumps(allowed_objects, ensure_ascii=True)}"""

    if object_attributes:
        lines = []
        for obj_type, attrs in object_attributes.items():
            parts = []
            for a in attrs:
                part = f"{a['name']} ({a['type']}"
                if a.get("values"):
                    part += f": {', '.join(str(v) for v in a['values'])}"
                part += ")"
                parts.append(part)
            lines.append(f"  {obj_type}: {', '.join(parts)}")
        prompt += "\nOBJECT ATTRIBUTES:\n" + "\n".join(lines)

    # Relation graph, byte-identical to LLM.build_schema_block at the "full" level so the
    # fine-tuned model sees at inference exactly the format it was trained on.
    if relations:
        def _render(triples):
            out = []
            for t in triples:
                source, target = t[0], t[1]
                qualifier = t[2] if len(t) > 2 else None
                suffix = f" ({qualifier})" if qualifier else ""
                out.append(f"  {source} -> {target}{suffix}")
            return out

        e2o_lines = _render(relations.get("e2o", []))
        if e2o_lines:
            prompt += ("\nEVENT-TO-OBJECT RELATIONS (only these event/object pairs are linked):\n"
                       + "\n".join(e2o_lines))
        o2o_lines = _render(relations.get("o2o", []))
        if o2o_lines:
            prompt += ("\nOBJECT-TO-OBJECT RELATIONS (only these object pairs are linked):\n"
                       + "\n".join(o2o_lines))
        if e2o_lines or o2o_lines:
            prompt += ("\nUse ONLY the relations listed above. If two entities are not directly "
                       "related, connect them through an intermediary that is.")

    prompt += f"""

Rules:
- Output ONLY raw JSON matching the OCPQ schema.
- The root object MUST contain both "nodes" and "edgeNames".
- "edgeNames" is an array of [[parent_index, child_index], "label"] pairs, one per parent-child
  link used in "nodes" (e.g. [[0, 1], "A"]). Do NOT omit it.
- Basic filters ("TimeBetweenEvents", "NotEqual", "EventAttributeValueFilter",
  "ObjectAttributeValueFilter", "BasicFilterCEL") must be wrapped as
  {{"type": "Filter", "filter": {{...}}}} when placed in "constraints".
- "O2E" and "O2O" belong in the node's "filters" array, never directly in "constraints".
- Use ONLY event and object names from the allowed lists.
- Never invent unsupported filter types such as "ExactlyOnce".
- Represent "exactly once" using the schema's supported structural encoding, not a custom operator.
- If a node contains both events and objects, include the required `O2E` filter(s).
- Keep all nodes connected in a single tree and avoid empty nodes.

Query: {query_nl}"""

    if include_reasoning:
        prompt += """

Think in this order before emitting JSON:
1. Map colloquial phrases to the allowed events and objects.
2. Choose the correct OCPQ pattern (for example, exactly once, within-after, same object).
3. Emit the final JSON only."""

    return prompt


def _make_training_rows(item):
    query_nl = item["nl"]
    ocpq_json = item["query"]
    object_attributes = item.get("object_attributes")
    relations = item.get("relations")
    allowed_events, allowed_objects = _resolve_allowed_entities(item)
    assistant_response = json.dumps(ocpq_json, ensure_ascii=True, indent=2)

    rows = []
    base_prompt = _make_user_prompt(query_nl, allowed_events, allowed_objects,
                                    include_reasoning=False, object_attributes=object_attributes,
                                    relations=relations)
    rows.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": base_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
    )

    reasoning_prompt = _make_user_prompt(query_nl, allowed_events, allowed_objects,
                                         include_reasoning=True, object_attributes=object_attributes,
                                         relations=relations)
    rows.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": reasoning_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
    )

    negative_prompt = (
        _make_user_prompt(query_nl, allowed_events, allowed_objects,
                          include_reasoning=False, object_attributes=object_attributes,
                          relations=relations)
        + f"""

Bad output patterns to avoid for this query:
- Unsupported filters like "ExactlyOnce"
- Hallucinated entities outside the allowed lists
- Disconnected nodes or empty nodes
- Missing O2E filters when both an event and an object appear in the same node

The correct answer must avoid those mistakes and still represent this query exactly."""
    )
    rows.append(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": negative_prompt},
                {"role": "assistant", "content": assistant_response},
            ]
        }
    )

    # Preserve nearby rewrites as extra supervised rows when available.
    for rewrite_key in ("nl_clean_1", "nl_clean_2"):
        rewrite_text = item.get(rewrite_key)
        if isinstance(rewrite_text, str) and rewrite_text.strip() and rewrite_text.strip() != query_nl.strip():
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _make_user_prompt(rewrite_text, allowed_events, allowed_objects,
                                                         include_reasoning=False,
                                                         object_attributes=object_attributes,
                                                         relations=relations),
                        },
                        {"role": "assistant", "content": assistant_response},
                    ]
                }
            )

    return rows

def convert_to_jsonl(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        exported_rows = 0
        missing_entity_rows = 0

        for item in dataset:
            if "nl" not in item or "query" not in item:
                continue

            allowed_events, allowed_objects = _resolve_allowed_entities(item)
            if not allowed_events or not allowed_objects:
                missing_entity_rows += 1

            for message_format in _make_training_rows(item):
                f.write(json.dumps(message_format, ensure_ascii=True) + "\n")
                exported_rows += 1

    print(
        f"Successfully converted {len(dataset)} source examples to {output_file} "
        f"with {exported_rows} training rows."
    )
    if missing_entity_rows:
        print(
            f"Warning: {missing_entity_rows} source examples still had incomplete allowed entity sets "
            "after fallback extraction."
        )

if __name__ == "__main__":
    input_dataset = "ocpq_dataset.json"
    output_jsonl = "finetune_ocpq.jsonl"
    convert_to_jsonl(input_dataset, output_jsonl)
