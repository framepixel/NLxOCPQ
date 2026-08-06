import json
import sys
from pathlib import Path


def _extract_allowed_list(prompt, label):
    marker = f"{label}: "
    for line in prompt.splitlines():
        if line.startswith(marker):
            payload = line[len(marker):].strip()
            try:
                value = json.loads(payload)
            except json.JSONDecodeError:
                return None
            return value if isinstance(value, list) else None
    return None


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_source_dataset(path):
    dataset = _load_json(path)
    total = len(dataset)
    with_events = 0
    with_objects = 0
    exactly_once = 0
    within_after = 0

    for item in dataset:
        if item.get("events"):
            with_events += 1
        if item.get("objects"):
            with_objects += 1

        nl = item.get("nl", "").lower()
        if "exactly once" in nl:
            exactly_once += 1
        if "within" in nl and "after" in nl:
            within_after += 1

    print(f"Source dataset: {path}")
    print(f"  rows: {total}")
    print(f"  rows with events: {with_events}/{total}")
    print(f"  rows with objects: {with_objects}/{total}")
    print(f"  exactly once rows: {exactly_once}")
    print(f"  within-after rows: {within_after}")


def audit_finetune_jsonl(path):
    total = 0
    missing_events = 0
    missing_objects = 0
    bad_system = 0
    negative_rows = 0

    decoder = json.JSONDecoder()
    raw_text = Path(path).read_text(encoding="utf-8")
    index = 0
    text_length = len(raw_text)

    while index < text_length:
        while index < text_length and raw_text[index].isspace():
            index += 1
        if index >= text_length:
            break

        row, next_index = decoder.raw_decode(raw_text, index)
        index = next_index
        total += 1
        messages = row.get("messages", [])
        if len(messages) < 3:
            bad_system += 1
            continue

        # Match the phrase actually used by convert_dataset_for_finetuning.SYSTEM_PROMPT
        # ("Never invent entities, events, objects, node names, or unsupported filter
        # types."). The previous literal assumed a wording the prompt has never used, so
        # this check reported every row as missing the tightened system prompt.
        system_content = messages[0].get("content", "")
        if "Never invent" not in system_content or "unsupported filter types" not in system_content:
            bad_system += 1

        prompt = messages[1].get("content", "")
        allowed_events = _extract_allowed_list(prompt, "ALLOWED EVENTS")
        allowed_objects = _extract_allowed_list(prompt, "ALLOWED OBJECTS")
        if not allowed_events:
            missing_events += 1
        if not allowed_objects:
            missing_objects += 1
        if "Bad output patterns to avoid" in prompt:
            negative_rows += 1

    print(f"Fine-tune JSONL: {path}")
    print(f"  rows: {total}")
    print(f"  rows missing allowed events: {missing_events}")
    print(f"  rows missing allowed objects: {missing_objects}")
    print(f"  rows with contrastive prompt section: {negative_rows}")
    print(f"  rows missing tightened system prompt: {bad_system}")


def main():
    source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ocpq_dataset.json")
    finetune_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("finetune_ocpq.jsonl")

    if source_path.exists():
        audit_source_dataset(source_path)
    else:
        print(f"Missing source dataset: {source_path}")

    if finetune_path.exists():
        audit_finetune_jsonl(finetune_path)
    else:
        print(f"Missing fine-tune JSONL: {finetune_path}")


if __name__ == "__main__":
    main()
