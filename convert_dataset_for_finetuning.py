import json
import os

def convert_to_jsonl(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            if "nl" not in item or "query" not in item:
                continue

            query_nl = item["nl"]
            reasoning = item.get("reasoning", "")
            ocpq_json = item["query"]
            mapped_events = item.get("events", [])
            mapped_objects = item.get("objects", [])

            # Construct the conversational format matching the LLM.py prompts.
            # For a direct fine-tuning, you might just map NL -> JSON.
            # But incorporating the reasoning step can be useful if doing Chain-of-Thought
            system_prompt = "You are a JSON-only API. You must output raw, valid JSON perfectly formatting according to the provided OCPQ schema."
            user_prompt = f"""Convert this natural language query to an OCPQ JSON graph.
ALLOWED EVENTS: {mapped_events}
ALLOWED OBJECTS: {mapped_objects}

Query: {query_nl}"""

            
            # Incorporate reasoning if we want CoT (optional, but good for LoRA)
            assistant_response = json.dumps(ocpq_json, indent=2)

            message_format = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ]
            }

            f.write(json.dumps(message_format) + "\n")

    print(f"Successfully converted {len(dataset)} examples to {output_file} for fine-tuning.")

if __name__ == "__main__":
    input_dataset = "ocpq_dataset.json"
    output_jsonl = "finetune_ocpq.jsonl"
    convert_to_jsonl(input_dataset, output_jsonl)
