import json
import xml.etree.ElementTree as ET
from pathlib import Path

testset = json.load(open("data/testset.json", "r"))
print(f"Total questions in testset: {len(testset)}")
ocel_paths = [q["ocel_path"] for q in testset]

path_occurrences = {}
for i, q in enumerate(testset):
    if q["ocel_path"] not in path_occurrences:
        path_occurrences[q["ocel_path"]] = 0
    path_occurrences[q["ocel_path"]] += 1

def local_tag(element):
    # Handles XML namespaces, such as {namespace}events
    return element.tag.rsplit("}", 1)[-1]

def xml_count(root, container_name):
    # Find the first matching container and count its direct child elements
    container = next(
        (element for element in root.iter()
         if local_tag(element) == container_name),
        None,
    )
    return len(list(container)) if container is not None else 0

print("For each path, this is how many questions in the testset have that path:")

for path, count in path_occurrences.items():
    try:
        text = Path(path).read_text(encoding="utf-8")

        try:
            log = json.loads(text)
            totals = {
                key: len(log.get(key, []))
                for key in ("events", "objects", "eventTypes", "objectTypes")
            }
        except json.JSONDecodeError:
            root = ET.fromstring(text)
            xml_names = {
                "events": "events",
                "objects": "objects",
                "eventTypes": "event-types",
                "objectTypes": "object-types",
            }
            totals = {
                key: xml_count(root, xml_name)
                for key, xml_name in xml_names.items()
            }


    except (OSError, ET.ParseError) as e:
        print(f"Could not read {path}: {e}")
        continue

    display_path = path.replace("\\", "/")
    print(f"Path: {display_path}")
    print(f"Total events in {display_path}: {totals['events']}")
    print(f"Total objects in {display_path}: {totals['objects']}")
    print(f"Total event types in {display_path}: {totals['eventTypes']}")
    print(f"Total object types in {display_path}: {totals['objectTypes']}")
    print(f"Total count of testset questions: {count}")
    print("--------------------------------------------------")