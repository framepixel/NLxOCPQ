patterns = [
    "EVENT_A happens before EVENT_B",
    "Whenever EVENT_A occurs, EVENT_B eventually occurs",
    "After EVENT_A, EVENT_B must not occur",
    "EVENT_A and EVENT_B refer to the same OBJECT",
    "EVENT_B happens within TIME hours after EVENT_A"
]

events = [
    "create order",
    "add item",
    "ship package",
    "pay order",
    "cancel order",
    "approve application",
    "submit application",
    "review application",
    "merge pull request",
    "open issue"
]

objects = [
    "orders",
    "items",
    "applications",
    "pull requests",
    "issues"
]

import random
import json

def generate_example():
    pattern = random.choice(patterns)

    A = random.choice(events)
    B = random.choice(events)
    obj = random.choice(objects)
    t = random.choice([1, 6, 24, 72])

    text = pattern.replace("EVENT_A", A)\
                  .replace("EVENT_B", B)\
                  .replace("OBJECT", obj)\
                  .replace("TIME", str(t))

    ocpq = {
        "pattern": pattern,
        "event_A": A,
        "event_B": B,
        "object": obj,
        "time": t
    }

    return {"nl": text, "query": ocpq}

dataset = [generate_example() for _ in range(200)]

with open("ocpq_dataset.json","w") as f:
    json.dump(dataset,f,indent=2)