# import json, collections
# for path in [
#     "../Logs/bpic2017-ocel2.json",
#     "../Logs/ocel2-p2p.json",
# ]:
#     with open(path) as f: d = json.load(f)
#     ev_attrs = collections.Counter()
#     for e in d.get("events", []):
#         for a in e.get("attributes", []):
#             ev_attrs[(a["name"], type(a["value"]).__name__)] += 1
#     print(path, dict(ev_attrs.most_common(5)))

import json, collections
for path, attr in [
    ("../Logs/lending_log_high.json", "Yearsofeducation"),
]:
    with open(path) as f: d = json.load(f)
    vals = collections.Counter(
        a["value"] for e in d.get("objects", [])
        for a in e.get("attributes", []) if a["name"] == attr
    )
    print(f"{attr}:", dict(vals.most_common(8)))
