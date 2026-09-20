#!/usr/bin/env python
"""Do the two copies of a log differ only in DECLARED TYPES, or in the data as well?

2026-09-12. The same log, `renting_log_high.json`, on two machines:

    local    57,321,096 bytes  md5 10907eb6d857aae39d21cbafa28a8d2e
             string 24, integer 1 (Age), boolean 1 (isMarried), float 1
    cluster  57,326,130 bytes  md5 d7ce5f131f1fd2ed9f73375e725c0b74
             string 26,                                        float 1

`extract_ocel_schema` agrees with its own file on both machines, so the extractor is fine and the
CORPORA are different. Across every cluster log there is not one `integer` and not one `boolean`.

Declared types alone cannot account for 5,034 bytes, so the question this answers is how deep the
difference goes -- and the answer decides how much prior work is affected:

  * types only  -> the PROMPT and the type-aware grammar were wrong on the cluster, but the
                   executed result sets are comparable, so F1, the floors, and every published
                   number still mean what they said.
  * data too    -> gold's result sets differ between the machines. `testset_discriminability.json`
                   -- the trivial floors, including the 0.7250 headline -- was computed on the
                   LOCAL corpus and is untracked, so it may not describe the runs at all.

Structural counts and a hash over sorted ids are enough to separate those without diffing 57 MB.

    python src/log_corpus_fingerprint.py                       # renting
    python src/log_corpus_fingerprint.py --log ../Logs/x.json

Run on both machines and compare the blocks line for line. Big logs (bpic2017 is 2.3 GB) are
skipped unless named explicitly.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402


def digest(values):
    """Order-independent hash, so a reordered export does not read as different data."""
    hasher = hashlib.md5()
    for value in sorted(values):
        hasher.update(str(value).encode("utf-8"))
        hasher.update(b"\x00")
    return hasher.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="default ../Logs/renting_log_high.json")
    args = ap.parse_args()

    logs_dir = (paths.REPO_ROOT / ".." / "Logs").resolve()
    log = (Path(args.log).resolve() if args.log
           else (logs_dir / "renting_log_high.json"))

    if not log.exists():
        print(f"MISSING: {log}")
        return 1

    raw = log.read_bytes()
    print(f"LOG   {log}")
    print(f"  size            : {len(raw)}")
    print(f"  md5             : {hashlib.md5(raw).hexdigest()}")

    document = json.loads(raw.decode("utf-8"))
    objects = document.get("objects") or []
    events = document.get("events") or []
    object_types = document.get("objectTypes") or []
    event_types = document.get("eventTypes") or []

    print(f"  objectTypes     : {len(object_types)}")
    print(f"  eventTypes      : {len(event_types)}")
    print(f"  objects         : {len(objects)}")
    print(f"  events          : {len(events)}")

    # Identity of the DATA, independent of declared types.
    print(f"  object ids hash : {digest(o.get('id') for o in objects)}")
    print(f"  event ids hash  : {digest(e.get('id') for e in events)}")
    print(f"  object type set : {digest(t.get('name') for t in object_types)}")
    print(f"  event type set  : {digest(t.get('name') for t in event_types)}")

    # Attribute VALUES, which is what a query actually filters on. If these match while the
    # declared types differ, only the declarations were rewritten.
    value_pairs = []
    for obj in objects:
        for attribute in (obj.get("attributes") or []):
            value_pairs.append(f"{obj.get('id')}|{attribute.get('name')}|{attribute.get('value')}")
    print(f"  object attr vals: {len(value_pairs)} pairs, hash {digest(value_pairs)}")

    print("  declared types per objectType:")
    for entry in sorted(object_types, key=lambda t: str(t.get("name"))):
        attributes = sorted((a.get("name"), a.get("type"))
                            for a in (entry.get("attributes") or []))
        if attributes:
            print(f"     {str(entry.get('name')):22s} {attributes}")

    print()
    print("Compare with the other machine:")
    print("  same ids/value hashes + different declared types -> DECLARATIONS ONLY.")
    print("  differing ids or value hashes -> the corpora are different data, and anything")
    print("  computed from one does not describe runs on the other.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
