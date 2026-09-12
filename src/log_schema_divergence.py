#!/usr/bin/env python
"""Do the cluster and the local machine agree about what the evaluation logs DECLARE?

On 2026-09-12 `src/typed_grammar_probe.py` reported, on the cluster:

    Age declared as    : 'string'
    typed defs created : 2 ['...__float', '...__string']

The same log read on the local machine, straight from the JSON with no cache and no helper:

    objectType Person -> Age integer, isCitizen string, isMarried boolean, throughputTime float

Those cannot both be right, and which one is right decides more than one experiment:

  * `build_schema_block` renders the declared type into the PROMPT. If the cluster believes `Age`
    is a string, then every run ever executed there TOLD the model `Age (String)` -- and the
    model's "error" of emitting `String{is_in:["40"]}` for "older than 40" is the model correctly
    following what it was told. Notes 34.3 calls that error the dominant failure sub-class and
    says the information "was never missing"; that was verified against a LOCALLY rendered block.
  * The type-aware grammar is built from the same extraction, so on the cluster it groups `Age`
    with the strings -- which is why the smoke emitted `Age -> String` with the constraint ON and
    nothing rejected it.
  * Gold uses `Integer{min:41}` on `Age`. Under the cluster's reading that gold is a type
    contradiction, so `tests/test_type_aware_filters.py` -- which asserts all 169 gold queries
    survive their own log's grammar -- should FAIL on the cluster while passing locally.

Note the difference is selective: `throughputTime -> float` survives while `integer` and `boolean`
collapse to `string`. A wholly different file would not usually land that way, which is why this
checks the raw bytes and the cache separately rather than assuming either.

Run it on BOTH machines and compare the blocks:

    python src/log_schema_divergence.py
    python src/log_schema_divergence.py --log ../Logs/renting_log_high.json
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402

PROBE_ATTRIBUTES = ("Age", "isMarried", "isCitizen", "throughputTime")


def raw_declarations(log_path):
    """Declared types straight from the file. No helper, no cache -- the ground truth."""
    with open(log_path, encoding="utf-8") as handle:
        document = json.load(handle)
    found = {}
    for key in ("objectTypes", "eventTypes"):
        for entry in (document.get(key) or []):
            for attribute in (entry.get("attributes") or []):
                name = attribute.get("name")
                if name is not None:
                    found.setdefault(name, str(attribute.get("type", "")).lower())
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=None, help="default ../Logs/renting_log_high.json")
    ap.add_argument("--all-logs", action="store_true",
                    help="Summarise every log in ../Logs instead of one.")
    args = ap.parse_args()

    logs_dir = (paths.REPO_ROOT / ".." / "Logs").resolve()
    if args.all_logs:
        targets = sorted(p for p in logs_dir.glob("*.json"))
    else:
        target = Path(args.log) if args.log else (logs_dir / "renting_log_high.json")
        targets = [target.resolve()]

    import collections
    from LogEvents import extract_ocel_schema

    print(f"host cwd     : {os.getcwd()}")
    print(f"logs dir     : {logs_dir}")
    print()

    for log in targets:
        print("=" * 78)
        print(f"LOG {log.name}")
        if not log.exists():
            print("   MISSING on this host.")
            continue
        data = log.read_bytes()
        print(f"   size       : {len(data)} bytes")
        print(f"   md5        : {hashlib.md5(data).hexdigest()}")
        print(f"   mtime      : {int(log.stat().st_mtime)}")

        raw = raw_declarations(str(log))
        raw_hist = collections.Counter(raw.values())
        print(f"   RAW  types : {dict(raw_hist)}  ({len(raw)} distinct names)")

        schema = extract_ocel_schema(str(log))
        via = {}
        for kind in ("object_attributes", "event_attributes"):
            for attrs in (schema.get(kind) or {}).values():
                for attribute in attrs:
                    via.setdefault(attribute["name"], str(attribute.get("type", "")).lower())
        via_hist = collections.Counter(via.values())
        print(f"   HELPER     : {dict(via_hist)}  ({len(via)} distinct names)")

        disagree = sorted(n for n in set(raw) | set(via) if raw.get(n) != via.get(n))
        if disagree:
            print(f"   *** extract_ocel_schema DISAGREES with the file on "
                  f"{len(disagree)} attribute(s) ***")
            for name in disagree[:12]:
                print(f"       {name:28s} file={raw.get(name)!r:12s} helper={via.get(name)!r}")
            print("       A helper that disagrees with the file is feeding wrong types into")
            print("       build_schema_block (the PROMPT) and into the type-aware grammar.")
        else:
            print("   helper agrees with the file on every attribute.")

        if not args.all_logs:
            print("   probe attributes:")
            for name in PROBE_ATTRIBUTES:
                print(f"       {name:16s} file={raw.get(name)!r:12s} helper={via.get(name)!r}")

    print()
    print("=" * 78)
    cache = paths.REPO_ROOT / ".schema_cache"
    print(f"schema cache : {cache}  exists={cache.exists()}")
    if cache.exists():
        entries = sorted(cache.glob("*"))
        print(f"   {len(entries)} entry/entries; newest: "
              f"{[e.name for e in entries[-3:]]}")
        print("   If the helper disagrees with the file, delete this directory and re-run:")
        print("   a stale entry written by an older extractor would survive a code change only")
        print("   if the cache key missed it.")


if __name__ == "__main__":
    main()
