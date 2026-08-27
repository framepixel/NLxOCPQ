#!/usr/bin/env python3
"""Replay one already-generated query and measure its result export. No model, no GPU.

WHY
---
Shard 60 was OOM-killed three times parsing a prediction's result export, and each time the
submit script's `trap ... EXIT` removed the scratch directory, so nobody could state the file's
size afterwards. Every threshold so far has been inferred from export *timings*.

The generated query is written to `<run_dir>/generated/case_NNN/<config_id>_rep0.json` BEFORE the
query is executed (eval_testset.py:1386), so a shard that died during the parse still has it. That
means the failing case can be replayed without loading a model: run the CLI on the saved query,
measure what it writes, and ask the guard whether it would refuse it.

Costs a couple of minutes on a CPU-only allocation instead of a full shard behind the GPU queue.

USAGE
    python diagnose_result_size.py --shard-dir eval_runs/_abandoned/shard_m5_s0_e0_c0of2_2026... \\
                                   --case 37

    # optional: also attempt the parse, to find out what it really costs. DANGEROUS -- this is
    # the thing that OOM-killed the shard. Only under a generous --mem allocation.
    python diagnose_result_size.py --shard-dir ... --case 37 --attempt-parse
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

GB = 1024 ** 3


def _resolve_case(shard_dir: Path, case_no: int):
    """Map a shard-local case number back to its testset row, exactly as the run did."""
    import math

    from eval_testset import _load_testset

    manifest = json.loads((shard_dir / "config_resolved.json").read_text(encoding="utf-8"))
    cfg = manifest.get("config") or {}
    rows = _load_testset(
        manifest.get("testset_path") or cfg.get("testset_path", "testset.json"),
        int(cfg.get("limit", 0) or 0),
        int(cfg.get("seed", 3407) or 3407),
        bool(cfg.get("shuffle", False)),
    )
    valid = [r for r in rows
             if r.get("nl_query", "") and r.get("ocel_path") and r.get("ocpq_query")]

    # The shard name carries its own case-half, which is the authority here -- the manifest's
    # copy of the config has already had the shard filters folded into it.
    half = int(shard_dir.name.split("_c")[1].split("of")[0])
    total = int(shard_dir.name.split("of")[1].split("_")[0])
    chunk = math.ceil(len(valid) / total)
    sliced = valid[half * chunk: half * chunk + chunk]
    if not 1 <= case_no <= len(sliced):
        raise SystemExit(f"FATAL: case {case_no} outside this shard's 1..{len(sliced)}")
    return sliced[case_no - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", required=True, type=Path)
    ap.add_argument("--case", required=True, type=int)
    ap.add_argument("--attempt-parse", action="store_true",
                    help="Also try the parse. This is what killed the shard -- use a big --mem.")
    args = ap.parse_args()

    shard_dir = args.shard_dir if args.shard_dir.is_absolute() else HERE / args.shard_dir
    if not shard_dir.is_dir():
        raise SystemExit(f"FATAL: {shard_dir} is not a directory")

    case_dir = shard_dir / "generated" / f"case_{args.case:03d}"
    queries = sorted(case_dir.glob("*.json")) if case_dir.is_dir() else []
    if not queries:
        raise SystemExit(
            f"FATAL: no generated query at {case_dir}.\n"
            f"       The shard died before writing it, so this case cannot be replayed."
        )
    query_path = queries[0]
    print(f"query : {query_path.relative_to(shard_dir)}  ({query_path.stat().st_size} bytes)")

    # What the dead run itself was configured with. This settles, from evidence rather than
    # inference, whether the size guard was active when it died -- and whether a --resume against
    # this directory is still possible (it is refused when config_fingerprint has moved on).
    manifest = json.loads((shard_dir / "config_resolved.json").read_text(encoding="utf-8"))
    print()
    print("that run was configured with:")
    print(f"  max_result_file_gb : {manifest.get('max_result_file_gb', 'ABSENT -- guard was OFF')}")
    print(f"  case_timeout_sec   : {manifest.get('case_timeout_sec')}")
    print(f"  config_fingerprint : {str(manifest.get('config_fingerprint'))[:16]}...")
    try:
        from eval_testset import _config_fingerprint
        live = _config_fingerprint(json.loads((HERE / "eval_config.json").read_text(encoding="utf-8")))
        same = live == manifest.get("config_fingerprint")
        print(f"  matches current    : {same}"
              f"{'' if same else '  -> --resume against this dir would be REFUSED'}")
    except Exception as exc:
        print(f"  (could not compare against the current config: {exc})")

    row = _resolve_case(shard_dir, args.case)
    ocel = row["ocel_path"]
    print(f"log   : {ocel}")
    print(f"nl    : {row['nl_query'][:96]}")

    from QueryExecutor import discard_result_file, run_ocpq_query

    print("\nexecuting (this is the CLI only -- no model is loaded) ...")
    t0 = time.time()
    stdout, stderr, result_path = run_ocpq_query(ocel, str(query_path))
    print(f"CLI wall time: {time.time() - t0:.1f}s")

    if not result_path or not os.path.exists(result_path):
        print("\nNo result file was produced -- this case fails at execution, not at parse.")
        print(f"stderr: {(stderr or '')[:400]}")
        return 0

    size = os.path.getsize(result_path)
    print(f"\n=== THE NUMBER NOBODY COULD STATE ===")
    print(f"  export size      : {size / GB:.3f} GB ({size} bytes)")
    for factor in (11.7, 15, 20, 25, 30):
        print(f"  projected @ {factor:>4}x : {size * factor / GB:>7.1f} GB")

    try:
        cfg = json.loads((HERE / "eval_config.json").read_text(encoding="utf-8"))
        cap = float(cfg.get("max_result_file_gb") or 0)
    except Exception:
        cap = 0.0

    from eval_testset import _result_file_too_large

    verdict = _result_file_too_large(result_path, cap)
    print(f"\n  configured cap   : {cap or 'DISABLED'} GB")
    print(f"  guard would fire : {verdict is not None}")
    if cap and verdict is None:
        print(f"  -> the guard PASSES this file. If the shard still dies here, the cap is too")
        print(f"     high: set it below {size / GB:.3f} GB.")
    elif verdict is not None:
        print(f"  -> the guard REFUSES it. The shard would record result_too_large and continue.")
    else:
        print(f"  -> no cap configured; this file would be parsed and may kill the process.")

    if args.attempt_parse:
        print("\nattempting the parse (this is what killed the shard) ...")
        from eval_testset import _load_result_ids
        t0 = time.time()
        ids, skipped, oversized = _load_result_ids(
            result_path, json.loads(query_path.read_text(encoding="utf-8")), max_gb=0)
        print(f"  parse survived in {time.time() - t0:.1f}s; "
              f"ids={'None' if ids is None else 'extracted'}, bindings_skipped={skipped}")

    discard_result_file(result_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
