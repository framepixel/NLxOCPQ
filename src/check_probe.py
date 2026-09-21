#!/usr/bin/env python3
"""Preflight for the agentic arm: can the probe serve every log, and at what cost?

WHY THIS EXISTS
---------------
`src/check_eval_deps.py` fails a job before it reaches a GPU when a model directory or a log is
missing. The agentic arm adds a second thing that can be missing at hour three of a fourteen-hour
job: a running OCPQ web server holding the right log. Three failures this catches on the login node
instead of mid-run:

  * **A log the server cannot import.** Three of the eleven logs are XML (`04_hospital.xml`,
    `03_hiring.xml`, `angular_github_commits_ocel.xml`). A probe that silently fails on those would
    degrade exactly those cases while the run looks healthy.
  * **The wrong log served.** The server holds ONE global OCEL with no sessions, so identity is
    asserted against `LogEvents.extract_ocel_schema` after every load. Here it is asserted for all
    eleven at once.
  * **Memory.** The resident server holds a whole log beside the LLM on the GPU and a 24 GB-capped
    `ocpq_cli` child, against `--mem=64G`. Measure the peak before a long job dies of it, not after.

It also prints each import's duration, which is what the per-log lazy loading in `OcpqProbe`
is trading against (notes 43).

USAGE
    python src/check_probe.py                 # every log in the testset
    python src/check_probe.py --limit 2       # the first two, for a quick check
    python src/check_probe.py --keep-server   # leave it running and print the URL
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from LogEvents import extract_ocel_schema  # noqa: E402
from ocpq_probe import OcpqProbe, ProbeError, ProbeServer, _http_json  # noqa: E402


def _testset_logs(testset_path, limit=0):
    """Every distinct log the testset references, in first-appearance order.

    Mirrors `corpus_manifest.testset_logs`' path handling: entries store repo-root-relative paths
    with Windows separators, so they are normalised before resolving.
    """
    rows = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    seen, logs = set(), []
    for row in rows:
        raw = (row.get("ocel_path") or "").replace("\\", "/")
        if not raw:
            continue
        resolved = (paths.REPO_ROOT / raw).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        logs.append(resolved)
    return logs[:limit] if limit else logs


def _server_rss_mb(process):
    """Resident memory of the server process in MB, or None where it cannot be read.

    Uses /proc rather than psutil: psutil is not a dependency of the eval environment and
    CLAUDE.md 2.2 makes adding one a new named environment.
    """
    if process is None or os.name == "nt":
        return None
    try:
        status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return round(int(line.split()[1]) / 1024.0, 1)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", default=str(paths.DATA / "testset.json"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Check only the first N distinct logs.")
    parser.add_argument("--keep-server", action="store_true",
                        help="Leave the server running and print its URL (for manual probing).")
    parser.add_argument("--out", default=None, help="Write the timings as JSON.")
    args = parser.parse_args()

    logs = _testset_logs(args.testset, args.limit)
    print(f"probe preflight: {len(logs)} log(s) from {args.testset}")

    missing = [log for log in logs if not log.is_file()]
    if missing:
        print("FATAL: logs not found:", file=sys.stderr)
        for log in missing:
            print(f"  - {log}", file=sys.stderr)
        return 1

    report, problems = [], []
    server = ProbeServer()
    try:
        started = time.time()
        server.start()
        print(f"server up on {server.base_url} in {time.time() - started:.1f}s "
              f"(workdir {server.workdir})")
        probe = OcpqProbe(server.base_url, max_tool_calls=10 ** 6, server=server)

        for index, log in enumerate(logs, start=1):
            entry = {"log": log.name, "bytes": log.stat().st_size}
            try:
                load_started = time.time()
                probe.ensure_log(str(log))
                entry["load_sec"] = round(time.time() - load_started, 1)
                schema = extract_ocel_schema(str(log))
                # One real probe per log: a minimal tree binding the first object type. It exercises
                # the same evaluation path the agent's try_query uses, so a log that imports but
                # cannot be queried is caught here too.
                object_type = (schema.get("object_types") or [None])[0]
                tree = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": [object_type]},
                                            "filters": [], "sizeFilters": [], "constraints": [],
                                            "evVarLabels": {}, "obVarLabels": {}, "labels": []},
                                           []]}],
                        "edgeNames": []}
                probe_started = time.time()
                result = probe.call("try_query", {"tree": tree})
                entry["probe_sec"] = round(time.time() - probe_started, 3)
                entry["situations"] = (result.get("nodes") or [{}])[0].get("situations")
                entry["probe_error"] = result.get("error")
                entry["rss_mb"] = _server_rss_mb(server.process)
                if entry.get("probe_error"):
                    problems.append(f"{log.name}: try_query failed -> {entry['probe_error']}")
            except ProbeError as exc:
                entry["error"] = str(exc)
                problems.append(f"{log.name}: {exc}")
            report.append(entry)
            print(f"  [{index:2d}/{len(logs)}] {log.name:42s} "
                  f"load {entry.get('load_sec', '-'):>6}s  "
                  f"probe {entry.get('probe_sec', '-'):>7}s  "
                  f"situations {str(entry.get('situations', '-')):>9}  "
                  f"rss {str(entry.get('rss_mb', '-')):>8} MB"
                  + (f"  ERROR {entry['error']}" if entry.get("error") else ""))

        peak = max((entry.get("rss_mb") or 0) for entry in report) if report else 0
        slowest = max((entry.get("load_sec") or 0) for entry in report) if report else 0
        print(f"\npeak server RSS {peak} MB; slowest import {slowest}s")
        if peak > 32000:
            problems.append(f"peak server RSS {peak} MB leaves little of a 64 GB job for the model")

        if args.out:
            Path(args.out).write_text(json.dumps({"logs": report, "peak_rss_mb": peak},
                                                 indent=2), encoding="utf-8")
            print(f"wrote {args.out}")

        if args.keep_server:
            print(f"\nserver left running at {server.base_url} (pid {server.process.pid}); "
                  f"stop it yourself when done")
            server.process = None            # hand ownership over rather than killing it on exit
    finally:
        if not args.keep_server:
            server.stop()

    if problems:
        print("\nFATAL: the probe cannot serve every log:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(f"\nprobe OK ({len(report)} log(s) imported, queried and identity-checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
