#!/usr/bin/env python3
"""Run one tree through BOTH OCPQ front ends and print what each counts, node by node.

WHY THIS EXISTS
---------------
The agent revises against the web server's counts (`try_query`) and is scored by `ocpq_cli`'s. The
two share `evaluate_box_tree`, so they must agree. In the scored agentic run (notes 43.15), A1's
final tree for six angular-log cases -- one identical tree -- got 51 root situations from the
cluster's server and 45 from its CLI. On the laptop, the same tree on a byte-identical log gets 17
from both, every time. So OCPQ is deterministic and its front ends agree when built together; what
differs on the cluster is the binaries. This script reproduces a disagreement on whichever host it
runs on, with the binaries' provenance beside the counts.

It is a diagnostic, not an evaluation: no metric is computed, and the CLI runs through
`QueryExecutor` as everywhere else.

USAGE
    python src/probe_cli_diff.py --run-dir eval_runs/agentic/<A1 dir> --disagreements
    python src/probe_cli_diff.py --run-dir eval_runs/agentic/<A1 dir> --case-id 17 --repeats 3
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import QueryExecutor
import ocpq_probe
import paths
from eval_testset import root_node_index


def _local_log(recorded_path, corpus):
    """The log to use on THIS host, checked against the run's manifest digest.

    A record names the path on the host that ran it; on another host the same basename under
    `paths.OCEL_LOGS` is only the same log if its sha256 matches (the logs have diverged before).
    """
    path = Path(recorded_path)
    if not path.exists():
        path = paths.OCEL_LOGS / path.name
    digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    want = (corpus.get(path.name) or {}).get("sha256")
    verdict = "matches the run" if digest and digest == want else f"DIFFERS from the run ({want})"
    print(f"  log {path}  sha256 {str(digest)[:16]} -- {verdict}")
    return str(path)


def _cli_counts(ocel_path, tree):
    """Per-node situation counts from ocpq_cli, or the error text."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(tree, fh)
        query_file = fh.name
    try:
        _stdout, stderr, result = QueryExecutor.run_ocpq_query(ocel_path, query_file)
        if not result:
            return f"no result ({str(stderr)[:200]})"
        try:
            with open(result, encoding="utf-8") as fh:
                doc = json.load(fh)
        finally:
            QueryExecutor.discard_result_file(result)
        ok = doc.get("Ok") if isinstance(doc, dict) else None
        if not isinstance(ok, dict):
            return f"error export: {str(doc)[:200]}"
        return [len(node.get("situations") or []) for node in ok.get("evaluationResults") or []]
    finally:
        os.unlink(query_file)


def _server_counts(probe, tree):
    result = probe.try_query({"tree": tree})
    if result.get("error"):
        return f"error: {str(result['error'])[:200]}"
    return [node.get("situations") for node in result.get("nodes") or []]


def _cases(run_dir, case_id, disagreements):
    """(case ids, ocel path, tree) per distinct final tree to test."""
    records = json.load(open(Path(run_dir) / "predictions.json", encoding="utf-8"))
    picked = {}
    for rec in records:
        agent = rec.get("agent") or {}
        agreement = agent.get("probe_vs_cli") or {}
        wanted = (rec.get("case_id") == case_id if case_id is not None
                  else disagreements and agreement.get("agree") is False)
        if not wanted or not isinstance(rec.get("prediction"), dict):
            continue
        key = (rec["ocel_path"], json.dumps(rec["prediction"], sort_keys=True))
        entry = picked.setdefault(key, {"cases": [], "recorded": agreement})
        entry["cases"].append(rec["case_id"])
    return [(v["cases"], ocel, json.loads(tree), v["recorded"]) for (ocel, tree), v in picked.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--case-id", type=int)
    parser.add_argument("--disagreements", action="store_true",
                        help="every final tree whose recorded probe_vs_cli disagreed")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.case_id is None and not args.disagreements:
        parser.error("pass --case-id N or --disagreements")

    provenance = QueryExecutor.ocpq_provenance(build=False)
    print(f"OCPQ rev {provenance.get('rev')} branch {provenance.get('branch')} "
          f"dirty {provenance.get('dirty')}")
    for name, binary in (provenance.get("binaries") or {}).items():
        print(f"  {name:16s} sha256 {str(binary.get('sha256'))[:16]}  mtime {binary.get('mtime')}")

    corpus = json.load(open(Path(args.run_dir) / "config_resolved.json",
                            encoding="utf-8")).get("corpus") or {}
    cases = _cases(args.run_dir, args.case_id, args.disagreements)
    if not cases:
        print("nothing to test")
        return 1
    with ocpq_probe.ProbeServer() as server:
        probe = ocpq_probe.OcpqProbe(server.base_url, max_tool_calls=10 ** 6, server=server)
        for case_ids, ocel_path, tree, recorded in cases:
            print(f"\ncases {case_ids}  {Path(ocel_path).name}  root node {root_node_index(tree)}")
            print(f"  recorded in the run: {recorded}")
            ocel_path = _local_log(ocel_path, corpus)
            probe.ensure_log(ocel_path)
            for attempt in range(args.repeats):
                print(f"  #{attempt}  cli {_cli_counts(ocel_path, tree)}  "
                      f"server {_server_counts(probe, tree)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
