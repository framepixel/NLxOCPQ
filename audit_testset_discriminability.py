"""Audit `testset.json` for entries the evaluation metric cannot discriminate.

A test case is only usable when its gold result is a *proper non-trivial subset*. Two
degenerate shapes make a case worthless regardless of how good the metric is:

* **Empty result** — any broken model query that also returns nothing scores a perfect F1.
* **Whole population, nothing distinguishing it** — any query that merely binds the right
  object type scores a perfect F1 without expressing the constraint at all. For compliance
  queries this shows up as a 0% or 100% violation rate.

The known cause of the 0%-violation class is OCPQ's `SAT` constraint being *vacuously*
satisfied: `SAT` is violated only when an existing child binding carries its own violation,
so an empty child set satisfies it trivially. The fix is to make absence itself the
violation via `SizeFilter{NumChilds, child_name, min: 1}` on the parent.

Result-set extraction is imported from `eval_testset.py` rather than reimplemented, so this
audit always measures exactly what the evaluation measures (CLAUDE.md: no metric logic
outside `eval_testset.py`).

Outputs (written to `--output-dir`, default `eval_runs/`):
  * `testset_discriminability.json` — per-entry verdict and counts.
  * `testset_discriminability_proposals.json` — proposed replacement gold queries for
    degenerate entries. **Proposals only** — `testset.json` is read-only and nothing is
    applied automatically.

Usage:
    python audit_testset_discriminability.py [--testset testset.json] [--limit N]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

from eval_testset import _extract_result_ids, root_node_index
from QueryExecutor import run_ocpq_query


VERDICT_OK = "ok"
VERDICT_EMPTY = "degenerate_empty"
VERDICT_SAME_AS_BASELINE = "degenerate_same_as_baseline"
VERDICT_EXEC_FAILED = "execution_failed"
VERDICT_BASELINE_FAILED = "baseline_execution_failed"

DEGENERATE_VERDICTS = {
    VERDICT_EMPTY,
    VERDICT_SAME_AS_BASELINE,
}

# Filters that express linkage between entities rather than an assertion about them.
# The no-op baseline keeps these so it stays bounded; every other filter is query logic
# and gets stripped. See `_build_baseline_query`.
STRUCTURAL_FILTER_TYPES = {"O2E", "O2O"}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _filter_vars_declared_at_root(query_filter: dict, root_data: dict) -> bool:
    """Whether every variable a structural filter references is declared at the root.

    A gold tree can place an `O2E`/`O2O` on the root that links to a variable introduced
    further down the tree. Lifting the root out on its own leaves that reference dangling
    and `ocpq_cli` rejects the whole query with `{"Err": "Could not get o1"}`, which the
    audit would otherwise misreport as an execution failure. Such a filter describes the
    tree's structure rather than the root box's own, so dropping it from the baseline is
    both necessary and correct.
    """
    event_vars = set(root_data.get("newEventVars") or {})
    object_vars = set(root_data.get("newObjectVars") or {})

    if query_filter.get("type") == "O2E":
        return (
            str(query_filter.get("object")) in object_vars
            and str(query_filter.get("event")) in event_vars
        )
    return (
        str(query_filter.get("object")) in object_vars
        and str(query_filter.get("other_object")) in object_vars
    )


def _build_baseline_query(gold_query: dict) -> Optional[dict]:
    """Build the no-op counterpart of a gold query: same root anchor, no query logic.

    Keeps only the root box's variable declarations and its **structural** filters
    (`O2E` / `O2O`). Everything that asserts something about the data is stripped:
    `sizeFilters`, `constraints`, every child node, and the non-structural filters
    (`TimeBetweenEvents`, `NotEqual`, `EventAttributeValueFilter`,
    `ObjectAttributeValueFilter`, `BasicFilterCEL`).

    The structural/non-structural split matters in both directions. Keeping `O2E`/`O2O`
    prevents the baseline from becoming an unbounded Cartesian product that exhausts
    memory — they say which entities are *linked*, not what the query claims. Stripping
    the rest is what makes the comparison meaningful for single-node queries: a query
    whose entire logic is one `ObjectAttributeValueFilter` would otherwise equal its own
    baseline by construction and be reported degenerate even though it discriminates
    perfectly well.

    Comparing the gold result against this baseline answers the question the metric
    actually cares about: does the query's logic change the root result at all? If it
    does not, then a model can score a perfect F1 by emitting the trivial query, and the
    test case cannot discriminate.

    Args:
        gold_query: The gold OCPQ query.

    Returns:
        The baseline query, or None when the gold query has no nodes.
    """
    nodes = gold_query.get("nodes") or []
    if not nodes:
        return None

    # The root is not guaranteed to be index 0 - a valid tree may list a child first.
    root = copy.deepcopy(nodes[root_node_index(gold_query)])
    root_data = root.get("Box", [None, None])[0]
    if not isinstance(root_data, dict):
        return None

    root_data["filters"] = [
        f
        for f in root_data.get("filters") or []
        if isinstance(f, dict)
        and f.get("type") in STRUCTURAL_FILTER_TYPES
        and _filter_vars_declared_at_root(f, root_data)
    ]
    root_data["sizeFilters"] = []
    root_data["constraints"] = []
    # CEL labels are computed annotations that routinely reference a child path
    # ("A.size()", "A.map(b, ...)"). With the children stripped that path no longer
    # resolves and the CLI aborts, so they must go too. They never change which
    # bindings the box produces, so removing them cannot alter the comparison.
    root_data["labels"] = []
    # `evVarLabels` / `obVarLabels` are deliberately kept: they change which variables
    # appear in a binding, so dropping them would change the situation tuple's shape and
    # make gold and baseline trivially unequal.
    root["Box"][1] = []

    return {"nodes": [root], "edgeNames": []}


def _classify(situations: set, baseline_situations: Optional[set]) -> str:
    """Assign a discriminability verdict by comparing gold against its no-op baseline."""
    if not situations:
        return VERDICT_EMPTY
    if baseline_situations is None:
        return VERDICT_BASELINE_FAILED
    if situations == baseline_situations:
        return VERDICT_SAME_AS_BASELINE
    return VERDICT_OK


def _edges(gold_query: dict) -> list:
    """Return `(parent_index, child_index, edge_label)` for every edge in the tree."""
    edges = []
    for edge in gold_query.get("edgeNames") or []:
        # Serialised as [[parent_index, child_index], "label"].
        if isinstance(edge, list) and len(edge) == 2 and isinstance(edge[0], list):
            parent_index, child_index = edge[0]
            edges.append((parent_index, child_index, edge[1]))
    return edges


def _already_guarded(node_data: dict, child_name: str) -> bool:
    """Whether this node already asserts the size of the given child path."""
    for constraint in node_data.get("constraints") or []:
        if (
            isinstance(constraint, dict)
            and constraint.get("type") == "SizeFilter"
            and (constraint.get("filter") or {}).get("child_name") == child_name
        ):
            return True
    for size_filter in node_data.get("sizeFilters") or []:
        if isinstance(size_filter, dict) and size_filter.get("child_name") == child_name:
            return True
    return False


def _candidate_repairs(gold_query: dict) -> list:
    """Enumerate `NumChilds min=1` repairs, one per unguarded edge in the tree.

    The degenerate class this targets is a child path whose emptiness is never asserted,
    so the child does no work the metric can observe. `SAT` has this problem because it
    is violated only when an *existing* child binding carries its own violation, meaning
    an empty child set satisfies it vacuously; a node with children and no constraint at
    all has it trivially. Both are repaired by making absence itself the violation.

    Which edge needs the assertion is not decidable from the tree shape alone — in a
    `root -> A -> B` chain, asserting on `A` is often trivially true while asserting on
    `B` is the real rule — so every unguarded edge is offered as a candidate and the
    caller verifies which one actually restores discrimination.

    Returns:
        `(candidate_query, parent_index, edge_label)` triples.
    """
    candidates = []
    for parent_index, _child_index, label in _edges(gold_query):
        nodes = gold_query.get("nodes") or []
        if parent_index >= len(nodes):
            continue
        parent_data = nodes[parent_index].get("Box", [None])[0]
        if not isinstance(parent_data, dict) or _already_guarded(parent_data, label):
            continue

        candidate = copy.deepcopy(gold_query)
        candidate_data = candidate["nodes"][parent_index]["Box"][0]
        candidate_data.setdefault("constraints", []).append(
            {
                "type": "SizeFilter",
                "filter": {"type": "NumChilds", "child_name": label, "min": 1, "max": None},
            }
        )
        candidates.append((candidate, parent_index, label))
    return candidates


def _propose_fix(
    entry: dict,
    verdict: str,
    baseline_situations: Optional[set],
    scratch: Path,
    index: int,
) -> Optional[dict]:
    """Search for a repair that verifiably restores discrimination, or return None.

    Every candidate is executed and kept only if it produces a strictly non-trivial
    violation rate *and* a root result that differs from the no-op baseline. An
    unverified proposal is worse than none: it looks authoritative while quietly
    encoding a different rule than the natural-language query states.
    """
    if verdict != VERDICT_SAME_AS_BASELINE:
        return None

    gold_query = entry["ocpq_query"]
    ocel_path = entry["ocel_path"]

    for candidate, parent_index, label in _candidate_repairs(gold_query):
        situations = _run_and_extract(
            ocel_path, candidate, scratch / f"cand_{index:03d}_{parent_index}_{label}.json"
        )
        if not situations:
            continue

        violated = sum(1 for _e, _o, is_violated in situations if is_violated)
        discriminating = 0 < violated < len(situations)
        differs_from_baseline = baseline_situations is None or situations != baseline_situations
        if not (discriminating and differs_from_baseline):
            continue

        return {
            "nl_query": entry.get("nl_query"),
            "reason": (
                f"Child path '{label}' of node {parent_index} was never asserted, so the "
                "root result matched a query with no children at all. Adding "
                f"NumChilds min=1 on that path makes absence the violation."
            ),
            "verified": {
                "root_situations": len(situations),
                "root_violations": violated,
                "violation_rate": round(violated / len(situations), 4),
            },
            "proposed_ocpq_query": candidate,
            "review_note": (
                "Verified to produce a non-trivial violation rate. Confirm it encodes the "
                "intended rule before applying - the audit checks discrimination, not meaning."
            ),
        }

    return None


def _run_and_extract(ocel_path: str, query: dict, query_path: Path) -> Optional[set]:
    """Execute one OCPQ query and return its root-node situation set.

    Returns:
        The situation set, or None when execution failed or produced no results.
    """
    _write_json(query_path, query)
    try:
        _stdout, _stderr, result_path = run_ocpq_query(ocel_path, str(query_path))
    except Exception:  # noqa: BLE001 - a crashing query is a reportable outcome, not a stop
        return None
    if not result_path:
        return None
    extracted = _extract_result_ids(_read_json(result_path), query)
    return extracted[2] if extracted else None


def audit(testset_path: str, output_dir: Path, limit: int = 0) -> dict:
    """Execute every gold query and classify how well it discriminates."""
    testset = _read_json(testset_path)
    if limit:
        testset = testset[:limit]

    scratch = output_dir / "_discriminability_queries"
    scratch.mkdir(parents=True, exist_ok=True)

    results = []
    proposals = {}

    for index, entry in enumerate(testset):
        nl_query = entry.get("nl_query", "")
        ocel_path = entry.get("ocel_path")
        gold_query = entry.get("ocpq_query")
        if not ocel_path or not gold_query:
            continue

        started = time.time()
        situations = _run_and_extract(ocel_path, gold_query, scratch / f"gold_{index:03d}.json")

        baseline_query = _build_baseline_query(gold_query)
        baseline_situations = (
            _run_and_extract(ocel_path, baseline_query, scratch / f"baseline_{index:03d}.json")
            if baseline_query
            else None
        )

        if situations is None:
            verdict = VERDICT_EXEC_FAILED
            situations = set()
        else:
            verdict = _classify(situations, baseline_situations)

        total = len(situations)
        violated = sum(1 for _e, _o, is_violated in situations if is_violated)

        record = {
            "index": index,
            "nl_query": nl_query,
            "ocel_path": ocel_path,
            "verdict": verdict,
            "root_situations": total,
            "root_violations": violated,
            "violation_rate": round(violated / total, 4) if total else None,
            "baseline_situations": (
                len(baseline_situations) if baseline_situations is not None else None
            ),
            "duration_sec": round(time.time() - started, 3),
        }
        results.append(record)

        if verdict in DEGENERATE_VERDICTS:
            proposal = _propose_fix(entry, verdict, baseline_situations, scratch, index)
            if proposal:
                proposals[str(index)] = proposal
                record["proposal_verified"] = proposal["verified"]

        flag = "  " if verdict == VERDICT_OK else "!!"
        baseline_note = record["baseline_situations"]
        print(
            f"{flag} [{index:3d}] {verdict:28s} "
            f"sits={total:6d} viol={violated:6d} base={baseline_note} | {nl_query[:52]}"
        )

    degenerate = [r for r in results if r["verdict"] in DEGENERATE_VERDICTS]
    failed = [r for r in results if r["verdict"] == VERDICT_EXEC_FAILED]

    summary = {
        "testset_path": testset_path,
        "total": len(results),
        "ok": len(results) - len(degenerate) - len(failed),
        "degenerate": len(degenerate),
        "execution_failed": len(failed),
        "by_verdict": {
            verdict: sum(1 for r in results if r["verdict"] == verdict)
            for verdict in sorted({r["verdict"] for r in results})
        },
        "degenerate_indices": [r["index"] for r in degenerate],
        "proposals_available": len(proposals),
        "entries": results,
    }

    _write_json(output_dir / "testset_discriminability.json", summary)
    _write_json(output_dir / "testset_discriminability_proposals.json", proposals)

    print()
    print(f"total              : {summary['total']}")
    print(f"discriminating     : {summary['ok']}")
    print(f"degenerate         : {summary['degenerate']}  -> {summary['degenerate_indices']}")
    print(f"execution failed   : {summary['execution_failed']}")
    print(f"auto-proposed fixes: {summary['proposals_available']}")
    print()
    print(f"wrote {output_dir / 'testset_discriminability.json'}")
    print(f"wrote {output_dir / 'testset_discriminability_proposals.json'}")
    print("testset.json is read-only - review proposals before applying anything.")

    return summary


def _make_stdout_unicode_safe() -> None:
    """Stop Windows' cp1252 default from killing the run on a non-ASCII query.

    Several testset queries contain characters outside cp1252 (e.g. the arrows in
    "Application -> initial interview" and euro signs). When stdout is redirected to a
    file on Windows it defaults to the ANSI codepage, so printing such a query raises
    UnicodeEncodeError and aborts the audit partway through.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass


def main() -> None:
    _make_stdout_unicode_safe()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default="testset.json")
    parser.add_argument("--output-dir", default="eval_runs")
    parser.add_argument("--limit", type=int, default=0, help="Audit only the first N entries.")
    args = parser.parse_args()

    audit(args.testset, Path(args.output_dir), args.limit)


if __name__ == "__main__":
    main()
