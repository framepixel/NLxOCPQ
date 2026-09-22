"""Label each `testset.json` entry with how well a result-only metric can score it.

**This is a stratification tool, not a filter.** It was once used to decide which entries to
delete; that was selection bias — removing items because models can pass them accidentally
changes the population being measured instead of fixing the instrument. Its output is now a
reported stratum, and a non-zero `non_discriminating` count is the expected, correct result.

An entry is `non_discriminating` when the gold result equals what a *no-op baseline* returns —
the same query with its constraints, size filters, labels and children stripped away. That
happens for two very different reasons, and only the first is a defect:

* **Defective gold** — e.g. OCPQ's `SAT` is violated only when an existing child binding
  carries its own violation, so an empty child set satisfies it vacuously and the query cannot
  detect the violation its own NL describes. The repair is to make absence itself the violation
  via `SizeFilter{NumChilds, child_name, min: 1}` on the parent.
* **A compliance rule the log fully satisfies** — the gold query is *correct*, binds every
  candidate, and flags nobody because nothing in that log breaks the rule. Nothing is wrong
  with the entry; a result-only metric simply cannot tell a correct query from a lazy one on it.

`baseline_f1` quantifies exactly that: the F1 the no-op baseline already scores against gold.
On a fully-compliant entry it is ~1.0, meaning a model expressing nothing still earns full
marks — so every model score must be read against this floor.

Result-set extraction and scoring are imported from `eval_testset.py` rather than
reimplemented, so this audit measures exactly what the evaluation measures (CLAUDE.md 2.5: no
metric logic outside `eval_testset.py`).

Outputs (written to `--output-dir`, default `eval_runs/`):
  * `testset_discriminability.json` — per-entry verdict, counts and `baseline_f1`.
  * `testset_discriminability_proposals.json` — proposed replacement gold queries. **Proposals
    only** — `testset.json` is read-only and nothing is applied automatically. Treat these as
    leads for the *defective* class above; a fully-compliant entry needs no repair.

The summary is checkpointed after every entry and `--resume` reuses what is already there,
so an interrupted run (a full pass takes over an hour, most of it on the 2.2 GB `bpic2017`
log) can be continued rather than restarted. `--log-filter` selects entries by log substring
and accepts a leading `!` to negate, which is how the cheap logs run first:

    python audit_testset_discriminability.py --log-filter '!bpic2017' --resume
    python audit_testset_discriminability.py --log-filter bpic2017 --resume

Usage:
    python audit_testset_discriminability.py [--testset testset.json] [--limit N]
        [--log-filter SUBSTR] [--resume]
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import result_store
from eval_testset import (
    SIMILARITY_VARIANTS,
    _extract_result_ids,
    _set_metrics,
    root_node_index,
)
from QueryExecutor import discard_result_file, run_ocpq_query


# Verdict names. `non_discriminating` was called `degenerate_same_as_baseline` while this script
# was used to *filter* the testset. It is now a stratification label, and "degenerate" actively
# misdescribes what those entries are: compliance rules their log fully satisfies, whose gold
# query is correct and simply flags nobody. Deleting them was the wrong remedy (it is selection
# bias); reporting them as their own stratum, against the trivial floor below, is the right one.
VERDICT_OK = "ok"
VERDICT_EMPTY = "empty_result"
VERDICT_SAME_AS_BASELINE = "non_discriminating"
VERDICT_EXEC_FAILED = "execution_failed"
VERDICT_BASELINE_FAILED = "baseline_execution_failed"

# Entries a result-only metric cannot score meaningfully. Not "bad entries".
NON_DISCRIMINATING_VERDICTS = {
    VERDICT_EMPTY,
    VERDICT_SAME_AS_BASELINE,
}
DEGENERATE_VERDICTS = NON_DISCRIMINATING_VERDICTS  # backwards-compatible alias

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


def _run_and_extract(ocel_path: str, query: dict, query_path: Path,
                     store_dir: Optional[Path] = None) -> Optional[set]:
    """Execute one OCPQ query and return its root-node situation set.

    When `store_dir` is given the full extracted triple is persisted there first. That is what
    makes the trivial floor re-derivable: the floor is the F1 of the *no-op baseline* query, and
    this is the only place that query is ever executed. Before the store, only the resulting
    scalar survived, so adding a metric variant meant re-running the whole audit -- exactly the
    situation `result_store.py` exists to prevent for predictions, and the one place still
    exposed to it.

    Args:
        ocel_path: Log to execute against.
        query: The OCPQ query.
        query_path: Where to write the query file for the CLI.
        store_dir: Result store to persist the extracted triple into, or None to skip.

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
    try:
        extracted = _extract_result_ids(_read_json(result_path), query)
    finally:
        # A full pass performs ~300 executions; the caller owns each result file and only
        # needs the situation set, so release it immediately rather than filling scratch.
        discard_result_file(result_path)
    if extracted and store_dir is not None:
        # Guarded: a store write failing must not lose a result already in hand, and the audit's
        # own outputs do not depend on it.
        try:
            result_store.store(store_dir, ocel_path, query, extracted)
        except Exception as exc:  # noqa: BLE001
            print(f"   warning: could not store result set: {exc}")
    return extracted[2] if extracted else None


# The always-violated bound. Every parent binding has fewer than 10**9 children whatever the log
# holds, so a `NumChilds min=IMPOSSIBLE_MIN` constraint flags all of them. Constraints flag without
# filtering, so the population is the no-op baseline's.
IMPOSSIBLE_NUMCHILDS_MIN = 10 ** 9
FLAG_EVERYTHING_EDGE = "ZZFLAGALL"
_EMPTY_BOX = {
    "newEventVars": {}, "newObjectVars": {}, "filters": [], "sizeFilters": [],
    "constraints": [], "evVarLabels": {}, "obVarLabels": {},
}


def _build_flag_everything_query(gold_query: dict) -> Optional[dict]:
    """The second degenerate strategy: bind the natural population and flag all of it.

    The no-op baseline strips the constraints, so it can never flag anything and its `violation`
    floor is 0 on every entry -- which made a model scoring 0.1360 there look like it had positive
    skill. It does not; it is far below what flagging everything achieves. A floor has to be the
    best score a strategy that ignores the question can reach, and this is the other one.

    Constructed as the no-op baseline plus an empty child and a `NumChilds` bound nothing can
    satisfy. **Its result is not obtained by executing this** -- `_flag_everything_situations`
    derives it from the stored baseline set for free. This function exists so the strategy is a
    query someone can inspect and re-run rather than an assertion, and it is what validated the
    derivation: executed against one entry of each of the eleven logs, the executed situation set
    equalled the derived one 11/11 (`Thesis/notes/11-evaluation-readiness.md` §27).

    Returns:
        The flag-everything query, or None when gold has no nodes.
    """
    base = _build_baseline_query(gold_query)
    if base is None:
        return None
    query = copy.deepcopy(base)
    root = query["nodes"][0]
    root["Box"][1] = [1]
    query["nodes"].append({"Box": [copy.deepcopy(_EMPTY_BOX), []]})
    query["edgeNames"] = [[[0, 1], FLAG_EVERYTHING_EDGE]]
    root["Box"][0].setdefault("constraints", []).append({
        "type": "SizeFilter",
        "filter": {"type": "NumChilds", "child_name": FLAG_EVERYTHING_EDGE,
                   "min": IMPOSSIBLE_NUMCHILDS_MIN, "max": None},
    })
    return query


def _flag_everything_situations(baseline_situations: Optional[set]) -> Optional[set]:
    """The flag-everything result, derived from the no-op baseline's rather than executed.

    Same population, every binding violated. Verified against real execution on all eleven logs;
    see `_build_flag_everything_query`.
    """
    if baseline_situations is None:
        return None
    return {(events, objects, True) for events, objects, _violated in baseline_situations}


def _degenerate_floors(gold_situations: set, baseline_situations: Optional[set]) -> tuple:
    """`(floors, detail)` -- the trivial floor per variant, and the strategies behind it.

    The floor is the best score obtainable **without understanding the question**, so it is the
    max over degenerate strategies, not the score of one arbitrarily chosen strategy. Using the
    no-op baseline alone understated it: flag-everything beats it on 54 of the 135 discriminating
    entries, and the strict floor is 0.7250 rather than 0.5148.

    `floors[variant]` is that max and is what skill is measured against; `detail[variant]` keeps
    both components so the choice is auditable. A variant undefined on this entry -- `violation`
    where gold flags nobody -- stays None in both, which is what makes the skill undefined
    downstream rather than silently zero.
    """
    strategies = {
        "noop": baseline_situations,
        "flagall": _flag_everything_situations(baseline_situations),
    }
    floors, detail = {}, {}
    for name, project in SIMILARITY_VARIANTS.items():
        scores = {}
        for strategy, situations in strategies.items():
            if situations is None:
                scores[strategy] = None
                continue
            projected = project(gold_situations, situations)
            scores[strategy] = (None if projected is None
                                else _set_metrics(*projected).get("f1"))
        usable = [v for v in scores.values() if v is not None]
        floors[name] = max(usable) if usable else None
        detail[name] = scores
    return floors, detail


def _baseline_floors(gold_situations: set, baseline_situations: Optional[set]) -> dict:
    """The no-op baseline's floor per variant. Superseded by `_degenerate_floors` as *the* floor.

    Kept because it is the determinism anchor: `baseline_f1` and this must keep reproducing the
    archived pre-floor audit exactly, which is how a re-run is shown not to have moved anything it
    should not have.
    """
    if baseline_situations is None:
        return {name: None for name in SIMILARITY_VARIANTS}
    floors = {}
    for name, project in SIMILARITY_VARIANTS.items():
        projected = project(gold_situations, baseline_situations)
        scored = None if projected is None else _set_metrics(*projected)
        floors[name] = (scored or {}).get("f1")
    return floors


def rebuild_floors(testset_path: str, output_dir: Path, store_dir: Optional[Path] = None) -> dict:
    """Recompute every floor from the stored situation sets. No `ocpq_cli`, no re-execution.

    The audit's expensive half is executing 338 queries; its floors are a pure function of the
    gold and no-op-baseline situation sets, both of which are in the result store. So a change to
    the floor definition -- adding a degenerate strategy, adding a variant -- costs minutes rather
    than the hour a full pass takes. This is the audit's counterpart to `--recompute`.

    Verdicts, `root_situations`, `root_violations` and `duration_sec` are carried over untouched:
    they are execution facts, and rewriting them here would silently launder a stale run into a
    fresh-looking one.

    Args:
        testset_path: Testset the summary was built from; refuses to run against a different one.
        output_dir: Directory holding `testset_discriminability.json`; rewritten in place.
        store_dir: Result store to read (default `<output_dir>/_discriminability_store`).

    Returns:
        The rewritten summary.

    Raises:
        FileNotFoundError: If the summary or the store is missing.
        ValueError: If the summary was built from a different testset.
    """
    summary_path = output_dir / "testset_discriminability.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No testset_discriminability.json in {output_dir}")
    store = Path(store_dir) if store_dir else output_dir / "_discriminability_store"
    if not store.is_dir():
        raise FileNotFoundError(
            f"No result store at {store}. Floors can only be rebuilt from stored situation "
            "sets; run a full audit once to populate it.")

    previous = _read_json(str(summary_path))
    if previous.get("testset_path") != testset_path:
        raise ValueError(
            f"summary was built from {previous.get('testset_path')!r}, not {testset_path!r}")

    testset = _read_json(testset_path)
    proposals_path = output_dir / "testset_discriminability_proposals.json"
    proposals = _read_json(str(proposals_path)) if proposals_path.exists() else {}

    results, unavailable, changed = [], 0, 0
    for record in previous.get("entries", []):
        index = record.get("index")
        entry = testset[index] if isinstance(index, int) and index < len(testset) else None
        if entry is None or entry.get("nl_query") != record.get("nl_query"):
            results.append(record)
            unavailable += 1
            continue
        gold = result_store.load(store, entry["ocel_path"], entry["ocpq_query"])
        baseline_query = _build_baseline_query(entry["ocpq_query"])
        baseline = (result_store.load(store, entry["ocel_path"], baseline_query)
                    if baseline_query else None)
        if gold is None:
            # No stored gold means no floor can be derived. Left exactly as it was and counted,
            # never recomputed from a partial input.
            results.append(record)
            unavailable += 1
            continue
        gold_situations = gold[2]
        baseline_situations = baseline[2] if baseline else None
        before = json.dumps(record.get("baseline_f1_variants"), sort_keys=True)
        floors, detail = _degenerate_floors(gold_situations, baseline_situations)
        record["baseline_f1_flagall"] = (detail.get("strict") or {}).get("flagall")
        record["trivial_floor"] = floors.get("strict")
        record["baseline_f1_variants"] = floors
        record["baseline_f1_variants_detail"] = detail
        if json.dumps(floors, sort_keys=True) != before:
            changed += 1
        results.append(record)

    summary = _build_summary(testset_path, results, proposals)
    _write_json(summary_path, summary)
    print(f"rebuilt floors for {len(results)} entries from {store}")
    print(f"  changed: {changed}   unavailable (left as-is): {unavailable}")
    print(f"wrote {summary_path}")
    return summary


def _log_selected(ocel_path: str, log_filter: str) -> bool:
    """Whether an entry's log matches the filter. A leading `!` negates the match."""
    negate = log_filter.startswith("!")
    needle = log_filter[1:] if negate else log_filter
    hit = needle.lower() in ocel_path.lower()
    return not hit if negate else hit


def _build_summary(testset_path: str, results: list, proposals: dict) -> dict:
    """Aggregate per-entry records into the summary payload written to disk."""
    ordered = sorted(results, key=lambda r: r["index"])
    nondisc = [r for r in ordered if r["verdict"] in NON_DISCRIMINATING_VERDICTS]
    failed = [r for r in ordered if r["verdict"] == VERDICT_EXEC_FAILED]

    def _mean_floor(rows):
        vals = [r["baseline_f1"] for r in rows if r.get("baseline_f1") is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    return {
        "testset_path": testset_path,
        "total": len(ordered),
        "ok": len(ordered) - len(nondisc) - len(failed),
        "non_discriminating": len(nondisc),
        "execution_failed": len(failed),
        "by_verdict": {
            verdict: sum(1 for r in ordered if r["verdict"] == verdict)
            for verdict in sorted({r["verdict"] for r in ordered})
        },
        "non_discriminating_indices": [r["index"] for r in nondisc],
        # The floor, per stratum. A trivial no-op query already scores this much against gold,
        # so any model number has to be read against it. On the non-discriminating stratum this
        # should sit at ~1.0, which is the quantitative statement of why those entries cannot be
        # scored by result comparison alone.
        "mean_baseline_f1": {
            "overall": _mean_floor(ordered),
            "discriminating": _mean_floor([r for r in ordered
                                           if r["verdict"] == VERDICT_OK]),
            "non_discriminating": _mean_floor(nondisc),
        },
        # Deprecated aliases, kept so notes and scripts written against the older naming stay
        # true. "degenerate" described these entries when this script was a filter; it is wrong
        # now that they are a reported stratum.
        "degenerate": len(nondisc),
        "degenerate_indices": [r["index"] for r in nondisc],
        "proposals_available": len(proposals),
        "entries": ordered,
    }


def audit(
    testset_path: str,
    output_dir: Path,
    limit: int = 0,
    log_filter: str = "",
    resume: bool = False,
) -> dict:
    """Execute every gold query and classify how well it discriminates.

    Args:
        testset_path: Path to the testset JSON.
        output_dir: Directory the summary and proposals are written to.
        limit: Audit only the first N entries (0 = all).
        log_filter: Case-insensitive substring; only entries whose `ocel_path` matches
            are audited. Lets the ~1 h `bpic2017` entries run separately from the rest.
        resume: Reuse verdicts already present in the summary file instead of re-executing
            them. A full pass takes over an hour, so a crash must not cost the whole run.

    Returns:
        The summary dict that was written to disk.
    """
    testset = _read_json(testset_path)
    if limit:
        testset = testset[:limit]

    scratch = output_dir / "_discriminability_queries"
    scratch.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "testset_discriminability.json"
    proposals_path = output_dir / "testset_discriminability_proposals.json"
    # Gold and no-op baseline result sets, so a future metric variant costs no execution. Sits
    # beside the summary rather than in `eval_runs/` proper: it belongs to this audit, and an
    # audit written to a scratch --output-dir must not write into the promoted store.
    store = output_dir / "_discriminability_store"

    results = []
    proposals = {}
    done: dict = {}
    if resume and summary_path.exists():
        previous = _read_json(str(summary_path))
        if previous.get("testset_path") == testset_path:
            done = {r["index"]: r for r in previous.get("entries", [])}
            proposals = _read_json(str(proposals_path)) if proposals_path.exists() else {}
            print(f"resuming: {len(done)} entries already audited")

    # Seed the checkpoint with everything already known, validated against the CURRENT testset.
    # Without this, an interrupted run destroys completed work: `results` starts empty and only
    # grows as the walk re-reaches each index, so a checkpoint written at index 57 contains 57
    # entries and the file loses every cached row beyond it. That cost a completed 159-entry
    # pass more than once. Keyed by index so a fresh result simply replaces its cached row.
    results_by_index = {
        i: r for i, r in done.items()
        if i < len(testset) and testset[i].get("nl_query") == r.get("nl_query")
    }

    # A cached row from before the per-variant floors existed carries none, and resuming onto it
    # would produce a summary that is silently half-populated -- some entries with matched floors
    # and some without, from one apparently complete run. Re-execute exactly those rows. This is
    # what makes `--resume` usable to *extend* an existing audit rather than only to finish an
    # interrupted one.
    _stale = [i for i, r in results_by_index.items() if "baseline_f1_variants" not in r]
    if _stale:
        print(f"resuming: {len(_stale)} cached entries predate the per-variant floors "
              "and will be re-executed")
        for i in _stale:
            del results_by_index[i]

    # A cached row is keyed by index and nl_query, so a gold query CORRECTED under an unchanged
    # question would keep its old verdict and floor (testset #25, 2026-09-22). The store is
    # content-addressed by (log, gold query): a row is current only while the store holds a blob
    # for the gold query the testset has NOW.
    _regolded = [i for i in results_by_index
                 if not result_store.blob_path(store, testset[i]["ocel_path"],
                                               testset[i]["ocpq_query"]).exists()]
    if _regolded:
        print(f"resuming: {len(_regolded)} cached entries have a gold query the store has never "
              f"executed and will be re-executed: {sorted(_regolded)}")
        for i in _regolded:
            del results_by_index[i]

    for index, entry in enumerate(testset):
        nl_query = entry.get("nl_query", "")
        ocel_path = entry.get("ocel_path")
        gold_query = entry.get("ocpq_query")
        if not ocel_path or not gold_query:
            continue
        if log_filter and not _log_selected(str(ocel_path), log_filter):
            continue  # already carried in results_by_index if it was audited before
        if index in results_by_index:
            print(f"   [{index:3d}] cached {results_by_index[index]['verdict']}")
            continue

        started = time.time()
        situations = _run_and_extract(ocel_path, gold_query,
                                      scratch / f"gold_{index:03d}.json", store)

        baseline_query = _build_baseline_query(gold_query)
        baseline_situations = (
            _run_and_extract(ocel_path, baseline_query,
                             scratch / f"baseline_{index:03d}.json", store)
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

        # The trivial floor: what the no-op baseline query already scores against gold under the
        # evaluation's own metric. This is what makes "this entry is uninformative" a number
        # rather than an assertion -- on a fully-compliant entry the floor is ~1.0, so a model
        # that expresses nothing still earns full marks, and any model score has to be read
        # against it. Scored with eval_testset._set_metrics so the floor and the reported metric
        # are computed by the same code (CLAUDE.md 2.5).
        baseline_scores = _set_metrics(situations, baseline_situations)
        # The second degenerate strategy, derived rather than executed (verified 11/11 across
        # every log). Scored under the same projections so the floor and the metric agree.
        floors, floor_detail = _degenerate_floors(situations, baseline_situations)
        flagall_scores = (floor_detail.get("strict") or {}).get("flagall")

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
            "baseline_f1": (baseline_scores or {}).get("f1"),
            "baseline_precision": (baseline_scores or {}).get("precision"),
            "baseline_recall": (baseline_scores or {}).get("recall"),
            # `baseline_f1` above is the NO-OP strategy's strict score and must never move: it is
            # the determinism anchor against the archived audit. The operative floor is the max
            # over degenerate strategies below -- see `_degenerate_floors`.
            "baseline_f1_flagall": flagall_scores,
            "trivial_floor": floors.get("strict"),
            # Per-variant floor, so a relaxed model F1 is never read against the strict floor.
            "baseline_f1_variants": floors,
            "baseline_f1_variants_detail": floor_detail,
            "duration_sec": round(time.time() - started, 3),
        }
        results_by_index[index] = record

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

        # Checkpoint after every entry: a crash mid-run then costs one query, not the run.
        _write_json(summary_path, _build_summary(testset_path,
                                                 list(results_by_index.values()),
                                                 proposals))
        _write_json(proposals_path, proposals)

    summary = _build_summary(testset_path, list(results_by_index.values()), proposals)
    _write_json(summary_path, summary)
    _write_json(proposals_path, proposals)

    floors = summary["mean_baseline_f1"]
    print()
    print(f"total              : {summary['total']}")
    print(f"discriminating     : {summary['ok']}   (trivial floor {floors['discriminating']})")
    print(f"non-discriminating : {summary['non_discriminating']}   "
          f"(trivial floor {floors['non_discriminating']})")
    print(f"execution failed   : {summary['execution_failed']}")
    print(f"auto-proposed fixes: {summary['proposals_available']}")
    print()
    print("Non-discriminating entries are NOT defects: their gold query is correct and the log "
          "simply complies.")
    print("They are a reported stratum, not a deletion list. The floor is what a no-op query "
          "already scores.")
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
    parser.add_argument("--testset", default="data/testset.json")
    parser.add_argument("--output-dir", default="eval_runs")
    parser.add_argument("--limit", type=int, default=0, help="Audit only the first N entries.")
    parser.add_argument(
        "--log-filter",
        default="",
        help="Only audit entries whose ocel_path contains this substring.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep verdicts already in the summary file instead of re-executing them.",
    )
    parser.add_argument(
        "--floors-only",
        action="store_true",
        help="Recompute every floor from the stored situation sets and rewrite the summary in "
             "place. No ocpq_cli, no re-execution; verdicts and situation counts are carried "
             "over untouched. Use after changing the floor definition or adding a variant.",
    )
    parser.add_argument(
        "--store-dir", default=None,
        help="Result store for --floors-only (default: <output-dir>/_discriminability_store).",
    )
    args = parser.parse_args()

    if args.floors_only:
        rebuild_floors(args.testset, Path(args.output_dir),
                       Path(args.store_dir) if args.store_dir else None)
        return

    audit(args.testset, Path(args.output_dir), args.limit, args.log_filter, args.resume)


if __name__ == "__main__":
    main()
