#!/usr/bin/env python3
"""Auto-detect OCPQ building-block labels for every testset entry.

Reads testset.json, inspects each entry's ocpq_query structure, and writes
testset_bb_labels.json — a sidecar file that maps nl_query to a sorted list of
BB IDs (e.g. ["F1", "F4", "C3", "T2"]).  The detection rules are fully
deterministic from the OCPQ JSON structure; no manual labelling is needed.

Usage:
    python tag_testset_bb.py [--testset testset.json] [--out testset_bb_labels.json]
"""

import argparse
import json
from collections import Counter


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _all_nodes(query: dict) -> list[dict]:
    return [node["Box"][0] for node in query.get("nodes", []) if "Box" in node]


def _all_filters(query: dict) -> list[dict]:
    """All filters appearing anywhere (direct in filters list, or wrapped in constraint)."""
    result = []
    for d in _all_nodes(query):
        result.extend(d.get("filters", []))
        for c in d.get("constraints", []):
            inner = c.get("filter")
            if isinstance(inner, dict):
                result.append(inner)
    return result


def _all_size_filters(query: dict) -> list[dict]:
    """All size-filter dicts (bare sizeFilters list + SizeFilter-wrapped in constraints)."""
    result = []
    for d in _all_nodes(query):
        result.extend(d.get("sizeFilters", []))
        for c in d.get("constraints", []):
            if c.get("type") == "SizeFilter":
                inner = c.get("filter")
                if isinstance(inner, dict):
                    result.append(inner)
    return result


def _all_constraints(query: dict) -> list[dict]:
    result = []
    for d in _all_nodes(query):
        result.extend(d.get("constraints", []))
    return result


def _longest_path(query: dict) -> int:
    """Length (in nodes) of the longest root-to-leaf path in the binding-box tree."""
    nodes = query.get("nodes", [])
    edges = query.get("edgeNames", [])
    if not nodes:
        return 0
    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for (src, dst), _ in edges:
        children[src].append(dst)

    def depth(node: int) -> int:
        kids = children.get(node, [])
        return 1 + (max(depth(k) for k in kids) if kids else 0)

    return depth(0)


def _has_branching(query: dict) -> bool:
    """True if any node has ≥ 2 distinct children (T4 branching)."""
    nodes = query.get("nodes", [])
    for i, node in enumerate(nodes):
        children = node["Box"][1] if "Box" in node and len(node["Box"]) > 1 else []
        if len(children) >= 2:
            return True
    return False


def detect_bb_labels(query: dict) -> list[str]:
    """Return a sorted list of BB IDs present in *query*."""
    labels: set[str] = set()
    nodes = _all_nodes(query)
    all_f = _all_filters(query)
    all_sf = _all_size_filters(query)
    all_c = _all_constraints(query)

    # --- Variables ---
    for d in nodes:
        for vs in d.get("newEventVars", {}).values():
            if isinstance(vs, list):
                if len(vs) == 1:
                    labels.add("V1")
                elif len(vs) > 1:
                    labels.add("V3")
        if len(d.get("newEventVars", {})) >= 2:
            labels.add("V5")
        for vs in d.get("newObjectVars", {}).values():
            if isinstance(vs, list):
                if len(vs) == 1:
                    labels.add("V2")
                elif len(vs) > 1:
                    labels.add("V4")
        if len(d.get("newObjectVars", {})) >= 2:
            labels.add("V6")

    # --- Filters ---
    for f in all_f:
        t = f.get("type")
        if t == "O2E":
            if f.get("qualifier") is None:
                labels.add("F1")
            else:
                labels.add("F3")
        elif t == "O2O":
            if f.get("qualifier") is None:
                labels.add("F2")
            else:
                labels.add("F3")
        elif t == "TimeBetweenEvents":
            mn = f.get("min_seconds", 0) or 0
            mx = f.get("max_seconds")
            if mx == 0:
                labels.add("F7")  # simultaneous
            elif mn > 0:
                labels.add("F6")  # min gap enforced
            elif mx is not None:
                labels.add("F5")  # deadline
            else:
                labels.add("F4")  # unbounded before
        elif t == "NotEqual":
            labels.add("F8")
        elif t == "EventAttributeValueFilter":
            labels.add("F9")
        elif t == "ObjectAttributeValueFilter":
            vf_type = (f.get("value_filter") or {}).get("type")
            vf_map = {"Float": "F10", "Integer": "F11", "Boolean": "F12",
                      "String": "F13", "Time": "F14"}
            if vf_type in vf_map:
                labels.add(vf_map[vf_type])
            at = (f.get("at_time") or {}).get("type")
            at_map = {"Always": "F15", "Sometime": "F16", "AtEvent": "F17"}
            if at in at_map:
                labels.add(at_map[at])
        elif t == "BasicFilterCEL":
            labels.add("F18")

    # --- Size filters ---
    for sf in all_sf:
        t = sf.get("type")
        if t == "NumChilds":
            if sf.get("min") is not None:
                labels.add("S1")
            if sf.get("max") is not None:
                labels.add("S2")
        elif t == "AdvancedCEL":
            labels.add("S3")

    # --- Constraints ---
    for c in all_c:
        t = c.get("type")
        if t == "Filter":
            labels.add("C1")
        elif t == "SizeFilter":
            labels.add("C2")
        elif t == "SAT":
            labels.add("C3")
        elif t == "ANY":
            labels.add("C4")
        elif t == "NOT":
            labels.add("C5")
        elif t == "OR":
            labels.add("C6")
        elif t == "AND":
            labels.add("C7")

    # --- Labels / output selectors ---
    for d in nodes:
        if d.get("labels"):
            labels.add("L1")
        if d.get("evVarLabels"):
            labels.add("L2")
        if d.get("obVarLabels"):
            labels.add("L3")

    # --- Tree structure ---
    n = len(query.get("nodes", []))
    longest = _longest_path(query)
    branching = _has_branching(query)
    if n == 1:
        labels.add("T1")
    if n == 2 and not branching:
        labels.add("T2")
    if n == 3 and not branching:
        labels.add("T3")
    if branching:
        labels.add("T4")
    if longest >= 4:
        labels.add("T5")

    # Canonical sort order matching taxonomy: V, F, S, C, L, T
    order = {b: i for i, b in enumerate([
        "V1","V2","V3","V4","V5","V6",
        "F1","F2","F3","F4","F5","F6","F7","F8","F9",
        "F10","F11","F12","F13","F14","F15","F16","F17","F18",
        "S1","S2","S3",
        "C1","C2","C3","C4","C5","C6","C7",
        "L1","L2","L3",
        "T1","T2","T3","T4","T5",
    ])}
    return sorted(labels, key=lambda x: order.get(x, 999))


# Canonical category order matching taxonomy: V, F, S, C, L, T
CATEGORY_ORDER = ["V", "F", "S", "C", "L", "T"]


def detect_category_labels(bb_labels: list[str]) -> list[str]:
    """Return the sorted set of BB categories (e.g. "F1" -> "F") present in *bb_labels*."""
    categories = {bb[0] for bb in bb_labels}
    order = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    return sorted(categories, key=lambda c: order.get(c, 999))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", default="data/testset.json")
    parser.add_argument("--out", default="data/testset_bb_labels.json")
    args = parser.parse_args()

    with open(args.testset, encoding="utf-8") as f:
        testset = json.load(f)

    results = []
    bb_coverage: Counter = Counter()
    category_coverage: Counter = Counter()

    for entry in testset:
        nl = entry["nl_query"]
        query = entry["ocpq_query"]
        if isinstance(query, str):
            query = json.loads(query)
        bbs = detect_bb_labels(query)
        categories = detect_category_labels(bbs)
        results.append({"nl_query": nl, "bb_labels": bbs, "category_labels": categories})
        for bb in bbs:
            bb_coverage[bb] += 1
        for cat in categories:
            category_coverage[cat] += 1

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote {len(results)} entries to {args.out}\n")

    # Coverage table
    all_bbs = [
        "V1","V2","V3","V4","V5","V6",
        "F1","F2","F3","F4","F5","F6","F7","F8","F9",
        "F10","F11","F12","F13","F14","F15","F16","F17","F18",
        "S1","S2","S3",
        "C1","C2","C3","C4","C5","C6","C7",
        "L1","L2","L3",
        "T1","T2","T3","T4","T5",
    ]
    print(f"{'BB':<6} {'Count':>5}  {'Bar'}")
    print("-" * 40)
    for bb in all_bbs:
        count = bb_coverage.get(bb, 0)
        bar = "#" * count
        flag = "  <- MISSING" if count == 0 else ""
        print(f"{bb:<6} {count:>5}  {bar}{flag}")

    missing = [bb for bb in all_bbs if bb_coverage.get(bb, 0) == 0]
    print(f"\nMissing BBs ({len(missing)}): {missing}")
    print(f"Total BBs covered: {len(all_bbs) - len(missing)}/{len(all_bbs)}")

    # Category coverage table
    category_names = {
        "V": "Variables", "F": "Filters", "S": "Size Filters",
        "C": "Constraints", "L": "Labels", "T": "Tree Structure",
    }
    print(f"\n{'Cat':<6} {'Name':<14} {'Count':>5}  {'Bar'}")
    print("-" * 50)
    for cat in CATEGORY_ORDER:
        count = category_coverage.get(cat, 0)
        bar = "#" * count
        flag = "  <- MISSING" if count == 0 else ""
        print(f"{cat:<6} {category_names[cat]:<14} {count:>5}  {bar}{flag}")


if __name__ == "__main__":
    main()
