"""Tree depth (the T5 label) is measured from the root the scorer uses, not from node 0.

`tag_testset_bb._longest_path` walked from node 0 until 2026-09-22. A tree that lists a child before
its parent has its root elsewhere, and depth(0) then measures a subpath: a four-box chain rooted at
node 3 carried no T-label at all. It could only ever drop T5, never add it.

`_root_index` mirrors `eval_testset.root_node_index` instead of importing it (eval_testset imports
the tagger), so this also pins that the two agree on every gold tree -- a disagreement would score
depth from one node and the answer from another.

Run from the repo root:  python tests/test_structural_metric_root.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tag_testset_bb import _longest_path, _root_index, detect_bb_labels  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def box(children=()):
    return {"Box": [{"newEventVars": {}, "newObjectVars": {}, "filters": [], "sizeFilters": [],
                     "constraints": [], "evVarLabels": {}, "obVarLabels": {}}, list(children)]}


# A four-box chain 3 -> 2 -> 1 -> 0: the root is the LAST node.
reversed_chain = {"nodes": [box(), box([0]), box([1]), box([2])],
                  "edgeNames": [[[3, 2], "A"], [[2, 1], "B"], [[1, 0], "C"]]}
forward_chain = {"nodes": [box([1]), box([2]), box([3]), box()],
                 "edgeNames": [[[0, 1], "A"], [[1, 2], "B"], [[2, 3], "C"]]}

print("--- depth is measured from the structural root ---")
check("the root of a reversed chain is its last node", _root_index(reversed_chain), 3)
check("a reversed four-box chain is four deep (was 1: depth from node 0, a leaf)",
      _longest_path(reversed_chain), 4)
check("the same chain written forwards is also four deep", _longest_path(forward_chain), 4)
check("so both carry the same T-labels",
      sorted(l for l in detect_bb_labels(reversed_chain) if l.startswith("T")),
      sorted(l for l in detect_bb_labels(forward_chain) if l.startswith("T")))
check("single box", _longest_path({"nodes": [box()], "edgeNames": []}), 1)
check("no nodes", _longest_path({"nodes": [], "edgeNames": []}), 0)

print("\n--- the mirror agrees with the scorer's root on every gold tree ---")
# Imported late: eval_testset pulls in the whole pipeline.
from eval_testset import root_node_index  # noqa: E402

testset = json.loads((ROOT / "data" / "testset.json").read_text(encoding="utf-8"))
disagree = [i for i, row in enumerate(testset)
            if _root_index(row["ocpq_query"]) != root_node_index(row["ocpq_query"])]
check("tag_testset_bb._root_index == eval_testset.root_node_index on all gold", disagree, [])
for tree in (reversed_chain, forward_chain):
    check("...and on the synthetic chains", _root_index(tree), root_node_index(tree))

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> depth is measured from the root the answer is scored from")
