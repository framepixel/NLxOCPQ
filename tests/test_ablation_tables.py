"""`ablation_tables.py` must count what the notes counted, from records it does not rescore.

  * shape adoption is gold's number of boxes in a PARSED prediction, and the conditional F1 is over
    the discriminating parsed records only (notes 11 §25.5);
  * "newly valid" means schema-invalid in the first rung, schema-valid here, on a discriminating
    question (notes 11 §31.3's 32 / 0.1427 / 25 / 2);
  * the arity split groups questions by gold's root arity as the run scored it.

Runs in the evaluation environment (`eval_testset` is imported for the root arity).

Run from the repo root:  python tests/test_ablation_tables.py
"""
import os
import sys
import tempfile
from pathlib import Path

os.environ["THESIS_TABLES_DIR"] = tempfile.mkdtemp()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ablation_tables as at  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def box(events=0, objects=1, children=()):
    return {"Box": [{"newEventVars": {f"e{i}": [] for i in range(events)},
                     "newObjectVars": {f"o{i}": [] for i in range(objects)}}, list(children)]}


def tree(*boxes):
    return {"nodes": list(boxes)}


def rec(question, f1=None, valid=True, disc=True, gold=None, prediction=None, tokens=100):
    return {"nl_query": question, "ocel_path": "/x/log.json", "config_id": "c", "repeat": 1,
            "discriminating": disc, "status": "success" if f1 is not None else "schema_invalid",
            "similarity": None if f1 is None else {"combined": {"f1": f1}},
            "execution": {"success": f1 is not None, "evaluated": f1 is not None},
            "metrics": {"schema_valid": valid}, "gold_query": gold or tree(box()),
            "prediction": prediction, "llm": {"completion_tokens": tokens, "calls": 1}}


two = tree(box(children=[1]), box())
one = tree(box())
rows = [rec("a", 1.0, gold=two, prediction=two),        # adopted, exact
        rec("b", 0.2, gold=two, prediction=one),        # parsed, not adopted
        rec("c", None, valid=False, gold=two),          # unparsed: no shape at all
        rec("d", 0.5, disc=False, gold=one, prediction=one)]  # adopted, not discriminating
level = at.hint_level(rows)
check("only parsed predictions count as trees", level["parsed"], 3)
check("adoption is over every parsed tree", level["adopted"], 2)
check("conditional F1 is over discriminating parsed records", level["f1_adopted"], (1.0, 1))
check("...and its complement excludes the unparsed", level["f1_not_adopted"], (0.2, 1))

first = [rec("a", None, valid=False), rec("b", None, valid=False), rec("c", None, valid=False),
         rec("d", 0.4)]
later = [rec("a", 0.0), rec("b", 1.0), rec("c", 0.3, disc=False), rec("d", 0.4)]
newly = at.decoding_level(later, first)["newly_valid"]
check("newly valid: invalid before, valid now, discriminating only", newly["n"], 2)
check("...with their zero and exact counts", (newly["zero"], newly["exact"]), (1, 1))
check("the first rung has no newly valid answers", at.decoding_level(first, None)["newly_valid"],
      None)

split = at.split_means([rec("p", 0.6, gold=tree(box(0, 1))), rec("q", 0.2, gold=tree(box(1, 1)))])
check("the split follows gold's root arity",
      (split["majority"], split["other"], split["all"]), ((0.6, 1), (0.2, 1), (0.4, 2)))

check("a share of nothing prints only the count", at._count(0, 0), "0")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
