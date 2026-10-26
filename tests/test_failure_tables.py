"""`failure_tables.py` must render the analyses' numbers, not recompute them.

  * every count and share in the grounding table comes from the anatomy JSON as written;
  * the depth table's F1 is the stored F1 sum over the questions at that depth;
  * a clearing combination is named by its classes, in words.

Run from the repo root:  python tests/test_failure_tables.py
"""
import os
import sys
import tempfile
from pathlib import Path

tables_dir = tempfile.mkdtemp()
os.environ["THESIS_TABLES_DIR"] = tables_dir
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import failure_tables as ft  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def arm(wrong, objects):
    return {"n": 20, "executed": 16,
            "buckets": {"unscored": 4, "exact": 2, "partial": 3, "over_broad": 2,
                        "over_narrow": 1, "disjoint": wrong},
            "entities": {"object_types": objects, "event_activities": 2, "both": 1, "n": wrong},
            "composition": {"node_count": 1, "filter_multiset": 2, "constraint_multiset": 0,
                            "n": wrong, "gold_boxes": 2.5, "pred_boxes": 1.5,
                            "constructs": {"SAT": {"gold": 7, "predicted": 1}}},
            "shape": {"shape_matched": {"n": 3, "mean_f1": 0.5},
                      "shape_mismatched": {"n": 2, "mean_f1": 0.25}},
            "depth": {"1": {"n": 4, "executed": 3, "f1_sum": 2.0, "exact": 1},
                      "2": {"n": 2, "executed": 1, "f1_sum": 0.5, "exact": 0},
                      "3": {"n": 2, "executed": 2, "f1_sum": 1.0, "exact": 1},
                      "4+": {"n": 1, "executed": 0, "f1_sum": 0.0, "exact": 0}}}


anatomy = {"conditions": {ft.PRIMARY: arm(8, 6), ft.BASE_PAIR: arm(4, 1)}}

text = ft.grounding(anatomy, []).read_text(encoding="utf-8")
check("answers without a result are counted", "No result & 4 & 4 \\\\" in text, True)
check("partly correct pools partial, over-broad and over-narrow",
      "F1 $<$ 1) & 6 & 6 \\\\" in text, True)
check("the wrong answers close the outcome rows",
      "Wrong (disjoint from gold) & 8 & 4 \\\\" in text, True)
check("a share prints its count and percentage", "Object types & 6 (75.0\\%) & 1 (25.0\\%)" in text,
      True)
check("SAT is gold / predicted", "\\texttt{SAT} constraints & 7 / 1 & 7 / 1" in text, True)
check("shape rows carry their n", "With gold's number of boxes & 0.5000 (3)" in text, True)

text = ft.depth(anatomy, []).read_text(encoding="utf-8")
check("depth F1 is the F1 sum over the questions",
      "1 & 4 & 75\\% & 0.5000 & 25.0\\%" in text, True)
check("the deepest bin is labelled in words", "4 or more & 1 & 0\\%" in text, True)

check("a clearing pair is named in words", ft._pair_label("link_mismatch+no_result"),
      "different root filters and no result")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
