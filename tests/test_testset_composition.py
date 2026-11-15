"""`testset_composition.py` must describe the testset with the evaluator's own definitions.

Pinned here:
  * `skill` is `eval_testset._skill_score`: the two-sided normalisation, bounded in [-1, 1],
    continuous at the floor, undefined at floor 1.0, and -1 for F1 0 whatever the floor;
  * a compliance query is one where ANY box carries constraints (what §11.2's 154 / 15 measured),
    and the root-only variant is the stricter reading;
  * the root of a tree is the box no box lists as a child.

Run from the repo root:  python tests/test_testset_composition.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from testset_composition import has_constraints, roots, skill  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


check("F1 at the floor scores 0", skill(0.6, 0.6), 0.0)
check("F1 1 scores 1", skill(1.0, 0.3), 1.0)
check("F1 0 scores -1 whatever the floor", (skill(0.0, 0.3), skill(0.0, 0.9998)), (-1.0, -1.0))
check("the upper branch divides by the headroom", round(skill(0.85, 0.5), 6), 0.7)
check("the lower branch divides by the floor", round(skill(0.25, 0.5), 6), -0.5)
check("floor 1.0 is undefined", skill(1.0, 1.0), None)
check("a floor of 0 never reaches the lower branch", skill(0.0, 0.0), 0.0)


def box(constraints=False):
    return {"newEventVars": {}, "newObjectVars": {}, "filters": [], "sizeFilters": [],
            "constraints": [{"type": "SAT", "child_names": ["A"]}] if constraints else []}


child_first = {"nodes": [{"Box": [box(constraints=True), []]}, {"Box": [box(), [0]]}]}
check("the root is the unreferenced box, here node 1", roots(child_first), [1])
check("any-box compliance sees the child's constraints", has_constraints(child_first), True)
check("root-only compliance does not", has_constraints(child_first, root_only=True), False)

# The skill label every table prints (analysis_common.SKILL_QUESTIONS) must match the audit: skill
# is defined only where the trivial floor is below 1.0 (review 2026-11-11).
import analysis_common as ac  # noqa: E402
check("skill is defined on 134 questions, as the tables say", ac.skill_question_count(),
      ac.SKILL_QUESTIONS)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
