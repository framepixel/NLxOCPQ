"""`schema_arm_identity.py` must pair the two validation-repair arms correctly.

Pinned here: records pair only when model, strategy, question and draw all match; only pairs in
which BOTH arms made one LLM call enter the identity check (a repair call makes the arms differ by
design); and a mismatch in either compared field is not counted as identical.

Run from the repo root:  python tests/test_schema_arm_identity.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from schema_arm_identity import identity  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def rec(question, enforced, calls=1, prompt=100, chars=50, model="m", strategy="s", f1=None):
    return {"model_label": model, "prompt_strategy": strategy, "nl_query": question,
            "ocel_path": "/x/log.json", "repeat": 1, "schema_enforced": enforced,
            "llm": {"calls": calls, "prompt_tokens": prompt, "completion_chars": chars},
            "similarity": None if f1 is None else {"combined": {"f1": f1}}}


records = [
    rec("q1", True), rec("q1", False),                       # identical single-call pair
    rec("q2", True, chars=60), rec("q2", False, chars=50),   # single-call, different length
    rec("q3", True, calls=2), rec("q3", False),              # repair ran on one arm
    rec("q4", True, model="other"), rec("q4", False),        # different model: no pair
]
report = identity(records)
check("pairs need model, strategy, question and draw to match", report["pairs"], 3)
check("only pairs with one call on both arms are compared", report["single_call_pairs"], 2)
check("identical prompts are counted", report["identical_prompt_tokens"], 2)
check("a different completion length is not identical", report["identical_completion_chars"], 1)
check("equal call counts", report["equal_call_counts"], 2)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
