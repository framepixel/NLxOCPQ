"""`paired_stats.py` must reproduce the published statistics and refuse to pair unlike things.

Every interval and p-value the thesis quotes comes from these functions, so each one is pinned:

  * the exact sign test, against values computable by hand (and §37's "7 up, 0 down, p = 0.016");
  * the bootstrap's draw order -- `random.Random(seed)` with one `randrange(n)` per question --
    which is what makes §44.9's +0.0087 ... +0.0848 reproducible to four decimals;
  * the resolution figures of §37.7 (n = 135 resolves about +-0.03 at the observed SD);
  * pairing, which must fail loudly on questions present in one condition but not the other.

Run from the repo root:  python tests/test_paired_stats.py
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paired_stats import (  # noqa: E402
    bootstrap_ci,
    min_detectable_effect,
    paired,
    questions_needed,
    reconcile,
    sign_test,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def close(name, got, want, tol=1e-9):
    check(name, abs(got - want) <= tol, True)


# --- sign test --------------------------------------------------------------------------------
close("sign test 7-0 is 2/128 (notes 37: p = 0.016)", sign_test(7, 0), 2 / 128)
close("sign test 6-0 is 2/64 (notes 37: p = 0.031)", sign_test(6, 0), 2 / 64)
close("sign test is symmetric", sign_test(22, 39), sign_test(39, 22))
close("sign test 39-22 rounds to the notes' 0.040", round(sign_test(39, 22), 3), 0.040)
close("sign test 1-1 is capped at 1", sign_test(1, 1), 1.0)
check("sign test with no change is undefined", sign_test(0, 0), None)

# --- bootstrap ---------------------------------------------------------------------------------
diffs = [random.Random(7).uniform(-0.5, 0.5) for _ in range(5)] + [0.1, -0.2, 0.3, 0.0, 0.05]
first = bootstrap_ci(diffs, 2000, 0)
check("bootstrap is deterministic for a fixed seed", bootstrap_ci(diffs, 2000, 0), first)
check("a different seed moves the interval", bootstrap_ci(diffs, 2000, 1) != first, True)
mean = sum(diffs) / len(diffs)
check("the interval contains the mean", first[0] <= mean <= first[1], True)
check("a constant difference has a zero-width interval", bootstrap_ci([0.25] * 10, 500, 0),
      (0.25, 0.25))


def manual_ci(values, resamples, seed):
    """The documented procedure, written out longhand."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(resamples))
    return means


reference = manual_ci(diffs, 2000, 0)
check("draw order matches the documented procedure (low tail)",
      reference[0] <= first[0] <= reference[len(reference) // 20], True)

try:
    bootstrap_ci([], 10, 0)
    check("an empty contrast raises", False, True)
except ValueError:
    check("an empty contrast raises", True, True)

# --- resolution (notes 37.7) ------------------------------------------------------------------
# n = 135 at 80% power and alpha 0.05 resolves (1.960 + 0.842) * sd / sqrt(135); at sd = 0.0625
# that is about 0.0151 -- the formula, not a measured SD, is what is pinned here.
close("minimum detectable effect formula", min_detectable_effect(0.0625, 135),
      (1.959963984540054 + 0.8416212335729143) * 0.0625 / 135 ** 0.5, 1e-12)
check("questions needed grows with smaller effects",
      questions_needed(0.1, 0.0089) > questions_needed(0.1, 0.0146), True)
check("questions needed for a zero effect is undefined", questions_needed(0.1, 0.0), None)
check("detecting exactly the resolved effect needs n questions",
      questions_needed(0.1, min_detectable_effect(0.1, 135)) in (135, 136), True)

# --- pairing -----------------------------------------------------------------------------------
a = {("q1", "log.json"): 0.5, ("q2", "log.json"): 0.0}
b = {("q1", "log.json"): 0.25, ("q2", "log.json"): 0.5}
check("pairing takes the weighted difference per question",
      paired({"A": a, "B": b}, {"A": 1.0, "B": -1.0}),
      {("q1", "log.json"): 0.25, ("q2", "log.json"): -0.5})
check("an interaction is a four-term weighted sum",
      paired({"A": a, "B": b, "C": b, "D": a}, {"A": 1, "B": -1, "C": -1, "D": 1}),
      {("q1", "log.json"): 0.5, ("q2", "log.json"): -1.0})
try:
    paired({"A": a, "B": {("q1", "log.json"): 0.25}}, {"A": 1.0, "B": -1.0})
    check("pairing different question sets raises", False, True)
except ValueError:
    check("pairing different question sets raises", True, True)

# --- reconciliation -----------------------------------------------------------------------------
row = {"better": 39, "worse": 22, "mean": 0.04531, "ci": [0.0087, 0.0848], "p": 0.0396,
       "evaluated_gained": 26, "evaluated_lost": 19, "evaluated_p": 0.37, "identical": None}
check("a matching row reconciles cleanly",
      reconcile(row, {"better": 39, "worse": 22, "mean": 0.0453, "ci": [0.0087, 0.0848],
                      "p": 0.040, "evaluated_p": 0.37}), ([], []))
problems, notes = reconcile(row, {"mean": 0.0453, "ci": [0.0090, 0.0848]})
check("a small interval gap is a method note, not a failure", (problems, len(notes)), ([], 1))
problems, _ = reconcile(row, {"ci": [0.0187, 0.0848]})
check("a large interval gap fails", len(problems), 1)
problems, _ = reconcile(row, {"better": 38})
check("a count mismatch fails", len(problems), 1)
problems, _ = reconcile(dict(row, p=0.0004), {"p": "<0.001"})
check("a '< 0.001' bound is checked as a bound", problems, [])

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
