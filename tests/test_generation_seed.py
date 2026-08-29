"""Per-sample generation seeding.

Two failure modes this guards against, both silent:

  * seeds that do NOT vary across repeats -> every sample identical -> self-consistency is
    voting on ten copies of one answer and the whole experiment is vacuous;
  * seeds derived from a counter rather than from record identity -> sample 7 of a 4-way
    case-shard split differs from sample 7 of a single job, so the run is not reproducible and
    the shard layout silently changes the result.

The GPU half -- that differing seeds actually produce differing text -- cannot run here and is
verification step 1 on the cluster.

Run from the repo root:  python tests/test_generation_seed.py
"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import LLM  # noqa: E402
from eval_testset import _record_key  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def derive(base, case_id, config_id, repeat, nl_query, ocel_path):
    """The derivation used in eval_testset's repeat loop, mirrored here."""
    if not base:
        return None
    key = _record_key(case_id, config_id, repeat, nl_query, ocel_path)
    return int(hashlib.sha256(f"{base}:{key}".encode("utf-8")).hexdigest()[:8], 16)


BASE, CFG, NL, LOG = 20260829, "arm|strategy|True", "Which orders were never paid?", "/L/om.json"

print("--- the setter ---")
LLM.set_generation_seed(1234)
check("round-trips", LLM.get_generation_seed(), 1234)
LLM.set_generation_seed(None)
check("None disables", LLM.get_generation_seed(), None)
LLM.set_generation_seed("77")
check("coerces to int", LLM.get_generation_seed(), 77)
LLM.set_generation_seed(None)

print("\n--- seeds vary across repeats (or the experiment is vacuous) ---")
seeds = [derive(BASE, 1, CFG, r, NL, LOG) for r in range(1, 11)]
check("10 repeats produce 10 distinct seeds", len(set(seeds)), 10)
check("none is None", all(s is not None for s in seeds), True)

print("\n--- stable across re-runs ---")
again = [derive(BASE, 1, CFG, r, NL, LOG) for r in range(1, 11)]
check("same inputs reproduce the same seeds", seeds, again)

print("\n--- independent of shard layout ---")
# case_id restarts at 1 in every case-shard, but the seed is derived from the ENTRY, so the same
# question in a different shard position must still seed identically.
in_half0 = derive(BASE, 1, CFG, 3, NL, LOG)
in_half1 = derive(BASE, 87, CFG, 3, NL, LOG)   # same question, different shard-local case number
check("same question, different case_id -> same seed", in_half0, in_half1)

print("\n--- different work gets different seeds ---")
check("different repeat", derive(BASE, 1, CFG, 1, NL, LOG) != derive(BASE, 1, CFG, 2, NL, LOG), True)
check("different question",
      derive(BASE, 1, CFG, 1, NL, LOG) != derive(BASE, 1, CFG, 1, "Other?", LOG), True)
check("different condition",
      derive(BASE, 1, CFG, 1, NL, LOG) != derive(BASE, 1, "other|s|True", 1, NL, LOG), True)
check("different base",
      derive(BASE, 1, CFG, 1, NL, LOG) != derive(BASE + 1, 1, CFG, 1, NL, LOG), True)
check("same question on a different log",
      derive(BASE, 1, CFG, 1, NL, LOG) != derive(BASE, 1, CFG, 1, NL, "/L/other.json"), True)

print("\n--- inertness: no base means no seeding ---")
check("base 0 -> None", derive(0, 1, CFG, 1, NL, LOG), None)
check("base None -> None", derive(None, 1, CFG, 1, NL, LOG), None)

print("\n--- seeds are in torch's accepted range ---")
check("all within [0, 2**32)", all(0 <= s < 2 ** 32 for s in seeds), True)

print("\n--- the greedy path is untouched ---")
src = (REPO / "src" / "LLM.py").read_text(encoding="utf-8")
i = src.index("if do_sample and generation_seed is not None:")
check("seeding is gated on do_sample", "do_sample and generation_seed is not None" in src[i:i + 60],
      True)
check("gate sits immediately before generate()",
      "local_model.generate(**generate_kwargs)" in src[i:i + 400], True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
