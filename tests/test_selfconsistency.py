"""The self-consistency aggregator, exercised against a real store on disk.

Builds a run directory whose result_store holds known situation sets, so the election, the
tie rule, the abstention rule and the oracle ceiling are all checked against outcomes computed
by hand rather than by the code under test.

Run from the repo root:  python tests/test_selfconsistency.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import result_store  # noqa: E402
from selfconsistency import _elect, _situations_key, analyse, breakdown  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def approx(name, got, want, tol=1e-6):
    ok = got is not None and abs(got - want) < tol
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got {got!r}  want {want!r}")
        failures.append(name)


# ---------------------------------------------------------------- key hashing
print("--- _situations_key ---")
A = (set(), set(), [((1, 2), (3,), False), ((4,), (5,), True)])
B = (set(), set(), [((4,), (5,), True), ((1, 2), (3,), False)])   # same set, other order
C = (set(), set(), [((1, 2), (3,), False)])
check("order-independent", _situations_key(A), _situations_key(B))
check("different sets differ", _situations_key(A) != _situations_key(C), True)
check("violation flag is part of identity",
      _situations_key((set(), set(), [((1,), (2,), True)]))
      != _situations_key((set(), set(), [((1,), (2,), False)])), True)
check("None passes through", _situations_key(None), None)

# ---------------------------------------------------------------- election
print("\n--- _elect ---")
def s(rep, key, size=3, trunc=False):
    return {"repeat": rep, "key": key, "size": size, "truncated": trunc, "ids": None, "f1": 0.0}

six = [s(i, "X") for i in range(1, 7)] + [s(i, f"Y{i}") for i in range(7, 11)]
check("6/10 majority elected", _elect(six, False)[:3], ("X", 6, 10))
check("not flagged as a tie", _elect(six, False)[3], False)

allsplit = [s(i, f"Z{i}") for i in range(1, 11)]
k, size, voters, tie = _elect(allsplit, False)
check("all-distinct falls back to lowest repeat", k, "Z1")
check("cluster size 1", size, 1)
check("flagged as a tie", tie, True)

abst = [s(1, None), s(2, None), s(3, "W"), s(4, "W")]
k, size, voters, tie = _elect(abst, False)
check("non-executing samples abstain, not vote for nothing", (k, size, voters), ("W", 2, 2))

# The rule that matters: 6 empties would out-vote 4 agreeing non-empty results.
empties = [s(i, "EMPTY", size=0) for i in range(1, 7)] + [s(i, "REAL", size=5) for i in range(7, 11)]
check("plain majority elects the empty set", _elect(empties, False)[0], "EMPTY")
check("filtered rule elects the real set", _elect(empties, True)[0], "REAL")
check("truncated results are filtered too",
      _elect([s(1, "T", size=9, trunc=True), s(2, "OK", size=4)], True)[0], "OK")
check("no voters at all", _elect([s(1, None), s(2, None)], False), (None, 0, 0, False))

# ---------------------------------------------------------------- end to end
print("\n--- analyse() against a real store ---")
tmp = Path(tempfile.mkdtemp(prefix="selfcons_"))
run = tmp / "run"
(run / "result_store").mkdir(parents=True)
store = run / "result_store"
LOG = "/logs/x.json"

GOLD = {"nodes": [{"gold": True}]}
gold_sit = [[[1], [10], False], [[2], [20], False], [[3], [30], False], [[4], [40], False]]
result_store.store(store, LOG, GOLD, (set(), set(), gold_sit), False)

# 6 samples return exactly gold; 3 return half of it; 1 never executed.
Q_RIGHT = {"nodes": [{"q": "right"}]}
Q_HALF  = {"nodes": [{"q": "half"}]}
result_store.store(store, LOG, Q_RIGHT, (set(), set(), gold_sit), False)
result_store.store(store, LOG, Q_HALF, (set(), set(), gold_sit[:2]), False)

def rec(repeat, query, f1):
    return {"case_id": 1, "nl_query": "Q?", "ocel_path": LOG, "repeat": repeat,
            "config_id": "arm|strategy|True", "gold_query": GOLD, "prediction": query,
            "status": "success" if query else "schema_invalid",
            "metrics": {"bindings_skipped": False},
            "similarity": None if f1 is None else {
                "combined": {"f1": f1, "precision": f1, "recall": f1}}}

records = ([rec(i, Q_RIGHT, 1.0) for i in range(1, 7)]
           + [rec(i, Q_HALF, 0.6667) for i in range(7, 10)]
           + [rec(10, None, None)])
(run / "predictions.json").write_text(json.dumps(records), encoding="utf-8")

out = analyse(run)
c = out["conditions"]["arm|strategy|True"]
check("one case", c["cases"], 1)
# single-sample baseline: (6*1.0 + 3*0.6667 + 0) / 10
approx("single-sample baseline averages the samples", c["single_sample_f1"], (6 + 3 * 0.6667) / 10)
approx("majority elects the 6-vote correct set", c["majority_f1"], 1.0)
approx("oracle is the best single sample", c["oracle_f1"], 1.0)
approx("gain is positive here", c["gain_majority"], 1.0 - (6 + 3 * 0.6667) / 10)
check("agreement recorded as 6 of 9 voters", c["agreement_histogram"], {"6/9": 1})
check("the non-executing sample abstained (9 voters, not 10)",
      list(c["agreement_histogram"])[0].endswith("/9"), True)
check("no ties", c["ties"], 0)
check("oracle never below majority", c["oracle_f1"] >= c["majority_f1"], True)

print("\n--- a case the vote gets WRONG ---")
run2 = tmp / "run2"
(run2 / "result_store").mkdir(parents=True)
st2 = run2 / "result_store"
result_store.store(st2, LOG, GOLD, (set(), set(), gold_sit), False)
result_store.store(st2, LOG, Q_RIGHT, (set(), set(), gold_sit), False)
result_store.store(st2, LOG, Q_HALF, (set(), set(), gold_sit[:2]), False)
# 7 agree on the WRONG answer, 3 on the right one.
recs2 = ([rec(i, Q_HALF, 0.6667) for i in range(1, 8)]
         + [rec(i, Q_RIGHT, 1.0) for i in range(8, 11)])
(run2 / "predictions.json").write_text(json.dumps(recs2), encoding="utf-8")
c2 = analyse(run2)["conditions"]["arm|strategy|True"]
approx("majority elects the wrong-but-popular answer", c2["majority_f1"], 0.6667, tol=1e-3)
approx("oracle still sees the right one", c2["oracle_f1"], 1.0)
check("headroom exposes the selection failure", c2["headroom_to_oracle"] > 0.3, True)
check("agreement 7/10", c2["agreement_histogram"], {"7/10": 1})

shutil.rmtree(tmp, ignore_errors=True)

print("\n--- breakdown (notes 22.2-22.7) ---")


def case(majority, cluster, voters, samples, log="a.json", disc=True, unloadable=0, floor=0.5):
    return {"majority": majority, "majority_cluster": cluster, "majority_voters": voters,
            "sample_f1": samples, "single_mean": sum(samples) / len(samples),
            "oracle": max(samples), "log": log, "discriminating": disc,
            "noop_floor": floor, "trivial_floor": floor, "unloadable": unloadable}


rows = [case(1.0, 9, 10, [1.0] * 9 + [0.0]),               # >= 90% consensus, exact
        case(0.0, 2, 4, [0.0] * 10, unloadable=1),         # 40-60%, one unloadable sample
        case(0.5, 1, 1, [0.5] + [0.0] * 9, log="retyped.json")]
b = breakdown(rows)
check("consensus is cluster / voters, over cases with >= 2 voters",
      {k: v["cases"] for k, v in b["consensus"].items()},
      {">=90%": 1, "60-90%": 0, "40-60%": 1, "<40%": 0})
approx("single-sample exact is the share of exact samples", b["exact_discriminating"]["single"],
       0.3)
approx("the notes' figure counts cases where every sample was exact",
       b["exact_discriminating"]["every_sample_as_in_notes"], 0.0)
approx("the gain from a one-voter case is attributed to it",
       b["gain_share_from_cases_with_at_most_one_voter"], 0.45 / (0.1 + 0.45))
approx("a case holding an unloadable sample scores 0 in the bound",
       b["majority_f1_if_unloadable_cases_scored_0"], 1.5 / 3)
check("a log can be left out", breakdown(rows, ["retyped.json"])["cases"], 2)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
