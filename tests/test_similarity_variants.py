"""The result-metric decompositions: do they relax the right thing, and only that?

`combined` keys each root binding as `(event_indices, object_indices, violated)`, so a prediction
that binds one extra variable changes every key and scores exactly 0. Measured on the
self-consistency store, that is 152 samples -- ~18% of executed discriminating samples -- whose
mean objects-only F1 is 0.5203 and whose `combined` F1 is 0.0000, only 4 of which actually select
the wrong objects. `SIMILARITY_VARIANTS` exists to quantify that, and to split the comparison into
"selected the right rows" and "flagged the right violations".

A relaxed metric that hands out credit it should not is worse than no metric, because the
resulting numbers look better rather than wrong. So the tests here are two-sided: each variant
must relax its own artefact AND leave a genuinely wrong answer at 0.

Two invariants carry the most weight:
  * `variants["strict"]` is `combined`, exactly. Six analysis modules read `combined`, and every
    published number depends on it; if the projection table ever drifts from the primary metric,
    the whole comparison silently changes basis.
  * `violation` is None, never 0, where gold flags nothing. Twenty-two of 135 discriminating
    entries are compliance rules their log fully satisfies -- scoring those 0 would measure the
    testset, and scoring them 1 would reward a query that flags nobody.

Run from the repo root:  python tests/test_similarity_variants.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_testset import (  # noqa: E402
    SIMILARITY_VARIANTS,
    _compute_similarity,
    _record_floor,
    _record_skill,
    _record_skill_variants,
    _record_variant_f1,
    _set_metrics,
    _skill_score,
    _variant_metrics,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def f1_of(scored):
    """F1 out of a `_set_metrics` dict, or None when the variant was undefined."""
    return None if scored is None else scored["f1"]


def ids(situations):
    """A `(event_ids, object_ids, situations)` triple as `_extract_result_ids` returns one."""
    events = {e for evs, _obs, _v in situations for e in evs}
    objects = {o for _evs, obs, _v in situations for o in obs}
    return events, objects, set(situations)


# Gold binds ONE OBJECT per situation and no events -- the shape of 73 of the 135 discriminating
# entries, and the shape the extra-variable artefact needs.
GOLD_OBJ = [((), (10,), False), ((), (11,), True), ((), (12,), False)]

print("--- the artefact: a prediction that binds one extra event variable ---")
# Same three objects, same flags, but each binding also carries an event.
PRED_EXTRA_EV = [((7,), (10,), False), ((8,), (11,), True), ((9,), (12,), False)]
v = _variant_metrics(set(GOLD_OBJ), set(PRED_EXTRA_EV))
check("strict scores it 0", f1_of(v["strict"]), 0.0)
check("population also 0 -- the flag is not what differs", f1_of(v["population"]), 0.0)
check("kind_projected recovers it", f1_of(v["kind_projected"]), 1.0)

print("\n--- the artefact does NOT extend to a genuinely wrong answer ---")
PRED_WRONG_OBJ = [((7,), (90,), False), ((8,), (91,), True), ((9,), (92,), False)]
v = _variant_metrics(set(GOLD_OBJ), set(PRED_WRONG_OBJ))
for name in SIMILARITY_VARIANTS:
    check(f"{name} gives no credit for disjoint objects", f1_of(v[name]), 0.0)

print("\n--- population: right rows, wrong violation flags ---")
PRED_FLAGS = [((), (10,), True), ((), (11,), False), ((), (12,), True)]
v = _variant_metrics(set(GOLD_OBJ), set(PRED_FLAGS))
check("strict scores it 0", f1_of(v["strict"]), 0.0)
check("population recovers the population", f1_of(v["population"]), 1.0)
check("violation still 0 -- it flagged the wrong one", f1_of(v["violation"]), 0.0)
check("kind_projected keeps the flag, so still 0", f1_of(v["kind_projected"]), 0.0)

print("\n--- violation is UNDEFINED where gold flags nothing ---")
GOLD_NO_VIOLATIONS = [((), (10,), False), ((), (11,), False)]
v = _variant_metrics(set(GOLD_NO_VIOLATIONS), set(GOLD_NO_VIOLATIONS))
check("violation is None, not 0", v["violation"], None)
check("violation is None, not 1 either", f1_of(v["violation"]), None)
check("a perfect prediction still scores 1 on strict", f1_of(v["strict"]), 1.0)
# ... and a query that flags nothing must not be rewarded for it on some other variant.
v = _variant_metrics(set(GOLD_OBJ), {((), (10,), False), ((), (11,), False), ((), (12,), False)})
check("gold DOES flag: violation is defined and 0 for a query that flags nobody",
      f1_of(v["violation"]), 0.0)

print("\n--- kind_projected is exactly strict where gold binds both kinds ---")
GOLD_BOTH = [((7,), (10,), False), ((8,), (11,), True)]
for label, pred in (("perfect", GOLD_BOTH),
                    ("wrong events", [((70,), (10,), False), ((80,), (11,), True)]),
                    ("wrong objects", [((7,), (99,), False), ((8,), (98,), True)]),
                    ("empty", [])):
    v = _variant_metrics(set(GOLD_BOTH), set(pred))
    check(f"kind_projected == strict ({label})",
          f1_of(v["kind_projected"]), f1_of(v["strict"]))

print("\n--- kind_projected cannot manufacture credit when gold binds nothing ---")
# Every situation would project onto the same blank key, handing any non-empty prediction a 1.0.
GOLD_BLANK = [((), (), False), ((), (), True)]
v = _variant_metrics(set(GOLD_BLANK), {((7,), (10,), False)})
check("falls back to strict rather than collapsing", f1_of(v["kind_projected"]),
      f1_of(v["strict"]))
check("... which is 0 here", f1_of(v["kind_projected"]), 0.0)

print("\n--- variants['strict'] IS combined, on every shape ---")
CASES = [
    ("identical", GOLD_OBJ, GOLD_OBJ),
    ("disjoint", GOLD_OBJ, PRED_WRONG_OBJ),
    ("partial", GOLD_OBJ, [((), (10,), False), ((), (91,), True)]),
    ("empty prediction", GOLD_OBJ, []),
    ("empty gold", [], PRED_EXTRA_EV),
    ("both empty", [], []),
    ("extra variable", GOLD_OBJ, PRED_EXTRA_EV),
    ("both kinds", GOLD_BOTH, GOLD_BOTH),
]
for label, gold, pred in CASES:
    sim = _compute_similarity(ids(gold), ids(pred))
    check(f"strict == combined ({label})", sim["variants"]["strict"], sim["combined"])
check("combined itself is untouched by the variants",
      _compute_similarity(ids(GOLD_OBJ), ids(PRED_EXTRA_EV))["combined"],
      _set_metrics(set(GOLD_OBJ), set(PRED_EXTRA_EV)))
check("a missing result still yields no similarity block at all",
      _compute_similarity(ids(GOLD_OBJ), None), None)

print("\n--- the strict convention: a missing result is 0, not the 1.0 of empty-vs-empty ---")
# `_set_metrics(set(), set())` returns f1 1.0, which is correct for "the query ran and returned
# nothing on an entry whose gold is also empty" and catastrophic for "the query never ran". The
# record-level accessor is what keeps those apart; it cost two wrong measurements to notice.
check("empty-vs-empty really is 1.0 at the set level", _set_metrics(set(), set())["f1"], 1.0)
NO_RESULT = {"similarity": None, "baseline_f1_variants": {"strict": 0.5, "population": 0.6,
                                                          "violation": 0.4,
                                                          "kind_projected": 0.5}}
check("no similarity -> 0 under the strict convention",
      _record_variant_f1(NO_RESULT, "strict", strict=True), 0.0)
check("no similarity -> None under the executed convention",
      _record_variant_f1(NO_RESULT, "strict", strict=False), None)

print("\n--- an entry where a variant is undefined stays undefined for a missing result ---")
# The floors dict is the authority: whether gold flags anything is a property of the ENTRY, and
# cannot be read off a similarity block that is not there. Substituting 0 would blame the model
# for the log's compliance.
UNDEF = {"similarity": None,
         "baseline_f1_variants": {"strict": 0.5, "population": 0.6, "violation": None,
                                  "kind_projected": 0.5}}
check("violation stays None", _record_variant_f1(UNDEF, "violation", strict=True), None)
check("the other variants still take the 0",
      _record_variant_f1(UNDEF, "population", strict=True), 0.0)

print("\n--- an audit file predating the floors degrades, it does not lie ---")
OLD = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(GOLD_OBJ))}
check("F1 is still computable", _record_variant_f1(OLD, "population", strict=True), 1.0)
check("every skill is undefined rather than read against the strict floor",
      _record_skill_variants(OLD), {name: None for name in SIMILARITY_VARIANTS})

print("\n--- skill uses each variant's OWN floor ---")
REC = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(PRED_EXTRA_EV)),
       "baseline_f1_variants": {"strict": 0.5, "population": 0.5, "violation": None,
                                "kind_projected": 0.5}}
skills = _record_skill_variants(REC)
check("strict: F1 0 against floor 0.5 -> -1", skills["strict"], -1.0)
check("kind_projected: F1 1 against floor 0.5 -> +1", skills["kind_projected"], 1.0)
check("violation: no floor -> undefined", skills["violation"], None)

print("\n--- the table the audit and the metric share ---")
check("four variants, strict first", list(SIMILARITY_VARIANTS),
      ["strict", "population", "violation", "kind_projected"])

print("\n--- the floor is the BEST trivial strategy, not one chosen strategy ---")
# The no-op baseline is gold with its constraints stripped, so it flags nobody and its violation
# floor is 0 on every entry -- which made a model scoring 0.1360 there look like positive skill.
# Flag-everything is the other degenerate strategy; the floor is the max. On the real testset it
# wins on 54/135 entries and lifts the strict floor from 0.5148 to 0.7250.
from audit_testset_discriminability import (  # noqa: E402
    _build_flag_everything_query,
    _degenerate_floors,
    _flag_everything_situations,
)

# Gold: three bound objects, the middle one flagged. Baseline: same population, none flagged.
GOLD_MIXED = {((), (10,), False), ((), (11,), True), ((), (12,), False)}
BASELINE = {((), (10,), False), ((), (11,), False), ((), (12,), False), ((), (13,), False)}
floors, detail = _degenerate_floors(GOLD_MIXED, BASELINE)

check("flag-everything keeps the population, flips every flag",
      _flag_everything_situations(BASELINE),
      {((), (10,), True), ((), (11,), True), ((), (12,), True), ((), (13,), True)})
check("no-op violation floor is 0 -- it cannot flag anything", detail["violation"]["noop"], 0.0)
check("flag-everything gives violation a real floor",
      detail["violation"]["flagall"] > 0.0, True)
check("...and that is the floor used", floors["violation"], detail["violation"]["flagall"])
for name in SIMILARITY_VARIANTS:
    parts = [v for v in detail[name].values() if v is not None]
    check(f"{name}: floor == max of the strategies", floors[name],
          max(parts) if parts else None)
check("population is unaffected -- it drops the flag both sides",
      detail["population"]["noop"], detail["population"]["flagall"])
# Undefinedness is a property of the entry and must survive the max.
no_flags = {((), (10,), False), ((), (11,), False)}
f2, d2 = _degenerate_floors(no_flags, BASELINE)
check("gold flags nothing -> violation floor stays None", f2["violation"], None)
check("...in the detail too", d2["violation"], {"noop": None, "flagall": None})
f3, d3 = _degenerate_floors(GOLD_MIXED, None)
check("no baseline at all -> every floor None", set(f3.values()), {None})

print("\n--- the flag-everything query is a real query, not an assertion ---")
# Validated by execution on one entry of each of the eleven logs (11/11 match, notes 27). Here we
# only assert the construct is what was validated: same root, plus a bound nothing can satisfy.
gold_q = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                              "filters": [], "sizeFilters": [], "constraints": [],
                              "evVarLabels": {}, "obVarLabels": {}}, []]}],
          "edgeNames": []}
fq = _build_flag_everything_query(gold_q)
check("two nodes: the root and an empty child", len(fq["nodes"]), 2)
check("root points at the child", fq["nodes"][0]["Box"][1], [1])
check("the child binds nothing", fq["nodes"][1]["Box"][0]["newObjectVars"], {})
c = fq["nodes"][0]["Box"][0]["constraints"][0]
check("the constraint is a NumChilds bound", c["filter"]["type"], "NumChilds")
check("...that no binding can reach", c["filter"]["min"] >= 10 ** 9, True)
check("root still declares gold's variable",
      fq["nodes"][0]["Box"][0]["newObjectVars"], {"0": ["orders"]})
check("malformed gold degrades", _build_flag_everything_query({"nodes": []}), None)

print("\n--- skill reads the true floor, and old records keep theirs ---")
# Two of gold's three situations, so F1 0.8: above a 0.5 floor, below a 0.9 one. A PARTIAL
# prediction is required to see the floor at all -- below the floor skill divides by the floor,
# so a fully wrong answer scores exactly -1 however high the floor goes. That invariance is
# deliberate (see `_skill_score`) and is asserted below rather than assumed.
PARTIAL = [((), (10,), False), ((), (11,), True)]
NEW = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(PARTIAL)),
       "baseline_f1": 0.5, "trivial_floor": 0.9}
OLDREC = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(PARTIAL)),
          "baseline_f1": 0.5}
check("uses trivial_floor when present", _record_floor(NEW), 0.9)
check("falls back to baseline_f1 when absent", _record_floor(OLDREC), 0.5)
check("a run predating the correction keeps the skill it published",
      _record_skill(OLDREC), _skill_score(0.8, 0.5))
check("the corrected floor scores the same prediction lower",
      _record_skill(NEW) < _record_skill(OLDREC), True)
check("...and flips it from beating trivial to losing to it",
      (_record_skill(OLDREC) > 0, _record_skill(NEW) < 0), (True, True))
# The invariance that made the naive version of this test wrong, pinned so it stays true.
WRONG_LOW = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(PRED_WRONG_OBJ)),
             "trivial_floor": 0.5}
WRONG_HIGH = {"similarity": _compute_similarity(ids(GOLD_OBJ), ids(PRED_WRONG_OBJ)),
              "trivial_floor": 0.9}
check("a fully wrong answer is -1 at any floor",
      (_record_skill(WRONG_LOW), _record_skill(WRONG_HIGH)), (-1.0, -1.0))

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
