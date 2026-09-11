"""The arity predictor: is its answer expressible, is it safe when it fails, and does the gate bite?

Forcing the root box's variable count to gold's moved result F1 from 0.2162 to 0.3109 (notes 32),
the largest effect in the project. That is an ORACLE. `predict_root_arity` is the component that
would supply that number without gold, and `src/arity_predictor_probe.py` is the gate that decides
whether it is worth 676 runs.

Three things are tested, and the third is the one that matters.

  * The `RootArity` grammar can express every arity gold uses, and refuses everything else. Same
    two-sided gate as notes 30: a constraint that cannot express the answer is a silent cap, and a
    constraint that accepts anything is not a constraint.
  * A failed prediction returns None rather than a guess. Forcing a WRONG arity makes the correct
    query unreachable, while not forcing leaves the model its own 43.8% chance -- so abstaining is
    strictly safer, and that has to be a property of the code rather than an intention.
  * **The gate rejects a predictor that is the constant plus noise.** The oracle's gain is
    concentrated in the 51 minority-arity cases (0.0766 -> 0.2228); a predictor that scores 70% by
    always answering (0,1) captures none of it while looking like progress. If this assertion ever
    stops holding, the gate has stopped protecting anything.

Run from the repo root:  python tests/test_arity_predictor.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from eval_testset import _gold_root_arity  # noqa: E402
from ocpq_schema import ROOT_ARITY_SCHEMA, RootArity  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


try:
    import xgrammar as xgr
except ImportError:
    print("FAIL  xgrammar is not installed.")
    sys.exit(1)

accepts = xgr.testing._is_grammar_accept_string
TESTSET = json.load(open(REPO / "data" / "testset.json", encoding="utf-8"))
GOLD_ARITIES = sorted({_gold_root_arity(e["ocpq_query"])[1:] for e in TESTSET})

print("--- the answer schema can express every arity gold actually uses ---")
g = xgr.Grammar.from_json_schema(RootArity, any_order=True, strict_mode=True)
print(f"        gold uses {len(GOLD_ARITIES)} distinct arities: {GOLD_ARITIES}")
unexpressible = [a for a in GOLD_ARITIES
                 if not accepts(g, '{"n_event_vars":%d,"n_object_vars":%d}' % a)]
check("every gold arity is expressible", unexpressible, [])

print("\n--- ...and it refuses everything that is not an arity ---")
for label, text in (
    ("prose", "two objects and one event"),
    ("markdown fence", '```json\n{"n_event_vars":0,"n_object_vars":1}'),
    ("a missing field", '{"n_event_vars":0}'),
    ("an unknown field", '{"n_event_vars":0,"n_object_vars":1,"n_other":2}'),
    ("a negative count", '{"n_event_vars":-1,"n_object_vars":1}'),
    ("an absurd count", '{"n_event_vars":999,"n_object_vars":1}'),
    ("a whole query", '{"nodes":[],"edgeNames":[]}'),
):
    check(f"rejects {label}", accepts(g, text), False)

print("\n--- the bound exists so a constrained decoder cannot stall on a huge integer ---")
# Gold never exceeds 2 of either kind; the cap is 4. Without it the grammar admits arbitrarily
# many digits and a confused model can emit them one token at a time until the budget runs out.
check("gold never needs more than 2 of either kind",
      max(max(a) for a in GOLD_ARITIES) <= 2, True)
props = ROOT_ARITY_SCHEMA["properties"]
check("the schema caps both fields",
      (props["n_event_vars"].get("maximum"), props["n_object_vars"].get("maximum")), (4, 4))

print("\n--- a failed prediction abstains; it never guesses ---")
# Forcing a wrong arity makes the correct query UNREACHABLE. Not forcing leaves the model its own
# 43.8%. So None is the only safe failure mode, and the caller treats it as "do not force".
import LLM  # noqa: E402

def _predict_with(monkey_content):
    """Run `predict_root_arity` against a canned model reply."""
    class _Msg:  # minimal stand-in for the OpenAI-shaped response
        def __init__(self, c): self.message = type("M", (), {"content": c})()
    original = LLM.call_hf_with_retry
    LLM.call_hf_with_retry = lambda **kw: type("R", (), {"choices": [_Msg(monkey_content)]})()
    try:
        return LLM.predict_root_arity("q", "/nonexistent.json", "m", schema_block="<schema>")
    finally:
        LLM.call_hf_with_retry = original

check("a well-formed reply parses", _predict_with('{"n_event_vars":1,"n_object_vars":2}'), (1, 2))
for label, reply in (("prose", "one event and two objects"), ("empty", ""),
                     ("half an object", '{"n_event_vars":1'),
                     ("out of range", '{"n_event_vars":99,"n_object_vars":1}'),
                     ("a whole query", '{"nodes":[],"edgeNames":[]}')):
    check(f"abstains on {label}", _predict_with(reply), None)

print("\n--- the schema override is restored even when the call raises ---")
# A leaked override would constrain the NEXT query generation to a two-integer object and fail
# every case after this one -- a failure that would look like a model collapse, not a bug here.
LLM.set_constrained_schema(None)
original = LLM.call_hf_with_retry
def _boom(**kw):
    raise RuntimeError("simulated backend failure")
LLM.call_hf_with_retry = _boom
try:
    result = LLM.predict_root_arity("q", "/nonexistent.json", "m", schema_block="<schema>")
finally:
    LLM.call_hf_with_retry = original
check("a raising call still abstains", result, None)
check("...and leaves the override cleared", LLM.get_constrained_schema(), None)

print("\n--- THE GATE must reject a predictor that is the constant plus noise ---")
from arity_predictor_probe import MAJORITY, report  # noqa: E402
import io, contextlib  # noqa: E402

golds = [_gold_root_arity(e["ocpq_query"])[1:] for e in TESTSET]

def _gate(rows):
    with contextlib.redirect_stdout(io.StringIO()):
        return report(rows)

constant = _gate([{"gold": list(gd), "predicted": list(MAJORITY)} for gd in golds])
check("a constant predictor FAILS the gate", constant["gate_passed"], False)
check("...even though its pooled accuracy is 63.3%",
      round(constant["accuracy"], 3), 0.633)
check("...because its minority recall is zero", constant["minority_recall"], 0.0)

perfect = _gate([{"gold": list(gd), "predicted": list(gd)} for gd in golds])
check("a perfect predictor passes", perfect["gate_passed"], True)

# The case the gate exists to catch: better than the constant overall, but only by being right
# more often on the majority class. It must still fail.
noisy = []
for i, gd in enumerate(golds):
    pred = list(gd) if (tuple(gd) == MAJORITY or i % 7 == 0) else list(MAJORITY)
    noisy.append({"gold": list(gd), "predicted": pred})
noisy_result = _gate(noisy)
check("a predictor that only finds the majority class fails, whatever its pooled score",
      noisy_result["gate_passed"] and noisy_result["minority_recall"] == 0.0, False)

# Abstentions are counted, never scored as correct.
abstaining = _gate([{"gold": list(gd), "predicted": None} for gd in golds])
check("abstentions are counted separately", abstaining["abstained"], len(golds))
check("...and never count as correct", abstaining["correct"], 0)

print("\n--- the probe resolves a REAL config, the way the evaluation does ---")
# This is here because guessing `expand_models`' signature cost a cluster job: the probe called
# `expand_models(models, ft_map)` when it takes and returns the whole CONFIG, and died on the first
# line after the model finished loading. A no-GPU resolution check catches that in milliseconds.
from arity_predictor_probe import _expand_model  # noqa: E402

cfg = json.loads((REPO / "configs" / "eval_config.json").read_text(encoding="utf-8"))
resolved = _expand_model(cfg, 1)
check("model-idx 1 resolves", isinstance(resolved, dict) and "name" in resolved, True)
check("...to the fine-tuned Llama the ablations use",
      resolved.get("label"), "llama31-8b-finetuned")
# The index must mean the same thing here as in eval_testset, or the probe measures another model.
import eval_testset as ET  # noqa: E402
ev_cfg = json.loads((REPO / "configs" / "eval_config.json").read_text(encoding="utf-8"))
if ev_cfg.get("auto_expand_finetuned"):
    from expand_models import expand_models  # noqa: E402
    ev_cfg = expand_models(ev_cfg)
check("the probe and eval_testset agree on what --model-idx 1 means",
      ev_cfg["models"][1].get("label"), resolved.get("label"))
# And an out-of-range index must fail loudly, naming the options.
try:
    _expand_model(cfg, 999)
    check("an out-of-range index raises", False, True)
except SystemExit as exc:
    check("an out-of-range index raises SystemExit", True, True)
    check("...and names the available models", "llama31-8b-finetuned" in str(exc), True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the answer is expressible, failure abstains, and the gate bites")
