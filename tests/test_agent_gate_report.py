"""`agent_gate_report.py` must apply the pre-registered gate rules exactly as written.

The gates decided which models the agentic evaluations score, so each criterion is pinned just
inside and just outside its threshold (notes 11 §44.4), along with the helpers that count repeats
and the root rule the mechanism statistics share with the metric:

  * a repeat is an action whose raw text was already sent in the same case;
  * a repeat's result is compared without the probe's per-call `evalVersion` counter;
  * a loop-driven timeout fails the gate even when the timeout rate is under 5%;
  * a tree the probe declined (`agree` None) is not a probe-CLI disagreement;
  * the root of a `try_query` tree is the first unreferenced box, 0 for a cycle -- exactly
    `eval_testset.root_node_index`.

Run from the repo root:  python tests/test_agent_gate_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_gate_report import _raw_repeats, _strip_counter, verdict  # noqa: E402
from agent_mechanism_stats import root_index, root_situations  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


EVAL2 = {"malformed_max": 0.05, "repeat_max": 0.10, "forced_close_max": 0.25,
         "timeout_max": 0.05, "loop_timeout_min_repeats": 2, "p90_case_seconds_max": 1500,
         "declared_min": 0.90, "probe_cli_agreement": 1.0, "no_pipeline_errors": True}


def passing_measures(**overrides):
    """A gate task that passes every evaluation-2 criterion, with fields overridden."""
    m = {"malformed_rate": 0.0, "repeat_rate": 0.08, "forced_close_rate": 0.04,
         "timeouts": 0, "timeout_rate": 0.0, "loop_timeouts": 0, "p90_case_seconds": 710.0,
         "p90_seconds_per_turn": None, "declared_rate": 1.0, "probe_cli_rate": 1.0,
         "pipeline_errors": [], "uptake": 0.74}
    m.update(overrides)
    return {"gate": m}


def failed(measures, criteria=EVAL2):
    passes, checks = verdict(measures, criteria)
    return passes, sorted({c["criterion"] for c in checks if not c["passes"]})


check("a clean task passes", failed(passing_measures()), (True, []))
check("repeats exactly at 10% pass", failed(passing_measures(repeat_rate=0.10))[0], True)
check("repeats above 10% fail (Glimmer, 12.4%)",
      failed(passing_measures(repeat_rate=0.124)), (False, ["repeats"]))
check("forced closes above 25% fail (Ministral-R, 73.1%)",
      failed(passing_measures(forced_close_rate=0.731)), (False, ["forced closes"]))
check("malformed above 5% fail", failed(passing_measures(malformed_rate=0.051)),
      (False, ["malformed"]))
check("a slow p90 fails", failed(passing_measures(p90_case_seconds=1500.5)),
      (False, ["p90 case s"]))
check("one timeout in 38 passes (Gemma 4 at the example gate)",
      failed(passing_measures(timeouts=1, timeout_rate=1 / 38))[0], True)
check("a loop-driven timeout fails even under 5%",
      failed(passing_measures(timeouts=1, timeout_rate=1 / 38, loop_timeouts=1)),
      (False, ["timeouts"]))
check("declared below 90% fails", failed(passing_measures(declared_rate=0.89)),
      (False, ["declared"]))
check("a pipeline error fails (Ministral-R's probe_unavailable)",
      failed(passing_measures(pipeline_errors=["probe_unavailable"])),
      (False, ["pipeline errors"]))
check("probe-CLI below 1.0 fails", failed(passing_measures(probe_cli_rate=0.97)),
      (False, ["probe-CLI"]))
check("no checkable probe comparison is not a failure",
      failed(passing_measures(probe_cli_rate=None))[0], True)
with_uptake = dict(EVAL2, uptake_min=0.25)
check("uptake below 25% fails the example gate (Ministral-R, 13%)",
      failed(passing_measures(uptake=0.13), with_uptake), (False, ["uptake"]))
check("uptake is not a criterion of the evaluation-2 gate",
      failed(passing_measures(uptake=0.0))[0], True)

EVAL1 = {"malformed_max": 0.05, "repeat_max": 0.10, "repeat_applies_to": "thought_off",
         "loop_timeout_min_repeats": 2,
         "p90_seconds_per_turn_max": {"thought_off": 30, "thought_on": 60},
         "declared_min": 0.90, "probe_cli_agreement": 1.0}
both = {"thought_off": passing_measures(p90_seconds_per_turn=27.2)["gate"],
        "thought_on": passing_measures(p90_seconds_per_turn=34.2, repeat_rate=0.137)["gate"]}
check("evaluation 1 reads repeats on the thought-off task only", failed(both, EVAL1)[0], True)
both["thought_off"] = dict(both["thought_off"], repeat_rate=0.214)
check("evaluation 1: Qwen's 21.4% thought-off repeats fail", failed(both, EVAL1),
      (False, ["repeats"]))
both = {"thought_off": passing_measures(p90_seconds_per_turn=32.9)["gate"],
        "thought_on": passing_measures(p90_seconds_per_turn=44.1)["gate"]}
check("evaluation 1: p90 s/turn has a per-task threshold (Gemma 4, 32.9 > 30)",
      failed(both, EVAL1), (False, ["p90 s/turn"]))

# --- repeats -----------------------------------------------------------------------------------
steps = [{"raw": "A"}, {"raw": "B"}, {"raw": "A"}, {"raw": "A"}, {"raw": None}]
check("repeats pair with the latest identical action", _raw_repeats(steps), [(0, 2), (2, 3)])
thought = [{"raw": '{"thought": "first idea", "tool": "sample", "args": {"n": 5}}'},
           {"raw": '{"thought": "second idea", "tool": "sample", "args": {"n": 5}}'}]
check("the same action with a different thought is a repeat", _raw_repeats(thought), [(0, 1)])
check("the evalVersion counter is ignored",
      _strip_counter("{'nodes': [], 'evalVersion': 3, 'structure': []}"),
      _strip_counter("{'nodes': [], 'evalVersion': 17, 'structure': []}"))
check("other differences are kept",
      _strip_counter("{'situations': 1, 'evalVersion': 3}")
      == _strip_counter("{'situations': 2, 'evalVersion': 3}"), False)

# --- the root rule -----------------------------------------------------------------------------


def action(nodes):
    return json.dumps({"tool": "try_query", "args": {"tree": {"nodes": nodes}}})


box = {"newEventVars": {}}
check("root is the unreferenced box", root_index(action([{"Box": [box, []]},
                                                         {"Box": [box, [0]]}])), 1)
check("a cycle falls back to 0", root_index(action([{"Box": [box, [1]]},
                                                    {"Box": [box, [0]]}])), 0)
check("an unparseable action has no root", root_index("{not json"), None)
step = {"raw": action([{"Box": [box, []]}, {"Box": [box, [0]]}]),
        "result": {"nodes": [{"node": 0, "situations": 5}, {"node": 1, "situations": 0}]}}
check("root situations read the root box, not node 0", root_situations(step), 0)
check("an unevaluated step has no count", root_situations({"raw": step["raw"]}), None)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
