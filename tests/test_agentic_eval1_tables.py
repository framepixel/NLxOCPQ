"""`agentic_eval1_tables.py` must count what the notes' evaluation-1 tables counted.

Each definition was calibrated against notes 11 §43.11, §43.13 and §43.15; these checks pin it on
a two-case synthetic run, so a change that drifts from the published numbers fails here first:

  * a trajectory step's action is stored JSON-encoded, sometimes twice;
  * a draft is "resubmitted" only when it equals the case's PREVIOUS draft, thought included;
  * a clean draft has no blocking structural defect (a cycle is one);
  * "without a declared answer" counts every final source but `declared`;
  * a thought is counted per turn that carries one.

Runs in the evaluation environment (ocpq_schema needs pydantic).
Run from the repo root:  python tests/test_agentic_eval1_tables.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent_gate_report import GateTask  # noqa: E402
from agentic_eval1_tables import _action, dev_measures  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


BOX = {"newEventVars": {}, "newObjectVars": {"0": ["book"]}, "filters": [], "sizeFilters": [],
       "constraints": []}
CLEAN = {"nodes": [{"Box": [BOX, []]}], "edgeNames": []}
CYCLE = {"nodes": [{"Box": [BOX, [0]]}], "edgeNames": [[[0, 0], "A"]]}


def step(action):
    return {"raw": json.dumps(json.dumps(action))}


def record(question, status, final_source, prediction, turns, llm_seconds):
    return {"nl_query": question, "ocel_path": "../Logs/order-management.json",
            "config_id": "c", "repeat": 1, "discriminating": True, "status": status,
            "prediction": prediction, "llm": {"duration_sec": llm_seconds},
            "total_duration_sec": llm_seconds + 1, "similarity": None,
            "metrics": {"structural": {"f1": 0.5}},
            "agent": {"turns": turns, "max_turns": 8, "malformed_actions": 0, "tool_calls": 2,
                      "final_source": final_source, "turn_summary": [], "probe_refused_trees": 1}}


check("a doubly encoded action is decoded", _action(json.dumps(json.dumps({"tool": "x"}))),
      {"tool": "x"})
check("an unparseable step has no action", _action("{not json"), None)

draft_a = {"tool": "try_query", "args": {"tree": CLEAN}}
draft_b = {"tool": "try_query", "args": {"tree": CYCLE}}
trajectories = [
    {"record_key": "k1", "agent": {"trajectory": [step(draft_a), step(draft_a), step(draft_b),
                                                  step(draft_a)]}},
    {"record_key": "k2", "agent": {"trajectory": [step({"thought": "look first",
                                                        "tool": "sample", "args": {}}),
                                                  step(draft_b)]}},
]
records = [record("q1", "success", "declared", CLEAN, 4, 40.0),
           record("q2", "agent_no_answer", "none", CYCLE, 2, 10.0)]

with tempfile.TemporaryDirectory() as tmp:
    run = Path(tmp)
    (run / "predictions.json").write_text(json.dumps(records), encoding="utf-8")
    (run / "agent_trajectories.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trajectories), encoding="utf-8")
    m = dev_measures(GateTask(str(run)))

check("every try_query is a draft", m["drafts"], 5)
check("only a repeat of the previous draft is a resubmission", m["resubmitted"], 1)
check("a cycle is not a clean draft", m["clean_drafts"], 3)
check("a final tree with a cycle is blocking", (m["finals_blocking"], m["finals"]), (1, 2))
check("no answer reads the status", m["no_answer"], 1)
check("anything but declared is not declared", m["not_declared"], 1)
check("a thought counts per turn", (m["thought_turns"], m["steps"]), (1, 6))
check("the longest question is in LLM seconds", m["longest_llm_seconds"], 40.0)
check("refused trees are summed", m["probe_refused"], 2)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
