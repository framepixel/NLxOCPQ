"""The harness hands the agent's probe the log and the model's own trees -- never gold -- and every
case starts from a clean slate.

`tests/test_ocpq_probe.py` proves the probe module cannot REACH gold. This proves the harness does
not HAND it any: a leak would come through an argument at a call site in `eval_testset.py`, and it
would look like a good result, not a crash. Scanned on the source, so a future edit that passes
`gold_query` to the probe fails here before any run.

Two more properties of the wiring (notes 43):
  * **Per-record resets** happen before the pipeline runs. Without them a case that dies early
    inherits the previous case's turns, tool calls and restarts.
  * **A case killed mid-loop keeps its turns.** The per-case cap is a signal raised inside the
    model call; the trajectory must already be published when it fires.

Run from the repo root:  python tests/test_agentic_no_gold.py
"""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


SOURCE = (ROOT / "src" / "eval_testset.py").read_text(encoding="utf-8")
TREE_AST = ast.parse(SOURCE)
# Every name in eval_testset that holds gold, or a row carrying it.
GOLD_NAMES = {"gold_query", "gt_record", "ground_truth_cache", "structure_hint", "row",
              "case_root_arity", "valid_testset", "testset"}
PROBE_RECEIVERS = {"agent_probe_obj", "probe_server"}
PROBE_FUNCTIONS = {"_probe_cli_agreement", "OcpqProbe", "ProbeServer", "set_agent_probe"}


def _names_in(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


print("--- every call that reaches the probe passes no gold ---")
probe_calls = []
for node in ast.walk(TREE_AST):
    if not isinstance(node, ast.Call):
        continue
    func = node.func
    receiver = func.value.id if (isinstance(func, ast.Attribute)
                                 and isinstance(func.value, ast.Name)) else None
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if receiver in PROBE_RECEIVERS or name in PROBE_FUNCTIONS:
        passed = set()
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            passed |= _names_in(arg)
        probe_calls.append((node.lineno, f"{receiver + '.' if receiver else ''}{name}", passed))
check("the probe is reached from the harness at all (the scan is not vacuous)",
      len(probe_calls) >= 5, True)
leaks = [(line, call, sorted(passed & GOLD_NAMES)) for line, call, passed in probe_calls
         if passed & GOLD_NAMES]
check("no probe call receives a gold-carrying name", leaks, [])
loads = [passed for _, call, passed in probe_calls if call.endswith("ensure_log")]
check("the log is loaded by its path alone", loads, [{"ocel_path"}])

print("\n--- per-record resets run before the pipeline ---")
lines = SOURCE.splitlines()


def first_line(fragment, after=0):
    for number, text in enumerate(lines[after:], start=after + 1):
        if fragment in text:
            return number
    return None


record_loop = first_line("for repeat_idx in range(1, repeats + 1):")
pipeline = first_line("nl_to_ocpq_pipeline(", after=record_loop)
for fragment in ("reset_last_agent_trajectory()", "agent_probe_obj.reset()",
                 "agent_probe_obj.ensure_log(ocel_path)"):
    at = first_line(fragment, after=record_loop)
    check(f"{fragment} is inside the record loop and before the pipeline",
          at is not None and record_loop < at < pipeline, True)
limit = first_line("with _case_time_limit(case_timeout_sec):", after=record_loop)
load = first_line("agent_probe_obj.ensure_log(ocel_path)", after=record_loop)
check("the log load is OUTSIDE the per-case time limit (before the `with`)", load < limit, True)

print("\n--- a case killed mid-loop keeps the turns it took ---")
import LLM  # noqa: E402

TREE = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                            "filters": [], "sizeFilters": [], "constraints": [],
                            "evVarLabels": {}, "obVarLabels": {}, "labels": []}, []]}],
        "edgeNames": []}


class CapFired(BaseException):
    """Stands in for eval_testset.CaseTimeout, raised by SIGALRM inside the model call."""


replies = ['{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE),
           '{"tool": "sample", "args": {"type": "orders"}}']


def model(model, messages, max_tokens, response_format, **kwargs):
    if not replies:
        raise CapFired("the per-case cap fired during turn 3")
    content = replies.pop(0)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                           usage=None)


class Probe:
    tool_calls = 0

    def __init__(self):
        self.trajectory = []

    def call(self, name, args):
        self.tool_calls += 1
        self.trajectory.append({"tool": name})
        return {"nodes": [{"node": 0, "situations": 4, "violations": 0}]}


LLM.call_hf_with_retry = model
LLM.extract_ocel_schema = lambda path: {"event_types": ["place order"], "object_types": ["orders"],
                                        "event_attributes": {}, "object_attributes": {},
                                        "e2o": [], "o2o": []}
LLM.build_schema_block = lambda schema, level=None: "ALLOWED OBJECTS: orders"
probe = Probe()
LLM.set_agent_probe(probe)
LLM.set_agent_max_turns(6)
try:
    LLM._run_pipeline_agentic("Which orders?", None, None,
                              output_file=str(ROOT / "eval_runs" / "_no_gold_test.json"),
                              ocel_path="../Logs/order-management.json", model_name="m",
                              schema_validation=False)
    check("the cap propagates out of the loop", "swallowed", "propagated")
except CapFired:
    check("the cap propagates out of the loop", "propagated", "propagated")
state = LLM.get_last_agent_trajectory()
check("the trajectory survives the kill", state is not None, True)
check("...with the turn that was running", state["turns"], 3)
check("...the two completed turns on it", [t.get("action") for t in state["trajectory"]],
      ["try_query", "sample"])
check("...their tool calls on the live tool trajectory", len(state["tool_trajectory"]), 2)
check("...and marked incomplete", state["complete"], False)
LLM.reset_last_agent_trajectory()
check("the per-record reset clears it for the next case", LLM.get_last_agent_trajectory(), None)
LLM.set_agent_probe(None)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the probe gets the log and the model's trees, never gold; cases start clean")
