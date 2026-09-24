"""The agentic loop keeps its protocol, its budget, and its distance from gold.

`LLM._run_pipeline_agentic` is the first pipeline here that generates in a loop, so it has failure
modes none of the others have -- and each one would look like a result rather than a bug:

  * **A leaked grammar override.** Every local generation gets the xgrammar processor attached, so
    an agent turn must be constrained to `AgentAction` or the model can only emit a BindingBoxTree
    and never a tool call. Worse, an override left set constrains the NEXT case's query. Both
    directions are asserted, including after an exception.
  * **Inventing an action.** `parse_llm_json` is right for queries and wrong for a control channel:
    it takes a greedy `{.*}` span and appends a closing brace to truncated output, which can turn a
    cut-off tool call into a different, valid one. `_parse_agent_action` must refuse instead, so the
    failure is counted (`agent_malformed_actions`) rather than silently executed.
  * **Losing a case to formatting.** A budget that runs out with a valid tree already produced must
    finalise with it (notes 43's pre-registered rule), or the tool rungs lose cases the one-shot
    rung wins by construction and the comparison measures protocol compliance, not capability.
  * **Gold.** `structure_hint` carries gold's tree shape and is passed to every strategy by
    `nl_to_ocpq_pipeline`. This arm must drop it.

Run from the repo root:  python tests/test_agent_action_protocol.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import LLM  # noqa: E402
import typing  # noqa: E402

from ocpq_schema import AGENT_TOOL_CALLS, AgentAction, AgentTurn  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def check_true(name, got):
    check(name, bool(got), True)


TREE = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                            "filters": [], "sizeFilters": [], "constraints": [],
                            "evVarLabels": {}, "obVarLabels": {}, "labels": []}, []]}],
        "edgeNames": []}
GOLD_SENTINEL = "SENTINEL_GOLD_SHAPE_MUST_NOT_APPEAR"


class ScriptedModel:
    """Replays queued assistant replies, recording what the loop sent and the grammar in force."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.schema_during_call = []

    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        self.sent.append([dict(message) for message in messages])
        self.schema_during_call.append(LLM.get_constrained_schema())
        reply = self.replies.pop(0) if self.replies else '{"final_tree": %s}' % json.dumps(TREE)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                               usage=None)


class FakeProbe:
    """Stands in for `ocpq_probe.OcpqProbe`: records calls, can refuse on an exhausted budget."""

    def __init__(self, results=None, budget=99):
        self.results = results or {}
        self.budget = budget
        self.tool_calls = 0
        self.trajectory = []
        self.calls = []

    def call(self, name, args):
        self.calls.append((name, args))
        if self.tool_calls >= self.budget:
            raise LLM.ocpq_probe.ProbeBudgetExhausted("budget spent")
        self.tool_calls += 1
        result = self.results.get(name, {"nodes": [{"node": 0, "situations": 7, "violations": 0}]})
        self.trajectory.append({"tool": name, "result": result})
        return result


def run_loop(replies, probe=None, max_turns=4, structure_hint=None, schema_validation=False):
    """Drive `_run_pipeline_agentic` with everything below the model stubbed out."""
    model = ScriptedModel(replies)
    LLM.call_hf_with_retry = model
    LLM.extract_ocel_schema = lambda path: {"event_types": ["place order"],
                                            "object_types": ["orders"],
                                            "event_attributes": {}, "object_attributes": {},
                                            "e2o": [], "o2o": []}
    LLM.build_schema_block = lambda schema, level=None: "ALLOWED OBJECTS: orders"
    LLM.run_ocpq_query_with_logging = lambda **kwargs: ("Evaluated in 1ms", None, "/tmp/result.json")
    LLM.set_agent_probe(probe)
    LLM.set_agent_max_turns(max_turns)
    outcome, error = None, None
    try:
        outcome = LLM._run_pipeline_agentic(
            "Which orders were paid late?", None, None,
            output_file=str(OUTPUT_FILE), ocel_path="../Logs/order-management.json",
            model_name="test-model", schema_validation=schema_validation,
            structure_hint=structure_hint)
    except Exception as exc:  # noqa: BLE001 - the loop's own signalling is under test
        error = exc
    return model, outcome, error


OUTPUT_FILE = Path(ROOT / "eval_runs" / "_agent_protocol_test_query.json")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("--- the parser refuses rather than invents ---")
action, reason = LLM._parse_agent_action('{"tool": "try_query", "args": {"tree": %s}}'
                                        % json.dumps(TREE))
check("a plain action parses", action.tool, "try_query")
action, reason = LLM._parse_agent_action(
    'Sure thing {"tool": "sample", "args": {"type": "orders"}} -- let me know {if} that helps')
check("prose on both sides does not swallow the object (parse_llm_json would)",
      (action.tool, action.args["type"]), ("sample", "orders"))
action, reason = LLM._parse_agent_action('```json\n{"tool": "sample", "args": {"type": "o"}}\n```')
check("a fenced block parses", action.tool, "sample")

print("\n--- the strict turn schema: each tool has typed arguments (gate smoke, 2026-09-23) ---")
action, reason = LLM._parse_agent_action('{"tool": "try_query", "args": {"tree": {}}}')
check("a try_query tree must be a real BindingBoxTree", action, None)
check_true("...and the reason names the call", "malformed try_query call" in reason)
action, reason = LLM._parse_agent_action('{"tool": "sample", "args": {}}')
check("a missing required argument is refused", action, None)
check_true("...naming the variant, not six union branches", "SampleCall" in reason)
action, reason = LLM._parse_agent_action('{"tool": "summon_gold", "args": {}}')
check_true("an unknown tool is refused at parse time, with the real list",
           action is None and "try_query" in reason)
action, reason = LLM._parse_agent_action('{"tool": "inspect_situations", "args": {}}')
check("optional arguments take their documented defaults", action.args,
      {"node": 0, "n": 5, "violated": None})
check("the grammar's tool variants are exactly the probe's tools",
      sorted(typing.get_args(c.model_fields["tool"].annotation)[0] for c in AGENT_TOOL_CALLS),
      sorted(spec["name"] for spec in LLM.ocpq_probe.TOOL_SPECS))

print("\n--- trees get the same container repair every other arm gets ---")
NO_FILTERS = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                                  "sizeFilters": [], "constraints": [], "evVarLabels": {},
                                  "obVarLabels": {}}, []]}], "edgeNames": []}
action, reason = LLM._parse_agent_action('{"final_tree": %s}' % json.dumps(NO_FILTERS))
check("a final tree missing `filters` (an any_order artefact) is repaired, not rejected",
      (action is not None, list(LLM._last_agent_parse_normalized)), (True, ["nodes[0].filters"]))
action, reason = LLM._parse_agent_action('{"tool": "try_query", "args": {"tree": %s}}'
                                        % json.dumps(NO_FILTERS))
check("...and so is a try_query tree", action.args["tree"]["nodes"][0]["Box"][0]["filters"], [])
check("the case's own normalisation record is untouched", LLM.get_last_normalized_fields(), [])
action, reason = LLM._parse_agent_action('{"tool": "sample", "args": {"type": "ord')
check("a truncated action is REFUSED, not auto-healed", action, None)
check_true("...with a reason the model can act on", "truncate" in reason)
action, reason = LLM._parse_agent_action('{"tool": "x", "final_tree": %s}' % json.dumps(TREE))
check("both fields at once is refused", action, None)
action, reason = LLM._parse_agent_action('{"note": "thinking about it"}')
check("neither field is refused", action, None)
action, reason = LLM._parse_agent_action('{"final_tree": {"nodes": "not a tree"}}')
check("a final_tree that is not a BindingBoxTree is refused", action, None)
check_true("...naming the field", "final_tree" in reason)
action, reason = LLM._parse_agent_action('{"final_tree": %s}' % json.dumps(TREE))
check("a valid final tree parses", isinstance(action.final_tree, type(action.final_tree)), True)
check("the parsed tree round-trips", action.final_tree.model_dump(by_alias=True)["edgeNames"], [])

print("\n--- the grammar is the PROTOCOL during a turn, and restored afterwards ---")
LLM.set_constrained_schema(None)
model, outcome, error = run_loop(['{"final_tree": %s}' % json.dumps(TREE)])
check("the turn was constrained to the strict AgentTurn union", model.schema_during_call,
      [AgentTurn])
check("the override is cleared afterwards", LLM.get_constrained_schema(), None)

LLM.set_constrained_schema(None)
model, outcome, error = run_loop([RuntimeError("boom")])
check("an exception inside the turn still clears the override", LLM.get_constrained_schema(), None)

previous = LLM.PromptTooLongError("too long")
LLM.set_constrained_schema(None)
model, outcome, error = run_loop([previous])
check("PromptTooLongError ends the loop instead of failing the case",
      type(error).__name__, "AgentNoFinalAnswer")
check("...and it is recorded as a turn", "prompt too long" in json.dumps(
    LLM.get_last_agent_trajectory()["trajectory"]), True)

print("\n--- a malformed turn is counted and corrected, never fatal ---")
model, outcome, error = run_loop(['not json at all',
                                  '{"final_tree": %s}' % json.dumps(TREE)])
check("the loop survives a malformed turn", error, None)
trajectory = LLM.get_last_agent_trajectory()
check("the malformed turn is counted", trajectory["malformed_actions"], 1)
check("and the loop still answered", trajectory["final_source"], "declared")
last_sent = model.sent[-1]
check("the correction is sent as a USER message", last_sent[-1]["role"], "user")
check_true("...quoting the protocol back",
           "one json object per turn" in last_sent[-1]["content"].lower())
roles = [message["role"] for message in last_sent]
check("no two consecutive messages share a role (Gemma's template rejects that)",
      any(a == b for a, b in zip(roles, roles[1:])), False)

print("\n--- tools: results come back as one user message per turn ---")
probe = FakeProbe()
model, outcome, error = run_loop(
    ['{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE),
     '{"final_tree": %s}' % json.dumps(TREE)], probe=probe)
check("the tool ran", [name for name, _ in probe.calls], ["try_query"])
trajectory = LLM.get_last_agent_trajectory()
check("the tool call is counted on the record", trajectory["tool_calls"], 1)
check("two turns were used", trajectory["turns"], 2)
check("the probe's own trajectory is carried over", len(trajectory["tool_trajectory"]), 1)
roles = [message["role"] for message in model.sent[-1]]
check("still strictly alternating after a tool result",
      any(a == b for a, b in zip(roles, roles[1:])), False)
check_true("the tool result reached the model",
           "situations" in model.sent[-1][-1]["content"])

print("\n--- the control rung has no tools, and cannot acquire them ---")
probe = FakeProbe()
model, outcome, error = run_loop(
    ['{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE),
     '{"final_tree": %s}' % json.dumps(TREE)], probe=None)
check("no tool was executed", probe.calls, [])
check_true("the model is told it has none",
           "no tools" in model.sent[-1][-1]["content"].lower())
check("the tool-call count is zero", LLM.get_last_agent_trajectory()["tool_calls"], 0)
check_true("the prompt itself withholds the tool list",
           "NO tools" in model.sent[0][1]["content"])

print("\n--- budget exhaustion: finalise with the best tree, or say there is none ---")
probe = FakeProbe(budget=1)
model, outcome, error = run_loop(
    ['{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE),
     '{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE),
     '{"tool": "try_query", "args": {"tree": %s}}' % json.dumps(TREE)],
    probe=probe, max_turns=3)
check("a spent budget does not fail the case", error, None)
check("the tree the model actually ran is finalised",
      LLM.get_last_agent_trajectory()["final_source"], "last_valid_probe")
check("and it is what gets returned", outcome[2]["edgeNames"], [])

model, outcome, error = run_loop(['nonsense', 'still nonsense'], max_turns=2)
check("nothing usable raises AgentNoFinalAnswer", type(error).__name__, "AgentNoFinalAnswer")
check("...which is a ValueError subclass, so existing handlers still see it",
      isinstance(error, ValueError), True)
check("...and the status is not phrased as an execution failure",
      "execut" in str(error).lower(), False)
check("the trajectory still records both turns",
      LLM.get_last_agent_trajectory()["turns"], 2)

print("\n--- gold cannot reach the loop ---")
model, outcome, error = run_loop(['{"final_tree": %s}' % json.dumps(TREE)],
                                 structure_hint=f"GOLD SHAPE: {GOLD_SENTINEL}")
sent = json.dumps(model.sent)
check("the structure hint appears in no message", GOLD_SENTINEL in sent, False)
check("nor in the trajectory", GOLD_SENTINEL in json.dumps(LLM.get_last_agent_trajectory()), False)
check("the agentic strategies are not marked oracle",
      [name for name in LLM.AGENTIC_PROMPT_STRATEGIES
       if name in LLM.ORACLE_PROMPT_STRATEGIES], [])
check("...and do not carry the load-bearing oracle prefix",
      [name for name in LLM.AGENTIC_PROMPT_STRATEGIES if name.startswith("oracle-")], [])

print("\n--- registration: a strategy the dispatcher cannot run is a silent dead end ---")
for name in LLM.AGENTIC_PROMPT_STRATEGIES:
    check(f"{name} is registered", name in LLM.PROMPT_STRATEGIES, True)
    check(f"{name} normalises to itself", LLM.normalize_prompt_strategy(name), name)
check("the short alias reaches the tool rung",
      LLM.normalize_prompt_strategy("agentic"), "agentic-log-tools")
check("the tool list is generated from the probe's registry, not written out",
      all(spec["name"] in LLM._agent_tool_block(FakeProbe())
          for spec in LLM.ocpq_probe.TOOL_SPECS), True)

print("\n--- the thought condition: reasoning is recorded, never acted on ---")
from ocpq_schema import AgentTurnWithThought  # noqa: E402

LLM.set_agent_thought(False)
model, outcome, error = run_loop(['{"final_tree": %s}' % json.dumps(TREE)])
check("off by default: the turn is constrained to the plain turn grammar",
      model.schema_during_call, [AgentTurn])
check("off: the prompt never mentions a thought",
      "thought" in json.dumps(model.sent[0]), False)
action, _ = LLM._parse_agent_action('{"thought": "orders first", "tool": "sample", "args": {}}')
check("off: a stray thought (free decoding only) is dropped by the plain schema",
      getattr(action, "thought", None), None)

LLM.set_agent_thought(True)
model, outcome, error = run_loop([
    '{"thought": "check the payment count first", "tool": "try_query", "args": {"tree": %s}}'
    % json.dumps(TREE),
    '{"thought": "counts look right", "final_tree": %s}' % json.dumps(TREE)],
    probe=FakeProbe())
state = LLM.get_last_agent_trajectory()
check("on: every turn is constrained to the thought turn grammar",
      set(model.schema_during_call), {AgentTurnWithThought})
check("on: the prompt carries the thought rule",
      LLM.AGENT_THOUGHT_RULE in model.sent[0][1]["content"], True)
check("on: a thought beside a tool call leaves the call intact",
      [t.get("action") for t in state["trajectory"]], ["try_query", "final_tree"])
check("on: the thought is recorded on its turn",
      [t.get("thought") for t in state["trajectory"]],
      ["check the payment count first", "counts look right"])
check("on: the thought never reaches the final tree", "thought" in json.dumps(outcome[2]), False)
check("on: the trajectory says the condition was on", state["thought_enabled"], True)
model, outcome, error = run_loop(['{"thought": "just thinking"}'], max_turns=1)
check("on: a thought ALONE is not an action -- counted malformed, not executed",
      (LLM.get_last_agent_trajectory()["malformed_actions"], outcome), (1, None))
LLM.set_agent_thought(False)

print("\n--- the agent gets the one-shot prompt's tree rules, verbatim (gate smoke, 2026-09-23) ---")
# Without them the model wrote `filters` in 0 of 12 turns (B0: O2E filters in 158/169), so A1 - A0
# would have measured the missing rules as well as the tools. B0's template is read from SOURCE,
# not re-typed, so any edit to either copy fails here.
import re  # noqa: E402

_src = (ROOT / "src" / "LLM.py").read_text(encoding="utf-8")
_start = _src.index("def _run_pipeline_allowed_entities_only(")
_b0 = _src[_start:_src.index("\ndef ", _start + 1)]
_m = re.search(r'\nRules:\n(.*?)\n\nQuery:', _b0, re.S)
check("B0's rules block located", _m is not None, True)
if _m:
    check("OCPQ_TREE_RULES is B0's rendered rules, character for character",
          LLM.OCPQ_TREE_RULES, _m.group(1).replace("{{", "{").replace("}}", "}"))
for probe in (FakeProbe(), None):
    model, outcome, error = run_loop(['{"final_tree": %s}' % json.dumps(TREE)], probe=probe)
    check(f"the {'tool' if probe else 'control'} rung's prompt carries the rules",
          LLM.OCPQ_TREE_RULES in model.sent[0][1]["content"], True)

OUTPUT_FILE.unlink(missing_ok=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the loop holds its protocol, its budget and its distance from gold")
