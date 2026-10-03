"""The Stage 5 protocol levers (notes 43.12): each does what it says, and none leaks when off.

The model gate (notes 43.11) found the arm's model holding the protocol and failing at tree
STRUCTURE: 71 of 75 finals carried a blocking defect, 178 of 298 try_query trees listed a node as its
own child. Five levers answer that, each a config flag:

  F  structural feedback -- a structure report on try_query, and a static check_tree tool in BOTH
     loop rungs (so A1 - A1' isolates log access; decided 2026-09-26)
  B  a declared final tree with blocking defects is returned ONCE while a turn remains
  E  finalise on the last tree the probe EVALUATED before the last schema-valid one (amends the
     pre-registered rule)
  X  the tree-linking rules and one worked example in an invented domain
  L  the last turn admits a final answer only

With every lever off, the prompt must be what the model gate saw: its hash is pinned below, so a
prompt edit is a deliberate, visible change and never a silent one.

Run from the repo root:  python tests/test_agent_stage5.py [--corpus]
  --corpus  also scan all 11 corpus logs (2.7 GB) for the worked example's type names.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import LLM  # noqa: E402
from eval_testset import (  # noqa: E402
    AGENT_LEVERS, _agent_budget, _agent_compliance, _agent_levers, _agent_protocol, _agent_suffix,
    _compact_agent_turns)
from ocpq_schema import (  # noqa: E402
    DEFECT_HINTS, PROBE_TOOL_NAMES, AgentTurn, BindingBoxTree, agent_turn_model,
    blocking_defects, structural_defects)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def check_true(name, got):
    check(name, bool(got), True)


CLEAN = {"nodes": [{"Box": [{"newEventVars": {}, "newObjectVars": {"0": ["orders"]},
                             "filters": [], "sizeFilters": [], "constraints": [],
                             "evVarLabels": {}, "obVarLabels": {}}, [1]]},
                   {"Box": [{"newEventVars": {"0": ["pay"]}, "newObjectVars": {},
                             "filters": [{"type": "O2E", "object": 0, "event": 0,
                                          "qualifier": None}],
                             "sizeFilters": [], "constraints": [], "evVarLabels": {},
                             "obVarLabels": {}}, []]}],
         "edgeNames": [[[0, 1], "A"]]}
# The gate's commonest shape: node 0 lists itself.
CYCLIC = {"nodes": [{"Box": [dict(CLEAN["nodes"][0]["Box"][0]), [0]]}], "edgeNames": []}
# Blocking but NOT refused by the probe: two roots (root_count).
TWO_ROOTS = {"nodes": [{"Box": [dict(CLEAN["nodes"][0]["Box"][0]), []]},
                       {"Box": [dict(CLEAN["nodes"][1]["Box"][0]), []]}], "edgeNames": []}


class ScriptedModel:
    def __init__(self, replies):
        self.replies, self.sent, self.grammars = list(replies), [], []

    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        self.sent.append([dict(m) for m in messages])
        self.grammars.append(getattr(LLM.get_constrained_schema(), "__name__", None))
        reply = self.replies.pop(0) if self.replies else '{"final_tree": %s}' % json.dumps(CLEAN)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
                               usage=None)


class FakeProbe:
    """Counts for every tree except those the probe refuses (a cycle), like the real one."""

    def __init__(self):
        self.tool_calls, self.trajectory, self.calls = 0, [], []

    def call(self, name, args):
        self.calls.append(name)
        self.tool_calls += 1
        tree = args.get("tree") or {}
        if any("[cycle]" in d for d in structural_defects(tree)):
            return {"error": "not evaluated: the tree is malformed", "defects": ["[cycle] ..."]}
        return {"nodes": [{"node": 0, "situations": 7, "violations": 0}]}


OUT = ROOT / "eval_runs" / "_agent_stage5_test_query.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


class CutModel(ScriptedModel):
    """Its first reply stops at the cap, reported as the local backend does: exactly `cap` tokens."""

    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        first = not self.sent
        out = super().__call__(model, messages, max_tokens, response_format, **kwargs)
        if first:
            out.usage = SimpleNamespace(completion_tokens=max_tokens)
        return out


def run(replies, probe=None, max_turns=4, thought=False, model_cls=ScriptedModel, **levers):
    model = model_cls(replies)
    LLM.call_hf_with_retry = model
    LLM.extract_ocel_schema = lambda path: {"event_types": ["pay"], "object_types": ["orders"],
                                            "event_attributes": {}, "object_attributes": {},
                                            "e2o": [], "o2o": []}
    LLM.build_schema_block = lambda schema, level=None: "ALLOWED OBJECTS: orders"
    LLM.run_ocpq_query_with_logging = lambda **kwargs: ("ok", None, None)
    LLM.set_agent_probe(probe)
    LLM.set_agent_max_turns(max_turns)
    LLM.set_agent_thought(thought)
    LLM.set_agent_protocol(**levers)
    outcome = error = None
    try:
        outcome = LLM._run_pipeline_agentic("Which orders?", None, None, output_file=str(OUT),
                                            ocel_path="../Logs/order-management.json",
                                            model_name="m", schema_validation=False)
    except Exception as exc:  # noqa: BLE001 - the loop's own signalling is under test
        error = exc
    return model, outcome, error, LLM.get_last_agent_trajectory()


def tool(name, tree):
    return json.dumps({"tool": name, "args": {"tree": tree}})


def final(tree):
    return json.dumps({"final_tree": tree})


print("--- the levers are explicit, and off means the model gate's protocol ---")
try:
    LLM.set_agent_protocol(structural_feedbak=True)
    check("a misspelt lever raises", "no error", "ValueError")
except ValueError:
    check("a misspelt lever raises", "ValueError", "ValueError")
LLM.set_agent_protocol(structural_feedback=True)
LLM.set_agent_protocol(worked_example=True)
check("setting replaces, never merges (no lever inherited from a previous condition)",
      LLM.get_agent_protocol()["structural_feedback"], False)
LLM.set_agent_thought(False)
LLM.set_agent_protocol()
# Pinned 2026-09-26 from the code the model gate ran (fingerprint 137f1fc52d), verified identical
# message-for-message against that commit. A change here is a change to the gate's condition.
check("levers off: the tool rung's prompt is the model gate's (hash pinned)",
      LLM.agent_prompt_sha256(True),
      "8d733044648a696881b990d2daf7eb4d4f74822e5b4c760956a2a58440e9bc80")
check("levers off: the control rung's prompt is the model gate's (hash pinned)",
      LLM.agent_prompt_sha256(False),
      "56b5e0ee601a1217647792e9509f36f33be94e507649184c7a4edb5b03415392")
check("levers off: the tool rung compiles AgentTurn itself",
      agent_turn_model(PROBE_TOOL_NAMES) is AgentTurn, True)
hashes = set()
for letters in ("", "F", "X", "L", "FBEXL"):
    LLM.set_agent_protocol(**{lever: letter in letters for _k, lever, letter in AGENT_LEVERS})
    hashes.add(LLM.agent_prompt_sha256(True))
check("every lever set that changes the prompt changes the hash (B and E change no text)",
      len(hashes), 5)
LLM.set_agent_protocol()

print("\n--- names: every lever set is its own directory; A0 keeps its id ---")
# The five Stage 5 levers; C (notes 43.15) and R (notes 43.17) are tested on their own below, and S
# (notes 45) in tests/test_example_search.py.
cfg = {key: True for key, lever, _letter in AGENT_LEVERS
       if lever not in ("cap_feedback", "repeat_guard", "example_search")}
check("letters in a fixed order", _agent_levers(cfg), "FBEXL")
check("none -> no suffix part", _agent_levers({}), "")
check("the suffix carries them after the turn cap, before the thought",
      _agent_suffix("agentic-log-tools", 8, 12, True, 2000, "FBEXL"),
      "_turns8_tools12_turntok2000_pFBEXL_thought")
check("...on the control rung too", _agent_suffix("agentic-self-refine", 8, 12, False, 2000, "FX"),
      "_turns8_turntok2000_pFX")
check("A0 never runs the loop, so the levers never name it",
      _agent_suffix("schema-allowed-entities-only", 8, 12, False, 2000, "FBEXL"), "")
check("_agent_budget reads them from the config", _agent_budget(cfg)[4], "FBEXL")
for name in ("eval_config.agentic.json", "eval_config.agentic_gate.json"):
    conf = json.load(open(ROOT / "configs" / name, encoding="utf-8"))
    # R failed its gate (notes 43.18), so evaluation 1 stays on the scored run's pFBEXL: C and R off.
    check(f"{name}: the five Stage 5 levers on, C and R off (pFBEXL, notes 43.18)",
          _agent_levers(conf), "FBEXL")

print("\n--- X: the worked example is valid, well-formed, and borrowed from nothing ---")
example = LLM.AGENT_WORKED_EXAMPLE
check("it validates as a BindingBoxTree",
      BindingBoxTree.model_validate(example) is not None, True)
check("it has no structural defect at all", structural_defects(example), [])
check("it uses NumChilds min 1 on the root, not a bare SAT (SAT is vacuous on an empty child)",
      (example["nodes"][0]["Box"][0]["constraints"][0]["filter"]["type"],
       example["nodes"][0]["Box"][0]["constraints"][0]["filter"]["min"]), ("NumChilds", 1))
names = {t for node in example["nodes"] for key in ("newEventVars", "newObjectVars")
         for types in node["Box"][0][key].values() for t in types}
gold_trees, gold_names = [], set()
for path in ("data/testset.json", "data/agent_devset.json"):
    for entry in json.load(open(ROOT / path, encoding="utf-8")):
        gold_trees.append(BindingBoxTree.model_validate(entry["ocpq_query"])
                          .model_dump(by_alias=True, mode="json"))
        for node in entry["ocpq_query"]["nodes"]:
            for key in ("newEventVars", "newObjectVars"):
                for types in (node["Box"][0].get(key) or {}).values():
                    gold_names.update(types)
check("its type names occur in no testset or dev-set gold", names & gold_names, set())
check("it is no gold tree", any(t == BindingBoxTree.model_validate(example).model_dump(
    by_alias=True, mode="json") for t in gold_trees), False)
if "--corpus" in sys.argv:
    manifest = json.load(open(ROOT / "data" / "corpus_manifest.json", encoding="utf-8"))
    logs = ROOT.parent / "Logs"
    for log in manifest["logs"]:
        data = (logs / log).read_bytes().lower()
        check(f"corpus {log}: none of {sorted(names)} as a name",
              [n for n in names if f'"{n}"'.encode() in data or f">{n}<".encode() in data], [])
else:
    # Scanned once on 2026-09-26 (grep over all 11 logs): neither `book` nor `shelve book` is a
    # type anywhere. The word `book` occurs only inside the activity "Book Vehicles"
    # (container_logistics) and in commit text (angular); `shelves` only in free text
    # (reasoning_benchmark). `borrow` occurs 19040x in lending_log_high ("Borrowers"), which is why
    # the example's first draft ("borrow book") was replaced.
    print("SKIP  corpus scan (2.7 GB; run with --corpus)")
model, outcome, error, state = run([final(CLEAN)], probe=FakeProbe(), worked_example=True)
task = model.sent[0][1]["content"]
check("on: the prompt carries the linking rules and the example",
      (LLM.AGENT_TREE_STRUCTURE_RULES in task, json.dumps(example) in task), (True, True))
model, outcome, error, state = run([final(CLEAN)], probe=None, worked_example=True)
check("...and so does the control rung (same prompt, only the tools differ)",
      json.dumps(example) in model.sent[0][1]["content"], True)
model, outcome, error, state = run([final(CLEAN)], probe=FakeProbe())
check("off: neither appears", json.dumps(example) in model.sent[0][1]["content"], False)

print("\n--- F: the structure report and check_tree ---")
check("a fix hint exists for every defect kind structural_defects emits",
      sorted(DEFECT_HINTS), sorted({"child_out_of_range", "children_mismatch", "cycle",
                                     "edge_name_out_of_range", "malformed_tree",
                                     "multiple_parents", "root_count",
                                     "unbound_optional_reference", "unbound_variable",
                                     "unknown_child_name"}))
report = LLM._structure_report(TWO_ROOTS)
check_true("the report names the defect and its fix", report and "-> fix:" in report[0]
           and "root_count" in report[0])
check("a clean tree reports nothing", LLM._structure_report(CLEAN), [])
probe = FakeProbe()
model, outcome, error, state = run([tool("try_query", TWO_ROOTS), final(CLEAN)], probe=probe,
                                   structural_feedback=True)
seen = json.loads(model.sent[1][-1]["content"])
check("try_query: the counts come back WITH the structure report",
      ("nodes" in seen, any("root_count" in d for d in seen.get("structure", []))), (True, True))
probe = FakeProbe()
model, outcome, error, state = run([tool("try_query", CYCLIC), final(CLEAN)], probe=probe,
                                   structural_feedback=True)
seen = json.loads(model.sent[1][-1]["content"])
check("a refused tree keeps the probe's own defects and gets no second report",
      ("defects" in seen, "structure" in seen), (True, False))
probe = FakeProbe()
model, outcome, error, state = run([tool("check_tree", CYCLIC), final(CLEAN)], probe=probe,
                                   structural_feedback=True)
seen = json.loads(model.sent[1][-1]["content"])
check("check_tree reports the cycle, with its fix",
      any("[cycle]" in d and "-> fix:" in d for d in seen.get("structure", [])), True)
check("...without touching the probe or its budget", (probe.calls, probe.tool_calls), ([], 0))
check("...and is counted", state["check_tree_calls"], 1)
model, outcome, error, state = run([tool("check_tree", CLEAN), final(CLEAN)], probe=None,
                                   structural_feedback=True)
seen = json.loads(model.sent[1][-1]["content"])
check("the CONTROL rung has check_tree too, and a clean tree gets the no-defect note",
      (seen.get("structure"), "no structural defect" in seen.get("note", "")), ([], True))
check("the control's grammar offers check_tree and the final answer only",
      model.grammars[0], "AgentTurn_check_tree")
check("the control's prompt lists check_tree and no log tool",
      ("check_tree(" in model.sent[0][1]["content"], "try_query(" in model.sent[0][1]["content"]),
      (True, False))
model, outcome, error, state = run([final(CLEAN)], probe=FakeProbe(), structural_feedback=True)
check("the tool rung's grammar is the five probe tools plus check_tree",
      model.grammars[0], "AgentTurn_" + "_".join(PROBE_TOOL_NAMES + ("check_tree",)))

print("\n--- B: a blocking final is returned once, while a turn remains ---")
model, outcome, error, state = run([final(TWO_ROOTS), final(CLEAN)], probe=FakeProbe(),
                                   bounce_blocking_final=True)
check("the first blocking final is returned with its defects",
      "cannot be evaluated as written" in model.sent[1][-1]["content"]
      and "root_count" in model.sent[1][-1]["content"], True)
check("...the corrected one is accepted (its root has the child link the first lacked)",
      (outcome[2]["nodes"][0]["Box"][1], state["final_source"]), ([1], "declared"))
check("...and the case records the bounce", state["final_bounced"], True)
model, outcome, error, state = run([final(TWO_ROOTS), final(TWO_ROOTS)], probe=FakeProbe(),
                                   bounce_blocking_final=True)
check("only ONCE: a second blocking final is accepted as the answer",
      (len(model.sent), state["final_source"]), (2, "declared"))
model, outcome, error, state = run([final(TWO_ROOTS)], probe=FakeProbe(), max_turns=1,
                                   bounce_blocking_final=True)
check("never on the last turn: nothing would be left to answer with",
      (state["final_bounced"], state["final_source"]), (False, "declared"))
model, outcome, error, state = run([final(CLEAN)], probe=FakeProbe(), bounce_blocking_final=True)
check("a clean final is never returned", (state["final_bounced"], len(model.sent)), (False, 1))
model, outcome, error, state = run([final(TWO_ROOTS), final(CLEAN)], probe=FakeProbe())
check("off: a blocking final is final", (state["final_bounced"], len(model.sent)), (False, 1))

print("\n--- E: finalise on the last tree the probe EVALUATED ---")
ONE_BOX = {"nodes": [CLEAN["nodes"][0] | {}], "edgeNames": []}
ONE_BOX["nodes"][0] = {"Box": [dict(CLEAN["nodes"][0]["Box"][0]), []]}
replies = [tool("try_query", ONE_BOX), tool("try_query", CYCLIC)]
model, outcome, error, state = run(replies, probe=FakeProbe(), max_turns=2,
                                   finalise_evaluated=True)
check("the evaluated tree wins over the later refused (cyclic) one",
      (state["final_source"], outcome[2]["nodes"][0]["Box"][1]), ("last_evaluated_probe", []))
model, outcome, error, state = run(replies, probe=FakeProbe(), max_turns=2)
check("off: the pre-registered rule -- the last schema-valid tree, the cyclic one",
      (state["final_source"], outcome[2]["nodes"][0]["Box"][1]), ("last_valid_probe", [0]))
model, outcome, error, state = run([tool("try_query", CYCLIC)], probe=FakeProbe(), max_turns=1,
                                   finalise_evaluated=True)
check("on, with nothing evaluated: falls back to the last schema-valid tree",
      state["final_source"], "last_valid_probe")

print("\n--- L: the last turn is an answer ---")
model, outcome, error, state = run([tool("try_query", CLEAN), tool("try_query", CLEAN),
                                    final(CLEAN)], probe=FakeProbe(), max_turns=3,
                                   reserve_last_turn=True)
check("earlier turns keep the tool grammar; the last admits a final answer only",
      model.grammars, ["AgentTurn", "AgentTurn", "AgentTurn_final_only"])
last = model.sent[2]
check("the model is told, in the message it is answering",
      last[-1]["content"].endswith(LLM.AGENT_LAST_TURN_NOTE), True)
check("...so the roles still alternate",
      all(a["role"] != b["role"] for a, b in zip(last[1:], last[2:])), True)
check("the note is not left behind in earlier messages",
      sum(LLM.AGENT_LAST_TURN_NOTE in m["content"] for m in last), 1)
model, outcome, error, state = run([final(CLEAN)], probe=FakeProbe(), max_turns=1,
                                   reserve_last_turn=True, thought=True)
check("a one-turn case: the task itself says so; the thought variant is final-only too",
      (model.sent[0][1]["content"].endswith(LLM.AGENT_LAST_TURN_NOTE), model.grammars),
      (True, ["AgentTurn_final_only_thought"]))

print("\n--- the compliance block reads structure whatever the levers ---")
recs = [{"status": "success", "agent": {"turns": 3, "max_turns": 8, "final_source": "declared",
                                          "final_blocking_defects": 0, "final_bounced": True,
                                          "check_tree_calls": 2,
                                          "turn_summary": [{"tree_blocking": 1},
                                                           {"tree_blocking": 0}, {}]}},
        {"status": "success", "agent": {"turns": 8, "max_turns": 8,
                                          "final_source": "last_evaluated_probe",
                                          "final_blocking_defects": 3, "final_bounced": False,
                                          "check_tree_calls": 0,
                                          "turn_summary": [{"tree_blocking": 2}]}}]
block = _agent_compliance(recs)
check("final blocking rate, clean tree-turn rate, bounce rate, mean check_tree calls",
      (block["agent_final_blocking_rate"], round(block["agent_tree_turn_clean_rate"], 3),
       block["agent_final_bounce_rate"], block["agent_mean_check_tree_calls"]),
      (0.5, 0.333, 0.5, 1.0))
check("last_evaluated_probe appears in the final-source counts",
      block["agent_final_source"].get("last_evaluated_probe"), 1)

print("\n--- C: a turn cut off at the per-turn cap is told so (notes 43.15) ---")
CUT = '{"tool": "sample", "args": {"type": "ord'
check("its letter comes last, so the frozen rungs' suffix is unchanged",
      (_agent_levers(dict(cfg, agent_cap_feedback=True)), _agent_levers(cfg)), ("FBEXLC", "FBEXL"))
model, outcome, error, state = run([CUT, final(CLEAN)], max_turns=3, model_cls=CutModel)
check("off: the cut is recorded, and counted apart from the malformed turns",
      (state["trajectory"][0].get("hit_cap"), state["cap_hits"], state["malformed_actions"]),
      (True, 1, 1))
check("off: the reply is the pre-43.15 one",
      "do not truncate the object" in model.sent[1][-1]["content"], True)
model, outcome, error, state = run([CUT, final(CLEAN)], max_turns=3, model_cls=CutModel,
                                   cap_feedback=True)
feedback = model.sent[1][-1]["content"]
check("on: the model is told it hit the limit, and not that it truncated",
      ("token limit for one reply" in feedback, "do not truncate" in feedback), (True, False))
check("...naming the cap the turn ran under", "2000-token limit" in feedback, True)
check("...and the case still ends on its declared answer", state["final_source"], "declared")
model, outcome, error, state = run([CUT, final(CLEAN)], max_turns=3, cap_feedback=True)
check("on: malformed but NOT at the cap keeps the parse error, and counts no cap hit",
      ("do not truncate the object" in model.sent[1][-1]["content"], state["cap_hits"]), (True, 0))
check("an endpoint's finish_reason 'length' counts as a cap hit",
      LLM._turn_hit_cap(SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")],
                                        usage=None), 2000), True)
check("the compact turn keeps hit_cap",
      _compact_agent_turns([{"turn": 1, "malformed": "x", "hit_cap": True}], 200)[0]["hit_cap"],
      True)
block = _agent_compliance([{"status": "success", "agent": {"turns": 4, "cap_hits": 1}},
                           {"status": "success", "agent": {"turns": 4, "cap_hits": 0}}])
check("the cap-hit rate is over the turns of the records that count hits",
      block["agent_cap_hit_turn_rate"], 0.125)
check("...and None, not 0, for a run from before 43.15",
      _agent_compliance([{"status": "success", "agent": {"turns": 4}}])["agent_cap_hit_turn_rate"],
      None)
LLM.set_agent_thought(False)
LLM.set_agent_protocol(**dict(_agent_protocol(cfg), cap_feedback=True))
check("C changes no prompt text: the tool rung's hash is still the frozen A1/A2 one",
      LLM.agent_prompt_sha256(True),
      "f70eb067d54d5d5e4fce3c9d34c57093330b6bc5de4e3f0b5b49943e0084b780")
LLM.set_agent_protocol()

print("\n--- R: an action identical to an earlier one is not re-run (notes 43.17) ---")
check("its letter comes after L and C, so every earlier suffix is unchanged",
      (_agent_levers(dict(cfg, agent_repeat_guard=True)),
       _agent_levers(dict(cfg, agent_cap_feedback=True, agent_repeat_guard=True))),
      ("FBEXLR", "FBEXLCR"))
TWICE = [tool("try_query", CLEAN), tool("try_query", CLEAN), final(CLEAN)]
probe = FakeProbe()
model, outcome, error, state = run(TWICE, probe=probe)
check("off: the repeat is re-run, as in every run before 43.17", probe.calls,
      ["try_query", "try_query"])
check("off: ...but recorded, with the turn it repeats",
      (state["trajectory"][1].get("repeat_of"), state["trajectory"][1].get("guarded"),
       state["repeat_turns"]), (1, None, 1))
probe = FakeProbe()
model, outcome, error, state = run(TWICE, probe=probe, repeat_guard=True)
check("on: the repeat is NOT re-run, and spends no tool budget",
      (probe.calls, probe.tool_calls), (["try_query"], 1))
check("on: the model is told which turn made it",
      model.sent[2][-1]["content"], LLM.AGENT_REPEAT_NOTE.format(turn=1))
check("on: ...the roles still alternate",
      all(a["role"] != b["role"] for a, b in zip(model.sent[2][1:], model.sent[2][2:])), True)
check("on: the turn is spent and recorded as guarded, and the case still ends on its answer",
      (state["trajectory"][1].get("guarded"), state["repeat_turns"], state["final_source"]),
      (True, 1, "declared"))
probe = FakeProbe()
model, outcome, error, state = run([tool("try_query", CLEAN), tool("try_query", TWO_ROOTS),
                                    final(CLEAN)], probe=probe, repeat_guard=True)
check("on: the same tool on a different tree is a new action, and runs",
      (probe.calls, state["repeat_turns"]), (["try_query", "try_query"], 0))
probe = FakeProbe()
model, outcome, error, state = run([tool("try_query", CLEAN), tool("check_tree", CLEAN),
                                    tool("try_query", CLEAN), final(CLEAN)], probe=probe,
                                   repeat_guard=True, structural_feedback=True)
check("on: a repeat of any EARLIER turn is caught, not only of the last one (A B A)",
      ([t.get("repeat_of") for t in state["trajectory"]], probe.calls),
      ([None, None, 1, None], ["try_query"]))
thought_turn = lambda text: json.dumps({"thought": text, "tool": "try_query",  # noqa: E731
                                        "args": {"tree": CLEAN}})
probe = FakeProbe()
model, outcome, error, state = run([thought_turn("first"), thought_turn("a new idea"),
                                    final(CLEAN)], probe=probe, thought=True, repeat_guard=True)
check("on: a new thought on the same action is still the same action",
      (probe.calls, state["trajectory"][1].get("repeat_of")), (["try_query"], 1))
model, outcome, error, state = run([tool("check_tree", CYCLIC), tool("check_tree", CYCLIC),
                                    final(CLEAN)], repeat_guard=True, structural_feedback=True)
check("on: the no-tools control rung is guarded too (its check_tree), counted once",
      (state["trajectory"][1].get("guarded"), state["check_tree_calls"]), (True, 1))
for thought, with_tools, label in ((False, True, "A1/A2"), (False, False, "A1-prime"),
                                   (True, True, "A1+thought")):
    LLM.set_agent_thought(thought)
    LLM.set_agent_protocol(**_agent_protocol(cfg))
    without = LLM.agent_prompt_sha256(with_tools)
    LLM.set_agent_protocol(**dict(_agent_protocol(cfg), repeat_guard=True))
    check(f"R changes no prompt text ({label})", LLM.agent_prompt_sha256(with_tools), without)
LLM.set_agent_protocol()
LLM.set_agent_thought(False)
check("the compact turn keeps repeat_of and guarded",
      _compact_agent_turns([{"turn": 2, "action": "try_query", "repeat_of": 1, "guarded": True}],
                           200)[0], {"turn": 2, "action": "try_query", "repeat_of": 1,
                                     "guarded": True})
block = _agent_compliance([{"status": "success", "agent": {"turns": 8, "repeat_turns": 2}},
                           {"status": "success", "agent": {"turns": 2, "repeat_turns": 0}}])
check("the repeat rate is over the turns of the records that count repeats",
      block["agent_repeat_turn_rate"], 0.2)
check("...and None, not 0, for a run from before 43.17",
      _agent_compliance([{"status": "success", "agent": {"turns": 4}}])["agent_repeat_turn_rate"],
      None)

print("\n--- THE FROZEN PROTOCOL (notes 43.14): the prompt the scored rungs run ---")
# Frozen 2026-09-26 after the Stage 5 dev run (notes 43.13), BEFORE any scored agentic run. The
# tool-rung hashes equal the ones the cluster recorded in that run's manifests (f70eb067...,
# d46ba7c2...), so this is the prompt that was measured. Derived from the RUNG config's own levers,
# so the config and the frozen prompt cannot drift apart. A change here after a scored run is a new
# condition, and every compared rung re-runs.
from eval_testset import _agent_protocol  # noqa: E402

FROZEN = {  # the three prompts a rung runs: (thought, tool rung?) -> sha256. A0 runs B0's prompt.
    (False, True): "f70eb067d54d5d5e4fce3c9d34c57093330b6bc5de4e3f0b5b49943e0084b780",   # A1, A2
    (False, False): "093b2eb0816183e57c61445d7f4b08fb471c319e52440df3be3967c70950d89a",  # A1-prime
    (True, True): "d46ba7c2d21d5c7c99cb03fae1092023089b60cb048dd4e8f490dda718956670",    # A1+thought
}
rung_cfg = json.load(open(ROOT / "configs" / "eval_config.agentic.json", encoding="utf-8"))
gate_cfg = json.load(open(ROOT / "configs" / "eval_config.agentic_gate.json", encoding="utf-8"))
check("the rung config runs the levers the dev run measured",
      _agent_protocol(rung_cfg), _agent_protocol(gate_cfg))
check("...and not C, which came after the scored run (notes 43.15)",
      _agent_protocol(rung_cfg)["cap_feedback"], False)
for (thought, with_tools), want in FROZEN.items():
    LLM.set_agent_thought(thought)
    LLM.set_agent_protocol(**_agent_protocol(rung_cfg))
    rung = "A1+thought" if thought else ("A1 and A2" if with_tools else "A1-prime")
    check(f"frozen: the prompt of {rung}", LLM.agent_prompt_sha256(with_tools), want)

LLM.set_agent_protocol()
LLM.set_agent_thought(False)
OUT.unlink(missing_ok=True)
print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> seven levers, each doing what it says, none leaking when off")
