"""Evaluation 2 with example lookup (notes 45): the search_examples tool, lever S.

Asserted here, without a GPU:
  * THE CONFIGS are evaluation 2's frozen configs with exactly these changes: lever S on, 3 examples per
    search, the two loop rungs (or A1*+S alone at the gate), and their own output directories.
  * OFF, NOTHING MOVES: the frozen prompt hashes and tool blocks are evaluation 2's. ON, both loop
    rungs list the tool, and the prompt says the examples are on other logs.
  * THE TOOL: the turn grammar admits it; retrieval is the most similar pool rows first, never one already
    shown in the case; whole examples are kept within the result cap; an unavailable pool or embedding
    model raises instead of falling back to the pool's first rows.
  * THE LOOP: a search is static (no probe call, no tool budget), is recorded with its pool rows and
    similarities, and a repeated search -- which returns new examples -- is not counted as a repeat.
  * IT NEVER TOUCHES GOLD: the tool's code names no testset, gold or log access.
  * NAMES AND SCRIPTS: the lever letter S comes last; the scripts' suffixes are the ones the harness
    builds; both scripts run Qwen3.8 (model index 1) only.

Run from the repo root:  python tests/test_example_search.py
"""
import ast
import inspect
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import LLM  # noqa: E402
from eval_testset import (  # noqa: E402
    AGENT_LEVERS, _agent_budget, _agent_compliance, _agent_levers, _agent_protocol, _agent_suffix,
    _compact_agent_turns, _decoding_suffix,
)
from ocpq_schema import PROBE_TOOL_NAMES, agent_turn_model  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def load(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


print("--- the configs: evaluation 2's, with lever S and nothing else changed ---")
CHANGED = {"agent_example_search", "agent_examples_per_search", "agent_example_search_note",
           "prompt_strategies", "prompt_strategies_note", "output_dir"}
for base, new, strategies, out in (
        ("eval_config.unleashed.json", "eval_config.unleashed_examples.json",
         ["agentic-log-tools", "agentic-self-refine"], "eval_runs/unleashed_examples"),
        ("eval_config.unleashed_gate.json", "eval_config.unleashed_examples_gate.json",
         ["agentic-log-tools"], "eval_runs/unleashed_examples_gate")):
    b, n = load(base), load(new)
    check(f"{new}: every other key is evaluation 2's", {k: v for k, v in n.items() if k not in CHANGED},
          {k: v for k, v in b.items() if k not in CHANGED})
    check(f"{new}: S on, 3 examples, the rungs, its own directory",
          (n["agent_example_search"], n["agent_examples_per_search"], n["prompt_strategies"],
           n["output_dir"]), (True, 3, strategies, out))
RUN, GATE = load("eval_config.unleashed_examples.json"), load("eval_config.unleashed_examples_gate.json")
check("the run is on the testset, the gate on the dev set",
      (RUN["testset_path"], GATE["testset_path"]), ("data/testset.json", "data/agent_devset.json"))
check("the levers are FBEXLS: evaluation 2's FBEXL plus S", _agent_levers(RUN), "FBEXLS")

print("\n--- off, nothing moves; on, both loop rungs have the tool ---")
FROZEN = {True: "f70eb067d54d5d5e4fce3c9d34c57093330b6bc5de4e3f0b5b49943e0084b780",
          False: "093b2eb0816183e57c61445d7f4b08fb471c319e52440df3be3967c70950d89a"}
LLM.set_agent_thought(False)
off = dict(_agent_protocol(RUN), example_search=False)
LLM.set_agent_protocol(**off)
check("S off: both frozen prompt hashes are evaluation 2's (tool rung, control)",
      (LLM.agent_prompt_sha256(True), LLM.agent_prompt_sha256(False)), (FROZEN[True], FROZEN[False]))
tool_off, control_off = LLM._agent_tool_block(object()), LLM._agent_tool_block(None)
LLM.set_agent_protocol(**_agent_protocol(RUN))
check("S on: both hashes change", (LLM.agent_prompt_sha256(True) != FROZEN[True],
                                  LLM.agent_prompt_sha256(False) != FROZEN[False]), (True, True))
tool_on, control_on = LLM._agent_tool_block(object()), LLM._agent_tool_block(None)
check("S on: the tool rung lists search_examples after evaluation 2's tools, and nothing else changes",
      (tool_on.startswith(tool_off), "- search_examples(text" in tool_on[len(tool_off):]), (True, True))
check("S on: the control keeps check_tree and gains search_examples, and still has no log tools",
      ("NO log tools" in control_on, "- check_tree(" in control_on, "- search_examples(" in control_on,
       "try_query" in control_on), (True, True, True, False))
prompt = LLM._agent_task_prompt("<S>", "<Q>", None)
check("S on: the prompt says the examples are on OTHER logs and their type names are not yours",
      "search_examples shows solved questions on OTHER logs" in prompt, True)
check("S off: the control's tool block is evaluation 2's exactly",
      control_off.startswith("You have NO log tools in this condition. One tool you may call:"), True)

print("\n--- the grammar and the parser ---")
turn = agent_turn_model(PROBE_TOOL_NAMES + ("check_tree", "search_examples"))
check("the turn grammar admits a search_examples call",
      turn.model_validate({"tool": "search_examples", "args": {"text": "at least twice"}}).root.tool,
      "search_examples")
check("...and the control's grammar too",
      agent_turn_model(("check_tree", "search_examples")).model_validate(
          {"tool": "search_examples", "args": {"text": "x"}}).root.tool, "search_examples")
action, reason = LLM._parse_agent_action('{"tool": "search_examples", "args": {"text": "twice"}}')
check("the parser reads it", (action.tool, action.args, reason), ("search_examples", {"text": "twice"}, None))

print("\n--- retrieval: most similar first, never one shown twice, whole examples, no fallback ---")
POOL = [{"nl": f"question {i}", "query": {"nodes": [{"Box": [{"i": i}, []]}], "edgeNames": []}}
        for i in range(10)]
EMB = np.array([[1.0, float(i)] for i in range(10)])           # row i: direction (1, i)
saved = (LLM.get_embedding, LLM.model, LLM.tokenizer, LLM.embedding_load_error)
LLM.get_embedding = lambda text: np.array([1.0, 9.0]) if "high" in text else np.array([1.0, 0.0])
LLM.model, LLM.tokenizer, LLM.embedding_load_error = object(), object(), None
LLM.set_agent_example_search(3, 4000)
result, picked = LLM._search_examples_result("high", POOL, EMB, [])
check("the most similar rows first (cosine to (1, 9): rows 9, 8, 7)", [i for i, _ in picked], [9, 8, 7])
check("each example is the pool row's question and tree, unchanged",
      result["examples"][0], {"question": "question 9", "tree": POOL[9]["query"]})
check("the result says the examples are on other logs", "OTHER logs" in result["note"], True)
result2, picked2 = LLM._search_examples_result("high", POOL, EMB, [9, 8, 7])
check("the same search again returns the NEXT most similar, never one already shown",
      [i for i, _ in picked2], [6, 5, 4])
LLM.set_agent_example_search(3, 120)
small, picked_small = LLM._search_examples_result("high", POOL, EMB, [])
check("within the result cap, whole examples are kept and the rest counted, never cut mid-JSON",
      (len(small["examples"]), small.get("omitted"), len(picked_small)), (1, 2, 1))
LLM.set_agent_example_search(3, 4000)
seen_text = []
LLM.get_embedding = lambda text: seen_text.append(text) or np.array([1.0, 0.0])
LLM._search_examples_result("x" * 2000, POOL, EMB, [])
check("the search text is cut to its cap before it is embedded",
      len(seen_text[0]), LLM.AGENT_SEARCH_TEXT_CHAR_CAP)
check("an empty text is the model's mistake, answered, not a crash",
      LLM._search_examples_result("  ", POOL, EMB, [])[0].get("error") is not None, True)
LLM.model = None
try:
    LLM._search_examples_result("high", POOL, EMB, [])
    check("no embedding model: refused, never the pool's first rows", "fell back", "raised")
except LLM.ExampleSearchUnavailable:
    check("no embedding model: refused, never the pool's first rows", "raised", "raised")
LLM.model = object()
try:
    LLM._search_examples_result("high", POOL, None, [])
    check("no pool embeddings: refused", "fell back", "raised")
except LLM.ExampleSearchUnavailable:
    check("no pool embeddings: refused", "raised", "raised")
try:
    LLM.set_agent_example_search(0)
    check("zero examples per search is refused", "accepted", "raised")
except ValueError:
    check("zero examples per search is refused", "raised", "raised")
LLM.set_agent_example_search(3, 4000)

print("\n--- the loop: static, recorded, a repeated search is not a repeat ---")


class Scripted:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        self.sent.append([dict(m) for m in messages])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=self.replies.pop(0)))], usage=SimpleNamespace(completion_tokens=50))


CLEAN = json.loads((ROOT / "data" / "agent_devset.json").read_text(encoding="utf-8"))[0]["ocpq_query"]
SEARCH = json.dumps({"tool": "search_examples", "args": {"text": "high"}})
LLM.get_embedding = lambda text: np.array([1.0, 9.0])
real_call = LLM.call_hf_with_retry
model = Scripted([SEARCH, SEARCH, json.dumps({"final_tree": CLEAN})])
LLM.call_hf_with_retry = model
LLM.extract_ocel_schema = lambda path: {"event_types": ["pay"], "object_types": ["orders"],
                                        "event_attributes": {}, "object_attributes": {},
                                        "e2o": [], "o2o": []}
LLM.build_schema_block = lambda schema, level=None: "ALLOWED OBJECTS: orders"
LLM.run_ocpq_query_with_logging = lambda **kwargs: ("ok", None, None)
LLM.set_agent_probe(None)
LLM.set_agent_max_turns(4)
LLM.set_agent_protocol(structural_feedback=True, example_search=True)
out = ROOT / "eval_runs" / "_example_search_test_query.json"
out.parent.mkdir(parents=True, exist_ok=True)
LLM._run_pipeline_agentic("Which orders?", POOL, EMB, output_file=str(out),
                          ocel_path="../Logs/order-management.json", model_name="m",
                          schema_validation=False)
state = LLM.get_last_agent_trajectory()
t1, t2 = state["trajectory"][0], state["trajectory"][1]
check("each search is recorded with its pool rows and similarities",
      ([i for i, _ in t1["examples"]], [i for i, _ in t2["examples"]]), ([9, 8, 7], [6, 5, 4]))
check("the second, identical search is NOT counted as a repeat (it returned new examples)",
      (t2.get("repeat_of"), state["repeat_turns"]), (None, 0))
check("calls and rows shown are counted on the case",
      (state["example_search_calls"], state["examples_shown"]), (2, 6))
check("the model sees the examples as the next user message",
      json.loads(model.sent[1][-1]["content"])["examples"][0]["question"], "question 9")
check("the case still ends on its declared answer", state["final_source"], "declared")
out.unlink(missing_ok=True)
LLM.call_hf_with_retry = real_call
LLM.get_embedding, LLM.model, LLM.tokenizer, LLM.embedding_load_error = saved
LLM.set_agent_protocol()
LLM.set_agent_max_turns(8)

print("\n--- it never touches gold, the testset or the log ---")
fn = ast.parse(inspect.getsource(LLM._search_examples_result)).body[0]
body = fn.body[1:] if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) \
    else fn.body                     # the code, without the docstring that says what it avoids
src = "\n".join(ast.unparse(node) for node in body)
check("the tool's code names no testset, gold, ground truth, probe or log",
      [w for w in ("testset", "gold", "ground_truth", "probe", "ocel", "devset") if w in src], [])

print("\n--- records, compliance, names and scripts ---")
check("the compact turn keeps the rows shown",
      _compact_agent_turns([{"turn": 1, "action": "search_examples", "examples": [[9, 0.99]]}],
                           200)[0]["examples"], [[9, 0.99]])
block = _agent_compliance([{"status": "success", "agent": {"turns": 3, "example_search_calls": 2}},
                           {"status": "success", "agent": {"turns": 3, "example_search_calls": 0}}])
check("compliance: the share of records that searched, and searches per record",
      (block["agent_example_search_rate"], block["agent_mean_example_searches"]), (0.5, 1))
check("...None for runs before notes 45",
      _agent_compliance([{"status": "success", "agent": {"turns": 1}}])["agent_example_search_rate"], None)
check("S is the last lever, so every earlier suffix is unchanged",
      (AGENT_LEVERS[-1], _agent_levers(dict(RUN, agent_example_search=False))),
      (("agent_example_search", "example_search", "S"), "FBEXL"))
decoding = _decoding_suffix(RUN["constrained_decoding"], False, RUN["generation_token_budget"],
                            any_order=RUN["constrained_any_order"],
                            bounded=RUN["constrained_bounded_shape"],
                            reasoning_budget=RUN["reasoning_token_budget"],
                            reasoning_history=RUN["reasoning_history"])
built = {i: decoding + _agent_suffix(s, *_agent_budget(RUN)) for i, s in enumerate(RUN["prompt_strategies"])}
RUNGS = (ROOT / "slurm" / "submit_unleashed_examples_rungs.sh").read_bytes().decode("utf-8")
GATE_SH = (ROOT / "slurm" / "submit_unleashed_examples_gate.sh").read_bytes().decode("utf-8")
suffixes = dict(re.findall(r'^\s+(\d)\) RUNG_NAME=.*\n\s+RUNG_SUFFIX="([^"]+)"', RUNGS, re.M))
check("rungs: every RUNG_SUFFIX is the suffix the harness builds", suffixes,
      {str(i): b for i, b in built.items()})
check("gate: its RUNG_SUFFIX is A1*+S's", re.search(r'^RUNG_SUFFIX="([^"]+)"', GATE_SH, re.M).group(1),
      built[0])
check("both scripts run Qwen3.8 only (model index 1)",
      ("MODEL_IDX=1" in GATE_SH, "MODEL_IDX=1" in RUNGS, '  1) ;;' in RUNGS), (True, True, True))
check("both scripts read their own configs and write their own directories",
      ("eval_config.unleashed_examples_gate.json" in GATE_SH, "eval_config.unleashed_examples.json" in RUNGS,
       "eval_runs/unleashed_examples_gate/" in GATE_SH, "eval_runs/unleashed_examples/" in RUNGS),
      (True, True, True, True))
for name, text in (("rungs", RUNGS), ("gate", GATE_SH)):
    check(f"{name}: no CR characters", text.count("\r"), 0)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> example lookup is static, recorded, gold-free, and evaluation 2 is unchanged without it")
