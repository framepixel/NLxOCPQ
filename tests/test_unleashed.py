"""Evaluation 2, "unleashed" (notes 44): reason first in the model's own channel, then act as before.

Asserted here, without a GPU:
  * THE CONFIGS carry evaluation 1's protocol unchanged (the same prompts -- the frozen hashes --, tools,
    grammar, per-turn action cap, levers, scorer settings) and differ from it only where notes 44
    decided: the reasoning budget, sampling with seeds and two draws, the 1800 s cap, the rungs
    A0*/A1*/A1'*, the four models in their pre-registered order, and their own output directories.
  * THE TWO-STEP TURN: the reasoning step runs WITHOUT the grammar, stops at the channel's closing
    token or at the budget (a "forced close", after which the harness closes the channel itself),
    and the action step continues from it UNDER the grammar. Only the action is returned as the
    content; the reasoning comes back beside it. With no reasoning configured, generation is exactly
    what it was (one call, no reasoning).
  * THE REAL TEMPLATES AND TOKENIZERS (pinned revisions, from the HF cache): each model's closing
    marker is one token, its reasoning switch reaches the prompt, an earlier turn carries no
    reasoning unless it is kept, and a KEPT reasoning is rendered exactly as the model's own template
    renders reasoning inside a tool loop (notes 44.6) -- Gemma's harness-written text included.
  * THE AGENT LOOP records the reasoning per turn; under "drop" keeps it out of the transcript, under
    "keep" hands it to the next turn; counts the per-turn action cap on the ACTION's tokens only; and
    no API backend ever receives the private reasoning marker.
  * NAMES AND SCRIPTS: `_think2048keep` names every evaluation-2 condition; the scripts' suffixes are the
    ones the harness builds; the shard and runtime mapping is executed in bash.

The template section reads the pinned files from the HF cache; on the cluster run it as
    HF_HOME=$HPCWORK/nlxocpq-hf python tests/test_unleashed.py
Run from the repo root:  python tests/test_unleashed.py
"""
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import LLM  # noqa: E402
from eval_testset import (  # noqa: E402
    _agent_budget, _agent_compliance, _agent_protocol, _agent_suffix, _compact_agent_turns,
    _decoding_suffix)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def load(name):
    return json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))


EVAL1 = load("eval_config.agentic.json")
RUN, GATE = load("eval_config.unleashed.json"), load("eval_config.unleashed_gate.json")
FROZEN = {  # notes 43.14; the prompts evaluation 2 must share with evaluation 1
    (False, True): "f70eb067d54d5d5e4fce3c9d34c57093330b6bc5de4e3f0b5b49943e0084b780",
    (False, False): "093b2eb0816183e57c61445d7f4b08fb471c319e52440df3be3967c70950d89a",
}
ORDER = ["ministral3-14b-reasoning", "qwen3.8-27b-thinking", "muse-glimmer-30b-reasoning",
         "gemma4-31b-thinking"]

print("--- the configs: evaluation 1's protocol, unleashed only where notes 44 decided ---")
SAME = ("constrained_decoding", "constrained_any_order", "constrained_bounded_shape",
        "generation_token_budget", "schema_enforced", "schema_validation", "corpus_check",
        "force_root_arity", "type_aware_filters", "structural_validation", "constant_root_arity",
        "agent_max_turns", "agent_max_tool_calls", "agent_turn_token_budget",
        "agent_structural_feedback", "agent_bounce_blocking_final", "agent_finalise_evaluated",
        "agent_worked_example", "agent_reserve_last_turn", "agent_tool_result_char_cap",
        "local_quantization")
for name, cfg in (("run", RUN), ("gate", GATE)):
    check(f"{name}: evaluation 1's protocol settings, unchanged",
          {k: cfg.get(k) for k in SAME}, {k: EVAL1.get(k) for k in SAME})
    check(f"{name}: the levers are pFBEXL -- C and R off, no thought field",
          (cfg["agent_cap_feedback"], cfg["agent_repeat_guard"], cfg["agent_thought"]),
          (False, False, False))
    check(f"{name}: reasoning 2048 (frozen at the gate stage, notes 44.5), temperature 1.0, fixed seeds, the 1800 s cap, preloaded",
          (cfg["reasoning_token_budget"], cfg["temperature"], bool(cfg["generation_seed_base"]),
           cfg["case_timeout_sec"], cfg["preload_model"]), (2048, 1.0, True, 1800, True))
    check(f"{name}: the four models, in the pre-registered order", [m["label"] for m in cfg["models"]],
          ORDER)
    check(f"{name}: earlier reasoning is KEPT (notes 44.6), and every model says how its template "
          "carries it", (cfg["reasoning_history"],
                         [m["reasoning"]["history"]["via"] for m in cfg["models"]]),
          ("keep", ["thinking_chunk", "reasoning_content", "reasoning_content", "splice"]))
    check(f"{name}: Qwen's preserve_thinking is its template's default (on)",
          cfg["models"][1]["chat_template_kwargs"].get("preserve_thinking"), True)
    check(f"{name}: one derived seed per generation (notes 44.7)", cfg["generation_seed_per_call"],
          True)
    check(f"{name}: each model's context cap is what its memory check verified (notes 44.3, 44.5)",
          [m["max_context_tokens"] for m in cfg["models"]], [65536, 65536, 65536, 49152])
    check(f"{name}: no adapters are expanded", cfg["auto_expand_finetuned"], False)
check("run: the testset, A0*/A1*/A1'*, two draws, its own directory",
      (RUN["testset_path"], RUN["prompt_strategies"], RUN["repeats"], RUN["output_dir"]),
      ("data/testset.json",
       ["schema-allowed-entities-only", "agentic-log-tools", "agentic-self-refine"], 2,
       "eval_runs/unleashed"))
check("gate: the dev set, A1* only, one draw, its own directory",
      (GATE["testset_path"], GATE["prompt_strategies"], GATE["repeats"], GATE["output_dir"]),
      ("data/agent_devset.json", ["agentic-log-tools"], 1, "eval_runs/unleashed_gate"))
check("gate and run load the same model entries", GATE["models"], RUN["models"])
for m in RUN["models"]:
    check(f"{m['label']}: pinned, no adapter, a reasoning channel, sampling, a runtime",
          (bool(re.fullmatch(r"[0-9a-f]{40}", m["revision"])), m["finetuned"],
           bool(m["reasoning"].get("close")), m["temperature"],
           m["runtime"]["hf_cache"], m["runtime"]["conda_env"] in ("nlxocpq-eval-v5",
                                                                   "nlxocpq-eval-glimmer")),
          (True, False, True, 1.0, "hpcwork", True))
for thought_tools, want in FROZEN.items():
    LLM.set_agent_thought(False)
    LLM.set_agent_protocol(**_agent_protocol(RUN))
    check(f"the frozen prompt of evaluation 1 ({'tools' if thought_tools[1] else 'control'}) "
          "is evaluation 2's", LLM.agent_prompt_sha256(thought_tools[1]), want)
LLM.set_agent_protocol()

print("\n--- the two-step turn: reason without the grammar, act under it ---")


class Tok:
    pad_token_id, eos_token_id = 0, 1
    MAP = {"[OPEN]": [50], "[CLOSE]": [51], "ANS": [52], "": []}

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append((messages, kwargs))
        return torch.tensor([[10, 11]])

    def encode(self, text, add_special_tokens=True):
        return self.MAP.get(text, [99])

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


class Model:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=131072)

    def __init__(self, reasoning_out):
        self.calls, self.reasoning_out = [], reasoning_out

    def generate(self, **kw):
        self.calls.append({k: (v.tolist() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()})
        new = [80, 81] if "logits_processor" in kw else self.reasoning_out
        return torch.cat([kw["input_ids"], torch.tensor([new])], dim=-1)


def generate(reasoning_out, reasoning=True, budget=3, sampling=None, prefix=None):
    tok, model = Tok(), Model(reasoning_out)
    LLM.local_tokenizer, LLM.local_model = tok, model
    LLM._ensure_local_gpu_model_loaded = lambda: None
    LLM.set_local_model_options(
        reasoning={"open": "[OPEN]", "close": "[CLOSE]", "answer_prefix": "ANS"} if reasoning else None,
        sampling=sampling, system_prompt_prefix=prefix)
    LLM.set_reasoning_token_budget(budget)
    response = LLM._call_local_gpu_chat_completion(
        [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q"}], 50, None)
    return tok, model, response


_real_processor = LLM._ocpq_logits_processor
LLM._ocpq_logits_processor = lambda: "GRAMMAR"
LLM.set_generation_temperature(1.0)
tok, model, response = generate([70, 71, 51])
first, second = model.calls
check("reasoning step: NO grammar, stops at the closing token or EOS, at most the budget",
      ("logits_processor" in first, first["eos_token_id"], first["max_new_tokens"]),
      (False, [51, 1], 3))
check("reasoning step: starts after the channel's opening text", first["input_ids"], [[10, 11, 50]])
check("action step: UNDER the grammar, after the reasoning, the closing token and the answer prefix",
      (second.get("logits_processor"), second["input_ids"]),
      (["GRAMMAR"], [[10, 11, 50, 70, 71, 51, 52]]))
check("only the action is the content; the reasoning comes back beside it",
      (response.choices[0].message.content, response.choices[0].message.reasoning), ("80 81", "70 71"))
check("the model closed the channel itself: no forced close; tokens counted apart",
      (response.forced_close, response.reasoning_tokens, response.action_tokens,
       response.usage.completion_tokens), (False, 2, 2, 4))
tok, model, response = generate([70, 71, 72])
check("at the budget the harness closes the channel: a forced close",
      (response.forced_close, model.calls[1]["input_ids"]),
      (True, [[10, 11, 50, 70, 71, 72, 51, 52]]))
tok, model, response = generate([70, 1])
check("a reasoning step that ends the turn is closed too, and counts as forced",
      (response.forced_close, response.choices[0].message.reasoning), (True, "70"))
tok, model, response = generate([70, 51], sampling={"top_p": 0.95, "top_k": 20})
check("the vendor's top_p and top_k reach both steps",
      [(c.get("top_p"), c.get("top_k")) for c in model.calls], [(0.95, 20), (0.95, 20)])
tok, model, response = generate([70, 51], prefix="VENDOR SWITCH")
check("a prompt-text reasoning switch is prepended to our system prompt, which follows unchanged",
      tok.template_calls[0][0][0]["content"], "VENDOR SWITCH\n\nSYS")
tok, model, response = generate([70, 51], reasoning=False)
check("no reasoning channel: ONE call, exactly as before, and no reasoning in the response",
      (len(model.calls), hasattr(response.choices[0].message, "reasoning"),
       "top_p" in model.calls[0]), (1, False, False))
tok, model, response = generate([70, 51], budget=None)
check("no reasoning budget (evaluation 1): ONE call, whatever the model declares",
      (len(model.calls), hasattr(response, "forced_close")), (1, False))
try:
    LLM.set_local_model_options(reasoning={"open": "x"})
    check("a reasoning channel without a closing marker is refused", "accepted", "ValueError")
except ValueError:
    check("a reasoning channel without a closing marker is refused", "ValueError", "ValueError")


class SpliceTok(Tok):
    """Renders to text, one char per id, as a template with a model-turn header would (Gemma)."""

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        self.template_calls.append((messages, kwargs))
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"
        return text if tokenize is False else torch.tensor([[10, 11]])

    def encode(self, text, add_special_tokens=True):
        return self.MAP[text] if text in self.MAP else [ord(c) for c in text]


tok, model = SpliceTok(), Model([70, 51])
LLM.local_tokenizer, LLM.local_model = tok, model
LLM.set_local_model_options(reasoning={"open": "[OPEN]", "close": "[CLOSE]", "answer_prefix": "ANS",
                                       "history": {"via": "splice", "after": "<assistant>",
                                                   "format": "T{reasoning}/T"}})
LLM.set_reasoning_token_budget(3)
LLM._call_local_gpu_chat_completion(
    [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", "_reasoning": "R1"},
     {"role": "user", "content": "RES"}], 50, None)
check("splice: the reasoning step starts after the spliced prompt and the channel opener",
      model.calls[0]["input_ids"][0][-1], 50)
prompt_text = "".join(chr(i) for i in model.calls[0]["input_ids"][0][:-1])
check("splice (Gemma): the generation prompt carries the kept reasoning after the earlier turn's "
      "header, and none after the new one", prompt_text,
      "<user>Q<assistant>TR1/TA<user>RES<assistant>")
check("splice: the template never sees the private marker",
      ["_reasoning" in m for m in tok.template_calls[0][0]], [False, False, False])
try:
    LLM._splice_reasoning("<assistant>A<user>RES<assistant>", [{"role": "assistant"}], [])
    check("splice: a count mismatch is refused, never guessed", "accepted", "ValueError")
except ValueError:
    check("splice: a count mismatch is refused, never guessed", "ValueError", "ValueError")
LLM._ocpq_logits_processor = _real_processor
LLM.set_local_model_options()
LLM.set_reasoning_token_budget(None)
LLM.set_generation_temperature(None)

print("\n--- the real templates and tokenizers (pinned revisions, from the HF cache) ---")
try:
    from huggingface_hub import hf_hub_download
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    files = {m["label"]: {f: hf_hub_download(m["name"], f, revision=m["revision"],
                                             local_files_only=True)
                          for f in ("chat_template.jinja", "tokenizer.json")}
             for m in RUN["models"]}
except Exception as exc:  # noqa: BLE001 - reported with the way to fix it
    files = None
    check(f"the pinned templates and tokenizers are cached ({type(exc).__name__}); run "
          "scripts/prefetch_pinned_models.sh, and HF_HOME=$HPCWORK/nlxocpq-hf on the cluster",
          False, True)
if files:
    # What the switch must put into the rendered prompt, and what the prompt must end with.
    EXPECT = {
        "ministral3-14b-reasoning": ("[THINK]Your thoughts", "[/INST]"),
        # Qwen's "medium" adds no instruction (only xhigh and low do): thinking on is the switch.
        "qwen3.8-27b-thinking": ("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n"),
        "muse-glimmer-30b-reasoning": ("Reasoning strength: medium.", "<|start|>assistant"),
        "gemma4-31b-thinking": ("<|think|>", "<|turn>model\n"),
    }
    for m in RUN["models"]:
        tok = Tokenizer.from_file(files[m["label"]]["tokenizer.json"])
        close = tok.encode(m["reasoning"]["close"], add_special_tokens=False)
        check(f"{m['label']}: its closing marker {m['reasoning']['close']!r} is ONE token",
              len(close.ids), 1)
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[loopcontrols])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(message))
        env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
        template = env.from_string(Path(files[m["label"]]["chat_template.jinja"]).read_text(
            encoding="utf-8"))
        system = "SYS"
        if m.get("system_prompt_prefix"):
            system = m["system_prompt_prefix"] + "\n\n" + system   # as the harness prepends it
        messages = [{"role": "system", "content": system}, {"role": "user", "content": "TASK"},
                    {"role": "assistant", "content": "ACTION"}, {"role": "user", "content": "RESULT"}]
        prompt = template.render(messages=messages, add_generation_prompt=True, bos_token="",
                                 eos_token="</s>", **m.get("chat_template_kwargs", {}))
        switch, tail = EXPECT[m["label"]]
        check(f"{m['label']}: the reasoning switch reaches the prompt", switch in prompt, True)
        if m["label"].startswith("qwen"):
            check(f"{m['label']}: ...at medium effort, not the template's default xhigh",
                  "Reasoning effort is set to xhigh" in prompt, False)
        check(f"{m['label']}: the prompt ends where the reasoning channel begins", prompt.endswith(tail),
              True)
        earlier = prompt.split("ACTION")[0][-40:]
        check(f"{m['label']}: without a kept reasoning, the earlier turn carries none",
              any(mark in earlier for mark in ("[THINK]Your", "to=self", "<|channel>thought\nA")), False)

        # KEEP (notes 44.6). The template's NATIVE rendering of a reasoning it keeps: the assistant
        # turn after the last user message, where every one of the four keeps it.
        LLM.set_local_model_options(reasoning=m["reasoning"])
        R = "MY EARLIER REASONING"
        native_msgs, native_splices = LLM._render_reasoning_history(
            [{"role": "system", "content": system}, {"role": "user", "content": "TASK"},
             {"role": "assistant", "content": "ACTION", "_reasoning": R}])
        if native_splices:   # Gemma: its own field, which it renders after the last user message
            native_msgs[-1]["reasoning_content"] = native_splices[0]
        native = template.render(messages=native_msgs, add_generation_prompt=False, bos_token="",
                                 eos_token="</s>", **m.get("chat_template_kwargs", {}))
        check(f"{m['label']}: (the template itself renders a reasoning in its last turn)", R in native,
              True)
        segment = native[native.index(R) - 40: native.index("ACTION") + len("ACTION")]
        kept_msgs, splices = LLM._render_reasoning_history(
            [{"role": "system", "content": system}, {"role": "user", "content": "TASK"},
             {"role": "assistant", "content": "ACTION", "_reasoning": R},
             {"role": "user", "content": "RESULT"}])
        kept = template.render(messages=kept_msgs, add_generation_prompt=True, bos_token="",
                               eos_token="</s>", **m.get("chat_template_kwargs", {}))
        if splices:
            kept = LLM._splice_reasoning(kept, kept_msgs, splices)
        check(f"{m['label']}: a KEPT earlier reasoning is rendered exactly as the template renders it",
              segment in kept, True)
        check(f"{m['label']}: ...once, and the prompt still ends where the reasoning channel begins",
              (kept.count(R), kept.endswith(tail)), (1, True))
        LLM.set_local_model_options()

print("\n--- the agent loop: reasoning recorded; dropped or kept; the cap on the action ---")
REAL_CALL = LLM.call_hf_with_retry


class Scripted:
    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        self.sent.append([dict(m) for m in messages])
        content, reasoning, forced = self.replies.pop(0)
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=content, reasoning=reasoning))],
            usage=SimpleNamespace(completion_tokens=1600 + len(content)))
        response.reasoning_tokens, response.action_tokens, response.forced_close = 1500, 100, forced
        return response


CLEAN = json.loads((ROOT / "data" / "agent_devset.json").read_text(encoding="utf-8"))[0]["ocpq_query"]
model = Scripted([(json.dumps({"tool": "check_tree", "args": {"tree": CLEAN}}), "I THINK HARD", True),
                  (json.dumps({"final_tree": CLEAN}), "DONE THINKING", False)])
LLM.call_hf_with_retry = model
LLM.extract_ocel_schema = lambda path: {"event_types": ["pay"], "object_types": ["orders"],
                                        "event_attributes": {}, "object_attributes": {},
                                        "e2o": [], "o2o": []}
LLM.build_schema_block = lambda schema, level=None: "ALLOWED OBJECTS: orders"
LLM.run_ocpq_query_with_logging = lambda **kwargs: ("ok", None, None)
LLM.set_agent_probe(None)
LLM.set_agent_max_turns(4)
LLM.set_agent_protocol(structural_feedback=True)
out = ROOT / "eval_runs" / "_unleashed_test_query.json"
out.parent.mkdir(parents=True, exist_ok=True)
LLM._run_pipeline_agentic("Which orders?", None, None, output_file=str(out),
                          ocel_path="../Logs/order-management.json", model_name="m",
                          schema_validation=False)
state = LLM.get_last_agent_trajectory()
turn = state["trajectory"][0]
check("the turn records its reasoning, its length and a forced close",
      (turn.get("reasoning"), turn.get("reasoning_tokens"), turn.get("forced_close")),
      ("I THINK HARD", 1500, True))
check("drop: the transcript the model sees next holds the action, never the reasoning",
      (any("I THINK HARD" in str(m) for m in model.sent[1]), "_reasoning" in model.sent[1][2]),
      (False, False))
check("1500 reasoning + 100 action tokens under a 2000-token action cap: no cap hit",
      turn.get("hit_cap"), None)
compact = _compact_agent_turns(state["trajectory"], 200)
check("the compact turn keeps the length and the forced close, not the text",
      (compact[0].get("reasoning_tokens"), compact[0].get("forced_close"), "reasoning" in compact[0]),
      (1500, True, False))
block = _agent_compliance([{"status": "success", "agent": {"turns": 2, "turn_summary": compact}}])
check("compliance: the forced-close rate and the mean reasoning length",
      (block["agent_forced_close_rate"], block["agent_mean_reasoning_tokens"]), (0.5, 1500))
check("...and None for a run without a reasoning step",
      _agent_compliance([{"status": "success", "agent": {"turns": 1, "turn_summary": [{}]}}])
      ["agent_forced_close_rate"], None)
# KEEP (notes 44.6): the same case, the reasoning handed to the next turn on the assistant message.
LLM.set_reasoning_history("keep")
model = Scripted([(json.dumps({"tool": "check_tree", "args": {"tree": CLEAN}}), "I THINK HARD", True),
                  (json.dumps({"final_tree": CLEAN}), "DONE THINKING", False)])
LLM.call_hf_with_retry = model
LLM._run_pipeline_agentic("Which orders?", None, None, output_file=str(out),
                          ocel_path="../Logs/order-management.json", model_name="m",
                          schema_validation=False)
check("keep: the next turn's assistant message carries this turn's reasoning, in full",
      (model.sent[1][2]["role"], model.sent[1][2].get("_reasoning")), ("assistant", "I THINK HARD"))
check("keep: ...and its content is still the action alone",
      model.sent[1][2]["content"].startswith('{"tool"'), True)
check("keep: the loop's other messages are untouched", [m.get("_reasoning") for m in model.sent[1]
                                                        if m["role"] != "assistant"], [None, None, None])
LLM.set_reasoning_history(None)
check("set_reasoning_history(None) is drop", LLM.get_reasoning_history(), "drop")
try:
    LLM.set_reasoning_history("summarise")
    check("an unknown reasoning_history is refused", False, True)
except ValueError:
    check("an unknown reasoning_history is refused", True, True)

# A model asked to keep reasoning without saying how its template carries it fails loudly.
LLM.set_local_model_options(reasoning={"close": "</think>"})
try:
    LLM._render_reasoning_history([{"role": "assistant", "content": "A", "_reasoning": "R"}])
    check("keep without a declared 'history' is refused", False, True)
except ValueError:
    check("keep without a declared 'history' is refused", True, True)
plain = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
check("drop: messages reach the template as they are (the very same list)",
      LLM._render_reasoning_history(plain)[0] is plain, True)
LLM.set_local_model_options()

# No API backend receives the private marker.
sent_kwargs = []
_saved = (LLM.hf_backend, LLM.openai_client, LLM._record_llm_call)
LLM.hf_backend = "custom-endpoint"
LLM.openai_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
    create=lambda **kw: sent_kwargs.append(kw) or SimpleNamespace(choices=[], usage=None))))
LLM._record_llm_call = lambda **kw: None
REAL_CALL("m", [{"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A", "_reasoning": "R"}], 10, None)
LLM.hf_backend, LLM.openai_client, LLM._record_llm_call = _saved
check("an endpoint backend never receives the private reasoning marker",
      [sorted(m) for m in sent_kwargs[0]["messages"]], [["content", "role"], ["content", "role"]])
LLM.call_hf_with_retry = REAL_CALL

LLM.set_agent_protocol()
out.unlink(missing_ok=True)

print("\n--- the verified-context limit (notes 44.7): no warning, the case ends, it is recorded ---")
LLM.set_local_model_options(reasoning={"close": "</think>"}, max_context_tokens=49152)
LLM.set_reasoning_token_budget(2048)
LLM._check_memory_budget(45000, 2000)   # 45000 + 2048 + 2000 = 49048: fits
try:
    LLM._check_memory_budget(45200, 2000)   # 49248: could outgrow the verified 49152
    check("a call that COULD outgrow the verified context is refused before generating", "ran", "raised")
except LLM.PromptTooLongError:
    check("a call that COULD outgrow the verified context is refused before generating", "raised",
          "raised")
LLM.set_local_model_options(max_context_tokens=49152)   # no reasoning channel: no reasoning budget
LLM._check_memory_budget(47000, 2000)
check("...counting the reasoning budget only for a model that reasons", True, True)
LLM.set_local_model_options()
LLM._check_memory_budget(10 ** 6, 2000)
check("...and no limit configured (every earlier model): no check", True, True)
LLM.set_reasoning_token_budget(None)


class Overflowing(Scripted):
    def __call__(self, model, messages, max_tokens, response_format, **kwargs):
        if len(self.sent) == 1:
            self.sent.append(messages)
            raise LLM.PromptTooLongError("context limit")
        return super().__call__(model, messages, max_tokens, response_format, **kwargs)


TRY = json.dumps({"tool": "check_tree", "args": {"tree": CLEAN}})
model = Overflowing([(TRY, "R", False)])
LLM.call_hf_with_retry = model
LLM.set_agent_max_turns(4)
LLM.set_agent_protocol(structural_feedback=True)
try:
    LLM._run_pipeline_agentic("Which orders?", None, None, output_file=str(out),
                              ocel_path="../Logs/order-management.json", model_name="m",
                              schema_validation=False)
    ended = "finalised"
except LLM.AgentNoFinalAnswer:
    ended = "no answer"   # check_tree yields no candidate tree, so there is none to finalise
state = LLM.get_last_agent_trajectory()
check("the limit ends the case like running out of turns (no tree here: no answer), recorded, the "
      "model never warned",
      (ended, state["context_limit_hit"], state["turns"],
       any("limit" in str(m.get("content")) for m in model.sent[0])),
      ("no answer", True, 2, False))
model = Scripted([(json.dumps({"final_tree": CLEAN}), "R", False)])
LLM.call_hf_with_retry = model
LLM._run_pipeline_agentic("Which orders?", None, None, output_file=str(out),
                          ocel_path="../Logs/order-management.json", model_name="m",
                          schema_validation=False)
check("...and a case that does not reach it records False",
      LLM.get_last_agent_trajectory()["context_limit_hit"], False)
LLM.call_hf_with_retry = REAL_CALL
LLM.set_agent_protocol()
out.unlink(missing_ok=True)
block = _agent_compliance([{"status": "success", "agent": {"turns": 2, "context_limit_hit": True}},
                           {"status": "success", "agent": {"turns": 2, "context_limit_hit": False}}])
check("compliance: the context-limit rate", block["agent_context_limit_rate"], 0.5)
check("...and None for runs that did not record it",
      _agent_compliance([{"status": "success", "agent": {"turns": 1}}])["agent_context_limit_rate"],
      None)

print("\n--- names and scripts ---")
check("_think2048keep follows _tok4000 in every evaluation-2 condition name",
      _decoding_suffix(True, False, 4000, any_order=False, bounded=True, reasoning_budget=2048,
                       reasoning_history="keep"),
      "_constrained_ordered_bounded_tok4000_think2048keep")
check("...and a 'drop' condition keeps the name it had (the 44.5 smokes)",
      _decoding_suffix(True, False, 4000, any_order=False, bounded=True, reasoning_budget=2048,
                       reasoning_history="drop"),
      "_constrained_ordered_bounded_tok4000_think2048")
check("...and is absent without a reasoning budget (every earlier name unchanged)",
      _decoding_suffix(True, False, 4000, any_order=False, bounded=True),
      "_constrained_ordered_bounded_tok4000")
RUNGS = (ROOT / "slurm" / "submit_unleashed_rungs.sh").read_bytes().decode("utf-8")
GATE_SH = (ROOT / "slurm" / "submit_unleashed_gate.sh").read_bytes().decode("utf-8")
decoding = _decoding_suffix(RUN["constrained_decoding"], False, RUN["generation_token_budget"],
                            any_order=RUN["constrained_any_order"],
                            bounded=RUN["constrained_bounded_shape"],
                            reasoning_budget=RUN["reasoning_token_budget"],
                            reasoning_history=RUN["reasoning_history"])
built = {i: decoding + _agent_suffix(s, *_agent_budget(RUN))
         for i, s in enumerate(RUN["prompt_strategies"])}
suffixes = dict(re.findall(r'^\s+(\d)\) RUNG_NAME=.*\n\s+RUNG_SUFFIX="([^"]+)"', RUNGS, re.M))
check("rungs: every RUNG_SUFFIX is the suffix the harness builds", suffixes,
      {str(i): b for i, b in built.items()})
check("gate: its RUNG_SUFFIX is A1*'s", re.search(r'^RUNG_SUFFIX="([^"]+)"', GATE_SH, re.M).group(1),
      built[1])
for name, text in (("rungs", RUNGS), ("gate", GATE_SH),
                   ("long-prompt smoke", (ROOT / "slurm" / "submit_long_prompt_smoke.sh")
                    .read_bytes().decode("utf-8"))):
    check(f"{name}: no CR characters", text.count("\r"), 0)
_bash = shutil.which("bash")
if _bash:
    start = RUNGS.index('if [ "$SMOKE" = "1" ]; then')
    end = RUNGS.index("STRATEGY_IDX=\"$RUNG\"")
    snippet = "SHARDS=6\n" + RUNGS[start:end] + 'echo "$RUNG $CASE_SHARD"\n'

    def mapping(task, smoke="0"):
        out = subprocess.run([_bash, "-c", f"TASK={task}; SMOKE={smoke}\n" + snippet],
                             capture_output=True, text=True)
        return out.stdout.strip() or f"exit {out.returncode}"
    check("task -> rung / case slice: 3 rungs x 6 slices",
          [mapping(t) for t in (0, 5, 6, 11, 12, 17)],
          ["0 0/6", "0 5/6", "1 0/6", "1 5/6", "2 0/6", "2 5/6"])
    check("...task 18 is not a rung", mapping(18), "exit 1")
    check("--smoke: task t is rung t on slice 8/11", [mapping(t, "1") for t in (0, 1, 2)],
          ["0 8/11", "1 8/11", "2 8/11"])
from model_runtime import runtime  # noqa: E402
os.environ["HPCWORK"] = "/w"
check("each model's runtime comes from its config entry",
      [runtime(str(ROOT / "configs" / "eval_config.unleashed.json"), i) for i in range(4)],
      ["nlxocpq-eval-v5|/w/nlxocpq-hf", "nlxocpq-eval-glimmer|/w/nlxocpq-hf",
       "nlxocpq-eval-glimmer|/w/nlxocpq-hf", "nlxocpq-eval-v5|/w/nlxocpq-hf"])

print("\n--- Gemma 4's attention: a kernel exists for grouped K/V at head dim 512 (notes 44.3) ---")
# The long-prompt smoke found none: flash caps the head dim at 256 and neither fused kernel takes
# grouped K/V. Under sdpa_kernels, transformers must repeat K/V itself, and only there.
import transformers.integrations.sdpa_attention as hf_sdpa  # noqa: E402
_gqa = hf_sdpa.use_gqa_in_sdpa
LLM.local_sdpa_kernels = ["flash_attention", "efficient_attention"]
with LLM._sdpa_context():
    check("inside: transformers repeats K/V (no enable_gqa)", hf_sdpa.use_gqa_in_sdpa(None, None), False)
check("after: transformers' own switch is restored", hf_sdpa.use_gqa_in_sdpa is _gqa, True)
try:
    with LLM._sdpa_context():
        raise KeyError("boom")
except KeyError:
    pass
check("...restored even when generation raises", hf_sdpa.use_gqa_in_sdpa is _gqa, True)
if torch.cuda.is_available():
    q = torch.randn(1, 32, 512, 512, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, 4, 512, 512, device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(num_key_value_groups=8, is_causal=True)
    with LLM._sdpa_context():
        out, _ = hf_sdpa.sdpa_attention_forward(layer, q, kv, kv, None)
    check("GPU: Gemma's global-layer shape runs under the allowed kernels", tuple(out.shape),
          (1, 512, 32, 512))
else:
    print("SKIP  GPU: Gemma's global-layer shape (no CUDA here; the long-prompt smoke covers it)")
LLM.local_sdpa_kernels = None
with LLM._sdpa_context():
    check("no kernels configured: transformers' switch untouched", hf_sdpa.use_gqa_in_sdpa is _gqa, True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> evaluation 2 reasons first and acts as evaluation 1 does; nothing else moved")
