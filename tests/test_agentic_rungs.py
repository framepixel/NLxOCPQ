"""The agentic rungs: five distinct conditions in five directories, none of them touching gold.

`slurm/submit_agentic_rungs.sh` (notes 43) runs A0, A1, A1-prime, A2 and A1+thought. Three rungs
share the model, strategy AND schema index -- A1, A2 and A1+thought all run `agentic-log-tools` --
so the only thing separating their shard directories is the suffix. When suffixes drifted before
(notes 32.2), `--resume` for one rung globbed another rung's directory and would have resumed into
it without a word. So every rung is parsed OUT OF THE SCRIPT, its flags are applied to the config
the way `eval_testset` applies them, and the directory suffix the harness would build is compared
with the script's RUNG_SUFFIX -- no string is written twice here.

Also asserted: nothing is an oracle; every rung shares one case cap (CLAUDE.md 6); A2 cannot run
before its budget is sized; the arm runs on a base model the gate picked, never a
fine-tuned adapter; the gate runs on the dev set only; the main matrix config is untouched;
and the `thought` condition changes the action schema and one rule line, and nothing else.

Run from the repo root:  python tests/test_agentic_rungs.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import LLM  # noqa: E402
from eval_testset import (  # noqa: E402
    _agent_shard_suffix,
    _agent_suffix,
    _agent_budget,
    _decoding_suffix,
)
from ocpq_schema import AgentAction, AgentActionWithThought  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def script_text(path):
    return (ROOT / path).read_bytes().decode("utf-8")


def flags_to_config(config, agent_args):
    """Apply the script's agent flags the way eval_testset._apply_shard_filters does."""
    cfg = dict(config)
    tokens = agent_args.split()
    for i, token in enumerate(tokens):
        if token == "--agent-max-turns":
            cfg["agent_max_turns"] = int(tokens[i + 1])
        elif token == "--agent-max-tool-calls":
            cfg["agent_max_tool_calls"] = int(tokens[i + 1])
        elif token == "--agent-thought":
            cfg["agent_thought"] = True
        elif token == "--no-agent-thought":
            cfg["agent_thought"] = False
    return cfg


def resolve(text, script):
    """Substitute the script's own variable assignments (A2_TURNS=...) into a string."""
    for name, value in re.findall(r"^([A-Z0-9_]+)=(\d+)$", script, flags=re.MULTILINE):
        text = text.replace("${" + name + "}", value).replace('"${' + name + '}"', value)
    return text.replace('"', "")


RUNGS_SCRIPT = script_text("slurm/submit_agentic_rungs.sh")
GATE_SCRIPT = script_text("slurm/submit_agentic_gate.sh")
CONFIG = load("configs/eval_config.agentic.json")
GATE_CONFIG = load("configs/eval_config.agentic_gate.json")

print("--- every rung's directory, derived from the script, is the one the harness builds ---")
blocks = re.findall(r"^  (\d)\) (RUNG_NAME=.*?);;", RUNGS_SCRIPT, flags=re.MULTILINE | re.DOTALL)
check("five rungs in the case statement", [task for task, _ in blocks], ["0", "1", "2", "3", "4"])
decoding = _decoding_suffix(CONFIG["constrained_decoding"], CONFIG["force_root_arity"],
                            CONFIG["generation_token_budget"], CONFIG["type_aware_filters"],
                            CONFIG["constant_root_arity"], CONFIG["structural_validation"],
                            CONFIG["constrained_any_order"], CONFIG["constrained_bounded_shape"])
check("the config's decoding suffix", decoding, "_constrained_ordered_bounded_tok4000")
# notes 43.8: no free whitespace, bounded tree lists, and a per-turn cap for the loop rungs.
for _name, _cfg in (("agentic", CONFIG), ("gate", load("configs/eval_config.agentic_gate.json"))):
    check(f"{_name}: the bounded grammar is on", _cfg["constrained_bounded_shape"], True)
    check(f"{_name}: the loop's per-turn cap is 2000 tokens", _cfg["agent_turn_token_budget"], 2000)
check("_bounded is named after _ordered, and not without constrained decoding",
      (_decoding_suffix(True, False, 4000, any_order=False, bounded=True),
       _decoding_suffix(False, False, 4000, any_order=False, bounded=True)),
      ("_constrained_ordered_bounded_tok4000", "_tok4000"))
check("the per-turn cap names both loop rungs; A0 never runs the loop, so it keeps its id",
      (_agent_suffix("agentic-log-tools", 8, 12, False, 2000),
       _agent_suffix("agentic-self-refine", 8, 12, False, 2000),
       _agent_suffix("schema-allowed-entities-only", 8, 12, False, 2000)),
      ("_turns8_tools12_turntok2000", "_turns8_turntok2000", ""))
check("no cap keeps every earlier id", _agent_suffix("agentic-log-tools", 8, 12, True),
      "_turns8_tools12_thought")
# notes 43.7: any_order=True admits a repeated key, and base Llama looped on one in 58/70 turns.
check("both agentic configs compile the ORDERED grammar",
      (CONFIG["constrained_any_order"], load("configs/eval_config.agentic_gate.json")
       ["constrained_any_order"]), (False, False))
check("...which names its directories (no resume into an any_order run)",
      _decoding_suffix(True, False, 4000, any_order=False), "_constrained_ordered_tok4000")
check("the default leaves every published directory name alone",
      (_decoding_suffix(True, False, 4000), _decoding_suffix(True, False, 4000, any_order=True)),
      ("_constrained_tok4000", "_constrained_tok4000"))
check("_ordered means nothing without constrained decoding, so it is not written",
      _decoding_suffix(False, False, 4000, any_order=False), "_tok4000")
suffixes = {}
for task, block in blocks:
    strategy_idx = int(re.search(r"STRATEGY_IDX=(\d+)", block).group(1))
    agent_args = resolve((re.search(r"AGENT_ARGS=\((.*?)\)", block).group(1)), RUNGS_SCRIPT)
    declared = resolve(re.search(r'RUNG_SUFFIX="([^"]+)"', block).group(1), RUNGS_SCRIPT)
    cfg = flags_to_config(CONFIG, agent_args)
    strategy = CONFIG["prompt_strategies"][strategy_idx]
    built = decoding + _agent_shard_suffix([strategy], cfg)
    check(f"task {task} ({strategy}): RUNG_SUFFIX == the shard suffix eval_testset builds",
          declared, built)
    check(f"task {task}: ...and its agent part == the config_id suffix",
          built[len(decoding):], _agent_suffix(strategy, *_agent_budget(cfg)))
    suffixes[task] = (strategy_idx, built)
check("five DIFFERENT directories",
      len({(s, b) for s, b in suffixes.values()}), 5)
same_strategy = [b for s, b in suffixes.values() if s == 1]
check("A1, A2 and A1+thought share strategy 1 -- and still get three different suffixes",
      (len(same_strategy), len(set(same_strategy))), (3, 3))
check("the resume glob is built from RUNG_SUFFIX in this run's directory",
      f"{CONFIG['output_dir']}/shard_m${{MODEL_IDX}}_s${{STRATEGY_IDX}}_e${{SCHEMA_IDX}}"
      "${RUNG_SUFFIX}_*" in RUNGS_SCRIPT, True)
# notes 43.17: the pFBEXLR runs never share a directory with the completed pFBEXL run of 43.15 --
# A0's name carries no lever, so in one directory the two protocols' A0 would be indistinguishable.
check("the rungs write to eval_runs/agentic_r, not to the completed run's eval_runs/agentic",
      CONFIG["output_dir"], "eval_runs/agentic_r")

print("\n--- the rung table is the intended experiment ---")
by_task = dict(blocks)
check("A0 is the one-shot strategy", CONFIG["prompt_strategies"][0], "schema-allowed-entities-only")
check("A1-prime is the no-tools control", CONFIG["prompt_strategies"][2], "agentic-self-refine")
check("A1-prime has A1's turn budget (compute-matched)",
      re.search(r"--agent-max-turns (\S+)", by_task["2"]).group(1),
      re.search(r"--agent-max-turns (\S+)", by_task["1"]).group(1))
check("A1+thought differs from A1 in the thought flag only",
      by_task["4"].split("AGENT_ARGS=")[1].split(")")[0].replace("--agent-thought", "X"),
      by_task["1"].split("AGENT_ARGS=")[1].split(")")[0].replace("--no-agent-thought", "X"))
# notes 43.13: sized from the Stage 5 dev run (6.4-7.0 s/turn, longest case 95 s at 8 turns ->
# ~190 s at 16), and recorded as a COMMITTED constant -- an exported variable may not survive sbatch.
check("A2's budget is the sized one: 16 turns, 24 tool calls, marked A2_SIZED=1",
      [re.search(rf"^{name}=(\d+)", RUNGS_SCRIPT, re.M).group(1)
       for name in ("A2_TURNS", "A2_TOOL_CALLS", "A2_SIZED")], ["16", "24", "1"])
check("A2 is double A1's turns and tool calls (only the budget moves)",
      (int(resolve("${A2_TURNS}", RUNGS_SCRIPT)), int(resolve("${A2_TOOL_CALLS}", RUNGS_SCRIPT))),
      (2 * int(re.search(r"--agent-max-turns (\d+)", by_task["1"]).group(1)),
       2 * int(re.search(r"--agent-max-tool-calls (\d+)", by_task["1"]).group(1))))
check("A2 still refuses to run when the constant says unsized",
      'if [ "${A2_SIZED}" != "1" ]' in by_task["3"], True)
check("...and nothing depends on an exported confirmation any more",
      "OCPQ_A2_CONFIRMED" in RUNGS_SCRIPT, False)
check("the tool rungs run the probe preflight", 'python "$PROJECT_DIR/src/check_probe.py"'
      in RUNGS_SCRIPT and 'if [ "$STRATEGY_IDX" = "1" ]' in RUNGS_SCRIPT, True)

print("\n--- nothing is an oracle, and every rung shares one cap ---")
for name, cfg in (("agentic", CONFIG), ("gate", GATE_CONFIG)):
    check(f"{name}: no oracle- strategy",
          [s for s in cfg["prompt_strategies"] if s.startswith("oracle-")], [])
    check(f"{name}: force_root_arity is off", cfg["force_root_arity"], False)
    check(f"{name}: every strategy is registered in LLM",
          [s for s in cfg["prompt_strategies"] if s not in LLM.PROMPT_STRATEGIES], [])
    check(f"{name}: corpus_check is error (CLAUDE.md 2.3)", cfg["corpus_check"], "error")
    check(f"{name}: case_timeout_sec is the matrix's 600", cfg["case_timeout_sec"], 600)
check("the rungs pass no per-rung cap (one config, one cap)", "--case-timeout" in RUNGS_SCRIPT,
      False)
_non_oracle = load("configs/eval_config.non_oracle.json")
# Models the agentic arm adds are APPENDED (Muse Glimmer, notes 43.16), so every index the two
# configs share still means the same model -- a prefix, not equality.
check("the model list starts with the non-oracle arm's, so a shared index means the same model",
      (CONFIG["models"][:len(_non_oracle["models"])], CONFIG["fine_tune_map"]),
      (_non_oracle["models"], _non_oracle["fine_tune_map"]))
check("...and what it appends is Muse Glimmer only",
      [m["label"] for m in CONFIG["models"][len(_non_oracle["models"]):]],
      ["muse-glimmer-30b-base"])

print("\n--- the gate: dev set only, and its own directory ---")
check("the gate reads the dev set", GATE_CONFIG["testset_path"], "data/agent_devset.json")
check("...never the testset", "testset.json" in GATE_CONFIG["testset_path"], False)
check("the gate writes to its own directory", GATE_CONFIG["output_dir"], "eval_runs/agentic_gate")
check("the gate measures the tool rung only", GATE_CONFIG["prompt_strategies"],
      ["agentic-log-tools"])
gate_suffixes = set(re.findall(r'RUNG_SUFFIX="([^"]+)"', GATE_SCRIPT))
check("gate suffixes are the A1 and A1+thought directories",
      gate_suffixes, {suffixes["1"][1], suffixes["4"][1]})
check("the gate's resume glob is in its own directory",
      "eval_runs/agentic_gate/shard_m${MODEL_IDX}" in GATE_SCRIPT, True)

print("\n--- the main matrix is untouched ---")
main_config = load("configs/eval_config.json")
check("no agentic strategy in the main matrix config (it would add cells silently)",
      [s for s in main_config["prompt_strategies"] if s.startswith("agentic")], [])

print("\n--- the thought condition changes the schema and one rule line, nothing else ---")
check("the plain action has no thought field", "thought" in AgentAction.model_fields, False)
check("the thought action has one", "thought" in AgentActionWithThought.model_fields, True)
check("...and it is optional (any_order can skip a required property anyway)",
      AgentActionWithThought.model_fields["thought"].is_required(), False)
check("...and everything else is identical",
      set(AgentActionWithThought.model_fields) - {"thought"}, set(AgentAction.model_fields))
LLM.set_agent_thought(False)
plain_rules, plain_model = LLM._agent_protocol_rules(), LLM._agent_action_model()
LLM.set_agent_thought(True)
thought_rules, thought_model = LLM._agent_protocol_rules(), LLM._agent_action_model()
LLM.set_agent_thought(False)
check("off: the plain schema", plain_model, AgentAction)
check("off: the rules do not mention a thought", "thought" in plain_rules, False)
check("on: the thought schema", thought_model, AgentActionWithThought)
check("on: the rules are the plain rules plus exactly one line",
      thought_rules, plain_rules + "\n" + LLM.AGENT_THOUGHT_RULE)

print("\n--- the scripts themselves ---")
for name, text in (("rungs", RUNGS_SCRIPT), ("gate", GATE_SCRIPT)):
    check(f"{name}: no CR characters (a CRLF script dies on the cluster)", text.count("\r"), 0)
    check(f"{name}: node and GPU are recorded (notes 37.5)",
          "nvidia-smi" in text and "hostname" in text, True)
    check(f"{name}: --case-shard is accepted as an argument", "--case-shard)" in text, True)
check("rungs: the array covers exactly the five rungs", "#SBATCH --array=0-4" in RUNGS_SCRIPT,
      True)
check("gate: the default array is the two BASE models x thought off/on (notes 43.6)",
      "#SBATCH --array=0,1,4,5" in GATE_SCRIPT, True)

print("\n--- the arm runs on a BASE model, and only the one the gate picked (notes 43.6) ---")
# The fine-tuned adapters emit one tree and never call a tool (2026-09-23 smoke: 0 calls in 10
# cases), and MODEL_IDX=1 -- fine-tuned Llama -- was hard-coded here until then. Even indices are
# base models in the config's expansion; the script must refuse the odd ones and refuse to guess.
from expand_models import expand_models  # noqa: E402

labels = [m["label"] for m in expand_models(CONFIG)["models"]]
check("the indices the guard accepts are base models",
      [labels[0], labels[2]], ["llama31-8b-base", "ministral3-14b-base"])
check("...and the ones it refuses are their adapters",
      [labels[1], labels[3]], ["llama31-8b-finetuned", "ministral3-14b-finetuned"])
check("MODEL_IDX is not hard-coded", re.search(r"^MODEL_IDX=\d", RUNGS_SCRIPT, re.M), None)
_loop = RUNGS_SCRIPT.index('while [ "$#" -gt 0 ]; do')
_guard_end = RUNGS_SCRIPT.index('echo "Model index: ${MODEL_IDX}"')
_snippet = ('MODEL_IDX="${OCPQ_AGENT_MODEL_IDX:-}"\n'
            + RUNGS_SCRIPT[_loop:_guard_end] + 'echo "OK ${MODEL_IDX}"\n')
_bash = shutil.which("bash")
if _bash is None:
    print("SKIP  the guard is executed (no bash on PATH)")
else:
    def guard(args, env_idx=None):
        env = dict(os.environ)
        env.pop("OCPQ_AGENT_MODEL_IDX", None)
        if env_idx is not None:
            env["OCPQ_AGENT_MODEL_IDX"] = env_idx
        proc = subprocess.run([_bash, "-c", _snippet, "guard", *args], env=env,
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()
    check("no model given: refuses", guard([])[0], 1)
    check("--model-idx 0 (Llama base): runs", guard(["--model-idx", "0"]), (0, "OK 0"))
    check("--model-idx=2 (Ministral base): runs", guard(["--model-idx=2"]), (0, "OK 2"))
    check("--model-idx 1 (fine-tuned Llama): refuses", guard(["--model-idx", "1"])[0], 1)
    check("--model-idx 3 (fine-tuned Ministral): refuses", guard(["--model-idx", "3"])[0], 1)
    check("the env var is a fallback", guard([], env_idx="2"), (0, "OK 2"))
    check("the argument wins over the env var", guard(["--model-idx", "0"], env_idx="2"),
          (0, "OK 0"))
    check("the argument coexists with --case-shard",
          guard(["--case-shard", "8/11", "--model-idx", "0"]), (0, "OK 0"))
    check("--model-idx 8 (Muse Glimmer base, the second arm): runs",
          guard(["--model-idx", "8"]), (0, "OK 8"))
    check("--model-idx 9 (nothing: Glimmer has no adapter): refuses",
          guard(["--model-idx", "9"])[0], 1)

print("\n--- the runtime follows the MODEL: Glimmer in its own env, on $HPCWORK (notes 43.16) ---")
check("index 8 is Muse Glimmer in the expansion", labels[8], "muse-glimmer-30b-base")
RUNTIME = script_text("slurm/model_runtime.sh")
check("model_runtime.sh: no CR characters", RUNTIME.count("\r"), 0)
for name, text, known in (("rungs", RUNGS_SCRIPT, 'echo "Model index: ${MODEL_IDX}"'),
                          ("gate", GATE_SCRIPT, '8|9) MODEL_IDX=8;')):
    check(f"{name}: sources the runtime and activates what it chose",
          ('source "$SCRIPT_DIR/model_runtime.sh"' in text
           and 'conda activate "$OCPQ_RUNTIME_ENV"' in text
           and 'export HF_HOME="$OCPQ_RUNTIME_HF_HOME"' in text), True)
    check(f"{name}: no environment is activated before the model is known",
          text.index("conda activate") > text.index(known), True)
    check(f"{name}: ...and nothing runs python before it",
          re.search(r"^python ", text[:text.index("conda activate")], re.M), None)
check("gate: tasks 8/9 are Glimmer, thought off/on",
      '8|9) MODEL_IDX=8; MODEL_NAME="muse-glimmer-30b-base"' in GATE_SCRIPT, True)
if _bash is not None:
    def runtime(idx, **env_over):
        env = {k: v for k, v in os.environ.items()
               if k not in ("OCPQ_EVAL_ENV", "HF_HOME", "HF_HUB_CACHE", "HPCWORK")}
        env.update(env_over)
        env.setdefault("HOME", "/home/u")
        snippet = (RUNTIME + '\nocpq_model_runtime "$1" || exit 1\n'
                   'echo "$OCPQ_RUNTIME_ENV|$OCPQ_RUNTIME_HF_HOME|$OCPQ_RUNTIME_HF_HUB_CACHE"\n')
        proc = subprocess.run([_bash, "-c", snippet, "runtime", idx], env=env,
                              capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()
    check("index 2 runs exactly where it did before",
          runtime("2", HOME="/home/u"),
          (0, "nlxocpq-eval-v5|/home/u/.cache/huggingface|/home/u/.cache/huggingface/hub"))
    check("...honouring the variables it honoured before",
          runtime("2", OCPQ_EVAL_ENV="e", HF_HOME="/h"), (0, "e|/h|/h/hub"))
    check("index 8 runs in the Glimmer env with its cache on $HPCWORK",
          runtime("8", HPCWORK="/w"),
          (0, "nlxocpq-eval-glimmer|/w/nlxocpq-hf|/w/nlxocpq-hf/hub"))
    check("...whatever OCPQ_EVAL_ENV and HF_HOME say",
          runtime("8", HPCWORK="/w", OCPQ_EVAL_ENV="nlxocpq-eval-v5", HF_HOME="/h"),
          (0, "nlxocpq-eval-glimmer|/w/nlxocpq-hf|/w/nlxocpq-hf/hub"))
    check("index 8 without $HPCWORK refuses", runtime("8")[0], 1)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> five distinct agentic rungs in five directories, a base model, no gold, one cap")
