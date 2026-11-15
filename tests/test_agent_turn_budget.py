"""The agent's per-turn token cap must bound a runaway turn, never an answer.

`agent_turn_token_budget` (notes 43.8) caps each loop turn so a turn that repeats itself costs
~48 s instead of ~95 s. That is only legitimate if every gold answer fits under the cap with room
to spare, in the evaluated models' OWN tokens -- a cap that cut a gold-sized tree would cap the
agent below gold and read as a finding. So each gold tree (testset and dev set) is serialised the
way the bounded grammar forces it -- standard `", "` / `": "` separators, no free whitespace, in
schema order -- wrapped in the LONGEST turn that carries a tree (`try_query`), tokenised with each
gate model's tokenizer, and required to fit within HEADROOM of the cap.

Tokenizers: the base model's own, from the HF cache (the cluster), at the config's pinned revision
when it has one. Where transformers cannot build it (Muse Glimmer's tokenizer class needs
transformers 5.x), its tokenizer.json from the same cache. Where the base model is absent, the
fine-tuned adapter's copy in models/adapters/, which is the same tokenizer (an adapter changes
weights, not the vocabulary). The output says which one was used. On the cluster, Glimmer's cache
is on $HPCWORK:  HF_HOME=$HPCWORK/nlxocpq-hf python tests/test_agent_turn_budget.py

Run from the repo root:  python tests/test_agent_turn_budget.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from expand_models import expand_models  # noqa: E402
from ocpq_schema import BindingBoxTree  # noqa: E402

# The largest gold turn must fit in this share of the cap. The rest is room for a thought, for a
# tree slightly larger than any gold one, and for tokenizer differences across models.
HEADROOM = 0.75
# The base models the rung script accepts (submit_agentic_rungs.sh): 8 Muse Glimmer (notes 43.16),
# 9 Qwen3.8-27B and 10 Gemma 4 31B-it (notes 43.18).
GATE_MODEL_INDICES = (0, 2, 8, 9, 10)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


try:
    from transformers import AutoTokenizer
except ImportError:
    print("FAIL  transformers is not installed; this check needs the models' tokenizers.")
    sys.exit(1)

configs = {name: json.load(open(REPO / "configs" / name, encoding="utf-8"))
           for name in ("eval_config.agentic.json", "eval_config.agentic_gate.json")}
caps = {name: cfg.get("agent_turn_token_budget") for name, cfg in configs.items()}
check("both agentic configs set a per-turn cap, and the same one", len(set(caps.values())), 1)
cap = next(iter(caps.values()))
check("the cap is below the run budget (else it bounds nothing)",
      cap is not None and cap < configs["eval_config.agentic.json"]["generation_token_budget"],
      True)

turns = []
for path in ("data/testset.json", "data/agent_devset.json"):
    for entry in json.load(open(REPO / path, encoding="utf-8")):
        tree = BindingBoxTree.model_validate(entry["ocpq_query"]).model_dump(by_alias=True,
                                                                             mode="json")
        turns.append(json.dumps({"tool": "try_query", "args": {"tree": tree}}))
print(f"        {len(turns)} gold trees, as try_query turns; longest {max(map(len, turns))} chars")

cfg = configs["eval_config.agentic.json"]
models = expand_models(cfg)["models"]
adapters = cfg.get("fine_tune_map", {})


def load_tokenizer(model):
    """`(count_tokens, source)`, or `(None, None)` if neither tokenizer can be found."""
    rev = {"revision": model["revision"]} if model.get("revision") else {}
    try:
        tok = AutoTokenizer.from_pretrained(model["name"], local_files_only=True, **rev)
        return (lambda text: len(tok(text, add_special_tokens=False)["input_ids"])), "base, cached"
    except Exception:  # noqa: BLE001 - fall through to the vocabulary file itself
        # A tokenizer class this transformers cannot build: ValueError for Glimmer's before 5.x,
        # AttributeError for Gemma 4's on 4.x. The vocabulary, read below, is what counts.
        pass
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        path = hf_hub_download(model["name"], "tokenizer.json", local_files_only=True, **rev)
        tok = Tokenizer.from_file(path)
        return ((lambda text: len(tok.encode(text, add_special_tokens=False).ids)),
                "base, cached tokenizer.json")
    except Exception:  # noqa: BLE001 - not cached; try the adapter's copy below
        pass
    mapped = adapters.get(model["name"], {})
    adapter = REPO / (mapped.get("name") if isinstance(mapped, dict) else mapped or "")
    if (adapter / "tokenizer.json").exists():
        # Read the vocabulary file itself: the adapters' tokenizer_config names a tokenizer class
        # from a newer transformers than some environments have, and the vocabulary is what counts.
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(adapter / "tokenizer.json"))
        return ((lambda text: len(tok.encode(text, add_special_tokens=False).ids)),
                f"adapter's copy, {adapter.name}")
    return None, None


for idx in GATE_MODEL_INDICES:
    model = models[idx]
    count_tokens, source = load_tokenizer(model)
    if count_tokens is None:
        check(f"{model['label']}: a tokenizer is available", False, True)
        continue
    counts = [count_tokens(t) for t in turns]
    worst = max(counts)
    print(f"        {model['label']} ({source}): longest gold turn {worst} tokens, "
          f"median {sorted(counts)[len(counts) // 2]}; cap {cap}, headroom limit "
          f"{int(cap * HEADROOM)}")
    check(f"{model['label']}: every gold turn fits within {int(HEADROOM * 100)}% of the cap",
          worst <= cap * HEADROOM, True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the per-turn cap bounds runaways, not answers")
