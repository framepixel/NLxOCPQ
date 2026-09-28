"""Per-model chat settings (notes 43.16): Muse Glimmer gets them, no earlier model is touched.

Muse Glimmer-30B's chat template ends the generation prompt at `<|start|>assistant`; the model then
chooses a channel -- ` to=user<|message|>` to answer, ` to=self<|message|>` to reason. Constrained
decoding would force `{` where the model expects that header, so the harness appends the answer
channel itself (`generation_prefix`) and asks for low reasoning strength (`chat_template_kwargs`),
and the weights are pinned to one hub commit (`revision`). Asserted here:

  * WITHOUT the settings, the local generation path is exactly what every recorded run used: the
    same `apply_chat_template` call, the same input ids, the same cached-model key behaviour.
  * WITH them, the kwargs reach the template and the prefix is appended after the generation prompt.
  * On Glimmer's REAL template and tokenizer (the pinned revision, from the HF cache): the prefix is
    what the template itself writes for a plain assistant turn, so the current turn looks exactly
    like the model's history -- also token for token, with no merge across the boundary.
  * The configs: Glimmer is appended LAST (index 8 after expansion), identical in both agentic
    configs, has no adapter, and no earlier model carries any of the new keys.

The template section reads Glimmer's files from the HF cache at the pinned revision. On the laptop
they are cached by `hf_hub_download`; on the cluster run it as
    HF_HOME=$HPCWORK/nlxocpq-hf python tests/test_model_chat_options.py
after scripts/prefetch_glimmer.sh.

Run from the repo root:  python tests/test_model_chat_options.py
"""
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

import LLM  # noqa: E402
from eval_testset import _configure_backend, _library_versions  # noqa: E402
from expand_models import expand_models  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


CONFIGS = {name: json.loads((ROOT / "configs" / name).read_text(encoding="utf-8"))
           for name in ("eval_config.agentic.json", "eval_config.agentic_gate.json")}
GLIMMER = next(m for m in CONFIGS["eval_config.agentic.json"]["models"]
               if m.get("label") == "muse-glimmer-30b-base")
NEW_KEYS = {"revision", "finetuned", "chat_template_kwargs", "generation_prefix"}

print("--- the configs: Glimmer appended last, pinned, and no earlier model touched ---")
for name, cfg in CONFIGS.items():
    labels = [m["label"] for m in expand_models(cfg)["models"]]
    check(f"{name}: Glimmer is index 8 after expansion, and nothing follows it",
          (labels.index("muse-glimmer-30b-base"), len(labels)), (8, 9))
    check(f"{name}: indices 0-7 are what they were",
          labels[:8], ["llama31-8b-base", "llama31-8b-finetuned", "ministral3-14b-base",
                       "ministral3-14b-finetuned", "ministral3-3b-base", "ministral3-3b-finetuned",
                       "gemma3-4b-base", "gemma3-4b-finetuned"])
    check(f"{name}: no earlier model carries a per-model chat key",
          [m["label"] for m in cfg["models"][:-1] if NEW_KEYS & set(m)], [])
    check(f"{name}: the Glimmer entry is the same in both configs",
          next(m for m in cfg["models"] if m.get("label") == "muse-glimmer-30b-base"), GLIMMER)
check("the revision is a full 40-hex commit", bool(re.fullmatch(r"[0-9a-f]{40}",
                                                                GLIMMER["revision"])), True)
check("the chat settings are the decided ones (notes 43.16)",
      (GLIMMER["chat_template_kwargs"], GLIMMER["generation_prefix"], GLIMMER["finetuned"]),
      ({"reasoning_strength": "low"}, " to=user<|message|>", False))
check("expand_models: a model with finetuned=false gets no invented adapter",
      [m["name"] for m in expand_models({"models": [{"name": "x", "finetuned": False}]})["models"]],
      ["x"])
check("...and a model without the key still gets its default adapter entry",
      [m["name"] for m in expand_models({"models": [{"name": "x"}]})["models"]],
      ["x", "x-finetuned"])

print("\n--- the options are set, and cleared, for every model before it is selected ---")
_configure_backend(dict(GLIMMER), {"backend": "local-gpu", "local_quantization": "4bit"})
check("Glimmer's entry sets all three",
      (LLM.local_model_revision, LLM.local_chat_template_kwargs, LLM.local_generation_prefix),
      (GLIMMER["revision"], {"reasoning_strength": "low"}, " to=user<|message|>"))
key_glimmer = LLM._local_model_key("m", None, "cuda", "auto", "4bit")
_configure_backend({"name": "unsloth/Ministral-3-14B-Instruct-2512-unsloth-bnb-4bit"},
                   {"backend": "local-gpu", "local_quantization": "4bit"})
check("the next model without them inherits nothing",
      (LLM.local_model_revision, LLM.local_chat_template_kwargs, LLM.local_generation_prefix),
      (None, {}, None))
check("the revision is part of the loaded model's key",
      LLM._local_model_key("m", None, "cuda", "auto", "4bit") != key_glimmer, True)
LLM.local_model = LLM.local_tokenizer = object()
LLM.local_model_key = LLM._local_model_key(LLM.hf_model, LLM.hf_token, "cuda", "auto", "4bit")
LLM.set_local_gpu_client(model_name=LLM.hf_model, device="cuda", dtype="auto", quantization="4bit")
check("re-selecting the same model keeps it loaded (no reload per case)",
      LLM.local_model is not None, True)
LLM.set_local_model_options(revision="0" * 40)
LLM.set_local_gpu_client(model_name=LLM.hf_model, device="cuda", dtype="auto", quantization="4bit")
check("a different revision evicts it", LLM.local_model, None)
LLM.set_local_model_options()

print("\n--- the generation path: identical without the settings, prefixed with them ---")


class FakeTokenizer:
    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
        return torch.tensor([[10, 11, 12]])

    def encode(self, text, add_special_tokens=True):
        return [90, 91] if text == " to=user<|message|>" else [99]

    def decode(self, ids, skip_special_tokens=True):
        return "{}"


class FakeModel:
    device = "cpu"
    config = SimpleNamespace(max_position_embeddings=131072)

    def __init__(self):
        self.inputs = []

    def generate(self, **kwargs):
        self.inputs.append(kwargs["input_ids"].tolist())
        return torch.cat([kwargs["input_ids"], torch.tensor([[7]])], dim=-1)


def generate_once():
    tok, model = FakeTokenizer(), FakeModel()
    LLM.local_tokenizer, LLM.local_model = tok, model
    LLM._ensure_local_gpu_model_loaded = lambda: None
    LLM._call_local_gpu_chat_completion([{"role": "user", "content": "q"}], 50, None)
    return tok.template_calls[0], model.inputs[0]


LLM.set_constrained_decoding(False)
LLM.set_local_model_options()
kwargs, ids = generate_once()
check("off: apply_chat_template gets exactly the pre-43.16 arguments",
      kwargs, {"add_generation_prompt": True, "return_tensors": "pt"})
check("off: the model sees the template's ids and nothing else", ids, [[10, 11, 12]])
LLM.set_local_model_options(chat_template_kwargs={"reasoning_strength": "low"},
                            generation_prefix=" to=user<|message|>")
kwargs, ids = generate_once()
check("on: the template variables reach the template",
      kwargs, {"add_generation_prompt": True, "return_tensors": "pt", "reasoning_strength": "low"})
check("on: the prefix's ids follow the generation prompt", ids, [[10, 11, 12, 90, 91]])
LLM.set_local_model_options()

print("\n--- Glimmer's REAL template and tokenizer (pinned revision, from the HF cache) ---")
try:
    from huggingface_hub import hf_hub_download
    files = {f: hf_hub_download(GLIMMER["name"], f, revision=GLIMMER["revision"],
                                local_files_only=True)
             for f in ("chat_template.jinja", "tokenizer.json", "tokenizer_config.json")}
except Exception as exc:  # noqa: BLE001 - reported as a failure with the way to fix it
    files = None
    check(f"Glimmer's template and tokenizer are cached ({type(exc).__name__}); run "
          "scripts/prefetch_glimmer.sh and HF_HOME=$HPCWORK/nlxocpq-hf on the cluster", False, True)

if files:
    from jinja2.ext import loopcontrols
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer

    def _raise(message):
        raise ValueError(message)

    # How transformers renders a chat template: a sandboxed environment with these options.
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[loopcontrols])
    env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
    env.globals["raise_exception"] = _raise
    template = env.from_string(Path(files["chat_template.jinja"]).read_text(encoding="utf-8"))
    bos = json.loads(Path(files["tokenizer_config.json"]).read_text(encoding="utf-8"))["bos_token"]
    bos = bos if isinstance(bos, str) else bos["content"]
    tok = Tokenizer.from_file(files["tokenizer.json"])

    def render(messages, **kwargs):
        return template.render(messages=messages, bos_token=bos, **kwargs)

    task = [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "TASK"}]
    prompt = render(task, add_generation_prompt=True, **GLIMMER["chat_template_kwargs"])
    check("the generation prompt ends at the channel choice",
          prompt.endswith("<|start|>assistant"), True)
    check("the system prompt carries the low reasoning strength, once",
          (prompt.count("Reasoning strength: low."), "Reasoning strength: high" in prompt), (1, False))
    check("...where the template's default would have said high",
          "Reasoning strength: high." in render(task, add_generation_prompt=True), True)
    history = render(task + [{"role": "assistant", "content": "ACTION"},
                             {"role": "user", "content": "RESULT"}], add_generation_prompt=True)
    check("the prefix is exactly how the template writes the model's own past turn",
          "<|start|>assistant" + GLIMMER["generation_prefix"] + "ACTION<|eot|>" in history, True)
    joined = tok.encode(prompt + GLIMMER["generation_prefix"], add_special_tokens=False).ids
    split = (tok.encode(prompt, add_special_tokens=False).ids
             + tok.encode(GLIMMER["generation_prefix"], add_special_tokens=False).ids)
    check("appending the prefix's ids = tokenising the whole text (no merge across the boundary)",
          split, joined)
    check("the prefix ends on the <|message|> special token",
          tok.encode(GLIMMER["generation_prefix"], add_special_tokens=False).tokens[-1],
          "<|message|>")

print("\n--- the loader: causal LM, then image-text-to-text, then multimodal (Glimmer's card) ---")
import transformers  # noqa: E402

UNREGISTERED = "Unrecognized configuration class X for this kind of AutoModel"


def fake_class(tag, error=None):
    def from_pretrained(name, **kwargs):
        if error:
            raise ValueError(error)
        return tag
    return SimpleNamespace(from_pretrained=from_pretrained)


def load_with(causal, image, multimodal):
    saved = {n: getattr(transformers, n, None) for n in
             ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM")}
    transformers.AutoModelForCausalLM = causal
    transformers.AutoModelForImageTextToText = image
    transformers.AutoModelForMultimodalLM = multimodal
    try:
        return LLM._load_text_generation_model("m")
    except ValueError as exc:
        return f"raised: {exc}"
    finally:
        for n, v in saved.items():
            if v is None:
                delattr(transformers, n)
            else:
                setattr(transformers, n, v)


check("a causal LM loads as before",
      load_with(fake_class("causal"), fake_class("image"), fake_class("mm")), "causal")
check("an image-text-to-text checkpoint loads as before (Ministral 3)",
      load_with(fake_class("", UNREGISTERED), fake_class("image"), fake_class("mm")), "image")
check("registered under neither: the multimodal class (Muse Glimmer)",
      load_with(fake_class("", UNREGISTERED), fake_class("", UNREGISTERED), fake_class("mm")), "mm")
check("any other error is raised, never papered over",
      load_with(fake_class("", UNREGISTERED), fake_class("", "out of memory"), fake_class("mm")),
      "raised: out of memory")

print("\n--- the model loads before the case timer, in the agentic configs only (notes 43.17) ---")
for name, cfg in CONFIGS.items():
    check(f"{name}: preload_model is on", cfg.get("preload_model"), True)
check("the main matrix config does not preload (its runs are unchanged)",
      json.loads((ROOT / "configs" / "eval_config.json").read_text(encoding="utf-8"))
      .get("preload_model"), None)
# The 2026-09-28 gate: preload_model was on in the config and never ran, because run_evaluation
# builds `global_cfg` from a fixed key list and the key was not on it. So: every key the harness
# READS from global_cfg must be one that run_evaluation PUTS into it.
import ast  # noqa: E402

_tree = ast.parse((ROOT / "src" / "eval_testset.py").read_text(encoding="utf-8"))
_set = next({k.value for k in n.value.keys if isinstance(k, ast.Constant)}
            for n in ast.walk(_tree)
            if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
            and any(isinstance(t, ast.Name) and t.id == "global_cfg" for t in n.targets))
_read = {n.args[0].value for n in ast.walk(_tree)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
         and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
         and n.func.value.id == "global_cfg" and n.args and isinstance(n.args[0], ast.Constant)}
check("every key read from global_cfg is set in it (else the config value is silently ignored)",
      sorted(_read - _set), [])
check("...preload_model among them", "preload_model" in _set, True)

loads = []
_real_ensure = LLM._ensure_local_gpu_model_loaded
LLM._ensure_local_gpu_model_loaded = lambda: loads.append(1)
saved_backend = LLM.hf_backend
LLM.hf_backend = "local-gpu"
check("local backend: it loads, and says how long it took",
      (isinstance(LLM.preload_local_model(), float), len(loads)), (True, 1))
LLM.hf_backend = "custom-endpoint"
check("an endpoint backend has nothing to load", (LLM.preload_local_model(), len(loads)), (None, 1))
LLM.hf_backend = saved_backend
LLM._ensure_local_gpu_model_loaded = _real_ensure

print("\n--- the manifest records the generation stack ---")
versions = _library_versions()
check("every recorded library has an entry (None when absent)",
      sorted(versions), sorted(["torch", "transformers", "tokenizers", "xgrammar", "bitsandbytes",
                                "accelerate", "peft"]))
check("...and torch, which this test imported, has a version", bool(versions["torch"]), True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> Glimmer's chat settings apply to Glimmer only; every other model is unchanged")
