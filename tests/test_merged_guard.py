"""The guard that must reject the 2026-08-19 failure: a PEFT wrapper state dict written under a
merged model's filename.

Builds real safetensors files with the exact key names from the failing LOAD REPORT.

Run:  python <this file>
"""
import shutil
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import merge_gemma as mg  # noqa: E402

failures = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


def raises(name, fn, needle):
    try:
        fn()
    except SystemExit as exc:
        if needle.lower() in str(exc).lower():
            print(f"PASS  {name}")
        else:
            print(f"FAIL  {name}: message lacked {needle!r}\n        {exc}")
            failures.append(name)
        return
    print(f"FAIL  {name} (no SystemExit)")
    failures.append(name)


tmp = Path(tempfile.mkdtemp(prefix="mergeguard_"))
t = lambda: torch.zeros(4, 4)

print("--- a genuinely merged checkpoint is accepted ---")
good = tmp / "good"; good.mkdir()
save_file({
    "model.language_model.layers.0.self_attn.q_proj.weight": t(),
    "model.language_model.layers.0.mlp.up_proj.weight": t(),
    "lm_head.weight": t(),
}, str(good / "model.safetensors"))
(good / "config.json").write_text("{}")
mg._assert_dir_is_loadable(good, "merged")
check("merged checkpoint accepted", True)

print("\n--- the actual failure: PEFT wrapper keys under a merged filename ---")
bad = tmp / "peft_wrapper"; bad.mkdir()
# Exactly the shapes from the LOAD REPORT that broke the reload.
save_file({
    "model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight": t(),
    "model.language_model.layers.0.self_attn.q_proj.lora_B.default.weight": t(),
    "model.language_model.layers.0.self_attn.q_proj.base_layer.weight": t(),
    "model.language_model.layers.0.mlp.up_proj.base_layer.weight": t(),
}, str(bad / "model.safetensors"))
(bad / "config.json").write_text("{}")
raises("PEFT wrapper state dict rejected",
       lambda: mg._assert_dir_is_loadable(bad, "fine-tuned arm"), "was NOT merged")

print("\n--- rejected even if only SOME tensors carry wrapper names ---")
mixed = tmp / "mixed"; mixed.mkdir()
save_file({
    "model.language_model.layers.0.self_attn.q_proj.weight": t(),
    "model.language_model.layers.0.mlp.up_proj.base_layer.weight": t(),
    "lm_head.weight": t(),
}, str(mixed / "model.safetensors"))
(mixed / "config.json").write_text("{}")
raises("partial contamination rejected",
       lambda: mg._assert_dir_is_loadable(mixed, "mixed"), "was NOT merged")

print("\n--- sharded merged checkpoint still accepted ---")
sharded = tmp / "sharded"; sharded.mkdir()
save_file({"model.language_model.layers.0.self_attn.q_proj.weight": t()},
          str(sharded / "model-00001-of-00002.safetensors"))
save_file({"lm_head.weight": t()},
          str(sharded / "model-00002-of-00002.safetensors"))
(sharded / "config.json").write_text("{}")
mg._assert_dir_is_loadable(sharded, "sharded merged")
check("sharded merged accepted", True)

print("\n--- contamination in the SECOND shard is still caught ---")
sh2 = tmp / "sharded_bad"; sh2.mkdir()
save_file({"model.language_model.layers.0.self_attn.q_proj.weight": t()},
          str(sh2 / "model-00001-of-00002.safetensors"))
save_file({"model.language_model.layers.1.mlp.up_proj.lora_A.default.weight": t()},
          str(sh2 / "model-00002-of-00002.safetensors"))
(sh2 / "config.json").write_text("{}")
raises("second-shard contamination caught",
       lambda: mg._assert_dir_is_loadable(sh2, "sharded bad"), "was NOT merged")

print("\n--- pre-existing checks still fire ---")
adap = tmp / "adap"; adap.mkdir()
save_file({"model.layers.0.q_proj.weight": t()}, str(adap / "model.safetensors"))
(adap / "adapter_config.json").write_text("{}")
raises("adapter_config.json still rejected",
       lambda: mg._assert_dir_is_loadable(adap, "hybrid"), "loaded as a LoRA")

empty = tmp / "empty"; empty.mkdir()
(empty / "config.json").write_text("{}")
raises("no weights still rejected",
       lambda: mg._assert_dir_is_loadable(empty, "empty"), "not loadable by LLM.py")

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
