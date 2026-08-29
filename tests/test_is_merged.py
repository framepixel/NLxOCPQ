"""is_merged_model_dir: the predicate LLM.py and merge_gemma.py now share.

Regression target: the exact on-disk layout unsloth produced on 2026-08-19, which the old
hard-coded filename tuple rejected and which broke all 48 Gemma smoke conditions.

Run from the NLxOCPQ directory:  python tests/test_is_merged.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from LLM import is_merged_model_dir  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got {got!r}  want {want!r}")
        failures.append(name)


def make(tmp, name, files):
    d = Path(tmp) / name
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text("{}")
    return str(d)


OLD_RULE = ("pytorch_model.bin", "pytorch_model-00001-of-00002.bin", "model.safetensors")


def old_accepts(files):
    return any(f in files for f in OLD_RULE)


tmp = tempfile.mkdtemp(prefix="ismerged_")

# THE regression: exactly what merge_gemma produced.
gemma = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
         "model.safetensors.index.json", "config.json", "tokenizer.json",
         "tokenizer_config.json", "tokenizer.model"]
print("--- the 2026-08-19 layout that broke 48 conditions ---")
check("old hard-coded rule REJECTED it (the bug)", old_accepts(gemma), False)
check("shared predicate accepts it", is_merged_model_dir(make(tmp, "gemma", gemma)), True)

print("\n--- other real layouts ---")
cases = [
    ("single-file safetensors", ["model.safetensors", "config.json"], True),
    ("legacy single bin", ["pytorch_model.bin", "config.json"], True),
    ("legacy two-shard bin", ["pytorch_model-00001-of-00002.bin",
                              "pytorch_model-00002-of-00002.bin"], True),
    ("sharded bin with index", ["pytorch_model-00001-of-00003.bin",
                                "pytorch_model.bin.index.json"], True),
    ("index only, shards not yet written", ["model.safetensors.index.json"], True),
    ("many shards", [f"model-0000{i}-of-00005.safetensors" for i in range(1, 6)], True),
]
for name, files, want in cases:
    check(name, is_merged_model_dir(make(tmp, name.replace(" ", "_"), files)), want)

print("\n--- must NOT be mistaken for merged weights ---")
neg = [
    ("LoRA adapter dir", ["adapter_config.json", "adapter_model.safetensors"], False),
    ("config + tokenizer only", ["config.json", "tokenizer.json"], False),
    ("empty dir", [], False),
    ("unrelated safetensors name", ["embeddings.safetensors"], False),
]
for name, files, want in neg:
    check(name, is_merged_model_dir(make(tmp, name.replace(" ", "_"), files)), want)

check("nonexistent path", is_merged_model_dir(str(Path(tmp) / "nope")), False)

print("\n--- an adapter dir is still caught by the caller's ordering ---")
# LLM.py checks adapter_config.json BEFORE this predicate, so a dir with both resolves as an
# adapter. Documenting the interaction: the predicate alone must not be read as "load as merged".
both = make(tmp, "both", ["adapter_config.json", "model.safetensors"])
check("predicate alone says True for a hybrid dir", is_merged_model_dir(both), True)
print("       (LLM.py's adapter branch runs first, so the hybrid loads as a LoRA -- and")
print("        merge_gemma._assert_dir_is_loadable rejects hybrids outright.)")

import shutil
shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
