#!/usr/bin/env python3
"""Fail fast if the environment cannot run an evaluation. Exits non-zero on the first problem.

Every eval job calls this before touching a GPU. `LLM.py` imports `sklearn` and `openai`
unguarded at module level, so a gap surfaces as an ImportError minutes into a 14-hour job that
has already been queued, scheduled and allocated a GPU.

Deliberately *imports* each module rather than checking `importlib.util.find_spec`. find_spec
only proves a module is findable; it says nothing about whether it imports. A scikit-learn
whose scipy is broken, or a torch built against the wrong CUDA, passes find_spec and then dies
at the real import. Importing costs a few seconds once per job and is the thing actually being
asserted.

    python check_eval_deps.py            # quiet on success, exit 0
    python check_eval_deps.py --verbose  # print resolved versions
"""
import argparse
import sys

# Third-party modules reached on the evaluation path (eval_testset -> LLM/QueryExecutor/...).
REQUIRED = [
    ("torch", "local-gpu inference"),
    ("transformers", "model + tokenizer loading"),
    ("peft", "LoRA adapter loading"),
    ("safetensors", "adapter weight loading"),
    ("huggingface_hub", "model resolution"),
    ("numpy", "embeddings"),
    ("sklearn", "few-shot retrieval (cosine_similarity, LLM.py:595)"),
    ("openai", "imported unconditionally by LLM.py:19; unused when backend=local-gpu"),
    ("pydantic", "BindingBoxTree schema validation"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print resolved versions")
    args = ap.parse_args()

    failures = []
    versions = []
    for name, why in REQUIRED:
        try:
            module = __import__(name)
        except Exception as exc:  # noqa: BLE001 - any import failure is disqualifying
            failures.append(f"{name} ({why}): {type(exc).__name__}: {exc}")
            continue
        versions.append((name, getattr(module, "__version__", "?")))

    if args.verbose:
        for name, version in versions:
            print(f"  {name:<16} {version}")

    if failures:
        print("FATAL: the evaluation environment is not usable.", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        print("\nBuild the eval environment with:  bash make_eval_env.sh", file=sys.stderr)
        return 1

    print(f"eval dependencies OK ({len(versions)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
