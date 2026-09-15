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

    python src/check_eval_deps.py            # quiet on success, exit 0
    python src/check_eval_deps.py --verbose  # print resolved versions
"""
import argparse
import os
import sys
from pathlib import Path

import paths

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
    # Only needed when a config sets `constrained_decoding: true`, but checked unconditionally:
    # the failure would otherwise surface hours into an array job, at the first generate() call.
    ("xgrammar", "constrained decoding (grammar-masked generation, LLM._ocpq_logits_processor)"),
]


def _resolve_ocel_path(raw_path, anchor_dir):
    """Resolve a testset `ocel_path` the way the evaluation does.

    Mirrors `eval_testset._resolve_ocel_path`, deliberately rather than importing it: importing
    `eval_testset` pulls in `LLM`, which calls `load_resources()` at module level and costs ~13 s
    plus an embedding-model load — far too heavy for a preflight whose job is to fail in
    milliseconds.

    The separator normalisation is the whole point. `testset.json` stores Windows-style
    `..\\Logs\\03_hiring.xml`; on Linux that is a single filename containing backslashes, so a
    naive `exists()` reports every log as missing. That false alarm blocked a smoke run.
    `scratchpad/test_path_resolution.py` asserts this function agrees with the evaluation's on
    every testset entry, so the duplication cannot drift unnoticed.
    """
    normalized = str(raw_path).replace("\\", os.sep).replace("/", os.sep)
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return (Path(anchor_dir) / candidate).resolve()


def check_data_files(config_path):
    """Verify the input data a run needs. Returns a list of problems.

    Imports are not the only way a job can waste GPU hours. The 2026-08-15 smoke run completed
    and produced 144 records with *no stratum labels at all*, because
    `eval_runs/testset_discriminability.json` had not been copied to the cluster: the code warns
    and carries on, which is right for a partial run but meant three GPU-hours of unusable
    strata. Data absence is cheap to detect here and expensive to detect afterwards.
    """
    import json

    problems = []

    required = [
        (paths.DATA / "testset.json", "the evaluation testset"),
        (paths.DATA / "testset_bb_labels.json",
         "building-block labels for the per-BB breakdown"),
        (paths.EVAL_RUNS / "testset_discriminability.json",
         "stratum labels and the trivial floor (skill metrics are dead without it)"),
        (paths.DATA / "ocpq_dataset.json", "few-shot retrieval corpus"),
        (paths.DATA / "embeddings.npy", "few-shot retrieval embeddings"),
    ]
    for path, why in required:
        if not path.exists():
            problems.append(f"missing {path} — {why}")

    # The OCEL logs are referenced by relative path from the testset; a missing one fails only
    # when that case is reached, potentially hours in.
    testset_path = paths.DATA / "testset.json"
    if testset_path.exists():
        try:
            rows = json.loads(testset_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            problems.append(f"testset.json is unreadable: {exc}")
            rows = []
        missing_logs = sorted({
            row["ocel_path"] for row in rows
            if row.get("ocel_path")
            and not _resolve_ocel_path(row["ocel_path"], paths.REPO_ROOT).exists()
        })
        for log in missing_logs[:5]:
            problems.append(f"missing OCEL log {_resolve_ocel_path(log, paths.REPO_ROOT)}")
        if len(missing_logs) > 5:
            problems.append(f"... and {len(missing_logs) - 5} more missing OCEL log(s)")

        # Present is not enough. For ten weeks the cluster had all eleven logs and three of them
        # were old copies (Thesis/notes/11-evaluation-readiness.md 35). Compared by size and
        # sha256 against the committed data/corpus_manifest.json; decided 2026-09-15 to stop.
        if rows and not missing_logs:
            from corpus_manifest import verify  # stdlib + paths only: still a light import
            problems.extend(f"corpus: {line}"
                            for line in verify(testset_path, paths.CORPUS_MANIFEST))

    return problems

# Mirrors `LLM.is_merged_model_dir` and LLM.py's adapter check, deliberately rather than importing
# them: importing LLM loads the whole pipeline (~13 s plus an embedding model), and this preflight
# must fail in milliseconds.
_MERGED_MARKERS = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model-00001-of-00002.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
_ADAPTER_MARKERS = ("adapter_config.json", "adapter_model.safetensors")
_LFS_POINTER_PREFIX = b"version https://git-lfs"


def check_model_dirs(config_path):
    """Verify every LOCAL model directory the config names exists and holds loadable weights.

    2026-09-15: the Gemma arms are merged 16-bit checkpoints that live outside git (src/merge_gemma.py,
    Thesis/notes/11-evaluation-readiness.md 17). The 2026-08-29 reorg changed the config from
    `adapters/...` to `models/adapters/...` but could not move untracked files, so on the cluster the
    checkpoints stayed at the old path. `LLM.py` treats a name that is not an existing directory as a
    Hugging Face repo id, so the first smoke run after the reorg lost all 24 Gemma runs to
    `HFValidationError` -- discovered at the first generation, after the job had queued and loaded,
    while every other arm ran normally. Hub ids (`org/name`) are left to the offline cache check.

    Args:
        config_path: The evaluation config, absolute or repo-root-relative.

    Returns:
        One problem per missing or unloadable local model directory, each prefixed `model:`.
    """
    import glob
    import json

    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = paths.REPO_ROOT / cfg_path
    if not cfg_path.is_file():
        return [f"model: evaluation config not found: {cfg_path}"]
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    names = set()
    for model in cfg.get("models") or []:
        if isinstance(model, dict) and model.get("name"):
            names.add(str(model["name"]))
    for base_name, finetuned in (cfg.get("fine_tune_map") or {}).items():
        names.add(str(base_name))
        if isinstance(finetuned, dict) and finetuned.get("name"):
            names.add(str(finetuned["name"]))

    problems = []
    for name in sorted(names):
        path = Path(name) if Path(name).is_absolute() else paths.REPO_ROOT / name
        looks_local = (Path(name).is_absolute() or path.is_dir()
                       or name.startswith(("models/", "./", "../", "/")))
        if not looks_local:
            continue
        if not path.is_dir():
            problems.append(
                f"model: {name} -- no such directory under {paths.REPO_ROOT}. LLM.py would treat it "
                "as a Hugging Face repo id and fail at the first generation")
            continue
        is_adapter = any((path / marker).exists() for marker in _ADAPTER_MARKERS)
        is_merged = (any((path / marker).exists() for marker in _MERGED_MARKERS)
                     or bool(glob.glob(str(path / "model-*-of-*.safetensors")))
                     or bool(glob.glob(str(path / "pytorch_model-*-of-*.bin"))))
        if not (is_adapter or is_merged):
            problems.append(f"model: {name} -- the directory holds neither a LoRA adapter nor "
                            "merged weights")
            continue
        # A directory checked out without `git lfs pull` has the right filenames and ~130-byte
        # pointer files instead of weights; that fails much later and far less legibly.
        for weights in sorted(path.glob("*.safetensors")):
            with open(weights, "rb") as handle:
                if handle.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                    problems.append(f"model: {weights} is a Git LFS pointer, not weights -- "
                                    "run `git lfs pull`")
                    break
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="print resolved versions")
    ap.add_argument("--skip-data", action="store_true",
                    help="check imports only, without the input-data files")
    ap.add_argument("--config", default="configs/eval_config.json",
                    help="config whose local model directories are checked")
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

    data_problems = [] if args.skip_data else check_data_files(None)
    if not args.skip_data:
        data_problems.extend(check_model_dirs(args.config))
    if data_problems:
        print("FATAL: required input data is missing or is not the pinned version.",
              file=sys.stderr)
        for line in data_problems:
            print(f"  - {line}", file=sys.stderr)
        if any(line.startswith("corpus:") for line in data_problems):
            print("\nThe logs differ from data/corpus_manifest.json. Sync them from the machine "
                  "the manifest was pinned on -- or, for a DELIBERATE corpus change, run "
                  "`python src/corpus_manifest.py --pin` and commit the manifest.",
                  file=sys.stderr)
        if any(line.startswith("model:") for line in data_problems):
            print("\nA local model directory is missing or unloadable. The Gemma arms are merged "
                  "checkpoints kept outside git (src/merge_gemma.py, notes 17): move them into "
                  "models/adapters/ or regenerate them in the training environment.",
                  file=sys.stderr)
        print("\nCopy the current NLxOCPQ files to the cluster, including eval_runs/ "
              "(a plain rsync that excludes it drops the discriminability audit).",
              file=sys.stderr)
        return 1

    suffix = "" if args.skip_data else " + input data + pinned corpus + model dirs"
    print(f"eval dependencies OK ({len(versions)} modules{suffix})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
