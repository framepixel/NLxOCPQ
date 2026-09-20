#!/usr/bin/env python3
"""Pin the OCEL logs an evaluation reads, and refuse to evaluate a different copy of them.

2026-09-12 (Thesis/notes/11-evaluation-readiness.md 35): the cluster evaluated OLD copies of three
logs for ten weeks. `renting_log_high.json`, `ocel2-p2p.json` and `container_logistics.json` had been
retyped on the laptop; the cluster kept its June files, gold returned nothing on 14 cases, and 56
cases were prompted with the old attribute types. Nothing recorded it: `pipeline_fingerprint` hashes
the code, `testset_hash` hashes the testset, and no field of any run described the logs themselves.

This closes that gap in two places:

  * `data/corpus_manifest.json` pins every testset log by size and sha256. It is committed, so a
    deliberate corpus change is a visible diff, made with `--pin` in the same commit.
  * `check_eval_deps.py` (every SLURM job's preflight) and `eval_testset.py` (every run, config key
    `corpus_check`) compare the logs on disk against it. A mismatch STOPS the job before a GPU is
    used -- decided 2026-09-15, because a warning is exactly what scrolled past the last time.
    Each run also records the digests of the logs it read in `config_resolved.json`.

Deliberately light: stdlib and `paths` only, so the preflight can import it in milliseconds.
Hashing itself is the cost -- ~3.5 GB for the whole testset corpus, bpic2017 alone 2.3 GB.

    python src/corpus_manifest.py --check     # exit 1 on any difference
    python src/corpus_manifest.py --pin       # after a DELIBERATE corpus change; commit the result
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import paths

MANIFEST_VERSION = 1
_CHUNK_BYTES = 1 << 20


def resolve_log_path(raw_path: str, anchor_dir: Path = paths.REPO_ROOT) -> Path:
    """Resolve a testset `ocel_path` the way the evaluation does.

    Mirrors `eval_testset._resolve_ocel_path` and `check_eval_deps._resolve_ocel_path`, including the
    separator normalisation: `testset.json` stores Windows-style `..\\Logs\\x.json`, which on Linux is
    one filename containing backslashes. Not imported from either, because importing `eval_testset`
    loads the whole pipeline.

    Args:
        raw_path: The `ocel_path` as written in the testset.
        anchor_dir: The directory relative paths are resolved against (the repo root).

    Returns:
        The absolute path of the log.
    """
    normalized = str(raw_path).replace("\\", os.sep).replace("/", os.sep)
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    return (Path(anchor_dir) / candidate).resolve()


def file_digest(path: Path) -> Dict[str, object]:
    """Size and sha256 of one file, streamed so a multi-GB log is never held in memory.

    Args:
        path: The file to hash.

    Returns:
        `{"bytes": int, "sha256": str}`.
    """
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            hasher.update(chunk)
            size += len(chunk)
    return {"bytes": size, "sha256": hasher.hexdigest()}


def digest_logs(log_paths: Iterable) -> Dict[str, Optional[Dict[str, object]]]:
    """Digest each distinct log, keyed by basename. A missing log maps to None.

    Keyed by basename rather than by path because the same log lives at a different absolute path on
    each machine, and it is the file's content that has to agree, not where it sits.

    Args:
        log_paths: Resolved log paths; duplicates are hashed once.

    Returns:
        `{basename: {"bytes", "sha256"} | None}`, sorted by basename.

    Raises:
        ValueError: If two different paths share a basename -- the key would be ambiguous.
    """
    by_name: Dict[str, Path] = {}
    for raw in log_paths:
        path = Path(raw)
        prior = by_name.get(path.name)
        if prior is not None and prior != path:
            raise ValueError(
                f"two logs share the basename {path.name!r} ({prior} and {path}); the corpus "
                "manifest keys logs by basename, so one of them must be renamed")
        by_name[path.name] = path
    return {name: (file_digest(path) if path.is_file() else None)
            for name, path in sorted(by_name.items())}


def testset_logs(testset_path: Path = paths.DATA / "testset.json") -> Dict[str, Path]:
    """Every log the testset references, keyed by basename.

    Args:
        testset_path: The testset JSON.

    Returns:
        `{basename: resolved path}`.
    """
    rows = json.loads(Path(testset_path).read_text(encoding="utf-8"))
    resolved = {resolve_log_path(row["ocel_path"]) for row in rows if row.get("ocel_path")}
    return {path.name: path for path in sorted(resolved)}


def load_pinned(manifest_path: Path = paths.CORPUS_MANIFEST) -> Optional[dict]:
    """The pinned manifest, or None when there is none.

    Args:
        manifest_path: Where the manifest lives.

    Returns:
        The parsed manifest, or None if the file does not exist.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def compare(observed: Dict[str, Optional[Dict[str, object]]], pinned: Optional[dict],
            manifest_path: Path = paths.CORPUS_MANIFEST) -> List[str]:
    """Every way the observed logs differ from the pinned ones. Empty means identical.

    Only the logs in `observed` are checked, so a shard that reads four of the eleven logs is not
    failed for the other seven.

    Args:
        observed: Output of `digest_logs`.
        pinned: Output of `load_pinned` (None if no manifest exists).
        manifest_path: Named in the messages, so a failure says which file to update.

    Returns:
        One human-readable problem per differing log, each naming the log.
    """
    if pinned is None:
        return [f"no pinned corpus manifest at {manifest_path} -- create it on the machine whose "
                "logs are authoritative with `python src/corpus_manifest.py --pin` and commit it"]
    pinned_logs = pinned.get("logs") or {}
    problems = []
    for name, got in sorted(observed.items()):
        want = pinned_logs.get(name)
        if got is None:
            problems.append(f"{name}: the log file is missing")
        elif want is None:
            problems.append(f"{name}: not in the pinned manifest {manifest_path}")
        elif got["bytes"] != want.get("bytes"):
            problems.append(f"{name}: {got['bytes']:,} bytes on disk, pinned "
                            f"{want.get('bytes'):,} -- a different copy of this log")
        elif got["sha256"] != want.get("sha256"):
            problems.append(f"{name}: sha256 {got['sha256'][:16]}... on disk, pinned "
                            f"{str(want.get('sha256'))[:16]}... -- same size, different content")
    return problems


def verify(testset_path: Path = paths.DATA / "testset.json",
           manifest_path: Path = paths.CORPUS_MANIFEST,
           basenames: Optional[Iterable[str]] = None) -> List[str]:
    """Compare the testset's logs on disk with the pinned manifest.

    The manifest is checked for existence first, so a missing manifest fails without hashing
    3.5 GB.

    Args:
        testset_path: The testset whose logs are checked.
        manifest_path: The pinned manifest.
        basenames: Restrict the check to these logs (e.g. one shard's slice). None checks all.

    Returns:
        Problems, as in `compare`. Empty means the corpus is the pinned one.
    """
    pinned = load_pinned(manifest_path)
    if pinned is None:
        return compare({}, None, manifest_path)
    logs = testset_logs(testset_path)
    if basenames is not None:
        wanted = set(basenames)
        logs = {name: path for name, path in logs.items() if name in wanted}
    return compare(digest_logs(logs.values()), pinned, manifest_path)


def pin(testset_path: Path = paths.DATA / "testset.json",
        manifest_path: Path = paths.CORPUS_MANIFEST) -> dict:
    """Write the manifest for the logs currently on disk.

    Refuses when a log is missing: a manifest pinning an absent file would make every later check
    fail for a reason that has nothing to do with the corpus.

    Args:
        testset_path: The testset whose logs are pinned.
        manifest_path: Where to write.

    Returns:
        The manifest that was written.

    Raises:
        FileNotFoundError: If any referenced log is missing.
    """
    logs = digest_logs(testset_logs(testset_path).values())
    missing = sorted(name for name, digest in logs.items() if digest is None)
    if missing:
        raise FileNotFoundError(f"cannot pin a corpus with missing logs: {missing}")
    manifest = {
        "version": MANIFEST_VERSION,
        "pinned_at": time.strftime("%Y-%m-%d"),
        "note": ("Size and sha256 of every OCEL log data/testset.json references. Checked by "
                 "check_eval_deps.py and eval_testset.py (corpus_check); a mismatch stops the run. "
                 "Regenerate ONLY after a deliberate corpus change, with "
                 "`python src/corpus_manifest.py --pin`, in the same commit. "
                 "See Thesis/notes/11-evaluation-readiness.md 35."),
        "logs": logs,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """CLI entry point: `--pin` or `--check`."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="exit 1 if any log differs")
    mode.add_argument("--pin", action="store_true", help="write the manifest for the logs on disk")
    ap.add_argument("--testset", default="data/testset.json")
    ap.add_argument("--manifest", default=None, help="default: data/corpus_manifest.json")
    args = ap.parse_args()

    testset_path = Path(args.testset)
    if not testset_path.is_absolute():
        testset_path = paths.REPO_ROOT / testset_path
    manifest_path = Path(args.manifest) if args.manifest else paths.CORPUS_MANIFEST

    started = time.time()
    if args.pin:
        manifest = pin(testset_path, manifest_path)
        total = sum(entry["bytes"] for entry in manifest["logs"].values())
        print(f"pinned {len(manifest['logs'])} log(s), {total / 1e9:.2f} GB, into {manifest_path} "
              f"in {time.time() - started:.1f}s")
        for name, entry in manifest["logs"].items():
            print(f"  {name:<40} {entry['bytes']:>14,}  {entry['sha256'][:16]}")
        return 0

    problems = verify(testset_path, manifest_path)
    elapsed = time.time() - started
    if problems:
        print(f"CORPUS MISMATCH against {manifest_path} ({elapsed:.1f}s):", file=sys.stderr)
        for line in problems:
            print(f"  - {line}", file=sys.stderr)
        return 1
    print(f"corpus OK: every testset log matches {manifest_path} ({elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
