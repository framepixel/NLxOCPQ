"""A run must refuse to evaluate on a copy of the logs other than the pinned one.

2026-09-12 (Thesis/notes/11-evaluation-readiness.md 35): the cluster evaluated old copies of three
logs for ten weeks, and no run recorded it -- `pipeline_fingerprint` hashes code, `testset_hash` the
testset, nothing the logs. `src/corpus_manifest.py` pins every log by size and sha256; the preflight
and `eval_testset._corpus_gate` compare against it, and a mismatch stops the job (decided
2026-09-15). This asserts both directions: the pinned corpus passes, and every kind of difference
fails, naming the log.

Run from the repo root:  python tests/test_corpus_manifest.py
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import corpus_manifest  # noqa: E402
import paths  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


tmp = Path(tempfile.mkdtemp(prefix="corpus_"))
logs = tmp / "Logs"
logs.mkdir()
A = logs / "a.json"
B = logs / "b.xml"
A_BYTES = b"A" * 1000
B_BYTES = b"<b/>" * 50
A.write_bytes(A_BYTES)
B.write_bytes(B_BYTES)
testset = tmp / "testset.json"
manifest = tmp / "corpus_manifest.json"


def write_testset(*log_paths):
    testset.write_text(json.dumps([{"nl_query": f"q{i}", "ocel_path": str(p)}
                                   for i, p in enumerate(log_paths)]), encoding="utf-8")


write_testset(A, A, B)   # a log referenced twice is pinned once

print("--- pin ---")
pinned = corpus_manifest.pin(testset, manifest)
check("one entry per distinct log", sorted(pinned["logs"]), ["a.json", "b.xml"])
check("size recorded", pinned["logs"]["a.json"]["bytes"], len(A_BYTES))
check("sha256 recorded", pinned["logs"]["a.json"]["sha256"], hashlib.sha256(A_BYTES).hexdigest())
check("the file on disk is what was returned",
      json.loads(manifest.read_text(encoding="utf-8"))["logs"], pinned["logs"])

print("\n--- the pinned corpus passes ---")
check("verify on the pinned corpus", corpus_manifest.verify(testset, manifest), [])

print("\n--- every kind of difference fails, naming the log ---")
A.write_bytes(b"B" + A_BYTES[1:])            # same size, one byte different
problems = corpus_manifest.verify(testset, manifest)
check("same size, different content -> one problem", len(problems), 1)
check("...naming a.json", "a.json" in problems[0], True)
check("...as a content difference", "same size, different content" in problems[0], True)
A.write_bytes(A_BYTES)

B.write_bytes(B_BYTES + b"\n")               # the renting case: a different size
problems = corpus_manifest.verify(testset, manifest)
check("different size -> one problem naming b.xml",
      (len(problems), "b.xml" in problems[0], "bytes on disk" in problems[0]), (1, True, True))
check("a shard that does not read b.xml is not failed for it",
      corpus_manifest.verify(testset, manifest, basenames={"a.json"}), [])
B.unlink()
problems = corpus_manifest.verify(testset, manifest)
check("a missing log fails", (len(problems), "missing" in problems[0]), (1, True))
B.write_bytes(B_BYTES)

C = logs / "c.json"
C.write_bytes(b"{}")
write_testset(A, B, C)
problems = corpus_manifest.verify(testset, manifest)
check("a log the manifest never pinned fails",
      (len(problems), "c.json" in problems[0], "not in the pinned manifest" in problems[0]),
      (1, True, True))
write_testset(A, B)

print("\n--- no manifest ---")
nowhere = tmp / "absent.json"
problems = corpus_manifest.verify(tmp / "no_testset_either.json", nowhere)
check("a missing manifest fails without reading the testset or hashing anything",
      (len(problems), "--pin" in problems[0]), (1, True))

print("\n--- pin refuses what it cannot pin ---")
write_testset(A, logs / "gone.json")
try:
    corpus_manifest.pin(testset, tmp / "never.json")
    check("pinning a missing log raises", False, True)
except FileNotFoundError:
    check("pinning a missing log raises", True, True)
check("...and writes nothing", (tmp / "never.json").exists(), False)
write_testset(A, B)

other = tmp / "other"
other.mkdir()
(other / "a.json").write_bytes(b"different")
try:
    corpus_manifest.digest_logs([A, other / "a.json"])
    check("two logs sharing a basename raise", False, True)
except ValueError:
    check("two logs sharing a basename raise", True, True)

print("\n--- path resolution matches the evaluation's ---")
check("a Windows-style relative path resolves against the repo root",
      corpus_manifest.resolve_log_path("..\\Logs\\x.json"),
      (paths.REPO_ROOT / ".." / "Logs" / "x.json").resolve())

print("\n--- eval_testset: the gate and the merge ---")
# Imported late: eval_testset pulls in the whole pipeline (LLM, torch). Everything above runs without it.
from eval_testset import (  # noqa: E402
    _PIPELINE_SOURCE_FILES,
    _corpus_gate,
    _merge_shards,
    _resolve_ocel_path,
)

rows = [{"ocel_path": str(A)}, {"ocel_path": str(B)}]
digests, problems = _corpus_gate(rows, "error", manifest)
check("gate passes the pinned corpus", (sorted(digests), problems), (["a.json", "b.xml"], []))

A.write_bytes(b"B" + A_BYTES[1:])
try:
    _corpus_gate(rows, "error", manifest)
    check("gate 'error' raises on a difference", False, True)
except RuntimeError:
    check("gate 'error' raises on a difference", True, True)
digests, problems = _corpus_gate(rows, "warn", manifest)
check("gate 'warn' continues and returns the problem", (len(problems), "a.json" in problems[0]),
      (1, True))
check("gate 'warn' still records the digests", digests["a.json"]["bytes"], len(A_BYTES))
A.write_bytes(A_BYTES)
check("gate 'off' neither hashes nor checks, even with a missing log",
      _corpus_gate([{"ocel_path": str(logs / "gone.json")}], "off", manifest), (None, []))
try:
    _corpus_gate(rows, "maybe", manifest)
    check("an unknown policy raises", False, True)
except ValueError:
    check("an unknown policy raises", True, True)
try:
    _corpus_gate(rows, "error", nowhere)
    check("gate 'error' with no manifest raises", False, True)
except RuntimeError:
    check("gate 'error' with no manifest raises", True, True)

check("corpus_manifest.py is part of the pipeline fingerprint",
      "corpus_manifest.py" in _PIPELINE_SOURCE_FILES, True)

testset_rows = json.loads((paths.DATA / "testset.json").read_text(encoding="utf-8"))
disagree = [r["ocel_path"] for r in testset_rows
            if Path(_resolve_ocel_path(r["ocel_path"], paths.REPO_ROOT))
            != corpus_manifest.resolve_log_path(r["ocel_path"])]
check("resolution agrees with eval_testset on every testset entry", disagree, [])

CFG = "m|s|True"


def rec(nl, log):
    f1 = 0.5
    return {"case_id": 1, "nl_query": nl, "ocel_path": log,
            "domain_seen": True, "discriminating": True, "baseline_f1": 0.4,
            "gold_query": {"nodes": []}, "prediction": {"nodes": []},
            "model": "m", "model_label": "m", "prompt_strategy": "schema-grammar-guided",
            "schema_enforced": True, "schema_validation": True, "repeat": 0,
            "config_id": CFG, "status": "success", "pipeline_status": "ok",
            "similarity": {"combined": {"f1": f1, "precision": f1, "recall": f1},
                           "events": {"jaccard": f1, "precision": f1, "recall": f1},
                           "objects": {"jaccard": f1, "precision": f1, "recall": f1}},
            "llm": {}, "execution": {"success": True, "duration_sec": 1.0},
            "total_duration_sec": 1.0,
            "metrics": {"schema_valid": True, "hallucinated_entities": False,
                        "unsupported_filters": False,
                        "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                        "bindings_skipped": False, "gold_bindings_skipped": False}}


def shard(name, log, corpus):
    d = tmp / "shards" / name
    d.mkdir(parents=True)
    (d / "predictions.json").write_text(json.dumps([rec(f"q {name}", log)]), encoding="utf-8")
    manifest_doc = {"pipeline_fingerprint": "same", "repeats": 1}
    if corpus is not None:
        manifest_doc["corpus"] = corpus
    (d / "config_resolved.json").write_text(json.dumps(manifest_doc), encoding="utf-8")
    return str(d)


def merged_summary(name, shard_dirs):
    out = tmp / "merged" / name
    _merge_shards(shard_dirs, out)
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


a_ok = {"a.json": pinned["logs"]["a.json"]}
b_ok = {"b.xml": pinned["logs"]["b.xml"]}
a_old = {"a.json": {"bytes": 999, "sha256": "0" * 64}}

summary = merged_summary("agree", [shard("h0", str(A), a_ok), shard("h1", str(B), b_ok)])
check("case-halves reading different logs, each the pinned copy -> consistent",
      (summary["corpus_consistent"], summary["corpus_conflicts"]), (True, []))

summary = merged_summary("conflict", [shard("x0", str(A), a_ok), shard("x1", str(A), a_old)])
check("two shards reading different copies of one log -> inconsistent, naming it",
      (summary["corpus_consistent"], summary["corpus_conflicts"]), (False, ["a.json"]))

summary = merged_summary("old", [shard("o0", str(A), a_ok), shard("o1", str(B), None)])
check("a shard that recorded no digests -> cannot be established (None)",
      (summary["corpus_consistent"], summary["corpus_unrecorded_shards"]), (None, 1))

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> only the pinned corpus is evaluated, and merges compare it per log")
