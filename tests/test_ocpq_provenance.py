"""Every run names the evaluator that computed it, and a merge refuses to hide a mix of evaluators.

Until 2026-09-22 nothing recorded which `ocpq_cli` produced a run's result sets, although it
computes every one of them and `cargo run --release` silently rebuilds it whenever the OCPQ source
is newer than the binary. Three ways the new record could still mislead:

  * **Recording the wrong binary.** A stale binary is hashed, then cargo replaces it on the first
    query. `cli_older_than_source` must say so, and `build=True` must let cargo build BEFORE the
    digest is taken.
  * **Crashing the run it describes.** Provenance that can raise would be switched off, and then
    record nothing. A missing git, checkout or binary is recorded, never raised.
  * **A merge that averages over two evaluators.** `ocpq_consistent` must be False when shards
    ran different binaries or one saw its binary change mid-run, and None when it cannot be told.

Run from the repo root:  python tests/test_ocpq_provenance.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import paths  # noqa: E402
import QueryExecutor  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def git(*args):
    return subprocess.run(["git", "-C", str(checkout), "-c", "user.name=t",
                           "-c", "user.email=t@example.invalid", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


tmp = Path(tempfile.mkdtemp(prefix="ocpq-provenance-"))
checkout = tmp / "OCPQ"
source = checkout / "backend" / "ocpq_cli" / "src" / "main.rs"
source.parent.mkdir(parents=True)
source.write_text("fn main() {}\n", encoding="utf-8")
(checkout / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
binary_name = "ocpq_cli" + (".exe" if os.name == "nt" else "")
binary = checkout / "target" / "release" / binary_name
binary.parent.mkdir(parents=True)
binary.write_bytes(b"pretend evaluator v1")
git("init", "-q")
git("add", "backend", "Cargo.toml")
git("commit", "-q", "-m", "pinned")
rev = git("rev-parse", "HEAD")

real_checkout = paths.OCPQ_CHECKOUT
paths.OCPQ_CHECKOUT = checkout
os.environ.pop("CARGO_TARGET_DIR", None)

print("--- what a run records ---")
now = time.time()
os.utime(source, (now - 100, now - 100))
os.utime(binary, (now, now))
record = QueryExecutor.ocpq_provenance(build=False)
check("the source revision", record["rev"], rev)
check("a clean checkout is not dirty", (record["dirty"], record["dirty_files"]), (False, []))
check("the binary's sha256 is of its bytes",
      record["binaries"]["ocpq_cli"]["sha256"],
      hashlib.sha256(b"pretend evaluator v1").hexdigest())
check("a web-server that was never built is recorded as absent, not an error",
      record["binaries"]["ocpq-web-server"], None)
check("a binary newer than its source is current", record["cli_older_than_source"], False)
check("the record is JSON-serialisable for config_resolved.json", bool(json.dumps(record)), True)

print("\n--- the two ways it goes stale ---")
source.write_text("fn main() { changed(); }\n", encoding="utf-8")
os.utime(source, (now + 100, now + 100))
record = QueryExecutor.ocpq_provenance(build=False)
check("an edited tracked file makes the checkout dirty, and is named",
      (record["dirty"], record["dirty_files"]), (True, ["backend/ocpq_cli/src/main.rs"]))
check("source newer than the binary: the next cargo run would replace it",
      record["cli_older_than_source"], True)
git("checkout", "-q", "--", ".")
os.utime(source, (now - 100, now - 100))

print("\n--- build=True lets cargo build BEFORE the digest is taken ---")
order = []
real_detect, real_find = QueryExecutor._detect_cli_mode, QueryExecutor._find_cargo_command


def fake_detect(cargo, cli_dir):
    order.append("build")
    binary.write_bytes(b"pretend evaluator v2")        # what cargo would do to a stale binary
    return "subcommand"


QueryExecutor._detect_cli_mode = fake_detect
QueryExecutor._find_cargo_command = lambda: "cargo"
record = QueryExecutor.ocpq_provenance(build=True)
check("cargo ran first", order, ["build"])
check("so the digest is of the REBUILT binary, the one that will actually evaluate",
      record["binaries"]["ocpq_cli"]["sha256"],
      hashlib.sha256(b"pretend evaluator v2").hexdigest())
check("and the CLI mode is recorded", record["cli_mode"], "subcommand")


def no_cargo():
    raise FileNotFoundError("Cargo was not found.")


QueryExecutor._find_cargo_command = no_cargo
record = QueryExecutor.ocpq_provenance(build=True)
check("no cargo: recorded as a build error, never raised",
      record["build_error"].startswith("FileNotFoundError"), True)
QueryExecutor._detect_cli_mode, QueryExecutor._find_cargo_command = real_detect, real_find

print("\n--- nothing to fingerprint is recorded, not raised ---")
os.environ["CARGO_TARGET_DIR"] = str(tmp / "elsewhere")
record = QueryExecutor.ocpq_provenance(build=False)
check("$CARGO_TARGET_DIR is where cargo puts it, so that is where it is looked for",
      record["binaries"]["ocpq_cli"], None)
check("...and with no binary, staleness is unknown rather than guessed",
      record["cli_older_than_source"], None)
os.environ.pop("CARGO_TARGET_DIR")
paths.OCPQ_CHECKOUT = tmp / "no-such-checkout"
record = QueryExecutor.ocpq_provenance(build=False)
check("a missing checkout: no revision, and the reason is kept",
      (record["rev"], bool(record["git_error"])), (None, True))
paths.OCPQ_CHECKOUT = real_checkout

print("\n--- the merge compares evaluators across shards ---")
# Imported late: eval_testset pulls in the whole pipeline. Everything above runs without it.
from eval_testset import _PIPELINE_SOURCE_FILES, _merge_shards  # noqa: E402

check("QueryExecutor.py, which records all this, is in the pipeline fingerprint",
      "QueryExecutor.py" in _PIPELINE_SOURCE_FILES, True)
CFG = "m|s|True"


def rec(nl):
    return {"case_id": 1, "nl_query": nl, "ocel_path": "a.json",
            "domain_seen": True, "discriminating": True, "baseline_f1": 0.4,
            "gold_query": {"nodes": []}, "prediction": {"nodes": []},
            "model": "m", "model_label": "m", "prompt_strategy": "schema-grammar-guided",
            "schema_enforced": True, "schema_validation": True, "repeat": 0,
            "config_id": CFG, "status": "success", "pipeline_status": "ok",
            "similarity": {"combined": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                           "events": {"jaccard": 0.5, "precision": 0.5, "recall": 0.5},
                           "objects": {"jaccard": 0.5, "precision": 0.5, "recall": 0.5}},
            "llm": {}, "execution": {"success": True, "duration_sec": 1.0},
            "total_duration_sec": 1.0,
            "metrics": {"schema_valid": True, "hallucinated_entities": False,
                        "unsupported_filters": False,
                        "structural": {"f1": 0.5, "precision": 0.5, "recall": 0.5},
                        "bindings_skipped": False, "gold_bindings_skipped": False}}


def shard(name, cli_sha, changed=False):
    d = tmp / "shards" / name
    d.mkdir(parents=True)
    (d / "predictions.json").write_text(json.dumps([rec(f"q {name}")]), encoding="utf-8")
    manifest = {"pipeline_fingerprint": "same", "repeats": 1}
    if cli_sha is not None:
        manifest["ocpq"] = {"rev": "r", "binaries": {"ocpq_cli": {"sha256": cli_sha}}}
    (d / "config_resolved.json").write_text(json.dumps(manifest), encoding="utf-8")
    (d / "summary.json").write_text(json.dumps({"ocpq_cli_changed_during_run": changed}),
                                    encoding="utf-8")
    return str(d)


def merged(name, dirs):
    out = tmp / "merged" / name
    _merge_shards(dirs, out)
    return json.loads((out / "summary.json").read_text(encoding="utf-8"))


s = merged("same", [shard("s0", "a" * 64), shard("s1", "a" * 64)])
check("every shard ran the same binary -> consistent",
      (s["ocpq_consistent"], list(s["ocpq_cli_sha256s"])), (True, ["a" * 64]))
s = merged("differ", [shard("d0", "a" * 64), shard("d1", "b" * 64)])
check("two binaries -> inconsistent, both named",
      (s["ocpq_consistent"], sorted(s["ocpq_cli_sha256s"])), (False, ["a" * 64, "b" * 64]))
s = merged("changed", [shard("c0", "a" * 64), shard("c1", "a" * 64, changed=True)])
check("one binary at every start, but it changed under a shard -> inconsistent",
      s["ocpq_consistent"], False)
s = merged("old", [shard("o0", "a" * 64), shard("o1", None)])
check("a shard from before 2026-09-22 -> cannot be established (None)",
      (s["ocpq_consistent"], s["ocpq_unrecorded_shards"]), (None, 1))

shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> every run names its evaluator, and a merge cannot hide a mix of them")
