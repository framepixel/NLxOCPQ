import os
import subprocess
import shutil
import glob
import json
import tempfile
import threading
import time
import signal
from urllib import request, error

import paths


# A pathological OCPQ query (e.g. an unbounded Cartesian product from a missing
# O2E filter) can exhaust system RAM and trigger a SLURM OOM-kill of the whole
# job, or run effectively forever. Cap the CLI subprocess so such a query is
# killed and surfaces as a counted execution failure instead. Both limits are
# overridable via environment variables (no magic numbers in the logic).
_OCPQ_CLI_TIMEOUT_SEC = int(os.getenv("OCPQ_CLI_TIMEOUT_SEC", "300"))
_OCPQ_CLI_MEM_LIMIT_GB = float(os.getenv("OCPQ_CLI_MEM_LIMIT_GB", "24"))

# Where each CLI invocation gets its private working directory. The Rust CLI writes its
# result to `ocpq-res-export-<UTC-second>.json` *relative to its cwd* and exposes no output
# flag (`EvaluateArgs` in OCPQ/backend/ocpq_cli/src/main.rs), so the working directory is the
# only handle we have on where the file lands. Point this at node-local scratch on SLURM.
_OCPQ_CLI_SCRATCH_DIR = os.getenv("OCPQ_CLI_SCRATCH_DIR") or tempfile.gettempdir()

_RESULT_EXPORT_GLOB = "ocpq-res-export-*.json"
_RUN_DIR_PREFIX = "ocpq-run-"


def _limit_child_memory():
    """Cap the child process's address space (Linux only).

    Runs in the forked child before exec. When the Rust CLI exceeds the cap its
    allocations fail and it aborts with a non-zero exit code, which the caller
    handles as an ordinary execution failure rather than letting the runaway
    query take down the entire SLURM allocation.
    """
    import resource

    limit_bytes = int(_OCPQ_CLI_MEM_LIMIT_GB * (1024 ** 3))
    resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))


def _find_cargo_command():
    cargo_candidates = []

    if os.name == "nt":
        home_dir = os.path.expanduser("~")
        cargo_candidates.extend([
            os.path.join(home_dir, ".cargo", "bin", "cargo.exe"),
            os.path.join(home_dir, ".cargo", "bin", "cargo.bat"),
            os.path.join(home_dir, ".cargo", "bin", "cargo.cmd"),
        ])
    else:
        cargo_candidates.extend([
            os.path.expanduser("~/.cargo/bin/cargo"),
            os.path.expanduser("~/.cargo/bin/cargo.exe"),
            "/usr/bin/cargo",
            "/usr/local/bin/cargo",
        ])

    cargo_candidates.append(shutil.which("cargo"))

    cargo_cmd = next(
        (candidate for candidate in cargo_candidates if candidate and os.path.exists(candidate)),
        None,
    )

    if cargo_cmd:
        return cargo_cmd

    searched = "\n- ".join(candidate for candidate in cargo_candidates if candidate)
    raise FileNotFoundError(
        "Cargo was not found. Install Rust via rustup or add cargo to PATH. "
        "Searched locations:\n- " + searched
    )


# Cached OCPQ CLI argument style. Older builds accept `--ocel/--bbox-tree`
# directly; newer builds nest the same flags under an `evaluate` subcommand.
# Detected once per process from the binary's --help output.
_CLI_MODE = None  # None=unknown, "subcommand", or "bare"
_CLI_MODE_LOCK = threading.Lock()


# Upper bound on captured CLI output, in characters. ocpq_cli evaluates bindings across rayon
# worker threads, and a malformed predicted query makes every one of them panic separately: a
# single bad query in the 2026-08-15 smoke test produced 46,016 identical
# "index out of bounds" lines, 6.8 MB of log across only 144 runs. The same text is stored in
# `query_stdout` on every record, so an uncapped dump inflates predictions.json too. Head and
# tail are both kept because the first lines identify the panic and the last usually carry the
# CLI's own summary.
_MAX_CLI_OUTPUT_CHARS = int(os.getenv("OCPQ_CLI_OUTPUT_CHAR_CAP", "8000"))


def _clip_cli_output(text):
    """Bound a CLI stdout/stderr dump, stating explicitly how much was dropped."""
    if not text or len(text) <= _MAX_CLI_OUTPUT_CHARS:
        return text
    head = _MAX_CLI_OUTPUT_CHARS * 2 // 3
    tail = _MAX_CLI_OUTPUT_CHARS - head
    omitted = len(text) - head - tail
    omitted_lines = text.count("\n", head, len(text) - tail)
    return (
        f"{text[:head]}\n"
        f"[... {omitted} characters / ~{omitted_lines} lines omitted "
        f"(cap {_MAX_CLI_OUTPUT_CHARS}); ocpq_cli panics once per worker thread ...]\n"
        f"{text[-tail:]}"
    )


def _cargo_invocation(cargo_cmd, cli_dir):
    """Build the cargo prefix that runs `ocpq_cli` regardless of the working directory.

    The CLI is launched from a private scratch directory (see `run_ocpq_query`), which sits
    outside the cargo workspace, so cargo can no longer discover the manifest by walking up
    from its cwd. `--manifest-path` pins it explicitly.
    """
    return [
        cargo_cmd, "run", "-q", "--release",
        "--manifest-path", os.path.join(cli_dir, "Cargo.toml"),
        "-p", "ocpq_cli", "--",
    ]


def _detect_cli_mode(cargo_cmd, cli_dir):
    """Return 'subcommand' if the CLI exposes an `evaluate` subcommand, else 'bare'.

    Detection is cached and guarded by a lock. The probe takes seconds, so publishing the
    `bare` default *before* it finished let a concurrent caller read the placeholder and
    build legacy-style arguments against a modern CLI, which fails with
    `unexpected argument '--ocel'`. The result is only assigned once it is final.
    """
    global _CLI_MODE
    if _CLI_MODE is not None:
        return _CLI_MODE
    with _CLI_MODE_LOCK:
        if _CLI_MODE is not None:  # another caller finished while we waited
            return _CLI_MODE
        detected = "bare"  # safe default = legacy interface used on the cluster
        try:
            probe = subprocess.run(
                _cargo_invocation(cargo_cmd, cli_dir) + ["--help"],
                cwd=cli_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=600,
            )
            help_text = (probe.stdout or "") + (probe.stderr or "")
            if "evaluate" in help_text and "Commands:" in help_text:
                detected = "subcommand"
        except Exception:
            pass  # keep the legacy default if probing fails
        _CLI_MODE = detected
    return _CLI_MODE


def _build_cli_args(cli_mode, ocel_abs, query_abs):
    """Build the argument list after the cargo `--` separator for the given mode."""
    flags = ["--ocel", ocel_abs, "--bbox-tree", query_abs]
    return (["evaluate"] + flags) if cli_mode == "subcommand" else flags


class _AmbiguousResultError(RuntimeError):
    """More than one result file appeared for a single CLI invocation.

    Impossible by construction once the working directory is private, so it signals a broken
    assumption rather than a query failure. Never resolved by guessing, because guessing is
    exactly what made the shared-directory version unsafe.
    """


def _collect_exported_result(run_dir):
    """Return the single result file the CLI wrote into its private run directory.

    Isolating the working directory is what makes this unambiguous. The previous
    implementation globbed the *shared* CLI directory and took the newest file by mtime,
    which under `submit_eval_array.sh` (24 concurrent tasks, one `SLURM_SUBMIT_DIR`) could
    hand one task another task's results — silently, and straight into `combined_f1`.
    """
    exports = glob.glob(os.path.join(run_dir, _RESULT_EXPORT_GLOB))
    if not exports:
        return None
    if len(exports) > 1:
        raise _AmbiguousResultError(
            f"OCPQ CLI produced {len(exports)} result files in {run_dir}; expected exactly one."
        )
    return exports[0]


def discard_result_file(result_path):
    """Delete a CLI result file and the private run directory created for it.

    Safe to call with None or an already-deleted path. Only removes the parent directory
    when it is one of ours, so a caller that moved the result elsewhere cannot lose data.
    """
    if not result_path:
        return
    parent = os.path.dirname(os.path.abspath(result_path))
    if os.path.basename(parent).startswith(_RUN_DIR_PREFIX):
        shutil.rmtree(parent, ignore_errors=True)
        return
    try:
        os.remove(result_path)
    except OSError:
        pass


def _terminate_process(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except Exception:
        process.terminate()

def run_ocpq_query(ocel_path, query_result_file):
    """
    Executes the generated OCPQ query using the Rust CLI backend.
    Output is suppressed except for the final query result.

    Each invocation runs in its own scratch directory, because the CLI writes its result
    relative to the working directory and offers no output flag. That makes the returned
    path unambiguous even when many evaluations run concurrently against one checkout.

    Returns:
        A `(stdout, stderr, result_path)` triple. `result_path` is None when the query did
        not produce a result, which callers treat as an execution failure. On success the
        caller owns the returned file and should pass it to `discard_result_file` once the
        result has been read.
    """
    # Relative arguments are repo-root-relative, not relative to this module: callers pass
    # what the testset stores (`..\Logs\x.json`, anchored at the repo root) or what a user
    # typed at the repo root. Anchoring on `__file__` would silently resolve them inside
    # `src/` and every query would fail to find its log.
    ocel_abs = os.path.abspath(os.path.join(paths.REPO_ROOT, ocel_path))
    query_abs = os.path.abspath(os.path.join(paths.REPO_ROOT, query_result_file))

    # Path to the Rust CLI cargo project
    cli_dir = os.path.join(paths.OCPQ_CHECKOUT, "backend", "ocpq_cli")

    if not os.path.exists(cli_dir):
        warning = f"warning: OCPQ CLI directory not found at {cli_dir}"
        print(warning)
        return warning, warning, None
        
    cargo_cmd = _find_cargo_command()

    # The -q (quiet) flag restricts cargo build output, only showing program output.
    # Older OCPQ builds accept `--ocel/--bbox-tree` directly; newer builds nest them
    # under an `evaluate` subcommand. Detect the style once and adapt, so the same
    # code works against both the cluster (legacy) and an updated local checkout.
    cli_mode = _detect_cli_mode(cargo_cmd, cli_dir)
    cmd = [
        *_cargo_invocation(cargo_cmd, cli_dir),
        *_build_cli_args(cli_mode, ocel_abs, query_abs),
    ]

    # Private working directory: the CLI's output filename is fixed and only second-resolution,
    # so a shared cwd lets concurrent runs collide and lets one run pick up another's result.
    os.makedirs(_OCPQ_CLI_SCRATCH_DIR, exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix=_RUN_DIR_PREFIX, dir=_OCPQ_CLI_SCRATCH_DIR)

    print(f"\nExecuting OCPQ CLI query locally...")

    process = None
    try:
        # start_new_session=True creates a new process group so we can cleanly kill BOTH cargo and the spawned Rust executable if Streamlit stops
        process = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            # Memory cap is POSIX-only (no fork/preexec_fn on Windows).
            preexec_fn=_limit_child_memory if os.name != "nt" else None,
        )

        # Wait for the process to finish, bounding the wall time so a runaway
        # query cannot hang the shard indefinitely.
        stdout, stderr = process.communicate(timeout=_OCPQ_CLI_TIMEOUT_SEC)
        
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, cmd, output=stdout, stderr=stderr
            )
            
        # Clipped before printing *and* before returning, so the cap protects the stored record
        # as much as the console.
        stdout = _clip_cli_output(stdout.strip() if stdout else stdout)
        print("--- Query Execution Result ---")
        print(stdout)
        print("-----------------------------\n")

        result_file = _collect_exported_result(run_dir)
        if result_file is None:
            shutil.rmtree(run_dir, ignore_errors=True)

        # On successful execution (exit code 0), we ignore `res.stderr` (e.g. rust/cargo warnings)
        return stdout, None, result_file

    except BaseException as e:
        # If the user presses "Stop" in Streamlit, it raises a custom ScriptRunner exception.
        # This BaseException catch perfectly hooks into that flow so we can kill the underlying process.
        if process and process.poll() is None:
            print("\nExecution interrupted (e.g. Streamlit Stop)! Force-killing the sub-processes...")
            try:
                _terminate_process(process)
            except Exception as kill_err:
                print(f"Failed to kill process: {kill_err}")

        # No result reaches the caller on any of these paths, so the scratch directory is ours
        # to remove; leaving it behind would recreate the disk growth this replaced. Done after
        # terminating the child, which still has this directory as its cwd. The one exception
        # is a multiple-export collision, which should be impossible — keep the directory so
        # whoever investigates can still see what the CLI actually wrote.
        if not isinstance(e, _AmbiguousResultError):
            shutil.rmtree(run_dir, ignore_errors=True)

        # A runaway query that hit the wall-time cap: kill it and report as a
        # normal execution failure so the shard keeps going.
        if isinstance(e, subprocess.TimeoutExpired):
            msg = (
                f"OCPQ query exceeded the {_OCPQ_CLI_TIMEOUT_SEC}s time limit "
                "and was terminated (likely an unbounded Cartesian search)."
            )
            print(f"--- Query Execution Timed Out ---\n{msg}\n-----------------------------\n")
            return "", msg, None

        # If it was a standard Cargo failure (e.g., syntax crash), parse it out as normal
        if isinstance(e, subprocess.CalledProcessError):
            print("--- Query Execution Failed ---")
            
            stdout_dump = e.stdout.strip() if e.stdout else ""
            stderr_dump = e.stderr.strip() if e.stderr else ""

            # Filter stderr to only include the panic message if it exists. Note this matches
            # only main-thread panics; the high-volume ones come from rayon workers
            # ("thread '<unnamed>'"), which is why the clip below is the real bound.
            if "thread 'main'" in stderr_dump:
                stderr_dump = stderr_dump[stderr_dump.find("thread 'main'"):]

            stdout_dump = _clip_cli_output(stdout_dump)
            stderr_dump = _clip_cli_output(stderr_dump)


            if stdout_dump:
                print("Stdout Dump:\n", stdout_dump)
            if stderr_dump:
                print("Stderr Dump:\n", stderr_dump)
            print("-----------------------------\n")
            return stdout_dump, stderr_dump, None
        else:
            # Re-raise Streamlit's StopException / KeyboardInterrupt so the UI actually stops
            raise e


def _http_post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def _http_post_bytes(url, payload_bytes):
    req = request.Request(url, data=payload_bytes, method="POST")
    with request.urlopen(req, timeout=120) as resp:
        return resp.status, resp.read()


def _http_get_status(url):
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=5) as resp:
        return resp.status


def _is_backend_alive(base_url):
    try:
        status = _http_get_status(base_url.rstrip("/") + "/ocel/info")
        return 200 <= status < 500
    except Exception:
        return False


def export_query_result_table(
    ocel_path,
    query_result_file,
    node_index=0,
    output_path=None,
    export_format="CSV",
    include_violation_status=True,
    include_ids=True,
    omit_header=False,
    labels=None,
    backend_url="http://127.0.0.1:3000",
    auto_start_backend=True,
):
    """
    Export OCPQ query situations as CSV/XLSX using OCPQ's built-in web-server endpoint.

    This uses the exact same backend flow as the UI:
    1) Upload OCEL to web-server state
    2) Evaluate bbox tree via /ocel/check-constraints-box
    3) Export one node's bindings via /ocel/export-bindings
    """
    labels = labels or []
    export_format = export_format.upper()
    if export_format not in {"CSV", "XLSX"}:
        raise ValueError("export_format must be 'CSV' or 'XLSX'")

    web_server_dir = os.path.join(paths.OCPQ_CHECKOUT, "backend", "web-server")

    # Repo-root-relative, for the same reason as in `run_ocpq_query` above.
    ocel_abs = os.path.abspath(os.path.join(paths.REPO_ROOT, ocel_path))
    query_abs = os.path.abspath(os.path.join(paths.REPO_ROOT, query_result_file))

    if not os.path.exists(ocel_abs):
        raise FileNotFoundError(f"OCEL file not found: {ocel_abs}")
    if not os.path.exists(query_abs):
        raise FileNotFoundError(f"Query tree file not found: {query_abs}")

    if output_path is None:
        ext = "csv" if export_format == "CSV" else "xlsx"
        output_path = os.path.join(
            script_dir,
            f"ocpq-bindings-node-{node_index}.{ext}",
        )

    started_backend = False
    backend_proc = None
    try:
        if auto_start_backend and not _is_backend_alive(backend_url):
            if not os.path.exists(web_server_dir):
                raise FileNotFoundError(
                    f"OCPQ web-server directory not found at {web_server_dir}"
                )

            cargo_cmd = shutil.which("cargo")
            if not cargo_cmd:
                cargo_cmd_windows = os.path.expanduser("~\\.cargo\\bin\\cargo.exe")
                cargo_cmd_unix = os.path.expanduser("~/.cargo/bin/cargo")
                if os.path.exists(cargo_cmd_windows):
                    cargo_cmd = cargo_cmd_windows
                elif os.path.exists(cargo_cmd_unix):
                    cargo_cmd = cargo_cmd_unix
                else:
                    cargo_cmd = "cargo"

            cmd = [cargo_cmd, "run", "-q", "--release"]
            backend_proc = subprocess.Popen(
                cmd,
                cwd=web_server_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            started_backend = True

            for _ in range(200):
                if _is_backend_alive(backend_url):
                    break
                time.sleep(0.25)
            else:
                raise RuntimeError(
                    f"Backend did not become ready at {backend_url} within timeout"
                )

        # 1) Upload OCEL to backend state
        if ocel_abs.endswith(".json"):
            upload_endpoint = "/ocel/upload-json"
        elif ocel_abs.endswith(".xml"):
            upload_endpoint = "/ocel/upload-xml"
        elif ocel_abs.endswith(".sqlite"):
            upload_endpoint = "/ocel/upload-sqlite"
        elif ocel_abs.endswith(".xes.gz"):
            upload_endpoint = "/ocel/upload-xes-conversion/.xes.gz"
        elif ocel_abs.endswith(".xes"):
            upload_endpoint = "/ocel/upload-xes-conversion/.xes"
        else:
            raise ValueError(
                "Unsupported OCEL file type. Use .json, .xml, .sqlite, .xes, or .xes.gz"
            )

        with open(ocel_abs, "rb") as f:
            payload_bytes = f.read()
        _http_post_bytes(backend_url.rstrip("/") + upload_endpoint, payload_bytes)

        # 2) Evaluate bbox tree (fills eval_res in backend state)
        with open(query_abs, "r", encoding="utf-8") as f:
            tree_json = json.load(f)
        _http_post_json(
            backend_url.rstrip("/") + "/ocel/check-constraints-box",
            {"tree": tree_json, "measurePerformance": False},
        )

        # 3) Export table for the selected node
        table_options = {
            "includeViolationStatus": include_violation_status,
            "includeIds": include_ids,
            "omitHeader": omit_header,
            "labels": labels,
            "format": export_format,
        }
        _status, table_bytes = _http_post_json(
            backend_url.rstrip("/") + "/ocel/export-bindings",
            [node_index, table_options],
        )

        with open(output_path, "wb") as f:
            f.write(table_bytes)
        return os.path.abspath(output_path)

    except error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        raise RuntimeError(
            f"HTTP error calling OCPQ backend: {e.code} {e.reason}. Response: {body}"
        ) from e
    finally:
        if started_backend and backend_proc is not None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=3)
            except Exception:
                backend_proc.kill()
