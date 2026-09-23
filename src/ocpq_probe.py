#!/usr/bin/env python3
"""Log-inspection tools for the agentic arm: what an analyst could look up, and nothing else.

WHY THIS EXISTS
---------------
Every intervention in this project so far repaired a query *before* it ran -- constrained decoding
(notes 31), forced arity (notes 32), the structural check (notes 37), the non-oracle stack (notes 41)
-- and each one raised the execution rate while leaving the answer essentially unchanged. Notes 40
located the failure as compositional: the model writes a tree of the wrong shape over the right
entities. Nothing that inspects only the tree can see that.

This module is the missing channel. It lets a model ask the LOG questions -- run this candidate tree
and tell me how many situations each box selects, show me five of them, what values does this
attribute really take -- so a revision can be driven by what the query *selects* rather than by
whether it parses.

WHAT THIS MODULE MAY NOT DO
---------------------------
* **It never sees gold.** No function here takes a gold query, a gold result, a testset row or a
  case id. It must not import `eval_testset` or `result_store`; `tests/test_ocpq_probe.py` asserts
  that by scanning the source, because in a production setting the gold query does not exist and any
  signal derived from it is a capability the deployed system would not have.
* **It never scores.** It returns counts, rows and error strings. Comparison against gold lives in
  `eval_testset.py` alone (CLAUDE.md 2.5), which is also what scores the agent's final answer: the
  tree goes through `QueryExecutor.run_ocpq_query` like every other arm, so this server is a PROBE
  surface and never an evaluation path (CLAUDE.md 4.4).

WHY A SERVER, AND WHY WE OWN IT
-------------------------------
`ocpq_cli` re-imports the log on every invocation -- 23-35 s against ~30 ms of evaluation -- so a
15-probe case would spend eight minutes importing. The OCPQ web server imports once and answers in
milliseconds. It holds ONE global OCEL with no sessions, though, so two servers sharing a port would
silently evaluate each other's log. Three guards, in order of how much they matter:

  1. the server is started BY THIS PROCESS on its own port and killed with it -- a server we did not
     start is never adopted, however healthy it looks;
  2. every load is followed by an identity assertion against `LogEvents.extract_ocel_schema`, so
     probing the wrong log is a loud failure rather than a plausible wrong answer;
  3. the log directory is a private directory of symlinks, so no job mutates the pinned corpus.

Requires the local OCPQ modification of 2026-09-20 (`OCPQ_WEB_PORT`, `OCPQ_WEB_SKIP_DEFAULT_OCEL`);
see `Thesis/notes/11-evaluation-readiness.md` 16 for the first such modification.

USAGE
    from ocpq_probe import ProbeServer, OcpqProbe
    with ProbeServer() as server:                       # owns the process, kills it on exit
        probe = OcpqProbe(server.base_url, max_tool_calls=12)
        probe.ensure_log("../Logs/order-management.json")
        print(probe.call("try_query", {"tree": tree}))

    python src/ocpq_probe.py --smoke ../Logs/order-management.json   # start, load, probe, stop
"""
import argparse
import atexit
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from LogEvents import extract_ocel_schema  # noqa: E402
from ocpq_schema import defect_kind, structural_defects  # noqa: E402
from QueryExecutor import _find_cargo_command, _terminate_process  # noqa: E402

# Defaults live here; every caller in the evaluation passes them from configs/eval_config.agentic.json
# so they land in the config fingerprint (CLAUDE.md 4.2).
DEFAULT_REQUEST_TIMEOUT_SEC = float(os.getenv("OCPQ_PROBE_REQUEST_TIMEOUT_SEC", "30"))
# A load is an OCEL import: 23-35 s for the big logs, and bpic2017 is the outlier.
DEFAULT_LOAD_TIMEOUT_SEC = float(os.getenv("OCPQ_PROBE_LOAD_TIMEOUT_SEC", "300"))
DEFAULT_STARTUP_TIMEOUT_SEC = float(os.getenv("OCPQ_PROBE_STARTUP_TIMEOUT_SEC", "180"))
# Tool results are read by the model. The harness clamps any single message at 20 000 chars
# (`LLM._clamp_message_history`), which would cut a result mid-JSON and hand the model something it
# reads as content; cap here instead, with an explicit marker saying what was dropped.
DEFAULT_RESULT_CHAR_CAP = int(os.getenv("OCPQ_PROBE_RESULT_CHAR_CAP", "4000"))
# Bounded self-repair of the server (see `OcpqProbe._recover`). Past either cap the probe raises
# `ProbeUnavailable`, because an arm that keeps restarting its tool surface is no longer measuring
# what a working tool surface gives the model.
DEFAULT_MAX_RESTARTS_PER_CASE = int(os.getenv("OCPQ_PROBE_MAX_RESTARTS_PER_CASE", "2"))
DEFAULT_MAX_RESTARTS_PER_RUN = int(os.getenv("OCPQ_PROBE_MAX_RESTARTS_PER_RUN", "10"))

# Defect kinds `try_query` refuses to send, because the evaluator PANICS on them rather than
# answering. MEASURED 2026-09-22 on the pinned server (nlxocpq-pinned 50f60c1) and on the ocpq_cli
# binary of 2026-08-28, one tree per kind:
#   child_out_of_range -- panics (structs.rs:784, index out of bounds). The server drops the
#                         connection but SURVIVES and keeps answering correctly; ocpq_cli exits 101.
#   unbound_variable   -- 400 "Could not get oN": an error document, passed through.
#   malformed_tree     -- 422 deserialisation error naming the field: passed through.
#   cycle, multiple_parents -- EVALUATED by both the server and ocpq_cli (exit 0, an `Ok` export),
#                         although `ocpq_schema.BLOCKING_DEFECT_KINDS` lists them. Passed through.
# Refusing more than the crash class would withhold counts for trees the scorer does evaluate.
PROBE_REFUSED_DEFECT_KINDS = frozenset({"child_out_of_range"})


class ProbeError(RuntimeError):
    """Base class: something went wrong in the probe layer itself, not in the model's query."""


class ProbeUnavailable(ProbeError):
    """The server is unreachable, did not start, or refused to load the log.

    Distinct from a failed query: a failed query is DATA the agent should see and learn from, while
    this means the tool surface is broken and the case cannot be measured as an agentic case.
    """


class ProbeBudgetExhausted(ProbeError):
    """The per-case tool-call budget is spent. The caller finalises with what it has."""


class LogIdentityError(ProbeError):
    """The loaded log is not the log that was asked for. Always fatal -- see guard 2 above."""


def _free_port():
    """An unused TCP port, chosen by the OS.

    There is a race between closing this socket and the server binding it. It is preferred over
    hashing a job id anyway: a collision here fails loudly at startup, whereas a fixed port silently
    hands us another job's server.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _http_json(url, payload=None, timeout=DEFAULT_REQUEST_TIMEOUT_SEC):
    """POST (or GET when `payload` is None) JSON and return `(status, parsed_body_or_text)`.

    Only the transport exceptions are caught, and they are re-raised as `ProbeUnavailable`. A bare
    `except Exception` here would swallow `eval_testset.CaseTimeout` -- SIGALRM fires inside the
    blocking socket read -- and the per-case cap would be silently inert for this arm alone, which
    CLAUDE.md 6 forbids ("the cap must be applied to every arm").
    """
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = response.status
    except urllib.error.HTTPError as exc:                      # a 4xx/5xx carries a usable message
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        status = exc.code
    except (urllib.error.URLError, OSError) as exc:
        raise ProbeUnavailable(f"probe request to {url} failed: {exc}") from exc
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body


class ProbeServer:
    """An OCPQ web server owned by this process: its own port, its own log directory, its own life.

    The working directory matters. The server resolves its log directory as `../data/` RELATIVE TO
    ITS CWD (`OCPQ/backend/web-server/src/load_ocel/mod.rs`), so running it inside a private
    directory gives per-job isolation for free -- no writing into the shared OCPQ checkout, and the
    pinned corpus (`data/corpus_manifest.json`) stays untouched because the log directory holds
    symlinks only.
    """

    def __init__(self, workdir=None, port=None, logs_dir=None, startup_timeout_sec=None):
        self.port = int(port) if port else _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.startup_timeout_sec = float(startup_timeout_sec or DEFAULT_STARTUP_TIMEOUT_SEC)
        self.logs_dir = Path(logs_dir or paths.OCEL_LOGS)
        root = Path(workdir) if workdir else Path(
            tempfile.mkdtemp(prefix="ocpq-probe-",
                             dir=os.getenv("OCPQ_CLI_SCRATCH_DIR") or tempfile.gettempdir()))
        self.workdir = root
        self.run_dir = root / "run"
        self.data_dir = root / "data"
        self.process = None
        self.log_path = root / "server.log"
        self._log_handle = None
        self._atexit_registered = False
        self.restarts = 0

    def link_log(self, ocel_path):
        """Expose one log to the server under its basename, and return that name.

        Symlinks, never copies: the corpus is 2.75 GB and pinned by sha256. On Windows a symlink
        needs Developer Mode or an elevated shell, so the failure says so rather than silently
        copying gigabytes.
        """
        source = Path(ocel_path).resolve()
        if not source.is_file():
            raise ProbeUnavailable(f"log not found: {source}")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        link = self.data_dir / source.name
        if link.exists() or link.is_symlink():
            return source.name
        try:
            os.symlink(source, link)
            return source.name
        except (OSError, NotImplementedError) as symlink_error:
            # Windows refuses symlinks without Developer Mode (WinError 1314). A hard link needs no
            # privilege on the same NTFS volume and is equally copy-free, which matters: the corpus
            # is 2.75 GB and pinned by sha256, so copying it would be both slow and a second copy
            # to keep honest. Linux takes the symlink path and never reaches here.
            try:
                os.link(source, link)
                return source.name
            except (OSError, NotImplementedError) as link_error:
                raise ProbeUnavailable(
                    f"could not link {source} into {self.data_dir}: symlink failed "
                    f"({symlink_error}); hard link failed ({link_error}). On Windows, enable "
                    "Developer Mode, or put the probe workdir on the same volume as the logs."
                ) from link_error

    def binary_path(self):
        """Where a release build of the server lands, if one exists."""
        name = "ocpq-web-server.exe" if os.name == "nt" else "ocpq-web-server"
        return paths.OCPQ_CHECKOUT / "target" / "release" / name

    def ensure_built(self, timeout_sec=None):
        """Build the server once, if it is not built yet. Returns the binary path.

        Building is separated from starting because the first release build of the axum stack takes
        many minutes, and a build that slow inside the readiness wait is indistinguishable from a
        server that will never answer -- which is exactly how the first probe smoke failed
        (2026-09-20). Afterwards the binary is spawned directly: no cargo in the hot path, and no
        rebuild check per job.
        """
        binary = self.binary_path()
        if binary.is_file():
            return binary
        manifest = paths.OCPQ_CHECKOUT / "backend" / "web-server" / "Cargo.toml"
        if not manifest.is_file():
            raise ProbeUnavailable(f"OCPQ web-server crate not found at {manifest}")
        print(f"building ocpq-web-server (first time; this takes several minutes) -> {binary}")
        build = subprocess.run(
            [_find_cargo_command(), "build", "--release",
             "--manifest-path", str(manifest), "-p", "ocpq-web-server"],
            capture_output=True, text=True,
            timeout=float(timeout_sec or os.getenv("OCPQ_PROBE_BUILD_TIMEOUT_SEC", "3600")))
        if build.returncode != 0 or not binary.is_file():
            raise ProbeUnavailable(
                f"could not build the OCPQ web server: {(build.stderr or '')[-800:]}")
        return binary

    def start(self):
        """Spawn the server and wait until it answers. Idempotent."""
        if self.process is not None:
            return self
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        command = [str(self.ensure_built())]
        environment = dict(os.environ)
        environment["OCPQ_WEB_PORT"] = str(self.port)
        # Without this the server imports order-management.json at startup -- 25-35 s and several GB
        # for a log we immediately replace. Requires the 2026-09-20 OCPQ modification.
        environment["OCPQ_WEB_SKIP_DEFAULT_OCEL"] = "1"
        # Appended, not truncated: after a restart the previous process's panic trace is the
        # evidence for why it was restarted.
        self._log_handle = open(self.log_path, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            command, cwd=str(self.run_dir), env=environment,
            stdout=self._log_handle, stderr=subprocess.STDOUT,
            start_new_session=True if os.name != "nt" else False)
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        self._wait_until_ready()
        return self

    def is_alive(self):
        """True while the server process we started is still running."""
        return self.process is not None and self.process.poll() is None

    def restart(self):
        """Kill the server and start a fresh one ON A NEW PORT. The caller must reload the log.

        A new port rather than the old one: on Windows a just-closed listening port is not reliably
        rebindable at once, and a restart that fails to bind would look like a second crash. The new
        process holds NO log -- `OcpqProbe._recover` reloads it and re-asserts its identity.
        """
        self.stop()
        self.port = _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.restarts += 1
        return self.start()

    def _wait_until_ready(self):
        deadline = time.time() + self.startup_timeout_sec
        last_error = None
        while time.time() < deadline:
            if self.process.poll() is not None:
                raise ProbeUnavailable(
                    f"OCPQ web server exited with code {self.process.returncode} during startup; "
                    f"see {self.log_path}")
            try:
                status, _ = _http_json(f"{self.base_url}/ocel/info", timeout=5)
                if status in (200, 400, 404):        # answering at all is what we are waiting for
                    return
            except ProbeUnavailable as exc:
                last_error = exc
            time.sleep(1.0)
        self.stop()
        raise ProbeUnavailable(
            f"OCPQ web server did not answer on {self.base_url} within "
            f"{self.startup_timeout_sec:.0f}s ({last_error}); see {self.log_path}")

    def stop(self):
        """Kill the server. Safe to call twice; registered with `atexit` so a crash still cleans up."""
        if self.process is not None and self.process.poll() is None:
            _terminate_process(self.process)
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class OcpqProbe:
    """The tool surface the agent sees: five questions about the log, and a call budget.

    Every tool returns a JSON-serialisable dict. Failures of the MODEL's query (an invalid tree, an
    undeclared variable) come back as `{"error": ...}` -- that is the signal the agent is meant to
    act on. Failures of the PROBE (a dead server, the wrong log) raise, because they mean the case
    cannot be measured.
    """

    def __init__(self, base_url, max_tool_calls, server=None, result_char_cap=None,
                 request_timeout_sec=None, load_timeout_sec=None,
                 max_restarts_per_case=None, max_restarts_per_run=None):
        self.base_url = base_url.rstrip("/")
        self.max_tool_calls = int(max_tool_calls)
        self.server = server
        self.result_char_cap = int(result_char_cap or DEFAULT_RESULT_CHAR_CAP)
        self.request_timeout_sec = float(request_timeout_sec or DEFAULT_REQUEST_TIMEOUT_SEC)
        self.load_timeout_sec = float(load_timeout_sec or DEFAULT_LOAD_TIMEOUT_SEC)
        self.max_restarts_per_case = int(
            DEFAULT_MAX_RESTARTS_PER_CASE if max_restarts_per_case is None
            else max_restarts_per_case)
        self.max_restarts_per_run = int(
            DEFAULT_MAX_RESTARTS_PER_RUN if max_restarts_per_run is None
            else max_restarts_per_run)
        self.loaded_ocel_path = None
        self.eval_version = None
        # The loaded log's type names, from the server's own /ocel/load answer. They are what the
        # typed lookup binds and what an unknown-type error lists.
        self.object_types = []
        self.event_types = []
        # The agent's last evaluated tree, and whether the server still holds ITS evaluation: the
        # typed lookup in `_scan_entities` evaluates a one-box query of its own, which replaces it.
        self._agent_tree = None
        self._agent_eval_current = False
        self.tool_calls = 0
        self.trajectory = []
        # Health counters. Per case: reset by `reset()` and meant for the run record, beside
        # agent_tool_calls -- a silent restart is how a degraded arm would hide. Per run: the cap.
        self.restarts_case = 0
        self.restarts_run = 0
        self.evaluator_panics = 0
        self.refused_trees = 0
        self._recovery_note = None

    # ------------------------------------------------------------------ log loading and identity

    def ensure_log(self, ocel_path, force=False):
        """Load `ocel_path` into the server if it is not already loaded, and verify it.

        Called from the case preamble, OUTSIDE the per-case time limit: an import is ~30 s and it is
        not the case's own work. Loading is lazy because the evaluation's case loop is the outer
        loop, so consecutive cases usually share a log.
        """
        resolved = str(Path(ocel_path).resolve())
        if not force and self.loaded_ocel_path == resolved:
            return False
        if self.server is not None:
            name = self.server.link_log(resolved)
        else:
            name = Path(resolved).name
        status, body = _http_json(f"{self.base_url}/ocel/load", {"name": name},
                                  timeout=self.load_timeout_sec)
        if status != 200 or not isinstance(body, dict):
            raise ProbeUnavailable(f"loading {name} failed with status {status}: {str(body)[:300]}")
        self._assert_identity(resolved, body)
        self.loaded_ocel_path = resolved
        self.eval_version = None
        self._agent_tree, self._agent_eval_current = None, False
        self.object_types = sorted(_type_names(body.get("object_types") or body.get("objectTypes")
                                               or []))
        self.event_types = sorted(_type_names(body.get("event_types") or body.get("eventTypes")
                                              or []))
        return True

    def _assert_identity(self, ocel_path, info):
        """Fail loudly if the server loaded a different log than we asked for.

        The server keeps ONE global OCEL. If anything replaced it -- another process on the same
        port, a load that silently fell back -- every probe answer would be plausible and wrong.
        Comparing type names against the log on disk costs ~1 ms and converts that into a crash.
        """
        schema = extract_ocel_schema(ocel_path)
        for field, expected_key in (("event_types", "event_types"), ("object_types", "object_types")):
            served = _type_names(info.get(field) or info.get(_camel(field)) or [])
            expected = set(schema.get(expected_key) or [])
            if served != expected:
                missing = sorted(expected - served)[:5]
                extra = sorted(served - expected)[:5]
                raise LogIdentityError(
                    f"server's {field} do not match {Path(ocel_path).name}: "
                    f"missing {missing}, unexpected {extra}")

    # ------------------------------------------------------------------------------- the tools

    def call(self, name, args):
        """Run one tool by name, count it against the budget, and record it in the trajectory.

        Returns the tool's result dict. Raises `ProbeBudgetExhausted` when the budget is spent and
        `ProbeUnavailable` when the surface is broken; an unknown tool or bad arguments come back as
        `{"error": ...}` because those are the model's mistakes and it should see them.
        """
        if self.tool_calls >= self.max_tool_calls:
            raise ProbeBudgetExhausted(
                f"tool-call budget of {self.max_tool_calls} is spent")
        handler = _TOOL_HANDLERS.get(name)
        started = time.time()
        self._recovery_note = None
        if handler is None:
            result = {"error": f"unknown tool {name!r}; available: {sorted(_TOOL_HANDLERS)}"}
        else:
            self.tool_calls += 1
            try:
                result = self._run_handler(handler, name, dict(args or {}))
            except TypeError as exc:                       # wrong/missing argument names
                result = {"error": f"bad arguments for {name}: {exc}"}
        entry = {
            "tool": name,
            "args": _clip(json.dumps(args, default=str), 1000),
            "result": _clip(json.dumps(result, default=str), self.result_char_cap),
            "duration_sec": round(time.time() - started, 3),
        }
        if self._recovery_note:
            entry["recovery"] = self._recovery_note
        self.trajectory.append(entry)
        return result

    def _run_handler(self, handler, name, args):
        """Run one tool; if the server process died under it, restart, reload, retry ONCE.

        Only a DEAD process is retried here. `try_query` handles its own failures first, because a
        failure there is usually the model's tree, and resending a tree that broke the evaluator
        would just break it again. The other tools send no model-authored structure, so a dead
        server under them is our fault and one retry on a fresh server is the right answer.
        """
        try:
            return handler(self, args)
        except ProbeUnavailable as exc:
            if self.server is None or self.server.is_alive():
                raise
            self._recover(f"server process died during {name}: {exc}")
            return handler(self, args)

    def _recover(self, reason):
        """Restart the owned server, reload the case's log, re-assert its identity.

        Raises `ProbeUnavailable` when there is nothing to recover with (no owned server, or no log
        was loaded) or a restart cap is spent. Every restart is counted and noted on the trajectory
        entry of the call that caused it, so it is never silent.
        """
        if self.server is None:
            raise ProbeUnavailable(f"{reason}; no owned server to restart")
        if (self.restarts_case >= self.max_restarts_per_case
                or self.restarts_run >= self.max_restarts_per_run):
            raise ProbeUnavailable(
                f"{reason}; restart cap reached ({self.restarts_case}/"
                f"{self.max_restarts_per_case} this case, {self.restarts_run}/"
                f"{self.max_restarts_per_run} this run)")
        log_path = self.loaded_ocel_path
        self.restarts_case += 1
        self.restarts_run += 1
        started = time.time()
        self.server.restart()
        self.base_url = self.server.base_url.rstrip("/")
        self.loaded_ocel_path = None
        self.eval_version = None
        if log_path:
            self.ensure_log(log_path)
        self._recovery_note = {"restarted": True, "reason": _clip(reason, 300),
                               "restart_sec": round(time.time() - started, 1)}

    def health(self):
        """Per-case counters for the run record. Zero everywhere is the only healthy answer."""
        return {"probe_restarts": self.restarts_case,
                "probe_evaluator_panics": self.evaluator_panics,
                "probe_refused_trees": self.refused_trees}

    def try_query(self, args):
        """Evaluate a candidate tree and report how many situations each box selects.

        The per-node counts are the composition signal: they say which box collapsed the population
        to nothing, which is the failure notes 40 identified and which no pre-execution check sees.
        """
        tree = args.get("tree")
        if not isinstance(tree, dict):
            return {"error": "try_query needs a BindingBoxTree object as `tree`"}
        refused = [defect for defect in structural_defects(tree)
                   if defect_kind(defect) in PROBE_REFUSED_DEFECT_KINDS]
        if refused:
            # The evaluator panics on these instead of answering (see PROBE_REFUSED_DEFECT_KINDS),
            # so the agent gets the defect -- better feedback than a dropped connection.
            self.refused_trees += 1
            return {"error": "not evaluated: the tree is malformed, and the evaluator would crash "
                             "on it rather than report an error", "defects": refused}
        try:
            status, body = _http_json(f"{self.base_url}/ocel/check-constraints-box",
                                      {"tree": tree, "measurePerformance": False},
                                      timeout=self.request_timeout_sec)
        except ProbeUnavailable as exc:
            return self._try_query_failed(exc)
        if status == 404:
            raise ProbeUnavailable("no OCEL loaded on the probe server")
        if status != 200 or not isinstance(body, dict):
            # The evaluation refused the tree. This is the agent's problem to fix, not ours.
            return {"error": _clip(str(body), 800)}
        self.eval_version = body.get("evalVersion")
        self._agent_tree, self._agent_eval_current = tree, True
        nodes = [{"node": index,
                  "situations": summary.get("situationCount"),
                  "violations": summary.get("situationViolatedCount")}
                 for index, summary in enumerate(body.get("nodeSummaries") or [])]
        return {"nodes": nodes, "bindingsSkipped": body.get("bindingsSkipped"),
                "evalVersion": body.get("evalVersion")}

    def _try_query_failed(self, exc):
        """Turn a transport failure on `try_query` into feedback, recovering the server if needed.

        Three cases, told apart by the process state and the exception, never by guessing:
          * the request TIMED OUT -- the server is still computing a runaway evaluation, holding CPU
            and memory for an answer nobody will read. Restart it, then tell the agent.
          * the process is DEAD -- restart it, then tell the agent. Not retried: the tree is the
            likely cause, and resending it would kill the new process too.
          * the process is ALIVE and the connection dropped -- the evaluator panicked on this tree
            and the server survived it (measured 2026-09-22: three consecutive panics, correct
            answers after each). Nothing to restart; the tree is the problem, and the agent is told.
        Without an owned server there is nothing to diagnose with, so the failure propagates.
        """
        if self.server is None:
            raise exc
        if _is_timeout(exc):
            self._recover(f"try_query timed out after {self.request_timeout_sec:.0f}s")
            return {"error": f"the evaluation did not finish within "
                             f"{self.request_timeout_sec:.0f}s and was stopped; the query likely "
                             f"binds far too many combinations -- narrow it"}
        if not self.server.is_alive():
            self._recover(f"server process died during try_query: {exc}")
            return {"error": "the evaluator crashed on this tree and was restarted; the tree is "
                             "probably malformed -- check child indices, edgeNames and variables"}
        self.evaluator_panics += 1
        return {"error": "the evaluator crashed on this tree without returning an error; the tree "
                         "is probably malformed -- check child indices, edgeNames and variables"}

    def inspect_situations(self, args):
        """Show rows the LAST evaluated tree actually selected, for one node.

        `try_query` gives a count; this says what is in it, which is what distinguishes "my tree
        selects 400 of the wrong things" from "my tree selects 400 of the right things".
        """
        if self.eval_version is None:
            return {"error": "call try_query first; there is no evaluation to page through"}
        if not self._agent_eval_current and self._agent_tree is not None:
            # A `sample` / `attribute_values` lookup evaluated its own one-box query since, so the
            # server no longer holds the agent's evaluation. Re-run the agent's tree -- not charged,
            # not on the trajectory: the agent asked to page ITS result, and that is what it gets.
            status, body = _http_json(f"{self.base_url}/ocel/check-constraints-box",
                                      {"tree": self._agent_tree, "measurePerformance": False},
                                      timeout=self.request_timeout_sec)
            if status != 200 or not isinstance(body, dict):
                return {"error": "your last try_query could not be re-evaluated; call it again"}
            self.eval_version = body.get("evalVersion")
            self._agent_eval_current = True
        node = int(args.get("node", 0) or 0)
        limit = max(1, min(int(args.get("n", 5) or 5), 20))
        violated = args.get("violated")
        status, body = _http_json(
            f"{self.base_url}/ocel/eval-results/page",
            {"evalVersion": self.eval_version, "nodeIndex": node, "offset": 0,
             "limit": limit, "violated": violated},
            timeout=self.request_timeout_sec)
        if status == 410:
            self.eval_version = None
            return {"error": "that evaluation expired; call try_query again"}
        if status != 200 or not isinstance(body, dict):
            return {"error": _clip(str(body), 500)}
        return {"node": node, "rows": body.get("rows"), "filteredCount": body.get("filteredCount")}

    def sample(self, args):
        """A handful of real entities of one event or object type, with their attributes."""
        type_name = args.get("type")
        if not type_name:
            return {"error": "sample needs a `type`"}
        limit = max(1, min(int(args.get("n", 5) or 5), 20))
        kind, found = self._scan_entities(type_name, limit)
        if kind is None:
            return self._unknown_type(type_name)
        if found is None:
            return {"error": f"looking up {type_name!r} failed; try again, or bind it in try_query"}
        return {"type": type_name, "kind": kind,
                "entities": [{"id": entity.get("id"),
                              "attributes": entity.get("attributes")} for entity in found],
                "sampled_from": (f"the first {limit} {kind}s of this type in evaluation order, not "
                                 f"a uniform sample") if found
                else f"this {kind} type has no entities"}

    def attribute_values(self, args):
        """The values one attribute actually takes, read off sampled entities.

        The schema block names attributes and their types; it does not say that `status` is one of
        {"open", "closed"}. That is the grounding a filter needs.
        """
        type_name, attribute = args.get("type"), args.get("attribute")
        if not type_name or not attribute:
            return {"error": "attribute_values needs a `type` and an `attribute`"}
        limit = max(1, min(int(args.get("n", 10) or 10), 25))
        kind, found = self._scan_entities(type_name, limit)
        if kind is None:
            return self._unknown_type(type_name)
        if found is None:
            return {"error": f"looking up {type_name!r} failed; try again, or bind it in try_query"}
        values = [entry.get("value") for entity in found
                  for entry in (entity.get("attributes") or [])
                  if entry.get("name") == attribute and entry.get("value") is not None]
        if not values:
            present = sorted({entry.get("name") for entity in found
                              for entry in (entity.get("attributes") or [])} - {None})
            return {"type": type_name, "attribute": attribute, "sampled": 0,
                    "distinct_values": [], "attributes_present": present[:25]}
        return {"type": type_name, "attribute": attribute, "sampled": len(values),
                "distinct_values": sorted({str(value) for value in values})[:25],
                "sampled_from": f"the first {limit} {kind}s of this type in evaluation order, not "
                                f"a uniform sample"}

    def activity_stats(self, args):
        """Timing statistics for an activity -- the one grounding fact absent from the schema block.

        A `TimeBetweenEvents` threshold written without it is a guess.
        """
        activity = args.get("type") or args.get("activity")
        if not activity:
            return {"error": "activity_stats needs an activity (event type) name as `type`"}
        # The handler takes a BARE JSON string, not an object (`Json<String>`, main.rs:436).
        status, body = _http_json(f"{self.base_url}/ocel/get-activity-statistics", activity,
                                  timeout=self.request_timeout_sec)
        if status != 200 or body is None:
            return {"error": _clip(str(body), 500)}
        return {"activity": activity, "statistics": body}

    def _unknown_type(self, type_name):
        """An honest answer for a name that is not a type of this log: say so, and list the real
        ones. (Until 2026-09-23 a prefix scan answered "no entities, check the spelling" for REAL
        types that happened to appear late in the log -- DiagnosticEquipment in 04_hospital.xml.)"""
        return {"error": f"{type_name!r} is not an object or event type of this log",
                "object_types": self.object_types, "event_types": self.event_types}

    def _scan_entities(self, type_name, want):
        """Up to `want` real entities of `type_name`, exactly, wherever they sit in the log.

        The server has no typed id route (`/ocel/sample-ids` returns the first N ids of ANY type),
        so the lookup evaluates a one-box query binding the type and pages its rows -- the same
        evaluator every other answer comes from. That replaces the server's current evaluation, so
        `_agent_eval_current` is cleared and `inspect_situations` re-runs the agent's own tree if it
        needs it.

        Returns:
            `(kind, entities)`: kind is "object" / "event", or None when the name is not a type of
            this log; entities is a list (possibly empty), or None when the lookup itself failed.
        """
        if type_name in self.object_types:
            kind, declared = "object", "newObjectVars"
        elif type_name in self.event_types:
            kind, declared = "event", "newEventVars"
        else:
            return None, []
        box = {"newEventVars": {}, "newObjectVars": {}, "filters": [], "sizeFilters": [],
               "constraints": [], "evVarLabels": {}, "obVarLabels": {}}
        box[declared] = {"0": [type_name]}
        try:
            status, body = _http_json(f"{self.base_url}/ocel/check-constraints-box",
                                      {"tree": {"nodes": [{"Box": [box, []]}], "edgeNames": []},
                                       "measurePerformance": False},
                                      timeout=self.request_timeout_sec)
        except ProbeUnavailable as exc:
            if self.server is not None and _is_timeout(exc):
                self._recover(f"type lookup for {type_name!r} timed out")
                return kind, None
            raise
        if status == 404:
            raise ProbeUnavailable("no OCEL loaded on the probe server")
        self._agent_eval_current = False
        if status != 200 or not isinstance(body, dict):
            return kind, None
        status, page = _http_json(f"{self.base_url}/ocel/eval-results/page",
                                  {"evalVersion": body.get("evalVersion"), "nodeIndex": 0,
                                   "offset": 0, "limit": want, "violated": None},
                                  timeout=self.request_timeout_sec)
        if status != 200 or not isinstance(page, dict):
            return kind, None
        ids = [pair[1] for row in page.get("rows") or [] for pair in row.get(kind + "s") or []
               if isinstance(pair, (list, tuple)) and len(pair) == 2][:want]
        route = "get-object" if kind == "object" else "get-event"
        found = []
        for entity_id in ids:
            status, payload = _http_json(f"{self.base_url}/ocel/{route}", {"id": entity_id},
                                         timeout=self.request_timeout_sec)
            if status == 200 and isinstance(payload, dict) and isinstance(payload.get(kind), dict):
                found.append(payload[kind])
        return kind, found

    # ------------------------------------------------------------------------------ bookkeeping

    def reset(self):
        """Clear the per-case budget, trajectory and health counters. The loaded log and the
        per-run restart count are kept -- they are per-run state."""
        self.tool_calls = 0
        self.trajectory = []
        # The previous case's evaluation must not be pageable from the next case.
        self.eval_version = None
        self._agent_tree, self._agent_eval_current = None, False
        self.restarts_case = 0
        self.evaluator_panics = 0
        self.refused_trees = 0

    def budget_remaining(self):
        return max(0, self.max_tool_calls - self.tool_calls)


def _is_timeout(exc):
    """Whether a `ProbeUnavailable` wraps a socket timeout (`urlopen` raises it bare, or inside a
    `URLError`). `socket.timeout` is an alias of `TimeoutError` from Python 3.10 on."""
    cause = exc.__cause__
    return isinstance(cause, TimeoutError) or isinstance(getattr(cause, "reason", None),
                                                         TimeoutError)


def _camel(snake):
    """`event_types` -> `eventTypes`. The served OCELInfo is snake_case in this build of the
    server; the frontend's TypeScript types are camelCase, so both spellings are accepted rather
    than pinning the identity check to one build."""
    head, *rest = snake.split("_")
    return head + "".join(part.title() for part in rest)


def _type_names(entries):
    """Type names from an OCELInfo list, which holds `{"name", "attributes"}` objects, not strings."""
    names = set()
    for entry in entries:
        if isinstance(entry, dict):
            if entry.get("name"):
                names.add(entry["name"])
        elif entry:
            names.add(str(entry))
    return names


def _clip(text, cap):
    """Truncate with a marker that says so, so the model never sees a silently cut-off value."""
    if text is None:
        return None
    if len(text) <= cap:
        return text
    return text[:cap] + f"... [truncated: {len(text)} chars, showing {cap}]"


# The registry is the single source of truth for BOTH the prompt's tool list and the dispatcher, so
# the two cannot drift -- a tool described to the model but not dispatched is a silent dead end.
TOOL_SPECS = [
    {"name": "try_query",
     "args": {"tree": "a complete BindingBoxTree JSON object"},
     "description": "Evaluate a candidate query against the log. Returns, per box, how many "
                    "situations it selects and how many are violations, or the evaluation error."},
    {"name": "inspect_situations",
     "args": {"node": "box index (default 0)", "n": "rows to show, max 20",
              "violated": "optional true/false to show only violations"},
     "description": "Show rows the last try_query actually selected, so you can see WHAT was "
                    "matched, not just how much."},
    {"name": "sample",
     "args": {"type": "an event or object type name", "n": "how many entities, max 20"},
     "description": "Real entities of one type with their attributes, to see what its data looks "
                    "like. The first n of that type, not a uniform sample. A name that is not a "
                    "type of this log is answered with the list of real types."},
    {"name": "attribute_values",
     "args": {"type": "an event or object type name", "attribute": "an attribute name",
              "n": "how many entities to sample"},
     "description": "The values an attribute actually takes, for writing a correct value filter."},
    {"name": "activity_stats",
     "args": {"type": "an activity (event type) name"},
     "description": "Timing statistics for an activity, for TimeBetweenEvents thresholds."},
]

_TOOL_HANDLERS = {
    "try_query": OcpqProbe.try_query,
    "inspect_situations": OcpqProbe.inspect_situations,
    "sample": OcpqProbe.sample,
    "attribute_values": OcpqProbe.attribute_values,
    "activity_stats": OcpqProbe.activity_stats,
}

assert {spec["name"] for spec in TOOL_SPECS} == set(_TOOL_HANDLERS), \
    "TOOL_SPECS and _TOOL_HANDLERS disagree: the model would be told about a tool that cannot run"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", metavar="OCEL_PATH",
                        help="Start a server, load this log, run one probe of each tool, stop.")
    parser.add_argument("--max-tool-calls", type=int, default=10)
    args = parser.parse_args()
    if not args.smoke:
        parser.error("nothing to do; pass --smoke <ocel path>")

    with ProbeServer() as server:
        probe = OcpqProbe(server.base_url, max_tool_calls=args.max_tool_calls, server=server)
        started = time.time()
        probe.ensure_log(args.smoke)
        print(f"loaded {Path(args.smoke).name} in {time.time() - started:.1f}s "
              f"on {server.base_url}")
        schema = extract_ocel_schema(str(Path(args.smoke).resolve()))
        object_type = (schema.get("object_types") or [None])[0]
        print(json.dumps(probe.call("sample", {"type": object_type, "n": 3}), indent=2)[:600])
        for entry in probe.trajectory:
            print(f"  {entry['tool']:20s} {entry['duration_sec']:>7.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
