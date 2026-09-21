"""The probe shows the agent the log, never gold, and never eats the per-case time limit.

`src/ocpq_probe.py` is the first component in this project that hands a model information at
inference time beyond the prompt. Three ways it could quietly invalidate every agentic number:

  * **Gold leaking in.** In production the gold query does not exist, so any signal derived from it
    is a capability the deployed system would not have. The module must not import `eval_testset`
    or `result_store`, and must not mention gold at all -- asserted here by scanning the source,
    because a leak would look like a good result rather than a crash.
  * **Swallowing `CaseTimeout`.** The per-case cap is SIGALRM (`eval_testset._case_time_limit`),
    which fires inside the blocking socket read. A bare `except Exception` around a request would
    catch it and leave this arm -- and only this arm -- effectively uncapped, which CLAUDE.md 6
    forbids. Asserted with a sentinel exception that must propagate.
  * **Silently probing the wrong log.** The OCPQ server holds ONE global OCEL with no sessions
    (notes 43), so a stray server on the same port answers happily about a different log. Every
    load is followed by an identity assertion; here it must raise on a mismatch.

It also may not score anything (CLAUDE.md 2.5): it returns counts, rows and error strings.

Run from the repo root:  python tests/test_ocpq_probe.py
"""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ocpq_probe  # noqa: E402
from ocpq_probe import (  # noqa: E402
    TOOL_SPECS,
    LogIdentityError,
    OcpqProbe,
    ProbeBudgetExhausted,
    ProbeUnavailable,
    _clip,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def check_true(name, got):
    check(name, bool(got), True)


class FakeTransport:
    """Stands in for `_http_json`. Records calls and replays queued `(status, body)` answers."""

    def __init__(self, answers):
        self.answers = dict(answers)
        self.calls = []

    def __call__(self, url, payload=None, timeout=None):
        path = url.split("/ocel/", 1)[1] if "/ocel/" in url else url
        self.calls.append((path, payload, timeout))
        answer = self.answers.get(path)
        if answer is None:
            return 404, None
        if callable(answer):
            return answer(payload)
        return answer


def probe_with(answers, **kwargs):
    transport = FakeTransport(answers)
    ocpq_probe._http_json = transport
    options = {"max_tool_calls": 5}
    options.update(kwargs)
    return OcpqProbe("http://127.0.0.1:0", **options), transport


print("--- the module cannot reach gold: an import ban, asserted on the source ---")
source = (ROOT / "src" / "ocpq_probe.py").read_text(encoding="utf-8")
tree = ast.parse(source)
imported = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported.update(alias.name.split(".")[0] for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.module:
        imported.add(node.module.split(".")[0])
check("does not import eval_testset", "eval_testset" in imported, False)
check("does not import result_store", "result_store" in imported, False)
check("does not import tag_testset_bb", "tag_testset_bb" in imported, False)

def _strip_docstrings(module):
    """Executable code only. Prose about gold is wanted -- it is how the rule is documented.

    `ast.unparse` keeps docstrings (they are ordinary string expressions), so they are dropped here
    before the ban is applied; otherwise the module's own explanation of the rule would trip it.
    """
    for node in ast.walk(module):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body.pop(0)
    return ast.unparse(module)


body_without_docstrings = _strip_docstrings(ast.parse(source))
for banned in ("gold_query", "ground_truth", "testset", "baseline_f1", "similarity"):
    check(f"no {banned!r} anywhere in executable code", banned in body_without_docstrings, False)

for scoring in ("def _f1", "precision", "recall", "jaccard", "skill"):
    check(f"computes no metric ({scoring!r} absent)", scoring in body_without_docstrings, False)

print("\n--- a tool result never silently loses its tail ---")
long_text = "x" * 5000
clipped = _clip(long_text, 100)
check("clipped to the cap plus a marker", clipped.startswith("x" * 100), True)
check_true("the marker says how much was dropped", "truncated: 5000 chars, showing 100" in clipped)
check("short text is untouched", _clip("abc", 100), "abc")
check("None stays None", _clip(None, 100), None)

print("\n--- try_query reports per-node counts, and a rejected tree is DATA, not a crash ---")
probe, transport = probe_with({
    "check-constraints-box": (200, {"nodeSummaries": [{"situationCount": 12,
                                                       "situationViolatedCount": 3},
                                                      {"situationCount": 0,
                                                       "situationViolatedCount": 0}],
                                    "bindingsSkipped": False, "evalVersion": 7})})
result = probe.call("try_query", {"tree": {"nodes": [], "edgeNames": []}})
check("per-node counts are the headline",
      result["nodes"], [{"node": 0, "situations": 12, "violations": 3},
                        {"node": 1, "situations": 0, "violations": 0}])
check("the evaluation version is remembered for paging", probe.eval_version, 7)
check("the call is counted against the budget", probe.tool_calls, 1)
check("and recorded in the trajectory", probe.trajectory[0]["tool"], "try_query")

probe, _ = probe_with({"check-constraints-box": (400, "Could not get o1")})
rejected = probe.call("try_query", {"tree": {"nodes": []}})
check("an evaluation error comes back as an error field", rejected, {"error": "Could not get o1"})
check("...and still counts against the budget", probe.tool_calls, 1)

probe, _ = probe_with({"check-constraints-box": (404, None)})
try:
    probe.call("try_query", {"tree": {"nodes": []}})
    check("no loaded OCEL raises ProbeUnavailable", "no raise", "ProbeUnavailable")
except ProbeUnavailable:
    check("no loaded OCEL raises ProbeUnavailable", "ProbeUnavailable", "ProbeUnavailable")

probe, _ = probe_with({"check-constraints-box": (200, {"nodeSummaries": []})})
check("a non-object tree is refused with an explanation, not a stack trace",
      "error" in probe.call("try_query", {"tree": "SELECT * FROM orders"}), True)

print("\n--- inspect_situations needs an evaluation, and says so ---")
probe, _ = probe_with({"eval-results/page": (200, {"rows": [{"objects": [[0, "o1"]]}],
                                                   "filteredCount": 1})})
check("without a prior try_query it explains itself",
      "call try_query first" in probe.call("inspect_situations", {"node": 0})["error"], True)
probe.eval_version = 7
check("with one, it returns rows",
      probe.call("inspect_situations", {"node": 0, "n": 3})["rows"], [{"objects": [[0, "o1"]]}])
probe, _ = probe_with({"eval-results/page": (410, "gone")})
probe.eval_version = 3
expired = probe.call("inspect_situations", {"node": 0})
check("a stale evaluation asks for a fresh try_query rather than failing the case",
      "expired" in expired["error"], True)
check("...and forgets the dead version", probe.eval_version, None)

print("\n--- sample / attribute_values filter by type themselves, because sample-ids does not ---")
entities = {
    "o1": {"object": {"id": "o1", "type": "orders",
                      "attributes": [{"name": "status", "value": "open"}]}},
    "o2": {"object": {"id": "o2", "type": "items",
                      "attributes": [{"name": "status", "value": "shipped"}]}},
    "o3": {"object": {"id": "o3", "type": "orders",
                      "attributes": [{"name": "status", "value": "closed"}]}},
}
answers = {
    "sample-ids": (200, {"object_ids": ["o1", "o2", "o3"], "event_ids": []}),
    "get-object": lambda payload: (200, entities.get(payload.get("id"))),
    "get-event": (404, None),
}
probe, transport = probe_with(answers)
sampled = probe.call("sample", {"type": "orders", "n": 5})
check("only entities of the requested type come back",
      [entity["id"] for entity in sampled["entities"]], ["o1", "o3"])
check_true("the result admits the sample is log-ordered", "not a uniform" in sampled["sampled_from"])
check("the untyped id request asks for the scan limit, not n",
      transport.calls[0][1], {"limit": probe.scan_limit})

probe, _ = probe_with(answers)
values = probe.call("attribute_values", {"type": "orders", "attribute": "status"})
check("distinct values are read off the sampled entities",
      values["distinct_values"], ["closed", "open"])
probe, _ = probe_with(answers)
missing = probe.call("attribute_values", {"type": "orders", "attribute": "nope"})
check("an unknown attribute lists what IS there instead of returning nothing",
      missing["attributes_present"], ["status"])

print("\n--- activity_stats posts a bare JSON string (Json<String> in main.rs) ---")
probe, transport = probe_with({"get-activity-statistics": (200, {"meanSec": 42})})
stats = probe.call("activity_stats", {"type": "place order"})
check("the payload is the activity name itself", transport.calls[0][1], "place order")
check("statistics come back under a named key", stats["statistics"], {"meanSec": 42})

print("\n--- the budget is a hard stop, and an unknown tool is the model's mistake ---")
probe, _ = probe_with({"check-constraints-box": (200, {"nodeSummaries": []})}, max_tool_calls=2)
probe.call("try_query", {"tree": {}})
probe.call("try_query", {"tree": {}})
check("budget remaining reaches zero", probe.budget_remaining(), 0)
try:
    probe.call("try_query", {"tree": {}})
    check("a spent budget raises", "no raise", "ProbeBudgetExhausted")
except ProbeBudgetExhausted:
    check("a spent budget raises", "ProbeBudgetExhausted", "ProbeBudgetExhausted")

probe, _ = probe_with({}, max_tool_calls=3)
unknown = probe.call("summon_gold", {})
check_true("an unknown tool name is reported to the model", "unknown tool" in unknown["error"])
check("...and is not charged to the budget", probe.tool_calls, 0)
check("...but is recorded, so a looping agent is visible", probe.trajectory[0]["tool"], "summon_gold")

print("\n--- reset clears the case, keeps the run ---")
probe, _ = probe_with({"check-constraints-box": (200, {"nodeSummaries": []})})
probe.loaded_ocel_path = "/logs/order-management.json"
probe.call("try_query", {"tree": {}})
probe.reset()
check("budget is restored", probe.tool_calls, 0)
check("trajectory is cleared", probe.trajectory, [])
check("the loaded log survives -- it is per-run state", probe.loaded_ocel_path,
      "/logs/order-management.json")

print("\n--- SIGALRM safety: the probe must not swallow what it did not expect ---")


class Sentinel(BaseException):
    """Stands in for eval_testset.CaseTimeout, which SIGALRM raises inside the socket read."""


def exploding_transport(url, payload=None, timeout=None):
    raise Sentinel("the per-case cap fired")


ocpq_probe._http_json = exploding_transport
probe = OcpqProbe("http://127.0.0.1:0", max_tool_calls=5)
try:
    probe.call("try_query", {"tree": {}})
    check("a timeout propagates out of the probe", "swallowed", "propagated")
except Sentinel:
    check("a timeout propagates out of the probe", "propagated", "propagated")
except Exception as exc:  # noqa: BLE001 - the point of the test
    check("a timeout propagates out of the probe", f"converted to {type(exc).__name__}",
          "propagated")

print("\n--- the identity assertion: a server holding another log is fatal ---")
ocpq_probe._http_json = FakeTransport({})
probe = OcpqProbe("http://127.0.0.1:0", max_tool_calls=5)
ocpq_probe.extract_ocel_schema = lambda path: {"event_types": ["place order", "pay"],
                                               "object_types": ["orders"]}
# The served shape is snake_case, and each type is an object with a name, not a bare string --
# verified against a running server on 2026-09-20. The camelCase spelling of the frontend's types
# is accepted too, so the check is not pinned to one build of the server.
served_info = {"event_types": [{"name": "place order", "attributes": []},
                               {"name": "pay", "attributes": []}],
               "object_types": [{"name": "orders", "attributes": []}]}
try:
    probe._assert_identity("/logs/order-management.json", served_info)
    check("the right log passes", "passed", "passed")
except LogIdentityError:
    check("the right log passes", "raised", "passed")
try:
    probe._assert_identity("/logs/order-management.json",
                           {"eventTypes": ["place order", "pay"], "objectTypes": ["orders"]})
    check("the camelCase spelling is accepted too", "passed", "passed")
except LogIdentityError:
    check("the camelCase spelling is accepted too", "raised", "passed")
try:
    probe._assert_identity("/logs/order-management.json",
                           {"event_types": [{"name": "admit patient"}],
                            "object_types": [{"name": "patients"}]})
    check("a different log raises LogIdentityError", "no raise", "LogIdentityError")
except LogIdentityError as exc:
    check("a different log raises LogIdentityError", "LogIdentityError", "LogIdentityError")
    check_true("...and names what was missing", "missing" in str(exc))

print("\n--- the tool registry cannot describe a tool that does not run ---")
check("every advertised tool has a handler",
      sorted(spec["name"] for spec in TOOL_SPECS), sorted(ocpq_probe._TOOL_HANDLERS))
check_true("every tool documents its arguments",
           all(isinstance(spec["args"], dict) and spec["args"] for spec in TOOL_SPECS))
check_true("every tool says what it is for",
           all(len(spec["description"]) > 40 for spec in TOOL_SPECS))
check_true("the registry is JSON-serialisable for the prompt", json.dumps(TOOL_SPECS))

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> the probe reads the log, never gold, and never eats the case cap")
