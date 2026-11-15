"""The agentic arm's outcomes are classified, identified and reported as what they are.

Stage 3 of the agentic plan (notes 43) wires the loop into `eval_testset`. What could make its
numbers quietly wrong, each checked here:

  * **Misfiled failures.** `agent_no_answer` and `probe_unavailable` never reached execution. Filed
    under `execution_failed` -- which a substring match on "execute" would do -- they would drag
    `execution_success_rate`, which must mean "the query ran" (CLAUDE.md 6).
  * **Rungs merged into one.** A1 and A2 share a strategy and differ only in budget; without the
    budget in `config_id`, `--merge-shards` de-duplicates one of them away (the 2026-08-27 failure).
    The one-shot rung A0, run in the same job, must keep its ordinary id.
  * **An unreported compliance failure.** The compliance block is pre-registered to be read
    before any F1, so it must exist for agentic conditions and must not change any other summary.
  * **A second model on the first model's endpoint.** `set_hf_client` keeps the previous
    `base_url`, so a mixed config is refused.
  * **Probe and scorer disagreeing unnoticed.** The agreement check compares the root counts.

Run from the repo root:  python tests/test_agentic_failure_status.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_testset import (  # noqa: E402
    _agent_compliance,
    _agent_suffix,
    _check_backends_consistent,
    _classify_run_status,
    _compact_agent_turns,
    _decoding_suffix,
    _probe_cli_agreement,
)
import ocpq_probe  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


print("--- statuses: never reached execution, so never an execution failure ---")
for status, message in (
        ("agent_no_answer", "the agent produced no usable query in 8 turn(s) (3 malformed)"),
        ("agent_no_answer", "could not execute: no final tree"),       # a wording with 'execute'
        ("probe_unavailable", "ProbeUnavailable: probe request to /ocel/load failed"),
        ("probe_unavailable", "LogIdentityError: execution log mismatch")):
    check(f"{status} stays {status} ({message[:34]}...)",
          _classify_run_status(status, message, False, {}, None, None), status)
check("a timeout is still a timeout",
      _classify_run_status("timeout", "case exceeded 600s during execution", False, {}, None,
                           None), "timeout")
check("an ordinary execution error is still execution_failed",
      _classify_run_status("error", "Refusing to execute: bad", False, {}, None, None),
      "execution_failed")

print("\n--- config_id: the budget IS the rung, and only agentic conditions carry it ---")
a1 = _agent_suffix("agentic-log-tools", 8, 12)
a2 = _agent_suffix("agentic-log-tools", 20, 30)
check("A1 and A2 get different ids", a1 != a2, True)
check("the tool rung names both budgets", a1, "_turns8_tools12")
check("the self-refine control names only turns (it has no tools)",
      _agent_suffix("agentic-self-refine", 8, 12), "_turns8")
check("the one-shot rung A0 keeps its ordinary id",
      _agent_suffix("schema-allowed-entities-only", 8, 12), "")
check("the run-wide decoding suffix is untouched by the agent settings",
      _decoding_suffix(True, False, 4000), "_constrained_tok4000")

print("\n--- the compliance block ---")


def agent_rec(status="success", **agent):
    block = {"turns": 3, "max_turns": 8, "malformed_actions": 0, "tool_calls": 2,
             "final_source": "declared", "post_execution_repair": False}
    block.update(agent)
    return {"status": status, "agent": block}


records = [
    agent_rec(malformed_actions=1, probe_restarts=1,
              probe_vs_cli={"agree": True, "probe_root_situations": 5, "cli_root_situations": 5}),
    agent_rec(turns=8, final_source="last_valid_probe"),                         # exhausted
    agent_rec(status="agent_no_answer", turns=8, final_source="none", malformed_actions=8),
    agent_rec(status="probe_unavailable", turns=0, final_source="none"),
]
block = _agent_compliance(records)
check("denominator is the agentic records", block["agent_n"], 4)
check("malformed rate is per TURN taken (9 of 19)", round(block["agent_malformed_action_rate"], 4),
      round(9 / 19, 4))
check("no-answer rate", block["agent_no_answer_rate"], 0.25)
check("budget exhaustion = every turn used, nothing declared",
      block["agent_budget_exhausted_rate"], 0.5)
check("final_source counted", block["agent_final_source"],
      {"declared": 1, "last_valid_probe": 1, "none": 2})
check("probe_unavailable counted", block["probe_unavailable_n"], 1)
check("restarts summed", block["probe_restarts"], 1)
check("agreement over the cases that have both counts",
      (block["probe_cli_agreement_rate"], block["probe_cli_agreement_n"]), (1.0, 1))
check("a condition with no agentic records gets NO block (no summary changes shape)",
      _agent_compliance([{"status": "success", "agent": None}]), {})

print("\n--- the record keeps what the mechanism analysis reads, clipped ---")
turns = _compact_agent_turns([
    {"turn": 1, "raw": "x" * 9000, "action": "try_query",
     "result": {"nodes": [{"node": 0, "situations": 0, "violations": 0}]}},
    {"turn": 2, "raw": "junk", "malformed": "not valid JSON " + "y" * 900},
    {"turn": 3, "action": "sample", "result": {"error": "no entities of type 'order'"}},
], cap=50)
check("try_query keeps its per-node counts", turns[0]["nodes"][0]["situations"], 0)
check("raw content is NOT on the record (it goes to the JSONL)", "raw" in turns[0], False)
check("a malformed reason is clipped to the cap", len(turns[1]["malformed"]), 50)
check("a tool error is kept", turns[2]["result_error"].startswith("no entities"), True)

print("\n--- mixed endpoints are refused ---")
cfg = {"backend": "custom-endpoint"}
try:
    _check_backends_consistent([{"name": "a", "base_url": "http://one/v1"}, {"name": "b"}], cfg)
    check("an endpoint model followed by a non-endpoint model raises", "no raise", "ValueError")
except ValueError:
    check("an endpoint model followed by a non-endpoint model raises", "ValueError", "ValueError")
_check_backends_consistent([{"name": "a", "base_url": "http://one/v1"},
                            {"name": "b", "base_url": "http://one/v1"}], cfg)
check("two models on one endpoint are fine", True, True)
_check_backends_consistent([{"name": "a", "backend": "local-gpu"},
                            {"name": "b", "backend": "local-gpu"}], {"backend": "local-gpu"})
check("local models are fine (they reset the endpoint)", True, True)

print("\n--- probe vs CLI agreement ---")


class Probe:
    def __init__(self, answer=None, raise_exc=None):
        self.answer, self.raise_exc = answer, raise_exc

    def try_query(self, args):
        if self.raise_exc:
            raise self.raise_exc
        return self.answer


TREE = {"nodes": [{"Box": [{}, [1]]}, {"Box": [{}, []]}], "edgeNames": [[[0, 1], "A"]]}
PRED = (set(), set(), {((1,), (), False), ((2,), (), True)})
agree = _probe_cli_agreement(Probe({"nodes": [{"node": 0, "situations": 2},
                                              {"node": 1, "situations": 9}]}), TREE, PRED, False)
check("equal root counts agree -- the ROOT node, not node 1", agree["agree"], True)
disagree = _probe_cli_agreement(Probe({"nodes": [{"node": 0, "situations": 3}]}), TREE, PRED,
                                True)
check("unequal counts disagree, with the truncation flag beside it",
      (disagree["agree"], disagree["cli_bindings_skipped"]), (False, True))
down = _probe_cli_agreement(Probe(raise_exc=ocpq_probe.ProbeUnavailable("gone")), TREE, PRED,
                            False)
check("a probe failure is 'unknown', never 'disagree'", down["agree"], None)
check("nothing executed -> nothing to compare", _probe_cli_agreement(Probe({}), TREE, None,
                                                                     False), None)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> agentic outcomes are classified, identified and reported as what they are")
