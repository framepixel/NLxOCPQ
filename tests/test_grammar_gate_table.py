"""`grammar_gate_table.py` must show the grammar gate's own invalid documents, unaltered.

Pinned here:
  * the table reads the INVALID dict of tests/test_constrained_decoding.py without running it, and
    has one row per document (8);
  * every fragment shown is literal text of its document, {...} standing for an abridged span, so
    a change to a test document that the table no longer matches stops the script;
  * a fragment that is not in its document is caught.

Run from the repo root:  python tests/test_grammar_gate_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from grammar_gate_table import DISPLAY, fragment_occurs, load_invalid, rows  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


invalid = load_invalid()
check("eight invalid documents in the gate", len(invalid), 8)
check("one row per document", len(rows(invalid)), 8)
check("every shown fragment is in its document",
      all(fragment_occurs(DISPLAY[k][1], doc) for k, doc in invalid.items()), True)
check("the ellipsis stands for any span", fragment_occurs('{"Box":[{...},[],99]}',
                                                          '{"Box":[{"a":1},[],99]}'), True)
check("order matters", fragment_occurs('{"Box":[{...},[],99]}', '{"Box":[[],99,{"a":1}]}'), False)
check("an invented fragment is caught", fragment_occurs('"type":"Nope2"', invalid["prose"]), False)

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
