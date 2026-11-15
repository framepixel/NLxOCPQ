"""`compliance_measures_table.py` must show every gate criterion, with the value the gate applies.

Pinned here:
  * every criterion of every gate in configs/agent_gate_criteria.json is read by some row, so a
    gate cannot be added to the config without the table showing it;
  * the cells render the thresholds the gates apply (evaluation 1: repeats and p90 seconds per turn
    as gated with the thought field off, without footnotes since 2026-11-15; evaluation 2 and the
    example gate: timeouts and p90 case seconds; the example gate alone: uptake);
  * a measure no gate reads is not in the table (since 2026-11-15).

Run from the repo root:  python tests/test_compliance_measures_table.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from compliance_measures_table import (NOT_GATED, cells, load_criteria,  # noqa: E402
                                       unconsumed_criteria)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


criteria = load_criteria()
rows = {name: gate_cells for _group, name, _what, gate_cells in cells(criteria)}

check("every gate criterion has a row", unconsumed_criteria(criteria), {})
check("malformed is gated at 5% everywhere", rows["Malformed actions"],
      ["$\\leq$\\,5\\%"] * 3)
check("evaluation 1 gates repeats with the thought off only", rows["Repeated actions"][0],
      "$\\leq$\\,10\\%")
check("p90 per turn is evaluation 1's", rows["p90 seconds per turn"],
      ["$\\leq$\\,30\\,s", NOT_GATED, NOT_GATED])
check("p90 per case is evaluation 2's and the example gate's", rows["p90 seconds per case"],
      [NOT_GATED, "$\\leq$\\,1,500\\,s", "$\\leq$\\,1,500\\,s"])
check("timeouts are capped only after evaluation 1", rows["Timeouts"],
      [NOT_GATED, "$\\leq$\\,5\\%", "$\\leq$\\,5\\%"])
check("no gate allows a loop-driven timeout", rows["Loop-driven timeouts"], ["none"] * 3)
check("uptake is the example gate's alone", rows["Example-search uptake"],
      [NOT_GATED, NOT_GATED, "$\\geq$\\,25\\%"])
check("submitted answers at least 90% everywhere", rows["Final source"], ["$\\geq$\\,90\\%"] * 3)
check("probe and CLI must agree on every case", rows["Probe/CLI agreement"], ["all cases"] * 3)
ungated = {"Cap hits", "Mean reasoning tokens", "No answer", "Budget exhausted", "Context limit",
           "Probe restarts"}
check("measures no gate reads are left out", sorted(ungated & set(rows)), [])
check("every row is checked by at least one gate",
      [name for name, gate_cells in rows.items() if gate_cells == [NOT_GATED] * 3], [])

# A criterion the table does not know must be caught, not dropped.
extra = {g: dict(c) for g, c in criteria.items()}
extra["eval2"]["new_gate_max"] = 0.1
check("an unknown criterion is reported", unconsumed_criteria(extra), {"eval2": ["new_gate_max"]})

print(f"\n{len(failures)} failure(s)")
sys.exit(1 if failures else 0)
