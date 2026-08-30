"""Inertness of the structure-hint parameter, and correct dispatch of the oracle strategies.

The ablation compares S0 (today's `allowed-entities-only` prompt) against S1/S2 (the same prompt
plus a hint). That comparison is only valid if adding the parameter left S0 byte-identical -- a
stray blank line or a reordered rule would make the baseline a different condition than the matrix
measured, and the difference would look like a result.

Rather than mock a model, this reconstructs the prompt f-string from the source and renders it
both ways, which is what actually ships.

Run from the repo root:  python tests/test_structure_hint_prompt.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import LLM  # noqa: E402
from eval_testset import ORACLE_STRUCTURE_LEVELS, STRUCTURE_HINT_LEVELS  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


SRC = (REPO / "src" / "LLM.py").read_text(encoding="utf-8")

print("--- strategy registration ---")
for name in ("oracle-skeleton-given", "oracle-skeleton-and-constraints-given"):
    check(f"{name} is a known strategy", name in LLM.PROMPT_STRATEGIES, True)
    check(f"{name} normalises to itself", LLM.normalize_prompt_strategy(name), name)
check("ORACLE_PROMPT_STRATEGIES matches the registry",
      set(LLM.ORACLE_PROMPT_STRATEGIES) <= set(LLM.PROMPT_STRATEGIES), True)
check("short alias works", LLM.normalize_prompt_strategy("skeleton-given"),
      "oracle-skeleton-given")
check("every oracle strategy has a hint level",
      set(ORACLE_STRUCTURE_LEVELS) == set(LLM.ORACLE_PROMPT_STRATEGIES), True)
check("every hint level is a real level",
      set(ORACLE_STRUCTURE_LEVELS.values()) <= set(STRUCTURE_HINT_LEVELS), True)
check("ordinary strategies get no level",
      any(s in ORACLE_STRUCTURE_LEVELS for s in
          ("schema-allowed-entities-only", "schema-grammar-guided",
           "schema-chain-of-thought-plus")), False)

print("\n--- the prompt is inert when no hint is given ---")
# Scope to THIS pipeline: LLM.py holds several `prompt = f"""` blocks and the first one belongs
# to a different strategy, so an unscoped search silently tests the wrong template.
_start = SRC.index("def _run_pipeline_allowed_entities_only(")
_end = SRC.index("\ndef ", _start + 1)
FUNC = SRC[_start:_end]
check("function body isolated", "{hint_block}" in FUNC, True)
m = re.search(r'prompt = f"""\n(.*?)\n"""', FUNC, re.S)
check("prompt template located", m is not None, True)
if m:
    template = m.group(1)
    def render(hint_block):
        return template.replace("{schema_block}", "<SCHEMA>") \
                       .replace("{hint_block}", hint_block) \
                       .replace("{query}", "<QUERY>") \
                       .replace("{{", "{").replace("}}", "}")
    without = render("")
    # The pre-parameter template read `{schema_block}\n\nRules:` -- schema, ONE blank line, rules.
    # `{schema_block}\n{hint_block}\nRules:` with an empty hint must render to the same three
    # characters between them. Reconstructing the old string by deleting the placeholder is what
    # a first draft of this test did, and it deletes the newline too, so it asserted the wrong
    # thing; spell the invariant out instead.
    check("exactly one blank line between schema and rules", "<SCHEMA>\n\nRules:" in without, True)
    check("no doubled blank line crept in", "\n\n\n" in without, False)
    check("the hint placeholder left no residue", "{hint_block}" in without, False)
    check("nothing else changed around the seam",
          without.startswith("Convert this natural language query to OCPQ JSON.\n\n"
                             "Output ONLY valid JSON.\n\n<SCHEMA>\n\nRules:"), True)
    check("query section intact", without.rstrip().endswith("Query:\n<QUERY>"), True)

    with_hint = render("\nTree structure -- produce EXACTLY this shape:\n  2 node(s).\n")
    check("hint lands between the schema and the rules",
          with_hint.index("<SCHEMA>") < with_hint.index("Tree structure")
          < with_hint.index("Rules:"), True)
    check("the query still comes last",
          with_hint.index("Rules:") < with_hint.index("<QUERY>"), True)
    check("all original rules survive",
          all(r in with_hint for r in (
              'The root object MUST contain both "nodes" and "edgeNames".',
              'Use ONLY event and object names from the allowed lists.',
              'Keep all nodes connected in a single tree and avoid empty nodes.')), True)

print("\n--- one code path, so the rungs differ only by the hint ---")
check("no duplicated pipeline was introduced",
      SRC.count("def _run_pipeline_allowed_entities_only"), 1)
check("the oracle branch reuses it",
      SRC.count("return _run_pipeline_allowed_entities_only("), 2)
check("failure-log stages are parameterised, not hardcoded",
      'stage="allowed_entities_only' in SRC, False)
check("stage_prefix defaults to the historical name",
      'stage_prefix="allowed_entities_only"' in SRC, True)

print("\n--- structure_hint cannot reach an ordinary strategy ---")
sig = re.search(r"def nl_to_ocpq_pipeline\((.*?)\n\):", SRC, re.S)
check("dispatcher accepts the kwarg", sig is not None and "structure_hint=None" in sig.group(1),
      True)
body = SRC[SRC.index("def nl_to_ocpq_pipeline("):]
body = body[:body.index("Unsupported prompt strategy")]
check("only the oracle branch forwards it", body.count("structure_hint=structure_hint"), 1)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
