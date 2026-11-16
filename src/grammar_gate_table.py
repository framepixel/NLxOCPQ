#!/usr/bin/env python3
"""The invalid documents the BindingBoxTree grammar must refuse, as an appendix table.

WHY
---
Ch. 3 says the grammar was checked against the 169 gold queries and against "the 8 invalid
documents" before use (notes 11 §30.2), and refers to an appendix for them. The documents are the
`INVALID` dict of `tests/test_constrained_decoding.py`, the gate itself. This script reads them
from that file without running it (running it needs xgrammar and a tokenizer), so the table shows
exactly what the gate tested. Each row shows the part of the document that makes it invalid; a
valid box in between is abridged as {...}, and the script checks that every shown fragment is a
literal part of the real document.

Usage (from the repo root):
    python src/grammar_gate_table.py            # writes tables/grammar_invalid_documents.tex
    python src/grammar_gate_table.py --no-write # prints the rows only
"""
import argparse
import ast
from typing import Dict, List, Tuple

import analysis_common as ac
import paths

TEST_PATH = paths.REPO_ROOT / "tests" / "test_constrained_decoding.py"
ELLIPSIS = "{...}"

# test key -> (what is wrong, the fragment shown; ELLIPSIS stands for an abridged valid part)
DISPLAY: Dict[str, Tuple[str, str]] = {
    "prose": ("prose instead of JSON", "The orders delivered late."),
    "markdown fence": ("a Markdown code fence before the JSON", '```json {"nodes":[]}'),
    "wrong top-level key": ("an unknown top-level key", '{"nodez":[]}'),
    "string where an int list belongs": ("a string where the list of child indices belongs",
                                         '{"Box":[{...},"notalist"]}'),
    "invalid filter type literal": ("an unknown filter type",
                                    '"filters":[{"type":"NotAFilter","object":0,"event":0}]'),
    "invalid value_filter type literal": ("an unknown value-filter type",
                                          '"value_filter":{"type":"Nope"}'),
    "missing required field": ("a box without its required fields",
                               '{"Box":[{"newEventVars":{}},[]]}'),
    "Box with three elements": ("a box with three elements instead of two",
                                '{"Box":[{...},[],99]}'),
}


def load_invalid(path=TEST_PATH) -> Dict[str, str]:
    """The test's INVALID dict, evaluated from its own source (only BOX, ATTR_FILTER, INVALID)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    namespace: dict = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("BOX", "ATTR_FILTER", "INVALID")):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
            exec(code, {"__builtins__": {}}, namespace)  # string literals and % only
    return namespace["INVALID"]


def fragment_occurs(fragment: str, document: str) -> bool:
    """Whether the shown fragment is the document's own text, with {...} standing for any span."""
    normalised = document.replace("\n", " ")
    position = 0
    for piece in fragment.split(ELLIPSIS):
        found = normalised.find(piece, position)
        if found < 0:
            return False
        position = found + len(piece)
    return True


def rows(invalid: Dict[str, str]) -> List[Tuple[str, str]]:
    if set(invalid) != set(DISPLAY):
        raise SystemExit(f"ERROR: the test's INVALID keys changed: test {sorted(invalid)}, "
                         f"table {sorted(DISPLAY)}")
    out = []
    for key, document in invalid.items():
        what, fragment = DISPLAY[key]
        if not fragment_occurs(fragment, document):
            raise SystemExit(f"ERROR: the fragment shown for {key!r} is not in its document")
        out.append((what, fragment))
    return out


def _tt(text: str) -> str:
    escaped = (text.replace("\\", "\\textbackslash{}").replace("{", "\\{").replace("}", "\\}")
               .replace("_", "\\_").replace("#", "\\#").replace("%", "\\%").replace("&", "\\&")
               .replace("$", "\\$").replace("^", "\\^{}").replace("~", "\\~{}")
               .replace("`", "\\textasciigrave{}").replace("\\{...\\}", "\\{\\dots\\}")
               .replace(",", ",\\allowbreak{}"))     # long JSON may break after a comma
    return f"\\texttt{{{escaped}}}"


def tabular(table_rows: List[Tuple[str, str]]) -> str:
    lines = [
        "  \\small",
        "  \\begin{tabularx}{\\textwidth}{@{}r>{\\raggedright\\arraybackslash}p{4.6cm}"
        ">{\\raggedright\\arraybackslash}X@{}}",
        "    \\toprule",
        "    & What is wrong & The part of the document that is wrong \\\\",
        "    \\midrule",
    ]
    for number, (what, fragment) in enumerate(table_rows, 1):
        lines.append(f"    {number} & {what} & {_tt(fragment)} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


CAPTION = ("The invalid documents the BindingBoxTree grammar had to refuse before it was used;"
           " \\texttt{\\{\\dots\\}} abridges a valid box.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-write", action="store_true", help="print the rows only")
    args = parser.parse_args(argv)
    table_rows = rows(load_invalid())
    for number, (what, fragment) in enumerate(table_rows, 1):
        print(f"{number}  {what:50} {fragment}")
    if not args.no_write:
        out = ac.write_tex("grammar_invalid_documents", tabular(table_rows), CAPTION,
                           "tab:grammar:invalid",
                           notes=[f"documents: {ac.repo_relative(TEST_PATH)} (INVALID), not run"])
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
