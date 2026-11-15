#!/usr/bin/env python3
"""Thesis tables for the failure anatomy (Ch. 4, "Failure anatomy").

It renders two analyses that already exist; it computes no metric of its own:

  * `src/failure_anatomy.py` on the reference matrix (notes 11 §40.2--§40.4): entity grounding and
    composition on wrong answers, the gold-depth frontier, and the shape of emitted trees;
  * `src/root_population_diagnostic.py` on the arity rungs N1 and N2 (notes 11 §36.3): why a
    scoreable answer selects the wrong rows, and the clearing oracle.

Produce the inputs first (both are cheap, CPU only):

    python src/failure_anatomy.py --run_dir eval_runs/matrix_v2/merged --pooled --training \\
        --out eval_runs/analysis/failure_anatomy.json
    python src/root_population_diagnostic.py --run_dir <N1 shard> \\
        --out eval_runs/analysis/root_population_N1.json      (and N2; see tables/README.md)

then
    python src/failure_tables.py

Tables (in `paths.THESIS_TABLES`): failure_grounding, failure_depth, failure_shape,
failure_populations.

The anatomy reads the as-run `merged`, not the re-golded reference: every recompute, the re-gold
included, loses the comparison of the 211 predictions whose result sets exceeded the store's cap
(notes 11 §40.1). They all scored 0, so F1 is unaffected, but they are wrong answers, and dropping
them changes the anatomy's denominators (63 wrong answers -> 53 in the fine-tuned arm). The
re-gold changed the gold of one question (notes 11 §43.2).
"""
import hashlib
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_common as ac  # noqa: E402
import paths  # noqa: E402
from matrix_tables import arm_label, arms  # noqa: E402

ANATOMY = paths.ANALYSIS_DIR / "failure_anatomy.json"
POPULATIONS = (("N1", paths.ANALYSIS_DIR / "root_population_N1.json"),
               ("N2", paths.ANALYSIS_DIR / "root_population_N2.json"))
# The two `failure_anatomy.py --condition` arms, as the script names them.
PRIMARY = "primary: llama31-8b-finetuned / schema-allowed-entities-only / schema=on"
BASE_PAIR = "base pair: llama31-8b-base / schema-chain-of-thought-plus / schema=on"
ARM_TITLES = ((PRIMARY, r"\makecell[r]{Fine-tuned,\\allowed entities only}"),
              (BASE_PAIR, r"\makecell[r]{Base,\\chain-of-thought plus}"))
# failure_anatomy buckets with some, but not complete, overlap with gold.
PARTLY = ("partial", "over_broad", "over_narrow")
DEPTHS = (("1", "1"), ("2", "2"), ("3", "3"), ("4+", r"4 or more"))
# Failure classes of root_population_diagnostic, in the order of notes 11 §36.3.
CLASSES = (("arity_mismatch", "Different number of root variables"),
           ("type_mismatch", "Same number, different types"),
           ("link_mismatch", "Same variables, different root filters"),
           ("size_filter_mismatch", "Same filters, different size filters"),
           ("residual", "Root node matches, rows still differ"))
LINK_SUBCLASSES = (("attribute", "attribute filter missing, extra or elsewhere"),
                   ("placement", "attribute filter where gold has none"),
                   ("bounds", "same attribute, different bounds"),
                   ("value_type", "same attribute, different value type"),
                   ("structural_link", "an object, event or time link differs"))
CLEARED = (("no_result", "no result"), ("arity_mismatch", "different number of variables"),
           ("type_mismatch", "different types"), ("link_mismatch", "different root filters"))


def _row(label: str, values: List[str]) -> str:
    return f"    {label} & " + " & ".join(values) + " \\\\"


def _share(count: int, n: int) -> str:
    return f"{count} ({ac.fmt_pct(count / n if n else None)})"


def _input_note(run_dir: str) -> str:
    """`input: <run> sha256 <prefix>` for a run directory recorded in an analysis JSON."""
    path = ac.resolve_run(run_dir)
    digest = hashlib.sha256((path / "predictions.json").read_bytes()).hexdigest()[:16]
    return f"input: {ac.repo_relative(path)} sha256 {digest}"


def _rung_f1(run_dir: str) -> float:
    """The rung's strict F1 over the discriminating questions: the clearing oracle's start."""
    values = ac.case_values(ac.load_run(run_dir).records, "strict_f1")
    return ac.mean(values.values())


def grounding(anatomy: dict, notes: List[str]) -> Path:
    """Wrong answers: does the model name the right types, and build the right tree?"""
    cells = [anatomy["conditions"][key] for key, _ in ARM_TITLES]
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrr@{}}",
             "    \\toprule",
             "    & " + " & ".join(title for _, title in ARM_TITLES) + " \\\\",
             "    \\midrule",
             # Every answer in one outcome (the author's rows, 2026-10-26), so the wrong answers
             # read against how many answers each arm returned at all. Partly correct pools
             # failure_anatomy's partial, over_broad and over_narrow buckets: each column sums to n.
             f"    \\multicolumn{{3}}{{@{{}}l}}{{\\textit{{All {cells[0]['n']} answers}}}} \\\\",
             _row("No result", [str(c["buckets"].get("unscored", 0)) for c in cells]),
             _row("Exactly correct (F1 = 1)", [str(c["buckets"].get("exact", 0)) for c in cells]),
             _row("Partly correct (0.05 $\\leq$ precision or recall, F1 $<$ 1)",
                  [str(sum(c["buckets"].get(k, 0) for k in PARTLY)) for c in cells]),
             _row("Wrong (disjoint from gold)", [str(c["entities"]["n"]) for c in cells]),
             "    \\midrule",
             "    \\multicolumn{3}{@{}l}{\\textit{Same as gold, as a share of the wrong answers}}"
             " \\\\"]
    for key, label, part in (("object_types", "Object types", "entities"),
                             ("event_activities", "Event types", "entities"),
                             ("both", "Object and event types", "entities"),
                             ("node_count", "Number of nodes", "composition"),
                             ("filter_multiset", "Filter types", "composition"),
                             ("constraint_multiset", "Constraint types", "composition")):
        lines.append(_row(label, [_share(c[part][key], c[part]["n"]) for c in cells]))
    lines += ["    \\midrule",
              "    \\multicolumn{3}{@{}l}{\\textit{Gold against prediction}} \\\\",
              _row("Mean nodes", [f"{c['composition']['gold_boxes']:.2f} / "
                                  f"{c['composition']['pred_boxes']:.2f}" for c in cells]),
              _row("\\texttt{SAT} constraints",
                   [f"{c['composition']['constructs'].get('SAT', {}).get('gold', 0)} / "
                    f"{c['composition']['constructs'].get('SAT', {}).get('predicted', 0)}"
                    for c in cells]),
              "    \\midrule",
              "    \\multicolumn{3}{@{}l}{\\textit{Mean F1 of the answers that returned a result}}"
              " \\\\"]
    for key, label in (("shape_matched", "With gold's number of nodes"),
                       ("shape_mismatched", "With a different number")):
        lines.append(_row(label, [f"{ac.fmt(c['shape'][key]['mean_f1'])} "
                                  f"({c['shape'][key]['n']})" for c in cells]))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "failure_grounding", "\n".join(lines),
        "What the wrong answers get right (Llama 3.1 8B, validation on, all 169 questions); a "
        "wrong answer shares (almost) nothing with gold.",
        "tab:abl:anatomy:grounding", notes=notes)


def depth(anatomy: dict, notes: List[str]) -> Path:
    """The capability frontier: outcome by the number of nodes in the gold query."""
    cells = [anatomy["conditions"][key]["depth"] for key, _ in ARM_TITLES]
    lines = ["  \\small",
             "  \\begin{tabular}{@{}lrrrrrrr@{}}",
             "    \\toprule",
             "    & & \\multicolumn{3}{c}{Fine-tuned, allowed entities only}"
             " & \\multicolumn{3}{c}{Base, chain-of-thought plus} \\\\",
             "    \\cmidrule(lr){3-5} \\cmidrule(l){6-8}",
             "    Gold nodes & Questions"
             + " & \\makecell[r]{Returned\\\\a result} & F1 & Exact" * 2 + " \\\\",
             "    \\midrule"]
    for key, label in DEPTHS:
        values = [str(cells[0][key]["n"])]
        for cell in cells:
            c = cell[key]
            values += [ac.fmt_pct(c["executed"] / c["n"], 0), ac.fmt(c["f1_sum"] / c["n"]),
                       ac.fmt_pct(c["exact"] / c["n"], 1)]
        lines.append(_row(label, values))
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return ac.write_tex(
        "failure_depth", "\n".join(lines),
        "Outcome by the number of nodes in the gold query: Llama 3.1 8B with validation on, all "
        "169 questions.",
        "tab:abl:anatomy:depth", notes=notes)


def _shape_cells(dist: dict, count: str) -> List[str]:
    return [count, ac.fmt_pct(dist["one_box_share"]), f"{dist['mean_boxes']:.2f}",
            ac.fmt_pct(dist["sat_share"])]


def shape(anatomy: dict, notes: List[str]) -> Path:
    """The shape of the trees: gold, the training corpus, and what each arm emits."""
    gold, training = anatomy["gold_distribution"], anatomy["training_distribution"]
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrr@{}}",
             "    \\toprule",
             "    & Trees & One node & \\makecell[r]{Mean\\\\nodes}"
             " & \\makecell[r]{With\\\\\\texttt{SAT}} \\\\",
             "    \\midrule",
             _row("Gold queries of the testset", _shape_cells(gold, str(gold["n"]))),
             _row("Fine-tuning set", _shape_cells(training, f"{training['n']:,}")),
             "    \\midrule",
             "    \\multicolumn{5}{@{}l}{\\textit{Generated, every condition of the arm}} \\\\"]
    for label in arms():
        e = anatomy["pooled"][label]["emitted"]
        lines.append(_row(arm_label(label), [str(e["n"]), ac.fmt_pct(e["one_box"] / e["n"]),
                                             f"{e['mean_boxes']:.2f}",
                                             ac.fmt_pct(e["with_sat"] / e["n"])]))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "failure_shape", "\n".join(lines),
        "The shape of the query trees: gold, the fine-tuning set, and the parsed trees of "
        "each arm over its six matrix conditions.",
        "tab:abl:anatomy:shape", notes=notes)


def populations(reports: List[tuple], notes: List[str]) -> Path:
    """Why a scoreable answer selects the wrong rows, at N1 and N2, and the clearing oracle."""
    blocks = []
    for _, report in reports:
        (block,) = report["by_config"].values()
        blocks.append(block)
    width = 1 + len(reports)
    header = " & ".join(rf"\makecell[r]{{{name}}}" for name, _ in reports)
    lines = ["  \\small",
             f"  \\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}X"
             f"{'r' * len(reports)}@{{}}}}",
             "    \\toprule",
             f"    & {header} \\\\",
             "    \\midrule",
             _row("Questions", [str(b["n"]) for b in blocks]),
             _row("No result", [str(b["no_result"]) for b in blocks]),
             _row("Result with gold's rows", [str(b["population_exact"]) for b in blocks]),
             _row("Result with other rows", [str(b["population_wrong"]) for b in blocks]),
             "    \\midrule",
             f"    \\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{Other rows: first difference in "
             f"the root node}}}} \\\\"]
    for key, label in CLASSES:
        lines.append(_row(label, [str(b["classes"].get(key, 0)) for b in blocks]))
        if key == "link_mismatch":
            lines += [_row(f"\\quad {sub}", [str(b["link_subclasses"].get(k, 0)) for b in blocks])
                      for k, sub in LINK_SUBCLASSES]
    lines += ["    \\midrule",
              f"    \\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{Mean F1 if one or two classes "
              f"were answered correctly (discriminating)}}}} \\\\",
              _row("None (the rung's F1)", [ac.fmt(_rung_f1(r["run_dir"])) for _, r in reports])]
    for key, label in CLEARED:
        lines.append(_row(f"Answers with {label}",
                          [ac.fmt(b["clearing_oracle_discriminating"].get(key)) for b in blocks]))
    # The oracle dict is sorted highest first, so its first entry is the best combination.
    best = [next(iter(b["clearing_oracle_discriminating"].items())) for b in blocks]
    lines.append(_row("Best pair of classes", [ac.fmt(value) for _, value in best]))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    pairs = "; ".join(f"{rung}: {_pair_label(name)}"
                      for (rung, _), (name, _) in zip(reports, best))
    return ac.write_tex(
        "failure_populations", "\n".join(lines),
        "Wrong rows at the arity rungs (fine-tuned Llama 3.1 8B, constrained decoding) over all "
        "169 questions, with the clearing oracle.",
        "tab:abl:anatomy:populations", notes=[*notes, f"best pair: {pairs}"])


def _pair_label(name: str) -> str:
    labels = dict(CLEARED, population_exact="gold's rows", size_filter_mismatch="size filters",
                  residual="residual")
    return " and ".join(labels.get(part, part) for part in name.split("+"))


def main() -> int:
    anatomy = ac._read_json(ANATOMY)
    notes = [_input_note(anatomy["run_dir"]), f"analysis: {ac.repo_relative(ANATOMY)}"]
    written = [grounding(anatomy, notes), depth(anatomy, notes), shape(anatomy, notes)]
    reports = [(name, ac._read_json(path)) for name, path in POPULATIONS]
    notes = [_input_note(r["run_dir"]) for _, r in reports]
    notes += [f"analysis: {ac.repo_relative(path)}" for _, path in POPULATIONS]
    notes.append("N1 and N2 ran on the corrected corpus (notes 11 §35)")
    written.append(populations(reports, notes))
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
