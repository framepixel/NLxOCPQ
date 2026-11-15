#!/usr/bin/env python3
"""Thesis tables for the ablation sections (Ch. 4) that no other script writes.

`rung_tables.py` gives each rung's level and `paired_stats.py` the paired differences (including
`paired_decoding_hint` and `paired_resolution`). This adds the rest:

  structure_hint      the structure-hint ablation (H0-H2; notes 11 §25, §26.5; re-run 2026-10-27)
  rescoring           the metric variants on the self-consistency run (§26, §27.3; re-run
                      2026-10-29)
  decoding            D0-D2: token budget and constrained decoding (§31; re-run 2026-10-27), with
                      the answers the constraint made valid
  arity_split         the arity oracle and the constant, split by gold's root arity (§33.1, §41.2)
  arity_predictor     the arity predictor against the constant, on the current gold (§42; IC-09)
  ablation_synthesis  every direction's best F1 against the trivial floor

Every condition is the one `configs/analysis_contrasts.json` defines, read through
`paired_stats.RunCache`, so each run is checked against its summary.json first. It computes no
metric: F1, skill, the variants and the evaluated flag are read from the records exactly as
`src/eval_testset.py` wrote them (CLAUDE.md 2.5).

Inputs besides the runs: `eval_runs/analysis/arity_probe_regold.json`
(`python src/arity_predictor_probe.py --rescore eval_runs/arity_probe/arity_probe_20260920.json
--regold --out eval_runs/analysis/arity_probe_regold.json`) and
`eval_runs/analysis/selfconsistency.json` (`src/selfconsistency.py`).

Runs in the evaluation environment: gold's root arity comes from `eval_testset._gold_root_arity`.

    python src/ablation_tables.py
"""
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_common as ac  # noqa: E402
import paths  # noqa: E402
import paired_stats as ps  # noqa: E402
from analyze_divergence import _node_count  # noqa: E402
from eval_testset import _gold_root_arity  # noqa: E402

TRIVIAL_FLOOR = 0.7250  # mean trivial floor over the 135 discriminating questions (CLAUDE.md 6)
MAJORITY = (0, 1)  # the most common root arity (events, objects) of gold and of the corpus
ARITY_PROBE = paths.ANALYSIS_DIR / "arity_probe_regold.json"
SELFCONS = paths.ANALYSIS_DIR / "selfconsistency.json"
# Re-run 2026-10-29 on the pinned corpus; the old-corpus run is logs/selfcons/merged_recomputed.
SELFCONS_RUN = "eval_runs/selfconsistency/merged"
SELFCONS_ARM = "llama31-8b-finetuned"
VARIANTS = (("strict", "Strict (primary)"), ("population", "Population (flag dropped)"),
            ("violation", "Violation (flagged rows only)"),
            ("kind_projected", "Kind-projected (extra variables ignored)"))

SPEC = ps.load_spec()
CACHE = ps.RunCache()


def records(name: str) -> List[dict]:
    """The records of one spec condition."""
    condition = SPEC["conditions"][name]
    return ps._condition_records(condition, CACHE.get(condition["run"]))


def runs_of(names: List[str]) -> List[ac.Run]:
    """The distinct runs behind the named conditions, for the provenance header."""
    found = {}
    for name in names:
        run = CACHE.get(SPEC["conditions"][name]["run"])
        found[run.provenance["run"]] = run
    return list(found.values())


def disc_mean(rows: List[dict], metric: str) -> Optional[float]:
    """Mean over the discriminating questions (each question's mean over its draws)."""
    return ac.mean(ac.case_values(rows, metric).values())


def exact_share(rows: List[dict]) -> Optional[float]:
    """Share of discriminating records whose F1 is 1 (`rung_tables.level`'s definition)."""
    return ac.mean(1.0 if ac.strict_f1(r) == 1.0 else 0.0
                   for r in rows if r.get("discriminating") is True)


def _row(label: str, values: List[str]) -> str:
    return f"    {label} & " + " & ".join(values) + " \\\\"


def _panel(title: str, width: int) -> str:
    return f"    \\multicolumn{{{width}}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"


def _count(count: int, total: int) -> str:
    return f"{count} ({ac.fmt_pct(count / total)})" if total else str(count)


def _tabularx(columns: int) -> List[str]:
    return ["  \\small",
            f"  \\begin{{tabularx}}{{\\textwidth}}{{@{{}}>{{\\raggedright\\arraybackslash}}X"
            f"{'r' * columns}@{{}}}}",
            "    \\toprule"]


# ------------------------------------------------------------------------------------------------
# Structure hint.
# ------------------------------------------------------------------------------------------------

def hint_level(rows: List[dict]) -> Dict[str, object]:
    """One structure-hint rung, with the shape measures of `structure_ablation_report.py`.

    Shape adoption is gold's number of boxes in a parsed prediction. The conditional F1 is over
    the discriminating records whose prediction parsed, split by adoption (notes 11 §25.5).
    """
    parsed = [r for r in rows if _node_count(r.get("prediction")) is not None]
    adopted = [r for r in parsed
               if _node_count(r.get("prediction")) == _node_count(r.get("gold_query"))]
    disc_parsed = [r for r in parsed if r.get("discriminating") is True]
    with_shape = [ac.strict_f1(r) for r in disc_parsed if r in adopted]
    without = [ac.strict_f1(r) for r in disc_parsed if r not in adopted]
    return {
        "questions": len(rows), "parsed": len(parsed), "adopted": len(adopted),
        "mean_boxes": statistics.mean(_node_count(r["prediction"]) for r in parsed),
        "evaluated": sum(ac.evaluated(r) for r in rows),
        "f1": disc_mean(rows, "strict_f1"), "kind_projected": disc_mean(rows, "kind_projected_f1"),
        "skill": disc_mean(rows, "skill"), "exact": exact_share(rows),
        "f1_adopted": (ac.mean(with_shape), len(with_shape)),
        "f1_not_adopted": (ac.mean(without), len(without)),
    }


def structure_hint() -> Path:
    names = ("H0", "H1", "H2")
    levels = [hint_level(records(n)) for n in names]
    n = levels[0]["questions"]
    lines = _tabularx(3) + [
        "    & \\makecell[r]{H0\\\\no hint} & \\makecell[r]{H1\\\\skeleton}"
        " & \\makecell[r]{H2\\\\skeleton and\\\\constraint types} \\\\",
        "    \\midrule",
        _panel(f"All {n} questions", 4),
        _row("Parsed trees", [str(lv["parsed"]) for lv in levels]),
        _row("Gold's number of boxes", [_count(lv["adopted"], n) for lv in levels]),
        _row("Mean boxes (parsed trees)", [f"{lv['mean_boxes']:.2f}" for lv in levels]),
        _row("Evaluated", [_count(int(lv["evaluated"]), n) for lv in levels]),
        "    \\midrule",
        _panel("135 discriminating questions", 4),
        _row("F1", [ac.fmt(lv["f1"]) for lv in levels]),
        _row("F1, kind-projected", [ac.fmt(lv["kind_projected"]) for lv in levels]),
        _row(ac.SKILL_LABEL, [ac.fmt(lv["skill"], 3) for lv in levels]),
        _row("Exact", [ac.fmt_pct(lv["exact"]) for lv in levels]),
        _row("F1 with gold's number of boxes",
             [f"{ac.fmt(lv['f1_adopted'][0])} ({lv['f1_adopted'][1]})" for lv in levels]),
        _row("F1 with another number",
             [f"{ac.fmt(lv['f1_not_adopted'][0])} ({lv['f1_not_adopted'][1]})" for lv in levels]),
        "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "structure_hint", "\n".join(lines),
        "Structure hint (fine-tuned Llama 3.1 8B, allowed entities only, free decoding): H1 "
        "and H2 give the model gold's tree as a scaffold (oracles).",
        "tab:abl:structure", runs_of(list(names)),
        notes=["re-run 2026-10-27 on the pinned corpus; the old-corpus run (notes 11 §25) is "
               "logs/structable/merged_recomputed"])


# ------------------------------------------------------------------------------------------------
# Rescoring.
# ------------------------------------------------------------------------------------------------

def rescoring() -> Path:
    run = CACHE.get(SELFCONS_RUN)
    arm = [r for r in run.records if r.get("model_label") == SELFCONS_ARM]
    bundle = run.summary["by_stratum"]["by_model"][SELFCONS_ARM]["discriminating"]["variants"]
    disc = {ac.entry_key(r): r for r in arm if r.get("discriminating") is True}
    lines = _tabularx(6) + [
        "    Variant & F1 & Records & \\makecell[r]{Trivial\\\\floor}"
        " & \\makecell[r]{Questions with\\\\floor 1.0} & Skill & \\makecell[r]{Skill\\\\records}"
        " \\\\",
        "    \\midrule"]
    for name, label in VARIANTS:
        block = bundle[name]
        at_one = sum(1 for r in disc.values()
                     if (r.get("baseline_f1_variants") or {}).get(name) == 1.0)
        lines.append(_row(label, [ac.fmt(block["result_set_f1_strict"]), f"{block['n']:,}",
                                  ac.fmt(block["mean_trivial_floor"]), str(at_one),
                                  ac.fmt(block["skill_score"], 3), f"{block['skill_n']:,}"]))
    strict = ac.case_values(arm, "strict_f1")
    projected = ac.case_values(arm, "kind_projected_f1")
    diffs = list(ps.paired({"kp": projected, "st": strict}, {"kp": 1.0, "st": -1.0}).values())
    settings = SPEC["settings"]
    low, high = ps.bootstrap_ci(diffs, settings["resamples"], settings["seed"],
                                settings["confidence"])
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    paired_text = (f"Paired over the {len(diffs)} questions, kind-projected $-$ strict is "
                   f"{ac.fmt(statistics.mean(diffs), signed=True)} "
                   f"[{ac.fmt(low, signed=True)}, {ac.fmt(high, signed=True)}] "
                   f"({better} questions better, {worse} worse; percentile bootstrap, "
                   f"\\num{{{settings['resamples']}}} resamples, seed {settings['seed']}).")
    return ac.write_tex(
        "rescoring", "\n".join(lines),
        "The result comparison under four projections (self-consistency run, fine-tuned "
        "Llama 3.1 8B, 135 discriminating questions); strict is the primary metric.",
        "tab:abl:rescoring", [run],
        notes=["re-run 2026-10-29 on the pinned corpus (old corpus: "
               "logs/selfcons/merged_recomputed)",
               "each variant is read against its own trivial floor; skill is undefined where "
               "that floor is 1.0, violation F1 where gold flags nothing",
               paired_text])


# ------------------------------------------------------------------------------------------------
# Decoding.
# ------------------------------------------------------------------------------------------------

def decoding_level(rows: List[dict], first: Optional[List[dict]]) -> Dict[str, object]:
    """One decoding rung; `first` is the table's first rung, for the newly valid answers."""
    tokens = [(r.get("llm") or {}).get("completion_tokens") for r in rows]
    calls = [(r.get("llm") or {}).get("calls") for r in rows]
    level = {
        "schema_valid": int(sum(ac.schema_valid(r) for r in rows)),
        "evaluated": disc_mean(rows, "evaluated"), "f1": disc_mean(rows, "strict_f1"),
        "exact": exact_share(rows),
        "median_tokens": statistics.median(t for t in tokens if t is not None),
        "mean_calls": statistics.mean(c for c in calls if c is not None),
        "newly_valid": None,
    }
    if first is not None:
        # Discriminating questions only, as notes 11 §31.3 counted them (32 / 0.1427 / 25 / 2).
        invalid = {ac.entry_key(r) for r in first if not ac.schema_valid(r)}
        newly = [r for r in rows if ac.entry_key(r) in invalid and ac.schema_valid(r)
                 and r.get("discriminating") is True]
        f1s = [ac.strict_f1(r) for r in newly]
        level["newly_valid"] = {"n": len(newly), "mean_f1": ac.mean(f1s),
                                "zero": sum(1 for f in f1s if f == 0.0),
                                "exact": sum(1 for f in f1s if f == 1.0)}
    return level


def decoding_table(name: str, conditions: List[tuple], caption: str, label: str,
                   notes: List[str]) -> Path:
    first = records(conditions[0][0])
    levels = [decoding_level(records(c), None if i == 0 else first)
              for i, (c, _) in enumerate(conditions)]
    n = len(first)
    width = len(conditions)
    newly = [lv["newly_valid"] for lv in levels]
    lines = _tabularx(width) + [
        "    & " + " & ".join(header for _, header in conditions) + " \\\\",
        "    \\midrule",
        _row(f"Schema-valid (of {n})", [_count(lv["schema_valid"], n) for lv in levels]),
        _row("Evaluated", [ac.fmt_pct(lv["evaluated"]) for lv in levels]),
        _row("F1", [ac.fmt(lv["f1"]) for lv in levels]),
        _row("Exact", [ac.fmt_pct(lv["exact"]) for lv in levels]),
        _row("Median completion tokens", [f"{lv['median_tokens']:.0f}" for lv in levels]),
        _row("Mean LLM calls", [f"{lv['mean_calls']:.2f}" for lv in levels]),
        "    \\midrule",
        _panel(f"Discriminating questions schema-invalid under {conditions[0][0]}, valid here",
               width + 1),
        _row("Questions", ["--" if v is None else str(v["n"]) for v in newly]),
        _row("Mean F1", ["--" if v is None else ac.fmt(v["mean_f1"]) for v in newly]),
        _row("F1 exactly 0", ["--" if v is None else _count(v["zero"], v["n"]) for v in newly]),
        _row("Exact", ["--" if v is None else str(v["exact"]) for v in newly]),
        "    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(name, "\n".join(lines), caption, label,
                        runs_of([c for c, _ in conditions]), notes=notes)


def decoding() -> List[Path]:
    common = ("schema validity, completion tokens and LLM calls are over all 169 questions; "
              "everything else over the 135 discriminating ones")
    # One table since the 2026-10-27 re-run put all three rungs on the pinned corpus. D0 and D2
    # repeat N0 and N1 of the arity ablation; the two differ on one question (run-to-run noise).
    return [
        decoding_table(
            "decoding", [("D0", "\\makecell[r]{D0 free,\\\\2,000 tokens}"),
                         ("D1", "\\makecell[r]{D1 free,\\\\4,000 tokens}"),
                         ("D2", "\\makecell[r]{D2 constrained,\\\\4,000 tokens}")],
            "Token budget and grammar-constrained decoding (fine-tuned Llama 3.1 8B, allowed "
            "entities only, validation on).",
            "tab:abl:decoding",
            ["re-run 2026-10-27 on the pinned corpus; the old-corpus runs (notes 11 §31) are "
             "eval_runs/decoding_ablation/*_20260910_*", common]),
    ]


# ------------------------------------------------------------------------------------------------
# Root arity.
# ------------------------------------------------------------------------------------------------

def gold_arity(record: dict) -> Optional[tuple]:
    """Gold's root arity (events, objects) as the run scored it."""
    arity = _gold_root_arity(record.get("gold_query"))
    return None if arity is None else tuple(arity[1:])


def split_means(rows: List[dict]) -> Dict[str, tuple]:
    """Mean discriminating F1 for the majority arity, the rest, and all: {group: (mean, n)}."""
    values = ac.case_values(rows, "strict_f1")
    by_key = {ac.entry_key(r): gold_arity(r) for r in rows}
    groups = {"majority": [v for k, v in values.items() if by_key[k] == MAJORITY],
              "other": [v for k, v in values.items() if by_key[k] != MAJORITY],
              "all": list(values.values())}
    return {name: (ac.mean(vals), len(vals)) for name, vals in groups.items()}


def arity_split() -> Path:
    pairs = (("N1", "N2"), ("B0", "B2"))
    splits = {name: split_means(records(name)) for pair in pairs for name in pair}
    lines = _tabularx(8) + [
        "    & \\multicolumn{4}{c}{Gold's arity forced (oracle)}"
        " & \\multicolumn{4}{c}{(0,~1) forced (constant)} \\\\",
        "    \\cmidrule(lr){2-5} \\cmidrule(l){6-9}",
        "    Gold root arity & $n$ & N1 free & N2 & $\\Delta$"
        " & $n$ & B0 free & B2 & $\\Delta$ \\\\",
        "    \\midrule"]
    for group, label in (("majority", "(0,~1)"), ("other", "any other"), ("all", "all")):
        values = []
        for free, forced in pairs:
            # The pair's two runs share a gold; the pairs need not (the re-gold moved one
            # discriminating question from (0, 1) to (1, 0) between N and B).
            (a, n), (b, n_forced) = splits[free][group], splits[forced][group]
            if n != n_forced:
                raise ValueError(f"{free} and {forced} split {group} differently: {n}, {n_forced}")
            values += [str(n), ac.fmt(a), ac.fmt(b), ac.fmt(b - a, signed=True)]
        lines.append(_row(label, values))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "arity_split", "\n".join(lines),
        "Forcing the root arity (events, objects), split by gold's root arity: fine-tuned "
        "Llama 3.1 8B, constrained decoding, 135 discriminating questions.",
        "tab:abl:arity:split", runs_of(["N1", "N2", "B0", "B2"]))


def arity_predictor() -> Path:
    probe = ac._read_json(ARITY_PROBE)
    summary, rows = probe["summary"], probe["rows"]
    n = summary["n"]
    per_class = sorted(summary["per_class"].items(), key=lambda kv: -kv[1]["n"])
    lines = _tabularx(3) + [
        "    Gold root arity & Questions & Predictor & Constant (0,~1) \\\\",
        "    \\midrule"]
    for arity, cell in per_class:
        constant = cell["n"] if arity == str(MAJORITY) else 0
        lines.append(_row(arity.replace(", ", ",~"), [
            str(cell["n"]), _count(cell["hit"], cell["n"]), _count(constant, cell["n"])]))
    majority_n = summary["per_class"][str(MAJORITY)]["n"]
    minority_n = n - majority_n
    lines += [
        "    \\midrule",
        _row("All (accuracy)", [str(n), _count(summary["correct"], n),
                                _count(majority_n, n)]),
        _row("Any arity but (0,~1) (recall)",
             [str(minority_n), _count(round(summary["minority_recall"] * minority_n), minority_n),
              _count(0, minority_n)]),
        _row("Abstained (not forced)", [str(n), str(summary["abstained"]), "0"]),
        _row("Wrong arity forced", [str(n), str(sum(summary["forced_wrong"].values())),
                                    str(minority_n)]),
        "    \\midrule",
        _panel("Estimated gain in summed F1 over the 169 questions", 4),
        _row("A wrong force loses the free score", ["", ac.fmt(summary["expected_value"], 2,
                                                               signed=True),
                                                    ac.fmt(summary["constant_expected_value"], 2,
                                                           signed=True)]),
        _row("A wrong force costs nothing", ["", ac.fmt(summary["expected_value_zero_loss"], 2,
                                                         signed=True),
                                             ac.fmt(summary["constant_expected_value_zero_loss"],
                                                    2, signed=True)]),
        "    \\bottomrule", "  \\end{tabularx}"]
    regolded = len((probe.get("regold") or {}).get("rows_changed") or [])
    return ac.write_tex(
        "arity_predictor", "\n".join(lines),
        "Predicting the root arity (events, objects) from the question against always "
        "answering (0,~1): fine-tuned Llama 3.1 8B, all 169 questions.",
        "tab:abl:arity:predictor",
        notes=[f"input: {ac.repo_relative(ARITY_PROBE)} (rescored from "
               f"{probe.get('rescored_from')}, --regold)",
               f"prompt v3; scored against the current gold ({regolded} question re-golded "
               "since the probe ran)"])


# ------------------------------------------------------------------------------------------------
# Synthesis.
# ------------------------------------------------------------------------------------------------

def synthesis() -> Path:
    """The best F1 each direction reached, against the trivial floor."""
    selfcons = ac._read_json(SELFCONS)["breakdown"]
    oracle = selfcons[f"{SELFCONS_ARM}_schema-allowed-entities-only_schema-on"]["all_cases"]
    best_of_10 = oracle["f1_discriminating"]["oracle"]
    hint = hint_level(records("H0"))
    entries = [
        ("Vote among sampled queries", "majority of 10 samples", "corrected",
         oracle["f1_discriminating"]["majority"]),
        ("Select among sampled queries", "best of 10 samples (oracle)", "corrected", best_of_10),
        ("A larger token budget", "D1", "corrected", disc_mean(records("D1"), "strict_f1")),
        ("Grammar-constrained decoding", "N1", "corrected", disc_mean(records("N1"), "strict_f1")),
        ("Gold's tree given as a hint", "H1 (oracle)", "corrected",
         disc_mean(records("H1"), "strict_f1")),
        ("Gold's number of boxes produced", "H0, conditional", "corrected", hint["f1_adopted"][0]),
        ("A structural check", "B1", "corrected", disc_mean(records("B1"), "strict_f1")),
        ("Root arity predicted", "not run (gate failed)", "corrected", None),
        ("Root arity constant", "B2", "corrected", disc_mean(records("B2"), "strict_f1")),
        ("Root arity from gold", "N2 (oracle)", "corrected", disc_mean(records("N2"), "strict_f1")),
    ]
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xlrr@{}}",
             "    \\toprule",
             "    Direction & Measured by & F1 & \\makecell[r]{Below the\\\\floor by} \\\\",
        "    \\midrule"]
    # Every entry is on the pinned corpus since the 2026-10 re-runs; the corpus field stays in
    # `entries` so that a future old-corpus row has to say so in its cell.
    for direction, rung, corpus, f1 in entries:
        if corpus != "corrected":
            rung = f"{rung}, {corpus} corpus"
        lines.append(_row(direction, [rung, ac.fmt(f1),
                                      "--" if f1 is None else ac.fmt(TRIVIAL_FLOOR - f1)]))
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return ac.write_tex(
        "ablation_synthesis", "\n".join(lines),
        "The best F1 each ablation direction reached on the 135 discriminating questions, "
        f"against the trivial floor of {TRIVIAL_FLOOR:.4f}.",
        "tab:abl:synthesis", runs_of(["D1", "N1", "H0", "H1", "B1", "B2", "N2"]),
        notes=[f"input: {ac.repo_relative(SELFCONS)} (best of 10)",
               "every row on the pinned corpus since the 2026-10-27/29 re-runs (TH-07 resolved)"])


def main() -> int:
    written = [structure_hint(), rescoring(), *decoding(), arity_split(), arity_predictor(),
               synthesis()]
    for path in written:
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
