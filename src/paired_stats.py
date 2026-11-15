#!/usr/bin/env python3
"""Every paired comparison the thesis reports: bootstrap CIs, sign tests and resolution.

WHY
---
Notes 11 quotes paired differences from §25 to §45, and none of them had a committed script
(Thesis/script/FLAGS.md NS-01, NS-07): `src/` contained no resampling code at all, and the
resample count varied between sections (4,000 in §39.4, 10,000 elsewhere). This script is the one
implementation, and `configs/analysis_contrasts.json` is the one list of what is compared.

WHAT A CONTRAST IS
------------------
A weighted sum of conditions, evaluated per question. A condition is a run directory, optionally
narrowed by `select` (e.g. one model's six strategy x validation conditions in the matrix); a
question's value in a condition is the mean over its records there -- over draws, and over pooled
configs. `{"A1": 1, "A1p": -1}` is the plain difference A1 - A1'; four terms give an interaction.
Only questions present in every condition are paired, and a condition missing a question is an
error, not a silent drop.

THE STATISTICS (all over the questions of the population, 135 discriminating by default)
-----------------------------------------------------------------------------------------
mean          the mean per-question difference.
95% CI        percentile bootstrap: resample questions with replacement, `resamples` times.
              The generator is Python's `random.Random(seed)` drawing `randrange(n)` once per
              resampled question -- the exact procedure behind the notes' figures. It reproduces
              §44.9's +0.0087 ... +0.0848 to four decimals, where numpy's generators do not.
better/worse  questions whose difference is > 0 / < 0, with the exact two-sided sign test.
evaluated     the same on the execution indicator (questions newly evaluated / no longer).
identical     questions with byte-identical predictions in both conditions (two-term contrasts,
              over all 169, one draw) -- the reproduction rows of §37 and §41.
resolution    `min_detectable_effect`: the smallest difference n questions resolve at the
              observed paired SD (two-sided alpha, power), and the n a given effect would need --
              the §37.7 "n = 135 resolves about +-0.03" argument.

Usage (from the repo root):
    python src/paired_stats.py                 # every contrast; JSON + LaTeX tables
    python src/paired_stats.py --only e2_log   # selected contrasts, nothing written
    python src/paired_stats.py --check         # reconcile against the values in the notes
"""
import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import analysis_common as ac
import paths

SPEC_PATH = paths.CONFIGS / "analysis_contrasts.json"

# --check tolerances: the notes round to four decimals (means, CIs) and to two or three (p).
MEAN_TOLERANCE = 5e-5
P_TOLERANCE = 5e-3
# Before §44.9 the notes' intervals came from several throwaway procedures (a normal
# approximation in §36.1, numpy's RandomState in most of §39.4 and §41, others unrecorded). Every
# mean and count reproduces; the intervals differ by Monte-Carlo noise, which at 10,000 resamples
# stays within this bound. A larger gap is a real disagreement and fails --check.
CI_METHOD_TOLERANCE = 5e-3

GROUP_CAPTIONS = {
    "matrix": "Paired differences between matrix arms",
    "ablations": "Paired differences in the ablations",
    "agentic": "Paired differences in the agentic evaluations",
    # Re-run on the pinned corpus on 2026-10-27 (their old-corpus runs stay reachable through each
    # contrast's run_map, for --check against notes 11 §25.4 and §31.2).
    "decoding_hint": "Paired differences in the token-budget and structure-hint ablations",
    # The *_detail groups hold the rows the chapter does not discuss (reproductions, replicate
    # contrasts, post-hoc subsets and single draws); their tables go to the appendix.
    "ablations_detail": "Reproductions and further contrasts of the ablations",
    "agentic_detail": "Post-hoc subsets and single draws of the agentic contrasts",
}


# --------------------------------------------------------------------------------------------
# Statistics. Pure functions; tests/test_paired_stats.py pins each one.
# --------------------------------------------------------------------------------------------

def sign_test(better: int, worse: int) -> Optional[float]:
    """Exact two-sided sign test on the questions that changed (ties are dropped).

    Returns:
        The p-value, or None when nothing changed.
    """
    n = better + worse
    if n == 0:
        return None
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_ci(diffs: Sequence[float], resamples: int, seed: int,
                 confidence: float = 0.95) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling questions with replacement.

    The draw order is part of the result: `random.Random(seed)` with one `randrange(n)` per
    resampled question, a fresh generator per call. Changing it changes the fourth decimal of
    every interval in the notes.
    """
    n = len(diffs)
    if n == 0:
        raise ValueError("no paired questions")
    rng = random.Random(seed)
    means = np.empty(resamples)
    for i in range(resamples):
        means[i] = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
    tail = 100 * (1 - confidence) / 2
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def _z(probability: float) -> float:
    return statistics.NormalDist().inv_cdf(probability)


def min_detectable_effect(sd: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest paired mean difference n questions detect (two-sided alpha, given power)."""
    return (_z(1 - alpha / 2) + _z(power)) * sd / math.sqrt(n)


def questions_needed(sd: float, effect: float, alpha: float = 0.05,
                     power: float = 0.8) -> Optional[int]:
    """Questions needed to detect `effect` at the observed paired SD; None for effect 0."""
    if effect == 0:
        return None
    return math.ceil(((_z(1 - alpha / 2) + _z(power)) * sd / abs(effect)) ** 2)


def paired(per_condition: Dict[str, Dict[ac.EntryKey, float]],
           weights: Dict[str, float]) -> Dict[ac.EntryKey, float]:
    """Per-question weighted sum over conditions, on the questions every condition has.

    Raises:
        ValueError: if the conditions do not cover the same questions -- pairing would otherwise
            silently drop the questions one arm failed to record.
    """
    keys = None
    for name in weights:
        these = set(per_condition[name])
        if keys is None:
            keys = these
        elif these != keys:
            missing = sorted(keys ^ these)[:3]
            raise ValueError(f"conditions cover different questions, e.g. {missing}")
    return {key: sum(w * per_condition[name][key] for name, w in weights.items())
            for key in sorted(keys or ())}


# --------------------------------------------------------------------------------------------
# Contrasts.
# --------------------------------------------------------------------------------------------

class RunCache:
    """Loads each run once, and checks it against its summary.json the first time."""

    def __init__(self) -> None:
        self._runs: Dict[str, ac.Run] = {}

    def get(self, run_dir: str) -> ac.Run:
        if run_dir not in self._runs:
            run = ac.load_run(run_dir)
            ac.assert_reproduces_summary(run)
            self._runs[run_dir] = run
        return self._runs[run_dir]

    @property
    def runs(self) -> List[ac.Run]:
        return list(self._runs.values())


def _condition_records(condition: dict, run: ac.Run) -> List[dict]:
    select = condition.get("select") or {}
    records = [r for r in run.records if all(r.get(k) == v for k, v in select.items())]
    if not records:
        raise ValueError(f"condition {condition} selects no records in {run.path}")
    return records


def _map_run(run_dir: str, run_map: Optional[Dict[str, str]]) -> str:
    for old, new in (run_map or {}).items():
        if run_dir.startswith(old):
            return new + run_dir[len(old):]
    return run_dir


def evaluate(contrast: dict, conditions: Dict[str, dict], cache: RunCache, settings: dict,
             run_map: Optional[Dict[str, str]] = None,
             resamples: Optional[int] = None,
             evaluated_population: Optional[str] = None) -> dict:
    """Compute one contrast.

    Args:
        contrast: An entry of the spec's `contrasts`.
        conditions: The spec's `conditions`.
        cache: Loaded runs.
        settings: The spec's `bootstrap` and `power` blocks, merged.
        run_map: Run-path prefix substitutions (--check against the runs the notes used).
        resamples: Override of the resample count (--check against §39.4's 4,000).
        evaluated_population: Population of the execution sign test, if not the contrast's
            own (--check against §37, which counted executions over all 169).

    Returns:
        The result row: n, mean, CI, sign tests, identical predictions, resolution.
    """
    weights = {name: float(w) for name, w in contrast["terms"].items()}
    subset = contrast.get("subset") or {}
    population = contrast.get("population", "discriminating")
    evaluated_population = evaluated_population or population
    records_of = {}
    for name in weights:
        condition = conditions[name]
        run = cache.get(_map_run(condition["run"], run_map))
        records_of[name] = _condition_records(condition, run)

    # A subset conditioned on execution (§44.9's "88 questions evaluated in all four runs"): keep
    # the questions every record of the named conditions evaluated, across all draws. It selects
    # on an outcome, so the spec must mark it post hoc.
    keep = None
    if subset.get("evaluated_in_all"):
        for name in subset["evaluated_in_all"]:
            condition = conditions[name]
            records = _condition_records(condition, cache.get(_map_run(condition["run"], run_map)))
            outcomes: Dict[ac.EntryKey, List[float]] = {}
            for record in records:
                outcomes.setdefault(ac.entry_key(record), []).append(ac.evaluated(record))
            evaluated_everywhere = {key for key, values in outcomes.items() if all(values)}
            keep = evaluated_everywhere if keep is None else keep & evaluated_everywhere

    def per_question(metric: str, stratum: str) -> Dict[ac.EntryKey, float]:
        values = {name: ac.case_values(records_of[name], metric, stratum,
                                       exclude_logs=subset.get("exclude_logs", ()),
                                       draw=subset.get("draw"))
                  for name in weights}
        if keep is not None:
            values = {name: {k: v for k, v in vals.items() if k in keep}
                      for name, vals in values.items()}
        return paired(values, weights)

    f1 = per_question("strict_f1", population)
    diffs = list(f1.values())
    better = sum(1 for d in diffs if d > 0)
    worse = sum(1 for d in diffs if d < 0)
    evaluated = list(per_question("evaluated", evaluated_population).values())
    gained = sum(1 for d in evaluated if d > 0)
    lost = sum(1 for d in evaluated if d < 0)
    count = resamples or settings["resamples"]
    low, high = bootstrap_ci(diffs, count, settings["seed"], settings["confidence"])
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    mean = sum(diffs) / len(diffs)
    return {
        "id": contrast["id"],
        "label": contrast["label"],
        "section": contrast.get("section"),
        "group": contrast.get("group"),
        "corpus": contrast.get("corpus"),
        "population": population,
        "subset": subset or None,
        "post_hoc": bool(subset.get("post_hoc")),
        "terms": contrast["terms"],
        "runs": {name: _map_run(conditions[name]["run"], run_map) for name in weights},
        "n": len(diffs),
        "mean": mean,
        "ci": [low, high],
        "resamples": count,
        "seed": settings["seed"],
        "better": better,
        "worse": worse,
        "p": sign_test(better, worse),
        "evaluated_gained": gained,
        "evaluated_lost": lost,
        "evaluated_p": sign_test(gained, lost),
        "identical": _identical(records_of, weights),
        "paired_sd": sd,
        "min_detectable_effect": min_detectable_effect(sd, len(diffs), settings["alpha"],
                                                       settings["power"]),
        "questions_needed": questions_needed(sd, mean, settings["alpha"], settings["power"]),
    }


def _identical(records_of: Dict[str, List[dict]], weights: Dict[str, float]) -> Optional[int]:
    """Questions (all 169) with byte-identical predictions; two single-draw conditions only."""
    if len(weights) != 2:
        return None
    canon = {}
    for name in weights:
        by_entry = {}
        for record in records_of[name]:
            if record.get("repeat") not in (None, 1) or len(
                    {r.get("config_id") for r in records_of[name]}) != 1:
                return None
            by_entry[ac.entry_key(record)] = json.dumps(record.get("prediction"), sort_keys=True)
        canon[name] = by_entry
    first, second = (canon[name] for name in weights)
    return sum(1 for key, text in first.items() if second.get(key) == text)


# --------------------------------------------------------------------------------------------
# Reconciliation against the notes.
# --------------------------------------------------------------------------------------------

def reconcile(row: dict, expected: dict) -> Tuple[List[str], List[str]]:
    """Compare a computed row with the published figures.

    Returns:
        (failures, method_notes). Failures are disagreements in a count, a mean or a p-value, or
        an interval further off than `CI_METHOD_TOLERANCE`; any of them means the script and the
        notes measure different things. Method notes are interval differences within that bound,
        explained by the notes' differing bootstrap procedures.
    """
    problems, notes = [], []
    # A figure published to fewer than four decimals is compared at its own precision.
    tolerance = 0.5 * 10 ** -expected["digits"] if "digits" in expected else MEAN_TOLERANCE
    for key in ("better", "worse", "identical", "evaluated_gained", "evaluated_lost"):
        if key in expected and row.get(key) != expected[key]:
            problems.append(f"{key}: computed {row.get(key)}, notes {expected[key]}")
    if "mean" in expected and abs(row["mean"] - expected["mean"]) > tolerance:
        problems.append(f"mean: computed {row['mean']:+.4f}, notes {expected['mean']:+.4f}")
    if "ci" in expected and expected.get("ci_method") == "normal approximation":
        # The notes' interval was mean +- z * sd / sqrt(n): recompute it from the same paired
        # differences and require it to match exactly, instead of excusing the gap as noise.
        half = _z(0.975) * row["paired_sd"] / math.sqrt(row["n"])
        normal = (row["mean"] - half, row["mean"] + half)
        if all(abs(got - want) <= tolerance for got, want in zip(normal, expected["ci"])):
            notes.append(f"CI: the notes' normal approximation reproduces exactly "
                         f"({normal[0]:+.4f}, {normal[1]:+.4f}); the thesis reports the "
                         f"bootstrap ({row['ci'][0]:+.4f}, {row['ci'][1]:+.4f})")
        else:
            problems.append(f"CI: normal approximation gives ({normal[0]:+.4f}, "
                            f"{normal[1]:+.4f}), notes {tuple(expected['ci'])}")
    elif "ci" in expected:
        method = expected.get("ci_method", "an unrecorded bootstrap")
        for got, want, side in zip(row["ci"], expected["ci"], ("low", "high")):
            gap = abs(got - want)
            if gap <= tolerance:
                continue
            text = f"CI {side}: computed {got:+.4f}, notes {want:+.4f} (notes: {method})"
            (notes if gap <= CI_METHOD_TOLERANCE else problems).append(text)
    for key in ("p", "evaluated_p"):
        if key not in expected:
            continue
        got, want = row.get(key), expected[key]
        if isinstance(want, str) and want.startswith("<"):
            if got is None or got >= float(want[1:]):
                problems.append(f"{key}: computed {got}, notes {want}")
        elif got is None or abs(got - want) > P_TOLERANCE:
            problems.append(f"{key}: computed {got}, notes {want}")
    return problems, notes


# --------------------------------------------------------------------------------------------
# Output.
# --------------------------------------------------------------------------------------------

def tex_table(rows: List[dict]) -> str:
    """The booktabs tabularx for one group of contrasts; the label column wraps.

    The sign tests' p values stay in the JSON (`p`, `evaluated_p`) but not in the table: the
    author dropped them from the thesis (Ch. 3 review, 2026-11-11); the counts stay.
    """
    lines = [
        "  \\small",
        "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrcr@{}}",
        "    \\toprule",
        "    Contrast & \\makecell[r]{Better /\\\\worse} & $\\Delta$F1 & 95\\% CI & "
        "\\makecell[r]{Evaluated:\\\\up / down} \\\\",
        "    \\midrule",
    ]
    for index, row in enumerate(rows):
        # A rule between the notes sections keeps each evaluation's contrasts together.
        if index and row["section"] != rows[index - 1]["section"]:
            lines.append("    \\midrule")
        label = row["label"] + (" (post hoc)" if row["post_hoc"] else "")
        label += " (old corpus)" if row["corpus"] == "old" else ""
        ci = f"{ac.fmt(row['ci'][0], signed=True)}, {ac.fmt(row['ci'][1], signed=True)}"
        # A difference of differences has no "better" question, so its counts and sign tests
        # stay in the JSON and out of the table.
        if len(row["terms"]) > 2:
            counts = evaluated = "--"
        else:
            counts = f"{row['better']} / {row['worse']}"
            evaluated = f"{row['evaluated_gained']} / {row['evaluated_lost']}"
        lines.append(f"    {label} & {counts} & {ac.fmt(row['mean'], signed=True)} & "
                     f"{ci} & {evaluated} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    return "\n".join(lines)


def write_tables(rows: List[dict], settings: dict, runs: List[ac.Run]) -> List[Path]:
    """One table per group; the caption states population, resamples and the seed."""
    written = []
    for group, caption in GROUP_CAPTIONS.items():
        group_rows = [r for r in rows if r["group"] == group]
        if not group_rows:
            continue
        populations = {r["population"] for r in group_rows}
        if len(populations) != 1:
            raise ValueError(f"group {group} mixes populations {populations}")
        population = ac.POPULATION_LABELS[populations.pop()]
        # Two printed lines at most (supervisor, 2026-11-06): the method (draws averaged per
        # question, percentile bootstrap, resamples and seed, exact sign tests) is stated once,
        # in the thesis's statistical-analysis section.
        full_caption = (f"{caption} over {population}; the method is in "
                        f"\\cref{{subsec:eval:stats}}.")
        used = {p for r in group_rows for p in r["runs"].values()}
        group_runs = [run for run in runs if run.provenance["run"] in used]
        written.append(ac.write_tex(f"paired_{group}", tex_table(group_rows), full_caption,
                                    f"tab:paired:{group}", group_runs))
    return written


def resolution_table(rows: List[dict], settings: dict, runs: List[ac.Run]) -> Path:
    """`paired_resolution`: what each ablation contrast could detect, and what it would need.

    Two-condition contrasts of the `ablations` and `ablations_detail` groups. Reproductions (same
    configuration run twice) are left out: their spread is run-to-run noise, not an effect to power.
    """
    chosen = [r for r in rows if r["group"] in ("ablations", "ablations_detail")
              and len(r["terms"]) == 2 and "reproduction" not in r["label"]]
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrrr@{}}",
             "    \\toprule",
             "    Contrast & $\\Delta$F1 & \\makecell[r]{Paired\\\\SD}"
             " & \\makecell[r]{Detectable\\\\at $n$ = 135}"
             " & \\makecell[r]{Questions\\\\needed} \\\\",
             "    \\midrule"]
    for index, row in enumerate(chosen):
        if index and row["section"] != chosen[index - 1]["section"]:
            lines.append("    \\midrule")
        needed = row["questions_needed"]
        lines.append(f"    {row['label']} & {ac.fmt(row['mean'], signed=True)} & "
                     f"{ac.fmt(row['paired_sd'])} & $\\pm${ac.fmt(row['min_detectable_effect'])}"
                     f" & {'--' if needed is None else f'{needed:,}'} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    used = {p for r in chosen for p in r["runs"].values()}
    caption = (
        "What each ablation contrast can resolve at $n = 135$: the smallest detectable difference "
        f"($\\alpha = {settings['alpha']}$, power {settings['power']}) and the questions needed.")
    return ac.write_tex("paired_resolution", "\n".join(lines), caption, "tab:paired:resolution",
                        [run for run in runs if run.provenance["run"] in used])


def load_spec(path: Path = SPEC_PATH) -> dict:
    """The contrast spec, with `bootstrap` and `power` merged into one settings dict."""
    with open(path, encoding="utf-8") as handle:
        spec = json.load(handle)
    spec["settings"] = {**spec["bootstrap"], **spec["power"]}
    return spec


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", metavar="ID", help="Compute only these contrasts "
                        "(prints them; writes nothing).")
    parser.add_argument("--check", action="store_true", help="Reconcile every contrast with an "
                        "`expected` block against the published figures, on the runs the notes "
                        "used; exits 1 on any difference.")
    parser.add_argument("--spec", type=Path, default=SPEC_PATH, help="Contrast spec.")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    contrasts = spec["contrasts"]
    if args.only:
        unknown = set(args.only) - {c["id"] for c in contrasts}
        if unknown:
            parser.error(f"unknown contrast ids: {sorted(unknown)}")
        contrasts = [c for c in contrasts if c["id"] in args.only]
    cache = RunCache()

    if args.check:
        failures = method_only = 0
        for contrast in contrasts:
            expected = contrast.get("expected")
            if not expected:
                continue
            row = evaluate(contrast, spec["conditions"], cache, spec["settings"],
                           run_map=expected.get("run_map"),
                           resamples=expected.get("resamples"),
                           evaluated_population=expected.get("evaluated_population"))
            problems, notes = reconcile(row, expected)
            failures += bool(problems)
            method_only += bool(notes) and not problems
            status = "DIFF" if problems else ("CI  " if notes else "OK  ")
            print(f"{status} {contrast['id']:<20} (notes {contrast.get('section')})")
            for line in problems + notes:
                print(f"       {line}")
        print(f"\n{failures} contrast(s) differ from the notes; {method_only} agree except for "
              f"interval differences within {CI_METHOD_TOLERANCE} from the notes' bootstrap "
              f"procedure.")
        return 1 if failures else 0

    rows = [evaluate(c, spec["conditions"], cache, spec["settings"]) for c in contrasts]
    for row in rows:
        print(f"{row['id']:<20} n={row['n']:<4} {row['mean']:+.4f} "
              f"[{row['ci'][0]:+.4f}, {row['ci'][1]:+.4f}]  {row['better']}/{row['worse']}  "
              f"evaluated {row['evaluated_gained']}/{row['evaluated_lost']}")
    if args.only:
        return 0
    out = ac.write_json("paired_stats", {"settings": spec["settings"], "contrasts": rows},
                        cache.runs)
    tables = write_tables(rows, spec["settings"], cache.runs)
    tables.append(resolution_table(rows, spec["settings"], cache.runs))
    print(f"\nwrote {out}")
    for table in tables:
        print(f"wrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
