#!/usr/bin/env python3
"""Majority-vote the N sampled queries per question by their RESULT SET.

WHY
---
The 8112-run matrix ran greedy at temperature 0 with one sample per case: 38.2% of generations
were runnable, 2.31% correct on the discriminating stratum. Self-consistency asks whether the
right answer is already being sampled and merely not selected -- draw N queries at temperature
> 0, execute them all, and return the result set that the most samples agree on. Correct queries
tend to agree with one another; wrong ones scatter. The vote happens in *result* space, not token
space, so two syntactically different queries that select the same situations count as one vote.

This reads a finished run. It generates nothing and re-executes nothing: every sample's result is
already in the run's content-addressed store.

WHAT IT REPORTS
---------------
single-sample   the mean over individual samples -- the fair baseline. NOT the greedy matrix
                number, which would confound the sampling temperature with the voting.
majority        F1 of the modal result set.
filtered        F1 of the modal result set after discarding empty and truncated results. With
                63% of executed best-config queries returning zero overlap and many of those
                empty, a plain majority can elect "returns nothing".
oracle          max F1 over the N samples. Free, and it upper-bounds ANY selection rule. If the
                oracle is high and the vote is low, selection is the bottleneck; if the oracle is
                also low, sampling never contained the right answer and no selector can help.

agreement       the diagnostic that decides what to do next. A modal cluster of 8/10 means the
                model is confidently and systematically wrong, and voting has nothing to work
                with -- evidence that the bottleneck is grounding, not decoding. 2/10 means the
                errors scatter and there is signal to exploit.

USAGE
    python src/selfconsistency.py --run eval_runs/selfconsistency/merged
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis_common as ac  # noqa: E402
import paired_stats as ps  # noqa: E402
import result_store  # noqa: E402
from eval_testset import _compute_similarity, _read_json, _write_json  # noqa: E402


def _situations_key(ids):
    """Canonical hash of a result set, order-independent.

    `load_with_meta` returns `(event_ids, object_ids, situations)`; only `situations` is hashed,
    matching what `combined` scores -- it preserves which entities co-occur inside one binding and
    whether that binding violated a constraint, which the flat entity sets discard.
    """
    if ids is None:
        return None
    situations = ids[2]
    canon = sorted(
        (tuple(sorted(ev)), tuple(sorted(ob)), bool(vi)) for ev, ob, vi in situations
    )
    return hashlib.sha256(
        json.dumps(canon, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _f1(record):
    return ((record.get("similarity") or {}).get("combined") or {}).get("f1")


def _elect(samples, skip_empty):
    """Return (winning_key, cluster_size, n_voters, was_tie).

    Only samples that produced a loadable result vote. A sample that never executed has no
    result set to agree with, so it is not a vote for "nothing" -- it is an abstention. It stays
    in the *case* denominator regardless, because failing to answer is still a wrong answer.

    Ties are broken by lowest repeat index and counted rather than hidden: a matrix where most
    cases tie is one where the vote is arbitrary, and that has to be visible in the output.
    """
    voters = [s for s in samples if s["key"] is not None]
    if skip_empty:
        voters = [s for s in voters if s["size"] > 0 and not s["truncated"]]
    if not voters:
        return None, 0, 0, False

    counts = Counter(s["key"] for s in voters)
    top = max(counts.values())
    tied = [k for k, c in counts.items() if c == top]
    if len(tied) > 1:
        first = {}
        for s in voters:
            first.setdefault(s["key"], s["repeat"])
        winner = min(tied, key=lambda k: first[k])
        return winner, top, len(voters), True
    return tied[0], top, len(voters), False


def analyse(run_dir: Path):
    records = _read_json(run_dir / "predictions.json")
    store_dir = run_dir / "result_store"
    if not store_dir.is_dir():
        raise SystemExit(
            f"FATAL: {store_dir} not found. The vote compares stored result sets, so the run must "
            f"have been produced with result_store enabled."
        )

    # Grouped by entry identity, not case_id: case_id restarts at 1 in every case-shard (see
    # _record_key), so grouping on it would merge two different questions from opposite halves.
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("nl_query"), r.get("config_id"))].append(r)

    per_condition = defaultdict(lambda: {
        "cases": 0, "single": [], "majority": [], "filtered": [], "oracle": [],
        "agreement": Counter(), "ties": 0, "no_voter_cases": 0, "unloadable_samples": 0,
    })
    per_case = []

    for (nl_query, config_id), samples in sorted(groups.items(), key=lambda kv: str(kv[0])):
        head = samples[0]
        ocel_path = head.get("ocel_path")
        gold_ids, _ = result_store.load_with_meta(store_dir, ocel_path, head.get("gold_query"))

        loaded = []
        for r in sorted(samples, key=lambda r: r.get("repeat") or 0):
            ids, skipped = result_store.load_with_meta(store_dir, ocel_path, r.get("prediction"))
            loaded.append({
                "repeat": r.get("repeat") or 0,
                "key": _situations_key(ids),
                "ids": ids,
                "size": 0 if ids is None else len(ids[2]),
                "truncated": bool(skipped) or bool((r.get("metrics") or {}).get("bindings_skipped")),
                "f1": _f1(r),
            })

        agg = per_condition[config_id]
        agg["cases"] += 1
        agg["unloadable_samples"] += sum(
            1 for s in loaded if s["key"] is None and s["f1"] is not None
        )

        # Baseline and ceiling come from the samples' own recorded scores, so they are computed
        # by exactly the same path as the matrix's headline metric.
        scored = [0.0 if s["f1"] is None else s["f1"] for s in loaded]
        agg["single"].append(sum(scored) / len(scored))
        agg["oracle"].append(max(scored))

        row = {"nl_query": nl_query, "config_id": config_id, "n_samples": len(loaded),
               "single_mean": round(sum(scored) / len(scored), 6),
               "oracle": round(max(scored), 6),
               # For the breakdowns (notes 22.2-22.7): the stratum, the log (to leave out the
               # logs the stale cluster corpus retyped, notes 35.2), the floors, every sample's
               # score, and the samples that executed but left no loadable result.
               "discriminating": head.get("discriminating"),
               "log": Path(str(ocel_path or "").replace("\\", "/")).name,
               "noop_floor": head.get("baseline_f1"),
               "trivial_floor": head.get("trivial_floor"),
               "sample_f1": [round(f, 6) for f in scored],
               "unloadable": sum(1 for s in loaded if s["key"] is None and s["f1"] is not None)}

        for rule, skip_empty in (("majority", False), ("filtered", True)):
            key, size, voters, tie = _elect(loaded, skip_empty)
            if key is None:
                agg[rule].append(0.0)
                if rule == "majority":
                    agg["no_voter_cases"] += 1
                row[rule] = 0.0
                row[f"{rule}_cluster"] = 0
                row[f"{rule}_voters"] = 0
                continue
            winner = next(s for s in loaded if s["key"] == key)
            sim = _compute_similarity(gold_ids, winner["ids"])
            f1 = ((sim or {}).get("combined") or {}).get("f1")
            f1 = 0.0 if f1 is None else f1
            agg[rule].append(f1)
            row[rule] = round(f1, 6)
            row[f"{rule}_cluster"] = size
            # Voter count, without which cluster size is ambiguous: "1" means both "the only
            # sample that executed" and "five samples that all disagreed", and those are
            # opposite situations. A case with one voter had no vote at all -- whatever gain it
            # shows measures execution success, not selection.
            row[f"{rule}_voters"] = voters
            if rule == "majority":
                agg["agreement"][f"{size}/{voters}"] += 1
                agg["ties"] += int(tie)
        per_case.append(row)

    summary = {"run": str(run_dir), "conditions": {}, "per_case": per_case}
    for config_id, a in per_condition.items():
        mean = lambda xs: (sum(xs) / len(xs)) if xs else None  # noqa: E731
        summary["conditions"][config_id] = {
            "cases": a["cases"],
            "single_sample_f1": mean(a["single"]),
            "majority_f1": mean(a["majority"]),
            "filtered_f1": mean(a["filtered"]),
            "oracle_f1": mean(a["oracle"]),
            "gain_majority": (mean(a["majority"]) - mean(a["single"])),
            "gain_filtered": (mean(a["filtered"]) - mean(a["single"])),
            "headroom_to_oracle": (mean(a["oracle"]) - mean(a["majority"])),
            "ties": a["ties"],
            "cases_with_no_voter": a["no_voter_cases"],
            "unloadable_samples": a["unloadable_samples"],
            "agreement_histogram": dict(sorted(a["agreement"].items())),
        }
    return summary


# The three logs the cluster held in a stale, untyped copy until 2026-09-12 (notes 11 §35.2). This
# run predates the fix, so its 56 cases on these logs were prompted with the old declarations; the
# breakdown is also reported without them, as the sensitivity check of §35.5 did for the arity run.
RETYPED_LOGS = ("renting_log_high.json", "ocel2-p2p.json", "container_logistics.json")
CONSENSUS_BINS = ((0.9, ">=90%"), (0.6, "60-90%"), (0.4, "40-60%"), (0.0, "<40%"))
VOTER_BINS = ((0, "0"), (1, "1"), (4, "2-4"), (9, "5-9"), (10 ** 9, "10"))


def _skill(f1, floor):
    """eval_testset._skill_score: two-sided normalisation against the case's floor."""
    if f1 is None or floor is None or floor >= 1.0:
        return None
    return (f1 - floor) / (1.0 - floor) if f1 >= floor else (f1 - floor) / floor


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def breakdown(cases, exclude_logs=()):
    """The notes-22 analyses for one condition's per-case rows.

    headline      F1 over every case (single sample, majority, oracle); exact matches and
                  per-case skill over the discriminating cases, against both the no-op floor the
                  notes used and the trivial floor `skill_score` uses since notes 27.
    consensus     cases with >= 2 voters, binned by modal cluster / voters (notes 22.5).
    voters        cases by number of voters (22.6), and the share of the vote's total gain that
                  comes from cases where no real vote could happen (<= 1 voter).
    bound         the majority F1 if every case holding an unloadable sample -- one that executed
                  but left no result to vote with -- were scored 0 (22.7).
    """
    cases = [c for c in cases if c["log"] not in set(exclude_logs)]
    disc = [c for c in cases if c["discriminating"]]
    out = {"cases": len(cases), "discriminating": len(disc)}
    out["f1"] = {"single": _mean(c["single_mean"] for c in cases),
                 "majority": _mean(c["majority"] for c in cases),
                 "oracle": _mean(c["oracle"] for c in cases)}
    out["f1_discriminating"] = {"single": _mean(c["single_mean"] for c in disc),
                                "majority": _mean(c["majority"] for c in disc),
                                "oracle": _mean(c["oracle"] for c in disc)}
    # "single" is one sample's expected outcome: the share of samples that are exact, the mean of
    # the samples' own skills. Notes 22.2 published two other quantities under that label, kept
    # here under their own names: exact = cases where ALL ten samples were exact (6.7%, against
    # 11.2% of samples), and skill of the case's MEAN F1 (-0.6108, against -0.6266).
    out["exact_discriminating"] = {
        "single": _mean(_mean(1.0 if f == 1.0 else 0.0 for f in c["sample_f1"]) for c in disc),
        "every_sample_as_in_notes": _mean(
            1.0 if all(f == 1.0 for f in c["sample_f1"]) else 0.0 for c in disc),
        "majority": _mean(1.0 if c["majority"] == 1.0 else 0.0 for c in disc),
        "oracle": _mean(1.0 if c["oracle"] == 1.0 else 0.0 for c in disc)}
    for floor in ("noop_floor", "trivial_floor"):
        out[f"skill_{floor}"] = {
            "single": _mean(_mean(_skill(f, c[floor]) for f in c["sample_f1"]) for c in disc),
            "case_mean_as_in_notes": _mean(_skill(c["single_mean"], c[floor]) for c in disc),
            "majority": _mean(_skill(c["majority"], c[floor]) for c in disc),
            "oracle": _mean(_skill(c["oracle"], c[floor]) for c in disc)}
    # Skill is undefined at floor 1.0, so its means run over fewer cases than F1's (134 of 135 on
    # the current testset, analysis_common.SKILL_QUESTIONS); the table labels the row with this.
    out["skill_cases_trivial_floor"] = sum(
        1 for c in disc if c["trivial_floor"] is not None and c["trivial_floor"] < 1.0)
    bins = {label: [] for _, label in CONSENSUS_BINS}
    for c in cases:
        if c["majority_voters"] < 2:
            continue
        share = c["majority_cluster"] / c["majority_voters"]
        label = next(label for low, label in CONSENSUS_BINS if share >= low)
        bins[label].append(c)
    out["consensus"] = {label: {"cases": len(group),
                                "majority_f1": _mean(c["majority"] for c in group),
                                "oracle_f1": _mean(c["oracle"] for c in group),
                                "exact": sum(1 for c in group if c["majority"] == 1.0)}
                        for label, group in bins.items()}
    voters = Counter(next(label for top, label in VOTER_BINS if c["majority_voters"] <= top)
                     for c in cases)
    out["voters"] = {label: voters.get(label, 0) for _, label in VOTER_BINS}
    gains = [(c["majority"] - c["single_mean"], c["majority_voters"]) for c in cases]
    total = sum(g for g, _ in gains)
    out["gain_share_from_cases_with_at_most_one_voter"] = (
        sum(g for g, n in gains if n <= 1) / total if total else None)
    out["unloadable_samples"] = sum(c["unloadable"] for c in cases)
    out["majority_f1_if_unloadable_cases_scored_0"] = _mean(
        0.0 if c["unloadable"] else c["majority"] for c in cases)
    # Paired per discriminating question, with the thesis bootstrap (paired_stats): does the vote
    # beat one sample, and how much does the oracle leave on the table above the vote?
    out["paired_discriminating"] = {
        name: _paired([c[b] - c[a] for c in disc])
        for name, a, b in (("majority_minus_single", "single_mean", "majority"),
                           ("oracle_minus_majority", "majority", "oracle"))}
    return out


def versus_greedy(cases: list, greedy: dict) -> dict:
    """Each selection rule minus one greedy query of the same condition, paired per question.

    Args:
        cases: One condition's per-case rows (discriminating ones are used).
        greedy: `{nl_query: strict F1}` of that condition's greedy run (temperature 0, one
            sample), e.g. its cell of the matrix.

    Returns:
        `{rule: _paired(...)}` for single (sample mean), majority and oracle.
    """
    disc = [c for c in cases if c["discriminating"]]
    missing = [c["nl_query"] for c in disc if c["nl_query"] not in greedy]
    if missing:
        raise SystemExit(f"FATAL: the greedy run lacks {len(missing)} question(s), e.g. "
                         f"{missing[0][:60]!r}")
    return {rule: _paired([c[key] - greedy[c["nl_query"]] for c in disc])
            for rule, key in (("single", "single_mean"), ("majority", "majority"),
                              ("oracle", "oracle"))}


def _paired(diffs: list) -> dict:
    """Mean paired difference, its percentile-bootstrap CI and the sign test (paired_stats)."""
    settings = ps.load_spec()["settings"]
    low, high = ps.bootstrap_ci(diffs, settings["resamples"], settings["seed"],
                                settings["confidence"])
    better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    return {"n": len(diffs), "mean": _mean(diffs), "ci": [low, high], "better": better,
            "worse": worse, "p": ps.sign_test(better, worse)}


# The two arms of the run, in the order the thesis presents them, with their table labels.
ARMS = (("llama31-8b-finetuned_schema-allowed-entities-only_schema-on",
         "Fine-tuned Llama 3.1 8B, allowed-entities-only"),
        ("llama31-8b-base_schema-chain-of-thought-plus_schema-on",
         "Base Llama 3.1 8B, chain-of-thought-plus"))
RULES = ("single", "majority", "oracle")


def _row(label: str, values: list) -> str:
    return f"    {label} & " + " & ".join(values) + " \\\\"


def write_tables(summary: dict, run_dir: Path) -> List[Path]:
    """Write `selfconsistency` (the three selection rules) and `selfconsistency_consensus`.

    Args:
        summary: The analysis output, with its `breakdown`.
        run_dir: The run it was computed from (its predictions sha256 goes in the header).

    Returns:
        The written paths.
    """
    digest = hashlib.sha256((run_dir / "predictions.json").read_bytes()).hexdigest()[:16]
    notes = [f"input: {ac.repo_relative(run_dir)} sha256 {digest}"]
    # A run checked against the pinned corpus says so in its summary (corpus_consistent, notes 11
    # §38.1). The 2026-08-30 run predates the check and read three stale logs (FLAGS TH-07).
    run_summary = ac._read_json(run_dir / "summary.json")
    old_corpus = run_summary.get("corpus_consistent") is not True
    tag = " (old corpus)" if old_corpus else ""
    if old_corpus:
        notes.append("old corpus: the run predates the 2026-09-12 corpus fix (FLAGS TH-07)")
    lines = ["  \\small",
             "  \\begin{tabularx}{\\textwidth}{@{}>{\\raggedright\\arraybackslash}Xrrr@{}}",
             "    \\toprule",
             "    & \\makecell[r]{Single\\\\sample} & \\makecell[r]{Majority\\\\vote}"
             " & \\makecell[r]{Best-of-10\\\\oracle} \\\\"]
    for config_id, title in ARMS:
        every = summary["breakdown"][config_id]["all_cases"]
        kept = summary["breakdown"][config_id]["without_retyped_logs"]
        paired = every["paired_discriminating"]
        gains = [paired["majority_minus_single"], paired["oracle_minus_majority"]]
        lines += ["    \\midrule", f"    \\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\",
                  _row(f"F1, {every['discriminating']} discriminating",
                       [ac.fmt(every["f1_discriminating"][r]) for r in RULES]),
                  _row(f"F1, all {every['cases']}", [ac.fmt(every["f1"][r]) for r in RULES])]
        if old_corpus:
            lines.append(_row(f"F1, the {kept['cases']} not on a retyped log",
                              [ac.fmt(kept["f1"][r]) for r in RULES]))
        lines += [_row("Exact, discriminating",
                       [ac.fmt_pct(every["exact_discriminating"][r]) for r in RULES]),
                  _row(f"Skill, {every['skill_cases_trivial_floor']} discriminating",
                       [ac.fmt(every["skill_trivial_floor"][r], 3) for r in RULES]),
                  _row("Gain over the column to the left",
                       ["--"] + [ac.fmt(g["mean"], signed=True) for g in gains]),
                  _row("\\quad 95\\% CI", ["--"] + [f"{ac.fmt(g['ci'][0], signed=True)}, "
                                                    f"{ac.fmt(g['ci'][1], signed=True)}"
                                                    for g in gains]),
                  _row("\\quad questions better / worse",
                       ["--"] + [f"{g['better']} / {g['worse']}" for g in gains])]
        greedy = every.get("paired_vs_greedy")
        if greedy:
            lines += [_row("Gain over one greedy query",
                           [ac.fmt(greedy[r]["mean"], signed=True) for r in RULES]),
                      _row("\\quad 95\\% CI", [f"{ac.fmt(greedy[r]['ci'][0], signed=True)}, "
                                               f"{ac.fmt(greedy[r]['ci'][1], signed=True)}"
                                               for r in RULES]),
                      _row("\\quad questions better / worse",
                           [f"{greedy[r]['better']} / {greedy[r]['worse']}" for r in RULES])]
    lines += ["    \\bottomrule", "  \\end{tabularx}"]
    written = [ac.write_tex(
        "selfconsistency", "\n".join(lines),
        f"Self-consistency{tag}: ten samples per question at temperature 0.7, scored as one "
        "sample, as a majority vote and as the best of the ten (oracle).",
        "tab:abl:selfcons", notes=notes)]

    config_id = ARMS[0][0]
    lines = ["  \\small",
             "  \\begin{tabular}{@{}lrrrr@{}}",
             "    \\toprule",
             "    \\makecell[l]{Samples agreeing\\\\with the vote} & Questions"
             " & \\makecell[r]{Majority\\\\F1} & \\makecell[r]{Oracle\\\\F1}"
             " & \\makecell[r]{Exact\\\\votes} \\\\",
             "    \\midrule"]
    for label, group in summary["breakdown"][config_id]["all_cases"]["consensus"].items():
        cell = label.replace(">=", "$\\geq$").replace("<", "$<$").replace("%", "\\%")
        cell = cell.replace("-", "--")
        lines.append(_row(cell, [str(group["cases"]), ac.fmt(group["majority_f1"]),
                                 ac.fmt(group["oracle_f1"]), str(group["exact"])]))
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    voters = summary["breakdown"][config_id]["all_cases"]["voters"]
    written.append(ac.write_tex(
        "selfconsistency_consensus", "\n".join(lines),
        f"Agreement and correctness under self-consistency{tag} (fine-tuned Llama 3.1 8B): "
        "questions binned by the share of samples that returned the winning result set.",
        "tab:abl:selfcons:consensus",
        notes=[*notes, "questions with at least two samples that returned a result, out of all "
               f"169; on {voters['0']} no sample returned a result, on {voters['1']} only one"]))
    return written


def _report(summary):
    for config_id, c in summary["conditions"].items():
        print(f"\n{'=' * 78}\n{config_id}\n{'=' * 78}")
        print(f"  cases {c['cases']}   ties {c['ties']}   "
              f"cases with no voter {c['cases_with_no_voter']}   "
              f"unloadable samples {c['unloadable_samples']}")
        print(f"\n  {'single sample (baseline)':<30}{c['single_sample_f1']:>9.4f}")
        print(f"  {'majority vote':<30}{c['majority_f1']:>9.4f}"
              f"   {c['gain_majority']:+.4f}")
        print(f"  {'majority, empties filtered':<30}{c['filtered_f1']:>9.4f}"
              f"   {c['gain_filtered']:+.4f}")
        print(f"  {'oracle best-of-N (ceiling)':<30}{c['oracle_f1']:>9.4f}"
              f"   headroom {c['headroom_to_oracle']:+.4f}")
        print("\n  agreement -- modal cluster size / voters:")
        hist = c["agreement_histogram"]
        total = sum(hist.values()) or 1
        for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:10]:
            print(f"    {k:>8}  {v:>4}  {v/total*100:>5.1f}%  {'#' * round(v/total*40)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="A finished run directory containing predictions.json and result_store/.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Where to write selfconsistency.json (default: inside --run).")
    ap.add_argument("--tables", action="store_true",
                    help="Also write the thesis tables selfconsistency{,_consensus}.tex.")
    ap.add_argument("--greedy", type=Path, default=None,
                    help="A greedy run holding the same conditions (e.g. the matrix): each rule "
                         "is also paired against its one greedy query.")
    args = ap.parse_args()

    run_dir = args.run if args.run.is_absolute() else Path.cwd() / args.run
    if not (run_dir / "predictions.json").is_file():
        raise SystemExit(f"FATAL: {run_dir / 'predictions.json'} not found.")

    summary = analyse(run_dir)
    # Runs older than notes 27 carry no trivial floor; take it from the current audit, by question.
    audit_path = Path(__file__).resolve().parents[1] / "eval_runs" / "testset_discriminability.json"
    if audit_path.exists():
        floors = {e["nl_query"]: e.get("trivial_floor")
                  for e in _read_json(audit_path)["entries"]}
        for row in summary["per_case"]:
            if row.get("trivial_floor") is None:
                row["trivial_floor"] = floors.get(row["nl_query"])
    summary["breakdown"] = {}
    for config_id in summary["conditions"]:
        rows = [c for c in summary["per_case"] if c["config_id"] == config_id]
        summary["breakdown"][config_id] = {
            "all_cases": breakdown(rows),
            "without_retyped_logs": breakdown(rows, RETYPED_LOGS),
        }
        if args.greedy:
            greedy = {r["nl_query"]: ac.strict_f1(r) for r in ac.load_run(args.greedy).records
                      if r.get("config_id") == config_id}
            summary["breakdown"][config_id]["all_cases"]["paired_vs_greedy"] = versus_greedy(
                rows, greedy)
    if args.greedy:
        summary["greedy_run"] = ac.repo_relative(ac.resolve_run(args.greedy))
    out = args.out or (run_dir / "selfconsistency.json")
    _write_json(out, summary)
    _report(summary)
    print(f"\nwrote {out}")
    if args.tables:
        for path in write_tables(summary, run_dir):
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
