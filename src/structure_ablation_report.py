#!/usr/bin/env python3
"""Score the structure ablation, and decide first whether it is interpretable at all.

WHY THIS EXISTS
---------------
The ablation hands the model part of the GOLD tree (an oracle condition) and asks whether correct
structure is what the model is missing. That question only has an answer if the model actually
ADOPTS the structure it was given. The 2026-08-30 smoke measured 18% compliance against a prose
hint -- below the 30% node agreement the same model reaches with no hint at all -- which would
have made a null result unreadable: "structure does not help" and "the model will not take
structural instruction" produce the same numbers.

So compliance is reported FIRST and gates everything below it. Under ~70% the rung's score is not
evidence about structure.

THE OTHER CONFOUND
------------------
Deeper trees select more, so the oracle rungs may hit the `max_result_file_gb` cap more often than
the baseline does. A refused export scores 0, which would penalise S1/S2 for a storage reason
rather than a capability one -- and bias the comparison toward "composition does not help", which
is the outcome the analysis already predicts. A prior-confirming bias is the kind worth measuring
rather than assuming away, so refusals are counted per rung and a matched re-score (dropping any
question refused in ANY rung, from EVERY rung) is reported beside the headline.

USAGE
    python src/structure_ablation_report.py --run eval_runs/structure_ablation/merged
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_testset import _read_json, _write_json  # noqa: E402

RUNG_ORDER = [
    "schema-allowed-entities-only",
    "oracle-skeleton-given",
    "oracle-skeleton-and-constraints-given",
]
RUNG_LABEL = {
    "schema-allowed-entities-only": "S0 baseline",
    "oracle-skeleton-given": "S1 + skeleton",
    "oracle-skeleton-and-constraints-given": "S2 + constraint types",
}
COMPLIANCE_FLOOR = 0.70


def _nodes(query):
    if not isinstance(query, dict):
        return None
    boxes = [n for n in (query.get("nodes") or []) if isinstance(n, dict) and n.get("Box")]
    return len(boxes) or None


def _arity(query):
    out = []
    for node in (query.get("nodes") or []) if isinstance(query, dict) else []:
        box = (node or {}).get("Box") or [{}]
        body = box[0] if isinstance(box[0], dict) else {}
        out.append((len(body.get("newEventVars") or {}), len(body.get("newObjectVars") or {})))
    return out


def _edges(query):
    out = []
    for edge in (query.get("edgeNames") or []) if isinstance(query, dict) else []:
        try:
            out.append((int(edge[0][0]), int(edge[0][1]), str(edge[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return sorted(out)


def _constraint_types(query):
    counts = collections.Counter()
    for node in (query.get("nodes") or []) if isinstance(query, dict) else []:
        box = (node or {}).get("Box") or [{}]
        body = box[0] if isinstance(box[0], dict) else {}
        for c in body.get("constraints") or []:
            if not isinstance(c, dict):
                continue
            kind = c.get("type")
            if kind == "Filter":
                inner = c.get("filter")
                kind = f"Filter:{inner.get('type')}" if isinstance(inner, dict) else "Filter"
            counts[kind] += 1
    return counts


def _f1(record):
    return ((record.get("similarity") or {}).get("combined") or {}).get("f1")


def _skill(f1, floor):
    if floor is None or f1 is None:
        return None
    if floor >= 1.0:                       # non-discriminating: undefined by construction
        return None
    return (f1 - floor) / (1 - floor) if f1 >= floor else (f1 - floor) / floor


def analyse(run_dir: Path):
    records = _read_json(run_dir / "predictions.json")
    by_rung = collections.defaultdict(list)
    for r in records:
        by_rung[r.get("prompt_strategy")].append(r)

    # A question is comparable only if every rung produced a record for it.
    per_rung_questions = {k: {r.get("nl_query") for r in v} for k, v in by_rung.items()}
    common = set.intersection(*per_rung_questions.values()) if per_rung_questions else set()
    refused_anywhere = {
        r.get("nl_query") for v in by_rung.values() for r in v
        if (r.get("metrics") or {}).get("result_too_large")
    }

    out = {"run": str(run_dir), "rungs": {}, "questions_total": len(common),
           "questions_refused_in_some_rung": len(refused_anywhere & common)}

    for rung in RUNG_ORDER:
        rows = [r for r in by_rung.get(rung, []) if r.get("nl_query") in common]
        if not rows:
            continue
        is_oracle = rung != "schema-allowed-entities-only"

        node_ok = arity_ok = edge_ok = ctype_ok = parsed = 0
        hinted = fallback = 0
        compliant, noncompliant = [], []
        for r in rows:
            if is_oracle:
                if r.get("structure_hint"):
                    hinted += 1
                elif r.get("structure_hint_level"):
                    fallback += 1
            gold, pred = r.get("gold_query"), r.get("prediction")
            if _nodes(pred) is None:
                continue
            parsed += 1
            shape_ok = _nodes(gold) == _nodes(pred)
            node_ok += shape_ok
            arity_ok += _arity(gold) == _arity(pred)
            edge_ok += _edges(gold) == _edges(pred)
            ctype_ok += _constraint_types(gold) == _constraint_types(pred)
            # The sub-analysis that survives low compliance: among the cases where the model DID
            # adopt the given shape, how well does it score? At 169 cases and ~36% compliance this
            # is ~60 per rung -- enough to read, and the causal counterpart of the observational
            # 0.3770 in notes 23.7. It is still a SELECTED subset (the model may comply on the
            # easier questions), so it is an upper estimate, not a clean causal number.
            if r.get("discriminating") is True and r.get("baseline_f1") is not None:
                (compliant if shape_ok else noncompliant).append(_f1(r) or 0.0)

        scored = [(_f1(r) or 0.0, r.get("baseline_f1"), r.get("discriminating")) for r in rows]
        disc = [(f, fl) for f, fl, d in scored if d is True and fl is not None]
        matched = [r for r in rows if r.get("nl_query") not in refused_anywhere]
        matched_disc = [(_f1(r) or 0.0, r.get("baseline_f1")) for r in matched
                        if r.get("discriminating") is True and r.get("baseline_f1") is not None]
        skills = [s for s in (_skill(f, fl) for f, fl in disc) if s is not None]

        out["rungs"][rung] = {
            "label": RUNG_LABEL[rung],
            "is_oracle": is_oracle,
            "n": len(rows),
            "hint_delivered": hinted if is_oracle else None,
            "hint_fallbacks": fallback if is_oracle else None,
            "parsed": parsed,
            "compliance_node_count": (node_ok / parsed) if parsed else None,
            "compliance_arity": (arity_ok / parsed) if parsed else None,
            "compliance_edges": (edge_ok / parsed) if parsed else None,
            "compliance_constraint_types": (ctype_ok / parsed) if parsed else None,
            "f1_all": statistics.mean(f for f, _, _ in scored),
            "f1_discriminating": statistics.mean(f for f, _ in disc) if disc else None,
            "skill_discriminating": statistics.mean(skills) if skills else None,
            "skill_n": len(skills),
            "exact_discriminating": (sum(1 for f, _ in disc if f >= 0.999) / len(disc))
                                    if disc else None,
            "execution_success_rate": sum(1 for r in rows if r.get("status") == "success") / len(rows),
            "result_too_large": sum(1 for r in rows
                                    if (r.get("metrics") or {}).get("result_too_large")),
            "f1_discriminating_matched": (statistics.mean(f for f, _ in matched_disc)
                                          if matched_disc else None),
            "matched_n": len(matched_disc),
            "mean_floor": statistics.mean(fl for _, fl in disc) if disc else None,
            "f1_when_shape_adopted": statistics.mean(compliant) if compliant else None,
            "n_shape_adopted": len(compliant),
            "f1_when_shape_not_adopted": statistics.mean(noncompliant) if noncompliant else None,
            "n_shape_not_adopted": len(noncompliant),
        }
    return out


def _report(summary):
    rungs = summary["rungs"]
    print(f"\n{'=' * 92}")
    print(f"STRUCTURE ABLATION -- {summary['questions_total']} questions common to all rungs")
    print(f"{'=' * 92}")

    print("\n1. HINT COMPLIANCE -- read this before any score below.\n"
          "   Did the model adopt the structure it was handed? If not, the scores say nothing\n"
          "   about whether structure helps.\n")
    print(f"   {'rung':<26}{'parsed':>8}{'node ct':>10}{'arity':>9}{'edges':>9}{'constr':>9}")
    blocked = []
    for rung in RUNG_ORDER:
        c = rungs.get(rung)
        if not c:
            continue
        pct = lambda v: "  --  " if v is None else f"{v * 100:5.0f}%"  # noqa: E731
        print(f"   {c['label']:<26}{c['parsed']:>8}{pct(c['compliance_node_count']):>10}"
              f"{pct(c['compliance_arity']):>9}{pct(c['compliance_edges']):>9}"
              f"{pct(c['compliance_constraint_types']):>9}")
        if c["is_oracle"] and (c["compliance_node_count"] or 0) < COMPLIANCE_FLOOR:
            blocked.append(c["label"])
        if c["is_oracle"] and c["hint_fallbacks"]:
            print(f"      WARNING: {c['hint_fallbacks']} case(s) got NO hint (malformed gold); "
                  f"exclude them before reporting.")

    if blocked:
        print(f"\n   *** HEADLINE NOT INTERPRETABLE: {', '.join(blocked)} below the "
              f"{COMPLIANCE_FLOOR:.0%} compliance floor. ***")
        print("   A FLAT result cannot be read as 'structure does not help' -- it is equally what\n"
              "   'the model will not take structural instruction' looks like. Read section 3\n"
              "   instead, which conditions on the cases where the shape WAS adopted, and treat\n"
              "   the compliance rate itself as a finding about the model.")
    else:
        print(f"\n   All oracle rungs clear the {COMPLIANCE_FLOOR:.0%} floor -- scores are "
              f"interpretable.")

    floor = next((c["mean_floor"] for c in rungs.values() if c["mean_floor"]), None)
    print(f"\n2. SCORES on the discriminating stratum"
          f"{f' (mean trivial floor {floor:.4f})' if floor else ''}\n")
    print(f"   {'rung':<26}{'F1':>9}{'skill':>9}{'exact':>9}{'exec':>9}{'too_large':>11}")
    for rung in RUNG_ORDER:
        c = rungs.get(rung)
        if not c:
            continue
        num = lambda v, f="{:.4f}": "  --  " if v is None else f.format(v)  # noqa: E731
        print(f"   {c['label']:<26}{num(c['f1_discriminating']):>9}"
              f"{num(c['skill_discriminating']):>9}"
              f"{num(c['exact_discriminating'], '{:.1%}'):>9}"
              f"{num(c['execution_success_rate'], '{:.1%}'):>9}{c['result_too_large']:>11}")

    refused = summary["questions_refused_in_some_rung"]
    if refused:
        print(f"\n   {refused} question(s) had an export refused in at least one rung. A refusal\n"
              "   scores 0, so an uneven count across rungs is a storage confound, not a\n"
              "   capability difference. Matched re-score, those questions dropped from ALL rungs:")
        for rung in RUNG_ORDER:
            c = rungs.get(rung)
            if c and c["f1_discriminating_matched"] is not None:
                print(f"      {c['label']:<26}{c['f1_discriminating_matched']:.4f}  "
                      f"(n={c['matched_n']})")
    else:
        print("\n   No export refusals in any rung -- the storage confound does not apply.")

    print("\n3. CONDITIONAL ON THE MODEL ADOPTING THE GIVEN SHAPE\n"
          "   The sub-analysis that survives low compliance. Selected, so an UPPER estimate:\n"
          "   the model may comply on the easier questions.\n")
    print(f"   {'rung':<26}{'shape adopted':>16}{'F1 |adopted':>13}{'F1 |not':>10}")
    for rung in RUNG_ORDER:
        c = rungs.get(rung)
        if not c:
            continue
        num = lambda v: "  --  " if v is None else f"{v:.4f}"  # noqa: E731
        print(f"   {c['label']:<26}{str(c['n_shape_adopted']) + '/' + str(c['n_shape_adopted'] + c['n_shape_not_adopted']):>16}"
              f"{num(c['f1_when_shape_adopted']):>13}{num(c['f1_when_shape_not_adopted']):>10}")
    if floor:
        print(f"\n   Trivial floor for comparison: {floor:.4f}")
        best_cond = max((c for c in rungs.values() if c["f1_when_shape_adopted"] is not None),
                        key=lambda c: c["f1_when_shape_adopted"], default=None)
        if best_cond and best_cond["n_shape_adopted"] >= 20:
            verdict = ("STILL BELOW the floor -- even when the correct shape is adopted, the "
                       "answer is not right.\n      Composition is not the missing ingredient."
                       if best_cond["f1_when_shape_adopted"] < floor else
                       "CLEARS the floor -- correct shape is sufficient on the cases where it "
                       "is adopted.")
            print(f"   -> {best_cond['label']} with the shape adopted: {verdict}")
        elif best_cond:
            print(f"   -> too few compliant cases (n={best_cond['n_shape_adopted']}) to read; "
                  f"needs the full 169-case run.")

    print("\n4. THE DECISION, against the floor rather than against S0\n")
    if blocked:
        print("   Headline deferred: compliance is too low for the unconditional scores to bear\n"
              "   on the question. Section 3 is what to read, alongside the compliance rate as a\n"
              "   result in its own right.")
    elif floor:
        best = max((c for c in rungs.values() if c["f1_discriminating"] is not None),
                   key=lambda c: c["f1_discriminating"])
        if best["f1_discriminating"] < floor:
            print(f"   Best rung ({best['label']}) scores {best['f1_discriminating']:.4f}, below "
                  f"the {floor:.4f} floor.\n"
                  "   -> The composition programme is bounded below the floor. Drop the training\n"
                  "      rebalance AND decomposition; the next move is grounding.")
        else:
            print(f"   {best['label']} clears the floor at {best['f1_discriminating']:.4f}.\n"
                  "   -> Structure is a live lever. See the rung ordering for which part of it.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="Merged structure-ablation run directory.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run_dir = args.run if args.run.is_absolute() else Path.cwd() / args.run
    if not (run_dir / "predictions.json").is_file():
        raise SystemExit(f"FATAL: {run_dir / 'predictions.json'} not found.")

    summary = analyse(run_dir)
    out = args.out or (run_dir / "structure_ablation.json")
    _write_json(out, summary)
    _report(summary)
    print(f"wrote {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
