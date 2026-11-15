#!/usr/bin/env python3
"""How far the testset sits from what the models saw: contamination, vocabulary, constructs, shapes.

WHY
---
The thesis claims the testset is unseen in four senses, and each claim had been checked once by
throwaway code (Thesis/script/FLAGS.md NS-08):

  * no question and no gold query occurs in either training corpus (notes 11 §3, §6.7);
  * the logs' type vocabulary does not occur in the training corpora (§4, superseded by §6.7 and
    §45.2). This is also what decides IC-06: `eval_testset._PRE_DELEAK_SEEN_LOGS` still labels
    five logs "seen" from a list hard-coded before the corpus was de-leaked, and the
    familiar/unfamiliar comparison of Ch. 4 may only use that label with a measured overlap
    beside it;
  * every building block the testset uses is also trained (§6.7: "blocks the testset uses and
    training never shows: none");
  * the retrieval pool's trees rarely share a gold tree's shape and are flatter (§45.2).

This script makes each one a committed measurement. It reads the corpora and the testset; it
generates and executes nothing.

THE CORPORA
-----------
pool       `data/ocpq_dataset.json`: 20,000 generated rows, the few-shot retrieval pool and the
           source the fine-tuning corpus was converted from.
finetune   `data/finetune_ocpq.jsonl`: the chat rows the adapters were trained on (each pool row
           in up to three variants). Its question is the prompt's `Query:` line, its gold the
           assistant message, its type vocabulary the prompt's `ALLOWED EVENTS/OBJECTS` lines.

DEFINITIONS
-----------
contamination      a testset question equal to a corpus question (exactly, and after case and
                   whitespace folding), or a gold query equal to a corpus query as canonical JSON.
type overlap       per log, the distinct event and object type names its gold queries declare, and
                   how many occur among the corpus's type names (exactly, and case-folded).
building blocks    `tag_testset_bb.detect_bb_labels` on every gold and every pool query.
name-free shape    the query with every string replaced by one placeholder and every number kept:
                   structure, filter and constraint types, variable indices and bounds survive,
                   names do not. This is the definition that reproduces §45.2's 5 of 169.
nearest question   (`--with-embeddings`, needs the evaluation environment) the cosine of each test
                   question's MiniLM embedding to its nearest pool row, computed with the
                   pipeline's own `LLM.get_embedding` and `data/embeddings.npy`.

Usage (from the repo root):
    python src/corpus_overlap_audit.py                    # everything but the embeddings
    python src/corpus_overlap_audit.py --with-embeddings  # in an env with torch + transformers
"""
import argparse
import ast
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import analysis_common as ac
import paths
from tag_testset_bb import detect_bb_labels

EXPECTED_PATH = paths.CONFIGS / "analysis_expected.json"

# A block this many times rarer (by share of rows) in training than in the testset is flagged as
# under-trained -- the §6.8 criterion that found F14 and F12 starved.
RARITY_FACTOR = 10

_ALLOWED_LINE = re.compile(r"^ALLOWED (EVENTS|OBJECTS): (\[.*\])\s*$", re.MULTILINE)
_QUERY_LINE = re.compile(r"^Query: (.*)$", re.MULTILINE)


# --------------------------------------------------------------------------------------------
# Loading.
# --------------------------------------------------------------------------------------------

def _literal(value: Any) -> Any:
    """Pool fields are stored either as values or as their Python repr; return the value."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def load_testset() -> List[dict]:
    with open(paths.DATA / "testset.json", encoding="utf-8") as handle:
        return json.load(handle)


def load_pool() -> List[dict]:
    """Pool rows with `query`, `events`, `objects` parsed."""
    with open(paths.DATA / "ocpq_dataset.json", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        for key in ("query", "events", "objects"):
            row[key] = _literal(row.get(key))
    return rows


def load_finetune() -> List[dict]:
    """Fine-tuning rows as {question, query, events, objects}."""
    rows = []
    with open(paths.DATA / "finetune_ocpq.jsonl", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            answer = next(m["content"] for m in messages if m["role"] == "assistant")
            allowed = {kind: json.loads(names) for kind, names in _ALLOWED_LINE.findall(user)}
            question = _QUERY_LINE.findall(user)
            try:
                query = json.loads(answer)
            except ValueError:
                query = None
            rows.append({"question": question[-1].strip() if question else None,
                         "query": query,
                         "events": allowed.get("EVENTS", []),
                         "objects": allowed.get("OBJECTS", [])})
    return rows


# --------------------------------------------------------------------------------------------
# Measures. Pure functions; tests/test_corpus_overlap_audit.py pins each one.
# --------------------------------------------------------------------------------------------

def fold(text: Optional[str]) -> str:
    """Case- and whitespace-folded text, for the near-exact contamination check."""
    return " ".join((text or "").casefold().split())


def canonical(query: Any) -> Optional[str]:
    """A query as canonical JSON (sorted keys), or None if it is not a query."""
    if not isinstance(query, dict):
        return None
    return json.dumps(query, sort_keys=True, ensure_ascii=False)


def declared_types(query: Any) -> Tuple[Set[str], Set[str]]:
    """(event types, object types) declared by any box of a query."""
    events: Set[str] = set()
    objects: Set[str] = set()
    for node in (query or {}).get("nodes") or []:
        box = node.get("Box") if isinstance(node, dict) else None
        if not isinstance(box, list) or not box or not isinstance(box[0], dict):
            continue
        for names in (box[0].get("newEventVars") or {}).values():
            events.update(names)
        for names in (box[0].get("newObjectVars") or {}).values():
            objects.update(names)
    return events, objects


def name_free_shape(query: Any) -> str:
    """The query with every string replaced by one placeholder; numbers and structure kept."""
    def blank(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: blank(item) for key, item in value.items()}
        if isinstance(value, list):
            return [blank(item) for item in value]
        return "s" if isinstance(value, str) else value
    return json.dumps(blank(query), sort_keys=True)


def box_count(query: Any) -> int:
    return len((query or {}).get("nodes") or [])


def contamination(test: List[dict], corpus_questions: Iterable[Optional[str]],
                  corpus_queries: Iterable[Any]) -> dict:
    """Testset questions and gold queries that occur in a corpus."""
    exact = set(q for q in corpus_questions if q)
    folded = {fold(q) for q in exact}
    queries = {canonical(q) for q in corpus_queries} - {None}
    return {
        "nl_exact": sum(1 for t in test if t["nl_query"] in exact),
        "nl_folded": sum(1 for t in test if fold(t["nl_query"]) in folded),
        "gold_exact": sum(1 for t in test if canonical(t["ocpq_query"]) in queries),
    }


def type_overlap(test: List[dict], vocabulary: Set[str]) -> List[dict]:
    """Per log: distinct gold type names and how many occur in the vocabulary."""
    folded_vocabulary = {name.casefold() for name in vocabulary}
    per_log: Dict[str, Set[str]] = defaultdict(set)
    for entry in test:
        events, objects = declared_types(entry["ocpq_query"])
        per_log[os.path.basename(entry["ocel_path"])].update(events | objects)
    rows = []
    for log in sorted(per_log, key=lambda name: (-len(per_log[name]), name)):
        names = per_log[log]
        exact = sorted(n for n in names if n in vocabulary)
        folded = sorted(n for n in names if n.casefold() in folded_vocabulary)
        rows.append({"log": log, "types": len(names), "exact": len(exact),
                     "exact_share": len(exact) / len(names) if names else None,
                     "folded": len(folded),
                     "folded_share": len(folded) / len(names) if names else None,
                     "exact_names": exact, "folded_names": folded})
    return rows


def block_coverage(test_queries: List[Any], train_queries: List[Any],
                   rarity_factor: float = RARITY_FACTOR) -> dict:
    """Building blocks per corpus, the blocks training never shows, and the under-trained ones."""
    test_counts = Counter(b for q in test_queries for b in detect_bb_labels(q))
    train_counts = Counter(b for q in train_queries if isinstance(q, dict)
                           for b in detect_bb_labels(q))
    n_test, n_train = len(test_queries), sum(1 for q in train_queries if isinstance(q, dict))
    blocks = []
    for block in sorted(set(test_counts) | set(train_counts),
                        key=lambda b: (b[0], int(b[1:]) if b[1:].isdigit() else 0)):
        test_share = test_counts[block] / n_test if n_test else 0.0
        train_share = train_counts[block] / n_train if n_train else 0.0
        blocks.append({"block": block, "testset": test_counts[block], "testset_share": test_share,
                       "training": train_counts[block], "training_share": train_share,
                       "under_trained": bool(test_counts[block]) and (
                           train_share * rarity_factor < test_share)})
    return {
        "testset_entries": n_test,
        "training_rows": n_train,
        "blocks": blocks,
        "testset_uses_training_never_shows": sorted(
            b["block"] for b in blocks if b["testset"] and not b["training"]),
        "under_trained": [b["block"] for b in blocks if b["under_trained"]],
    }


def shape_stats(test_queries: List[Any], pool_queries: List[Any]) -> dict:
    """§45.2: gold shapes found in the pool, and how flat each side is."""
    pool_shapes = {name_free_shape(q) for q in pool_queries}
    return {
        "gold_shapes_in_pool": sum(1 for q in test_queries if name_free_shape(q) in pool_shapes),
        "gold_one_box_share": sum(1 for q in test_queries if box_count(q) == 1) / len(test_queries),
        "pool_one_box_share": sum(1 for q in pool_queries if box_count(q) == 1) / len(pool_queries),
        "gold_mean_boxes": statistics.mean(box_count(q) for q in test_queries),
        "pool_mean_boxes": statistics.mean(box_count(q) for q in pool_queries),
    }


def nearest_cosines(test: List[dict]) -> dict:
    """Nearest-pool-question cosines with the pipeline's own embedding (evaluation env only)."""
    import numpy as np
    import LLM  # torch + transformers; loads the pool, its embeddings and MiniLM at import

    if LLM.embedding_load_error:
        raise SystemExit(f"embeddings unavailable: {LLM.embedding_load_error}")
    pool = np.asarray(LLM.embeddings)
    best = [float(np.max(pool @ LLM.get_embedding(t["nl_query"]))) for t in test]
    return {
        "nearest_cosine_max": max(best),
        "nearest_cosine_median": float(np.median(best)),
        "nearest_cosine_p90": float(np.percentile(best, 90)),
        "nearest_cosine_at_least_090": sum(1 for b in best if b >= 0.90),
        "per_question": [{"nl_query": t["nl_query"], "nearest_cosine": b}
                         for t, b in zip(test, best)],
    }


# --------------------------------------------------------------------------------------------
# Reconciliation and output.
# --------------------------------------------------------------------------------------------

def reconcile(results: dict, expected: dict) -> List[str]:
    """Every published figure the measurement does not reproduce (rounded as published)."""
    problems = []
    for key, entry in expected.items():
        if key.startswith("_") or key not in results:
            continue
        got, want = results[key], entry["value"]
        if isinstance(want, float):
            digits = len(str(want).split(".")[1])
            same = round(got, digits) == want
        else:
            same = got == want
        if not same:
            problems.append(f"{key}: computed {got}, notes {want} ({entry['section']})")
    return problems


def overlap_table(rows: List[dict], pool_types: int) -> str:
    lines = ["  \\small",
             "  \\begin{tabular}{@{}lrrr@{}}",
             "    \\toprule",
             "    Log & Types in gold & \\makecell[r]{in training\\\\(exact)} & "
             "\\makecell[r]{in training\\\\(any case)} \\\\",
             "    \\midrule"]
    for row in rows:
        log = row["log"].replace("_", "\\_")
        lines.append(f"    \\texttt{{{log}}} & {row['types']} & {row['exact']} & {row['folded']} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return "\n".join(lines)


def coverage_table(coverage: dict) -> str:
    lines = ["  \\small",
             "  \\begin{tabular}{@{}lrrrr@{}}",
             "    \\toprule",
             "    Block & Testset & Share & Corpus & Share \\\\",
             "    \\midrule"]
    for row in coverage["blocks"]:
        lines.append(f"    {row['block']} & {row['testset']} & {ac.fmt_pct(row['testset_share'])} & "
                     f"{row['training']} & {ac.fmt_pct(row['training_share'])} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--with-embeddings", action="store_true",
                        help="Also compute nearest-question cosines (needs torch + transformers).")
    parser.add_argument("--no-write", action="store_true", help="Print only.")
    args = parser.parse_args(argv)

    test = load_testset()
    pool = load_pool()
    finetune = load_finetune()
    gold = [t["ocpq_query"] for t in test]
    pool_queries = [r["query"] for r in pool]

    pool_events = {e for r in pool for e in (r["events"] or [])}
    pool_objects = {o for r in pool for o in (r["objects"] or [])}
    finetune_vocabulary = {n for r in finetune for n in r["events"] + r["objects"]}
    results: Dict[str, Any] = {
        "pool_rows": len(pool),
        "finetune_rows": len(finetune),
        "pool_event_types": len(pool_events),
        "pool_object_types": len(pool_objects),
        "finetune_type_names": len(finetune_vocabulary),
    }
    pool_contamination = contamination(
        test, (q for r in pool for q in (r.get("nl"), r.get("nl_clean_1"), r.get("nl_clean_2"))),
        pool_queries)
    finetune_contamination = contamination(
        test, (r["question"] for r in finetune), (r["query"] for r in finetune))
    results["contamination"] = {"pool": pool_contamination, "finetune": finetune_contamination}
    results["contamination_nl"] = pool_contamination["nl_folded"] + finetune_contamination["nl_folded"]
    results["contamination_gold"] = (pool_contamination["gold_exact"]
                                     + finetune_contamination["gold_exact"])

    vocabulary = pool_events | pool_objects | finetune_vocabulary
    overlap = type_overlap(test, vocabulary)
    results["type_overlap"] = overlap
    results["logs_with_any_exact_type_overlap"] = sum(1 for r in overlap if r["exact"])
    results["logs_with_any_folded_type_overlap"] = sum(1 for r in overlap if r["folded"])

    coverage = block_coverage(gold, pool_queries)
    results["block_coverage"] = coverage
    results["blocks_testset_uses_training_never_shows"] = coverage[
        "testset_uses_training_never_shows"]
    results.update(shape_stats(gold, pool_queries))
    if args.with_embeddings:
        results.update(nearest_cosines(test))

    with open(EXPECTED_PATH, encoding="utf-8") as handle:
        expected = json.load(handle)["corpus_overlap_audit"]
    problems = reconcile(results, expected)

    print(f"corpora: pool {results['pool_rows']} rows ({results['pool_event_types']} event, "
          f"{results['pool_object_types']} object types), fine-tuning {results['finetune_rows']} rows")
    print(f"contamination: pool {pool_contamination}, fine-tuning {finetune_contamination}")
    print("type overlap per log (exact / case-folded):")
    for row in overlap:
        print(f"  {row['log']:<34} {row['types']:>3} types  {row['exact']:>2} / {row['folded']:>2}"
              f"  {row['folded_names'] if row['folded_names'] else ''}")
    print(f"blocks the testset uses and training never shows: "
          f"{coverage['testset_uses_training_never_shows'] or 'none'}; "
          f">{RARITY_FACTOR}x rarer in training: {coverage['under_trained'] or 'none'}")
    print(f"shapes: {results['gold_shapes_in_pool']}/169 gold shapes in the pool; one-box "
          f"{results['pool_one_box_share']:.3f} pool vs {results['gold_one_box_share']:.3f} gold; "
          f"mean boxes {results['pool_mean_boxes']:.2f} vs {results['gold_mean_boxes']:.2f}")
    if args.with_embeddings:
        print(f"nearest question: max {results['nearest_cosine_max']:.3f}, median "
              f"{results['nearest_cosine_median']:.3f}, p90 {results['nearest_cosine_p90']:.3f}, "
              f">= 0.90: {results['nearest_cosine_at_least_090']}")
    print(f"\n{len(problems)} difference(s) from the notes:")
    for problem in problems:
        print(f"  {problem}")
    if args.no_write:
        return 0
    name = "corpus_overlap_audit" + ("_embeddings" if args.with_embeddings else "")
    out = ac.write_json(name, {"results": results, "differences_from_notes": problems})
    tables = [
        ac.write_tex("corpus_type_overlap", overlap_table(overlap, len(vocabulary)),
                     "Event and object type names declared by each log's gold queries, and how "
                     "many occur among the training corpora's type names, exactly and ignoring "
                     "case.", "tab:corpus:overlap"),
        ac.write_tex("corpus_block_coverage", coverage_table(coverage),
                     "Building blocks in the testset's gold queries and in the synthetic corpus's "
                     "queries; shares are of entries and rows.", "tab:corpus:blocks"),
    ]
    print(f"\nwrote {out}")
    for table in tables:
        print(f"wrote {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
