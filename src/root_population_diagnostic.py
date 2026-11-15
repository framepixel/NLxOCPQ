#!/usr/bin/env python3
"""Why the model's queries return the wrong rows: a decomposition of `wrong_population`.

`wrong_population` is the largest failure class after `schema_invalid` -- 748 records, 23.8% of
diverging predictions in the self-consistency run. Notes 26 sharpened it: the model scores 0.2935
on selecting the right rows against a trivial floor of 0.8881, three times below a do-nothing
query. Two facts make that hard to explain:

  * 95 of the 169 gold queries have NO root filters at all, so their population is fixed entirely
    by which variable types the root box declares; and
  * notes 23.3 says the model already gets ~80% of the entity types right.

Those do not fit together, and nothing said why. This script answers it by comparing gold's root
box with the prediction's, feature by feature, on every record whose population differs.

**It classifies; it does not score.** Every number it reports is a count of records falling into a
structural class, and the F1 values it prints come from the record as `eval_testset.py` wrote them.
No metric logic lives here (CLAUDE.md 2.5).

Two things it is careful about, both learned from a prototype that got them wrong:

  * `link_mismatch` keys a filter on its OPERANDS as well as its type -- `(type, object, event,
    other_object, qualifier)`. Keying on the type alone reported filters as identical when they
    linked entirely different variables, which inflated the residual class to 12.8%.
  * `arity_mismatch` overlaps the notes-26 metric artefact by construction: an extra root variable
    changes every situation tuple whatever else is right. So that class additionally reports how
    many of its records are REPAIRED by kind-projection, which separates "bound an extra column"
    from "selected different things".

Usage:
    python root_population_diagnostic.py --run_dir eval_runs/<run_id>
                                         [--store eval_runs/<run_id>/result_store]
                                         [--out root_population_report.json]
"""

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import result_store
from eval_testset import root_node_index

# First match wins, so the order is the claim: arity subsumes types, types subsume links, and
# anything surviving all four is not explained by the root box at all.
CLASS_ORDER = [
    "arity_mismatch",
    "type_mismatch",
    "link_mismatch",
    "size_filter_mismatch",
    "residual",
]
CLASS_MEANS = {
    "arity_mismatch": "declared a different NUMBER of root variables",
    "type_mismatch": "same variable count, different declared TYPES",
    "link_mismatch": "same variables, different root FILTERS (type, operands or qualifier)",
    "size_filter_mismatch": "same filters, different root sizeFilters",
    "residual": "root box matches on every feature, population still differs",
}


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def root_box(query: Any) -> Optional[dict]:
    """The root box's data dict, or None when the query has no readable root.

    Located with `root_node_index`, never assumed to be index 0: a valid tree may list a child
    before its parent, and the whole point of this script is that the root decides the population.
    """
    if not isinstance(query, dict):
        return None
    nodes = query.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return None
    index = root_node_index(query)
    if index >= len(nodes):
        return None
    node = nodes[index]
    box = node.get("Box") if isinstance(node, dict) else None
    if not isinstance(box, list) or not box or not isinstance(box[0], dict):
        return None
    return box[0]


def _types(var_map: Any) -> tuple:
    """The declared entity types of one variable map, order-insensitive.

    Which slot a type landed in is the query author's arbitrary choice -- the same reasoning that
    makes `_binding_entity_indices` sort by variable index rather than compare slot identity -- so
    the comparison is over the multiset of types, sorted.
    """
    if not isinstance(var_map, dict):
        return ()
    out = []
    for value in var_map.values():
        if isinstance(value, list):
            out.append(tuple(sorted(str(v) for v in value)))
        else:
            out.append((str(value),))
    return tuple(sorted(out))


def _strip_nulls(node: Any) -> Any:
    """Drop `None`-valued keys, recursively. An absent optional and an explicit null are the same.

    Applies to nested objects too: a `value_filter` of `{"type":"Integer","min":41,"max":null}` and
    one of `{"type":"Integer","min":41}` are the same filter, and gold and the model disagree about
    which to write.
    """
    if isinstance(node, dict):
        return {k: _strip_nulls(v) for k, v in node.items() if v is not None}
    if isinstance(node, list):
        return [_strip_nulls(v) for v in node]
    return node


def _filter_key(query_filter: Any) -> tuple:
    """A filter keyed on what it LINKS, not merely on what kind of filter it is.

    The whole filter is the key, canonicalised, rather than an allowlist of fields. Two earlier
    versions of this function enumerated the fields they thought mattered and both were wrong:
    the first omitted the operands, so an `O2E` joining object 0 to event 0 keyed the same as one
    joining object 1 to event 0; the second guessed `attribute`/`value` for the attribute filters
    when `src/ocpq_schema.py` actually declares `attribute_name` and a nested `value_filter`, so
    an `Integer{min: 41}` and a `String{is_in: [...]}` on the same attribute keyed as identical.
    Both mistakes sent real, explained differences into `residual` -- the one class that means
    "the root box does not explain this" and so must never absorb causes it does explain.

    Keying on the whole dict is exact by construction and cannot drift from the schema. Almost
    nothing is normalised away: an `Integer` bound and a `String` membership test really are
    different filters selecting different rows, so they must key differently.

    Two things ARE normalised, because they are serialisation and not meaning:

    * key ORDER, since gold and the model write the same filter's fields in different orders;
    * `None`-valued keys, since Pydantic treats an absent optional and an explicitly-null one
      identically. Without this, gold's `{"type":"O2E","object":0,"event":0}` and the model's
      `{...,"filterLabel":null}` -- the same filter -- keyed as different, and 4 of the 31
      `link_mismatch` cases at N2 (2 of 22 at N1) were this artefact. That inflated the class by
      13%, and the oracle computed on the inflated number is what nearly sent a week into
      grounding. See notes 34.
    """
    if not isinstance(query_filter, dict):
        return ("<unreadable>",)
    return (json.dumps(_strip_nulls(query_filter), sort_keys=True, default=str),)


def _filters(box: dict) -> tuple:
    return tuple(sorted((_filter_key(f) for f in (box.get("filters") or [])), key=repr))


def _size_filters(box: dict) -> tuple:
    """Root sizeFilters, keyed whole for the same reason `_filter_key` is."""
    return tuple(sorted((_filter_key(f) for f in (box.get("sizeFilters") or [])), key=repr))


def classify(gold_query: Any, prediction: Any) -> str:
    """Which root-box feature first explains a population difference. First match wins."""
    gold_box, pred_box = root_box(gold_query), root_box(prediction)
    if gold_box is None or pred_box is None:
        return "unreadable_root"

    gold_events, gold_objects = _types(gold_box.get("newEventVars")), _types(
        gold_box.get("newObjectVars"))
    pred_events, pred_objects = _types(pred_box.get("newEventVars")), _types(
        pred_box.get("newObjectVars"))

    if (len(gold_events), len(gold_objects)) != (len(pred_events), len(pred_objects)):
        return "arity_mismatch"
    if (gold_events, gold_objects) != (pred_events, pred_objects):
        return "type_mismatch"
    if _filters(gold_box) != _filters(pred_box):
        return "link_mismatch"
    if _size_filters(gold_box) != _size_filters(pred_box):
        return "size_filter_mismatch"
    return "residual"


# Filters that restrict by an attribute value (or a CEL expression over attributes), as opposed to
# the structural O2E / O2O / TimeBetweenEvents / NotEqual links between variables.
ATTRIBUTE_FILTERS = ("EventAttributeValueFilter", "ObjectAttributeValueFilter", "BasicFilterCEL")
LINK_SUBCLASS_ORDER = ["placement", "attribute", "value_type", "bounds", "structural_link"]
LINK_SUBCLASS_MEANS = {
    "placement": "gold's root restricts no attribute; the prediction adds an attribute filter there",
    "attribute": "an attribute filter is missing, extra, or on a different attribute",
    "value_type": "same attribute, different value type (e.g. Boolean against Integer)",
    "bounds": "same attribute and value type, different bounds or values",
    "structural_link": "attribute filters agree; an O2E/O2O/time/inequality link differs",
}


def _attribute_key(query_filter: dict) -> tuple:
    name = "cel" if query_filter.get("type") == "BasicFilterCEL" else query_filter.get(
        "attribute_name")
    return query_filter.get("type"), name


def link_subclass(gold_query: Any, prediction: Any) -> str:
    """Why two root boxes' filters differ, for a `link_mismatch` record. First match wins.

    Notes 11 §36.3 decomposed N2's 26 `link_mismatch` cases by hand (10 / 7 / 6 / 2 / 1). These
    rules make that decomposition reproducible. They reproduce it to within the cases that have
    two causes at once, where a hand reading picked one and the rules take the first in
    `LINK_SUBCLASS_ORDER` (N2: 11 / 7 / 6 / 1 / 1).
    """
    gold_filters = (root_box(gold_query) or {}).get("filters") or []
    pred_filters = (root_box(prediction) or {}).get("filters") or []
    gold_attrs = {_attribute_key(f): f for f in gold_filters
                  if isinstance(f, dict) and f.get("type") in ATTRIBUTE_FILTERS}
    pred_attrs = {_attribute_key(f): f for f in pred_filters
                  if isinstance(f, dict) and f.get("type") in ATTRIBUTE_FILTERS}
    if not gold_filters and pred_filters:
        return "placement"
    if set(gold_attrs) != set(pred_attrs):
        return "attribute"
    for key, gold_filter in gold_attrs.items():
        gold_value = gold_filter.get("value_filter") or {}
        pred_value = pred_attrs[key].get("value_filter") or {}
        if gold_value.get("type") != pred_value.get("type"):
            return "value_type"
    for key, gold_filter in gold_attrs.items():
        if _filter_key(gold_filter) != _filter_key(pred_attrs[key]):
            return "bounds"
    return "structural_link"


def clearing_oracle(rows: list, max_classes: int = 2) -> dict:
    """Mean strict F1 if every record of the chosen classes were answered perfectly.

    Args:
        rows: (class, strict F1) per record of one stratum; the class is `no_result`,
            `population_exact` or a `CLASS_ORDER` label.
        max_classes: Largest combination of classes to clear.

    Returns:
        {"a+b": mean F1}, highest first -- notes 11 §36.3's table. It bounds what fixing those
        failure classes could buy; it is not a prediction of any fix.
    """
    classes = sorted({label for label, _ in rows})
    oracle = {}
    for size in range(1, max_classes + 1):
        for combo in combinations(classes, size):
            oracle["+".join(combo)] = sum(1.0 if label in combo else f1
                                          for label, f1 in rows) / len(rows)
    return dict(sorted(oracle.items(), key=lambda item: -item[1]))


def _population(situations: Any) -> Optional[set]:
    """The row set with the violation flag dropped: which rows, not which are flagged."""
    if situations is None:
        return None
    return {(events, objects) for events, objects, _violated in situations}


def _kind_projected_equal(gold_situations: set, pred_situations: set) -> bool:
    """Whether blanking the entity kinds gold never binds makes the two populations agree.

    This is the notes-26 projection applied to the population question. Where it returns True the
    prediction picked out the right things and merely bound an extra column; where it returns
    False the prediction is about something else.
    """
    keep_events = any(events for events, _objects, _violated in gold_situations)
    keep_objects = any(objects for _events, objects, _violated in gold_situations)
    if not keep_events and not keep_objects:
        return False

    def project(situation):
        events, objects, _violated = situation
        return (events if keep_events else (), objects if keep_objects else ())

    return {project(s) for s in gold_situations} == {project(s) for s in pred_situations}


def analyse(run_dir: Path, store_dir: Path) -> dict:
    """Classify every record of a finished run. Returns the report written to disk."""
    records = _read_json(str(run_dir / "predictions.json"))

    by_config = defaultdict(lambda: {
        "classes": Counter(),
        "link_subclasses": Counter(),
        "arity_repaired_by_kind_projection": 0,
        "population_exact": 0,
        "no_result": 0,
        "n": 0,
        "discriminating_rows": [],
    })
    gold_cache: dict = {}

    for record in records:
        config_id = record.get("config_id", "<none>")
        bucket = by_config[config_id]
        bucket["n"] += 1

        ocel_path = record.get("ocel_path")
        gold_query = record.get("gold_query")
        key = (record.get("nl_query"), ocel_path)
        if key not in gold_cache:
            loaded = result_store.load(store_dir, ocel_path, gold_query) if gold_query else None
            gold_cache[key] = loaded[2] if loaded else None
        gold_situations = gold_cache[key]

        prediction = record.get("prediction")
        loaded = result_store.load(store_dir, ocel_path, prediction) if prediction else None
        pred_situations = loaded[2] if loaded else None

        # The record's strict F1 as eval_testset wrote it, for the clearing oracle.
        strict_f1 = ((record.get("similarity") or {}).get("combined") or {}).get("f1") or 0.0

        # A prediction with no stored result cannot be classified here at all -- it never parsed,
        # never executed, or its blob is absent. Counted, never guessed at.
        if gold_situations is None or pred_situations is None:
            bucket["no_result"] += 1
            label = "no_result"
        elif _population(gold_situations) == _population(pred_situations):
            bucket["population_exact"] += 1
            label = "population_exact"
        else:
            label = classify(gold_query, prediction)
            bucket["classes"][label] += 1
            if label == "arity_mismatch" and _kind_projected_equal(gold_situations,
                                                                   pred_situations):
                bucket["arity_repaired_by_kind_projection"] += 1
            if label == "link_mismatch":
                bucket["link_subclasses"][link_subclass(gold_query, prediction)] += 1
        if record.get("discriminating") is True:
            bucket["discriminating_rows"].append((label, strict_f1))

    report = {"run_dir": str(run_dir), "store_dir": str(store_dir),
              "class_meanings": CLASS_MEANS, "by_config": {}}
    for config_id, bucket in by_config.items():
        scoreable = bucket["n"] - bucket["no_result"]
        wrong = sum(bucket["classes"].values())
        report["by_config"][config_id] = {
            "n": bucket["n"],
            "no_result": bucket["no_result"],
            "scoreable": scoreable,
            "population_exact": bucket["population_exact"],
            "population_wrong": wrong,
            "classes": {name: bucket["classes"].get(name, 0)
                        for name in CLASS_ORDER + ["unreadable_root"]},
            "class_share_of_wrong": {
                name: (bucket["classes"].get(name, 0) / wrong) if wrong else None
                for name in CLASS_ORDER + ["unreadable_root"]
            },
            "arity_repaired_by_kind_projection": bucket["arity_repaired_by_kind_projection"],
            "link_subclasses": {name: bucket["link_subclasses"].get(name, 0)
                                for name in LINK_SUBCLASS_ORDER},
            "clearing_oracle_discriminating": (clearing_oracle(bucket["discriminating_rows"])
                                               if bucket["discriminating_rows"] else {}),
        }
    report["link_subclass_meanings"] = LINK_SUBCLASS_MEANS
    return report


def _print_report(report: dict) -> None:
    for config_id, block in report["by_config"].items():
        print(f"\n{config_id}")
        print(f"   {block['n']} records | {block['no_result']} with no stored result | "
              f"{block['scoreable']} scoreable")
        print(f"   population exact: {block['population_exact']}  "
              f"wrong: {block['population_wrong']}")
        if not block["population_wrong"]:
            continue
        print(f"   {'class':24s}{'n':>7}{'share of wrong':>17}")
        for name in CLASS_ORDER + ["unreadable_root"]:
            count = block["classes"][name]
            if not count:
                continue
            share = block["class_share_of_wrong"][name]
            print(f"   {name:24s}{count:>7}{share:>16.1%}")
        repaired = block["arity_repaired_by_kind_projection"]
        arity = block["classes"]["arity_mismatch"]
        if arity:
            print(f"   of the {arity} arity mismatches, {repaired} ({repaired / arity:.1%}) are "
                  f"REPAIRED by kind-projection")
            print(f"   -> {repaired} bound an extra column; {arity - repaired} selected "
                  f"different things")
        if block["classes"]["link_mismatch"]:
            parts = ", ".join(f"{name} {count}" for name, count in block["link_subclasses"].items()
                              if count)
            print(f"   link_mismatch by sub-class: {parts}")
        oracle = block.get("clearing_oracle_discriminating") or {}
        if oracle:
            top = list(oracle.items())[:4]
            print("   clearing oracle (discriminating): "
                  + ", ".join(f"{combo} {value:.4f}" for combo, value in top))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True,
                        help="A finished run or merged directory containing predictions.json.")
    parser.add_argument("--store", default=None,
                        help="Result store (default: <run_dir>/result_store).")
    parser.add_argument("--out", default="root_population_report.json",
                        help="Output filename, written inside --run_dir.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    store_dir = Path(args.store) if args.store else run_dir / "result_store"
    if not store_dir.is_dir():
        raise SystemExit(
            f"No result store at {store_dir}. This diagnostic reads stored root result sets; a "
            "run recorded without one cannot be classified without re-executing its queries.")

    report = analyse(run_dir, store_dir)
    _print_report(report)
    # A bare filename lands inside --run_dir (the original behaviour); a path is used as given, so
    # a finished run directory need not be written to (CLAUDE.md 2.3).
    out_path = Path(args.out) if Path(args.out).parent != Path(".") else run_dir / args.out
    _write_json(out_path, report)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
