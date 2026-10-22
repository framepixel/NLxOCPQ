"""The root-population classifier: does each class catch what it claims, in the right order?

The classes are ordered and first match wins, so the order IS the claim: arity subsumes types,
types subsume links. A prediction that differs in two ways must land in the earlier class, or the
report double-counts causes and the resulting shares are meaningless.

The check that matters most is `link_mismatch` on OPERANDS. A prototype keyed filters on their
type and qualifier only, which reported an `O2E` joining object 0 to event 0 as identical to one
joining object 1 to event 0 -- different queries with different populations -- and inflated the
residual class to 12.8%. That case is pinned below by name.

Run from the repo root:  python tests/test_root_population_diagnostic.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from root_population_diagnostic import (  # noqa: E402
    CLASS_ORDER,
    _kind_projected_equal,
    _population,
    classify,
    clearing_oracle,
    link_subclass,
    root_box,
)

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def box(events=None, objects=None, filters=None, size_filters=None, children=None):
    return {"Box": [{"newEventVars": events or {}, "newObjectVars": objects or {},
                     "filters": filters or [], "sizeFilters": size_filters or [],
                     "constraints": [], "evVarLabels": {}, "obVarLabels": {}},
                    children or []]}


def tree(*nodes, edges=None):
    return {"nodes": list(nodes), "edgeNames": edges or []}


O2E_0_0 = {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
O2E_1_0 = {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
O2E_0_0_Q = {"type": "O2E", "object": 0, "event": 0, "qualifier": "places"}

GOLD = tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0],
                events={"0": ["place order"]}))

print("--- each class, in isolation ---")
check("identical roots -> residual",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0],
                             events={"0": ["place order"]}))), "residual")
check("an extra object variable -> arity_mismatch",
      classify(GOLD, tree(box(objects={"0": ["orders"], "1": ["items"]}, filters=[O2E_0_0],
                             events={"0": ["place order"]}))), "arity_mismatch")
check("a missing event variable -> arity_mismatch",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0]))),
      "arity_mismatch")
check("same count, different type -> type_mismatch",
      classify(GOLD, tree(box(objects={"0": ["items"]}, filters=[O2E_0_0],
                             events={"0": ["place order"]}))), "type_mismatch")
check("same vars, a filter dropped -> link_mismatch",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, events={"0": ["place order"]}))),
      "link_mismatch")
check("same vars+filters, different sizeFilters -> size_filter_mismatch",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0],
                             events={"0": ["place order"]},
                             size_filters=[{"type": "SizeFilter", "child_name": "A",
                                            "min": 1, "max": None}]))),
      "size_filter_mismatch")

print("\n--- the bug this classifier exists to not have: OPERANDS ---")
# Same filter type, same qualifier, different operands. Keyed on type alone these look identical
# and the record lands in `residual`, which is how the prototype hid 12.8% of its cases.
check("a filter differing only in its OPERANDS is link_mismatch, not residual",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, filters=[O2E_1_0],
                             events={"0": ["place order"]}))), "link_mismatch")
check("a filter differing only in its QUALIFIER is link_mismatch too",
      classify(GOLD, tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0_Q],
                             events={"0": ["place order"]}))), "link_mismatch")

# Attribute filters, in the shape `src/ocpq_schema.py` actually declares them: `attribute_name`
# and a nested `value_filter`, NOT `attribute`/`value`. A real case from the self-consistency run
# -- "Which applicants older than 40 were rejected as prospective tenants?" -- has gold testing
# Integer{min: 41} and the model testing String{is_in: ["40".."45"]} on the same attribute. Those
# select 0 and 1921 rows respectively, and a key built from guessed field names called them equal.
ATTR = {"type": "ObjectAttributeValueFilter", "object": 0, "attribute_name": "Age",
        "at_time": {"type": "Always"},
        "value_filter": {"type": "Integer", "min": 41, "max": None}}
ATTR_STRING_TEST = dict(ATTR, value_filter={"type": "String", "is_in": ["40", "41"]})
ATTR_OTHER_BOUND = dict(ATTR, value_filter={"type": "Integer", "min": 18, "max": None})
ATTR_OTHER_NAME = dict(ATTR, attribute_name="Status")
GOLD_ATTR = tree(box(objects={"0": ["Person"]}, filters=[ATTR]))
for label, variant in (("a String membership test vs an Integer bound", ATTR_STRING_TEST),
                       ("a different numeric bound", ATTR_OTHER_BOUND),
                       ("a different attribute_name", ATTR_OTHER_NAME)):
    check(f"{label} -> link_mismatch",
          classify(GOLD_ATTR, tree(box(objects={"0": ["Person"]}, filters=[variant]))),
          "link_mismatch")
check("an identical attribute filter is still residual",
      classify(GOLD_ATTR, tree(box(objects={"0": ["Person"]}, filters=[dict(ATTR)]))),
      "residual")
# Gold and the model serialise the same filter's fields in different orders; that is not a
# difference. (The real case above has gold ordering value_filter before at_time, and the model
# the other way round.)
REORDERED = {"at_time": {"type": "Always"}, "type": "ObjectAttributeValueFilter",
             "value_filter": {"max": None, "min": 41, "type": "Integer"},
             "attribute_name": "Age", "object": 0}
check("field ORDER is not a difference",
      classify(GOLD_ATTR, tree(box(objects={"0": ["Person"]}, filters=[REORDERED]))),
      "residual")

print("\n--- first match wins: an earlier cause hides a later one ---")
both = tree(box(objects={"0": ["items"], "1": ["parts"]}, filters=[O2E_1_0],
                events={"0": ["ship"]},
                size_filters=[{"type": "SizeFilter", "child_name": "A", "min": 9, "max": None}]))
check("arity + type + link + size all differ -> arity_mismatch", classify(GOLD, both),
      "arity_mismatch")
check("type + link differ, arity does not -> type_mismatch",
      classify(GOLD, tree(box(objects={"0": ["items"]}, filters=[O2E_1_0],
                             events={"0": ["ship"]}))), "type_mismatch")
check("the order the report depends on", CLASS_ORDER,
      ["arity_mismatch", "type_mismatch", "link_mismatch", "size_filter_mismatch", "residual"])

print("\n--- variable SLOT is not identity; the declared types are ---")
# Which key a type landed under is the query author's arbitrary choice, exactly as it is for the
# result metric. Two roots declaring the same types under swapped keys are the same root.
swapped = tree(box(objects={"1": ["orders"]}, filters=[O2E_0_0], events={"3": ["place order"]}))
check("same types under different variable keys -> not a type_mismatch",
      classify(GOLD, swapped), "residual")

print("\n--- root location and malformed input ---")
# A valid tree may list a child before its parent, and the root is what decides the population.
child_first = tree(box(objects={"9": ["items"]}), box(objects={"0": ["orders"]},
                                                      filters=[O2E_0_0],
                                                      events={"0": ["place order"]},
                                                      children=[0]))
check("root located, not assumed to be index 0",
      root_box(child_first)["newObjectVars"], {"0": ["orders"]})
check("...so this reads as identical to gold", classify(GOLD, child_first), "residual")
check("prediction that is not a dict -> unreadable_root", classify(GOLD, "nope"),
      "unreadable_root")
check("prediction with no nodes -> unreadable_root", classify(GOLD, {"nodes": []}),
      "unreadable_root")
check("prediction whose node is not a Box -> unreadable_root",
      classify(GOLD, {"nodes": [{"NotABox": []}]}), "unreadable_root")

print("\n--- population and the kind-projection repair ---")
GOLD_SIT = {((), (10,), False), ((), (11,), True)}
check("population drops the flag", _population(GOLD_SIT), {((), (10,)), ((), (11,))})
check("None stays None", _population(None), None)
check("an extra event column is REPAIRED by kind-projection",
      _kind_projected_equal(GOLD_SIT, {((7,), (10,), False), ((8,), (11,), True)}), True)
check("different objects are NOT repaired",
      _kind_projected_equal(GOLD_SIT, {((7,), (90,), False), ((8,), (91,), True)}), False)
check("gold binding nothing is never 'repaired'",
      _kind_projected_equal({((), (), False)}, {((7,), (10,), False)}), False)

print("\n--- link_mismatch sub-classes (notes 36.3) ---")
AGE_40 = {"type": "ObjectAttributeValueFilter", "object": 0, "attribute_name": "age",
          "value_filter": {"type": "Integer", "min": 41, "max": None}}
AGE_41 = dict(AGE_40, value_filter={"type": "Integer", "min": 40, "max": None})
AGE_BOOL = dict(AGE_40, value_filter={"type": "Boolean", "is_true": True})
INCOME = dict(AGE_40, attribute_name="income")
BARE = tree(box(objects={"0": ["orders"]}))
with_age = tree(box(objects={"0": ["orders"]}, filters=[AGE_40]))
check("gold's root has no filter, the prediction adds one -> placement",
      link_subclass(BARE, with_age), "placement")
check("an attribute filter dropped -> attribute",
      link_subclass(with_age, tree(box(objects={"0": ["orders"]}, filters=[O2E_0_0]))),
      "attribute")
check("a different attribute -> attribute",
      link_subclass(with_age, tree(box(objects={"0": ["orders"]}, filters=[INCOME]))),
      "attribute")
check("same attribute, other value type -> value_type",
      link_subclass(with_age, tree(box(objects={"0": ["orders"]}, filters=[AGE_BOOL]))),
      "value_type")
check("same attribute and type, other bound -> bounds",
      link_subclass(with_age, tree(box(objects={"0": ["orders"]}, filters=[AGE_41]))),
      "bounds")
check("attribute filters equal, link differs -> structural_link",
      link_subclass(tree(box(objects={"0": ["orders"]}, filters=[AGE_40, O2E_0_0])),
                    tree(box(objects={"0": ["orders"]}, filters=[AGE_40, O2E_0_0_Q]))),
      "structural_link")

print("\n--- clearing oracle ---")
rows = [("no_result", 0.0), ("no_result", 0.0), ("link_mismatch", 0.2), ("population_exact", 1.0)]
oracle = clearing_oracle(rows)
check("clearing one class sets its records to 1", round(oracle["no_result"], 6), 0.8)
check("pairs are cleared together", round(oracle["link_mismatch+no_result"], 6), 1.0)
check("the result is sorted best first", list(oracle)[0], "link_mismatch+no_result")

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> all checks green")
