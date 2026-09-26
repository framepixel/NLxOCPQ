from pydantic import BaseModel, Field, RootModel
from typing import List, Dict, Optional, Union, Literal, Any, Tuple

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated

# --- TYPES AND VARIABLES ---

class VariableEvent(BaseModel):
    Event: int

class VariableObject(BaseModel):
    Object: int

QueryVariable = Union[VariableEvent, VariableObject]

# --- VALUE FILTERS ---

class ValueFilterFloat(BaseModel):
    type: Literal["Float"]
    min: Optional[float] = None
    max: Optional[float] = None

class ValueFilterInteger(BaseModel):
    type: Literal["Integer"]
    min: Optional[int] = None
    max: Optional[int] = None

class ValueFilterBoolean(BaseModel):
    type: Literal["Boolean"]
    is_true: bool

class ValueFilterString(BaseModel):
    type: Literal["String"]
    is_in: List[str]

class ValueFilterTime(BaseModel):
    type: Literal["Time"]
    from_: Optional[str] = Field(None, alias="from")
    to: Optional[str] = None

QueryValueFilter = Annotated[Union[
    ValueFilterFloat, ValueFilterInteger, ValueFilterBoolean, 
    ValueFilterString, ValueFilterTime
], Field(discriminator="type")]

class TimepointAlways(BaseModel):
    type: Literal["Always"]

class TimepointSometime(BaseModel):
    type: Literal["Sometime"]

class TimepointAtEvent(BaseModel):
    type: Literal["AtEvent"]
    event: int

QueryObjectValueFilterTimepoint = Annotated[Union[
    TimepointAlways, TimepointSometime, TimepointAtEvent
], Field(discriminator="type")]

FilterLabel = Literal["IGNORED", "INCLUDED", "EXCLUDED"]

# --- OCPQ FILTERS ---

class FilterO2E(BaseModel):
    type: Literal["O2E"]
    object: int
    event: int
    qualifier: Optional[str] = None
    filter_label: Optional[FilterLabel] = Field(None, alias="filterLabel")

class FilterO2O(BaseModel):
    type: Literal["O2O"]
    object: int
    other_object: int
    qualifier: Optional[str] = None
    filter_label: Optional[FilterLabel] = Field(None, alias="filterLabel")

class FilterTimeBetweenEvents(BaseModel):
    type: Literal["TimeBetweenEvents"]
    from_event: int
    to_event: int
    min_seconds: Optional[float] = None
    max_seconds: Optional[float] = None

class FilterNotEqual(BaseModel):
    type: Literal["NotEqual"]
    var_1: QueryVariable
    var_2: QueryVariable

class FilterEventAttributeValueFilter(BaseModel):
    type: Literal["EventAttributeValueFilter"]
    event: int
    attribute_name: str
    value_filter: QueryValueFilter

class FilterObjectAttributeValueFilter(BaseModel):
    type: Literal["ObjectAttributeValueFilter"]
    object: int
    attribute_name: str
    at_time: QueryObjectValueFilterTimepoint
    value_filter: QueryValueFilter

class FilterBasicFilterCEL(BaseModel):
    type: Literal["BasicFilterCEL"]
    cel: str

QueryFilter = Annotated[Union[
    FilterO2E, FilterO2O, FilterTimeBetweenEvents, FilterNotEqual, 
    FilterEventAttributeValueFilter, FilterObjectAttributeValueFilter, FilterBasicFilterCEL
], Field(discriminator="type")]

# --- OCPQ SIZE FILTERS ---

class SizeFilterNumChilds(BaseModel):
    type: Literal["NumChilds"]
    child_name: str
    min: Optional[int] = None
    max: Optional[int] = None

class SizeFilterAdvancedCEL(BaseModel):
    type: Literal["AdvancedCEL"]
    cel: str

QuerySizeFilter = Annotated[Union[
    SizeFilterNumChilds, SizeFilterAdvancedCEL
], Field(discriminator="type")]

# --- OCPQ CONSTRAINTS ---

class ConstraintFilterWrap(BaseModel):
    type: Literal["Filter"]
    filter: QueryFilter

class ConstraintSizeFilterWrap(BaseModel):
    type: Literal["SizeFilter"]
    filter: QuerySizeFilter

class ConstraintSAT(BaseModel):
    type: Literal["SAT"]
    child_names: List[str]

class ConstraintANY(BaseModel):
    type: Literal["ANY"]
    child_names: List[str]

class ConstraintNOT(BaseModel):
    type: Literal["NOT"]
    child_names: List[str]

class ConstraintOR(BaseModel):
    type: Literal["OR"]
    child_names: List[str]

class ConstraintAND(BaseModel):
    type: Literal["AND"]
    child_names: List[str]

QueryConstraint = Annotated[Union[
    ConstraintFilterWrap, ConstraintSizeFilterWrap, ConstraintSAT, 
    ConstraintANY, ConstraintNOT, ConstraintOR, ConstraintAND
], Field(discriminator="type")]

# --- CEL LABELS ---

class CELLabel(BaseModel):
    label: str
    cel: str

# --- TREE NODES ---

class BindingBoxData(BaseModel):
    newEventVars: Dict[str, List[str]]
    newObjectVars: Dict[str, List[str]]
    filters: List[QueryFilter]
    sizeFilters: List[QuerySizeFilter]
    constraints: List[QueryConstraint]
    evVarLabels: Dict[str, FilterLabel]
    obVarLabels: Dict[str, FilterLabel]
    labels: List[CELLabel] = Field(default_factory=list)

class BindingBoxTreeNodeBox(BaseModel):
    Box: Tuple[BindingBoxData, List[int]]

class BindingBoxTree(BaseModel):
    nodes: List[BindingBoxTreeNodeBox]
    edgeNames: List[Tuple[Tuple[int, int], str]]

class RootArity(BaseModel):
    """How many variables the ROOT binding box declares -- the prediction target of the arity
    predictor.

    Forcing this to gold's value took result F1 from 0.2162 to 0.3109 (notes 32), the largest
    effect any intervention in this project has had. That condition is an ORACLE; predicting the
    pair from the question is what turns it into a system.

    Bounded at 4 because gold never exceeds 2 of either kind and the model has no business
    proposing more: an unbounded integer lets a constrained decoder emit 999 and stall.
    """
    n_event_vars: int = Field(ge=0, le=4)
    n_object_vars: int = Field(ge=0, le=4)

class AgentAction(BaseModel):
    """One turn of the agentic protocol: look something up in the log, or answer.

    The agentic arm has no native tool-calling to lean on -- the local backend is raw
    `transformers.generate` with a chat template, and the fine-tuned adapters were trained to emit
    one JSON object -- so a turn is a JSON object either way, and this is its shape. Constraining
    generation to it (`LLM.set_constrained_schema`) is what makes a well-formed turn a guarantee
    rather than a hope, which is the same argument notes 33 made for `RootArity`.

    Exactly one of `tool` or `final_tree` is expected; `LLM._parse_agent_action` enforces that,
    because a grammar that permits both permits neither to be trusted. `args` is deliberately
    free-form: each tool documents its own arguments in `ocpq_probe.TOOL_SPECS`, and a wrong
    argument is a mistake the model should see and correct, not a parse failure.
    """
    tool: Optional[str] = None
    args: Optional[Dict[str, Any]] = None
    final_tree: Optional[BindingBoxTree] = None


class AgentActionWithThought(AgentAction):
    """`AgentAction` plus a short free-text `thought` -- the agentic arm's reasoning condition.

    A SEPARATE model rather than a field on `AgentAction`, because the grammar is compiled with
    `strict_mode=True`, which forbids unknown keys: under constrained decoding a model on the plain
    schema CANNOT emit a thought. That makes the "off" rung clean and byte-identical to the one
    Stage 3 verified, and makes this model the only difference between the two (notes 43).

    Why a protocol field and not native reasoning: the per-turn grammar masks everything that is not
    this JSON, so a reasoning model's own thinking tokens can never be emitted here. A field lets
    EVERY evaluated model reason, inside the grammar, whether or not it has a native mode.

    OPTIONAL, not required, deliberately: xgrammar's `any_order=True` can skip a required property
    (see `LLM._fill_missing_box_containers`), so "required" would not actually be guaranteed. The
    uptake -- how many turns carried a thought -- is recorded instead of assumed.
    """
    thought: Optional[str] = None


# --- THE STRICT TURN GRAMMAR (2026-09-23) ------------------------------------------------------
#
# `AgentAction` / `AgentActionWithThought` above are what a PARSED turn becomes. The models below
# are what the GRAMMAR is compiled from, and they are much tighter. The gate smoke of 2026-09-23
# (eval_runs/agentic_gate_smoke, 10 cases) showed why a single all-optional object is not enough:
#   * every field optional + `args: Dict[str, Any]` let the model write a `final_tree` AND an
#     `args.tree` in ONE object, run on to the 4000-token limit, and be cut off mid-JSON -- about
#     180 s a turn; three such turns spent the whole 600 s case cap (2 of 5 cases timed out);
#   * the tree inside `try_query` was free-form JSON, so its shape went unconstrained
#     (`edgeNames[0][0]: invalid type`, undeclared references) while `final_tree` was checked.
# A union of one variant per tool plus one final-answer variant lets the grammar admit exactly one
# well-formed action per turn, with typed arguments and a real BindingBoxTree wherever a tree goes.
# The tool names here must match `ocpq_probe.TOOL_SPECS`; tests/test_agent_action_protocol.py
# asserts they do.


class TryQueryArgs(BaseModel):
    tree: BindingBoxTree


class InspectSituationsArgs(BaseModel):
    node: int = 0
    n: int = 5
    violated: Optional[bool] = None


class SampleArgs(BaseModel):
    type: str
    n: int = 5


class AttributeValuesArgs(BaseModel):
    type: str
    attribute: str
    n: int = 10


class ActivityStatsArgs(BaseModel):
    type: str


class TryQueryCall(BaseModel):
    tool: Literal["try_query"]
    args: TryQueryArgs


class InspectSituationsCall(BaseModel):
    tool: Literal["inspect_situations"]
    args: InspectSituationsArgs


class SampleCall(BaseModel):
    tool: Literal["sample"]
    args: SampleArgs


class AttributeValuesCall(BaseModel):
    tool: Literal["attribute_values"]
    args: AttributeValuesArgs


class ActivityStatsCall(BaseModel):
    tool: Literal["activity_stats"]
    args: ActivityStatsArgs


class FinalAnswer(BaseModel):
    final_tree: BindingBoxTree


# The thought variants spell their fields out, `thought` FIRST, rather than subclassing the plain
# calls: a subclass appends its field, and under the ordered grammar (`constrained_any_order`
# false, notes 43.7) property order is generation order -- the thought would come after the action
# and the whole tree, which is reasoning written after the decision it was meant to inform.


class TryQueryCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["try_query"]
    args: TryQueryArgs


class InspectSituationsCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["inspect_situations"]
    args: InspectSituationsArgs


class SampleCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["sample"]
    args: SampleArgs


class AttributeValuesCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["attribute_values"]
    args: AttributeValuesArgs


class ActivityStatsCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["activity_stats"]
    args: ActivityStatsArgs


class FinalAnswerWithThought(BaseModel):
    thought: Optional[str] = None
    final_tree: BindingBoxTree


AGENT_TOOL_CALLS = (TryQueryCall, InspectSituationsCall, SampleCall, AttributeValuesCall,
                    ActivityStatsCall)


class AgentTurn(RootModel[Union[TryQueryCall, InspectSituationsCall, SampleCall,
                                AttributeValuesCall, ActivityStatsCall, FinalAnswer]]):
    """One agent turn, as the grammar admits it: exactly one tool call or one final answer."""


class AgentTurnWithThought(RootModel[Union[
        TryQueryCallWithThought, InspectSituationsCallWithThought, SampleCallWithThought,
        AttributeValuesCallWithThought, ActivityStatsCallWithThought, FinalAnswerWithThought]]):
    """`AgentTurn` for the reasoning condition: every variant may carry a `thought`. A separate
    class NAME, because the grammar cache (LLM._ocpq_logits_processor) is keyed by it."""


# --- STAGE 5 (notes 43.12): check_tree, and a turn grammar per condition -------------------------
#
# `check_tree` is the STATIC structure check: `structural_defects` on the model's tree, no log. Both
# loop rungs get it under `agent_structural_feedback`, so the no-tools control A1' has the same
# structural help as A1 and A1 - A1' isolates LOG ACCESS (decided 2026-09-26).


class CheckTreeArgs(BaseModel):
    tree: BindingBoxTree


class CheckTreeCall(BaseModel):
    tool: Literal["check_tree"]
    args: CheckTreeArgs


class CheckTreeCallWithThought(BaseModel):
    thought: Optional[str] = None
    tool: Literal["check_tree"]
    args: CheckTreeArgs


_TURN_VARIANTS = {
    # name: (plain variant, thought variant)
    "try_query": (TryQueryCall, TryQueryCallWithThought),
    "inspect_situations": (InspectSituationsCall, InspectSituationsCallWithThought),
    "sample": (SampleCall, SampleCallWithThought),
    "attribute_values": (AttributeValuesCall, AttributeValuesCallWithThought),
    "activity_stats": (ActivityStatsCall, ActivityStatsCallWithThought),
    "check_tree": (CheckTreeCall, CheckTreeCallWithThought),
}
PROBE_TOOL_NAMES = ("try_query", "inspect_situations", "sample", "attribute_values",
                    "activity_stats")
_TURN_MODEL_CACHE: Dict[Tuple, Any] = {}


def agent_turn_model(tools: Tuple[str, ...], thought: bool = False, final_only: bool = False):
    """The turn grammar for a condition: exactly one call to one of `tools`, or a final answer.

    With the five probe tools, no `check_tree` and not `final_only` this returns `AgentTurn` /
    `AgentTurnWithThought` THEMSELVES, so every condition without the Stage 5 levers compiles the
    grammar it compiled before. Otherwise a union is built once per combination, with a class name
    that spells the combination -- the grammar cache is keyed by the name.

    `final_only` (the reserved last turn) admits the final answer alone.
    """
    tools = tuple(tools)
    if not final_only and tools == PROBE_TOOL_NAMES:
        return AgentTurnWithThought if thought else AgentTurn
    key = (() if final_only else tools, bool(thought), bool(final_only))
    if key not in _TURN_MODEL_CACHE:
        final = FinalAnswerWithThought if thought else FinalAnswer
        calls = [] if final_only else [_TURN_VARIANTS[name][1 if thought else 0]
                                       for name in tools]
        variants = tuple(calls + [final])
        root = RootModel[Union[variants]] if len(variants) > 1 else RootModel[final]
        name = ("AgentTurn_" + ("final_only" if final_only else "_".join(tools) or "no_tools")
                + ("_thought" if thought else ""))
        _TURN_MODEL_CACHE[key] = type(name, (root,), {"__doc__": agent_turn_model.__doc__})
    return _TURN_MODEL_CACHE[key]


# A one-line fix per defect kind, appended to each defect the agent is shown (notes 43.12). The
# measured failures of the model gate, 298 try_query trees: children_mismatch 205, cycle 192,
# root_count 168, unbound_variable 101, multiple_parents 90 -- the model writes plausible boxes and
# does not know how a tree LINKS them. Static text about the representation, never about a case.
DEFECT_HINTS = {
    "cycle": "a Box's child list holds the indices of OTHER nodes, never the node's own index",
    "children_mismatch": "node p's child list must hold exactly the c of every [[p, c], name] "
                         "entry in edgeNames, and nothing else",
    "root_count": "node 0 is the root: the ONE node that appears in no child list; every other "
                  "node appears in exactly one",
    "multiple_parents": "a node appears in exactly one child list -- it has one parent",
    "child_out_of_range": "child indices must be node indices that exist (0 .. number of nodes-1)",
    "edge_name_out_of_range": "edgeNames entries must link existing nodes, parent first",
    "unbound_variable": "a filter may only read variables declared in its own box or an ancestor "
                        "box; declare the variable there or move the filter",
    "unbound_optional_reference": "an optional reference names a variable no box in scope "
                                  "declares; declare it or drop the reference",
    "unknown_child_name": "SAT/NOT/NumChilds name a child by its edgeNames name; use a name this "
                          "node's own edgeNames entries give",
    "malformed_tree": "each node is {\"Box\": [box data, [child indices]]}",
}


class QueryVerification(BaseModel):
    is_correct: bool
    issues: List[str]
    fixed_query: BindingBoxTree

class ExtractedStructure(BaseModel):
    events: List[str]
    objects: List[str]
    relations: List[str]
    constraints: List[str]
    requires_nesting: bool
    # Optional: only populated by strategies that explicitly ask for schema relation
    # grounding (currently schema-grammar-guided). Left empty by strategies that don't
    # reference these fields in their prompt, since they carry a default.
    object_event_links: List[str] = Field(default_factory=list)
    object_object_links: List[str] = Field(default_factory=list)

class QueryTreeNode(BaseModel):
    id: int
    description: str
    events: List[str]
    objects: List[str]

class QueryTreeEdge(BaseModel):
    from_: int = Field(alias="from")
    to: int
    label: str

class QueryTree(BaseModel):
    nodes: List[QueryTreeNode]
    edges: List[QueryTreeEdge]

OCPQ_JSON_SCHEMA = BindingBoxTree.model_json_schema()
ROOT_ARITY_SCHEMA = RootArity.model_json_schema()
AGENT_ACTION_SCHEMA = AgentAction.model_json_schema()
AGENT_ACTION_WITH_THOUGHT_SCHEMA = AgentActionWithThought.model_json_schema()
AGENT_TURN_SCHEMA = AgentTurn.model_json_schema()
AGENT_TURN_WITH_THOUGHT_SCHEMA = AgentTurnWithThought.model_json_schema()
VERIFICATION_JSON_SCHEMA = QueryVerification.model_json_schema()
EXTRACTED_STRUCTURE_SCHEMA = ExtractedStructure.model_json_schema()
QUERY_TREE_SCHEMA = QueryTree.model_json_schema()

# --- STRUCTURAL VALIDATION ---
#
# Pydantic checks that a tree has the right SHAPE. It cannot check that the tree is EVALUABLE, and
# ocpq_cli does not reject an unevaluable tree loudly: `evaluate_box_tree` returns an `Err` before
# printing "Evaluated in", and `run_evaluate` serialises that `Result` straight into the export and
# exits 0 (OCPQ/backend/ocpq_cli/src/main.rs). On 2026-09-13 that was 88 of the 507 executions of
# the three arity rungs. Replayed through the CLI, 61 of the 68 distinct failing trees reported
# "Could not get oN": a filter names a variable no enclosing box declares, so the step planner tries
# to bind it in a box that never declared it (backend/shared/src/binding_box/expand_step.rs).
# See Thesis/notes/11-evaluation-readiness.md §36.
#
# Every defect is a string "[kind] message", so one list can be counted and shown to a model. The
# kinds split by what the Rust evaluator does when it reaches them:
#   BLOCKING -- it errors or panics. Only these may trigger a repair.
#   advisory -- it tolerates them: an optional reference with no value is treated as absent
#               (`if let Some(..)`), and an edge name nobody walks is never looked up. Reported,
#               never repaired. On 2026-09-13, 26 trees that evaluated normally carried one.
# CORRECTION (2026-09-22, measured on the pinned OCPQ, one tree per kind): `cycle` and
# `multiple_parents` do NOT always block -- ocpq_cli evaluated both (exit 0, an `Ok` export; the
# cycle tree to 0 situations). They stay in this set deliberately: across matrix_v2 only 8 of 5500
# predictions carry them as their ONLY blocking kind (3 of which never executed anyway), the
# non-oracle rung's structval shards carry them 0 times, and a tree whose boxes do not form one tree
# is broken whatever the evaluator makes of it. `child_out_of_range` panics the evaluator.
# See TODO (2026-09-22) and ocpq_probe.PROBE_REFUSED_DEFECT_KINDS.
BLOCKING_DEFECT_KINDS = frozenset({
    "malformed_tree",
    "child_out_of_range",
    "multiple_parents",
    "cycle",
    "root_count",
    "unbound_variable",
})

_EVENT_REFERENCE_FIELDS = ("event", "from_event", "to_event")
_OBJECT_REFERENCE_FIELDS = ("object", "other_object")
_CHILD_NAME_CONSTRAINTS = ("SAT", "ANY", "NOT", "OR", "AND")


def defect_kind(defect: str) -> str:
    """The `kind` of a "[kind] message" defect string, or "" if it has none."""
    if isinstance(defect, str) and defect.startswith("[") and "]" in defect:
        return defect[1:defect.index("]")]
    return ""


def blocking_defects(defects: List[str]) -> List[str]:
    """The defects the evaluator errors on -- the only ones worth spending a repair on."""
    return [defect for defect in defects if defect_kind(defect) in BLOCKING_DEFECT_KINDS]


def _unhashable(value: Any) -> bool:
    """True for a value that cannot be a set member (a list or dict where a name belongs).

    Only these become defects: they are exactly the inputs that used to raise TypeError in
    `structural_defects`. A missing name (None) keeps its advisory `unknown_child_name` reading.
    """
    try:
        hash(value)
    except TypeError:
        return True
    return False


def _is_index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _variable_keys(declared: Any) -> set:
    """The variable indices a `newEventVars` / `newObjectVars` map declares."""
    if not isinstance(declared, dict):
        return set()
    keys = set()
    for key in declared:
        try:
            keys.add(int(key))
        except (TypeError, ValueError):
            continue
    return keys


def _filter_references(query_filter: dict) -> List[Tuple[str, str, int]]:
    """Every variable one filter reads, as `(defect kind if unbound, "event"/"object", index)`.

    A plain operand (`event`, `object`, `from_event`, ...) is looked up with `ok_or_else(..)?` and
    errors when unbound. An at-event timepoint and a NotEqual operand are looked up as optional
    values and silently treated as absent, so an unbound one is advisory.
    """
    references: List[Tuple[str, str, int]] = []
    for field in _EVENT_REFERENCE_FIELDS:
        if _is_index(query_filter.get(field)):
            references.append(("unbound_variable", "event", query_filter[field]))
    for field in _OBJECT_REFERENCE_FIELDS:
        if _is_index(query_filter.get(field)):
            references.append(("unbound_variable", "object", query_filter[field]))
    at_time = query_filter.get("at_time")
    if isinstance(at_time, dict) and _is_index(at_time.get("event")):
        references.append(("unbound_optional_reference", "event", at_time["event"]))
    if query_filter.get("type") == "NotEqual":
        for side in ("var_1", "var_2"):
            variable = query_filter.get(side)
            if isinstance(variable, dict):
                if _is_index(variable.get("Event")):
                    references.append(("unbound_optional_reference", "event", variable["Event"]))
                if _is_index(variable.get("Object")):
                    references.append(("unbound_optional_reference", "object",
                                       variable["Object"]))
    return references


def structural_defects(tree: Any) -> List[str]:
    """What would stop ocpq_cli evaluating this tree, as far as can be told without running it.

    Checks what Pydantic cannot: that the nodes form one tree, that each node's child list agrees
    with `edgeNames`, that every child name a size filter or constraint uses is one of the node's
    own children, and that every variable a filter reads is declared by the node or an ancestor.
    Variables declared by a sibling or a descendant are NOT in scope.

    Use `blocking_defects` to keep only the kinds the evaluator errors on. An empty list means none
    of these defects were found -- not that the query is correct.

    Args:
        tree: A BindingBoxTree as a plain dict, typically model output.

    Returns:
        Deterministically ordered, de-duplicated "[kind] message" strings. Never raises: it runs
        on model output, and a malformed prediction must be reportable rather than fatal.
    """
    defects: List[str] = []
    if not isinstance(tree, dict) or not isinstance(tree.get("nodes"), list):
        return ["[malformed_tree] no `nodes` list"]
    nodes = tree["nodes"]
    count = len(nodes)
    if count == 0:
        return defects

    boxes: List[Tuple[dict, list]] = []
    for index, node in enumerate(nodes):
        box = node.get("Box") if isinstance(node, dict) else None
        if (not isinstance(box, (list, tuple)) or len(box) != 2 or not isinstance(box[0], dict)
                or not isinstance(box[1], list)):
            defects.append(f"[malformed_tree] node {index} is not a Box of (data, children)")
            boxes.append(({}, []))
        else:
            boxes.append((box[0], box[1]))

    # The child lists are what evaluation walks, so they define the parent of each node.
    parents: Dict[int, int] = {}
    for index, (_data, children) in enumerate(boxes):
        for child in children:
            if not _is_index(child) or not 0 <= child < count:
                defects.append(f"[child_out_of_range] node {index} lists child {child!r}, but the "
                               f"tree has nodes 0..{count - 1}")
                continue
            if child == index:
                defects.append(f"[cycle] node {index} lists itself as a child")
                continue
            if child in parents and parents[child] != index:
                defects.append(f"[multiple_parents] node {child} is a child of both node "
                               f"{parents[child]} and node {index}")
                continue
            parents[child] = index

    edge_children: Dict[int, set] = {}
    child_names: Dict[int, set] = {}
    edges = tree.get("edgeNames")
    for entry in edges if isinstance(edges, list) else []:
        try:
            (parent, child), name = entry
        except (TypeError, ValueError):
            defects.append(f"[malformed_tree] edgeNames entry {entry!r} is not "
                           f"[[parent, child], name]")
            continue
        if not (_is_index(parent) and _is_index(child)
                and 0 <= parent < count and 0 <= child < count):
            defects.append(f"[edge_name_out_of_range] edgeNames entry [[{parent!r}, {child!r}], "
                           f"{name!r}] points outside nodes 0..{count - 1}")
            continue
        if _unhashable(name):
            # A list or dict here cannot go in the name set below (it used to raise TypeError), and
            # the evaluator refuses it at deserialisation anyway: edge names are strings.
            defects.append(f"[malformed_tree] edgeNames entry [[{parent}, {child}], {name!r}]: "
                           f"an edge name must be a string")
            continue
        edge_children.setdefault(parent, set()).add(child)
        child_names.setdefault(parent, set()).add(name)

    for index, (_data, children) in enumerate(boxes):
        listed = {c for c in children if _is_index(c) and 0 <= c < count and c != index}
        named = edge_children.get(index, set())
        if listed != named:
            defects.append(f"[children_mismatch] node {index} lists children {sorted(listed)} "
                           f"but edgeNames link it to {sorted(named)}")

    in_cycle = set()
    for index in range(count):
        current, steps = index, 0
        while current in parents and steps <= count:
            current, steps = parents[current], steps + 1
        if steps > count:
            in_cycle.add(index)
    if in_cycle:
        defects.append(f"[cycle] nodes {sorted(in_cycle)} are their own ancestors")

    roots = [index for index in range(count) if index not in parents]
    if len(roots) != 1:
        defects.append(f"[root_count] {len(roots)} node(s) have no parent {roots}; a tree has "
                       f"exactly one root")

    def in_scope(index: int) -> Tuple[set, set]:
        events, objects, seen = set(), set(), set()
        current: Optional[int] = index
        while current is not None and current not in seen:
            seen.add(current)
            data = boxes[current][0]
            events |= _variable_keys(data.get("newEventVars"))
            objects |= _variable_keys(data.get("newObjectVars"))
            current = parents.get(current)
        return events, objects

    for index, (data, _children) in enumerate(boxes):
        scope = dict(zip(("event", "object"), in_scope(index)))
        names = child_names.get(index, set())
        shown_names = sorted(str(name) for name in names)

        def check_filter(query_filter: dict, where: str) -> None:
            filter_type = query_filter.get("type")
            for kind, variable_kind, variable in _filter_references(query_filter):
                if variable in scope[variable_kind]:
                    continue
                if kind == "unbound_variable":
                    defects.append(f"[unbound_variable] node {index}: {where}{filter_type} reads "
                                   f"{variable_kind} variable {variable}, which neither node "
                                   f"{index} nor its ancestors declare")
                else:
                    defects.append(f"[unbound_optional_reference] node {index}: {where}"
                                   f"{filter_type} refers to {variable_kind} variable {variable}, "
                                   f"which is not in scope, so the engine treats it as absent")

        def check_child_name(size_filter: dict, where: str) -> None:
            if size_filter.get("type") != "NumChilds":
                return
            name = size_filter.get("child_name")
            if _unhashable(name):
                defects.append(f"[malformed_tree] node {index}: {where}NumChilds child_name "
                               f"{name!r} is not a string")
                return
            if name not in names:
                defects.append(f"[unknown_child_name] node {index}: {where}NumChilds names child "
                               f"{name!r}, but node {index}'s children are named {shown_names}")

        for query_filter in data.get("filters") or []:
            if isinstance(query_filter, dict):
                check_filter(query_filter, "")
        for size_filter in data.get("sizeFilters") or []:
            if isinstance(size_filter, dict):
                check_child_name(size_filter, "size filter ")
        for constraint in data.get("constraints") or []:
            if not isinstance(constraint, dict):
                continue
            constraint_type = constraint.get("type")
            inner = constraint.get("filter")
            if constraint_type == "Filter" and isinstance(inner, dict):
                check_filter(inner, "constraint ")
            elif constraint_type == "SizeFilter" and isinstance(inner, dict):
                check_child_name(inner, "constraint ")
            elif constraint_type in _CHILD_NAME_CONSTRAINTS:
                for name in constraint.get("child_names") or []:
                    if _unhashable(name):
                        defects.append(f"[malformed_tree] node {index}: {constraint_type} "
                                       f"child name {name!r} is not a string")
                        continue
                    if name not in names:
                        defects.append(f"[unknown_child_name] node {index}: {constraint_type} "
                                       f"names child {name!r}, but node {index}'s children are "
                                       f"named {shown_names}")

    return list(dict.fromkeys(defects))
