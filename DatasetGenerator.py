#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import InferenceClient

from pydantic import ValidationError

from ocpq_schema import BindingBoxTree

# In[2]:


train_with_rewrites = False

# In[3]:


patterns = [
    # Generic patterns
    "The EVENT_A event happens before the EVENT_B event.",
    "Whenever the EVENT_A event occurs, the EVENT_B event eventually occurs.",
    "After the EVENT_A event, the EVENT_B event must not occur.",
    "The EVENT_A and EVENT_B events must refer to the same OBJECT_A.",
    "The EVENT_B event happens within TIME TIME_UNIT after the EVENT_A event.",
    # Parent-child pattern
    "Each OBJECT_A must have at least NUM associated OBJECT_B for which the EVENT_A event occurs.",
    # Multi-child pattern
    "For any OBJECT_A with at most NUM associated OBJECT_B, the EVENT_B event must occur within TIME TIME_UNIT after the EVENT_A event.",
    # SAT pattern 
    # "A constraint modeling that for any OBJECT_A, the EVENT_B event should happen fully within TIME TIME_UNIT of the EVENT_A event.",
    "For any OBJECT_A, EVENT_B must be completed within TIME TIME_UNIT after EVENT_A.",
    # Deep nested pattern
    "Ensure that any OBJECT_A has at least NUM associated OBJECT_B where the EVENT_B event has occurred at least NUM2 times.",
    # Exactly once pattern
    # "A constraint modeling that the EVENT_A event should occur exactly once for each OBJECT_A.",
    "EVENT_A must occur exactly once for each OBJECT_A.",
    # At least once pattern
    "After the EVENT_A event occurs for a given OBJECT_A, the EVENT_B event should happen at least once.",
    # CEL Aggregation pattern
    "Find all OBJECT_A whose total ATT_LABEL is OP_TEXT NUM3, where the total is calculated by summing the ATT_NAME of every associated OBJECT_B that had an EVENT_A event."
]

# In[4]:


domains = {
    # Generic domains
    "e-commerce": {
        "events": ["create order", "add item", "pay order", "ship package", "cancel order"],
        "objects": ["customers", "employees", "items", "orders", "packages", "products"]
    },
    "recruitment": {
        "events": ["submit application", "review application", "approve application", "reject application"],
        "objects": ["applications", "candidates"]
    },
    "software_dev": {
        "events": ["open issue", "submit pull request", "review pull request", "merge pull request", "close issue"],
        "objects": ["issues", "pull requests", "commits"]
    },

    # Domains from actual log files
    "distribution": {
        "events": ["Apply Discount", "Apply Payment", "Approve Order", "Approve Return", "Generate Invoice", 
                   "Generate Pick List", "Generate Shipping Label", "Handle Payment Discrepancy", 
                   "Issue Credit Note", "Issue Payment Reminder", "Pack Items", "Perform Credit Check", 
                   "Pick Items", "Process Return Request", "Receive Customer Order", "Receive Payment", 
                   "Reject Order", "Schedule Order Fulfillment", "Send Invoice to Customer", "Ship Order", 
                   "Update Inventory", "Validate Customer Order"],
        "objects": ["CreditNote", "Customer", "Dispute", "Invoice", "Order", "Payment", "Product", "Return", "Shipment"]
    },
    "procurement": {
        "events": ["Approve PO", "Approve PR", "Create PO", "Create PO (Batch)", "Create PR", "Make Payment", 
                   "Receive Goods", "Receive Invoice", "Reject PR", "Send PO to Supplier", 
                   "Three-way Match - Mismatch Found", "Three-way Match - Successful"],
        "objects": ["Goods Receipt", "Purchase Requisition", "Payment", "Invoice", "Purchase Order", "Supplier"]
    },
    "02_p2p": {
        "events": ["Approve PO", "Approve PR", "Create PO", "Create PO (Batch)", "Create PR", "Make Payment", 
                   "Receive Goods", "Receive Invoice", "Reject PR", "Send PO to Supplier", "Three-way Match - Mismatch Found", 
                   "Three-way Match - Successful"],
        "objects": ["Goods Receipt", "Invoice", "Payment", "Purchase Order", "Purchase Requisition", "Supplier"]
    },
    "hiring": {
        "events": ['Approve Job Requisition', 'Attend Orientation Training', 'Candidate Accept Offer', 'Candidate Reject Offer', 
                   'Complete Assessment', 'Complete Background Check', 'Complete HR Paperwork', 'Complete Onboarding', 
                   'Conduct Initial Interview', 'Conduct Panel Interview', 'Create Job Requisition', 'Evaluate Interview', 
                   'Initiate Background Check', 'Post Job Opening', 'Prepare Offer Letter', 'Review Assessment', 
                   'Schedule Initial Interview', 'Schedule Panel Interview', 'Screen Resume', 'Send Assessment', 
                   'Send Offer Letter', 'Send Rejection Letter', 'Start Onboarding', 'Submit Application'],
        "objects": ['Interviewer', 'HiringManager', 'Recruiter', 'Assessment', 'Candidate', 'JobRequisition']
    },
    "hospital": {
        "events": ['Appointment Scheduling', 'Assign Bed', 'Diagnostic Test Order', 'Initial Evaluation', 
                   'Insurance Verification', 'Patient Arrival', 'Perform Diagnostic Test', 'Pre-Admission Consultation', 
                   'Registration'],
        "objects": ['Patient', 'InsuranceCompany', 'Physician', 'LabTechnician', 'Nurse', 'DiagnosticEquipment', 'Room', 'Bed']
    },
    "container_logistics": {
        "events": ['Book Vehicles', 'Bring to Loading Bay', 'Collect Goods', 'Create Transport Document', 'Depart', 
                   'Drive to Terminal', 'Load Truck', 'Load to Vehicle', 'Order Empty Containers', 'Pick Up Empty Container', 
                   'Place in Stock', 'Register Customer Order', 'Reschedule Container', 'Weigh'],
        "objects": ['Container', 'Customer Order', 'Forklift', 'Handling Unit', 'Transport Document', 'Truck', 'Vehicle']
    },
    "lending": {
        "events": ['Application Rejected', 'Appointment Denied', 'Hand In Credit Appliaction', 'Loan Denied', 
                   'Make Visit to Assess Colatteral', 'Request Appointment', 'Request Co-Signer On Loan', 
                   'Set Appointment', 'Sign Loan Agreement', 'Submit File to Underwriter', 'Verify Borrowers Information', 
                   'skipped_examination'],
        "objects": ['Person']
    },
    "renting": {
        "events": ["Accept Late Payment", "Apply for Viewing Appointment", "Check Paperwork", "Evict Tenant", "Extensive Screening",
                   "Hand In The Paperwork", "Issue Warning", "Miss Rent Payment", "Move In", "Pay Rent", "Reject Prospective Tenant",
                   "Screen Prospective Tenant", "Set Appointment", "Sign Contract", "Tenant Cancels Appartment", "View The Property"],
        "objects": ["Person"]
    }
}

# In[4b]:


# =====================================================================================
# Extended building-block coverage.
#
# The patterns above cover ~13 OCPQ building blocks. The testset evaluates ~35, so the
# fine-tuning / retrieval pool must teach the rest: qualifiers (F3), TBE min>0 (F6) and
# simultaneous (F7), NotEqual (F8), event/object attribute-value filters (F9-F17) across
# at_time modes, BasicFilterCEL (F18), AdvancedCEL + CEL labels (S3/L1), ANY/OR/AND
# constraints (C4/C6/C7), INCLUDED var labels (L2/L3), deep chains (T5), and multi-type
# variables (V3/V4).
#
# These are kept in a self-contained generator (`generate_extended_example`) with an
# explicit builder registry so they cannot collide with the substring-based dispatch of
# the original patterns. Every builder returns the same example dict shape as
# `generate_example`: {"nl", "events", "objects", "query", "reasoning"}.
# =====================================================================================

# Object attributes per domain, used for attribute-value filters and CEL aggregations.
# Each tuple: (object_type, attribute_name, kind[, string_values]).
# kind in {"Float", "Integer", "String", "Boolean", "Time"}. Attribute names follow the
# real OCEL logs where known (e.g. Container.AmountofHandlingUnits, Person.Creditscore)
# and use plausible synthetic names elsewhere so each example stays internally consistent.
OBJECT_ATTRIBUTES = {
    "e-commerce":          [("orders", "price", "Float"), ("products", "weight", "Float"),
                            ("orders", "orderDate", "Time")],
    "distribution":        [("Order", "totalAmount", "Float"), ("Product", "weight", "Float")],
    "procurement":         [("Purchase Order", "amount", "Float")],
    "02_p2p":              [("Purchase Order", "amount", "Float")],
    "hiring":              [("Candidate", "yearsExperience", "Integer"),
                            ("JobRequisition", "openings", "Integer")],
    "hospital":            [("Patient", "age", "Integer"), ("Patient", "isInpatient", "Boolean")],
    "container_logistics": [("Container", "AmountofHandlingUnits", "Float"),
                            ("Container", "Status", "String", ["full", "empty"])],
    "lending":             [("Person", "Creditscore", "Integer"), ("Person", "Age", "Integer"),
                            ("Person", "Yearsofeducation", "Integer"),
                            ("Person", "hasGuarantor", "Boolean"),
                            ("Person", "registrationDate", "Time")],
    "renting":             [("Person", "monthlyIncome", "Integer")],
    "recruitment":         [("candidates", "yearsExperience", "Integer")],
    "software_dev":        [("pull requests", "linesChanged", "Integer")],
}

# Event attributes used for EventAttributeValueFilter (F9) and attribute-comparison CEL.
EVENT_ATTRIBUTES = ["text", "outcome", "status"]
EVENT_ATTRIBUTE_VALUES = ["failed", "passed", "approved", "rejected", "completed"]

# Common object-relationship qualifiers observed across the logs (F3).
QUALIFIERS = ["comprises", "order", "item", "product", "CR for TD", "CR contains HU"]

TIME_UNITS_SECONDS = {
    "hour(s)": 3600, "day(s)": 86400, "week(s)": 604800, "minute(s)": 60,
}

# Pre-computed frozensets for O(1) domain detection by event-list fingerprint.
_DOMAIN_EVENT_SETS: dict = {dom: frozenset(dd["events"]) for dom, dd in domains.items()}


def _domain_attributes(domain: str) -> dict:
    """Return {obj_type: [{"name", "type", "values"}]} for the domain's known attributes."""
    result: dict = {}
    for spec in OBJECT_ATTRIBUTES.get(domain, []):
        obj, attr, kind = spec[0], spec[1], spec[2]
        svals = list(spec[3]) if len(spec) > 3 else None
        result.setdefault(obj, []).append({"name": attr, "type": kind, "values": svals})
    return result


def _detect_domain(example_events: list) -> str:
    """Return the domain name whose event list matches example_events exactly, or None."""
    ev_set = frozenset(example_events)
    for dom, dom_evs in _DOMAIN_EVENT_SETS.items():
        if ev_set == dom_evs:
            return dom
    return None


def _box(children, ev=None, ob=None, filters=None, size_filters=None,
         constraints=None, ev_labels=None, ob_labels=None, labels=None):
    """Build a single BindingBox node. `children` is the list of child node indices."""
    data = {
        "newEventVars": ev or {},
        "newObjectVars": ob or {},
        "filters": filters or [],
        "sizeFilters": size_filters or [],
        "constraints": constraints or [],
        "evVarLabels": ev_labels or {},
        "obVarLabels": ob_labels or {},
    }
    if labels:
        data["labels"] = labels
    return {"Box": [data, children]}


def _o2e(obj, event, qualifier=None):
    return {"type": "O2E", "object": obj, "event": event, "qualifier": qualifier}


def _o2o(obj, other, qualifier=None):
    return {"type": "O2O", "object": obj, "other_object": other, "qualifier": qualifier}


def _tbe(from_event, to_event, min_seconds=0, max_seconds=None):
    return {"type": "TimeBetweenEvents", "from_event": from_event, "to_event": to_event,
            "min_seconds": min_seconds, "max_seconds": max_seconds}


def _rand_value_filter(kind, string_values=None):
    """Return (value_filter_dict, human_readable_phrase) for a given attribute kind."""
    if kind == "Float":
        thr = random.choice([50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0])
        if random.random() < 0.5:
            return {"type": "Float", "min": thr, "max": None}, f"at least {thr:g}"
        return {"type": "Float", "min": None, "max": thr}, f"at most {thr:g}"
    if kind == "Integer":
        thr = random.choice([2, 5, 10, 18, 25, 30, 580, 620, 650])
        if random.random() < 0.5:
            return {"type": "Integer", "min": thr, "max": None}, f"at least {thr}"
        return {"type": "Integer", "min": None, "max": thr}, f"at most {thr}"
    if kind == "String":
        vals = string_values or ["active", "inactive"]
        v = random.choice(vals)
        return {"type": "String", "is_in": [v]}, f"equal to '{v}'"
    if kind == "Boolean":
        b = random.choice([True, False])
        return {"type": "Boolean", "is_true": b}, ("true" if b else "false")
    # Time
    year = random.choice([2020, 2021, 2022, 2023])
    return ({"type": "Time", "from": f"{year}-01-01T00:00:00Z", "to": f"{year}-12-31T23:59:59Z"},
            f"during {year}")


def _pick_object_attribute(kinds=None):
    """Pick a (domain, object, attr_name, kind, string_values) tuple, optionally by kind."""
    candidates = []
    for dom, specs in OBJECT_ATTRIBUTES.items():
        for spec in specs:
            obj, name, kind = spec[0], spec[1], spec[2]
            svals = spec[3] if len(spec) > 3 else None
            if kinds is None or kind in kinds:
                candidates.append((dom, obj, name, kind, svals))
    return random.choice(candidates)


def _cel_sum(edge, obj_index, attr):
    """CEL aggregation idiom: sum an object attribute over an edge's child bindings.
    obj_index is the 0-based object-variable index; CEL uses 1-based naming (o1, o2...)."""
    return f"{edge}.map(b, b['o{obj_index + 1}'].attr('{attr}')).sum().int()"


def _two_events(domain_data):
    a = random.choice(domain_data["events"])
    others = [e for e in domain_data["events"] if e != a]
    b = random.choice(others) if others else a
    return a, b


def _three_events(domain_data):
    evs = domain_data["events"]
    if len(evs) >= 3:
        return tuple(random.sample(evs, 3))
    a, b = _two_events(domain_data)
    return a, b, b


# --- Extended builders (one per missing building block) -------------------------------

def _build_qualifier():
    """F3: O2O / O2E restricted to a named qualifier."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    qual = random.choice(QUALIFIERS)
    if len(dd["objects"]) >= 2 and random.random() < 0.5:
        obj_a, obj_b = random.sample(dd["objects"], 2)
        nl = f"An {obj_a} must only be linked to an {obj_b} through the '{qual}' relationship."
        tree = {
            "nodes": [
                _box([1], ob={"0": [obj_a]}),
                _box([], ob={"1": [obj_b]}, filters=[_o2o(0, 1, qual)]),
            ],
            "edgeNames": [[[0, 1], "A"]],
        }
        reasoning = (f"Step 1: The query involves objects '{obj_a}' and '{obj_b}'. "
                     f"Step 2: They must be related specifically via the '{qual}' qualifier. "
                     f"Step 3: We place '{obj_a}' in the root and connect a child '{obj_b}' node "
                     f"bound with an O2O filter whose qualifier is '{qual}'.")
    else:
        event = random.choice(dd["events"])
        obj_a = random.choice(dd["objects"])
        nl = f"A '{event}' event must be related to the {obj_a} object via the '{qual}' qualifier."
        tree = {
            "nodes": [_box([], ev={"0": [event]}, ob={"0": [obj_a]}, filters=[_o2e(0, 0, qual)])],
            "edgeNames": [],
        }
        reasoning = (f"Step 1: The query involves event '{event}' and object '{obj_a}'. "
                     f"Step 2: They must be linked through the named qualifier '{qual}'. "
                     f"Step 3: A single node binds both, with an O2E filter carrying qualifier '{qual}'.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_tbe_min():
    """F6: TimeBetweenEvents with min_seconds > 0."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    unit = random.choice(list(TIME_UNITS_SECONDS.keys()))
    n = random.choice([1, 2, 3, 6])
    secs = n * TIME_UNITS_SECONDS[unit]
    nl = f"The '{b}' event must occur at least {n} {unit} after the '{a}' event."
    tree = {
        "nodes": [_box([], ev={"0": [a], "1": [b]},
                       filters=[_tbe(0, 1, min_seconds=secs, max_seconds=None)])],
        "edgeNames": [],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}'. "
                 f"Step 2: '{b}' must come at least {n} {unit} ({secs} seconds) after '{a}'. "
                 f"Step 3: A single node holds both events with a TimeBetweenEvents filter whose "
                 f"min_seconds is {secs} and max_seconds is unbounded.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_tbe_simultaneous():
    """F7: TimeBetweenEvents with max_seconds = 0 (events overlap in time)."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    nl = f"The '{a}' and '{b}' events must occur at the same time."
    tree = {
        "nodes": [_box([], ev={"0": [a], "1": [b]},
                       filters=[_tbe(0, 1, min_seconds=0, max_seconds=0)])],
        "edgeNames": [],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}'. "
                 f"Step 2: They must happen simultaneously. "
                 f"Step 3: A single node binds both events with a TimeBetweenEvents filter where "
                 f"both min_seconds and max_seconds are 0.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_not_equal():
    """F8 (+V5): two event variables that must be distinct instances."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"When a '{a}' and a '{b}' event occur for the same {obj}, the two events must be "
          f"distinct occurrences.")
    tree = {
        "nodes": [_box([], ev={"0": [a], "1": [b]}, ob={"0": [obj]},
                       filters=[_o2e(0, 0), _o2e(0, 1),
                                {"type": "NotEqual", "var_1": {"Event": 0}, "var_2": {"Event": 1}}])],
        "edgeNames": [],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}' and object '{obj}'. "
                 f"Step 2: Both events are tied to the same '{obj}' but must be different events. "
                 f"Step 3: A single node binds both events and the object via O2E filters, and a "
                 f"NotEqual filter forces the two event variables to be distinct.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_event_attr():
    """F9: EventAttributeValueFilter on an event attribute, followed by another event."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    attr = random.choice(EVENT_ATTRIBUTES)
    val = random.choice(EVENT_ATTRIBUTE_VALUES)
    nl = f"A '{a}' event whose {attr} is '{val}' must be followed by a '{b}' event."
    tree = {
        "nodes": [
            _box([1], ev={"0": [a]},
                 filters=[{"type": "EventAttributeValueFilter", "event": 0,
                           "attribute_name": attr, "value_filter": {"type": "String", "is_in": [val]}}],
                 constraints=[{"type": "SAT", "child_names": ["A"]},
                              {"type": "SizeFilter",
                               "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}]),
            _box([], ev={"1": [b]}, filters=[_tbe(0, 1, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"]],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}', filtered on the event "
                 f"attribute '{attr}'. "
                 f"Step 2: Only '{a}' events whose {attr} equals '{val}' count, and each must be "
                 f"followed by '{b}'. "
                 f"Step 3: The root binds '{a}' with an EventAttributeValueFilter and requires a "
                 f"satisfying child; the child node binds '{b}' after '{a}' via TimeBetweenEvents.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_object_attr(at_time_mode):
    """F10-F17: ObjectAttributeValueFilter across kinds and at_time modes."""
    domain, obj, attr, kind, svals = _pick_object_attribute()
    dd = domains[domain]
    vf, phrase = _rand_value_filter(kind, svals)
    event = random.choice(dd["events"])

    if at_time_mode == "AtEvent":
        at_time = {"type": "AtEvent", "event": 0}
        nl = (f"At the time of the '{event}' event, the {obj}'s {attr} must be {phrase}.")
        filt = {"type": "ObjectAttributeValueFilter", "object": 0, "attribute_name": attr,
                "at_time": at_time, "value_filter": vf}
        tree = {
            "nodes": [_box([], ev={"0": [event]}, ob={"0": [obj]},
                           filters=[_o2e(0, 0)],
                           constraints=[{"type": "Filter", "filter": filt}])],
            "edgeNames": [],
        }
        when = f"evaluated at the moment of the '{event}' event"
    else:
        at_time = {"type": at_time_mode}
        word = "always" if at_time_mode == "Always" else "at some point"
        nl = (f"A {obj} whose {attr} is {word} {phrase} must have a '{event}' event recorded.")
        filt = {"type": "ObjectAttributeValueFilter", "object": 0, "attribute_name": attr,
                "at_time": at_time, "value_filter": vf}
        tree = {
            "nodes": [
                _box([1], ob={"0": [obj]},
                     filters=[filt],
                     constraints=[{"type": "SizeFilter",
                                   "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}]),
                _box([], ev={"0": [event]}, filters=[_o2e(0, 0)]),
            ],
            "edgeNames": [[[0, 1], "A"]],
        }
        when = f"holding {'at all times' if at_time_mode == 'Always' else 'at some time-point'}"
    reasoning = (f"Step 1: The query involves object '{obj}' (attribute '{attr}') and event '{event}'. "
                 f"Step 2: The {attr} must be {phrase}, {when}. "
                 f"Step 3: An ObjectAttributeValueFilter with at_time '{at_time_mode}' restricts the "
                 f"'{obj}' bindings before the structural check on '{event}'.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_basic_cel():
    """F18: BasicFilterCEL (inequality or attribute comparison between two events)."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    obj = random.choice(dd["objects"])
    if random.random() < 0.5:
        cel = "e1 != e2"
        nl = (f"A '{a}' and a '{b}' event for the same {obj} must be different events.")
    else:
        attr = random.choice(EVENT_ATTRIBUTES)
        cel = f'e1.attr("{attr}") == e2.attr("{attr}")'
        nl = (f"A '{a}' and a '{b}' event for the same {obj} must share the same {attr}.")
    tree = {
        "nodes": [_box([], ev={"0": [a], "1": [b]}, ob={"0": [obj]},
                       filters=[_o2e(0, 0), _o2e(0, 1)],
                       constraints=[{"type": "Filter", "filter": {"type": "BasicFilterCEL", "cel": cel}}])],
        "edgeNames": [],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}' and object '{obj}'. "
                 f"Step 2: A relationship between the two events is expressed directly in CEL. "
                 f"Step 3: A single node binds both events to '{obj}' and adds a BasicFilterCEL "
                 f"constraint with the expression `{cel}`.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_advanced_cel():
    """S3 + L1: AdvancedCEL size filter over a CEL label aggregating an object attribute.
    Mirrors the gold testset form: a box-level `labels` field plus a SizeFilter-wrapped
    AdvancedCEL constraint that references the label by name."""
    domain, obj_b, attr, kind, _ = _pick_object_attribute(kinds=["Float", "Integer"])
    dd = domains[domain]
    obj_a = random.choice([o for o in dd["objects"] if o != obj_b] or [obj_b])
    event = random.choice(dd["events"])
    label = random.choice(["total_volume", "total_amount", "total_value"])
    op_text = random.choice(["below", "above", "at least"])
    op = {"below": "<", "above": ">", "at least": ">="}[op_text]
    num = random.choice([100, 1000, 5000, 50000, 300000])
    nl = (f"Find all {obj_a} whose total {label} is {op_text} {num}, where the total is the sum of "
          f"the {attr} of every associated {obj_b} that had a '{event}' event.")
    tree = {
        "nodes": [
            _box([1], ob={"0": [obj_a]},
                 constraints=[{"type": "SizeFilter",
                               "filter": {"type": "AdvancedCEL", "cel": f"{label} {op} {num}"}}],
                 labels=[{"label": label, "cel": _cel_sum("A", 1, attr)}]),
            _box([2], ob={"1": [obj_b]}, filters=[_o2o(0, 1)],
                 size_filters=[{"type": "NumChilds", "child_name": "B", "min": 1, "max": None}]),
            _box([], ev={"0": [event]}, filters=[_o2e(1, 0)]),
        ],
        "edgeNames": [[[0, 1], "A"], [[1, 2], "B"]],
    }
    reasoning = (f"Step 1: The query involves objects '{obj_a}', '{obj_b}', event '{event}', and the "
                 f"aggregated attribute '{attr}'. "
                 f"Step 2: For each '{obj_a}', the sum of '{attr}' over its '{obj_b}' (with '{event}') "
                 f"must be {op_text} {num}. "
                 f"Step 3: Node 0 carries a CEL label '{label}' summing the attribute over edge A and "
                 f"an AdvancedCEL SizeFilter comparing it to {num}; node 1 links '{obj_b}' via O2O; "
                 f"node 2 binds '{event}' via O2E.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_any():
    """C4: ANY (existential over child bindings)."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"For any '{b}' event of a {obj}, at least one '{a}' event must have occurred beforehand.")
    tree = {
        "nodes": [
            _box([1], ev={"0": [b]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "ANY", "child_names": ["A"]}]),
            _box([], ev={"1": [a]}, filters=[_o2e(0, 1), _tbe(1, 0, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"]],
    }
    reasoning = (f"Step 1: The query involves events '{a}' and '{b}' and object '{obj}'. "
                 f"Step 2: Each '{b}' must have at least one prior '{a}' for the same '{obj}'. "
                 f"Step 3: The root binds '{b}' and '{obj}'; an ANY constraint requires at least one "
                 f"satisfying child binding '{a}' before '{b}' (TimeBetweenEvents).")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_or():
    """C6 (+V3): OR over two child branches after an anchor event."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b, c = _three_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"After a '{a}' event, a {obj} must undergo either a '{b}' or a '{c}' event.")
    tree = {
        "nodes": [
            _box([1, 2], ev={"0": [a]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "OR", "child_names": ["A", "B"]}]),
            _box([], ev={"1": [b]}, filters=[_o2e(0, 1), _tbe(0, 1, min_seconds=0, max_seconds=None)]),
            _box([], ev={"2": [c]}, filters=[_o2e(0, 2), _tbe(0, 2, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"], [[0, 2], "B"]],
    }
    reasoning = (f"Step 1: The query involves events '{a}', '{b}', '{c}' and object '{obj}'. "
                 f"Step 2: After '{a}', at least one of '{b}' or '{c}' must follow for the '{obj}'. "
                 f"Step 3: The root binds '{a}' and '{obj}' and branches to two child nodes ('{b}', "
                 f"'{c}'); an OR constraint over edges A and B requires at least one to be satisfied.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_and():
    """C7: AND over two child branches that must both precede an anchor event."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b, c = _three_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"Before a '{c}' event, a {obj} must have completed both a '{a}' and a '{b}' event.")
    tree = {
        "nodes": [
            _box([1], ev={"0": [c]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "SAT", "child_names": ["A"]},
                              {"type": "SizeFilter",
                               "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}]),
            _box([2, 3], constraints=[{"type": "AND", "child_names": ["B", "C"]}]),
            _box([], ev={"1": [a]}, filters=[_o2e(0, 1), _tbe(1, 0, min_seconds=0, max_seconds=None)]),
            _box([], ev={"2": [b]}, filters=[_o2e(0, 2), _tbe(2, 0, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"], [[1, 2], "B"], [[1, 3], "C"]],
    }
    reasoning = (f"Step 1: The query involves events '{a}', '{b}', '{c}' and object '{obj}'. "
                 f"Step 2: Before '{c}', the '{obj}' must have both '{a}' and '{b}'. "
                 f"Step 3: The root binds '{c}' and '{obj}'; an intermediate node uses an AND "
                 f"constraint requiring both child branches ('{a}' via edge B, '{b}' via edge C), "
                 f"each ordered before '{c}'.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_ev_label():
    """L2: evVarLabels INCLUDED — return the anchor events for objects meeting a follow-up."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"List all '{a}' events for {obj} that subsequently had a '{b}' event.")
    tree = {
        "nodes": [
            _box([1], ev={"0": [a]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "SAT", "child_names": ["A"]},
                              {"type": "SizeFilter",
                               "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}],
                 ev_labels={"0": "INCLUDED"}),
            _box([], ev={"1": [b]}, filters=[_o2e(0, 1), _tbe(0, 1, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"]],
    }
    reasoning = (f"Step 1: The query asks to return '{a}' events for '{obj}' with a later '{b}'. "
                 f"Step 2: Only '{a}' events whose '{obj}' also had '{b}' afterwards qualify. "
                 f"Step 3: The root binds '{a}' and '{obj}', marks the '{a}' variable INCLUDED for "
                 f"output, and requires a satisfying child '{b}' (TimeBetweenEvents) via SizeFilter.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_ob_label():
    """L3: obVarLabels INCLUDED — return the objects that went through a two-step sequence."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    a, b = _two_events(dd)
    obj = random.choice(dd["objects"])
    nl = (f"List all {obj} that had a '{a}' event followed by a '{b}' event.")
    tree = {
        "nodes": [
            _box([1], ev={"0": [a]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "SAT", "child_names": ["A"]},
                              {"type": "SizeFilter",
                               "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}],
                 ob_labels={"0": "INCLUDED"}),
            _box([], ev={"1": [b]}, filters=[_o2e(0, 1), _tbe(0, 1, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"]],
    }
    reasoning = (f"Step 1: The query asks to return '{obj}' objects with '{a}' then '{b}'. "
                 f"Step 2: A qualifying '{obj}' has a '{a}' event followed by a '{b}' event. "
                 f"Step 3: The root binds '{a}' and '{obj}', marks the '{obj}' variable INCLUDED for "
                 f"output, and requires a satisfying later '{b}' child via SizeFilter.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_deep_chain():
    """T5: 4+-level deep chain enforcing an ordered sequence of events for an object."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    evs = dd["events"]
    seq = random.sample(evs, 4) if len(evs) >= 4 else (evs * 4)[:4]
    a, b, c, d = seq
    obj = random.choice(dd["objects"])
    nl = (f"A {obj} must follow the sequence: '{a}', then '{b}', then '{c}', then '{d}'.")
    tree = {
        "nodes": [
            _box([1], ev={"0": [a]}, ob={"0": [obj]}, filters=[_o2e(0, 0)],
                 constraints=[{"type": "SAT", "child_names": ["A"]}]),
            _box([2], ev={"1": [b]}, filters=[_o2e(0, 1), _tbe(0, 1, min_seconds=0, max_seconds=None)],
                 constraints=[{"type": "SAT", "child_names": ["B"]}]),
            _box([3], ev={"2": [c]}, filters=[_o2e(0, 2), _tbe(1, 2, min_seconds=0, max_seconds=None)],
                 constraints=[{"type": "SAT", "child_names": ["C"]}]),
            _box([], ev={"3": [d]}, filters=[_o2e(0, 3), _tbe(2, 3, min_seconds=0, max_seconds=None)]),
        ],
        "edgeNames": [[[0, 1], "A"], [[1, 2], "B"], [[2, 3], "C"]],
    }
    reasoning = (f"Step 1: The query involves the ordered events '{a}', '{b}', '{c}', '{d}' for a "
                 f"'{obj}'. "
                 f"Step 2: Each event must occur after the previous one for the same '{obj}'. "
                 f"Step 3: A four-level chain binds '{obj}' at the root and threads each successive "
                 f"event through SAT-linked child nodes with TimeBetweenEvents between consecutive steps.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_multi_type_event():
    """V3: a single event variable accepting multiple event types."""
    domain = random.choice(list(domains.keys()))
    dd = domains[domain]
    if len(dd["events"]) >= 3:
        a, b, c = random.sample(dd["events"], 3)
    else:
        a, b = _two_events(dd)
        c = b
    nl = (f"Either a '{a}' or a '{b}' event must be followed by a '{c}' event.")
    tree = {
        "nodes": [_box([], ev={"0": [a, b], "1": [c]},
                       filters=[_tbe(0, 1, min_seconds=0, max_seconds=None)])],
        "edgeNames": [],
    }
    reasoning = (f"Step 1: The query involves a multi-type event ('{a}' or '{b}') and event '{c}'. "
                 f"Step 2: Whichever of '{a}'/'{b}' occurs must be followed by '{c}'. "
                 f"Step 3: A single node declares one event variable accepting both '{a}' and '{b}', "
                 f"with a TimeBetweenEvents filter to '{c}'.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


def _build_multi_type_object():
    """V4: a single object variable accepting multiple object types."""
    domain = random.choice([d for d in domains if len(domains[d]["objects"]) >= 2])
    dd = domains[domain]
    obj_a, obj_b = random.sample(dd["objects"], 2)
    event = random.choice(dd["events"])
    nl = (f"Every {obj_a} or {obj_b} must participate in at least one '{event}' event.")
    tree = {
        "nodes": [
            _box([1], ob={"0": [obj_a, obj_b]},
                 constraints=[{"type": "SizeFilter",
                               "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}]),
            _box([], ev={"0": [event]}, filters=[_o2e(0, 0)]),
        ],
        "edgeNames": [[[0, 1], "A"]],
    }
    reasoning = (f"Step 1: The query involves a multi-type object ('{obj_a}' or '{obj_b}') and event "
                 f"'{event}'. "
                 f"Step 2: Each such object must take part in at least one '{event}'. "
                 f"Step 3: The root declares one object variable accepting both types and requires at "
                 f"least one child '{event}' bound via O2E.")
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": tree, "reasoning": reasoning}


EXT_BUILDERS = [
    _build_qualifier,
    _build_tbe_min,
    _build_tbe_simultaneous,
    _build_not_equal,
    _build_event_attr,
    lambda: _build_object_attr("Always"),
    lambda: _build_object_attr("Sometime"),
    lambda: _build_object_attr("AtEvent"),
    _build_basic_cel,
    _build_advanced_cel,
    _build_any,
    _build_or,
    _build_and,
    _build_ev_label,
    _build_ob_label,
    _build_deep_chain,
    _build_multi_type_event,
    _build_multi_type_object,
]

# Fraction of generated examples drawn from the extended (full-coverage) builders.
EXT_RATIO = 0.5


def generate_extended_example():
    """Generate one example covering an extended OCPQ building block."""
    return random.choice(EXT_BUILDERS)()

# In[5]:


from transformers import pipeline, BitsAndBytesConfig

import torch

import os
import re

# In[6]:


if train_with_rewrites:
    # Create 4-bit config to compress the ~15GB model to ~5-6GB
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Initialize the local pipeline with 4-bit quantization
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"quantization_config": bnb_config},
        device_map="auto",
        token=os.getenv("HF_TOKEN")
    )

# In[7]:


def rewrite_sentences_batch(sentences, batch_size=4):
    # Determine padding token if needed for processing batches
    if pipe.tokenizer.pad_token_id is None:
        pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id
        
    # CRITICAL FIX: Causal LMs require left-padding for batched generation!
    pipe.tokenizer.padding_side = "left"

    messages_batch = []
    for sentence in sentences:
        prompt = f"""
Rewrite the following sentence to make it sound natural and grammatically correct.
Provide TWO variations:
1. "nl_clean_1": A slightly more formal, straight-forward rewrite.
2. "nl_clean_2": A more varied and naturally flowing rewrite.

Rules:
- Keep the exact meaning.
- Do not add or remove information.
- Keep technical terms unchanged.
- Output ONLY valid JSON containing the keys "nl_clean_1" and "nl_clean_2".
- Do not include any explanations, greetings, or backticks.

Sentence:
"{sentence}"
"""
        messages_batch.append([
            {"role": "system", "content": "You are a JSON-only API. You must output raw, valid JSON tightly formatted, with no markdown formatting, no backticks, and no conversational text."},
            {"role": "user", "content": prompt}
        ])
    
    # Run the pipeline in batches (Processes `batch_size` sentences concurrently)
    outputs = pipe(
        messages_batch,
        max_new_tokens=500,
        temperature=0.5,
        do_sample=True,
        batch_size=batch_size
    )
    
    results = []
    for i, out in enumerate(outputs):
        content = out[0]["generated_text"][-1]["content"]
        try:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            clean_content = match.group(0) if match else content.strip()
            
            # Auto-heal: If the generation randomly cut off just the closing bracket, fix it
            if not clean_content.endswith("}"):
                clean_content += "\n}"
                
            data = json.loads(clean_content)
            results.append(data)
        except Exception as e:
            print(f"Failed to parse JSON for sample {i}:\nRaw Output: {content}")
            results.append({"nl_clean_1": sentences[i], "nl_clean_2": sentences[i]})
            
    return results

# In[8]:


def generate_example():
    # Draw a share of examples from the extended full-coverage builders so the dataset
    # spans all OCPQ building blocks, not just the original ~13 patterns below.
    if random.random() < EXT_RATIO:
        return generate_extended_example()

    pattern = random.choice(patterns)

    # Pick a domain first to ensure semantic consistency
    domain = random.choice(list(domains.keys()))
    domain_data = domains[domain]

    A = random.choice(domain_data["events"])
    # Ensure event A and B are different for better logical examples
    B = random.choice([e for e in domain_data["events"] if e != A])
    
    obj_a = random.choice(domain_data["objects"])
    
    # Fetch a second object if the domain has more than one
    obj_b = random.choice([o for o in domain_data["objects"] if o != obj_a]) if len(domain_data["objects"]) > 1 else obj_a

    # Randomly select a time unit for patterns that require it
    time_unit = random.choice(["hour(s)", "week(s)", "day(s)", "minute(s)", "second(s)", "month(s)", "year(s)"])
    
    # Calculate time scaling based on the time unit
    if time_unit == "year(s)":
        t = random.choice([1, 1.5, 2, 3])
        t_sec = int(t * 31536000) # Years to seconds
    elif time_unit == "month(s)":
        t = random.choice([1, 1.5, 2, 3, 6])
        t_sec = int(t * 2592000) # Months to seconds (approx 30 days)
    elif time_unit == "week(s)":
        t = random.choice([1, 1.5, 2, 3, 3.5, 4])
        t_sec = int(t * 604800) # Weeks to seconds
    elif time_unit == "day(s)":
        t = random.choice([1, 5, 10, 30])
        t_sec = int(t * 86400) # Days to seconds
    elif time_unit == "minute(s)":
        t = random.choice([5, 10, 30, 45])
        t_sec = int(t * 60) # Minutes to seconds
    elif time_unit == "second(s)":
        t = random.choice([10, 30, 60, 500])
        t_sec = int(t) # Seconds
    else: # hours
        t = random.choice([1, 6, 24, 72])
        t_sec = int(t * 3600) # Hours to seconds
    
    # Generate random numbers for pluralization and quantity-based patterns
    num = random.choice([1, 2, 5, 10, 50, 100])
    num2 = random.choice([1, 2, 3, 4, 5])

    # Generate random attributes for CEL scripts
    att_label = random.choice(["order_volume", "total_amount", "total_price"])
    att_name = random.choice(["price", "amount", "value"])
    op_text = random.choice(["below", "above", "exactly", "at least"])
    
    # Map operator language to CEL math operators
    op_map = {"below": "<", "above": ">", "exactly": "==", "at least": ">="}
    op_num = op_map[op_text]
    num3 = random.choice([500, 1000, 5000, 10000, 50000, 300000])

    # Replace placeholders in the pattern with the generated values
    text = pattern.replace("EVENT_A", A)\
                  .replace("EVENT_B", B)\
                  .replace("OBJECT_A", obj_a)\
                  .replace("OBJECT_B", obj_b)\
                  .replace("TIME_UNIT", time_unit)\
                  .replace("TIME", str(t))\
                  .replace("NUM2", str(num2))\
                  .replace("NUM3", str(num3))\
                  .replace("NUM", str(num))\
                  .replace("ATT_LABEL", att_label)\
                  .replace("ATT_NAME", att_name)\
                  .replace("OP_TEXT", op_text)

    # Generate reasoning for Least-to-Most prompting
    is_single_event = "exactly once" in pattern or "at least once" in pattern and "the EVENT_B event should happen" not in pattern

    if "exactly once" in pattern:
        reasoning = f"Step 1: The query involves the event '{A}' and the object '{obj_a}'. "
        reasoning += f"Step 2: The constraint specifies that '{A}' must occur exactly once for '{obj_a}'. "
        reasoning += f"Step 3: To prevent unbounded Cartesian products, we connect the logic structurally. We place '{obj_a}' in the root node, and link to a connected child node containing '{A}'. We bind them with an explicit `O2E` filter in the child node, and enforce the exactly 1 constraint via a SizeFilter in the parent."
    elif "at least once" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and object '{obj_a}'. "
        reasoning += f"Step 2: The constraint specifies that after '{A}' occurs for '{obj_a}', '{B}' must also occur at least once. "
        reasoning += f"Step 3: To prevent unbounded searches, we structure this as a connected tree. The parent node binds '{A}' and '{obj_a}' explicitly with an `O2E` filter, and links to a connected child node for '{B}'. We bind '{B}' back to the object using an `O2E` filter strictly inside this child node."
    elif "total is calculated" in pattern:
        reasoning = f"Step 1: The query involves event '{A}', objects '{obj_a}', '{obj_b}' and attributes '{att_label}', '{att_name}'. "
        reasoning += f"Step 2: The constraint limits '{obj_a}' based on the sum of '{att_name}' for all its '{obj_b}' linked to '{A}'. The sum must be {op_text} {num3}. "
        reasoning += f"Step 3: We establish a 3-node connected hierarchy. Node 0 is for '{obj_a}' with an AdvancedCEL SizeFilter mapping the sum to '{att_label}'. Node 1 is an object node for '{obj_b}' linking to Node 0. Node 2 is an event node for '{A}' connecting to Node 1 via O2E."
    elif "at least NUM2" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: The constraint expects '{obj_a}' to have at least {num} '{obj_b}' where '{B}' occurs at least {num2} times after '{A}'. "
        reasoning += f"Step 3: To prevent slow queries, we construct a fully connected 3-node structure: '{obj_a}' links to '{obj_b}'+'{A}' (bound explicitly with `O2E`), which then connects to a child node for '{B}' (also bound with `O2E`). Filters are embedded directly inside these connected child nodes to safely bound the search."
    elif "completed within" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}' (if applicable). "
        reasoning += f"Step 2: '{B}' must be completed within {t} {time_unit} after '{A}' for any '{obj_a}'. "
        reasoning += f"Step 3: To ensure safe query bounds, we use a connected parent-child structure. The SAT constraint is placed in the parent for '{A}', linking to a connected child node for '{B}'. Inside the connected child node, we place the `O2E` bindings and the TimeBetweenEvents filter ({t_sec} seconds) so they safely constrain the search path."
    elif "at most NUM" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: For any '{obj_a}' with at most {num} '{obj_b}', '{B}' must occur within {t} {time_unit} after '{A}'. "
        reasoning += f"Step 3: To avoid query explosions, we connect the nodes structurally. The root node for '{A}' and '{obj_a}' (bound via `O2E`) branches to two connected child nodes: one for '{obj_b}' (capped by SizeFilter max {num}) and one for '{B}'. Inside the '{B}' child node, we explicitly place the `O2E` and TimeBetweenEvents ({t_sec} secs) filters to constrain execution."
    elif "for which the EVENT_A" in pattern:
        reasoning = f"Step 1: The query involves event '{A}' and objects '{obj_a}' and '{obj_b}'. "
        reasoning += f"Step 2: Each '{obj_a}' must have at least {num} associated '{obj_b}' where '{A}' occurs. "
        reasoning += f"Step 3: To prevent unbounded searches, we establish a connected hierarchy. A parent node for '{obj_a}' with a SizeFilter (at least {num}) links to a connected child node containing '{A}' and '{obj_b}'. We explicitly bind them using `O2O` and `O2E` filters strictly inside this child node."
    elif "before" in pattern or "eventually" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{A}' occurs before or eventually leads to '{B}'. "
        reasoning += f"Step 3: To inherently prevent isolated events from forming a Cartesian product, we map them into a single connected node with a TimeBetweenEvents filter from '{A}' to '{B}' with no maximum time."
    elif "within TIME" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{B}' happens within {t} {time_unit} after '{A}'. "
        reasoning += f"Step 3: To inherently prevent isolated events from forming a Cartesian product, we map them into a single connected node with a TimeBetweenEvents filter from '{A}' to '{B}' up to {t_sec} seconds."
    elif "must not occur" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}'. "
        reasoning += f"Step 2: '{B}' must not occur after '{A}'. "
        reasoning += f"Step 3: A single node referencing a child edge inside a NOT constraint."
    elif "refer to the same" in pattern:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and object '{obj_a}'. "
        reasoning += f"Step 2: '{A}' and '{B}' must refer to the same '{obj_a}'. "
        reasoning += f"Step 3: To explicitly correlate them and avoid unbounded Cartesian searches, we place '{A}', '{B}', and '{obj_a}' into a single node and add exact `O2E` link filters for both events."
    else:
        reasoning = f"Step 1: The query involves events '{A}' and '{B}', and objects '{obj_a}' and '{obj_b}' (if applicable). "
        reasoning += "Step 2: Relate the identified elements. Step 3: Map to generic OCPQ single node."

    # Pattern handling: multi-node vs single-node trees
    if "exactly once" in pattern:
        # Two-node exactly once mapping (min: 1, max: 1)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": 1}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
        
    elif "at least once" in pattern:
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": None}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None},
                                {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "total is calculated" in pattern:
        # AdvancedCEL 3-node structure. Mirrors the gold testset form: a box-level `labels`
        # field defines the aggregation, and a SizeFilter-wrapped AdvancedCEL constraint
        # compares the named label against the threshold.
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [
                                {
                                    "type": "SizeFilter",
                                    "filter": {"type": "AdvancedCEL", "cel": f"{att_label} {op_num} {num3}"}
                                }
                            ],
                            "evVarLabels": {},
                            "obVarLabels": {},
                            "labels": [
                                {"label": att_label, "cel": f"A.map(b, b['o2'].attr('{att_name}')).sum().int()"}
                            ]
                        },
                        [1] # Contains child 1 (O2O)
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None}
                            ],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "B", "min": 1, "max": None}],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [2] # Contains child 2 (O2E)
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[1, 2], "B"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "at least NUM2" in pattern:
        # Deep nested tree structure (A -> B -> C)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "A", "min": None, "max": num}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None},
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "B", "min": num2, "max": None}],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [2] # Connects to child B
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 1, "event": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[1, 2], "B"]] # Ordered hierarchical edges
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    elif "completed within" in pattern:
        # Parent-child structure using the SAT constraint
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [],
                            "constraints": [{"type": "SAT", "child_names": ["A"]}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child A
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A], "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None},
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [
                                {
                                    "type": "Filter",
                                    "filter": {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec}
                                }
                            ],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
        
    elif "at most NUM" in pattern:
        # Multi-node structure with two distinct child branches
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "A", "min": None, "max": num}],
                            "constraints": [{"type": "SizeFilter", "filter": {"type": "NumChilds", "child_name": "B", "min": 1, "max": None}}],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1, 2] # Connects to child A and B
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "1": [B] },
                            "newObjectVars": {},
                            "filters": [
                                {"type": "O2E", "object": 0, "event": 1, "qualifier": None},
                                {"type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"], [[0, 2], "B"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}
    
    elif "for which the EVENT_A" in pattern:
        # Construct the OCPQ tree representing the parent-child node logic 
        # (e.g. Customers with at least 50 orders involving payment reminder)
        ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": {},
                            "newObjectVars": { "0": [obj_a] },
                            "filters": [],
                            "sizeFilters": [{"type": "NumChilds", "child_name": "A", "min": num, "max": None}],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [1] # Connects to child node at index 1
                    ]
                },
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A] },
                            "newObjectVars": { "1": [obj_b] },
                            "filters": [
                                {"type": "O2O", "object": 0, "other_object": 1, "qualifier": None},
                                {"type": "O2E", "object": 1, "event": 0, "qualifier": None}
                            ],
                            "sizeFilters": [],
                            "constraints": [],
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # No further children
                    ]
                }
            ],
            "edgeNames": [[[0, 1], "A"]]
        }
        return {"nl": text,
                "events": domain_data["events"], 
                "objects": domain_data["objects"], 
                "query": ocpq_tree, 
                "reasoning": reasoning}

    # Dynamically inject constraints based on single-node patterns
    constraints = []
    filters = []
    if "before" in pattern or "eventually" in pattern:
        constraints = [{
            "type": "Filter",
            "filter": { "type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": None }
        }]
    elif "within TIME" in pattern:
        constraints = [{
            "type": "Filter",
            "filter": { "type": "TimeBetweenEvents", "from_event": 0, "to_event": 1, "min_seconds": 0, "max_seconds": t_sec }
        }]
    elif "must not occur" in pattern:
        # Requires a child edge with a NOT constraint to be completely valid OCPQ
        constraints = [{
            "type": "NOT",
            "child_names": ["edge_1"]
        }]
    elif "refer to the same" in pattern:
        filters = [
            { "type": "O2E", "object": 0, "event": 0, "qualifier": None },
            { "type": "O2E", "object": 0, "event": 1, "qualifier": None }
        ]

    ocpq_tree = {
            "nodes": [
                {
                    "Box": [
                        {
                            "newEventVars": { "0": [A], "1": [B] },
                            "newObjectVars": { "0": [obj_a] } if "refer to the same" in pattern else {},
                            "filters": filters,
                            "sizeFilters": [],
                            "constraints": constraints,
                            "evVarLabels": {},
                            "obVarLabels": {}
                        },
                        [] # IDs of any connected child nodes in the tree
                    ]
                }
            ],
            "edgeNames": []
    }

    return {"nl": text, 
            "events": domain_data["events"], 
            "objects": domain_data["objects"], 
            "query": ocpq_tree, 
            "reasoning": reasoning}


def _build_targeted_curriculum_examples():
    hospital_events = domains["hospital"]["events"]
    hospital_objects = domains["hospital"]["objects"]

    targeted_examples = [
        {
            "nl": "For any person, the diagnosis happens exactly once.",
            "nl_clean_1": "For any person, the diagnosis must happen exactly once.",
            "nl_clean_2": "Each person should receive the diagnosis exactly one time.",
            "events": hospital_events,
            "objects": hospital_objects,
            "query": {
                "nodes": [
                    {
                        "Box": [
                            {
                                "newEventVars": {},
                                "newObjectVars": {"0": ["Patient"]},
                                "filters": [],
                                "sizeFilters": [],
                                "constraints": [
                                    {
                                        "type": "SizeFilter",
                                        "filter": {"type": "NumChilds", "child_name": "A", "min": 1, "max": 1},
                                    }
                                ],
                                "evVarLabels": {},
                                "obVarLabels": {},
                            },
                            [1],
                        ]
                    },
                    {
                        "Box": [
                            {
                                "newEventVars": {"0": ["Initial Evaluation"]},
                                "newObjectVars": {},
                                "filters": [{"type": "O2E", "object": 0, "event": 0, "qualifier": None}],
                                "sizeFilters": [],
                                "constraints": [],
                                "evVarLabels": {},
                                "obVarLabels": {},
                            },
                            [],
                        ]
                    },
                ],
                "edgeNames": [[[0, 1], "A"]],
            },
            "reasoning": (
                "Step 1: Interpret the colloquial term 'person' as the allowed object 'Patient' and "
                "'diagnosis' as the allowed event 'Initial Evaluation'. Step 2: The phrase 'exactly once' "
                "must not introduce a custom filter type. Step 3: Represent it with a parent object node "
                "and a SizeFilter min=1,max=1 on a child event node that is linked back with O2E."
            ),
        },
        {
            "nl": "For any person, the diagnosis must be completed within 3 days after the registration at ER.",
            "nl_clean_1": "For any person, diagnosis must be completed within 3 days after ER registration.",
            "nl_clean_2": "Each person should complete diagnosis within 3 days after registering at the ER.",
            "events": hospital_events,
            "objects": hospital_objects,
            "query": {
                "nodes": [
                    {
                        "Box": [
                            {
                                "newEventVars": {},
                                "newObjectVars": {"0": ["Patient"]},
                                "filters": [],
                                "sizeFilters": [],
                                "constraints": [{"type": "SAT", "child_names": ["A"]}],
                                "evVarLabels": {},
                                "obVarLabels": {},
                            },
                            [1],
                        ]
                    },
                    {
                        "Box": [
                            {
                                "newEventVars": {"0": ["Registration"], "1": ["Initial Evaluation"]},
                                "newObjectVars": {},
                                "filters": [
                                    {"type": "O2E", "object": 0, "event": 0, "qualifier": None},
                                    {"type": "O2E", "object": 0, "event": 1, "qualifier": None},
                                ],
                                "sizeFilters": [],
                                "constraints": [
                                    {
                                        "type": "Filter",
                                        "filter": {
                                            "type": "TimeBetweenEvents",
                                            "from_event": 0,
                                            "to_event": 1,
                                            "min_seconds": 0,
                                            "max_seconds": 259200,
                                        },
                                    }
                                ],
                                "evVarLabels": {},
                                "obVarLabels": {},
                            },
                            [],
                        ]
                    },
                ],
                "edgeNames": [[[0, 1], "A"]],
            },
            "reasoning": (
                "Step 1: Interpret 'person' as 'Patient', 'registration at ER' as 'Registration', and "
                "'diagnosis' as 'Initial Evaluation'. Step 2: The temporal constraint is 'within 3 days after', "
                "so it must become a TimeBetweenEvents max_seconds of 259200. Step 3: Keep the tree connected "
                "with a parent SAT node and child event node using O2E bindings for both events."
            ),
        },
    ]

    return targeted_examples

# In[9]:


def _is_valid_query(query):
    """Validate a generated OCPQ tree against the Pydantic schema. Returns True if it parses."""
    try:
        BindingBoxTree.model_validate(query)
        return True
    except ValidationError:
        return False


def _census(raw_examples):
    """Tally building-block usage across generated queries for a coverage report."""
    counts = {}

    def bump(key):
        counts[key] = counts.get(key, 0) + 1

    for ex in raw_examples:
        query = ex.get("query", {})
        for node in query.get("nodes", []):
            data = node.get("Box", [{}])[0]
            for f in data.get("filters", []):
                bump(f"filter:{f.get('type')}")
                if f.get("type") in ("O2E", "O2O") and f.get("qualifier"):
                    bump("filter:qualifier")
                if f.get("type") == "ObjectAttributeValueFilter":
                    bump(f"at_time:{f.get('at_time', {}).get('type')}")
            for sf in data.get("sizeFilters", []):
                bump(f"size:{sf.get('type')}")
            for c in data.get("constraints", []):
                bump(f"constraint:{c.get('type')}")
                inner = c.get("filter", {})
                if isinstance(inner, dict) and inner.get("type"):
                    bump(f"wrapped:{inner.get('type')}")
                    if inner.get("type") == "ObjectAttributeValueFilter":
                        bump(f"at_time:{inner.get('at_time', {}).get('type')}")
                    vf = inner.get("value_filter")
                    if isinstance(vf, dict) and vf.get("type"):
                        bump(f"value_filter:{vf.get('type')}")
            for f in data.get("filters", []):
                vf = f.get("value_filter")
                if isinstance(vf, dict) and vf.get("type"):
                    bump(f"value_filter:{vf.get('type')}")
            if data.get("evVarLabels"):
                bump("label:evVarLabels")
            if data.get("obVarLabels"):
                bump("label:obVarLabels")
            if data.get("labels"):
                bump("label:CEL")
            for vs in list(data.get("newEventVars", {}).values()) + list(data.get("newObjectVars", {}).values()):
                if isinstance(vs, list) and len(vs) > 1:
                    bump("var:multi-type")
        bump(f"depth:{len(query.get('nodes', []))}-nodes")
    return counts


def generate_dataset(num_samples=500, batch_size=4, original_sentence=True, first_rewrite=True, second_rewrite=True, output_file="ocpq_dataset.json"):
    dataset = []
    seen = set()
    invalid_dropped = 0

    # Block all testset NLs from entering the fine-tuning pool (contamination guard).
    testset_path = "testset.json"
    if os.path.exists(testset_path):
        with open(testset_path, "r", encoding="utf-8") as _f:
            _ts = json.load(_f)
        for _item in _ts:
            seen.add(_item.get("nl_query", _item.get("nl", "")).strip())
        print(f"Blocked {len(_ts)} testset NLs from training pool.")

    # 1. Seed the dataset with targeted curriculum examples for fragile semantics.
    raw_examples = []
    for example in _build_targeted_curriculum_examples():
        if example["nl"] not in seen:
            seen.add(example["nl"])
            dom = _detect_domain(example.get("events", []))
            if dom:
                example["object_attributes"] = _domain_attributes(dom)
            raw_examples.append(example)

    # 2. Instantly generate the remaining unique logical sentences locally.
    #    Every query is validated against the OCPQ schema before it is accepted, so the
    #    many extended builders cannot leak malformed trees into the training pool.
    print("Generating pure logical combinations...")
    while len(raw_examples) < num_samples:
        ex = generate_example()
        if ex["nl"] in seen:
            continue
        if not _is_valid_query(ex["query"]):
            invalid_dropped += 1
            continue
        seen.add(ex["nl"])
        dom = _detect_domain(ex.get("events", []))
        if dom:
            ex["object_attributes"] = _domain_attributes(dom)
        raw_examples.append(ex)

    # 3. Process the LLM rewrites in parallel batches
    print(f"Batch processing text inference on GPU (batch_size={batch_size})...")
    for i in range(0, len(raw_examples), batch_size):
        batch = raw_examples[i:i+batch_size]
        sentences = [ex["nl"] for ex in batch]
        
        if first_rewrite or second_rewrite:
            # Send 'batch_size' queries to the GPU at once instead of 1
            rewrites_batch = rewrite_sentences_batch(sentences, batch_size=batch_size)
        else:
            rewrites_batch = [{} for _ in sentences]
            
        for j, ex in enumerate(batch):
            updated_example = {}
            if original_sentence:
                updated_example["nl"] = ex["nl"]

            if first_rewrite:
                updated_example["nl_clean_1"] = ex.get("nl_clean_1", rewrites_batch[j].get("nl_clean_1", ex["nl"]))
            elif "nl_clean_1" in ex:
                updated_example["nl_clean_1"] = ex["nl_clean_1"]

            if second_rewrite:
                updated_example["nl_clean_2"] = ex.get("nl_clean_2", rewrites_batch[j].get("nl_clean_2", ex["nl"]))
            elif "nl_clean_2" in ex:
                updated_example["nl_clean_2"] = ex["nl_clean_2"]

            updated_example["events"] = ex["events"]
            updated_example["objects"] = ex["objects"]
            updated_example["query"] = ex["query"]
            updated_example["reasoning"] = ex["reasoning"]
            dataset.append(updated_example)
            
        print(f"Processed {len(dataset)} / {num_samples} examples", end="\r")

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print("\nDataset generation completed!")
    if invalid_dropped:
        print(f"Dropped {invalid_dropped} schema-invalid queries during generation.")
    print("Building-block coverage census:")
    for key, count in sorted(_census(raw_examples).items()):
        print(f"  {key}: {count}")
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset(num_samples=20000, first_rewrite=False, second_rewrite=False)
    print(dataset[:5])
