#!/usr/bin/env python
# coding: utf-8

# In[1]:


import hashlib
import os
import random
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import InferenceClient

from pydantic import ValidationError

from ocpq_schema import BindingBoxTree

# In[2]:


train_with_rewrites = False

# Corpus generation is nothing but randomness -- 74 `random.*` draws -- and was previously
# unseeded, so no generated corpus could be reproduced and every regeneration silently
# produced a different training set. CLAUDE.md 2.1 fixes the canonical seed at 3407.
#
# Seeding is sufficient on its own here; PYTHONHASHSEED is not needed. Every `random.choice`
# / `random.sample` draws from a list built in dict or list insertion order, or from an
# explicitly sorted sequence (`_e2o_object`, `_o2o_pair`, `_no_direct_path_triples`), and
# every set-derived output field is sorted before being emitted (`_query_relations`). The
# `random` module is the only randomness source: no numpy RNG, no uuid, no clock.
GLOBAL_SEED = int(os.getenv("OCPQ_DATASET_SEED", "3407"))
random.seed(GLOBAL_SEED)

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
    # "e-commerce" vocabulary was previously customers/employees/items/orders/packages/
    # products, which is a 100% object-vocabulary match (plus the "pay order" event) with
    # Logs/order-management.json, a real evaluation log referenced by testset.json. Renamed
    # to a disjoint vocabulary (verified against every log testset.json evaluates against)
    # while keeping the plural/lowercase naming convention, for the same de-leaking reason
    # as the three invented domains further below.
    "e-commerce": {
        "events": ["open cart", "add line item", "settle cart", "dispatch parcel", "void cart"],
        "objects": ["shoppers", "workers", "line items", "carts", "parcels", "articles"]
    },
    "recruitment": {
        "events": ["submit application", "review application", "approve application", "reject application"],
        "objects": ["applications", "candidates"]
    },
    "software_dev": {
        "events": ["open issue", "submit pull request", "review pull request", "merge pull request", "close issue"],
        "objects": ["issues", "pull requests", "commits"]
    },

    # De-leaked replacement domains (invented; vocabulary intentionally disjoint from
    # every evaluation log so the fine-tuned models cannot memorise eval entity names).
    # Attribute naming conventions are varied on purpose (camelCase, snake_case, is-prefix)
    # so the model learns to copy the attribute name from the schema rather than guess.
    "library_circulation": {
        "events": ["Reserve Book", "Borrow Book", "Renew Loan", "Return Book",
                   "Pay Late Fee", "Report Book Lost"],
        "objects": ["Member", "Book", "Loan", "Branch"]
    },
    "restaurant_service": {
        "events": ["Place Reservation", "Seat Guests", "Take Order", "Serve Dish",
                   "Split Bill", "Process Payment"],
        "objects": ["Guest", "Table", "Order", "Dish", "Bill"]
    },
    "insurance_claims": {
        "events": ["File Claim", "Assign Adjuster", "Inspect Damage", "Approve Claim",
                   "Deny Claim", "Issue Settlement"],
        "objects": ["Policyholder", "Claim", "Policy", "Adjuster"]
    },

    # String-typed domains. These model a log exported by a tool that erases attribute
    # types -- PM4PY writes every attribute as a string -- so EVERY attribute below is
    # declared "String", including ones that are semantically boolean ("0"/"1") or numeric
    # ("120"/"240"). Real evaluation logs look like this (renting's `isCitizen` is a String
    # holding "0"/"1"), while every other synthetic domain declares clean Boolean/Integer
    # types. Without these the model only ever sees the idealised typing and has no reason
    # to emit {"type":"String","is_in":["0"]} for something that reads like a boolean.
    #
    # Typing is uniform within a domain on purpose: an export tool erases all types or
    # none, so mixing would teach that typing is arbitrary rather than a property of the
    # source. Values stay categorical and are only ever compared for equality -- a
    # threshold on a string-typed number would require a CEL cast, which is deliberately
    # out of scope (see Thesis/notes/11-evaluation-readiness.md).
    # Vocabulary verified disjoint from all 392 type/attribute names across the 11
    # evaluation logs, same de-leaking rule as the invented domains above.
    "gym": {
        "events": ["Book Class", "Check In Athlete", "Start Session", "Cancel Booking",
                   "Renew Membership", "Report Injury"],
        "objects": ["Athlete", "ClassSession", "Membership", "Trainer", "Locker"]
    },
    "waste_collection": {
        "events": ["Register Bin", "Schedule Round", "Collect Bin", "Weigh Load",
                   "Report Contamination", "Close Round"],
        "objects": ["Household", "Bin", "CollectionRound", "Depot", "Crew"]
    },

    # Domains from actual log files (verified NOT to overlap any evaluation log's vocabulary)
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
    # `Time` and `Boolean` are the two starved kinds: before the additions below the whole
    # generator declared ONE Time attribute (`cartDate`) and three Booleans, which put F14 and
    # F12 more than 10x rarer in training than in the testset. That ceiling is structural --
    # the sampler cannot draw a kind that no attribute declares -- so it is raised here rather
    # than by regenerating until the distribution looks better, which would be selection on the
    # outcome. Added only to the invented domains: `distribution`/`procurement`/`02_p2p` have
    # tables derived from real logs and must keep describing them faithfully.
    "e-commerce":          [("carts", "price", "Float"), ("articles", "weight", "Float"),
                            ("carts", "cartDate", "Time"),
                            ("carts", "isGift", "Boolean")],
    "distribution":        [("Order", "totalAmount", "Float"), ("Product", "weight", "Float")],
    "procurement":         [("Purchase Order", "amount", "Float")],
    "02_p2p":              [("Purchase Order", "amount", "Float")],
    "recruitment":         [("candidates", "yearsExperience", "Integer")],
    "software_dev":        [("pull requests", "linesChanged", "Integer")],
    # De-leaked replacement domains. Naming conventions are intentionally mixed
    # (camelCase / snake_case / is-prefix) to teach schema-copying over guessing.
    "library_circulation": [("Member", "membershipYears", "Integer"),
                            ("Member", "is_student", "Boolean"),
                            ("Book", "page_count", "Integer"),
                            ("Loan", "fineAmount", "Float"),
                            ("Loan", "due_date", "Time"),
                            ("Book", "is_reference", "Boolean")],
    "restaurant_service":  [("Guest", "partySize", "Integer"),
                            ("Guest", "isVip", "Boolean"),
                            ("Bill", "total_amount", "Float"),
                            ("Order", "prepMinutes", "Integer"),
                            ("Order", "placedAt", "Time"),
                            ("Table", "isOutdoor", "Boolean")],
    "insurance_claims":    [("Policyholder", "age", "Integer"),
                            ("Policyholder", "has_prior_claims", "Boolean"),
                            ("Claim", "claimAmount", "Float"),
                            ("Policy", "coverage_limit", "Integer"),
                            ("Claim", "filedOn", "Time")],

    # String-typed domains: every attribute is declared "String" with an explicit value
    # list, mirroring a PM4PY-style export. The 4th tuple element is read by
    # `_domain_attributes` and rendered into the prompt as `name (String: v1, v2)`, so the
    # model can see that `isStudent` is encoded "0"/"1" rather than true/false. Three
    # shapes are covered because the testset needs all three: boolean-as-string,
    # numeric-as-string, and word categoricals. Values are compared for equality only.
    "gym":      [("Athlete", "isStudent", "String", ["0", "1"]),
                            ("Athlete", "membership_tier", "String", ["1", "2", "3"]),
                            ("ClassSession", "sessionStatus", "String", ["Held", "Cancelled"]),
                            ("Membership", "is_frozen", "String", ["0", "1"]),
                            ("Trainer", "trainerGrade", "String", ["1", "2"])],
    "waste_collection":    [("Household", "isCommercial", "String", ["0", "1"]),
                            ("Bin", "bin_size", "String", ["120", "240", "660"]),
                            ("CollectionRound", "roundOutcome", "String", ["Completed", "Aborted"]),
                            ("Crew", "shift_code", "String", ["A", "B"]),
                            ("Depot", "depotZone", "String", ["1", "2", "3"])],
}

# Authoritative per-domain relation graphs. Every builder that needs an O2O pair or an
# E2O (event, object) pair draws from these instead of sampling any two names at random,
# so two different generated examples for the same domain can never assert contradictory
# relations (e.g. one claiming Member-Book are directly linked while another claims they
# aren't). For `distribution`/`procurement`/`02_p2p` these were extracted/derived from the
# real logs at Logs/01_o2c.xml and Logs/02_p2p.xml (E2O read directly via
# LogEvents.extract_ocel_schema; O2O has no real data in those logs, so it's derived from
# object types co-occurring under the same event type). The other domains have no backing
# log, so both tables are hand-authored, independently of each other.

# domain -> [(object_a, object_b), ...] valid O2O pairs.
OBJECT_RELATIONS = {
    "e-commerce": [("carts", "shoppers"), ("carts", "line items"), ("line items", "articles"),
                   ("workers", "line items"), ("carts", "parcels"), ("workers", "parcels")],
    "recruitment": [("applications", "candidates")],
    "software_dev": [("pull requests", "commits"), ("issues", "pull requests")],
    "library_circulation": [("Member", "Loan"), ("Loan", "Book"), ("Loan", "Branch")],
    "restaurant_service": [("Guest", "Table"), ("Guest", "Order"), ("Order", "Dish"),
                            ("Order", "Bill"), ("Guest", "Bill")],
    "insurance_claims": [("Policyholder", "Claim"), ("Claim", "Policy"), ("Claim", "Adjuster")],
    "gym": [("Athlete", "Membership"), ("Athlete", "ClassSession"),
                       ("ClassSession", "Trainer"), ("Athlete", "Locker")],
    "waste_collection": [("Household", "Bin"), ("Bin", "CollectionRound"),
                         ("CollectionRound", "Crew"), ("CollectionRound", "Depot")],
    # Derived from Logs/01_o2c.xml E2O co-occurrence.
    "distribution": [("Customer", "Order"), ("Customer", "Invoice"), ("Invoice", "Payment"),
                      ("Order", "Shipment"), ("Invoice", "Order"), ("Order", "Product"),
                      ("Customer", "Dispute"), ("Dispute", "Invoice"), ("CreditNote", "Return"),
                      ("Customer", "Return"), ("Order", "Return")],
    # Derived from Logs/02_p2p.xml E2O co-occurrence.
    "procurement": [("Goods Receipt", "Purchase Order"), ("Invoice", "Purchase Order"),
                     ("Purchase Order", "Purchase Requisition"), ("Goods Receipt", "Invoice"),
                     ("Invoice", "Payment"), ("Invoice", "Supplier"), ("Payment", "Supplier"),
                     ("Purchase Order", "Supplier")],
}
OBJECT_RELATIONS["02_p2p"] = OBJECT_RELATIONS["procurement"]  # same underlying log

# domain -> [(event, object), ...] valid E2O pairs.
EVENT_OBJECT_RELATIONS = {
    "e-commerce": [("open cart", "carts"), ("open cart", "shoppers"),
                   ("add line item", "carts"), ("add line item", "line items"),
                   ("add line item", "articles"), ("add line item", "workers"),
                   ("settle cart", "carts"), ("settle cart", "shoppers"),
                   ("dispatch parcel", "carts"), ("dispatch parcel", "parcels"),
                   ("dispatch parcel", "workers"), ("void cart", "carts"),
                   ("void cart", "shoppers")],
    "recruitment": [("submit application", "applications"), ("submit application", "candidates"),
                     ("review application", "applications"), ("review application", "candidates"),
                     ("approve application", "applications"), ("approve application", "candidates"),
                     ("reject application", "applications"), ("reject application", "candidates")],
    "software_dev": [("open issue", "issues"), ("submit pull request", "pull requests"),
                      ("submit pull request", "commits"), ("review pull request", "pull requests"),
                      ("merge pull request", "pull requests"), ("merge pull request", "commits"),
                      ("close issue", "issues"), ("close issue", "pull requests")],
    "library_circulation": [("Reserve Book", "Member"), ("Reserve Book", "Book"),
                             ("Reserve Book", "Loan"), ("Borrow Book", "Member"),
                             ("Borrow Book", "Book"), ("Borrow Book", "Loan"),
                             ("Borrow Book", "Branch"), ("Renew Loan", "Member"),
                             ("Renew Loan", "Loan"), ("Return Book", "Book"),
                             ("Return Book", "Loan"), ("Return Book", "Branch"),
                             ("Pay Late Fee", "Member"), ("Pay Late Fee", "Loan"),
                             ("Report Book Lost", "Book"), ("Report Book Lost", "Loan")],
    "restaurant_service": [("Place Reservation", "Guest"), ("Place Reservation", "Table"),
                            ("Seat Guests", "Guest"), ("Seat Guests", "Table"),
                            ("Take Order", "Guest"), ("Take Order", "Order"),
                            ("Take Order", "Table"), ("Serve Dish", "Order"),
                            ("Serve Dish", "Dish"), ("Split Bill", "Guest"),
                            ("Split Bill", "Bill"), ("Split Bill", "Order"),
                            ("Process Payment", "Guest"), ("Process Payment", "Bill")],
    "insurance_claims": [("File Claim", "Policyholder"), ("File Claim", "Claim"),
                          ("File Claim", "Policy"), ("Assign Adjuster", "Claim"),
                          ("Assign Adjuster", "Adjuster"), ("Inspect Damage", "Claim"),
                          ("Inspect Damage", "Adjuster"), ("Approve Claim", "Claim"),
                          ("Approve Claim", "Policyholder"), ("Deny Claim", "Claim"),
                          ("Deny Claim", "Policyholder"), ("Issue Settlement", "Claim"),
                          ("Issue Settlement", "Policyholder")],
    "gym": [("Book Class", "Athlete"), ("Book Class", "ClassSession"),
                        ("Check In Athlete", "Athlete"), ("Check In Athlete", "Locker"),
                        ("Check In Athlete", "Membership"), ("Start Session", "ClassSession"),
                        ("Start Session", "Trainer"), ("Cancel Booking", "Athlete"),
                        ("Cancel Booking", "ClassSession"), ("Renew Membership", "Athlete"),
                        ("Renew Membership", "Membership"), ("Report Injury", "Athlete"),
                        ("Report Injury", "ClassSession"), ("Report Injury", "Trainer")],
    "waste_collection": [("Register Bin", "Household"), ("Register Bin", "Bin"),
                          ("Schedule Round", "CollectionRound"), ("Schedule Round", "Crew"),
                          ("Schedule Round", "Depot"), ("Collect Bin", "Bin"),
                          ("Collect Bin", "CollectionRound"), ("Collect Bin", "Crew"),
                          ("Weigh Load", "CollectionRound"), ("Weigh Load", "Depot"),
                          ("Report Contamination", "Bin"), ("Report Contamination", "Household"),
                          ("Close Round", "CollectionRound"), ("Close Round", "Crew")],
    # Real E2O extracted from Logs/01_o2c.xml (qualifier dropped, distinct (event,object) pairs).
    "distribution": [("Apply Discount", "Order"), ("Apply Payment", "Invoice"),
                      ("Apply Payment", "Payment"), ("Approve Order", "Customer"),
                      ("Approve Order", "Order"), ("Approve Return", "Return"),
                      ("Generate Invoice", "Invoice"), ("Generate Invoice", "Order"),
                      ("Generate Pick List", "Order"), ("Generate Pick List", "Product"),
                      ("Generate Shipping Label", "Order"), ("Generate Shipping Label", "Shipment"),
                      ("Handle Payment Discrepancy", "Customer"),
                      ("Handle Payment Discrepancy", "Dispute"),
                      ("Handle Payment Discrepancy", "Invoice"),
                      ("Issue Credit Note", "CreditNote"), ("Issue Credit Note", "Return"),
                      ("Issue Payment Reminder", "Customer"), ("Issue Payment Reminder", "Invoice"),
                      ("Pack Items", "Order"), ("Perform Credit Check", "Customer"),
                      ("Perform Credit Check", "Order"), ("Pick Items", "Product"),
                      ("Process Return Request", "Customer"), ("Process Return Request", "Order"),
                      ("Process Return Request", "Return"), ("Receive Customer Order", "Customer"),
                      ("Receive Customer Order", "Order"), ("Receive Payment", "Invoice"),
                      ("Receive Payment", "Payment"), ("Reject Order", "Customer"),
                      ("Reject Order", "Order"), ("Schedule Order Fulfillment", "Order"),
                      ("Send Invoice to Customer", "Customer"),
                      ("Send Invoice to Customer", "Invoice"), ("Ship Order", "Order"),
                      ("Ship Order", "Shipment"), ("Update Inventory", "Product"),
                      ("Validate Customer Order", "Order")],
    # Real E2O extracted from Logs/02_p2p.xml.
    "procurement": [("Approve PO", "Purchase Order"), ("Approve PR", "Purchase Requisition"),
                     ("Create PO", "Purchase Order"), ("Create PO", "Purchase Requisition"),
                     ("Create PO (Batch)", "Purchase Order"),
                     ("Create PO (Batch)", "Purchase Requisition"),
                     ("Create PR", "Purchase Requisition"), ("Make Payment", "Invoice"),
                     ("Make Payment", "Payment"), ("Make Payment", "Supplier"),
                     ("Receive Goods", "Goods Receipt"), ("Receive Goods", "Purchase Order"),
                     ("Receive Invoice", "Invoice"), ("Receive Invoice", "Purchase Order"),
                     ("Reject PR", "Purchase Requisition"), ("Send PO to Supplier", "Purchase Order"),
                     ("Send PO to Supplier", "Supplier"),
                     ("Three-way Match - Mismatch Found", "Goods Receipt"),
                     ("Three-way Match - Mismatch Found", "Invoice"),
                     ("Three-way Match - Mismatch Found", "Purchase Order"),
                     ("Three-way Match - Successful", "Goods Receipt"),
                     ("Three-way Match - Successful", "Invoice"),
                     ("Three-way Match - Successful", "Purchase Order")],
}
EVENT_OBJECT_RELATIONS["02_p2p"] = EVENT_OBJECT_RELATIONS["procurement"]

# Singular, natural-language surface form for each object type. Several domains use plural
# (e-commerce, recruitment, software_dev) or PascalCase/compound (distribution, procurement)
# type names that would read ungrammatically substituted directly into a sentence — used by
# the randomized no-direct-path generator, which has to phrase an NL sentence for any object
# type the relation graph hands it, not just a hand-picked few.
OBJECT_SINGULAR = {
    "e-commerce": {"shoppers": "shopper", "workers": "worker", "line items": "line item",
                   "carts": "cart", "parcels": "parcel", "articles": "article"},
    "recruitment": {"applications": "application", "candidates": "candidate"},
    "software_dev": {"issues": "issue", "pull requests": "pull request", "commits": "commit"},
    "library_circulation": {"Member": "member", "Book": "book", "Loan": "loan", "Branch": "branch"},
    "restaurant_service": {"Guest": "guest", "Table": "table", "Order": "order", "Dish": "dish",
                            "Bill": "bill"},
    "insurance_claims": {"Policyholder": "policyholder", "Claim": "claim", "Policy": "policy",
                          "Adjuster": "adjuster"},
    "distribution": {"CreditNote": "credit note", "Customer": "customer", "Dispute": "dispute",
                      "Invoice": "invoice", "Order": "order", "Payment": "payment",
                      "Product": "product", "Return": "return", "Shipment": "shipment"},
    "procurement": {"Goods Receipt": "goods receipt", "Purchase Requisition": "purchase requisition",
                     "Payment": "payment", "Invoice": "invoice", "Purchase Order": "purchase order",
                     "Supplier": "supplier"},
    # The CamelCase types need an explicit split, exactly as `CreditNote` does above: the
    # fallback in _build_no_direct_path is `name.lower()`, which turns `ClassSession` into
    # "classsession". The *type names* stay CamelCase on purpose -- real evaluation logs use
    # every convention (`HiringManager` and `JobRequisition` in 03_hiring, `DiagnosticEquipment`
    # in 04_hospital, `Customer Order` in container_logistics, `purchase_order` in ocel2-p2p),
    # so the model has to copy whatever the schema declares rather than assume one style.
    "gym": {"Athlete": "athlete", "ClassSession": "class session", "Membership": "membership",
             "Trainer": "trainer", "Locker": "locker"},
    "waste_collection": {"Household": "household", "Bin": "bin",
                          "CollectionRound": "collection round", "Depot": "depot", "Crew": "crew"},
}
OBJECT_SINGULAR["02_p2p"] = OBJECT_SINGULAR["procurement"]


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

# Pre-computed event -> {valid object types} map per domain, built once from
# EVENT_OBJECT_RELATIONS since _e2o_object() is called thousands of times per run.
_EVENT_TO_OBJECTS: dict = {}
for _dom, _pairs in EVENT_OBJECT_RELATIONS.items():
    _by_event: dict = {}
    for _ev, _obj in _pairs:
        _by_event.setdefault(_ev, set()).add(_obj)
    _EVENT_TO_OBJECTS[_dom] = _by_event
del _dom, _pairs, _by_event, _ev, _obj

# Pre-computed object -> {events valid for it} map per domain (reverse of _EVENT_TO_OBJECTS),
# needed when the object is picked first (e.g. it's constrained by OBJECT_ATTRIBUTES) and a
# compatible event must be found afterwards.
_OBJECT_TO_EVENTS: dict = {}
for _dom, _pairs in EVENT_OBJECT_RELATIONS.items():
    _by_obj: dict = {}
    for _ev, _obj in _pairs:
        _by_obj.setdefault(_obj, set()).add(_ev)
    _OBJECT_TO_EVENTS[_dom] = _by_obj
del _dom, _pairs, _by_obj, _ev, _obj


def _o2o_pair(domain: str, fixed: str = None):
    """Return a random valid (obj_a, obj_b) O2O pair for domain from OBJECT_RELATIONS.

    If `fixed` is given, returns (fixed, other) where `other` is drawn from pairs that
    actually contain `fixed`, so an object chosen elsewhere (e.g. by OBJECT_ATTRIBUTES) is
    kept as-is instead of being overridden. Falls back to an unconstrained random pair (or
    partner) if the domain has no table entry, or no table pair contains `fixed`.
    """
    pairs = OBJECT_RELATIONS.get(domain)
    dd = domains[domain]
    if fixed is not None:
        partners = [b if a == fixed else a for a, b in (pairs or []) if fixed in (a, b)]
        if partners:
            return fixed, random.choice(partners)
        others = [o for o in dd["objects"] if o != fixed] or [fixed]
        return fixed, random.choice(others)
    if pairs:
        return random.choice(pairs)
    if len(dd["objects"]) >= 2:
        a, b = random.sample(dd["objects"], 2)
        return a, b
    return dd["objects"][0], dd["objects"][0]


def _e2o_object(domain: str, events):
    """Return a random object type E2O-valid for every event in `events`.

    Uses the intersection of each event's valid-object set (so a single object variable
    bound to multiple events picks a type that plausibly relates to all of them). Falls
    back to the union if no object is valid for every event, then to any object in the
    domain if the domain has no EVENT_OBJECT_RELATIONS entry at all.
    """
    if isinstance(events, str):
        events = [events]
    dd = domains[domain]
    ev_to_objs = _EVENT_TO_OBJECTS.get(domain)
    if not ev_to_objs:
        return random.choice(dd["objects"])
    sets = [ev_to_objs.get(ev, set()) for ev in events]
    candidates = set.intersection(*sets) if sets else set()
    if not candidates:
        candidates = set.union(*sets) if sets else set()
    if not candidates:
        return random.choice(dd["objects"])
    return random.choice(sorted(candidates))


def _e2o_event(domain: str, objects):
    """Return a random event E2O-valid for at least one object in `objects`.

    Reverse of `_e2o_object`, used when the object(s) are already fixed (e.g. picked via
    OBJECT_ATTRIBUTES, or a multi-type object variable) and a compatible event is needed.
    Uses the union (any given object being touched by the event is sufficient) since the
    call sites either have a single fixed object or an either/or multi-type object.
    """
    if isinstance(objects, str):
        objects = [objects]
    dd = domains[domain]
    ob_to_evs = _OBJECT_TO_EVENTS.get(domain)
    if not ob_to_evs:
        return random.choice(dd["events"])
    candidates = set.union(*(ob_to_evs.get(o, set()) for o in objects)) if objects else set()
    if not candidates:
        return random.choice(dd["events"])
    return random.choice(sorted(candidates))


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


def _query_relations(query: dict) -> dict:
    """Derive the E2O and O2O relations a query actually uses, as name-level triples.

    O2E/O2O filters reference variables by index into a node's newEventVars /
    newObjectVars; this resolves those indices back to the event/object type names.
    The result is used to show the model exactly the relations (and their qualifiers)
    its target query relies on, so it learns to read relation names from the schema
    instead of inventing them. Because the graph is derived from the gold query, the
    gold can never reference a relation absent from the shown schema.

    Returns:
        {"e2o": [[event, object, qualifier], ...], "o2o": [[object, object, qualifier], ...]}
        with qualifiers as strings or null, sorted and de-duplicated.
    """
    e2o: set = set()
    o2o: set = set()
    for node in (query or {}).get("nodes", []):
        if not isinstance(node, dict):
            continue
        box = node.get("Box")
        if not isinstance(box, list) or not box or not isinstance(box[0], dict):
            continue
        data = box[0]
        ev_vars = data.get("newEventVars", {}) or {}
        ob_vars = data.get("newObjectVars", {}) or {}

        def _names(vars_map, idx):
            val = vars_map.get(str(idx), vars_map.get(idx, []))
            return [v for v in val if isinstance(v, str)] if isinstance(val, list) else []

        for f in data.get("filters", []) or []:
            if not isinstance(f, dict):
                continue
            qualifier = f.get("qualifier")
            if f.get("type") == "O2E":
                for obj_name in _names(ob_vars, f.get("object")):
                    for ev_name in _names(ev_vars, f.get("event")):
                        e2o.add((ev_name, obj_name, qualifier))
            elif f.get("type") == "O2O":
                for a_name in _names(ob_vars, f.get("object")):
                    for b_name in _names(ob_vars, f.get("other_object")):
                        o2o.add((a_name, b_name, qualifier))
    return {
        "e2o": sorted([list(t) for t in e2o]),
        "o2o": sorted([list(t) for t in o2o]),
    }


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
        obj_a, obj_b = _o2o_pair(domain)
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
        obj_a = _e2o_object(domain, [event])
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
    obj = _e2o_object(domain, [a, b])
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
    event = _e2o_event(domain, [obj])

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
    obj = _e2o_object(domain, [a, b])
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
    _, obj_a = _o2o_pair(domain, fixed=obj_b)
    event = _e2o_event(domain, [obj_b])
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
    obj = _e2o_object(domain, [a, b])
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
    obj = _e2o_object(domain, [a, b, c])
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
    obj = _e2o_object(domain, [a, b, c])
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
    obj = _e2o_object(domain, [a, b])
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
    obj = _e2o_object(domain, [a, b])
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
    obj = _e2o_object(domain, [a, b, c, d])
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
    obj_a, obj_b = _o2o_pair(domain)
    event = _e2o_event(domain, [obj_a, obj_b])
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


def _no_direct_path_triples(domain):
    """Return every (A, B, C) triple for domain where A-B and B-C are valid O2O pairs in
    OBJECT_RELATIONS but A-C is not — the shape the no-direct-path skill needs (A and C get
    named in the NL, B is the required unmentioned intermediary)."""
    adjacency: dict = {}
    edge_set = set()
    for a, b in OBJECT_RELATIONS.get(domain, []):
        edge_set.add((a, b))
        edge_set.add((b, a))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    triples = []
    for b, neighbors in adjacency.items():
        neighbors = sorted(neighbors)
        for i, a in enumerate(neighbors):
            for c in neighbors[i + 1:]:
                if (a, c) not in edge_set:
                    triples.append((a, b, c))
    return triples


# Pre-computed once: {domain: [(A, B, C), ...]} of valid no-direct-path triples.
_NO_PATH_TRIPLES: dict = {dom: _no_direct_path_triples(dom) for dom in OBJECT_RELATIONS}

# Domain-agnostic phrasing templates for the randomized no-direct-path generator, since it
# has to phrase a sentence for arbitrary triples it wasn't hand-authored for.
_NO_PATH_TEMPLATES = [
    ("who", "is connected to"),
    ("that", "is linked to"),
    ("who", "is associated with"),
    ("that", "relates to"),
]


def _build_no_direct_path():
    """Randomized counterpart to the hand-authored examples in
    _build_targeted_curriculum_examples: draws a random (A, B, C) no-direct-path triple from
    _NO_PATH_TRIPLES (any domain, not just the 8 hand-picked ones) so this skill is drawn from
    EXT_BUILDERS at the same rate as every other building block instead of being capped at a
    fixed handful. Reads the same OBJECT_RELATIONS table every other O2O site now draws from,
    so it can never contradict a 'these two ARE related' example generated elsewhere.
    """
    candidates = [dom for dom, triples in _NO_PATH_TRIPLES.items() if triples]
    domain = random.choice(candidates)
    a, b, c = random.choice(_NO_PATH_TRIPLES[domain])
    dd = domains[domain]
    singular = OBJECT_SINGULAR.get(domain, {})
    a_nl = singular.get(a, a.lower())
    c_nl = singular.get(c, c.lower())
    subj, verb = random.choice(_NO_PATH_TEMPLATES)
    article = "an" if c_nl[:1].lower() in "aeiou" else "a"

    node = _box(children=[], ob={"0": [a], "1": [b], "2": [c]}, filters=[_o2o(0, 1), _o2o(1, 2)])
    query = {"nodes": [node], "edgeNames": []}
    nl = f"Find each {a_nl} {subj} {verb} {article} {c_nl}."
    reasoning = (
        f"Step 1: The query names '{a}' and '{c}', but the relation graph has no direct "
        f"'{a}' -> '{c}' link. Step 2: The graph does link '{a}' -> '{b}' and '{b}' -> '{c}', "
        f"so '{b}' is the required (unmentioned) intermediary. Step 3: Bind all three object "
        f"variables in one node and chain them with two O2O filters: ({a},{b}) and ({b},{c})."
    )
    return {"nl": nl, "events": dd["events"], "objects": dd["objects"],
            "query": query, "reasoning": reasoning}


EXT_BUILDERS = [
    _build_qualifier,
    _build_no_direct_path,
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

    # Pick obj_a/obj_b consistent with OBJECT_RELATIONS/EVENT_OBJECT_RELATIONS, matching
    # the exact E2O/O2O shape the tree below actually builds for this pattern.
    if "total is calculated" in pattern or "for which the EVENT_A" in pattern:
        # obj_b-A O2E, obj_a-obj_b O2O
        obj_b = _e2o_object(domain, [A])
        _, obj_a = _o2o_pair(domain, fixed=obj_b)
    elif "at most NUM" in pattern:
        # obj_a-A O2E, obj_a-B O2E, obj_a-obj_b O2O
        obj_a = _e2o_object(domain, [A, B])
        _, obj_b = _o2o_pair(domain, fixed=obj_a)
    elif "at least NUM2" in pattern:
        # obj_a-A O2E; obj_b-A O2E and obj_b-B O2E (no O2O between them)
        obj_b = _e2o_object(domain, [A, B])
        obj_a = _e2o_object(domain, [A])
    elif "exactly once" in pattern:
        obj_a = _e2o_object(domain, [A])
        obj_b = obj_a
    elif ("at least once" in pattern or "completed within" in pattern
          or "refer to the same" in pattern):
        obj_a = _e2o_object(domain, [A, B])
        obj_b = obj_a
    else:
        # Patterns with no object at all in the tree ("before"/"eventually"/"within TIME"/
        # "must not occur") — obj_a/obj_b are unused, so an unconstrained pick is fine.
        obj_a = random.choice(domain_data["objects"])
        obj_b = (random.choice([o for o in domain_data["objects"] if o != obj_a])
                 if len(domain_data["objects"]) > 1 else obj_a)

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


def _plural_verb(verb):
    """Agree a singular-subject verb phrase with a plural subject (only be-verbs change)."""
    if verb.startswith("is "):
        return "are " + verb[3:]
    if verb.startswith("was "):
        return "were " + verb[4:]
    return verb


def _build_no_direct_path_examples():
    """Examples where two named entities have NO direct relation and the gold query must
    route through an unmentioned intermediary object.

    This is the exact skill the pipeline was missing: the NL names A and C, the relation
    graph links A->B and B->C but never A->C, so the model must introduce B. Because the
    shown schema is derived from the gold (see _query_relations), the A->C edge is never
    present, making the no-path premise explicit to the model.
    """
    # (domain, A, B_intermediary, C, subject word "who"/"that", verb phrase,
    #  NL-singular form of A, NL-singular form of C). The NL-singular forms are decoupled
    # from the schema type strings (A/C) because some domains use plural type names
    # (e.g. "customers", "issues") that would otherwise read ungrammatically in the sentence.
    specs = [
        ("library_circulation", "Member", "Loan", "Book", "who", "borrowed", "member", "book"),
        ("restaurant_service", "Guest", "Order", "Dish", "who", "was served", "guest", "dish"),
        ("insurance_claims", "Policyholder", "Claim", "Policy", "who", "filed against",
         "policyholder", "policy"),
        ("e-commerce", "carts", "line items", "articles", "that", "included",
         "cart", "article"),
        ("software_dev", "commits", "pull requests", "issues", "that", "closed",
         "commit", "issue"),
        ("distribution", "Customer", "Order", "Shipment", "who", "received",
         "customer", "shipment"),
        # Real Logs/02_p2p.xml co-occurrence shows Invoice-Supplier DO co-occur (via "Make
        # Payment"), so Supplier->PO->Invoice would contradict OBJECT_RELATIONS. Goods
        # Receipt and Purchase Requisition never co-occur there, so that pair is used instead.
        ("procurement", "Goods Receipt", "Purchase Order", "Purchase Requisition", "that",
         "is associated with", "goods receipt", "purchase requisition"),
        ("02_p2p", "Goods Receipt", "Purchase Order", "Purchase Requisition", "that",
         "is associated with", "goods receipt", "purchase requisition"),
    ]
    for domain, a, b, c, _subj, _verb, _a_nl, _c_nl in specs:
        assert (a, b) in OBJECT_RELATIONS[domain] or (b, a) in OBJECT_RELATIONS[domain], \
            f"{domain}: {a}-{b} must be a valid O2O pair"
        assert (b, c) in OBJECT_RELATIONS[domain] or (c, b) in OBJECT_RELATIONS[domain], \
            f"{domain}: {b}-{c} must be a valid O2O pair"
        assert (a, c) not in OBJECT_RELATIONS[domain] and (c, a) not in OBJECT_RELATIONS[domain], \
            f"{domain}: {a}-{c} must NOT be a direct O2O pair (contradicts the no-path claim)"
    examples = []
    for domain, a, b, c, subj, verb, a_nl, c_nl in specs:
        dd = domains[domain]
        node = _box(
            children=[],
            ob={"0": [a], "1": [b], "2": [c]},
            filters=[_o2o(0, 1), _o2o(1, 2)],
        )
        query = {"nodes": [node], "edgeNames": []}
        article = "an" if c_nl[:1].lower() in "aeiou" else "a"
        examples.append({
            "nl": f"Find each {a_nl} {subj} {verb} {article} {c_nl}.",
            "nl_clean_1": f"List every {a_nl} that {verb} {article} {c_nl}.",
            "nl_clean_2": f"Which {a_nl}s {_plural_verb(verb)} {article} {c_nl}?",
            "events": dd["events"],
            "objects": dd["objects"],
            "query": query,
            "reasoning": (
                f"Step 1: The query names '{a}' and '{c}', but the relation graph has no direct "
                f"'{a}' -> '{c}' link. Step 2: The graph does link '{a}' -> '{b}' and '{b}' -> '{c}', "
                f"so '{b}' is the required (unmentioned) intermediary. Step 3: Bind all three object "
                f"variables in one node and chain them with two O2O filters: ({a},{b}) and ({b},{c})."
            ),
        })
    return examples


def _build_targeted_curriculum_examples():
    # Uses the de-leaked `insurance_claims` domain (vocabulary disjoint from every eval log).
    claims_events = domains["insurance_claims"]["events"]
    claims_objects = domains["insurance_claims"]["objects"]

    targeted_examples = [
        {
            "nl": "For any policyholder, the claim is approved exactly once.",
            "nl_clean_1": "For any policyholder, the claim must be approved exactly once.",
            "nl_clean_2": "Each policyholder should have their claim approved exactly one time.",
            "events": claims_events,
            "objects": claims_objects,
            "query": {
                "nodes": [
                    {
                        "Box": [
                            {
                                "newEventVars": {},
                                "newObjectVars": {"0": ["Policyholder"]},
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
                                "newEventVars": {"0": ["Approve Claim"]},
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
                "Step 1: Interpret the colloquial term 'policyholder' as the allowed object 'Policyholder' "
                "and 'approved' as the allowed event 'Approve Claim'. Step 2: The phrase 'exactly once' "
                "must not introduce a custom filter type. Step 3: Represent it with a parent object node "
                "and a SizeFilter min=1,max=1 on a child event node that is linked back with O2E."
            ),
        },
        {
            "nl": "For any policyholder, the claim must be approved within 3 days after it is filed.",
            "nl_clean_1": "For any policyholder, the claim must be approved within 3 days after filing.",
            "nl_clean_2": "Each policyholder should have their claim approved within 3 days after filing it.",
            "events": claims_events,
            "objects": claims_objects,
            "query": {
                "nodes": [
                    {
                        "Box": [
                            {
                                "newEventVars": {},
                                "newObjectVars": {"0": ["Policyholder"]},
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
                                "newEventVars": {"0": ["File Claim"], "1": ["Approve Claim"]},
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
                "Step 1: Interpret 'policyholder' as 'Policyholder', 'filed' as 'File Claim', and "
                "'approved' as 'Approve Claim'. Step 2: The temporal constraint is 'within 3 days after', "
                "so it must become a TimeBetweenEvents max_seconds of 259200. Step 3: Keep the tree connected "
                "with a parent SAT node and child event node using O2E bindings for both events."
            ),
        },
    ]

    targeted_examples.extend(_build_no_direct_path_examples())
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


def generate_dataset(num_samples=500, batch_size=4, original_sentence=True, first_rewrite=True, second_rewrite=True, output_file="ocpq_dataset.json", seed=None):
    # Re-seed per call, not only at import: that makes one invocation reproducible no matter
    # what else drew from the shared `random` module first (an interactive session, another
    # import). Without it, `import DatasetGenerator; generate_dataset(...)` and
    # `python DatasetGenerator.py` could diverge.
    effective_seed = GLOBAL_SEED if seed is None else int(seed)
    random.seed(effective_seed)

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
            example.setdefault("relations", _query_relations(example["query"]))
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
        ex.setdefault("relations", _query_relations(ex["query"]))
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
            if "object_attributes" in ex:
                updated_example["object_attributes"] = ex["object_attributes"]
            if "relations" in ex:
                updated_example["relations"] = ex["relations"]
            updated_example["query"] = ex["query"]
            updated_example["reasoning"] = ex["reasoning"]
            dataset.append(updated_example)
            
        print(f"Processed {len(dataset)} / {num_samples} examples", end="\r")

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    # Provenance sidecar. The seed alone does not reproduce a corpus -- the code consuming the
    # random stream has to match too -- so record a hash of this generator alongside it.
    # Mirrors the embeddings.meta.json guard that catches embeddings built from another corpus.
    generator_digest = hashlib.sha256()
    with open(__file__, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            generator_digest.update(chunk)
    meta_path = os.path.splitext(output_file)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "seed": effective_seed,
            "rows": len(dataset),
            "num_samples_requested": num_samples,
            "generator_sha256": generator_digest.hexdigest(),
            "generator": os.path.basename(__file__),
        }, f, indent=2)
    print(f"\nWrote {meta_path} (seed {effective_seed}, "
          f"generator sha256 {generator_digest.hexdigest()[:12]}...)")

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
