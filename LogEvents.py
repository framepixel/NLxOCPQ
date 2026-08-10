#!/usr/bin/env python
# coding: utf-8
import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Optional


# Schemas are re-extracted for every test case in the evaluation loop, and extraction
# walks every event and object in the log. Cache by (path, mtime, size) so each log is
# parsed once per process — wall-time is the binding constraint on the cluster.
_SCHEMA_CACHE: dict[tuple, dict] = {}

# Bump whenever the *shape* of an extracted schema changes. The cache key is otherwise
# derived only from the log file, so a code change that adds a field would keep serving
# the old shape from `.schema_cache/` forever and the change would silently do nothing.
#   v2: string-typed attributes carry a bounded `values` list.
_SCHEMA_FORMAT_VERSION = 2

# Distinct values kept per string attribute. The prompt enumerates them so the model can
# see that `isCitizen` is encoded "0"/"1" rather than "true"/"false" — training did exactly
# this (`Status (String: full, empty)`) and evaluation did not, which is the divergence this
# closes. Attributes above the cap are identifiers (`Case` has 10 000 distinct values), and
# carry no `values`, so they render as a bare `(String)` exactly as before.
_MAX_ATTRIBUTE_VALUES = 10


class _BoundedValues:
    """Collects distinct attribute values, refusing to grow far past the cap.

    Bounding matters: `bpic2017-ocel2.json` has 1.2 M events, and an unbounded set per
    attribute would hold a large share of the log in memory. Growth stops one past the cap,
    which is still enough to tell "fits" from "too many".
    """

    def __init__(self, cap: int) -> None:
        self._cap = cap
        self._seen: dict[tuple, set] = {}

    def add(self, type_name: Optional[str], attr_name: Optional[str], value: Any) -> None:
        if not type_name or not attr_name or value is None:
            return
        text = str(value).strip()
        if not text:
            return  # empty strings are real but unguessable, and render as noise
        bucket = self._seen.setdefault((type_name, attr_name), set())
        if len(bucket) <= self._cap:
            bucket.add(text)

    def attach(self, attributes_by_type: dict) -> None:
        """Add a sorted `values` list to each string attribute that stayed within the cap."""
        for type_name, attrs in (attributes_by_type or {}).items():
            for attr in attrs or []:
                if str(attr.get("type", "")).lower() != "string":
                    continue
                bucket = self._seen.get((type_name, attr.get("name")))
                if bucket and len(bucket) <= self._cap:
                    # Sorted rather than insertion-ordered: the prompt must be identical
                    # across runs (CLAUDE.md 2.1), and dict/set order is not a guarantee.
                    attr["values"] = sorted(bucket)


def _normalize_qualifier(qualifier: Optional[str]) -> Optional[str]:
    """Map absent/placeholder qualifiers to None.

    OCEL XML writes the literal string "None" when a relationship has no qualifier.
    """
    if qualifier is None:
        return None
    qualifier = str(qualifier).strip()
    if qualifier in ("", "None", "none", "null"):
        return None
    return qualifier


def _schema_cache_key(ocel_path: str) -> tuple:
    stat = os.stat(ocel_path)
    return (os.path.abspath(ocel_path), stat.st_mtime_ns, stat.st_size,
            _SCHEMA_FORMAT_VERSION)


def _extract_schema_json(ocel_path: str) -> dict[str, Any]:
    """Extract the type-level schema from an OCEL 2.0 JSON log."""
    with open(ocel_path, "r", encoding="utf-8") as f:
        log = json.load(f)

    event_attributes = {
        et["name"]: [
            {"name": a.get("name"), "type": a.get("type")}
            for a in et.get("attributes", []) or []
        ]
        for et in log.get("eventTypes", []) or []
    }
    object_attributes = {
        ot["name"]: [
            {"name": a.get("name"), "type": a.get("type")}
            for a in ot.get("attributes", []) or []
        ]
        for ot in log.get("objectTypes", []) or []
    }

    objects = log.get("objects", []) or []
    id_to_type = {o.get("id"): o.get("type") for o in objects}

    # Values are gathered inside the relation walks below rather than in a separate pass —
    # these loops already visit every event and object, and the log may be gigabytes.
    event_values = _BoundedValues(_MAX_ATTRIBUTE_VALUES)
    object_values = _BoundedValues(_MAX_ATTRIBUTE_VALUES)

    e2o: set[tuple] = set()
    for event in log.get("events", []) or []:
        etype = event.get("type")
        for attr in event.get("attributes", []) or []:
            event_values.add(etype, attr.get("name"), attr.get("value"))
        for rel in event.get("relationships", []) or []:
            otype = id_to_type.get(rel.get("objectId"))
            if etype and otype:
                e2o.add((etype, otype, _normalize_qualifier(rel.get("qualifier"))))

    o2o: set[tuple] = set()
    for obj in objects:
        otype = obj.get("type")
        for attr in obj.get("attributes", []) or []:
            object_values.add(otype, attr.get("name"), attr.get("value"))
        for rel in obj.get("relationships", []) or []:
            target = id_to_type.get(rel.get("objectId"))
            if otype and target:
                o2o.add((otype, target, _normalize_qualifier(rel.get("qualifier"))))

    event_values.attach(event_attributes)
    object_values.attach(object_attributes)

    return {
        "event_types": sorted(event_attributes.keys()),
        "object_types": sorted(object_attributes.keys()),
        "event_attributes": event_attributes,
        "object_attributes": object_attributes,
        "e2o": sorted(e2o),
        "o2o": sorted(o2o),
    }


def _extract_schema_xml(ocel_path: str) -> dict[str, Any]:
    """Extract the type-level schema from an OCEL 2.0 XML log."""
    root = ET.parse(ocel_path).getroot()

    def _type_attributes(section_tag: str, item_tag: str) -> dict[str, list]:
        result: dict[str, list] = {}
        section = root.find(section_tag)
        if section is None:
            return result
        for item in section.findall(item_tag):
            name = item.get("name")
            if not name:
                continue
            attrs_node = item.find("attributes")
            attrs = []
            if attrs_node is not None:
                for a in attrs_node.findall("attribute"):
                    attrs.append({"name": a.get("name"), "type": a.get("type")})
            result[name] = attrs
        return result

    event_attributes = _type_attributes("event-types", "event-type")
    object_attributes = _type_attributes("object-types", "object-type")

    id_to_type: dict[str, str] = {}
    objects_node = root.find("objects")
    if objects_node is not None:
        for obj in objects_node.findall("object"):
            if obj.get("id"):
                id_to_type[obj.get("id")] = obj.get("type")

    # Dialects differ: some logs nest <relationship object-id=... />, others nest
    # <object object-id=... />. Accept any child element carrying an object-id.
    def _related_ids(parent: ET.Element) -> list[tuple[str, Optional[str]]]:
        rels = parent.find("objects")
        if rels is None:
            return []
        found = []
        for rel in rels:
            object_id = rel.get("object-id") or rel.get("objectId")
            if object_id:
                found.append((object_id, _normalize_qualifier(rel.get("qualifier"))))
        return found

    def _collect_values(item: ET.Element, type_name: Optional[str],
                        sink: "_BoundedValues") -> None:
        """Gather this element's attribute values. XML nests them as <attributes>."""
        for attrs_node in item.findall("attributes"):
            for a in attrs_node.findall("attribute"):
                sink.add(type_name, a.get("name"), a.text)

    event_values = _BoundedValues(_MAX_ATTRIBUTE_VALUES)
    object_values = _BoundedValues(_MAX_ATTRIBUTE_VALUES)

    e2o: set[tuple] = set()
    events_node = root.find("events")
    if events_node is not None:
        for event in events_node.findall("event"):
            etype = event.get("type")
            if etype is None:
                continue
            _collect_values(event, etype, event_values)
            for object_id, qualifier in _related_ids(event):
                otype = id_to_type.get(object_id)
                if otype:
                    e2o.add((etype, otype, qualifier))

    # OCEL XML permits object-to-object relationships in the same nested form.
    o2o: set[tuple] = set()
    if objects_node is not None:
        for obj in objects_node.findall("object"):
            otype = obj.get("type")
            if otype is None:
                continue
            _collect_values(obj, otype, object_values)
            for object_id, qualifier in _related_ids(obj):
                target = id_to_type.get(object_id)
                if target:
                    o2o.add((otype, target, qualifier))

    event_values.attach(event_attributes)
    object_values.attach(object_attributes)

    return {
        "event_types": sorted(event_attributes.keys()),
        "object_types": sorted(object_attributes.keys()),
        "event_attributes": event_attributes,
        "object_attributes": object_attributes,
        "e2o": sorted(e2o),
        "o2o": sorted(o2o),
    }


def _disk_cache_path(ocel_path: str) -> str:
    """Sidecar path holding the cached schema for a log."""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".schema_cache")
    stem = os.path.basename(ocel_path).replace(".", "_")
    return os.path.join(cache_dir, f"{stem}.schema.json")


def _load_disk_cache(ocel_path: str, key: tuple) -> Optional[dict]:
    path = _disk_cache_path(ocel_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    # Invalidate if the log changed (mtime/size) or if the extractor's output shape
    # changed (_SCHEMA_FORMAT_VERSION) — the whole key is persisted, so a version bump
    # retires every cached file without anyone having to remember to delete them.
    if payload.get("key") != list(key):
        return None
    schema = payload.get("schema")
    if not isinstance(schema, dict):
        return None
    # JSON round-trips tuples to lists; restore tuple form for the relation graphs.
    schema["e2o"] = [tuple(t) for t in schema.get("e2o", [])]
    schema["o2o"] = [tuple(t) for t in schema.get("o2o", [])]
    return schema


def _store_disk_cache(ocel_path: str, key: tuple, schema: dict) -> None:
    path = _disk_cache_path(ocel_path)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"key": list(key), "schema": schema}, f)
    except Exception:
        pass  # a cache failure must never break extraction


def extract_ocel_schema(ocel_path: str) -> dict[str, Any]:
    """Extract the full type-level schema of an OCEL log.

    Unlike `events_and_objects_from_ocel`, which returns only entity names, this also
    returns declared attribute names/types and the type-level E2O and O2O relation
    graphs. These are what the LLM needs in order to emit a correct `attribute_name`
    or a valid `O2E`/`O2O` filter instead of guessing.

    Args:
        ocel_path: Path to an OCEL 2.0 log (.json/.jsonocel or .xml/.xmlocel).

    Returns:
        A dict with keys:
          event_types:       sorted list of event type names
          object_types:      sorted list of object type names
          event_attributes:  {event_type: [{"name", "type"}, ...]}
          object_attributes: {object_type: [{"name", "type"}, ...]}
          e2o:               sorted list of (event_type, object_type, qualifier|None)
          o2o:               sorted list of (object_type, object_type, qualifier|None)

    Raises:
        ValueError: If the file extension is not a recognised OCEL format.
    """
    key = _schema_cache_key(ocel_path)
    cached = _SCHEMA_CACHE.get(key)
    if cached is not None:
        return cached

    # Large logs (e.g. bpic2017 takes ~80s) would otherwise be re-parsed by every Slurm
    # shard; persist the result so extraction happens once per log, not once per process.
    cached = _load_disk_cache(ocel_path, key)
    if cached is not None:
        _SCHEMA_CACHE[key] = cached
        return cached

    lowered = ocel_path.lower()
    if lowered.endswith((".json", ".jsonocel")):
        schema = _extract_schema_json(ocel_path)
    elif lowered.endswith((".xml", ".xmlocel")):
        schema = _extract_schema_xml(ocel_path)
    else:
        raise ValueError(f"Unrecognised OCEL format for schema extraction: {ocel_path}")

    _SCHEMA_CACHE[key] = schema
    _store_disk_cache(ocel_path, key, schema)
    return schema


def events_and_objects_from_ocel(ocel_path):
    # path to your OCEL file (e.g. .json, .jsonocel, .xml, .xmlocel)
    ocel_path = ocel_path

    ocel = {"events": [], "objects": []}

    if ocel_path.endswith(".json") or ocel_path.endswith(".jsonocel") or ocel_path.endswith(".jsonocel"):
        with open(ocel_path, "r", encoding="utf-8") as f:
            ocel_data = json.load(f)

            # Support OCEL 2.0 root-level eventTypes / objectTypes
            if "eventTypes" in ocel_data:
                ocel["events"] = ocel_data["eventTypes"]
            else:
                ocel["events"] = ocel_data.get("ocel:events", {})
                if isinstance(ocel["events"], dict):
                    ocel["events"] = [dict(v, id=k) for k, v in ocel["events"].items()]
                elif not ocel["events"]:
                    ocel["events"] = ocel_data.get("events", [])

            if "objectTypes" in ocel_data:
                ocel["objects"] = ocel_data["objectTypes"]
            else:
                ocel["objects"] = ocel_data.get("ocel:objects", {})
                if isinstance(ocel["objects"], dict):
                    ocel["objects"] = [dict(v, id=k) for k, v in ocel["objects"].items()]
                elif not ocel["objects"]:
                    ocel["objects"] = ocel_data.get("objects", [])

    elif ocel_path.endswith(".xml") or ocel_path.endswith(".xmlocel"):
        import xml.etree.ElementTree as ET
        tree = ET.parse(ocel_path)
        root = tree.getroot()

        # Parse events
        events_node = root.find(".//events")
        if events_node is not None:
            for ev in events_node.findall("event"):
                ev_dict = dict(ev.attrib)
                ocel["events"].append(ev_dict)

        # Parse objects
        objects_node = root.find(".//objects")
        if objects_node is not None:
            for obj in objects_node.findall("object"):
                obj_dict = dict(obj.attrib)
                ocel["objects"].append(obj_dict)


    # Get all event names (checking for 'name' (OCEL 2.0 specs), 'type', 'activity', or 'ocel:activity' attributes)
    # This collects unique values and preserves string representation.
    event_names = sorted({
        e.get("name") or e.get("type") or e.get("activity") or e.get("ocel:activity") 
        for e in ocel.get("events", []) 
        if (e.get("name") or e.get("type") or e.get("activity") or e.get("ocel:activity")) is not None
    })
    # print("Event names:", event_names)


    # Get object names: OCEL distinguishes object types and object IDs.
    # If objects have a 'name' attribute, collect that; otherwise show type:ID.
    object_names = set()
    unique_objects = set()  # to track unique object types
    for obj in ocel.get("objects", []):
        obj_id = obj.get("id", "UNKNOWN_ID")
        # object type stored in 'name', 'type' or 'ocel:type'
        otype = obj.get("name") or obj.get("type") or obj.get("ocel:type") or "UNKNOWN_TYPE"

        # optional human-readable name attribute
        oname = obj.get("ocel:name") or f"{otype}:{obj_id}"
        # However if the 'ocel:name' is missing and it's OCEL 2.0 objectTypes, otype IS the object name.
        if obj.get("name"): 
            oname = obj.get("name") + ":" + obj_id

        obj_type, obj_id = oname.split(":") if ":" in oname else (oname, "UNKNOWN_ID")
        if obj.get("name"):
            obj_type = obj.get("name") # Override for OCEL 2.0 list

        unique_objects.add(obj_type)
        object_names.add((obj_type, obj_id))

    object_names = sorted(object_names)
    # print("Object types:", sorted(list(unique_objects)))

    return event_names, sorted(list(unique_objects))




