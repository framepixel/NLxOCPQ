#!/usr/bin/env python
# coding: utf-8
import json

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
    print("Event names:", event_names)


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
    print("Object types:", sorted(list(unique_objects)))

    return event_names, sorted(list(unique_objects))




