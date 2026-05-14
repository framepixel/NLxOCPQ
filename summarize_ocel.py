#!/usr/bin/env python3
# coding: utf-8
"""summarize_ocel.py

Reads an OCEL file (JSON/JSONOCEL or XML/XMLOCEL) and prints a summary:
- event names and counts
- object types, counts, and combined attribute keys

Usage:
    python3 NLxOCPQ/summarize_ocel.py path/to/log.json
    python3 NLxOCPQ/summarize_ocel.py path/to/log.json --json
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any


def load_ocel(ocel_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (events, objects) as lists of dicts.

    Supports OCEL 2.0 (eventTypes/objectTypes) and common OCEL serializations.
    """
    events: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []

    if ocel_path.endswith(('.json', '.jsonocel')):
        with open(ocel_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # events
        if 'eventTypes' in data:
            # OCEL 2.0: eventTypes may be a dict or list
            et = data['eventTypes']
            if isinstance(et, dict):
                events = [dict(v, id=k) for k, v in et.items()]
            else:
                events = et
        else:
            ev = data.get('ocel:events') or data.get('events') or {}
            if isinstance(ev, dict):
                events = [dict(v, id=k) for k, v in ev.items()]
            else:
                events = ev or []

        # objects
        if 'objectTypes' in data:
            ot = data['objectTypes']
            if isinstance(ot, dict):
                objects = [dict(v, id=k) for k, v in ot.items()]
            else:
                objects = ot
        else:
            ob = data.get('ocel:objects') or data.get('objects') or {}
            if isinstance(ob, dict):
                objects = [dict(v, id=k) for k, v in ob.items()]
            else:
                objects = ob or []

    elif ocel_path.endswith(('.xml', '.xmlocel')):
        import xml.etree.ElementTree as ET
        tree = ET.parse(ocel_path)
        root = tree.getroot()

        # events
        events_node = root.find('.//events')
        if events_node is not None:
            for ev in events_node.findall('event'):
                events.append(dict(ev.attrib))

        # objects
        objects_node = root.find('.//objects')
        if objects_node is not None:
            for obj in objects_node.findall('object'):
                objects.append(dict(obj.attrib))

    else:
        raise ValueError('Unsupported file extension; provide .json/.jsonocel or .xml/.xmlocel')

    return events, objects


def summarize(events: List[Dict[str, Any]], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Events: collect activity/name/type values and counts
    event_names = []
    for e in events:
        name = e.get('name') or e.get('type') or e.get('activity') or e.get('ocel:activity')
        if name is not None:
            event_names.append(str(name))
    ev_counter = Counter(event_names)
    out['events'] = {
        'unique_names': sorted(list(ev_counter.keys())),
        'counts': dict(ev_counter),
        'total_event_definitions': len(events),
    }

    # Objects: group by type/name and collect attribute keys and sample objects
    obj_types: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for o in objects:
        # object type/name: OCEL may store in 'name', 'type', or 'ocel:type'
        otype = o.get('name') or o.get('type') or o.get('ocel:type') or 'UNKNOWN_TYPE'
        obj_types[str(otype)].append(o)

    obj_summary: Dict[str, Any] = {}
    for typ, items in obj_types.items():
        keys = set()
        samples = []
        for it in items[:5]:  # keep up to 5 samples
            keys.update(it.keys())
            samples.append(it)
        obj_summary[typ] = {
            'count': len(items),
            'attribute_keys': sorted(list(keys)),
            'samples': samples,
        }

    out['objects'] = {
        'total_object_definitions': len(objects),
        'types': obj_summary,
    }

    return out


def pretty_print(summary: Dict[str, Any], stream=sys.stdout) -> None:
    # Events
    ev = summary['events']
    print(f"Total event definitions: {ev.get('total_event_definitions', 0)}", file=stream)
    print(f"Unique event names ({len(ev['unique_names'])}):", file=stream)
    for name, cnt in sorted(ev['counts'].items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {name}: {cnt}", file=stream)
    print('', file=stream)

    # Objects
    obj = summary['objects']
    print(f"Total object definitions: {obj.get('total_object_definitions', 0)}", file=stream)
    print(f"Object types ({len(obj['types'])}):", file=stream)
    for typ, info in sorted(obj['types'].items(), key=lambda x: (-x[1]['count'], x[0])):
        print(f"- {typ}: {info['count']} definitions", file=stream)
        print(f"    attribute keys: {', '.join(info['attribute_keys'])}", file=stream)
        # Print small samples (id and up to 3 attributes)
        for s in info['samples']:
            sid = s.get('id') or s.get('ocel:id') or s.get('object_id') or '<no-id>'
            # show up to 3 other attributes
            attrs = {k: v for k, v in s.items() if k not in ('id', 'name', 'type', 'ocel:id', 'ocel:name')}
            sample_items = ', '.join(f"{k}={v}" for k, v in list(attrs.items())[:3])
            print(f"    sample: id={sid} {sample_items}", file=stream)
        print('', file=stream)


def build_arg_parser():
    p = argparse.ArgumentParser(description='Summarize an OCEL log (events and objects)')
    p.add_argument('ocel_path', help='Path to the OCEL file (.json, .jsonocel, .xml, .xmlocel)')
    p.add_argument('--json', action='store_true', help='Emit machine-readable JSON summary')
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        events, objects = load_ocel(args.ocel_path)
    except Exception as e:
        print('Error loading OCEL:', e, file=sys.stderr)
        return 2

    summary = summarize(events, objects)
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        pretty_print(summary)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
