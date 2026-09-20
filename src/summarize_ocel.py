#!/usr/bin/env python3
# coding: utf-8
"""summarize_ocel.py

Reads an OCEL file (JSON/JSONOCEL, XML/XMLOCEL, or SQLite) and prints a summary:
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

        # events — prefer actual instances over type-schema definitions
        ev_raw = data.get('ocel:events') or data.get('events')
        if ev_raw is not None:
            if isinstance(ev_raw, dict):
                events = [dict(v, id=k) for k, v in ev_raw.items()]
            else:
                events = list(ev_raw) or []
        elif 'eventTypes' in data:
            # fall back to type schema if no instance list exists
            et = data['eventTypes']
            if isinstance(et, dict):
                events = [dict(v, id=k) for k, v in et.items()]
            else:
                events = list(et) or []

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

    elif ocel_path.endswith(('.sqlite', '.db', '.sqlite3')):
        import sqlite3

        con = sqlite3.connect(ocel_path)
        con.row_factory = sqlite3.Row
        cur = con.cursor()

        # Discover all tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0].lower(): row[0] for row in cur.fetchall()}

        # ── Events ──────────────────────────────────────────────────────────
        # OCEL 2.0 (pm4py) stores events in an 'event' table with ocel_id /
        # ocel_type / ocel_time columns, plus per-type 'event_<type>' tables.
        # A 'event_map_type' table maps type names to their table-name suffix.
        event_tbl = tables.get('event')
        if event_tbl:
            # Detect id column name
            cur.execute(f'PRAGMA table_info("{event_tbl}")')
            ev_cols = {r['name'].lower(): r['name'] for r in cur.fetchall()}
            id_col = ev_cols.get('ocel_id') or ev_cols.get('event_id') or ev_cols.get('id')
            type_col = ev_cols.get('ocel_type') or ev_cols.get('event_type') or ev_cols.get('type') or ev_cols.get('activity')

            cur.execute(f'SELECT * FROM "{event_tbl}"')
            for row in cur.fetchall():
                d = dict(row)
                ev: Dict[str, Any] = {}
                if id_col:
                    ev['id'] = d.get(id_col)
                if type_col:
                    ev['type'] = d.get(type_col)
                ev.update({k: v for k, v in d.items() if k not in (id_col, type_col)})
                events.append(ev)

            # Pull per-type attributes from event_map_type + event_<suffix> tables
            if 'event_map_type' in tables:
                cur.execute(f'SELECT * FROM "{tables["event_map_type"]}"')
                for row in cur.fetchall():
                    d = dict(row)
                    etype = d.get('ocel_type') or d.get('event_type')
                    suffix = d.get('ocel_type_map') or d.get('table_name')
                    if not suffix:
                        continue
                    attr_tbl = f'event_{suffix}'
                    if attr_tbl.lower() not in tables:
                        continue
                    cur.execute(f'PRAGMA table_info("{tables[attr_tbl.lower()]}")')
                    attr_cols = [r['name'] for r in cur.fetchall() if r['name'].lower() not in ('ocel_id', 'event_id')]
                    # attach attribute keys to each event with this type
                    for ev in events:
                        if ev.get('type') == etype:
                            for c in attr_cols:
                                ev.setdefault(c, None)

        # ── Objects ─────────────────────────────────────────────────────────
        obj_tbl = tables.get('object')
        if obj_tbl:
            cur.execute(f'PRAGMA table_info("{obj_tbl}")')
            ob_cols = {r['name'].lower(): r['name'] for r in cur.fetchall()}
            id_col = ob_cols.get('ocel_id') or ob_cols.get('object_id') or ob_cols.get('id')
            type_col = ob_cols.get('ocel_type') or ob_cols.get('object_type') or ob_cols.get('type')

            cur.execute(f'SELECT * FROM "{obj_tbl}"')
            for row in cur.fetchall():
                d = dict(row)
                ob: Dict[str, Any] = {}
                if id_col:
                    ob['id'] = d.get(id_col)
                if type_col:
                    ob['type'] = d.get(type_col)
                ob.update({k: v for k, v in d.items() if k not in (id_col, type_col)})
                objects.append(ob)

            # Pull per-type attributes from object_map_type + object_<suffix> tables
            if 'object_map_type' in tables:
                cur.execute(f'SELECT * FROM "{tables["object_map_type"]}"')
                for row in cur.fetchall():
                    d = dict(row)
                    otype = d.get('ocel_type') or d.get('object_type')
                    suffix = d.get('ocel_type_map') or d.get('table_name')
                    if not suffix:
                        continue
                    attr_tbl = f'object_{suffix}'
                    if attr_tbl.lower() not in tables:
                        continue
                    cur.execute(f'PRAGMA table_info("{tables[attr_tbl.lower()]}")')
                    attr_cols = [r['name'] for r in cur.fetchall() if r['name'].lower() not in ('ocel_id', 'object_id')]
                    for ob in objects:
                        if ob.get('type') == otype:
                            for c in attr_cols:
                                ob.setdefault(c, None)

        con.close()

        if not events and not objects:
            raise ValueError(
                f'No OCEL event/object tables found in {ocel_path}. '
                f'Available tables: {sorted(tables.keys())}'
            )

    else:
        raise ValueError('Unsupported file extension; provide .json/.jsonocel, .xml/.xmlocel, or .sqlite/.db/.sqlite3')

    return events, objects


def summarize(events: List[Dict[str, Any]], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Events: collect activity/name/type values, counts, and attribute keys per type
    _EVENT_META_KEYS = {'name', 'type', 'activity', 'ocel:activity', 'id', 'ocel:id',
                        'timestamp', 'ocel:timestamp', 'time', 'ocel:time',
                        'relationships', 'ocel:omap', 'ocel:vmap', 'attributes'}
    event_names = []
    event_attrs: Dict[str, set] = defaultdict(set)
    for e in events:
        name = e.get('name') or e.get('type') or e.get('activity') or e.get('ocel:activity')
        if name is not None:
            name = str(name)
            event_names.append(name)
            # unwrap nested "attributes" list (OCEL 2.0: [{name, value}, ...])
            nested = e.get('attributes') or e.get('ocel:vmap')
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict) and 'name' in item:
                        event_attrs[name].add(item['name'])
            elif isinstance(nested, dict):
                event_attrs[name].update(nested.keys())
            # also collect any flat keys not in meta
            for k in e:
                if k not in _EVENT_META_KEYS:
                    event_attrs[name].add(k)
    ev_counter = Counter(event_names)
    out['events'] = {
        'unique_names': sorted(list(ev_counter.keys())),
        'counts': dict(ev_counter),
        'attributes': {n: sorted(list(ks)) for n, ks in event_attrs.items()},
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
    attrs_map = ev.get('attributes', {})
    for name, cnt in sorted(ev['counts'].items(), key=lambda x: (-x[1], x[0])):
        attr_keys = attrs_map.get(name, [])
        attr_str = f"  [attrs: {', '.join(attr_keys)}]" if attr_keys else ''
        print(f"  - {name}: {cnt}{attr_str}", file=stream)
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
    p.add_argument('ocel_path', help='Path to the OCEL file (.json, .jsonocel, .xml, .xmlocel, .sqlite, .db, .sqlite3)')
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
