import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from QueryExecutor import discard_result_file, run_ocpq_query
from convert_json import convert_to_ocpq


DEFAULT_OCEL_PATH = Path("../Logs/order-management.json")


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_testset(path: Path):
    if not path.exists():
        return []

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    data = json.loads(raw)
    if isinstance(data, list):
        return data

    raise ValueError(f"Expected `{path}` to contain a JSON array.")


def _write_testset(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(_dump_testset_with_compact_query(payload))
        f.write("\n")


def _dump_testset_with_compact_query(payload):
    """Pretty-print the testset, but keep `ocpq_query` compact on one line."""
    if not isinstance(payload, list):
        return json.dumps(payload, indent=2, ensure_ascii=False)

    lines = ["["]
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            rendered = json.dumps(item, ensure_ascii=False)
            comma = "," if idx < len(payload) - 1 else ""
            lines.append(f"  {rendered}{comma}")
            continue

        compact_query = json.dumps(item.get("ocpq_query"), ensure_ascii=False, separators=(",", ":"))
        compact_results = json.dumps(item.get("results"), ensure_ascii=False, separators=(",", ":"))
        # Remove compacted keys from the pretty-printed object so we can inject them on one line.
        item_without_query = {k: v for k, v in item.items() if k not in ("ocpq_query", "results")}
        has_results = "results" in item

        rendered_obj = json.dumps(item_without_query, indent=2, ensure_ascii=False)
        obj_lines = rendered_obj.splitlines()

        # Remove opening/closing braces from rendered object so we can inject compact fields.
        inner_lines = obj_lines[1:-1]
        lines.append("  {")

        if inner_lines:
            for line in inner_lines:
                lines.append(f"  {line}")
            lines[-1] = lines[-1] + ","

        if has_results:
            lines.append(f'    "ocpq_query": {compact_query},')
            lines.append(f'    "results": {compact_results}')
        else:
            lines.append(f'    "ocpq_query": {compact_query}')
        closing = "  }," if idx < len(payload) - 1 else "  }"
        lines.append(closing)

    lines.append("]")
    return "\n".join(lines)


def _read_result_file(result_file: Optional[str]):
    if not result_file:
        return None

    result_path = Path(result_file)
    if not result_path.exists():
        return None

    try:
        return _load_json(result_path)
    except json.JSONDecodeError:
        return result_path.read_text(encoding="utf-8")


def build_testset(
    test_set_path="testset.json",
    query_path="query.json",
    is_a_graph_query=False,
    nl_query=None,
    ocel_path=str(DEFAULT_OCEL_PATH),
):
    test_set_path = Path(test_set_path)
    query_path = Path(query_path)
    ocel_path = Path(ocel_path)

    if not query_path.exists():
        raise FileNotFoundError(f"Query file not found: {query_path}")
    if not ocel_path.exists():
        raise FileNotFoundError(f"OCEL file not found: {ocel_path}")

    query_json = _load_json(query_path)

    if is_a_graph_query:
        with tempfile.TemporaryDirectory() as tmpdir:
            ocpq_query_path = Path(tmpdir) / "graph_free_query.json"
            convert_to_ocpq(str(query_path), str(ocpq_query_path))
            stdout, stderr, last_file = run_ocpq_query(str(ocel_path), str(ocpq_query_path))
            executed_query_path = ocpq_query_path
            executed_query_json = _load_json(ocpq_query_path)
    else:
        stdout, stderr, last_file = run_ocpq_query(str(ocel_path), str(query_path))
        executed_query_path = query_path
        executed_query_json = query_json

    result_payload = _read_result_file(last_file)
    # The payload is already in memory and the record does not keep the path, so release the
    # caller-owned result file rather than leaving it in scratch.
    discard_result_file(last_file)

    record = {
        # "timestamp": datetime.now(timezone.utc).isoformat(),
        "nl_query": nl_query,
        # "is_graph_query": is_a_graph_query,
        # "query_path": str(query_path),
        # "executed_query_path": str(executed_query_path),
        "ocpq_query": executed_query_json,
        "ocel_path": str(ocel_path) if ocel_path else DEFAULT_OCEL_PATH,
        # "stdout": stdout,
        # "stderr": stderr,
        # "results": result_payload,
    }

    testset = _load_testset(test_set_path)
    testset.append(record)
    _write_testset(test_set_path, testset)

    return record


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Append one executed OCPQ query run to a JSON testset."
    )
    parser.add_argument("query_path", help="Path to the query JSON file.")
    parser.add_argument("test_set_path", help="Path to the testset JSON file.")
    parser.add_argument(
        "--graph-query",
        action="store_true",
        help="Convert the input React Flow query into flattened OCPQ JSON before execution.",
    )
    parser.add_argument(
        "--nl-query",
        default=None,
        help="Natural language query associated with the execution.",
    )
    parser.add_argument(
        "--ocel-path",
        default=str(DEFAULT_OCEL_PATH),
        help=f"Path to the OCEL file (default: {DEFAULT_OCEL_PATH}).",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    build_testset(
        test_set_path=args.test_set_path,
        query_path=args.query_path,
        is_a_graph_query=args.graph_query,
        nl_query=args.nl_query,
        ocel_path=args.ocel_path,
    )
