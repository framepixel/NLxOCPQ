#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

PROMPT_STRATEGIES = [
    "schema-grammar-guided",
    "schema-allowed-entities-only",
    "schema-few-shot",
    "schema-chain-of-thought-plus",
    "schema-least-to-most",
]
DEFAULT_PROMPT_STRATEGY = "schema-grammar-guided"


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Run NLxOCPQ end-to-end in the console and persist run artifacts to files."
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language process query.",
    )
    parser.add_argument(
        "--query-file",
        help="Optional text file containing the natural language query.",
    )
    parser.add_argument(
        "--ocel",
        required=True,
        help="Path to the OCEL log file (.json or .xml) used for extraction and execution.",
    )
    parser.add_argument(
        "--backend",
        choices=["local-gpu", "hf-provider"],
        required=True,
        help="Inference backend: local GPU model execution or Hugging Face providers.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier or local model path.",
    )
    parser.add_argument(
        "--hf-provider",
        default="",
        help="Optional Hugging Face provider override (only for --backend hf-provider).",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN", ""),
        help="Optional Hugging Face token. Falls back to HF_TOKEN environment variable.",
    )
    parser.add_argument(
        "--prompt-strategy",
        default=DEFAULT_PROMPT_STRATEGY,
        choices=PROMPT_STRATEGIES,
        help="Prompting strategy used by the NL-to-OCPQ pipeline.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for run artifacts (default: results).",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional fixed run directory name. If omitted, a timestamped name is used.",
    )
    parser.add_argument(
        "--local-device",
        default="cuda",
        help="Device preference for local GPU backend (default: cuda).",
    )
    parser.add_argument(
        "--local-dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for local GPU backend.",
    )
    return parser


def _resolve_query(args):
    if args.query_file:
        with open(args.query_file, "r", encoding="utf-8") as f:
            query_from_file = f.read().strip()
        if not query_from_file:
            raise ValueError("--query-file was provided but the file is empty.")
        return query_from_file

    if args.query and args.query.strip():
        return args.query.strip()

    raise ValueError("Provide a query as positional argument or with --query-file.")


def _safe_json_payload(value):
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                try:
                    return json.loads(stripped)
                except Exception:
                    return {"raw_text": value}
            return {"raw_text": value}
        return value
    return {"raw_repr": str(value)}


def _unique_run_dir(base_dir, requested_name):
    base_name = requested_name or f"query_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    candidate = base_dir / base_name
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        alt = base_dir / f"{base_name}_{idx:02d}"
        if not alt.exists():
            return alt
        idx += 1


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main():
    parser = _build_parser()
    args = parser.parse_args()

    try:
        query = _resolve_query(args)
    except Exception as exc:
        parser.error(str(exc))

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = _unique_run_dir(output_root, args.run_name.strip())
    run_dir.mkdir(parents=True, exist_ok=False)

    debug_log = run_dir / "debug.log"
    
    def debug_print(msg):
        print(f"[DEBUG] {msg}", flush=True)
        with open(debug_log, "a") as f:
            f.write(f"[DEBUG] {msg}\n")

    generated_query_path = run_dir / "generated_ocpq.json"
    debug_print("Starting main()")

    try:
        debug_print("Importing LLM module...")
        from LLM import (
            dataset,
            embeddings,
            generate_query_result_summary,
            get_hf_backend_description,
            nl_to_ocpq_pipeline,
            set_hf_client,
            set_local_gpu_client,
        )
        debug_print("LLM import successful")
    except Exception as exc:
        debug_print(f"LLM import failed: {exc}")
        summary = {
            "status": "failed",
            "run_dir": str(run_dir.resolve()),
            "error": f"Failed to import runtime dependencies: {exc}",
        }
        _write_json(run_dir / "run_summary.json", summary)
        print(f"Run directory: {run_dir.resolve()}")
        print("Status: failed")
        print(summary["error"])
        sys.exit(1)

    input_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "ocel_path": args.ocel,
        "backend": args.backend,
        "model": args.model,
        "hf_provider": args.hf_provider,
        "prompt_strategy": args.prompt_strategy,
        "local_device": args.local_device,
        "local_dtype": args.local_dtype,
    }
    _write_json(run_dir / "query_input.json", input_payload)

    debug_print(f"Backend: {args.backend}, Model: {args.model}")
    
    if args.backend == "local-gpu":
        debug_print("Calling set_local_gpu_client()...")
        set_local_gpu_client(
            model_name=args.model,
            token=args.hf_token or None,
            device=args.local_device,
            dtype=args.local_dtype,
        )
        debug_print("set_local_gpu_client() completed")
    else:
        debug_print("Calling set_hf_client()...")
        set_hf_client(
            token=args.hf_token or None,
            provider=args.hf_provider or None,
            model_name=args.model,
        )
        debug_print("set_hf_client() completed")

    summary = {
        "status": "failed",
        "backend_description": get_hf_backend_description(),
        "run_dir": str(run_dir.resolve()),
        "files": {
            "query_input": "query_input.json",
            "generated_ocpq": "generated_ocpq.json",
            "extracted_structure": "extracted_structure.json",
            "planned_query_tree": "planned_query_tree.json",
            "query_stdout": "query_executor_stdout.txt",
            "query_stderr": "query_executor_stderr.txt",
            "query_result": "query_result.json",
            "result_summary": "result_summary.json",
        },
    }

    try:
        debug_print("Calling nl_to_ocpq_pipeline()...")
        structure, tree, ocpq_json, query_stdout, query_result_file_path = nl_to_ocpq_pipeline(
            query=query,
            dataset=dataset,
            embeddings=embeddings,
            output_file=str(generated_query_path),
            ocel_path=args.ocel,
            model_name=args.model,
            prompt_strategy=args.prompt_strategy,
        )
        debug_print("nl_to_ocpq_pipeline() completed")

        _write_json(run_dir / "extracted_structure.json", _safe_json_payload(structure))
        _write_json(run_dir / "planned_query_tree.json", _safe_json_payload(tree))

        if not generated_query_path.exists():
            _write_json(generated_query_path, _safe_json_payload(ocpq_json))

        stdout_text = query_stdout if isinstance(query_stdout, str) else str(query_stdout)
        with open(run_dir / "query_executor_stdout.txt", "w", encoding="utf-8") as f:
            f.write(stdout_text)

        stderr_text = ""
        if isinstance(query_stdout, str) and query_stdout.lower().startswith("error running query automatically"):
            stderr_text = query_stdout
        with open(run_dir / "query_executor_stderr.txt", "w", encoding="utf-8") as f:
            f.write(stderr_text)

        copied_result_file = None
        if query_result_file_path and os.path.exists(query_result_file_path):
            copied_result_file = run_dir / "query_result.json"
            shutil.copyfile(query_result_file_path, copied_result_file)
            
            # Generate human-readable summary from the result
            try:
                with open(copied_result_file, "r", encoding="utf-8") as f:
                    result_data = json.load(f)
                result_summary = generate_query_result_summary(result_data)
                _write_json(run_dir / "result_summary.json", result_summary)
            except Exception as summary_exc:
                debug_print(f"Failed to generate result summary: {summary_exc}")

        summary["status"] = "ok"
        summary["query_result_export_found"] = bool(copied_result_file)
        summary["query_result_export_source"] = query_result_file_path
        debug_print("Pipeline completed successfully")
    except Exception as exc:
        debug_print(f"Pipeline error: {exc}")
        summary["error"] = str(exc)
    finally:
        _write_json(run_dir / "run_summary.json", summary)

    print(f"Run directory: {run_dir.resolve()}")
    print(f"Status: {summary['status']}")
    print(f"Backend: {summary['backend_description']}")
    if summary.get("error"):
        print(f"Error: {summary['error']}")

    if summary["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
