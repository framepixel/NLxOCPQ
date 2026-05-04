#!/usr/bin/env python3
"""Convert a merged HF model directory to GGUF for Ollama/llama.cpp.

Requires a local llama.cpp checkout with the conversion script and quantize binary.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Iterable

DEFAULT_OUTTYPE = "f16"
DEFAULT_QUANTIZE = "Q4_K_M"


def _candidate_paths(root: str, names: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for name in names:
        paths.append(os.path.join(root, name))
    return paths


def _find_converter(llama_cpp_dir: str) -> str:
    candidates = []
    candidates += _candidate_paths(llama_cpp_dir, ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"])
    candidates += _candidate_paths(
        os.path.join(llama_cpp_dir, "scripts"),
        ["convert-hf-to-gguf.py", "convert_hf_to_gguf.py"],
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find convert-hf-to-gguf.py in llama.cpp. "
        "Set --llama-cpp-dir to your llama.cpp checkout."
    )


def _find_quantize_binary(llama_cpp_dir: str) -> str:
    candidates = []
    candidates += _candidate_paths(llama_cpp_dir, ["quantize", "llama-quantize"])
    candidates += _candidate_paths(
        os.path.join(llama_cpp_dir, "build"),
        ["quantize", "llama-quantize"],
    )
    candidates += _candidate_paths(
        os.path.join(llama_cpp_dir, "build", "bin"),
        ["quantize", "llama-quantize"],
    )
    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "Could not find quantize binary in llama.cpp build output. "
        "Build llama.cpp first (cmake --build)."
    )


def _ensure_output_path(path: str, overwrite: bool) -> None:
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"Output file exists: {path}. Use --overwrite to replace it.")


def _normalize_quantize(value: str | None) -> str | None:
    if value is None:
        return None
    lowered = value.strip().lower()
    if lowered in {"none", "no", "false", "0", "skip"}:
        return None
    return value


def convert_hf_to_gguf(
    model_dir: str,
    output_path: str,
    llama_cpp_dir: str,
    outtype: str,
    quantize: str | None,
    keep_f16: bool,
    overwrite: bool,
) -> None:
    converter = _find_converter(llama_cpp_dir)

    _ensure_output_path(output_path, overwrite)

    f16_path = output_path
    if quantize:
        base, ext = os.path.splitext(output_path)
        f16_path = f"{base}.f16{ext or '.gguf'}"
        _ensure_output_path(f16_path, overwrite)

    cmd = [sys.executable, converter, model_dir, "--outfile", f16_path]
    if outtype:
        cmd += ["--outtype", outtype]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if quantize:
        quantize_bin = _find_quantize_binary(llama_cpp_dir)
        quant_cmd = [quantize_bin, f16_path, output_path, quantize]
        print("Running:", " ".join(quant_cmd))
        subprocess.run(quant_cmd, check=True)

        if not keep_f16:
            try:
                os.remove(f16_path)
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HF model to GGUF.")
    parser.add_argument("--model-dir", required=True, help="Merged HF model directory.")
    parser.add_argument("--output", required=True, help="Output GGUF path.")
    parser.add_argument(
        "--llama-cpp-dir",
        default="./llama.cpp",
        help="Path to local llama.cpp checkout.",
    )
    parser.add_argument(
        "--outtype",
        default=DEFAULT_OUTTYPE,
        help="Output type for convert step (e.g. f16).",
    )
    parser.add_argument(
        "--quantize",
        default=DEFAULT_QUANTIZE,
        help="Quantization type (e.g. Q4_K_M). Use 'none' to skip.",
    )
    parser.add_argument(
        "--keep-f16",
        action="store_true",
        help="Keep the intermediate f16 GGUF file after quantization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model_dir = os.path.abspath(args.model_dir)
    if not os.path.isdir(model_dir):
        print(f"Model directory does not exist: {model_dir}", file=sys.stderr)
        return 1

    output_path = os.path.abspath(args.output)
    llama_cpp_dir = os.path.abspath(args.llama_cpp_dir)

    quantize = _normalize_quantize(args.quantize)

    try:
        convert_hf_to_gguf(
            model_dir=model_dir,
            output_path=output_path,
            llama_cpp_dir=llama_cpp_dir,
            outtype=args.outtype,
            quantize=quantize,
            keep_f16=args.keep_f16,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
