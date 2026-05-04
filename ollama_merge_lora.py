#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a base model and save HF weights.

This is CPU-friendly but memory-heavy. Expect slow merges on CPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MAX_SHARD_SIZE = "2GB"
DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE_MAP = "cpu"


def _maybe_auth_token_kwargs() -> dict[str, Any]:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
    )
    return {"token": token} if token else {}


def _read_adapter_base_model(adapter_path: str) -> str | None:
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return None
    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None
    base_model = config.get("base_model_name_or_path")
    if isinstance(base_model, str) and base_model.strip():
        return base_model.strip()
    return None


def _ensure_output_dir(output_dir: str, overwrite: bool) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        return
    if not os.path.isdir(output_dir):
        raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    if os.listdir(output_dir) and not overwrite:
        raise ValueError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to continue."
        )


def _dtype_from_arg(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'. Use: {', '.join(mapping)}")
    return mapping[name]


def merge_lora(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    dtype: torch.dtype,
    device_map: str,
    max_shard_size: str,
) -> None:
    auth_kwargs = _maybe_auth_token_kwargs()
    tokenizer_source = adapter_path if os.path.isdir(adapter_path) else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **auth_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
        **auth_kwargs,
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    merged_model.eval()

    print(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter directory.")
    parser.add_argument(
        "--base-model",
        default=None,
        help=(
            "Base model name or local path. If omitted, read from adapter_config.json."
        ),
    )
    parser.add_argument("--output-dir", required=True, help="Directory for merged HF model.")
    parser.add_argument(
        "--dtype",
        default=DEFAULT_DTYPE,
        choices=["float16", "float32", "bfloat16"],
        help="Torch dtype for loading base model.",
    )
    parser.add_argument(
        "--device-map",
        default=DEFAULT_DEVICE_MAP,
        help="Device map for transformers (e.g. cpu, auto).",
    )
    parser.add_argument(
        "--max-shard-size",
        default=DEFAULT_MAX_SHARD_SIZE,
        help="Max shard size when saving HF weights (e.g. 2GB).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already contains files.",
    )
    parser.add_argument(
        "--allow-quantized-base",
        action="store_true",
        help="Allow base models that appear to be quantized (e.g. bnb-4bit).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    adapter_path = os.path.abspath(args.adapter_path)
    if not os.path.isdir(adapter_path):
        print(f"Adapter path does not exist: {adapter_path}", file=sys.stderr)
        return 1

    base_model = args.base_model or _read_adapter_base_model(adapter_path)
    if not base_model:
        print(
            "Base model not provided and not found in adapter_config.json.",
            file=sys.stderr,
        )
        return 1

    if "bnb-4bit" in base_model.lower() and not args.allow_quantized_base:
        print(
            "Warning: base model looks quantized (bnb-4bit). "
            "For GGUF conversion, prefer the full-precision base model. "
            "Pass --allow-quantized-base to silence this warning.",
            file=sys.stderr,
        )

    output_dir = os.path.abspath(args.output_dir)
    try:
        _ensure_output_dir(output_dir, args.overwrite)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    dtype = _dtype_from_arg(args.dtype)

    merge_lora(
        base_model=base_model,
        adapter_path=adapter_path,
        output_dir=output_dir,
        dtype=dtype,
        device_map=args.device_map,
        max_shard_size=args.max_shard_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
