#!/usr/bin/env python3
"""
Utility to expand a config's `models` list into paired pre-trained + fine-tuned entries.

Usage:
  python expand_models.py --config eval_config.json --out eval_config.expanded.json

Behavior:
  - If `fine_tune_map` exists in the config, use it to map base model names to fine-tuned names.
  - Otherwise, generate a fine-tuned entry by appending `-finetuned` to the model name and label.
  - The fine-tuned entries include `derived_from` and `auto_generated` markers.
"""
import argparse
import json
import os
from typing import Any, Dict


def expand_models(cfg: Dict[str, Any]) -> Dict[str, Any]:
    models = cfg.get("models", [])
    ft_map = cfg.get("fine_tune_map", {}) or {}
    expanded = []
    for m in models:
        expanded.append(m)
        name = m.get("name")
        label = m.get("label", name)

        if name in ft_map:
            mapped = ft_map[name]
            if isinstance(mapped, dict):
                ft_name = mapped.get("name")
                ft_label = mapped.get("label", ft_name or (label + "-finetuned"))
            else:
                ft_name = mapped
                ft_label = label + "-finetuned"
        else:
            ft_name = f"{name}-finetuned"
            ft_label = f"{label}-finetuned"

        if not ft_name or ft_name == name:
            # skip invalid or identical mapping
            continue

        expanded.append({
            "name": ft_name,
            "label": ft_label,
            "derived_from": name,
            "auto_generated": True,
        })

    new_cfg = dict(cfg)
    new_cfg["models"] = expanded
    return new_cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", required=True, help="Path to eval config JSON")
    p.add_argument("--out", "-o", help="Output path for expanded config (defaults to <config>.expanded.json)")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    new_cfg = expand_models(cfg)

    out_path = args.out or (os.path.splitext(args.config)[0] + ".expanded.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_cfg, f, indent=2, ensure_ascii=False)

    print(f"Wrote expanded config to {out_path}")


if __name__ == "__main__":
    main()
