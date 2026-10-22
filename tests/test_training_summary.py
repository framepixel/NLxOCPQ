"""`training_summary.py` must read adapters and logs without torch, and miss nothing.

Pinned here: the safetensors header is read exactly (8-byte little-endian length, then JSON), so
the parameter count is the sum of the tensor shapes; a layer missing one projection is reported
as incomplete; a vision-tower tensor is counted; and the log parser takes the best of all
evaluations, not the last, and the final runtime line.

Run from the repo root:  python tests/test_training_summary.py
"""
import json
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from training_summary import PROJECTIONS, inspect_adapter, parse_log  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


def write_adapter(directory: Path, layers: int, drop=None, vision=False) -> None:
    header = {"__metadata__": {"format": "pt"}}
    for layer in range(layers):
        for module in PROJECTIONS:
            if (layer, module) == drop:
                continue
            for part, shape in (("lora_A", [16, 64]), ("lora_B", [64, 16])):
                header[f"base_model.model.model.layers.{layer}.mlp.{module}.{part}.weight"] = {
                    "dtype": "F32", "shape": shape, "data_offsets": [0, 0]}
    if vision:
        header["base_model.model.vision_tower.layers.0.q_proj.lora_A.weight"] = {
            "dtype": "F32", "shape": [1, 1], "data_offsets": [0, 0]}
    raw = json.dumps(header).encode("utf-8")
    (directory / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    (directory / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 16}))


with tempfile.TemporaryDirectory() as root:
    base = Path(root)
    write_adapter(base, layers=2)
    report = inspect_adapter(base)
    check("tensors = layers x 7 x 2", report["lora_tensors"], 2 * 7 * 2)
    check("parameters are the sum of the shapes", report["trainable_params"], 2 * 7 * 2 * 1024)
    check("a complete adapter is complete", report["tensors_per_layer_and_module_complete"], True)
    write_adapter(base, layers=2, drop=(1, "v_proj"))
    check("a missing projection is incomplete",
          inspect_adapter(base)["tensors_per_layer_and_module_complete"], False)
    write_adapter(base, layers=1, vision=True)
    check("a vision-tower tensor is counted", inspect_adapter(base)["vision_tensors"], 1)

    log = base / "job.out"
    log.write_text("\n".join([
        "Loaded 60018 supervised rows from finetune_ocpq.jsonl",
        "Train rows: 53994",
        "Validation rows: 6024 (10.0% of rows)",
        " Num examples = 53,994 | Num Epochs = 1 | Total steps = 6,750",
        " Trainable parameters = 24,707,072 of 3,873,797,120 (0.64% trained)",
        "{'eval_loss': '0.0188', 'epoch': '0.03'}",
        "{'eval_loss': '0.0150', 'epoch': '0.5'}",
        "{'eval_loss': '0.0160', 'epoch': '1'}",
        "{'train_runtime': '1.682e+04', 'train_samples_per_second': '3.21', "
        "'train_loss': '0.02029', 'epoch': '1'}",
    ]), encoding="utf-8")
    parsed = parse_log(log)
    check("rows and split", (parsed["corpus_rows"], parsed["train_rows"],
                             parsed["validation_rows"]), (60018, 53994, 6024))
    check("steps with a thousands separator", parsed["steps"], 6750)
    check("trainable and total parameters", (parsed["trainable_params_logged"],
                                             parsed["total_params"]), (24707072, 3873797120))
    check("the best evaluation, not the last", parsed["best_eval_loss"], 0.0150)
    check("the final runtime line", (parsed["train_runtime_sec"], parsed["train_loss"]),
          (16820.0, 0.02029))

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("all checks passed")
