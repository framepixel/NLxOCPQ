"""B, the non-oracle rungs: the four conditions must be distinct, and none of them may touch gold.

2026-09-17. The two best non-oracle components had never run together: notes 37 measured the
structural check (0.2500) and notes 33.4/38.4 only ESTIMATED the constant (0,1) root arity
(0.2449-0.2713 against N1's 0.2354). `slurm/submit_non_oracle_rungs.sh` runs both as a 2x2 so each
effect and their interaction are separable.

Two things can silently ruin it, and both are asserted here:

  * A RUNG SUFFIX that disagrees with `eval_testset._decoding_suffix`. All four rungs share
    model/strategy/schema indices, so the suffix is the only thing separating their directories and
    their `config_id`s. When these drifted apart before (notes 32.2), `--resume` for a non-oracle rung
    globbed the ORACLE directory and would have resumed into it without a word.
  * An ORACLE flag creeping in. B's entire value is that it is system performance rather than a
    ceiling, so `--force-root-arity` must appear only in its negated form, and `oracle_conditions`
    must stay empty.

Run from the repo root:  python tests/test_non_oracle_rungs.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_testset import _decoding_suffix  # noqa: E402

failures = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        print(f"        got  {got!r}\n        want {want!r}")
        failures.append(name)


script_path = ROOT / "slurm" / "submit_non_oracle_rungs.sh"
script = script_path.read_bytes().decode("utf-8")
config_path = ROOT / "configs" / "eval_config.non_oracle.json"
config = json.loads(config_path.read_text(encoding="utf-8"))

# (task, constant_root_arity, structural_validation) -> the suffix eval_testset will build.
RUNGS = [
    (0, False, False),
    (1, False, True),
    (2, True, False),
    (3, True, True),
]

print("--- the suffix the harness builds is the suffix the script declares ---")
for task, constant_arity, structural in RUNGS:
    suffix = _decoding_suffix(
        constrained_decoding=True,
        force_root_arity=False,
        generation_token_budget=4000,
        type_aware_filters=False,
        constant_root_arity=constant_arity,
        structural_validation=structural,
    )
    check(f"task {task} declares RUNG_SUFFIX={suffix}",
          f'RUNG_SUFFIX="{suffix}"' in script, True)

print("\n--- the four rungs are four DIFFERENT conditions ---")
declared = re.findall(r'RUNG_SUFFIX="([^"]+)"', script)
check("four suffixes declared", len(declared), 4)
check("...all distinct", len(set(declared)), 4)
check("the resume glob is built from RUNG_SUFFIX", "${RUNG_SUFFIX}_*" in script, True)
check("the resume glob looks in this run's output dir",
      "eval_runs/non_oracle/shard_m" in script, True)

print("\n--- nothing here is an oracle ---")
check("--force-root-arity appears only negated",
      re.findall(r"(?<!-)(?<!no-)--force-root-arity", script), [])
check("every rung passes --no-force-root-arity", script.count("--no-force-root-arity"), 4)
check("the config does not turn on gold-derived forcing", config.get("force_root_arity"), False)
check("no oracle- prompt strategy in the config",
      [s for s in config.get("prompt_strategies", []) if s.startswith("oracle-")], [])

print("\n--- the 2x2 is actually a 2x2 ---")
check("two rungs carry the structural check", script.count(" --structural-validation"), 2)
check("two rungs carry the constant arity", script.count(" --constant-root-arity"), 2)
check("two rungs switch the structural check off", script.count("--no-structural-validation"), 2)
check("two rungs switch the constant arity off", script.count("--no-constant-root-arity"), 2)
check("every rung is constrained (the arity pin tightens that grammar)",
      script.count("--constrained-decoding"), 4)
check("every rung shares one token budget", script.count("--token-budget 4000"), 4)

print("\n--- run isolation ---")
check("the config writes to its own directory", config["output_dir"], "eval_runs/non_oracle")
check("the script points at that config", "configs/eval_config.non_oracle.json" in script, True)
main_config = json.loads((ROOT / "configs" / "eval_config.json").read_text(encoding="utf-8"))
check("the MAIN config still has the constant arity off",
      bool(main_config.get("constant_root_arity", False)), False)
check("the MAIN config still has the structural check off",
      bool(main_config.get("structural_validation", False)), False)

print("\n--- the script itself ---")
check("no CR characters (a CRLF script dies on the cluster)", script.count("\r"), 0)
check("the array covers exactly the four rungs", "#SBATCH --array=0-3" in script, True)
check("a task outside 0-3 is refused", "is not a rung; this array is 0-3" in script, True)
check("node and GPU are recorded (notes 37.5)", "nvidia-smi" in script and "hostname" in script, True)
check("--case-shard is accepted as an argument", "--case-shard)" in script, True)

print()
if failures:
    print(f"RESULT: FAIL -> {len(failures)} check(s): {failures}")
    sys.exit(1)
print("RESULT: PASS -> four distinct non-oracle rungs, no gold anywhere in them")
