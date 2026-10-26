#!/usr/bin/env python3
"""Shared loading, per-question values and output for the thesis analysis scripts.

WHY
---
The thesis quotes paired differences, per-arm tables and agent statistics that were computed by
throwaway code (Thesis/script/FLAGS.md NS-01, NS-02, NS-05..NS-07). The scripts that replace that
code -- `paired_stats.py`, `matrix_tables.py`, `agent_gate_report.py`,
`agent_mechanism_stats.py` -- all need the same four things, and each is easy to get quietly
wrong, so they live here once:

  * **Which records belong to one question.** `case_id` is NOT an identity: it restarts at 1 in
    every case shard, so a merged run holds 29 distinct `case_id`s for 135 questions. Pairing on
    it silently compares different questions. The entry key is the one `eval_testset._record_key`
    uses: the question text plus the log's basename (`entry_key`).
  * **What one record scores.** `eval_testset.py` is the single source of truth for metrics
    (CLAUDE.md 2.5), but importing it pulls in `LLM.py` and with it torch and transformers. So the
    per-record values are *read*, never computed: strict F1 is `similarity.combined.f1` with the
    strict-zero convention of `_build_summary._result_f1_values`, skill is the stored
    `metrics.skill`, structural F1 is `metrics.structural.f1`.
  * **Proof that the reading is right.** `assert_reproduces_summary` recomputes every condition's
    means from the records and compares them with the run's own `summary.json`. A script that
    passes it reads the records exactly as the evaluator aggregated them; one that fails stops
    before it prints a number.
  * **Where results go, and what they say about their inputs.** JSON goes to
    `eval_runs/analysis/`, LaTeX tables to the thesis (`paths.THESIS_TABLES`). Both carry the
    command line, the sha256 of every input, and the run's pipeline fingerprints, so Appendix D can
    list one command per table.

Every figure here is per question: a question's value is the mean over its draws (`repeat`)
before anything is paired or pooled, which is how notes 11 §44.4 pre-registered it.
"""
import datetime
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import paths

# Tolerance of the reproduction guard. The means are recomputed from the same floats the
# evaluator summed, in the same order, so any real disagreement is far larger than this.
SUMMARY_TOLERANCE = 1e-6

POPULATIONS = ("discriminating", "non_discriminating", "all")

# Human-readable population labels, used in every table caption so that no number appears without
# its denominator (FLAGS.md IC-08: summary.json's top-level F1 is over 169, the notes quote 135).
POPULATION_LABELS = {
    "discriminating": "the 135 discriminating questions",
    "non_discriminating": "the 34 non-discriminating questions",
    "all": "all 169 questions",
}

EntryKey = Tuple[str, str]


# --------------------------------------------------------------------------------------------
# Per-record values. Each mirrors the evaluator's aggregation in `eval_testset._build_summary`;
# `assert_reproduces_summary` is the check that they still do.
# --------------------------------------------------------------------------------------------

def strict_f1(record: dict) -> float:
    """Root-situation F1, scoring a record without a comparable result as 0.

    Mirrors `_build_summary._result_f1_values` (strict branch): a query that did not parse, did
    not execute or timed out counts as a failure to answer, not as missing.
    """
    value = ((record.get("similarity") or {}).get("combined") or {}).get("f1")
    return 0.0 if value is None else float(value)


def executed_f1(record: dict) -> Optional[float]:
    """Root-situation F1 only where a result exists; None otherwise. Never report it alone."""
    value = ((record.get("similarity") or {}).get("combined") or {}).get("f1")
    return None if value is None else float(value)


def variant_f1(record: dict, variant: str) -> Optional[float]:
    """One `SIMILARITY_VARIANTS` F1 under the strict convention (`eval_testset`, same rule).

    A record without a result scores 0, except where the variant is undefined on its entry (the
    audit's floor is present but None, e.g. `violation` where gold flags nothing): None there.
    """
    similarity = record.get("similarity")
    if similarity is not None:
        scored = (similarity.get("variants") or {}).get(variant)
        return None if scored is None else float(scored.get("f1"))
    floors = record.get("baseline_f1_variants") or {}
    if floors and variant in floors and floors[variant] is None:
        return None
    return 0.0


def kind_projected_f1(record: dict) -> Optional[float]:
    """The `kind_projected` diagnostic F1 (CLAUDE.md 6.2); never a primary score."""
    return variant_f1(record, "kind_projected")


def skill(record: dict) -> Optional[float]:
    """The stored strict skill score; None where the floor is 1.0 (undefined, by design)."""
    value = (record.get("metrics") or {}).get("skill")
    return None if value is None else float(value)


# ocpq_cli's stdout markers, copied from `eval_testset` (_CLI_*_MARKER). Used only for records
# written before `execution.evaluated` existed (the 2026-09 ablation runs).
_CLI_EVALUATED_MARKER = "Evaluated in"
_CLI_EXPORTED_MARKER = "Exported result in"
_CLI_CLIPPED_MARKER = "lines omitted (cap "


def _cli_reported_evaluation(query_stdout: Any) -> Optional[bool]:
    """Whether the CLI's stdout says it evaluated the tree (`eval_testset`, same name).

    True when the evaluation line is present, False when the CLI exported without it, None when
    the stdout is absent or clipped -- a clipped stdout never proves that evaluation did not
    happen.
    """
    if not query_stdout:
        return None
    text = str(query_stdout)
    if _CLI_EVALUATED_MARKER in text:
        return True
    if _CLI_CLIPPED_MARKER in text:
        return None
    if _CLI_EXPORTED_MARKER in text:
        return False
    return None


def evaluated(record: dict) -> float:
    """1.0 if ocpq_cli exited cleanly AND evaluated the tree, else 0.0.

    Mirrors `eval_testset._execution_succeeded`, the corrected meaning of
    `execution_success_rate` (2026-09-14), including its stdout fallback for records written
    before `execution.evaluated` existed.
    """
    execution = record.get("execution") or {}
    if not execution.get("success"):
        return 0.0
    if record.get("status") == "execution_failed" or execution.get("cli_error"):
        return 0.0
    if execution.get("evaluated") is False:
        return 0.0
    return 0.0 if _cli_reported_evaluation(record.get("query_stdout")) is False else 1.0


def structural_f1(record: dict) -> Optional[float]:
    """Building-block F1 of the generated query against gold (`metrics.structural.f1`)."""
    value = (((record.get("metrics") or {}).get("structural")) or {}).get("f1")
    return None if value is None else float(value)


def schema_valid(record: dict) -> float:
    """1.0 if the generated query passed the schema check, else 0.0."""
    return 1.0 if (record.get("metrics") or {}).get("schema_valid") else 0.0


METRICS: Dict[str, Callable[[dict], Optional[float]]] = {
    "strict_f1": strict_f1,
    "executed_f1": executed_f1,
    "skill": skill,
    "evaluated": evaluated,
    "structural_f1": structural_f1,
    "schema_valid": schema_valid,
    "kind_projected_f1": kind_projected_f1,
}

# summary.json key for each metric, for the reproduction guard.
SUMMARY_KEYS = {
    "strict_f1": "result_set_f1_strict",
    "executed_f1": "result_set_f1_executed",
    "skill": "skill_score",
    "evaluated": "execution_success_rate",
    "structural_f1": "structural_f1",
    "schema_valid": "schema_valid_rate",
}


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of the non-None values, None if there are none (the evaluator's `_mean`)."""
    kept = [v for v in values if v is not None]
    return sum(kept) / len(kept) if kept else None


# --------------------------------------------------------------------------------------------
# Identity and selection.
# --------------------------------------------------------------------------------------------

def entry_key(record: dict) -> EntryKey:
    """Identity of a testset entry: (question text, log basename), as in `_record_key`.

    Raises:
        ValueError: if the record has no `nl_query`. `case_id` is never used as a fallback here,
            because it restarts in every case shard and would pair different questions.
    """
    question = record.get("nl_query")
    if not question:
        raise ValueError(f"record without nl_query (case_id={record.get('case_id')}); "
                         "cannot identify its testset entry")
    return question, os.path.basename(record.get("ocel_path") or "")


def in_population(record: dict, population: str) -> bool:
    """Whether a record belongs to a stratum (`discriminating`, `non_discriminating`, `all`)."""
    if population == "all":
        return True
    if population == "discriminating":
        return record.get("discriminating") is True
    if population == "non_discriminating":
        return record.get("discriminating") is False
    raise ValueError(f"unknown population {population!r}; expected one of {POPULATIONS}")


def case_values(records: List[dict], metric: str, population: str = "discriminating",
                config_ids: Optional[Iterable[str]] = None,
                exclude_logs: Iterable[str] = (), draw: Optional[int] = None,
                ) -> Dict[EntryKey, float]:
    """One value per question: the metric averaged over draws and, if given, over conditions.

    Args:
        records: Prediction records of one or more runs.
        metric: A key of `METRICS`.
        population: Stratum to keep.
        config_ids: Conditions to pool. None keeps every condition in `records`. Pooling several
            (notes 11 §39.4 averages each question over its six strategy x validation conditions)
            averages within the question before anything is paired.
        exclude_logs: Log basenames to drop (a post-hoc subset; the caller must label it so).
        draw: Keep only this `repeat` (per-draw analyses); None averages over all draws.

    Returns:
        `{entry_key: value}`. Questions whose metric is undefined on every record (skill on a
        floor-1.0 entry) are absent, so the caller sees the reduced denominator.

    Raises:
        ValueError: on a duplicated (entry, condition, draw) -- a merge error that would
            silently double-weight a question.
    """
    wanted = None if config_ids is None else set(config_ids)
    excluded = set(exclude_logs)
    reader = METRICS[metric]
    seen = set()
    per_entry: Dict[EntryKey, List[float]] = defaultdict(list)
    for record in records:
        if wanted is not None and record.get("config_id") not in wanted:
            continue
        if not in_population(record, population):
            continue
        if draw is not None and record.get("repeat") != draw:
            continue
        key = entry_key(record)
        if key[1] in excluded:
            continue
        identity = (key, record.get("config_id"), record.get("repeat"))
        if identity in seen:
            raise ValueError(f"duplicate record {identity}: the run was merged twice")
        seen.add(identity)
        value = reader(record)
        if value is not None:
            per_entry[key].append(value)
    if wanted is not None:
        found = {r.get("config_id") for r in records} & wanted
        if found != wanted:
            raise ValueError(f"conditions not in the run: {sorted(wanted - found)}")
    return {key: sum(vals) / len(vals) for key, vals in per_entry.items()}


# --------------------------------------------------------------------------------------------
# Runs.
# --------------------------------------------------------------------------------------------

class Run:
    """One finished run directory, read-only.

    Attributes:
        path: The run directory.
        records: `predictions.json`.
        summary: `summary.json`, or None for a directory without one.
        provenance: Input hashes and consistency flags (see `provenance`).
    """

    def __init__(self, path: Path, records: List[dict], summary: Optional[dict]):
        self.path = path
        self.records = records
        self.summary = summary
        self.provenance = _provenance(path, summary)

    @property
    def config_ids(self) -> List[str]:
        """The run's conditions, in first-seen order."""
        return list(dict.fromkeys(r.get("config_id") for r in self.records))


def resolve_run(run_dir: Any) -> Path:
    """A run path relative to the repo root (or absolute), checked to hold predictions.json."""
    path = Path(run_dir)
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    if not (path / "predictions.json").exists():
        raise FileNotFoundError(f"no predictions.json in {path}")
    return path


def load_run(run_dir: Any) -> Run:
    """Load a merged or single-shard run directory without modifying it."""
    path = resolve_run(run_dir)
    records = _read_json(path / "predictions.json")
    summary_path = path / "summary.json"
    summary = _read_json(summary_path) if summary_path.exists() else None
    return Run(path, records, summary)


def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def sha256_of(path: Path) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(path: Path, summary: Optional[dict]) -> dict:
    summary = summary or {}
    return {
        "run": repo_relative(path),
        "predictions_sha256": sha256_of(path / "predictions.json"),
        "pipeline_fingerprints": sorted((summary.get("pipeline_fingerprints") or {}).keys()),
        "pipeline_consistent": summary.get("pipeline_consistent"),
        "corpus_consistent": summary.get("corpus_consistent"),
    }


def repo_relative(path: Path) -> str:
    """A path relative to the repo root, with forward slashes (absolute if outside it)."""
    try:
        return path.resolve().relative_to(paths.REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def git_head() -> Optional[str]:
    """The NLxOCPQ commit the analysis ran on, with `-dirty` if the tree has changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=paths.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=paths.REPO_ROOT, capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{head}-dirty" if dirty else head


# --------------------------------------------------------------------------------------------
# The reproduction guard.
# --------------------------------------------------------------------------------------------

def reproduction_errors(run: Run) -> List[str]:
    """Every disagreement between the records' means and the run's `summary.json`.

    Checks, per condition over all its records, and for the pooled discriminating stratum:
    strict and executed F1, skill, execution rate, structural F1 and schema validity.

    Returns:
        Human-readable mismatches; empty when the records read exactly as the evaluator
        aggregated them.
    """
    if run.summary is None:
        return [f"{run.path}: no summary.json to check against"]
    errors = []
    by_config = run.summary.get("by_config") or {}
    for config_id in run.config_ids:
        recorded = ((by_config.get(config_id) or {}).get("metrics")) or {}
        if not recorded:
            errors.append(f"{config_id}: missing from summary.json")
            continue
        records = [r for r in run.records if r.get("config_id") == config_id]
        errors.extend(_compare(config_id, records, recorded))
    pooled = ((run.summary.get("by_stratum") or {}).get("discriminating")) or {}
    if pooled:
        records = [r for r in run.records if r.get("discriminating") is True]
        errors.extend(_compare("by_stratum.discriminating", records, pooled))
    return errors


def cli_exited(record: dict) -> float:
    """1.0 if ocpq_cli exited 0 -- `eval_testset`'s `cli_exit_success_rate`, not evaluation."""
    return 1.0 if (record.get("execution") or {}).get("success") else 0.0


def _compare(label: str, records: List[dict], recorded: dict) -> List[str]:
    errors = []
    for metric, key in SUMMARY_KEYS.items():
        if key not in recorded:
            continue
        measure = METRICS[metric]
        # A summary written before 2026-09-14 has no `cli_exit_success_rate`, and its
        # `execution_success_rate` meant exactly that: ocpq_cli exited 0 (notes 11 §36.4). Check
        # it under that meaning; the tables report `evaluated` from the records either way.
        if key == "execution_success_rate" and "cli_exit_success_rate" not in recorded:
            measure = cli_exited
        got = mean(measure(r) for r in records)
        want = recorded[key]
        if got is None and want is None:
            continue
        if got is None or want is None or abs(got - want) > SUMMARY_TOLERANCE:
            errors.append(f"{label}: {key} records give {got}, summary.json says {want}")
    return errors


def assert_reproduces_summary(run: Run) -> None:
    """Stop unless the records reproduce `summary.json` (see `reproduction_errors`)."""
    errors = reproduction_errors(run)
    if errors:
        listing = "\n  ".join(errors[:20])
        raise SystemExit(f"{run.path}: records do not reproduce summary.json "
                         f"({len(errors)} mismatches):\n  {listing}")


# --------------------------------------------------------------------------------------------
# Output.
# --------------------------------------------------------------------------------------------

def command_line() -> str:
    """The command that produced the output, as typed from the repo root."""
    return " ".join(["python", *sys.argv])


def write_json(name: str, payload: dict, runs: Iterable[Run] = ()) -> Path:
    """Write `eval_runs/analysis/<name>.json` with a provenance header.

    Args:
        name: Output stem.
        payload: The analysis result.
        runs: The runs it read; their provenance is recorded.

    Returns:
        The written path.
    """
    paths.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out = paths.ANALYSIS_DIR / f"{name}.json"
    document = {
        "generated": {
            "command": command_line(),
            "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "nlxocpq_commit": git_head(),
            "inputs": [run.provenance for run in runs],
        },
        **payload,
    }
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return out


def write_tex(name: str, tabular: str, caption: str, label: str,
              runs: Iterable[Run] = (), notes: Iterable[str] = ()) -> Path:
    """Write a booktabs table fragment to `<THESIS_TABLES>/<name>.tex`.

    Args:
        name: Output stem; the chapter includes it with `\\input{tables/<name>}`.
        tabular: The complete `tabular` environment (booktabs rules only, no vertical lines).
        caption: Caption text. It must name the population; see `POPULATION_LABELS`.
        label: `\\label` key, e.g. `tab:matrix:arms`.
        runs: Runs the table was computed from (their predictions sha256 go in the header).
        notes: Extra `%` comment lines for the header (post-hoc subsets, old corpus, ...).

    Returns:
        The written path.
    """
    paths.THESIS_TABLES.mkdir(parents=True, exist_ok=True)
    out = paths.THESIS_TABLES / f"{name}.tex"
    header = [
        "% GENERATED -- do not edit. Regenerate from the NLxOCPQ repo root with:",
        f"%   {command_line()}",
        f"% NLxOCPQ commit: {git_head()}",
    ]
    header += [f"% input: {r.provenance['run']} sha256 {r.provenance['predictions_sha256'][:16]}"
               for r in runs]
    header += [f"% note: {line}" for line in notes]
    body = "\n".join([
        *header,
        # The author's placement (set by hand on the first tables, 2026-10-26): here, else top,
        # bottom or a float page.
        "\\begin{table}[hbtp]",
        "  \\centering",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        tabular.rstrip(),
        "\\end{table}",
        "",
    ])
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(body)
    return out


def fmt(value: Optional[float], digits: int = 4, signed: bool = False) -> str:
    """A number for a LaTeX table cell: fixed decimals, `--` for undefined, math minus."""
    if value is None:
        return "--"
    text = f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"
    return text.replace("-", "$-$").replace("+", "$+$") if signed else text.replace("-", "$-$")


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    """A rate as a percentage for a LaTeX cell (`\\%` escaped)."""
    return "--" if value is None else f"{100 * value:.{digits}f}\\%"
