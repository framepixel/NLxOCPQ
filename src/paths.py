"""Filesystem anchors for the repo.

Every path in this project is *repo-root-relative*, never CWD-relative. Before the
reorg the two coincided -- all code sat at the root, so `open('testset.json')` and
`Path(__file__).parent / 'testset.json'` meant the same thing. They no longer do:
code lives in `src/`, data in `data/`, configs in `configs/`. Anything that used
`__file__`'s directory or a bare relative filename resolves through here instead,
so there is exactly one place to look when a path is wrong.

Two anchors point *outside* this repo -- the OCPQ Rust checkout and the OCEL log
collection are sibling directories, not vendored. Both are env-overridable because
the cluster lays them out differently from a local clone.
"""
import os
from pathlib import Path

SRC = Path(__file__).resolve().parent
REPO_ROOT = SRC.parent

DATA = REPO_ROOT / "data"
CONFIGS = REPO_ROOT / "configs"
MODELS = REPO_ROOT / "models"
EVAL_RUNS = REPO_ROOT / "eval_runs"
SCHEMA_CACHE = REPO_ROOT / ".schema_cache"
UPLOADED_LOGS = REPO_ROOT / "uploaded_logs"
# Size and sha256 of every testset log; a run on any other copy stops. See corpus_manifest.py.
CORPUS_MANIFEST = DATA / "corpus_manifest.json"

# Sibling checkouts, outside this repo.
OCPQ_CHECKOUT = Path(os.getenv("OCPQ_REPO_DIR", REPO_ROOT.parent / "OCPQ"))
OCEL_LOGS = Path(os.getenv("OCPQ_LOGS_DIR", REPO_ROOT.parent / "Logs"))

# Outputs of the post-run analysis scripts (src/analysis_common.py and its users). JSON stays in
# this repo; the LaTeX tables go straight into the thesis checkout, a sibling directory, so one
# command regenerates a table the thesis \inputs.
ANALYSIS_DIR = EVAL_RUNS / "analysis"
THESIS_TABLES = Path(
    os.getenv("THESIS_TABLES_DIR", REPO_ROOT.parent / "Thesis" / "script" / "tables"))
