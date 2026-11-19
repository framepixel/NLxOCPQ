# NLxOCPQ

Translating natural-language questions about object-centric event logs into
[OCPQ](https://github.com/aarkue/OCPQ) queries with open-weight LLMs, and benchmarking how well
that works.

This repository holds the code, data, fine-tuned adapters and analysis outputs of the Master's
thesis *Benchmarking Large Language Models for Natural-Language Object-Centric Process Querying*
(Ayoub Lamrani, RWTH Aachen University, 2026).

## Overview

Given a question such as *"For any person, the diagnosis must be completed within 3 days after the
registration at ER"* and an OCEL 2.0 log, the pipeline

1. extracts the log's schema (event and object types, attributes, E2O/O2O relations),
2. retrieves similar examples from a synthetic corpus of 20,000 question/query pairs,
3. prompts the model with one of three strategies (`schema-allowed-entities-only`,
   `schema-grammar-guided`, `schema-chain-of-thought-plus`), optionally with the output
   constrained to the OCPQ JSON schema,
4. validates the query and, if enabled, lets the model repair schema or execution errors once,
5. executes the query with OCPQ's `ocpq_cli` and returns the result.

A second, agentic arm gives the model tools to inspect the log and try queries against a running
OCPQ server before it answers.

What was evaluated:

- **Models:** Llama 3.1 8B, Ministral 3 3B, Ministral 3 14B and Gemma 3 4B, each as the base model
  and with a LoRA adapter fine-tuned on the synthetic corpus, all run in 4-bit. The agentic arm
  adds larger open-weight models (see `configs/eval_config.agentic.json`).
- **Testset:** 169 hand-written questions with gold OCPQ queries over 11 public OCEL logs
  (`data/testset.json`). 135 of them are *discriminating*: a do-nothing query can't score
  perfectly on them.
- **Metrics:** the primary metric is `result_set_f1_strict`, the F1 between the predicted and
  gold result sets of the root binding box, with unparsable or failed queries scoring 0. It is
  reported next to `structural_f1` (did the query use the right constructs?),
  `execution_success_rate`, and a `skill_score` that rescales F1 against the best trivial
  strategy for each question.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | All Python, flat (modules import each other by bare name). Entry points: `cli.py`, `ui.py`, `eval_testset.py`, `train_unsloth.py` |
| `configs/` | Evaluation configs: the main matrix, the agentic arm and every ablation |
| `data/` | Testset, synthetic corpus (`ocpq_dataset.json`), its embeddings, the fine-tuning set and the log manifest |
| `models/adapters/` | The four thesis LoRA adapters (Git LFS) |
| `eval_runs/analysis/` | Analysis JSON behind every table in the thesis |
| `slurm/` | Batch scripts for training and evaluation on the RWTH cluster |
| `scripts/` | Environment setup and model prefetching for the cluster |
| `requirements/` | Pinned dependencies, one file per environment |
| `tests/` | Assertion scripts, run one by one with `python tests/test_<name>.py` |
| `docs/ocpq-nlxocpq-pinned.patch` | The two local changes to OCPQ the evaluation depends on |
| `notebooks/` | Exploration only |

`models/lora_model_ocpq/` is an early Llama 3 8B adapter that predates the thesis models. It is
kept for provenance only.

## Setup

### 1. Clone with Git LFS

The adapters and the larger data files are stored in Git LFS:

```bash
git lfs install
git clone https://github.com/framepixel/NLxOCPQ.git
```

### 2. Directory layout

OCPQ and the event logs live next to this repository, not inside it:

```
parent/
├── NLxOCPQ/   this repository
├── OCPQ/      OCPQ checkout (step 3)
└── Logs/      the OCEL logs (step 4)
```

Set `OCPQ_REPO_DIR` and `OCPQ_LOGS_DIR` to use other locations.

### 3. OCPQ

Install Rust with [rustup](https://rustup.rs/). The evaluation uses OCPQ at commit `2c5e0d6`
with two local changes. Later versions replace the `/ocel/*` HTTP API that the agentic arm talks
to.

```bash
git clone https://github.com/aarkue/OCPQ.git
cd OCPQ
git checkout 2c5e0d631bd10a8ecea1cea3c3d8703d5942b4a3
git am ../NLxOCPQ/docs/ocpq-nlxocpq-pinned.patch
```

`src/QueryExecutor.py` runs `ocpq_cli` through `cargo run --release`, so the first query also
compiles it. The agentic arm builds OCPQ's web server on first use, which takes several minutes.

### 4. Event logs

All 11 logs are public:

| Log file | Source |
|---|---|
| `03_hiring.xml`, `04_hospital.xml` | [10.5281/zenodo.13879980](https://doi.org/10.5281/zenodo.13879980) |
| `hospital_log_high.json`, `lending_log_high.json`, `renting_log_high.json` | [10.5281/zenodo.18333683](https://doi.org/10.5281/zenodo.18333683) |
| `container_logistics.json` | [10.5281/zenodo.18373888](https://doi.org/10.5281/zenodo.18373888) |
| `order-management.json` | [10.5281/zenodo.18373906](https://doi.org/10.5281/zenodo.18373906) |
| `ocel2-p2p.json` | [10.5281/zenodo.8412920](https://doi.org/10.5281/zenodo.8412920) |
| `reasoning_benchmark.json` | [10.5281/zenodo.18333784](https://doi.org/10.5281/zenodo.18333784) |
| `bpic2017-ocel2.json` | [aarkue/ocpq-eval](https://github.com/aarkue/ocpq-eval/blob/main/bpic2017-ocel2.zip), converted from the [BPI Challenge 2017](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) |
| `angular_github_commits_ocel.xml` | [10.5281/zenodo.8430332](https://doi.org/10.5281/zenodo.8430332) |

`data/corpus_manifest.json` pins every log by size and SHA-256, and an evaluation refuses to
start if a log on disk differs. Three logs were modified for the thesis: `renting_log_high`,
`ocel2-p2p` and `container_logistics`. Some string attributes were retyped to integer or
boolean, and `DepartureDate` in `container_logistics` became a timestamp, so that OCPQ can filter
on them. The modified copies are not part of this repository, so the original downloads of those
three logs fail the check. Setting `"corpus_check": "warn"` in the config lets a run continue,
but scores on those logs are then not comparable to the thesis.

### 5. Python environments

| Environment | Requirements | Used for |
|---|---|---|
| Base | `requirements/requirements.txt` | `cli.py`, `ui.py`, analysis scripts |
| Training | `requirements/requirements.unsloth-train-v5.txt` (conda, Python 3.10, CUDA 12.6) | `train_unsloth.py`, `merge_gemma.py` |
| Evaluation | `scripts/make_eval_env.sh` clones the training env and adds `scikit-learn`, `openai` and `xgrammar` | `eval_testset.py` |

Evaluation loads the adapters with the same torch/transformers/peft build that trained them,
which is why its environment is a clone of the training one rather than a fresh install.

## Usage

All commands run from the repository root.

### Single query

```bash
python src/cli.py \
    --backend local-gpu \
    --model unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --ocel ../Logs/hospital_log_high.json \
    --prompt-strategy schema-grammar-guided \
    "For any person, the diagnosis must be completed within 3 days after the registration at ER."
```

- `--model` also takes an adapter directory such as
  `models/adapters/llama31-8b-fine-tuned-quantized`. The base model is read from its
  `adapter_config.json`.
- `--backend hf-provider` runs the model through Hugging Face Inference Providers instead
  (`--hf-provider`, and `--hf-token` or `HF_TOKEN`).

Each run writes a timestamped directory under `results/` with the extracted schema, the generated
query, the `ocpq_cli` output, the query result and a `run_summary.json`.

### Web UI

```bash
streamlit run src/ui.py
```

The UI sends requests to Hugging Face Inference Providers or to any OpenAI-compatible
chat-completions endpoint (set in the sidebar or through `HF_BASE_URL`).

### Evaluation

```bash
python src/eval_testset.py --config configs/eval_config.json --dry-run   # planned run counts
python src/eval_testset.py --config configs/eval_config.json
```

A run writes `eval_runs/<run_id>/` with `config_resolved.json`, `predictions.json`,
`summary.json` (with per-stratum metrics under `by_stratum`), `result_store/` and error-analysis
reports (`bb_breakdown.json`, `divergence_report.json`, `failure_log.jsonl`).

| Config | Experiment |
|---|---|
| `eval_config.json`, `eval_config.matrix_v2.json` | Main matrix: 8 models × 3 strategies × 2 schema settings |
| `eval_config.agentic.json`, `eval_config.agentic_gate.json` | Agentic arm and its model gate |
| `eval_config.non_oracle.json` | Pipeline rungs without gold-derived (oracle) hints |
| `eval_config.unleashed*.json` | Agentic evaluation 2: each model with its own native reasoning |
| `eval_config.{decoding,arity,structural,structure,typed}_ablation.json`, `eval_config.selfconsistency.json` | Ablations |
| `eval_config.smoke*.json` | Short smoke runs |

The full matrix takes tens of GPU hours, so on the cluster it runs as a SLURM array of shards that
are merged afterwards:

```bash
sbatch slurm/submit_eval_smoke.sh            # all conditions on a 4-case slice; run this first
sbatch slurm/submit_eval_array.sh            # 96 shards = 48 conditions × 2 case halves
python src/eval_testset.py --merge-shards eval_runs/shard_* --merge-output eval_runs/merged
```

`--recompute <run_dir>` rescores a finished run from its result store into a new directory,
without a GPU or OCPQ. `--resume-from <run_dir>` continues an interrupted run. For dead shards,
resubmit the array script with `--resume`, e.g.
`sbatch --array=48-71%24 slurm/submit_eval_array.sh --resume`.

The SLURM scripts target the RWTH CLAIX cluster. Their `--partition` and `--account` lines need
changing for any other cluster.

### Fine-tuning

`src/train_unsloth.py` trains a LoRA adapter with Unsloth on `data/finetune_ocpq.jsonl` and is
configured through environment variables, e.g. `OCPQ_QUANTIZED_MODEL`, `OCPQ_CHAT_TEMPLATE`,
`OCPQ_MAX_SEQ_LENGTH`, `OCPQ_NUM_TRAIN_EPOCHS`, `OCPQ_LEARNING_RATE` and
`OCPQ_ADAPTER_OUTPUT_DIR`. The thesis adapters come from one script per model:

```bash
sbatch slurm/submit_job_v5_llama31_8b.sh
sbatch slurm/submit_job_v5_ministral3_3b.sh
sbatch slurm/submit_job_v5_ministral3_14b.sh
sbatch slurm/submit_job_v5_gemma3_4b.sh
```

Keep `OCPQ_MAX_SEQ_LENGTH` at 4096 or more. The trainer truncates the end of each example, which
is the JSON query the model is supposed to learn.

Gemma 3 is evaluated from merged 16-bit checkpoints rather than base model plus adapter. Its
adapter was trained against Unsloth's pre-quantized checkpoint, which plain transformers cannot
load. `python src/merge_gemma.py` (in the training environment) writes both Gemma arms to
`models/adapters/gemma3-4b-base-16bit/` and `models/adapters/gemma3-4b-merged-16bit/`. These are
not in the repository.

`data/finetune_ocpq.jsonl` is generated from `data/ocpq_dataset.json` by
`src/convert_dataset_for_finetuning.py` and should not be edited by hand.

### Thesis tables

The analysis scripts in `src/` (`main_tables.py`, `matrix_tables.py`, `paired_stats.py`,
`rung_tables.py`, `ablation_tables.py`, `agentic_eval1_tables.py`, ...) read finished runs and
write JSON to `eval_runs/analysis/` and LaTeX tables to `$THESIS_TABLES_DIR`. The run directories
are not committed because of their size, but the analysis JSON is, and most of those files record
the commit they were produced from.

## License

The code is released under the [MIT License](LICENSE). The adapters in `models/` are derived from
their base models and remain subject to those models' licences (Llama 3.1 Community License,
Gemma Terms of Use, and the Ministral 3 licence). The event logs are not included and are
distributed under their own terms.

## Citation

```bibtex
@mastersthesis{lamrani2026nlxocpq,
  author = {Ayoub Lamrani},
  title  = {Benchmarking Large Language Models for Natural-Language Object-Centric Process Querying},
  school = {RWTH Aachen University},
  year   = {2026}
}
```
