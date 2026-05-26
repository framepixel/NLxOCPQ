# NLxOCPQ
Natural language to OCPQ queries converter

## Automatic Execution
The `LLM.py` script utilizes `QueryExecutor.py` to automatically execute the generated queries right after generation using the Rust backend `ocpq_cli`. Output noise from `cargo` is suppressed to only show you the relevant query results (via the `-q` parameter).

### Dependencies & Required Repositories
To utilize automatic execution successfully, note the following dependencies:

1. **Rust & Cargo**: You must have Rust installed. Set it up using [rustup](https://rustup.rs/). The script currently expects `cargo` to be in your PATH or at `~/.cargo/bin/cargo`.
2. **OCPQ CLI Repository**: `OCPQ/backend/ocpq_cli` must exist in the root of the project (i.e. one directory level up from this `NLxOCPQ` directory) and be building correctly.
3. **Event Log**: By default, `LLM.py` assumes there is an object-centric event log named `hospital_log_high.json` at the parent directory (`../hospital_log_high.json`).

### Colab GPU Serving
If your local machine has no GPU, you can run the fine-tuned LoRA model in Google Colab and point the Streamlit UI at it through the existing custom-endpoint mode.

The repository now includes:

- [`colab_serve.py`](colab_serve.py) for serving the adapter as an OpenAI-compatible chat endpoint.
- [`requirements.colab-serve.txt`](requirements.colab-serve.txt) for the minimal Colab inference stack.
- [`COLAB_SERVER.md`](COLAB_SERVER.md) with the launch steps.

The UI accepts a custom `HF_BASE_URL`, and custom-endpoint mode no longer requires a Hugging Face token if your Colab server is open or uses its own auth.

## Console Runner
You can run the full NL-to-OCPQ pipeline without Streamlit using `cli.py`.

Each invocation creates a timestamped run directory in `results/` and writes multiple artifacts:

- `query_input.json`
- `extracted_structure.json`
- `planned_query_tree.json`
- `generated_ocpq.json`
- `query_executor_stdout.txt`
- `query_executor_stderr.txt`
- `query_result.json` (if the backend exported one)
- `run_summary.json`

### Backend Selection
- `--backend local-gpu`: run the selected model locally on CUDA with `transformers`
- `--backend hf-provider`: run via Hugging Face Inference Providers

### Examples

Run with local GPU:

```bash
python cli.py \
	--backend local-gpu \
	--model meta-llama/Meta-Llama-3-8B-Instruct \
	--ocel ../hospital_log_high.json \
	"For any person, the diagnosis must be completed within 3 days after the registration at ER."
```

Run with HF inference provider:

```bash
python cli.py \
	--backend hf-provider \
	--model meta-llama/Meta-Llama-3-8B-Instruct \
	--hf-provider together \
	--hf-token "$HF_TOKEN" \
	--ocel ../hospital_log_high.json \
	"For any person, the diagnosis must be completed within 3 days after the registration at ER."
```

## Testset Evaluation
Use `eval_testset.py` to evaluate the handcrafted `testset.json` across models, prompting strategies, and schema settings.

1. Copy the template config and edit it:

```bash
cp eval_config.example.json eval_config.json
```

2. Run the evaluator:

```bash
python eval_testset.py --config eval_config.json
```

Outputs are stored under `eval_runs/` and include:

- `predictions.json` (per-run records)
- `summary.json` (aggregated metrics + consistency)
- `config_resolved.json` (run manifest)

## Fine-tuning an LLM with train_unsloth.py

You can fine-tune a base LLM using `train_unsloth.py`. The script auto-detects available CUDA resources and will choose between a quantized (4-bit) or full-precision training setup by default. You can override behavior with environment variables.

Quick checklist
- Ensure CUDA is available and appropriate drivers are installed.
- Install training dependencies (see `requirements.unsloth-train.txt` or install `unsloth`, `trl`, `datasets`, `peft`, `transformers`).
- Prepare a fine-tuning JSONL dataset (the repo uses `finetune_ocpq.jsonl`).

Important environment variables
- `OCPQ_FINETUNE_DATASET`: path to the JSONL dataset (default: `finetune_ocpq.jsonl`).
- `OCPQ_ADAPTER_OUTPUT_DIR`: directory to write the LoRA adapter (default: `adapters/<run_label>`).
- `OCPQ_FORCE_4BIT`: if set (`1`, `true`) force the quantized 4-bit model path.
- `OCPQ_FORCE_FULL_PRECISION`: if set (`1`, `true`) force full-precision loading.
- `OCPQ_QUANTIZED_MODEL`: HF/local id for the quantized model (default: `unsloth/llama-3-8b-Instruct-bnb-4bit`).
- `OCPQ_FULL_PRECISION_MODEL`: HF/local id for the full-precision model (default: `meta-llama/Meta-Llama-3-8B-Instruct`).
- `OCPQ_MIN_FULL_PRECISION_GB`: free VRAM threshold (GB) above which full-precision is chosen by default (default: `32`).

How the script decides and labels outputs
- The script probes CUDA and available free VRAM; if free VRAM is below `OCPQ_MIN_FULL_PRECISION_GB` it will select the quantized model path.
- The saved adapter folder and a printed `Training label` include a descriptive name: `<model>-fine-tuned-quantized` or `<model>-fine-tuned-full-precision` (slugified). This makes it easy to reference the adapter in evaluation configs.

Example commands
- Auto-detect mode (recommended):
```bash
OCPQ_FINETUNE_DATASET=finetune_ocpq.jsonl python train_unsloth.py
```
- Force quantized 4-bit training:
```bash
OCPQ_FORCE_4BIT=1 OCPQ_FINETUNE_DATASET=finetune_ocpq.jsonl python train_unsloth.py
```
- Force full-precision training and set custom model id:
```bash
OCPQ_FORCE_FULL_PRECISION=1 OCPQ_FULL_PRECISION_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
	OCPQ_FINETUNE_DATASET=finetune_ocpq.jsonl python train_unsloth.py
```

Using the adapter in evaluation
- After training, point your evaluation config to the adapter directory. There are two options:
	- Use the adapter directory as a local model name in `eval_config.json` (set `models[].name` to the adapter path). The evaluator's backend will detect a local adapter directory and load it.
	- Or use the `fine_tune_map` in `eval_config.json` to map the base model to the adapter (see `eval_config.example.json`). Example mapping entry:
```json
"fine_tune_map": {
	"meta-llama/Meta-Llama-3-8B-Instruct": {
		"name": "adapters/meta-llama-3-8b-instruct-fine-tuned-quantized",
		"label": "meta-llama-3-8b-instruct-fine-tuned-quantized"
	}
}
```

Notes
- The script prints `CUDA resources detected:` and `Training label:` so you can confirm which path and label were chosen.
- If you plan to run multiple experiments, keep `OCPQ_ADAPTER_OUTPUT_DIR` or the run label descriptive so the evaluation harness can automatically expand and reference the correct adapter.
