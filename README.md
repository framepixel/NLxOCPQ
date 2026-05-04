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
