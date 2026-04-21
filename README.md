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
