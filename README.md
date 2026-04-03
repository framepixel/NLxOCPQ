# NLxOCPQ
Natural language to OCPQ queries converter

## Automatic Execution
The `LLM.py` script utilizes `QueryExecutor.py` to automatically execute the generated queries right after generation using the Rust backend `ocpq_cli`. Output noise from `cargo` is suppressed to only show you the relevant query results (via the `-q` parameter).

### Dependencies & Required Repositories
To utilize automatic execution successfully, note the following dependencies:

1. **Rust & Cargo**: You must have Rust installed. Set it up using [rustup](https://rustup.rs/). The script currently expects `cargo` to be in your PATH or at `~/.cargo/bin/cargo`.
2. **OCPQ CLI Repository**: `OCPQ/backend/ocpq_cli` must exist in the root of the project (i.e. one directory level up from this `NLxOCPQ` directory) and be building correctly.
3. **Event Log**: By default, `LLM.py` assumes there is an object-centric event log named `hospital_log_high.json` at the parent directory (`../hospital_log_high.json`).
