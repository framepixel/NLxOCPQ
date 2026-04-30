# Colab Server Setup

This repository now includes a lightweight inference server for running the fine-tuned LoRA adapter on a Colab GPU and exposing it through an HTTPS tunnel.

## What it does

- Loads the base model used for training.
- Attaches the saved adapter in `lora_model_ocpq/`.
- Serves an OpenAI-compatible `POST /v1/chat/completions` endpoint.
- Exposes `GET /health` for quick checks.

The Streamlit UI already supports a custom endpoint through `HF_BASE_URL`, so no UI rewrite is required once the Colab endpoint is reachable.

## Colab steps

1. Open a Colab notebook with a GPU runtime.
2. Mount your Drive or upload the `NLxOCPQ` folder so the adapter directory is available.
3. Install the minimal server dependencies:

```bash
pip install -r requirements.colab-serve.txt
```

4. Start the server:

```bash
python colab_serve.py --host 0.0.0.0 --port 8000 --adapter-path /content/NLxOCPQ/lora_model_ocpq
```

5. Publish port 8000 with a tunnel such as `cloudflared` or `ngrok`.
6. Copy the public HTTPS URL into the Streamlit sidebar as the custom endpoint base URL.

## Environment variables

- `COLAB_BASE_MODEL` - overrides the base model name.
- `COLAB_ADAPTER_PATH` - path to the LoRA adapter directory.
- `COLAB_LOAD_IN_4BIT` - set to `0` to disable 4-bit loading.
- `HF_TOKEN` - optional token for gated base-model downloads in Colab.

If you prefer notebook cells over command-line flags, set the paths once at the top of the notebook and reuse them:

```python
BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
ADAPTER_PATH = "/content/NLxOCPQ/lora_model_ocpq"
```

Then start the server with:

```bash
python colab_serve.py --host 0.0.0.0 --port 8000 --base-model "$BASE_MODEL" --adapter-path "$ADAPTER_PATH"
```

## Notes

- The server is intentionally single-node and lightweight.
- Streaming is not enabled.
- The endpoint accepts `response_format` and converts JSON schema requests into a stronger instruction, but it does not enforce schema output at the transport layer.
