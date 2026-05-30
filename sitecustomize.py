import os

if os.getenv("OCPQ_STABILIZE_RUNTIME", "0").strip().lower() in {"1", "true", "yes", "on"}:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    os.environ.setdefault("TQDM_MONITOR_INTERVAL", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import tqdm

        tqdm.tqdm.monitor_interval = 0
    except Exception:
        pass
