#!/usr/bin/env python3
"""Serve the fine-tuned LoRA model from a Colab GPU as an OpenAI-compatible API.

This script is intentionally lightweight so it can run inside Google Colab with a
single GPU. It loads the base model, attaches the saved LoRA adapter directory,
and exposes:

- GET /health
- POST /v1/chat/completions

The Streamlit UI in this repo can point `HF_BASE_URL` to the tunneled HTTPS URL
and keep using `InferenceClient.chat_completion(...)` without changing the
prompt pipeline.
"""

from __future__ import annotations

import argparse
import re
import json
import os
import time
import traceback
from typing import Any, Literal

# Help avoid CUDA fragmentation by default in Colab runtimes.
# Allow user override via environment before starting the server.
os.environ.setdefault("PYTORCH_ALLOC_CONF", os.getenv("PYTORCH_ALLOC_CONF", "expandable_segments:True"))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from ocpq_schema import BindingBoxTree
except Exception:
    BindingBoxTree = None

DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
DEFAULT_ADAPTER_PATH = "lora_model_ocpq"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
PROMPT_TOKEN_MARGIN = 32


def _runtime_default(env_name: str, fallback: str) -> str:
    value = os.getenv(env_name)
    return value if value else fallback


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class JsonSchemaSpec(BaseModel):
    name: str | None = None
    description: str | None = None
    schema_: dict[str, Any] = Field(default_factory=dict, alias="schema")
    strict: bool | None = None


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = False
    response_format: dict[str, Any] | None = None


class ChatCompletionChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionChoiceMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict[str, int]


app = FastAPI(title="Colab LoRA Server")
model = None
tokenizer = None
model_name = None


def _extract_json_payload(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _read_adapter_base_model_name(adapter_path: str) -> str | None:
    adapter_config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_config_path):
        return None

    try:
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return None

    base_model_name = config.get("base_model_name_or_path")
    return base_model_name if isinstance(base_model_name, str) and base_model_name.strip() else None


def _looks_like_lora_adapter(path: str) -> bool:
    return os.path.exists(os.path.join(path, "adapter_config.json")) and os.path.exists(
        os.path.join(path, "adapter_model.safetensors")
    )


def _looks_like_full_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return False
    if not os.path.exists(os.path.join(path, "config.json")):
        return False
    return any(
        os.path.exists(os.path.join(path, candidate))
        for candidate in ("model.safetensors", "pytorch_model.bin", "model-00001-of-0000X.safetensors")
    ) or any(name.endswith(".safetensors") for name in os.listdir(path))


def _resolve_model_paths(base_model_name: str, adapter_path: str) -> tuple[str, str | None, str]:
    if _looks_like_full_model_dir(adapter_path):
        return adapter_path, None, "full_model_dir"

    if _looks_like_lora_adapter(adapter_path):
        adapter_base_model = _read_adapter_base_model_name(adapter_path)
        if base_model_name == adapter_path and adapter_base_model:
            return adapter_base_model, adapter_path, "adapter_with_base_from_metadata"
        if base_model_name == DEFAULT_BASE_MODEL and adapter_base_model:
            return adapter_base_model, adapter_path, "adapter_with_base_from_metadata"
        return base_model_name, adapter_path, "adapter_with_explicit_base"

    return base_model_name, adapter_path, "explicit_paths"


def _infer_context_limit() -> int:
    candidates: list[int] = []

    tokenizer_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_max_length, int) and 0 < tokenizer_max_length < 10_000_000:
        candidates.append(tokenizer_max_length)

    max_position_embeddings = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(max_position_embeddings, int) and max_position_embeddings > 0:
        candidates.append(max_position_embeddings)

    generation_max_length = getattr(getattr(model, "generation_config", None), "max_length", None)
    if isinstance(generation_max_length, int) and generation_max_length > 0:
        candidates.append(generation_max_length)

    env_limit = os.getenv("COLAB_MAX_CONTEXT_TOKENS")
    if env_limit:
        try:
            candidates.append(int(env_limit))
        except ValueError:
            pass

    return min(candidates) if candidates else 8192


def _validate_ocpq_payload(generated_text: str) -> tuple[bool, str | None, str | None]:
    if BindingBoxTree is None:
        return False, None, "BindingBoxTree could not be imported on server runtime."

    payload_text = _extract_json_payload(generated_text)
    try:
        parsed = json.loads(payload_text)
    except Exception as exc:
        return False, None, f"JSON parse failed: {type(exc).__name__}: {exc}"

    try:
        validated = BindingBoxTree.model_validate(parsed)
        normalized = validated.model_dump(by_alias=True)
        return True, json.dumps(normalized, ensure_ascii=True), None
    except ValidationError as exc:
        return False, None, f"BindingBoxTree validation failed: {exc}"


def _is_ocpq_schema_request(response_format: dict[str, Any] | None) -> bool:
    if not isinstance(response_format, dict):
        return False
    if response_format.get("type") != "json_schema":
        return False
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return False
    return (json_schema.get("name") or "").strip() == "ocpq_schema"


def _generate_assistant_text(
    prepared_messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | None,
) -> tuple[str, int, int]:
    do_sample = temperature > 0.0

    chat_inputs = tokenizer.apply_chat_template(
        prepared_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    chat_input_ids = None
    attention_mask = None

    if torch.is_tensor(chat_inputs):
        chat_input_ids = chat_inputs.to(model.device)
    else:
        if hasattr(chat_inputs, "get"):
            chat_input_ids = chat_inputs.get("input_ids")
            attention_mask = chat_inputs.get("attention_mask")
        if chat_input_ids is None and hasattr(chat_inputs, "input_ids"):
            chat_input_ids = getattr(chat_inputs, "input_ids")
        if attention_mask is None and hasattr(chat_inputs, "attention_mask"):
            attention_mask = getattr(chat_inputs, "attention_mask")

        if chat_input_ids is None:
            raise HTTPException(status_code=500, detail="Tokenizer output missing input_ids.")

        chat_input_ids = chat_input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

    if not torch.is_tensor(chat_input_ids):
        raise HTTPException(status_code=500, detail="Tokenizer input_ids are not a tensor.")

    context_limit = _infer_context_limit()
    max_prompt_tokens = max(256, context_limit - max_new_tokens - PROMPT_TOKEN_MARGIN)
    if chat_input_ids.size(-1) > max_prompt_tokens:
        chat_input_ids = chat_input_ids[:, -max_prompt_tokens:]
        if attention_mask is not None:
            attention_mask = attention_mask[:, -max_prompt_tokens:]

    input_length = int(chat_input_ids.size(-1))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    generate_inputs: dict[str, Any] = {
        "input_ids": chat_input_ids,
        **{key: value for key, value in generation_kwargs.items() if value is not None},
    }
    if attention_mask is not None:
        generate_inputs["attention_mask"] = attention_mask

    # Try generation, and on OOM attempt a small number of retries with reduced max_new_tokens.
    attempts = 0
    max_attempts = 2
    current_max_new_tokens = max_new_tokens
    output_ids = None

    while True:
        try:
            # ensure the generation args reflect the current_max_new_tokens
            generate_inputs["max_new_tokens"] = current_max_new_tokens
            with torch.inference_mode():
                output_ids = model.generate(**generate_inputs)
            break
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "out of memory" in msg or "outofmemory" in msg:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

                attempts += 1
                # If we can reduce token budget, try again with half the tokens.
                if attempts <= max_attempts and current_max_new_tokens > 64:
                    new_val = max(64, current_max_new_tokens // 2)
                    if new_val < current_max_new_tokens:
                        current_max_new_tokens = new_val
                        continue

                # Give a helpful error if retries exhausted.
                raise HTTPException(
                    status_code=500,
                    detail=(
                        "Internal model server error: OutOfMemoryError: CUDA ran out of memory during generation. "
                        "I attempted automatic retries by reducing `max_new_tokens` but the GPU still ran out of memory. "
                        "Mitigations: restart the server with a smaller base model, enable 4-bit loading (set COLAB_LOAD_IN_4BIT=1), "
                        "reduce the request `max_tokens` from the client, or set the environment variable `PYTORCH_ALLOC_CONF=expandable_segments:True` before starting the server. "
                        f"Original error: {exc}"
                    ),
                ) from exc
            raise

    generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
    if stop:
        stop_sequences = [stop] if isinstance(stop, str) else list(stop)
        cutoff_positions = [generated_text.find(stop_sequence) for stop_sequence in stop_sequences if stop_sequence]
        cutoff_positions = [position for position in cutoff_positions if position >= 0]
        if cutoff_positions:
            generated_text = generated_text[: min(cutoff_positions)].rstrip()

    prompt_tokens = int(input_length)
    completion_tokens = int(output_ids.shape[-1] - input_length)
    return generated_text, prompt_tokens, completion_tokens


def _maybe_add_auth_token_kwargs() -> dict[str, Any]:
    auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if auth_token:
        return {"token": auth_token}
    return {}


def _load_model(base_model_name: str, adapter_path: str, load_in_4bit: bool) -> tuple[Any, Any]:
    resolved_base_model_name, resolved_adapter_path, load_mode = _resolve_model_paths(base_model_name, adapter_path)
    tokenizer_source = resolved_adapter_path or resolved_base_model_name
    tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True, **_maybe_add_auth_token_kwargs())

    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token

    model_load_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        **_maybe_add_auth_token_kwargs(),
    }

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        model_load_kwargs["quantization_config"] = bnb_config

    print(f"Resolved model load mode: {load_mode}")
    print(f"Resolved base model: {resolved_base_model_name}")
    print(f"Resolved adapter path: {resolved_adapter_path or '(none)'}")
    print(f"Tokenizer source: {tokenizer_source}")
    print(f"4-bit loading enabled: {load_in_4bit}")

    base_model = AutoModelForCausalLM.from_pretrained(resolved_base_model_name, **model_load_kwargs)

    loaded_model = (
        PeftModel.from_pretrained(base_model, resolved_adapter_path)
        if resolved_adapter_path is not None
        else base_model
    )
    loaded_model.eval()

    generation_config = getattr(loaded_model, "generation_config", None)
    if generation_config is not None and getattr(generation_config, "max_length", None):
        generation_config.max_length = None

    return loaded_model, tokenizer_obj


def _normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    fragments.append(str(text))
            elif item is not None:
                fragments.append(str(item))
        return "\n".join(fragments)
    if content is None:
        return ""
    if isinstance(content, dict):
        text = content.get("text")
        if text is not None:
            return str(text)
        return json.dumps(content, ensure_ascii=True)
    return str(content)


def _normalize_role(role: str) -> str:
    # Llama chat templates typically support system/user/assistant.
    # Map unknown roles to user so tool messages do not crash generation.
    if role in {"system", "user", "assistant"}:
        return role
    return "user"


def _apply_response_format(messages: list[dict[str, str]], response_format: dict[str, Any] | None) -> list[dict[str, str]]:
    if not response_format:
        return messages

    format_type = response_format.get("type")
    if format_type == "json_object":
        return [{"role": "system", "content": "Return only valid JSON."}, *messages]

    if format_type == "json_schema":
        schema_spec = JsonSchemaSpec.model_validate(response_format.get("json_schema", {}))
        schema_name = schema_spec.name or "response"
        schema_text = json.dumps(schema_spec.schema_, indent=2, ensure_ascii=True)
        instruction = (
            f"Return only valid JSON matching the schema '{schema_name}'.\n"
            f"Schema:\n{schema_text}"
        )
        return [{"role": "system", "content": instruction}, *messages]

    return messages


@app.on_event("startup")
def _startup() -> None:
    global model, tokenizer, model_name

    base_model_name = os.getenv("COLAB_BASE_MODEL", DEFAULT_BASE_MODEL)
    adapter_path = os.getenv("COLAB_ADAPTER_PATH", DEFAULT_ADAPTER_PATH)
    load_in_4bit = os.getenv("COLAB_LOAD_IN_4BIT", "1").strip() not in {"0", "false", "False"}

    model_name = base_model_name
    print(f"Loading base model: {base_model_name}")
    print(f"Loading LoRA adapter: {adapter_path}")
    model, tokenizer = _load_model(base_model_name, adapter_path, load_in_4bit)
    print("Model ready.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": model_name or "unloaded"}


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        if model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported in this lightweight server.")

        messages = [
            {"role": _normalize_role(message.role), "content": _normalize_content(message.content)}
            for message in request.messages
        ]
        messages = _apply_response_format(messages, request.response_format)

        max_new_tokens = request.max_tokens or 1024
        temperature = 0.0 if request.temperature is None else request.temperature
        top_p = 1.0 if request.top_p is None else request.top_p

        generated_text, prompt_tokens, completion_tokens = _generate_assistant_text(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=request.stop,
        )

        if _is_ocpq_schema_request(request.response_format):
            max_validation_retries = int(os.getenv("COLAB_OCPQ_SCHEMA_RETRIES", "2"))
            is_valid, normalized_text, validation_error = _validate_ocpq_payload(generated_text)

            for _ in range(max_validation_retries):
                if is_valid:
                    break
                repair_messages = [
                    *messages,
                    {"role": "assistant", "content": generated_text},
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not satisfy the required OCPQ schema. "
                            f"Error: {validation_error}\n"
                            "Return ONLY valid JSON matching the schema. No markdown fences."
                        ),
                    },
                ]
                generated_text, prompt_tokens, completion_tokens = _generate_assistant_text(
                    repair_messages,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    stop=request.stop,
                )
                is_valid, normalized_text, validation_error = _validate_ocpq_payload(generated_text)

            if not is_valid:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Model output could not be validated against OCPQ schema after retries. "
                        f"Last error: {validation_error}"
                    ),
                )

            # Enforce transport-level schema by returning normalized validated JSON only.
            generated_text = normalized_text or generated_text

        created_at = int(time.time())

        total_tokens = prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            id=f"chatcmpl-{created_at}",
            created=created_at,
            model=request.model or model_name or DEFAULT_BASE_MODEL,
            choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content=generated_text))],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal model server error: {type(exc).__name__}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the fine-tuned OCPQ LoRA model.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--base-model",
        default=_runtime_default("COLAB_BASE_MODEL", DEFAULT_BASE_MODEL),
        help="Base model repo/path to load before applying the adapter.",
    )
    parser.add_argument(
        "--adapter-path",
        default=_runtime_default("COLAB_ADAPTER_PATH", DEFAULT_ADAPTER_PATH),
        help="Local or mounted path to the LoRA adapter directory.",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.environ["COLAB_BASE_MODEL"] = args.base_model
    os.environ["COLAB_ADAPTER_PATH"] = args.adapter_path
    os.environ["COLAB_LOAD_IN_4BIT"] = "1" if args.load_in_4bit else "0"
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
