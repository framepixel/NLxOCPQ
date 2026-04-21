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
import json
import os
import time
import traceback
from typing import Any, Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE_MODEL = "unsloth/llama-3-8b-Instruct-bnb-4bit"
DEFAULT_ADAPTER_PATH = "lora_model_ocpq"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Any


class JsonSchemaSpec(BaseModel):
    name: str | None = None
    description: str | None = None
    schema: dict[str, Any] = Field(default_factory=dict)
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


def _maybe_add_auth_token_kwargs() -> dict[str, Any]:
    auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if auth_token:
        return {"token": auth_token}
    return {}


def _load_model(base_model_name: str, adapter_path: str, load_in_4bit: bool) -> tuple[Any, Any]:
    tokenizer_obj = AutoTokenizer.from_pretrained(adapter_path, use_fast=True, **_maybe_add_auth_token_kwargs())

    if tokenizer_obj.pad_token is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **_maybe_add_auth_token_kwargs(),
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **_maybe_add_auth_token_kwargs(),
        )

    merged_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model.eval()
    return merged_model, tokenizer_obj


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
        schema_text = json.dumps(schema_spec.schema, indent=2, ensure_ascii=True)
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
        do_sample = temperature > 0.0

        chat_input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        chat_input_ids = chat_input_ids.to(model.device)
        input_length = chat_input_ids.shape[-1]

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "top_p": top_p if do_sample else None,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        with torch.inference_mode():
            output_ids = model.generate(chat_input_ids, **{key: value for key, value in generation_kwargs.items() if value is not None})

        generated_text = tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True).strip()
        if request.stop:
            stop_sequences = [request.stop] if isinstance(request.stop, str) else list(request.stop)
            cutoff_positions = [generated_text.find(stop_sequence) for stop_sequence in stop_sequences if stop_sequence]
            cutoff_positions = [position for position in cutoff_positions if position >= 0]
            if cutoff_positions:
                generated_text = generated_text[: min(cutoff_positions)].rstrip()
        created_at = int(time.time())

        prompt_tokens = int(input_length)
        completion_tokens = int(output_ids.shape[-1] - input_length)
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
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    os.environ.setdefault("COLAB_BASE_MODEL", args.base_model)
    os.environ.setdefault("COLAB_ADAPTER_PATH", args.adapter_path)
    os.environ.setdefault("COLAB_LOAD_IN_4BIT", "1" if args.load_in_4bit else "0")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
