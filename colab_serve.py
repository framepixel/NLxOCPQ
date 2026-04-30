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

    input_length = int(chat_input_ids.size(-1))
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    generate_inputs: dict[str, Any] = {
        "input_ids": chat_input_ids,
        **{key: value for key, value in generation_kwargs.items() if value is not None},
    }
    if attention_mask is not None:
        generate_inputs["attention_mask"] = attention_mask

    with torch.inference_mode():
        output_ids = model.generate(**generate_inputs)

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
