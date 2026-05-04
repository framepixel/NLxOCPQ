#!/usr/bin/env python3
"""Serve Ollama models via an OpenAI-compatible chat endpoint.

This keeps the Streamlit UI working with a local Ollama backend.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

try:
    from ocpq_schema import BindingBoxTree
except Exception:
    BindingBoxTree = None

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "ocpq-v0"
DEFAULT_TIMEOUT_SEC = 120


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


app = FastAPI(title="Ollama Local Server")


def _ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL)


def _ollama_default_model() -> str:
    return os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _ollama_timeout() -> float:
    value = os.getenv("OLLAMA_TIMEOUT", str(DEFAULT_TIMEOUT_SEC))
    try:
        return float(value)
    except ValueError:
        return float(DEFAULT_TIMEOUT_SEC)


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
    if role in {"system", "user", "assistant"}:
        return role
    return "user"


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


def _apply_response_format(
    messages: list[dict[str, str]], response_format: dict[str, Any] | None
) -> list[dict[str, str]]:
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


def _ollama_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop: str | list[str] | None,
) -> tuple[str, int, int]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "num_predict": max_new_tokens,
    }
    if stop:
        options["stop"] = [stop] if isinstance(stop, str) else list(stop)

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }

    timeout = _ollama_timeout()
    url = f"{base_url.rstrip('/')}/api/chat"
    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    message = data.get("message") or {}
    content = message.get("content") or ""
    prompt_tokens = int(data.get("prompt_eval_count") or 0)
    completion_tokens = int(data.get("eval_count") or 0)
    return str(content).strip(), prompt_tokens, completion_tokens


def _list_ollama_models(base_url: str) -> list[str]:
    url = f"{base_url.rstrip('/')}/api/tags"
    timeout = _ollama_timeout()
    with httpx.Client(timeout=timeout) as client:
        response = client.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    models = data.get("models", [])
    names = []
    for item in models:
        name = item.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


@app.get("/health")
def health() -> dict[str, Any]:
    base_url = _ollama_base_url()
    models = _list_ollama_models(base_url)
    return {
        "status": "ok",
        "ollama_base_url": base_url,
        "models": models,
    }


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    try:
        if request.stream:
            raise HTTPException(status_code=400, detail="Streaming is not supported.")

        model_name = request.model or _ollama_default_model()

        messages = [
            {"role": _normalize_role(message.role), "content": _normalize_content(message.content)}
            for message in request.messages
        ]
        messages = _apply_response_format(messages, request.response_format)

        max_new_tokens = request.max_tokens or 1024
        temperature = 0.0 if request.temperature is None else request.temperature
        top_p = 1.0 if request.top_p is None else request.top_p

        generated_text, prompt_tokens, completion_tokens = _ollama_chat(
            base_url=_ollama_base_url(),
            model=model_name,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=request.stop,
        )

        if _is_ocpq_schema_request(request.response_format):
            max_validation_retries = int(os.getenv("OLLAMA_OCPQ_SCHEMA_RETRIES", "2"))
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
                generated_text, prompt_tokens, completion_tokens = _ollama_chat(
                    base_url=_ollama_base_url(),
                    model=model_name,
                    messages=repair_messages,
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

            generated_text = normalized_text or generated_text

        created_at = int(time.time())
        total_tokens = int(prompt_tokens + completion_tokens)

        return ChatCompletionResponse(
            id=f"chatcmpl-{created_at}",
            created=created_at,
            model=model_name,
            choices=[ChatCompletionChoice(message=ChatCompletionChoiceMessage(content=generated_text))],
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(total_tokens),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {type(exc).__name__}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve Ollama via OpenAI-compatible API.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--ollama-base-url",
        default=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument(
        "--default-model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        help="Default Ollama model name to use when request.model is not provided.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=_ollama_timeout(),
        help="HTTP timeout in seconds for Ollama requests.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    os.environ["OLLAMA_BASE_URL"] = args.ollama_base_url
    os.environ["OLLAMA_MODEL"] = args.default_model
    os.environ["OLLAMA_TIMEOUT"] = str(args.timeout)

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
