from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("openrouter-proxy")

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

app = FastAPI(title="Ollama-Compatible OpenRouter Proxy")


class OllamaChatRequest(BaseModel):
    model: str | None = None
    stream: bool | None = None
    format: str | None = None
    options: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _openrouter_headers() -> dict[str, str]:
    api_key = _env("OPENROUTER_API_KEY")
    logger.info("OPENROUTER_API_KEY present=%s length=%s", bool(api_key), len(api_key) if api_key else 0)
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY no está configurada")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    referer = _env("OPENROUTER_HTTP_REFERER")
    title = _env("OPENROUTER_X_TITLE")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers


def _openrouter_payload(req: OllamaChatRequest) -> dict[str, Any]:
    model = _env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    payload: dict[str, Any] = {
        "model": model,
        "messages": req.messages,
    }

    # Map temperature if provided by Ollama client
    if req.options and "temperature" in req.options:
        payload["temperature"] = req.options["temperature"]

    # Optional: enforce JSON responses if requested + enabled
    force_json = _env("OPENROUTER_FORCE_JSON", "0").lower() in {"1", "true", "yes"}

    if req.format == "json" and force_json:
        payload["response_format"] = {"type": "json_object"}

    # Optional: provider routing controls
    def _csv_env(name: str) -> list[str]:
        raw = _env(name)
        return [s.strip() for s in raw.split(",") if s.strip()] if raw else []

    provider: dict[str, Any] = {}
    only = _csv_env("OPENROUTER_PROVIDER_ONLY")
    order = _csv_env("OPENROUTER_PROVIDER_ORDER")
    ignore = _csv_env("OPENROUTER_PROVIDER_IGNORE")
    quantizations = _csv_env("OPENROUTER_QUANTIZATIONS")
    allow_fallbacks = _env("OPENROUTER_ALLOW_FALLBACKS")

    if only:
        provider["only"] = only
    if order:
        provider["order"] = order
    if ignore:
        provider["ignore"] = ignore
    if quantizations:
        provider["quantizations"] = quantizations
    if allow_fallbacks:
        provider["allow_fallbacks"] = allow_fallbacks.lower() in {"1", "true", "yes"}

    if provider:
        payload["provider"] = provider

    return payload


def _log_openrouter_response(status_code: int, body: str) -> None:
    enabled = _env("OPENROUTER_LOG_RAW", "1").lower() in {"1", "true", "yes"}
    if not enabled:
        return
    snippet = (body or "")[:2000]
    logger.info("OpenRouter response status=%s body(trunc)=%s", status_code, snippet)


@app.get("/api/tags")
async def tags() -> dict[str, Any]:
    model = _env("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    return {
        "models": [
            {
                "name": model,
                "model": model,
                "modified_at": "",
                "size": 0,
                "digest": "",
                "details": {},
            }
        ]
    }


@app.post("/api/chat")
async def chat(req: OllamaChatRequest) -> dict[str, Any]:
    base_url = _env("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    timeout_s = float(_env("OPENROUTER_TIMEOUT_S", "120"))

    headers = _openrouter_headers()
    print(headers, "holi2")
    payload = _openrouter_payload(req)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        resp = await client.post(f"{base_url}/chat/completions", headers=headers, json=payload)

    #_log_openrouter_response(resp.status_code, resp.text)

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    try:
        data = resp.json()
    except Exception:
        raise HTTPException(status_code=502, detail=f"Respuesta no JSON de OpenRouter: {resp.text}")

    if isinstance(data, dict) and data.get("error"):
        raise HTTPException(status_code=resp.status_code, detail=data.get("error"))

    choices = data.get("choices") or []
    print(data, "holi")
    if not choices:
        raise HTTPException(status_code=502, detail=f"Respuesta inválida de OpenRouter (sin choices): {data}")

    message = (choices[0] or {}).get("message") or {}
    content = message.get("content") or ""
    if not content:
        raise HTTPException(status_code=502, detail=f"Respuesta inválida de OpenRouter (sin content): {data}")

    # Minimal Ollama-compatible shape used by backend
    return {
        "model": payload.get("model"),
        "created": int(time.time()),
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
