import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    # Optional dependency; continue if dotenv is not available.
    pass

from services.errors import UpstreamServiceError


def get_llm_provider() -> str:
    """
    Selecciona proveedor LLM:
      - LOCAL_LLM_PROVIDER: "ollama" (default) | "openrouter"
    """
    return (os.environ.get("LOCAL_LLM_PROVIDER") or "ollama").strip().lower()


def get_local_llm_model_name() -> str:
    """
    Selección de modelo local (open-weight) con flag para prod:
      - LOCAL_LLM_MODEL: override absoluto (si existe, se usa tal cual)
      - LOCAL_LLM_VARIANT: "small" | "large" (default: small)
      - LOCAL_LLM_SMALL_MODEL: default llama3.1:8b
      - LOCAL_LLM_LARGE_MODEL: default llama3.3:70b-instruct
    """
    override = (os.environ.get("LOCAL_LLM_MODEL") or "").strip()
    if override:
        return override

    variant = (os.environ.get("LOCAL_LLM_VARIANT") or "small").strip().lower()
    # NOTE: Ollama tags can vary; use simple defaults and allow override via env vars.
    small = (os.environ.get("LOCAL_LLM_SMALL_MODEL") or "llama3.1:8b").strip()
    large = (os.environ.get("LOCAL_LLM_LARGE_MODEL") or "llama3.3:70b-instruct").strip()

    return large if variant == "large" else small


def get_ollama_base_url() -> str:
    return (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").strip().rstrip("/")


def get_openrouter_base_url() -> str:
    return (os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").strip().rstrip("/")


def get_openrouter_model_name() -> str:
    return (os.environ.get("OPENROUTER_MODEL") or "meta-llama/llama-3.1-8b-instruct").strip()


def _parse_csv(value: str) -> list[str]:
    parts = [p.strip() for p in (value or "").split(",")]
    return [p for p in parts if p]


def parse_json_loose(text: str) -> Any:
    def _raw_decode_first(s: str) -> Any:
        dec = json.JSONDecoder()
        s2 = s.lstrip()
        obj, _end = dec.raw_decode(s2)
        return obj

    t = (text or "").strip()
    if not t:
        raise ValueError("Empty model output")

    try:
        return _raw_decode_first(t)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return _raw_decode_first(fenced.group(1))

    # Try decoding from the first JSON-looking bracket to ignore trailing text ("extra data").
    start_any = min([p for p in [t.find("["), t.find("{")] if p != -1], default=-1)
    if start_any != -1:
        try:
            return _raw_decode_first(t[start_any:])
        except json.JSONDecodeError:
            pass

    # Last attempt: scan for any '{' or '[' and try raw_decode from there.
    for m in re.finditer(r"[\[{]", t):
        try:
            return _raw_decode_first(t[m.start() :])
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model output")


def ollama_chat_json(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
) -> Any:
    """
    Llama a Ollama /api/chat con format=json y stream=false.
    Devuelve el JSON parseado del contenido del mensaje.
    """
    model_name = model or get_local_llm_model_name()
    base_url = get_ollama_base_url()
    temp = float(os.environ.get("LOCAL_LLM_TEMPERATURE") or "0.2") if temperature is None else float(temperature)
    tout = float(os.environ.get("LOCAL_LLM_TIMEOUT_S") or "60") if timeout_s is None else float(timeout_s)
    print(tout)

    url = f"{base_url}/api/chat"
    payload = {
        "model": model_name,
        "stream": False,
        "format": "json",
        "options": {"temperature": temp},
        "messages": [{"role": "user", "content": prompt}],
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=tout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            print(f"Ollama raw response (truncado): {raw[:800]}")
            outer = json.loads(raw)
            print(f"Ollama parsed response: {outer}")
    except urllib.error.HTTPError as e:
        
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise UpstreamServiceError("Ollama", f"HTTP {e.code}: {detail}", status_code=502) from e
    except urllib.error.URLError as e:
        raise UpstreamServiceError(
            "Ollama",
            f"No se pudo conectar a {base_url} (¿ollama está corriendo?). Detalle: {e}",
            status_code=503,
        ) from e

    content = ((outer.get("message") or {}).get("content") or "").strip()
    if not content:
        raise UpstreamServiceError("Ollama", f"Respuesta inválida (sin message.content): {outer}", status_code=502)

    return parse_json_loose(content)


def _build_openrouter_provider_config() -> dict[str, Any] | None:
    """
    Construye config de providers para OpenRouter si hay envs.
    Env soportadas:
      - OPENROUTER_PROVIDER_ONLY (csv)
      - OPENROUTER_PROVIDER_ORDER (csv)
      - OPENROUTER_PROVIDER_IGNORE (csv)
      - OPENROUTER_QUANTIZATIONS (csv)
      - OPENROUTER_ALLOW_FALLBACKS (0/1)
    """
    only = _parse_csv(os.environ.get("OPENROUTER_PROVIDER_ONLY") or "")
    order = _parse_csv(os.environ.get("OPENROUTER_PROVIDER_ORDER") or "")
    ignore = _parse_csv(os.environ.get("OPENROUTER_PROVIDER_IGNORE") or "")
    quant = _parse_csv(os.environ.get("OPENROUTER_QUANTIZATIONS") or "")

    provider: dict[str, Any] = {}
    if order:
        provider["order"] = order
    if only and not order:
        provider["order"] = only
    if ignore:
        provider["ignore"] = ignore
    if quant:
        provider["quantizations"] = quant

    allow = os.environ.get("OPENROUTER_ALLOW_FALLBACKS")
    if allow is not None and allow != "":
        provider["allow_fallbacks"] = str(allow).strip().lower() not in {"0", "false", "no", "off"}
    elif only:
        provider["allow_fallbacks"] = False

    return provider or None


def openrouter_chat_json(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
) -> Any:
    """
    Llama a OpenRouter /chat/completions.
    Devuelve el JSON parseado del contenido del mensaje.
    """
    api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Falta OPENROUTER_API_KEY para usar OpenRouter.")

    model_name = model or get_openrouter_model_name()
    base_url = get_openrouter_base_url()
    temp = float(os.environ.get("OPENROUTER_TEMPERATURE") or "0.2") if temperature is None else float(temperature)
    tout = float(os.environ.get("OPENROUTER_TIMEOUT_S") or "120") if timeout_s is None else float(timeout_s)
    max_tokens = os.environ.get("OPENROUTER_MAX_TOKENS")
    force_json = (os.environ.get("OPENROUTER_FORCE_JSON") or "0").strip().lower() in {"1", "true", "yes"}
    log_raw = (os.environ.get("OPENROUTER_LOG_RAW") or "0").strip().lower() in {"1", "true", "yes"}
    log_prompt = (os.environ.get("OPENROUTER_LOG_PROMPT") or "0").strip().lower() in {"1", "true", "yes"}

    url = f"{base_url}/chat/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
    }
    if max_tokens:
        try:
            payload["max_tokens"] = int(max_tokens)
        except ValueError:
            pass
    if force_json:
        payload["response_format"] = {"type": "json_object"}

    provider = _build_openrouter_provider_config()
    if provider:
        payload["provider"] = provider

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        if log_prompt:
            print(f"[OpenRouter] Prompt length: {len(prompt)} chars")
            print("[OpenRouter] Prompt:")
            print(prompt)
        with urllib.request.urlopen(req, timeout=tout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            if log_raw:
                print(f"OpenRouter raw response (truncado): {raw[:1200]}")
            outer = json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise UpstreamServiceError("OpenRouter", f"HTTP {e.code}: {detail}", status_code=502) from e
    except urllib.error.URLError as e:
        raise UpstreamServiceError(
            "OpenRouter",
            f"No se pudo conectar a {base_url}. Detalle: {e}",
            status_code=503,
        ) from e

    choices = outer.get("choices") or []
    if not choices:
        raise UpstreamServiceError("OpenRouter", f"Respuesta inválida (sin choices): {outer}", status_code=502)

    msg = choices[0].get("message") or {}
    content = (msg.get("content") or "").strip()
    if not content:
        # Algunas respuestas devuelven texto directo en choices[0].text
        content = (choices[0].get("text") or "").strip()
    if not content:
        raise UpstreamServiceError("OpenRouter", f"Respuesta inválida (sin content): {outer}", status_code=502)

    return parse_json_loose(content)


def llm_chat_json(
    prompt: str,
    model: str | None = None,
    temperature: float | None = None,
    timeout_s: float | None = None,
) -> Any:
    """
    Wrapper que enruta a Ollama u OpenRouter según LOCAL_LLM_PROVIDER.
    """
    provider = get_llm_provider()
    if provider == "openrouter":
        return openrouter_chat_json(prompt, model=model, temperature=temperature, timeout_s=timeout_s)
    return ollama_chat_json(prompt, model=model, temperature=temperature, timeout_s=timeout_s)
