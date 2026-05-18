"""
Tarea 2 – Extracción de contribuciones científicas.

Función pública: analyze_contributions(segments, model)
→ Recibe los segmentos de la Tarea 1 y devuelve fragmentos anotados.

Para reemplazar el mock por un modelo real, edita únicamente
_call_real_model() y pon la lógica de inferencia ahí.
"""

import re
import time
import random
import json
import os
import hashlib
from pathlib import Path
import urllib.request
import urllib.error
from typing import Any
import logging

from services.models import MODELS
from services.local_llm import llm_chat_json, parse_json_loose
from services.errors import UpstreamServiceError
from services.s3_hf import download_hf_model_from_s3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nota: Tarea 2 es binaria (is_contribution). No clasificamos tipo de contribución.
# Usamos un "perfil" solo para elegir patrones de highlight (no se expone al cliente).
# ---------------------------------------------------------------------------
_LABEL_TO_HIGHLIGHT_PROFILE = {
    "METH":  "Metodológica",
    "RES":   "Empírica",
    "CONTR": "Recurso",
    "DISC":  "Conceptual",
    "INTRO": "Conceptual",
    "BACK":  "Conceptual",
    "LIM":   "Conceptual",
    "CONC":  "Metodológica",
}

# Labels más propensos a tener contribuciones
_HIGH_CONTRIBUTION_LABELS = {"CONTR", "METH", "RES"}
_MED_CONTRIBUTION_LABELS = {"DISC", "CONC", "INTRO"}

# Frases de contribución por tipo (para el highlight mock)
_HIGHLIGHT_PATTERNS = {
    "Metodológica": [
        r"propone?mos\s+(?:un|una|el|la)\s+\w+(?:\s+\w+){1,5}",
        r"nuevo\s+(?:método|enfoque|algoritmo|sistema|marco|modelo)\s+\w+(?:\s+\w+){0,4}",
        r"implementa(?:mos|ción)\s+\w+(?:\s+\w+){1,4}",
        r"utilizamos\s+(?:un|una)\s+\w+(?:\s+\w+){1,5}",
    ],
    "Empírica": [
        r"obtuv(?:imos|o)\s+(?:un|una)\s+\w+(?:\s+\w+){1,4}",
        r"f1\s*(?:score|=|de)\s*[\d.,]+",
        r"precisión\s+de\s+[\d.,]+\s*%",
        r"supera(?:mos|ndo)?\s+(?:el|la|los)\s+\w+(?:\s+\w+){1,4}",
        r"mejora(?:mos|ndo)?\s+(?:en|el|la)\s+\w+(?:\s+\w+){1,4}",
    ],
    "Recurso": [
        r"corpus\s+\w+(?:\s+\w+){0,4}",
        r"dataset\s+\w+(?:\s+\w+){0,4}",
        r"recurso\s+(?:léxico|lingüístico|anotado)\s+\w+(?:\s+\w+){0,3}",
        r"publicamos\s+\w+(?:\s+\w+){1,4}",
        r"ponemos\s+a\s+disposición\s+\w+(?:\s+\w+){1,4}",
    ],
    "Conceptual": [
        r"definimos\s+(?:el|la|un|una)\s+\w+(?:\s+\w+){1,4}",
        r"proponemos\s+(?:un|una)\s+(?:marco|taxonomía|definición)\s+\w+(?:\s+\w+){0,4}",
        r"concepto\s+de\s+\w+(?:\s+\w+){1,4}",
        r"demostram(?:os|amos)\s+que\s+\w+(?:\s+\w+){1,4}",
    ],
}


def _find_highlight(text: str, contribution_type: str, rng: random.Random) -> str:
    """Extrae una frase clave del texto para resaltar como contribución."""
    patterns = _HIGHLIGHT_PATTERNS.get(contribution_type, [])
    lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            start, end = match.span()
            return text[start:end].strip()

    # Fallback: devolver las primeras palabras de una oración con contribución
    sentences = re.split(r"[.;]", text)
    for sent in sentences:
        if len(sent.split()) >= 6:
            words = sent.strip().split()
            return " ".join(words[:min(10, len(words))])

    return text[:80].strip()


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    s = str(value).strip().lower()
    if s in {"true", "t", "1", "yes", "y", "si", "sí"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    return default


def _mock_analyze(segments: list[dict], model: str) -> dict:
    """
    Mock de extracción de contribuciones.
    Usa los segmentos reales de la Tarea 1.
    """
    model_config = MODELS[model]
    rng = random.Random(hash(str(segments)[:80]) ^ hash(model))

    fragments = []
    for seg in segments:
        label = seg["label"]
        text = seg["text"]

        # Probabilidad de que este fragmento sea contribución según su label y modelo
        if label in _HIGH_CONTRIBUTION_LABELS:
            base_prob = 0.80
        elif label in _MED_CONTRIBUTION_LABELS:
            base_prob = 0.40
        else:
            base_prob = 0.20

        # Ajuste por modelo
        if model == "encoder":
            prob_noise = rng.uniform(-0.10, 0.05)
        elif model == "llm":
            prob_noise = rng.uniform(-0.05, 0.08)
        else:  # api
            prob_noise = rng.uniform(0.0, 0.10)

        is_contribution = rng.random() < (base_prob + prob_noise)

        # Confianza
        if is_contribution:
            base_conf = 0.78 + rng.uniform(0, 0.18)
            if model == "api":
                base_conf += 0.04
            elif model == "encoder":
                base_conf -= 0.03
        else:
            base_conf = 0.30 + rng.uniform(0, 0.30)

        confidence = round(min(max(base_conf, 0.10), 0.99), 2)

        highlight = ""
        if is_contribution:
            profile = _LABEL_TO_HIGHLIGHT_PROFILE.get(label, "Conceptual")
            highlight = _find_highlight(text, profile, rng)

        fragments.append({
            "paragraph_index": seg["paragraph_index"],
            "text": text,
            "is_contribution": is_contribution,
            "contribution_type": None,
            "confidence": confidence,
            "highlight": highlight,
            "source_label": label,
        })

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
        },
    }


# ---------------------------------------------------------------------------
# Punto de entrada público — REEMPLAZA ESTE BLOQUE CON EL MODELO REAL
# ---------------------------------------------------------------------------

def analyze_contributions(segments: list[dict], model: str, encoder_variant: str = "roberta") -> dict:
    """
    Analiza los segmentos e identifica contribuciones científicas.

    Args:
        segments: Lista de segmentos con label (salida de analyze_segments).
        model:    Identificador del modelo ("encoder", "llm", "api").

    Returns:
        Dict con "fragments" y "stats".

    TODO: Reemplazar _mock_analyze() por _call_real_model() cuando las APIs estén disponibles.
    """
    if model == "llm":
        return _call_local_llm(segments)

    if model == "encoder":
        return _call_encoder_model(segments, encoder_variant=encoder_variant)

    if model == "api":
        return _call_gemini_api(segments)

    time.sleep(MODELS[model]["simulated_delay_s"] * 0.5)
    return _mock_analyze(segments, model)


# ---------------------------------------------------------------------------
# Stub para el modelo real
# ---------------------------------------------------------------------------

def _call_real_model(segments: list[dict], model: str) -> dict:
    """
    STUB — Implementar con llamada real al modelo.

    El formato de retorno debe ser idéntico al de _mock_analyze().
    """
    raise NotImplementedError("Real model not yet configured.")


def _call_local_llm(segments: list[dict]) -> dict:
    """
    LLM open-weight local (p.ej. Ollama) para decidir si cada segmento expresa
    explícitamente una contribución científica (binario).
    """
    items = [{"paragraph_index": s["paragraph_index"], "text": s["text"], "label": s["label"]} for s in segments]
    prompt = (
        "Decide si cada fragmento (párrafo) expresa EXPLÍCITAMENTE una contribución científica.\n"
        "Marca positivo SOLO si hay formulación explícita de aporte/novelty (p.ej. 'proponemos', 'presentamos', 'nuestra contribución', 'ponemos a disposición').\n"
        "Devuelve SOLO JSON (sin Markdown) como ARREGLO, un item por entrada con:\n"
        '  - "paragraph_index": int\n'
        '  - "is_contribution": boolean\n'
        '  - "confidence": number 0..1\n\n'
        "IMPORTANTE: NO incluyas el campo 'text' en la salida. NO devuelvas claves extra.\n\n"
        "Entrada (JSON):\n"
        f"{json.dumps(items, ensure_ascii=False)}"
        "ejemplo de salida esperada:\n"
        "{result: [{\"paragraph_index\": 0, \"is_contribution\": true, \"confidence\": 0.92}, {\"paragraph_index\": 1, \"is_contribution\": false, \"confidence\": 0.15}, ...]}"
    )

    started = time.perf_counter()
    parsed = llm_chat_json(prompt)
    elapsed = round(time.perf_counter() - started, 2)
    parsed = parsed.get("result") if isinstance(parsed, dict) and "result" in parsed else parsed

    if isinstance(parsed, dict):
        parsed = parsed.get("labels") or parsed.get("fragments") or parsed.get("items") or parsed

    if not isinstance(parsed, list):
        raise TypeError("Salida del modelo inválida: se esperaba un arreglo JSON (o un objeto con 'labels').")

    by_idx = {}
    for it in parsed:
        if isinstance(it, dict) and isinstance(it.get("paragraph_index"), int):
            by_idx[int(it["paragraph_index"])] = it

    fragments = []
    for seg in segments:
        idx = int(seg["paragraph_index"])
        out = by_idx.get(idx, {})
        is_contribution = _parse_bool(out.get("is_contribution", False), default=False)
        try:
            confidence = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        profile = _LABEL_TO_HIGHLIGHT_PROFILE.get(seg["label"], "Conceptual")
        highlight = _find_highlight(seg["text"], profile, random.Random(idx))
        if not is_contribution:
            highlight = ""

        fragments.append(
            {
                "paragraph_index": idx,
                "text": seg["text"],
                "is_contribution": is_contribution,
                "contribution_type": None,
                "confidence": confidence,
                "highlight": highlight,
                "source_label": seg["label"],
            }
        )

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
            "time_seconds": elapsed,
        },
    }


def _http_post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        raise UpstreamServiceError("Gemini", f"HTTP {e.code}: {detail}", status_code=502) from e
    except urllib.error.URLError as e:
        raise UpstreamServiceError("Gemini", f"Error de red llamando a Gemini: {e}", status_code=503) from e


def _extract_gemini_text(response_data: dict[str, Any]) -> str:
    candidates = response_data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Respuesta Gemini sin candidates: {response_data}")
    content = (candidates[0].get("content") or {})
    parts = content.get("parts") or []
    texts = []
    for part in parts:
        t = part.get("text")
        if t:
            texts.append(t)
    text = "\n".join(texts).strip()
    if not text:
        raise RuntimeError(f"Respuesta Gemini sin texto en parts: {response_data}")
    return text


def _call_gemini_api(segments: list[dict]) -> dict:
    """
    API comercial (Gemini) para Tarea 2 (binario).
    Devuelve SOLO is_contribution + confidence por párrafo (sin tipo).

    Env (mismo set que Task1):
      - GEMINI_API_KEY (requerida)
      - GEMINI_MODEL (opcional, default: gemini-1.5-flash)
      - GEMINI_API_BASE (opcional, default: https://generativelanguage.googleapis.com/v1beta)
      - GEMINI_TEMPERATURE (opcional, default: 0.2)
      - GEMINI_MAX_OUTPUT_TOKENS (opcional, default: 2048)
      - GEMINI_TIMEOUT_S (opcional, default: 45)
      - GEMINI_RESPONSE_MIME_TYPE (opcional, por ejemplo: application/json)
    """
    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Falta GEMINI_API_KEY. Para usar el modelo 'api', define GEMINI_API_KEY en el entorno.")

    model_id = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
    api_base = (os.environ.get("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")
    temperature = float(os.environ.get("GEMINI_TEMPERATURE") or "0.2")
    max_output_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "2048")
    timeout_s = float(os.environ.get("GEMINI_TIMEOUT_S") or "60")
    structured_output = (os.environ.get("GEMINI_STRUCTURED_OUTPUT") or "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    response_mime_type = (os.environ.get("GEMINI_RESPONSE_MIME_TYPE") or "").strip()
    if structured_output and not response_mime_type:
        response_mime_type = "application/json"

    items = [{"paragraph_index": s["paragraph_index"], "text": s["text"], "label": s["label"]} for s in segments]
    prompt = (
        "Decide si cada fragmento (párrafo) expresa EXPLÍCITAMENTE una contribución científica.\n"
        "Marca positivo SOLO si hay formulación explícita de aporte/novelty (p.ej. 'proponemos', 'presentamos', 'nuestra contribución', 'ponemos a disposición').\n"
        "Devuelve SOLO JSON (sin Markdown), como un ARREGLO con la misma cantidad de items que la entrada.\n"
        "Cada item debe tener: paragraph_index (int), is_contribution (boolean), confidence (0..1).\n\n"
        "IMPORTANTE: NO incluyas el campo 'text' en la salida. NO devuelvas claves extra.\n\n"
        "Entrada (JSON):\n"
        f"{json.dumps(items, ensure_ascii=False)}"
        "ejemplo de salida esperada:\n"
        "{result: [{\"paragraph_index\": 0, \"is_contribution\": true, \"confidence\": 0.92}, {\"paragraph_index\": 1, \"is_contribution\": false, \"confidence\": 0.15}, ...]}"
    )

    url = f"{api_base}/models/{model_id}:generateContent?key={api_key}"
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    logger.info(
        "Task2 Gemini request: model=%s fragments=%d responseMimeType=%s",
        model_id,
        len(segments),
        generation_config.get("responseMimeType") or "",
    )

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]},
        ],
        "generationConfig": generation_config,
    }

    started = time.perf_counter()
    response_data = _http_post_json(url, payload, timeout_s=timeout_s)
    elapsed = round(time.perf_counter() - started, 2)

    llm_text = _extract_gemini_text(response_data)
    logger.debug("Task2 Gemini raw text (truncado): %s", llm_text[:800])
    try:
        parsed = parse_json_loose(llm_text)
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "No se pudo parsear JSON desde Gemini (Task2). Texto recibido (truncado): %s",
            llm_text[:1200],
        )
        raise ValueError("Gemini devolvió una respuesta que no es JSON válido (Task2).") from e

    if isinstance(parsed, dict):
        parsed = parsed.get("result") or parsed.get("fragments") or parsed.get("items") or parsed

    if not isinstance(parsed, list):
        raise TypeError("Salida del modelo inválida: se esperaba un arreglo JSON.")

    by_idx: dict[int, dict[str, Any]] = {}
    for it in parsed:
        if isinstance(it, dict) and isinstance(it.get("paragraph_index"), int):
            by_idx[int(it["paragraph_index"])] = it

    fragments = []
    for seg in segments:
        idx = int(seg["paragraph_index"])
        out = by_idx.get(idx, {})
        is_contribution = _parse_bool(out.get("is_contribution", False), default=False)
        try:
            confidence = float(out.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        profile = _LABEL_TO_HIGHLIGHT_PROFILE.get(seg["label"], "Conceptual")
        highlight = _find_highlight(seg["text"], profile, random.Random(idx)) if is_contribution else ""

        fragments.append(
            {
                "paragraph_index": idx,
                "text": seg["text"],
                "is_contribution": is_contribution,
                "contribution_type": None,
                "confidence": confidence,
                "highlight": highlight,
                "source_label": seg["label"],
            }
        )

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
            "time_seconds": elapsed,
        },
    }


_TASK2_ENCODER_CACHE: dict[str, tuple[Any, Any, str, int]] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def task2_encoder_is_configured(encoder_variant: str) -> bool:
    """
    Verifica si hay configuración (URI o path) para el encoder de Task2 sin forzar descargas.
    """
    variant = (encoder_variant or "roberta").strip().lower()
    if variant not in {"roberta", "scibert"}:
        return False

    mlflow_var = f"TASK2_ENCODER_{variant.upper()}_MLFLOW_MODEL_URI"
    path_var = f"TASK2_ENCODER_{variant.upper()}_MODEL_PATH"

    if (os.environ.get(mlflow_var) or "").strip() or (os.environ.get("TASK2_ENCODER_MLFLOW_MODEL_URI") or "").strip():
        return True

    model_dir = (os.environ.get(path_var) or "").strip() or (os.environ.get("TASK2_ENCODER_MODEL_PATH") or "").strip()
    if model_dir:
        return Path(model_dir).exists()

    default_dir = "roberta_bne_task2" if variant == "roberta" else "scibert_task2"
    return (_repo_root() / "src" / "models" / default_dir).exists()


def _download_model_from_mlflow(model_uri: str, cache_prefix: str) -> Path:
    if model_uri.strip().startswith("s3://"):
        cache_root = (os.environ.get("MODEL_CACHE_DIR") or str(_repo_root() / "artifacts" / "model_cache")).strip()
        cache_root_path = Path(cache_root)
        cache_root_path.mkdir(parents=True, exist_ok=True)

        key = hashlib.sha1(model_uri.encode("utf-8")).hexdigest()[:12]
        dst = cache_root_path / f"{cache_prefix}_{key}"
        dst.mkdir(parents=True, exist_ok=True)

        existing_cfgs = list(dst.rglob("config.json"))
        if existing_cfgs:
            parent = existing_cfgs[0].parent
            if (parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists():
                return parent
        return download_hf_model_from_s3(model_uri, dst)

    try:
        import mlflow
        from mlflow.artifacts import download_artifacts
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Para cargar el modelo Task2 desde MLflow instala dependencias: pip install mlflow boto3"
        ) from e

    tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    cache_root = (os.environ.get("MODEL_CACHE_DIR") or str(_repo_root() / "artifacts" / "model_cache")).strip()
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha1(model_uri.encode("utf-8")).hexdigest()[:12]
    dst = cache_root_path / f"{cache_prefix}_{key}"
    dst.mkdir(parents=True, exist_ok=True)

    # Reuse if already there (including nested)
    existing_cfgs = list(dst.rglob("config.json"))
    if existing_cfgs:
        parent = existing_cfgs[0].parent
        if (parent / "model.safetensors").exists() or (parent / "pytorch_model.bin").exists():
            return parent

    def _is_s3_listbucket_access_denied(exc: Exception) -> bool:
        message = str(exc)
        if "AccessDenied" in message and "ListBucket" in message:
            return True
        return "not authorized to perform: s3:ListBucket" in message

    def _runs_uri_to_s3_prefix(uri: str) -> str | None:
        if not uri.startswith("runs:/"):
            return None
        tail = uri[len("runs:/") :].lstrip("/")
        if not tail:
            return None
        parts = tail.split("/", 1)
        run_id = parts[0].strip()
        artifact_path = parts[1].strip("/") if len(parts) > 1 else ""
        if not run_id:
            return None
        try:
            from mlflow.tracking import MlflowClient
        except Exception:  # noqa: BLE001
            return None
        artifact_uri = (MlflowClient().get_run(run_id).info.artifact_uri or "").strip()
        if not artifact_uri.startswith("s3://"):
            return None
        if artifact_path:
            return f"{artifact_uri.rstrip('/')}/{artifact_path}"
        return artifact_uri

    logger.info("Task2 encoder: descargando artefacto MLflow (%s) a %s", model_uri, str(dst))
    try:
        downloaded_path = download_artifacts(artifact_uri=model_uri, dst_path=str(dst))
    except Exception as e:  # noqa: BLE001
        fallback_s3_uri = _runs_uri_to_s3_prefix(model_uri)
        if fallback_s3_uri and _is_s3_listbucket_access_denied(e):
            logger.warning(
                "Task2 encoder: MLflow falló por ListBucket denegado; usando fallback S3 sin listado. "
                "model_uri=%s s3_uri=%s",
                model_uri,
                fallback_s3_uri,
            )
            return download_hf_model_from_s3(fallback_s3_uri, dst)
        raise
    downloaded = Path(downloaded_path)
    if downloaded.is_file():
        downloaded = downloaded.parent

    if not (downloaded / "config.json").exists():
        candidates = list(downloaded.rglob("config.json"))
        if candidates:
            downloaded = candidates[0].parent

    if not (downloaded / "config.json").exists():
        raise RuntimeError(f"No se encontró config.json dentro del artefacto descargado: {downloaded}")

    return downloaded


def _resolve_task2_encoder_model_path(encoder_variant: str) -> Path:
    variant = (encoder_variant or "roberta").strip().lower()
    if variant not in {"roberta", "scibert"}:
        raise ValueError("encoder_variant inválido para Task2. Usa 'roberta' o 'scibert'.")

    mlflow_var = f"TASK2_ENCODER_{variant.upper()}_MLFLOW_MODEL_URI"
    path_var = f"TASK2_ENCODER_{variant.upper()}_MODEL_PATH"

    mlflow_uri = (os.environ.get(mlflow_var) or "").strip()
    if not mlflow_uri:
        mlflow_uri = (os.environ.get("TASK2_ENCODER_MLFLOW_MODEL_URI") or "").strip()
    if mlflow_uri:
        return _download_model_from_mlflow(mlflow_uri, cache_prefix=f"task2_encoder_{variant}")

    model_dir = (os.environ.get(path_var) or "").strip()
    if not model_dir:
        model_dir = (os.environ.get("TASK2_ENCODER_MODEL_PATH") or "").strip()
    if model_dir:
        return Path(model_dir)

    # default local path (if you export locally)
    default_dir = "roberta_bne_task2" if variant == "roberta" else "scibert_task2"
    model_path = _repo_root() / "src" / "models" / default_dir
    if not model_path.exists():
        raise FileNotFoundError(
            "No hay modelo encoder de Task2 configurado.\n"
            "Configura una de estas variables:\n"
            f"  - {mlflow_var} (recomendado, runs:/.../hf_model)\n"
            f"  - {path_var} (path local al modelo exportado)\n"
            "O usa model='llm' / model='api' para contribuciones."
        )
    return model_path


def _load_task2_encoder(encoder_variant: str):
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Para usar el encoder en Task2 instala dependencias: pip install torch transformers safetensors"
        ) from e

    model_path = _resolve_task2_encoder_model_path(encoder_variant=encoder_variant)
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo Task2 en {model_path}")

    has_weights = any(
        (model_path / name).exists()
        for name in [
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        ]
    )
    if not has_weights:
        files = sorted([p.name for p in model_path.iterdir() if p.is_file()])
        raise FileNotFoundError(
            "El artefacto descargado para Task2 no contiene pesos del modelo.\n"
            f"Directorio: {model_path}\n"
            f"Archivos encontrados: {files}\n"
            "Se esperaba 'model.safetensors' (o 'pytorch_model.bin').\n"
            "Solución: re-exporta/re-sube a S3 una carpeta HF que incluya los pesos."
        )
    cache_key = str(model_path.resolve())
    if cache_key in _TASK2_ENCODER_CACHE:
        return _TASK2_ENCODER_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Determine positive class id
    label2id = getattr(model.config, "label2id", None) or {}
    pos_id = None
    for k in ["POS", "CONTRIBUTION", "YES", "TRUE", "1"]:
        if k in label2id:
            pos_id = int(label2id[k])
            break
    if pos_id is None:
        pos_id = 1  # common convention

    _TASK2_ENCODER_CACHE[cache_key] = (tokenizer, model, device, pos_id)
    return _TASK2_ENCODER_CACHE[cache_key]


def _call_encoder_model(segments: list[dict], encoder_variant: str) -> dict:
    import math

    tokenizer, model, device, pos_id = _load_task2_encoder(encoder_variant=encoder_variant)
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Falta torch para inferencia del encoder Task2.") from e

    threshold = float(os.environ.get("TASK2_ENCODER_THRESHOLD") or "0.5")
    batch_size = int(os.environ.get("TASK2_ENCODER_BATCH_SIZE") or "8")
    batch_size = max(1, min(batch_size, 64))

    texts = [str(s["text"]) for s in segments]
    started = time.perf_counter()

    probs_pos: list[float] = []
    with torch.no_grad():
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # if model has only 1 logit, fallback
            if probs.shape[-1] == 1:
                chunk_probs = probs.squeeze(-1).tolist()
            else:
                chunk_probs = probs[:, pos_id].tolist()
            probs_pos.extend([float(p) for p in chunk_probs])

    elapsed = round(time.perf_counter() - started, 2)

    fragments = []
    for seg, ppos in zip(segments, probs_pos, strict=False):
        idx = int(seg["paragraph_index"])
        is_contribution = bool(ppos >= threshold)
        confidence = float(ppos) if is_contribution else float(1.0 - ppos)
        if math.isnan(confidence):
            confidence = 0.0
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        profile = _LABEL_TO_HIGHLIGHT_PROFILE.get(seg["label"], "Conceptual")
        highlight = _find_highlight(seg["text"], profile, random.Random(idx)) if is_contribution else ""

        fragments.append(
            {
                "paragraph_index": idx,
                "text": seg["text"],
                "is_contribution": is_contribution,
                "contribution_type": None,
                "confidence": confidence,
                "highlight": highlight,
                "source_label": seg["label"],
            }
        )

    positives = [f for f in fragments if f["is_contribution"]]
    avg_conf_pos = (
        round(sum(f["confidence"] for f in positives) / len(positives), 3)
        if positives else 0.0
    )

    return {
        "fragments": fragments,
        "stats": {
            "total_fragments": len(fragments),
            "positive": len(positives),
            "negative": len(fragments) - len(positives),
            "avg_confidence_positive": avg_conf_pos,
            "time_seconds": elapsed,
        },
    }
