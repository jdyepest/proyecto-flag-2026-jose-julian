"""
Tarea 1 – Segmentación retórica.

Función pública: analyze_segments(text, model)
→ Devuelve lista de segmentos con label y confianza.

Para reemplazar el mock por un modelo real, edita únicamente
_call_real_model() y pon la lógica de inferencia ahí.
"""

import re
import time
import random
import os
import json
import hashlib
from pathlib import Path
import urllib.request
import urllib.error
from typing import Any
import logging

from services.models import MODELS
from services.models import RHETORICAL_LABELS
from services.local_llm import llm_chat_json, parse_json_loose
from services.errors import UpstreamServiceError
from services.s3_hf import download_hf_model_from_s3

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords por categoría (usadas en el mock para simular decisión)
# ---------------------------------------------------------------------------
_KEYWORDS = {
    "INTRO": [
        "introducción", "este trabajo", "en este artículo", "el presente",
        "motivación", "este estudio", "objetivo", "propósito", "presente trabajo",
    ],
    "BACK": [
        "antecedentes", "trabajos previos", "revisión de literatura", "estado del arte",
        "investigaciones anteriores", "han propuesto", "fue propuesto", "en la literatura",
        "según", "de acuerdo con", "autores",
    ],
    "METH": [
        "metodología", "método", "metodológico", "procedimiento", "experimento",
        "implementación", "arquitectura", "entrenamiento", "corpus", "dataset",
        "conjunto de datos", "evaluamos", "utilizamos", "se utilizó", "fine-tuning",
        "hiperparámetros", "configuración",
    ],
    "RES": [
        "resultado", "tabla", "figura", "obtuvo", "obtuvimos", "rendimiento",
        "accuracy", "f1", "precisión", "recall", "métricas", "muestra",
        "se observa", "se puede ver", "en la tabla", "en la figura",
    ],
    "DISC": [
        "discusión", "análisis", "interpretamos", "esto sugiere", "esto indica",
        "podemos inferir", "se debe a", "explicar", "probable", "parece",
        "comparado con", "en comparación",
    ],
    "CONTR": [
        "contribución", "aporte", "propuesta", "novedad", "innovación",
        "original", "nueva", "nuevo", "presentamos", "proponemos",
        "a diferencia de", "primer trabajo",
    ],
    "LIM": [
        "limitación", "limitaciones", "trabajo futuro", "futuras investigaciones",
        "no se consideró", "fuera del alcance", "restricción", "sesgo",
    ],
    "CONC": [
        "conclusión", "conclusiones", "en conclusión", "en resumen",
        "resumiendo", "concluimos", "hemos mostrado", "hemos demostrado",
        "finalmente", "en definitiva",
    ],
}

# Orden posicional esperado (para párrafos sin keywords claros)
_POSITIONAL_ORDER = ["INTRO", "BACK", "METH", "RES", "DISC", "CONTR", "LIM", "CONC"]


def _split_paragraphs(text: str) -> list[str]:
    """Divide el texto en párrafos no vacíos."""
    paragraphs = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in paragraphs if len(p.strip()) > 20]


def _keyword_score(paragraph: str) -> dict[str, float]:
    """Calcula score por categoría según keywords."""
    lower = paragraph.lower()
    scores = {}
    for label, kws in _KEYWORDS.items():
        count = sum(1 for kw in kws if kw in lower)
        scores[label] = count
    return scores


def _mock_label_paragraph(
    paragraph: str,
    index: int,
    total: int,
    model: str,
    rng: random.Random,
) -> tuple[str, float]:
    """
    Asigna un label y confianza simulados a un párrafo.

    Estrategia:
    1. Calcular scores por keywords.
    2. Si hay un ganador claro, usarlo con confianza alta.
    3. Si hay empate, usar posición en el texto como desempate.
    4. Añadir variación según modelo.
    """
    scores = _keyword_score(paragraph)
    max_score = max(scores.values())

    # Fuerza el primer y último párrafo a INTRO/CONC si no tienen keywords fuertes
    if index == 0 and max_score < 2:
        label = "INTRO"
        base_conf = 0.82
    elif index == total - 1 and max_score < 2:
        label = "CONC"
        base_conf = 0.80
    elif max_score > 0:
        # Tomar el label con mayor score
        label = max(scores, key=lambda k: scores[k])
        # Confianza proporcional al score
        base_conf = 0.72 + min(max_score * 0.05, 0.22)
    else:
        # Sin keywords: usar posición relativa
        pos_ratio = index / max(total - 1, 1)
        order_idx = int(pos_ratio * (len(_POSITIONAL_ORDER) - 1))
        label = _POSITIONAL_ORDER[order_idx]
        base_conf = 0.62 + rng.uniform(0, 0.10)

    # Ajuste por modelo (encoder es más rápido pero varía más)
    if model == "encoder":
        noise = rng.uniform(-0.06, 0.06)
    elif model == "llm":
        noise = rng.uniform(-0.04, 0.05)
    else:  # api
        noise = rng.uniform(-0.02, 0.04)

    confidence = round(min(max(base_conf + noise, 0.50), 0.99), 2)
    return label, confidence


def _mock_analyze(text: str, model: str) -> dict:
    """
    Mock completo de segmentación retórica.
    Usa el texto real del usuario para producir resultados coherentes.
    """
    model_config = MODELS[model]
    rng = random.Random(hash(text[:100]) ^ hash(model))  # determinista por texto+modelo

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    total = len(paragraphs)
    segments = []
    for i, para in enumerate(paragraphs):
        label, confidence = _mock_label_paragraph(para, i, total, model, rng)
        segments.append({
            "paragraph_index": i,
            "text": para,
            "label": label,
            "confidence": confidence,
        })

    word_count = sum(len(p.split()) for p in paragraphs)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 3)
    elapsed = round(model_config["simulated_delay_s"] + rng.uniform(0.1, 0.4), 2)

    return {
        "segments": segments,
        "stats": {
            "total_paragraphs": total,
            "total_words": word_count,
            "avg_confidence": avg_conf,
            "time_seconds": elapsed,
        },
    }


# ---------------------------------------------------------------------------
# Punto de entrada público — REEMPLAZA ESTE BLOQUE CON EL MODELO REAL
# ---------------------------------------------------------------------------

def analyze_segments(text: str, model: str, encoder_variant: str = "roberta") -> dict:
    """
    Analiza el texto y devuelve segmentos retóricos con labels y confianzas.

    Args:
        text:  Texto del artículo científico.
        model: Identificador del modelo ("encoder", "llm", "api").

    Returns:
        Dict con "segments" y "stats".

    TODO: Reemplazar _mock_analyze() por _call_real_model() cuando las APIs estén disponibles.
    """
    if model == "api":
        return _call_real_model(text, model)

    if model == "llm":
        return _call_local_llm(text)

    if model == "encoder":
        return _call_encoder_model(text, encoder_variant=encoder_variant)

    # Simular latencia de red/inferencia (solo en mock)
    time.sleep(MODELS[model]["simulated_delay_s"])
    return _mock_analyze(text, model)


# ---------------------------------------------------------------------------
# Stub para el modelo real (implementar cuando esté disponible)
# ---------------------------------------------------------------------------

def _call_real_model(text: str, model: str) -> dict:
    """
    Llamada real a Gemini (Google Generative Language API).

    Config por variables de entorno:
      - GEMINI_API_KEY (requerida)
      - GEMINI_MODEL (opcional, default: gemini-1.5-flash)
      - GEMINI_API_BASE (opcional, default: https://generativelanguage.googleapis.com/v1beta)
      - GEMINI_TEMPERATURE (opcional, default: 0.2)
      - GEMINI_MAX_OUTPUT_TOKENS (opcional, default: 4096)
      - GEMINI_TIMEOUT_S (opcional, default: 30)
      - GEMINI_RESPONSE_MIME_TYPE (opcional, por ejemplo: application/json)

    El modelo recibe TODOS los párrafos en una sola llamada y debe devolver
    un JSON con una entrada por párrafo:
      {"paragraph_index": 0, "label": "INTRO", "confidence": 0.91}
    """

    api_key = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("Falta GEMINI_API_KEY. Para usar el modelo 'api', define GEMINI_API_KEY en el entorno.")

    model_id = (os.environ.get("GEMINI_MODEL") or "gemini-2.5-flash-lite").strip()
    api_base = (os.environ.get("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta").strip().rstrip("/")
    temperature = float(os.environ.get("GEMINI_TEMPERATURE") or "0.2")
    max_output_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") or "4096")
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

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    prompt = _build_gemini_prompt(paragraphs)

    url = f"{api_base}/models/{model_id}:generateContent?key={api_key}"
    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }
    if response_mime_type:
        generation_config["responseMimeType"] = response_mime_type

    logger.info(
        "Task1 Gemini request: model=%s paragraphs=%d responseMimeType=%s",
        model_id,
        len(paragraphs),
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
    logger.debug("Task1 Gemini raw text (truncado): %s", llm_text[:800])
    try:
        parsed = parse_json_loose(llm_text)
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "No se pudo parsear JSON desde Gemini (Task1). Texto recibido (truncado): %s",
            llm_text[:1200],
        )
        raise ValueError("Gemini devolvió una respuesta que no es JSON válido (Task1).") from e

    segments = _normalize_segments_output(parsed, paragraphs)
    word_count = sum(len(p.split()) for p in paragraphs)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 3) if segments else 0.0

    return {
        "segments": segments,
        "stats": {
            "total_paragraphs": len(paragraphs),
            "total_words": word_count,
            "avg_confidence": avg_conf,
            "time_seconds": elapsed,
        },
    }


def _call_local_llm(text: str) -> dict:
    """
    LLM open-weight local (p.ej. Ollama). Usa 1 llamada con TODOS los párrafos.
    Selección de modelo ~3B vs 7/8B vía env en services/local_llm.py.
    """
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    input_items = [{"paragraph_index": i, "text": p} for i, p in enumerate(paragraphs)]
    prompt = (
        "Clasifica cada párrafo de un artículo científico en español en UNA etiqueta retórica.\n"
        "Devuelve SOLO JSON (sin Markdown), un ARREGLO con la misma cantidad de items que la entrada.\n"
        "Cada item debe tener: paragraph_index (int), label (one of INTRO,BACK,METH,RES,DISC,CONTR,LIM,CONC), confidence (0..1).\n"
        "IMPORTANTE: Usa 'RES' (no 'RESU').\n\n"
        "IMPORTANTE: NO incluyas el campo 'text' en la salida. NO devuelvas claves extra.\n\n"
        "Entrada (JSON):\n"
        f"{json.dumps(input_items, ensure_ascii=False)}"
        "ejemplo de salida esperada:\n"
        '{ "result" : [{"paragraph_index": 0, "label": "INTRO", "confidence": 0.92}, {"paragraph_index": 1, "label": "METH", "confidence": 0.88}, ...]}'
    )

    logger.info("Task1 local LLM request: paragraphs=%d", len(paragraphs))

    started = time.perf_counter()
    parsed = llm_chat_json(prompt)
    elapsed = round(time.perf_counter() - started, 2)
    logger.debug("Task1 local LLM raw text (truncado): %s", str(parsed)[:800])
    parsed = parsed.get("result") if isinstance(parsed, dict) and "result" in parsed else parsed

    # Reusar normalización existente
    segments = _normalize_segments_output(parsed, paragraphs)
    word_count = sum(len(p.split()) for p in paragraphs)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 3) if segments else 0.0

    return {
        "segments": segments,
        "stats": {
            "total_paragraphs": len(paragraphs),
            "total_words": word_count,
            "avg_confidence": avg_conf,
            "time_seconds": elapsed,
        },
    }


def _build_gemini_prompt(paragraphs: list[str]) -> str:
    labels = ", ".join(RHETORICAL_LABELS)
    input_items = [{"paragraph_index": i, "text": p} for i, p in enumerate(paragraphs)]
    input_json = json.dumps(input_items, ensure_ascii=False)
    return (
        "Clasifica cada párrafo de un artículo científico en español en UNA etiqueta retórica.\n"
        f"Etiquetas válidas: {labels}.\n"
        "Devuelve SOLO un JSON (sin Markdown, sin explicación), como un arreglo con la MISMA cantidad de entradas que párrafos.\n"
        "IMPORTANTE: NO incluyas el campo 'text' en la salida. NO devuelvas claves extra.\n"
        "Cada entrada debe ser un objeto con:\n"
        '  - "paragraph_index": entero (0..N-1)\n'
        '  - "label": una de las etiquetas válidas\n'
        '  - "confidence": número entre 0 y 1\n'
        "\n"
        "Entrada (JSON):\n"
        f"{input_json}\n"
    )


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
    """
    Extrae texto del primer candidato. Formato típico:
      candidates[0].content.parts[*].text
    """
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


def _parse_llm_json(text: str) -> Any:
    """
    Intenta parsear JSON aunque venga con cercas ```json ... ```.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(
        r"```(?:json)?\s*([\[{].*?[\]}])\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        return json.loads(fenced.group(1))

    # Fallback: tomar desde el primer '[' hasta el último ']' si existe, si no, '{' ... '}'
    start_list = text.find("[")
    end_list = text.rfind("]")
    if start_list != -1 and end_list != -1 and end_list > start_list:
        return json.loads(text[start_list : end_list + 1])

    start_obj = text.find("{")
    end_obj = text.rfind("}")
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return json.loads(text[start_obj : end_obj + 1])

    raise ValueError("No se pudo parsear JSON desde la respuesta del modelo.")


def _normalize_segments_output(parsed: Any, paragraphs: list[str]) -> list[dict[str, Any]]:
    """
    Acepta:
      - lista de objetos (uno por párrafo), o
      - dict con clave 'segments' que contiene esa lista.
    Siempre devuelve segmentos en orden y asegura paragraph_index/text correctos.
    """
    if isinstance(parsed, dict) and "segments" in parsed:
        items = parsed["segments"]
    else:
        items = parsed

    if not isinstance(items, list):
        raise TypeError("Salida del modelo inválida: se esperaba un arreglo JSON (o un objeto con 'segments').")

    # Intentar mapear por paragraph_index si existe y es usable
    by_index: dict[int, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("paragraph_index")
        if isinstance(idx, int) and 0 <= idx < len(paragraphs):
            by_index[idx] = item

    segments: list[dict[str, Any]] = []
    for i, para in enumerate(paragraphs):
        item = by_index.get(i)
        if item is None and i < len(items) and isinstance(items[i], dict):
            item = items[i]
        if item is None:
            raise ValueError(f"Salida del modelo no incluye el párrafo {i}.")

        label = str(item.get("label") or "").strip().upper()
        if label not in RHETORICAL_LABELS:
            raise ValueError(f"Label inválido para párrafo {i}: '{label}'. Debe ser uno de {RHETORICAL_LABELS}.")

        conf = item.get("confidence")
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = round(min(max(confidence, 0.0), 1.0), 2)

        segments.append(
            {
                "paragraph_index": i,
                "text": para,
                "label": label,
                "confidence": confidence,
            }
        )

    return segments


_ENCODER_CACHE: dict[str, tuple[Any, Any, str]] = {}


def _repo_root() -> Path:
    # .../app/backend/services/segmentation.py -> repo root is 3 parents above /app
    return Path(__file__).resolve().parents[3]


def _normalize_encoder_label(label: str) -> str:
    """
    El modelo fine-tuned usa 'RESU' pero el backend/UI usa 'RES'.
    Normalizamos para mantener compatibilidad sin tocar el frontend.
    """
    label = (label or "").strip().upper()
    if label == "RESU":
        return "RES"
    return label


def _load_encoder_model(encoder_variant: str):
    """
    Carga el encoder local de Task1 desde:
      - env TASK1_ENCODER_MODEL_PATH, o
      - env TASK1_ENCODER_<VARIANT>_MODEL_PATH, o
      - src/models/<variant> (por defecto)

    Requiere dependencias: torch, transformers, safetensors.
    """
    global _ENCODER_CACHE

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Para usar model='encoder' instala dependencias: pip install torch transformers safetensors"
        ) from e

    model_path = _resolve_task1_encoder_model_path(encoder_variant=encoder_variant)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo encoder en {model_path}. "
            "Define TASK1_ENCODER_MODEL_PATH o TASK1_ENCODER_<VARIANT>_MODEL_PATH "
            "apuntando al directorio del modelo."
        )

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
            "El modelo encoder de Task1 no contiene pesos.\n"
            f"Directorio: {model_path}\n"
            f"Archivos encontrados: {files}\n"
            "Se esperaba 'model.safetensors' (o 'pytorch_model.bin')."
        )

    cache_key = str(model_path.resolve())
    if cache_key in _ENCODER_CACHE:
        return _ENCODER_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    _ENCODER_CACHE[cache_key] = (tokenizer, model, device)
    return _ENCODER_CACHE[cache_key]


def _resolve_task1_encoder_model_path(encoder_variant: str) -> Path:
    """
    Resuelve el path del encoder para Task1.

    Fuentes (prioridad):
      1) TASK1_ENCODER_MLFLOW_MODEL_URI (recomendado si usas una sola variable por tarea)
         TASK1_ENCODER_<VARIANT>_MLFLOW_MODEL_URI
         e.g.:
           runs:/<run_id>/hf_model
           models:/scitext-task1-encoder/Production
      2) TASK1_ENCODER_MODEL_PATH (recomendado si usas una sola variable por tarea)
         TASK1_ENCODER_<VARIANT>_MODEL_PATH
      3) src/models/<variant> (default)
    """
    variant = (encoder_variant or "roberta").strip().lower()
    if variant not in {"roberta", "scibert"}:
        raise ValueError("encoder_variant inválido. Usa 'roberta' o 'scibert'.")

    mlflow_var = f"TASK1_ENCODER_{variant.upper()}_MLFLOW_MODEL_URI"
    path_var = f"TASK1_ENCODER_{variant.upper()}_MODEL_PATH"

    mlflow_uri = (os.environ.get("TASK1_ENCODER_MLFLOW_MODEL_URI") or "").strip()
    if not mlflow_uri:
        mlflow_uri = (os.environ.get(mlflow_var) or "").strip()
    if mlflow_uri:
        return _download_model_from_mlflow(mlflow_uri)

    model_dir = (os.environ.get("TASK1_ENCODER_MODEL_PATH") or "").strip()
    if not model_dir:
        model_dir = (os.environ.get(path_var) or "").strip()
    if not model_dir:
        default_dir = "roberta_bne_task1" if variant == "roberta" else "scibert_task1"
        model_dir = str(_repo_root() / "src" / "models" / default_dir)
    return Path(model_dir)


def _download_model_from_mlflow(model_uri: str) -> Path:
    """
    Descarga un artefacto de MLflow (S3 o registry) a un cache local y devuelve el directorio.
    Requiere: pip install mlflow boto3
    Variables útiles:
      - MLFLOW_TRACKING_URI (si usas runs:/ o models:/)
      - MODEL_CACHE_DIR (default: artifacts/model_cache)
      - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
      - MLFLOW_S3_ENDPOINT_URL (si usas MinIO)
    """
    if model_uri.strip().startswith("s3://"):
        cache_root = (os.environ.get("MODEL_CACHE_DIR") or str(_repo_root() / "artifacts" / "model_cache")).strip()
        cache_root_path = Path(cache_root)
        cache_root_path.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(model_uri.encode("utf-8")).hexdigest()[:12]
        dst = cache_root_path / f"task1_encoder_{key}"
        dst.mkdir(parents=True, exist_ok=True)
        if (dst / "config.json").exists() and (
            (dst / "model.safetensors").exists()
            or (dst / "pytorch_model.bin").exists()
            or (dst / "model.safetensors.index.json").exists()
            or (dst / "pytorch_model.bin.index.json").exists()
        ):
            return dst
        return download_hf_model_from_s3(model_uri, dst)

    try:
        import mlflow
        from mlflow.artifacts import download_artifacts
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Para cargar el modelo desde MLflow define TASK1_ENCODER_MLFLOW_MODEL_URI y "
            "instala dependencias: pip install mlflow boto3"
        ) from e

    tracking_uri = (os.environ.get("MLFLOW_TRACKING_URI") or "").strip()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    cache_root = (os.environ.get("MODEL_CACHE_DIR") or str(_repo_root() / "artifacts" / "model_cache")).strip()
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    key = hashlib.sha1(model_uri.encode("utf-8")).hexdigest()[:12]
    dst = cache_root_path / f"task1_encoder_{key}"
    dst.mkdir(parents=True, exist_ok=True)

    # If already downloaded, reuse (even if nested)
    if (dst / "config.json").exists() and ((dst / "model.safetensors").exists() or (dst / "pytorch_model.bin").exists()):
        return dst
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

    logger.info("Task1 encoder: descargando artefacto MLflow (%s) a %s", model_uri, str(dst))
    try:
        downloaded_path = download_artifacts(artifact_uri=model_uri, dst_path=str(dst))
    except Exception as e:  # noqa: BLE001
        fallback_s3_uri = _runs_uri_to_s3_prefix(model_uri)
        if fallback_s3_uri and _is_s3_listbucket_access_denied(e):
            logger.warning(
                "Task1 encoder: MLflow falló por ListBucket denegado; usando fallback S3 sin listado. "
                "model_uri=%s s3_uri=%s",
                model_uri,
                fallback_s3_uri,
            )
            return download_hf_model_from_s3(fallback_s3_uri, dst)
        raise
    downloaded = Path(downloaded_path)

    # download_artifacts might return a file or folder; we want the folder containing config/model.
    if downloaded.is_file():
        downloaded = downloaded.parent

    # If MLflow created a nested folder, try to locate typical HF files.
    if not (downloaded / "config.json").exists():
        candidates = list(downloaded.rglob("config.json"))
        if candidates:
            downloaded = candidates[0].parent

    if not (downloaded / "config.json").exists():
        raise RuntimeError(f"No se encontró config.json dentro del artefacto descargado: {downloaded}")

    return downloaded


def _call_encoder_model(text: str, encoder_variant: str = "scibert") -> dict:
    """
    Segmentación real con el encoder RoBERTa local.
    Devuelve {segments:[...], stats:{...}} como el mock.
    """
    import math

    tokenizer, model, device = _load_encoder_model(encoder_variant=encoder_variant)
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Falta torch para inferencia del encoder.") from e

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        paragraphs = [text.strip()]

    started = time.perf_counter()
    segments: list[dict[str, Any]] = []

    batch_size = int(os.environ.get("TASK1_ENCODER_BATCH_SIZE") or "8")
    batch_size = max(1, min(batch_size, 64))

    id2label = getattr(model.config, "id2label", None) or {}

    with torch.no_grad():
        for offset in range(0, len(paragraphs), batch_size):
            batch = paragraphs[offset : offset + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confs, preds = torch.max(probs, dim=-1)

            for j, para in enumerate(batch):
                idx = offset + j
                pred_id = int(preds[j].item())
                raw_label = id2label.get(str(pred_id)) or id2label.get(pred_id) or str(pred_id)
                label = _normalize_encoder_label(str(raw_label))
                if label not in RHETORICAL_LABELS:
                    label = "INTRO" if idx == 0 else "CONC" if idx == (len(paragraphs) - 1) else "BACK"

                confidence = float(confs[j].item())
                if math.isnan(confidence):
                    confidence = 0.0
                confidence = round(min(max(confidence, 0.0), 1.0), 2)

                segments.append(
                    {
                        "paragraph_index": idx,
                        "text": para,
                        "label": label,
                        "confidence": confidence,
                    }
                )

    elapsed = round(time.perf_counter() - started, 2)
    word_count = sum(len(p.split()) for p in paragraphs)
    avg_conf = round(sum(s["confidence"] for s in segments) / len(segments), 3) if segments else 0.0

    return {
        "segments": segments,
        "stats": {
            "total_paragraphs": len(paragraphs),
            "total_words": word_count,
            "avg_confidence": avg_conf,
            "time_seconds": elapsed,
        },
    }
