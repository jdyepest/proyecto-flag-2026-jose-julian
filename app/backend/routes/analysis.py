"""
Rutas de análisis: POST /api/analyze
"""

import uuid
from flask import Blueprint, request, jsonify

from services.segmentation import analyze_segments
from services.contributions import analyze_contributions, task2_encoder_is_configured
from services.models import MODELS

analysis_bp = Blueprint("analysis", __name__)

# Almacén en memoria de análisis (reemplazable por DB)
_analysis_store: dict[str, dict] = {}


@analysis_bp.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analiza un documento científico.
    ---
    tags:
      - analysis
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              example: "En este trabajo presentamos...\n\nNuestra metodología..."
            model:
              type: string
              enum: [encoder, llm, api]
              default: encoder
            tasks:
              type: array
              items:
                type: string
                enum: [segmentation, contributions]
              example: ["segmentation", "contributions"]
            encoder_variant:
              type: string
              enum: [roberta, scibert]
              default: scibert
    responses:
      200:
        description: Resultado del análisis
        schema:
          type: object
          properties:
            id: {type: string}
            model: {type: string}
            model_name: {type: string}
            segmentation:
              type: object
              nullable: true
            contributions:
              type: object
              nullable: true
      400:
        description: Error de validación
        schema:
          type: object
          properties:
            error: {type: string}
    """
    body = request.get_json(force=True)

    text = (body.get("text") or "").strip()
    model = body.get("model", "encoder")
    tasks = body.get("tasks", ["segmentation", "contributions"])
    encoder_variant = (body.get("encoder_variant") or "scibert").strip().lower()

    # Validaciones básicas
    if not text:
        return jsonify({"error": "El campo 'text' es requerido."}), 400
    if len(text) < 50:
        return jsonify({"error": "El texto es demasiado corto (mín. 50 caracteres)."}), 400
    if model not in MODELS:
        return jsonify({"error": f"Modelo '{model}' no válido. Opciones: {list(MODELS.keys())}"}), 400
    if encoder_variant not in {"roberta", "scibert"}:
        return jsonify({"error": "encoder_variant no válido. Opciones: ['roberta','scibert']"}), 400
    if model == "encoder" and "contributions" in tasks and not task2_encoder_is_configured(encoder_variant):
        return jsonify(
            {
                "error": (
                    "Task2 encoder no configurado para contribuciones. "
                    "Define TASK2_ENCODER_<VARIANT>_MLFLOW_MODEL_URI (runs:/.../hf_model) "
                    "o usa model='llm'/'api'."
                )
            }
        ), 400

    analysis_id = str(uuid.uuid4())
    result: dict = {
        "id": analysis_id,
        "model": model,
        "model_name": MODELS[model]["name"],
        "segmentation": None,
        "contributions": None,
    }

    # Tarea 1: Segmentación retórica
    segmentation_data = None
    if "segmentation" in tasks:
        segmentation_data = analyze_segments(text, model, encoder_variant=encoder_variant)
        result["segmentation"] = segmentation_data

    # Tarea 2: Extracción de contribuciones (requiere segmentos)
    if "contributions" in tasks:
        segments = (
            segmentation_data["segments"]
            if segmentation_data
            else analyze_segments(text, model, encoder_variant=encoder_variant)["segments"]
        )
        result["contributions"] = analyze_contributions(segments, model, encoder_variant=encoder_variant)

    # Guardar en memoria para /api/compare
    _analysis_store[analysis_id] = {
        "text": text,
        "model": model,
        "encoder_variant": encoder_variant,
        "result": result,
    }

    return jsonify(result), 200


def get_analysis_store():
    """Expone el store para uso en otras rutas."""
    return _analysis_store
