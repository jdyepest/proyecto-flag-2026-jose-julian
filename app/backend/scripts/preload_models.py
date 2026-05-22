"""
Precarga modelos encoder en build/startup cuando están configurados como s3://...

Este script evita que la primera inferencia pague el costo de descargar
artefactos HF desde S3. Solo resuelve y descarga URIs S3; cualquier ruta local,
models:/ o runs:/ se omite en build porque normalmente dependen de servicios
externos de runtime (por ejemplo, MLflow).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    here = Path(__file__).resolve()
    backend_root = here.parents[1]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))


_bootstrap_path()

from services.contributions import _resolve_task2_encoder_model_path
from services.segmentation import _resolve_task1_encoder_model_path


logging.basicConfig(level=(os.environ.get("LOG_LEVEL") or "INFO").upper())
logger = logging.getLogger(__name__)


def _resolve_first_env(candidates: list[str]) -> str:
    for name in candidates:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


def _should_preload(uri: str) -> bool:
    return uri.startswith("s3://")


def _preload_task1() -> None:
    for variant in ("roberta", "scibert"):
        uri = _resolve_first_env(
            [
                "TASK1_ENCODER_MLFLOW_MODEL_URI",
                f"TASK1_ENCODER_{variant.upper()}_MLFLOW_MODEL_URI",
            ]
        )
        if not _should_preload(uri):
            if uri:
                logger.info("Task1 %s: skip preload (uri no es s3): %s", variant, uri)
            continue
        path = _resolve_task1_encoder_model_path(variant)
        logger.info("Task1 %s: modelo precargado en %s", variant, path)


def _preload_task2() -> None:
    for variant in ("roberta", "scibert"):
        uri = _resolve_first_env(
            [
                "TASK2_ENCODER_MLFLOW_MODEL_URI",
                f"TASK2_ENCODER_{variant.upper()}_MLFLOW_MODEL_URI",
            ]
        )
        if not _should_preload(uri):
            if uri:
                logger.info("Task2 %s: skip preload (uri no es s3): %s", variant, uri)
            continue
        path = _resolve_task2_encoder_model_path(variant)
        logger.info("Task2 %s: modelo precargado en %s", variant, path)


def main() -> int:
    logger.info("Iniciando precarga de modelos encoder desde S3")
    _preload_task1()
    _preload_task2()
    logger.info("Precarga de modelos finalizada")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
