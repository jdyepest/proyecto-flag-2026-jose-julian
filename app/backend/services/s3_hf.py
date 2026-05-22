import json
import logging
import os
from pathlib import Path
from typing import Iterable


logger = logging.getLogger(__name__)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """
    Parse s3://bucket/key... into (bucket, key_prefix).
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"URI no es s3://...: {uri}")
    rest = uri[len("s3://") :]
    if "/" not in rest:
        return rest, ""
    bucket, key = rest.split("/", 1)
    return bucket, key.strip("/")


def _download_object(s3, bucket: str, key: str, dst_path: Path) -> bool:
    """
    Download an S3 object to dst_path.
    Returns True if downloaded, False if object does not exist.
    Raises for permission errors.
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        body = resp["Body"]
        with dst_path.open("wb") as f:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        return True
    except Exception as e:  # noqa: BLE001
        if dst_path.exists():
            dst_path.unlink(missing_ok=True)
        # Best-effort: detect missing object without requiring ListBucket.
        code = getattr(getattr(e, "response", None), "get", lambda _k, _d=None: _d)("Error", {}).get("Code")  # type: ignore[attr-defined]
        if code in {"404", "NoSuchKey", "NotFound"}:
            return False
        # botocore ClientError has .response dict
        resp = getattr(e, "response", None)
        if isinstance(resp, dict):
            err = (resp.get("Error") or {}).get("Code")
            if err in {"404", "NoSuchKey", "NotFound"}:
                return False
        raise


def _try_download_any(s3, bucket: str, prefix: str, names: Iterable[str], dst_dir: Path) -> str | None:
    for name in names:
        key = f"{prefix}/{name}" if prefix else name
        if _download_object(s3, bucket, key, dst_dir / name):
            return name
    return None


def download_hf_model_from_s3(prefix_uri: str, dst_dir: str | Path) -> Path:
    """
    Download a HuggingFace model folder from an S3 prefix WITHOUT listing the bucket.

    This is useful when IAM policies deny s3:ListBucket but allow s3:GetObject.
    Supports sharded checkpoints if an index json is present.
    """
    try:
        import boto3
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Para descargar desde S3 instala boto3: pip install boto3") from e

    bucket, prefix = _parse_s3_uri(prefix_uri)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    endpoint_url = (os.environ.get("AWS_S3_ENDPOINT_URL") or os.environ.get("MLFLOW_S3_ENDPOINT_URL") or "").strip()
    if endpoint_url:
        s3 = boto3.client("s3", endpoint_url=endpoint_url)
    else:
        s3 = boto3.client("s3")

    logger.info("Descargando modelo HF desde S3 (sin ListBucket): s3://%s/%s -> %s", bucket, prefix, str(dst))

    # Required-ish config/tokenizer files (some tokenizers don't have all of them).
    required = [
        "config.json",
        "tokenizer.json",
    ]
    optional = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.model",
        "generation_config.json",
        "training_args.bin",
    ]

    for name in required:
        key = f"{prefix}/{name}" if prefix else name
        ok = _download_object(s3, bucket, key, dst / name)
        if not ok:
            raise FileNotFoundError(f"No se encontró {name} en s3://{bucket}/{prefix}")

    for name in optional:
        key = f"{prefix}/{name}" if prefix else name
        _download_object(s3, bucket, key, dst / name)

    # Weights: try single-file first, then index-based sharded.
    weights = _try_download_any(
        s3,
        bucket,
        prefix,
        names=["model.safetensors", "pytorch_model.bin"],
        dst_dir=dst,
    )
    if weights:
        return dst

    index_name = _try_download_any(
        s3,
        bucket,
        prefix,
        names=["model.safetensors.index.json", "pytorch_model.bin.index.json"],
        dst_dir=dst,
    )
    if not index_name:
        files = sorted([p.name for p in dst.iterdir() if p.is_file()])
        raise FileNotFoundError(
            "No se encontraron pesos del modelo en el prefix S3.\n"
            f"s3://{bucket}/{prefix}\n"
            f"Descargado: {files}\n"
            "Se esperaba model.safetensors (o pytorch_model.bin) o un archivo *.index.json."
        )

    index_data = json.loads((dst / index_name).read_text(encoding="utf-8"))
    weight_map = index_data.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    if not shard_names:
        raise ValueError(f"Index sin weight_map válido: {index_name}")

    for shard in shard_names:
        key = f"{prefix}/{shard}" if prefix else shard
        ok = _download_object(s3, bucket, key, dst / shard)
        if not ok:
            raise FileNotFoundError(f"Shard faltante según {index_name}: s3://{bucket}/{prefix}/{shard}")

    return dst
