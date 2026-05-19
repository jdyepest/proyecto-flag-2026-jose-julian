#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/mlruns_recovered"
PORT="${MLFLOW_UI_PORT:-5006}"
PREPARE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-only)
      PREPARE_ONLY=1
      shift
      ;;
    --dir)
      TARGET_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      echo "Usage: $0 [--prepare-only] [--dir /abs/path] [--port 5006]"
      exit 1
      ;;
  esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "[error] directory not found: $TARGET_DIR"
  exit 1
fi

echo "[ok] backend store: file://$TARGET_DIR"
echo "[ok] experiments found:"
find "$TARGET_DIR" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort

if [[ "$PREPARE_ONLY" -eq 1 ]]; then
  echo "[ok] prepare-only mode, not launching MLflow UI"
  exit 0
fi

if ! command -v mlflow >/dev/null 2>&1; then
  echo "[error] mlflow command not found in PATH"
  echo "Run with your env active, e.g.: source .venv/bin/activate"
  exit 1
fi

echo "[run] http://localhost:${PORT}"
export MLFLOW_SERVER_ALLOWED_HOSTS="${MLFLOW_SERVER_ALLOWED_HOSTS:-*}"
export MLFLOW_SERVER_CORS_ALLOWED_ORIGINS="${MLFLOW_SERVER_CORS_ALLOWED_ORIGINS:-*}"
export MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE="${MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE:-true}"
exec mlflow ui --backend-store-uri "file://${TARGET_DIR}" --host 0.0.0.0 --port "$PORT"
