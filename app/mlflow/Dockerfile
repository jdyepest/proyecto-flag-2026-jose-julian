FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir mlflow boto3 python-dotenv

COPY .env /app/.env

EXPOSE 5006

CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:////app/mlflow.db --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT:-s3://bucket-artifacts-models-2026-03/mlflow} --host 0.0.0.0 --port 5006"]
