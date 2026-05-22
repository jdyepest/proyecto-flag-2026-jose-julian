# Stage 1: fetch DVC data at build time
FROM python:3.11-slim AS dvc

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN
ARG AWS_DEFAULT_REGION

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
    AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    DVC_NO_SCM=1

WORKDIR /app

RUN pip install --no-cache-dir "dvc[s3]"

COPY .dvc /app/.dvc
COPY .dvcignore /app/.dvcignore

# Force no-SCM mode locally and pull tracked data from DVC remote
RUN printf "[core]\n    no_scm = true\n" > /app/.dvc/config.local
RUN sed -i '/profile = default/d' /app/.dvc/config


# Stage 2: runtime image
FROM python:3.11-slim

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_SESSION_TOKEN
ARG AWS_DEFAULT_REGION
ARG AWS_S3_ENDPOINT_URL
ARG MLFLOW_S3_ENDPOINT_URL
ARG MODEL_CACHE_DIR=artifacts/model_cache
ARG TASK1_ENCODER_MLFLOW_MODEL_URI
ARG TASK1_ENCODER_ROBERTA_MLFLOW_MODEL_URI
ARG TASK1_ENCODER_SCIBERT_MLFLOW_MODEL_URI
ARG TASK2_ENCODER_MLFLOW_MODEL_URI
ARG TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI
ARG TASK2_ENCODER_SCIBERT_MLFLOW_MODEL_URI

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/backend/requirements.txt /app/app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/app/backend/requirements.txt

# Runtime deps used by backend services (encoder/mlflow/S3).
# Avoid installing the root "requirments.txt" because it includes many
# notebook/dev packages that are not required in production and often fail
# in slim container builds.
RUN pip install --no-cache-dir \
    boto3 \
    mlflow \
    "transformers<5" \
    torch \
    safetensors

COPY app/backend /app/app/backend
COPY app/frontend /app/app/frontend
COPY .env /app/.env

# Si los modelos están configurados como s3://..., precárgalos en build para
# evitar la descarga costosa durante la primera request.
RUN AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN} \
    AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    AWS_S3_ENDPOINT_URL=${AWS_S3_ENDPOINT_URL} \
    MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL} \
    MODEL_CACHE_DIR=${MODEL_CACHE_DIR} \
    TASK1_ENCODER_MLFLOW_MODEL_URI=${TASK1_ENCODER_MLFLOW_MODEL_URI} \
    TASK1_ENCODER_ROBERTA_MLFLOW_MODEL_URI=${TASK1_ENCODER_ROBERTA_MLFLOW_MODEL_URI} \
    TASK1_ENCODER_SCIBERT_MLFLOW_MODEL_URI=${TASK1_ENCODER_SCIBERT_MLFLOW_MODEL_URI} \
    TASK2_ENCODER_MLFLOW_MODEL_URI=${TASK2_ENCODER_MLFLOW_MODEL_URI} \
    TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI=${TASK2_ENCODER_ROBERTA_MLFLOW_MODEL_URI} \
    TASK2_ENCODER_SCIBERT_MLFLOW_MODEL_URI=${TASK2_ENCODER_SCIBERT_MLFLOW_MODEL_URI} \
    python /app/app/backend/scripts/preload_models.py

# Copy DVC data from builder

EXPOSE 5000
ENV PORT=5000

CMD ["python", "app/backend/main.py"]
