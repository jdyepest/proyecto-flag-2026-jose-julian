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
COPY data_lake/*.dvc /app/data_lake/

# Force no-SCM mode locally and pull tracked data from DVC remote
RUN printf "[core]\n    no_scm = true\n" > /app/.dvc/config.local
RUN sed -i '/profile = default/d' /app/.dvc/config


# Stage 2: runtime image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY app/backend/requirements.txt /app/app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/app/backend/requirements.txt

COPY requirments.txt /app/app/backend/requirements_global.txt
RUN pip install --no-cache-dir -r /app/app/backend/requirements_global.txt

COPY app/backend /app/app/backend
COPY app/frontend /app/app/frontend
COPY .env /app/.env

# Copy DVC data from builder

EXPOSE 5000
ENV PORT=5000

CMD ["python", "app/backend/main.py"]
