# Manual de Instalación — SciText-ES (Local y Docker)

**Versión:** 1.2  \
**Audiencia:** Equipo técnico que instala la app localmente (con o sin Docker).  \
**Alcance:** Instalación completa de backend + frontend estático + proxy LLM + datos/modelos vía DVC o S3. MLflow queda como opción de compatibilidad.

---

## 1. Ruta base de trabajo

Trabaja siempre desde la raíz del repositorio. En adelante, `REPO/` se refiere a esa ruta.

```bash
cd /ruta/maia-proyecto-desarrollo-soluciones
```

---

## 2. Requisitos

### 2.1 Sin Docker
- Python 3.10+ (recomendado 3.11)
- `pip`
- Git
- DVC (incluido en `requirments.txt`)
- (Opcional) MLflow solo si usarás modelos `encoder` vía `runs:/...` o `models:/...`
- (Opcional) Ollama o el proxy OpenRouter (`app/ollama`) si usarás `model=llm`

### 2.2 Con Docker
- Docker Desktop (o Docker Engine + Compose)
- AWS credentials disponibles en el entorno para el `dvc pull` del build

---

## 3. Configuración de `.env` (muy importante)

El archivo `.env` vive en la raíz del repo (`REPO/.env`). Si no existe, créalo. Evita subir credenciales al repositorio.

Ejemplo mínimo (valores de ejemplo, reemplaza los tuyos):

```env
PORT=5000
LOG_LEVEL=INFO
LOG_FILE=artifacts/logs/backend.log

# Rutas de servicios (ver tabla abajo)
MLFLOW_TRACKING_URI=http://127.0.0.1:5006
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Gemini (modelo api)
GEMINI_API_KEY=TU_API_KEY
GEMINI_MODEL=gemini-2.5-flash-lite

# OpenRouter (si usas proxy LLM app/ollama)
OPENROUTER_API_KEY=TU_API_KEY
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Encoder (Task1/Task2) recomendado: una variable por tarea
TASK1_ENCODER_MLFLOW_MODEL_URI=s3://<bucket>/<prefix-task1>/hf_model
TASK2_ENCODER_MLFLOW_MODEL_URI=s3://<bucket>/<prefix-task2>/hf_model

# Alternativa local después de dvc pull
# TASK1_ENCODER_MODEL_PATH=Modelos/task1/.../artifacts/model
# TASK2_ENCODER_MODEL_PATH=Modelos/task2/.../artifacts/hf_model

# AWS (necesario para DVC y/o S3)
AWS_ACCESS_KEY_ID=TU_ACCESS_KEY
AWS_SECRET_ACCESS_KEY=TU_SECRET
AWS_SESSION_TOKEN=TU_SESSION_TOKEN
AWS_DEFAULT_REGION=us-east-1
```

### Cambios de rutas en `.env` según entorno

| Variable | Local (sin Docker) | Docker Compose |
| --- | --- | --- |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5006` | `http://mlflow:5006` |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | `http://ollama:11434` |

Notas:
`http://mlflow:5006` y `http://ollama:11434` funcionan dentro de la red Docker. Desde tu navegador siempre usarás `http://localhost:5000`.

---

## 4. Instalación SIN Docker (paso a paso)

1. Crear y activar entorno virtual.

```bash
cd /ruta/maia-proyecto-desarrollo-soluciones
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Instalar dependencias con rutas correctas.

```bash
pip install -r app/backend/requirements.txt
pip install -r requirments.txt
```

Nota: el archivo global se llama `requirments.txt` (sin la segunda "e").

3. Descargar datos y modelos con DVC.

```bash
dvc pull data_lake/clean_parquet.dvc data_lake/datasets.dvc Modelos.dvc
```

Si `dvc pull` falla, verifica que `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN` y `AWS_DEFAULT_REGION` estén definidos.

Si no puedes descargar desde DVC o S3, usa los respaldos manuales en Drive y materializa la carpeta `Modelos/` localmente:

- Respaldo 1: [Drive - runs/modelos (1)](https://drive.google.com/drive/folders/1udVwPnN5Ep80qQ7ck2DjayhqiktzHWat?usp=sharing)
- Respaldo 2: [Drive - runs/modelos (2)](https://drive.google.com/drive/folders/1BWNaBJQ4-ktDpig4SMOlpVp0NffhKmTg?usp=sharing)

4. Levantar servicios opcionales (según el modelo que uses).

Si usarás `model=encoder` con `s3://...` directo o con `TASK*_ENCODER_MODEL_PATH`, no necesitas levantar MLflow.

Solo si usarás URIs `runs:/...` o `models:/...`, inicia MLflow en otra terminal:

```bash
mlflow server \
  --backend-store-uri sqlite:///./mlflow.db \
  --default-artifact-root s3://<tu-bucket>/mlflow \
  --host 0.0.0.0 --port 5006
```

Si usarás `model=llm` con el proxy OpenRouter, inicia el servicio `app/ollama` en otra terminal:

```bash
cd app/ollama
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn httpx python-dotenv
uvicorn app:app --host 0.0.0.0 --port 11434
```

5. Levantar el backend (sirve también el frontend).

```bash
cd /ruta/maia-proyecto-desarrollo-soluciones
python app/backend/main.py
```

Backend en: `http://localhost:5000`

Nota: si `TASK1_ENCODER_MLFLOW_MODEL_URI` o `TASK2_ENCODER_MLFLOW_MODEL_URI` apuntan a `s3://...`, la primera inferencia descargará los artefactos al cache local y puede tardar más de lo normal.

---

## 5. Instalación CON Docker (local)

1. Ajusta `REPO/.env` con las variables y rutas correctas para Docker.

2. Verifica que las credenciales AWS estén definidas (necesarias para el `dvc pull` del build o para descargar desde S3).

3. Construye y levanta los servicios.

```bash
docker compose up --build
```

Servicios expuestos:
- Backend: `http://localhost:5000`
- MLflow: `http://localhost:5006`
- Proxy LLM: `http://localhost:11434`

---

## 6. Verificación rápida

1. Abre `http://localhost:5000`.
2. Pega un texto de prueba y ejecuta el análisis.
3. Si usas API directa, prueba `POST /api/analyze` desde Postman.

---

## 7. Solución de problemas

- `dvc pull` falla: revisa credenciales AWS, acceso al bucket y, si hace falta, usa los respaldos de Drive para poblar `Modelos/`.
- `backend` responde 500 en `model=encoder`: revisa `TASK1_ENCODER_MLFLOW_MODEL_URI` y `TASK2_ENCODER_MLFLOW_MODEL_URI`, o confirma que `TASK1_ENCODER_MODEL_PATH` y `TASK2_ENCODER_MODEL_PATH` apunten a carpetas válidas dentro de `Modelos/`.
- `model=llm` no responde: revisa `OLLAMA_BASE_URL` y que el proxy esté activo.
- `model=api` falla: revisa `GEMINI_API_KEY`.
- `MLflow no responde`: revisa `MLFLOW_TRACKING_URI` y el puerto `5006` solo si decidiste usar `runs:/...` o `models:/...`.

---

**Fin del manual.**
