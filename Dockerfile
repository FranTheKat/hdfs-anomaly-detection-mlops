# Imagen de DEMOSTRACIÓN del servicio de inferencia. El modelo incluido fue
# RECHAZADO por el quality gate: la imagen solo arranca con APP_MODE=demo y no se publica.
#   docker build -t hdfs-anomaly:demo-local .
# Base fijada por etiqueta y digest (índice multiplataforma consultado el 2026-11-23).
FROM python:3.14.7-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Dependencias fijadas (bloqueo completo generado con uv a partir de requirements-api.in).
COPY requirements-api.txt .
RUN pip install --no-compile -r requirements-api.txt \
    && useradd --system --uid 10001 --no-create-home app

# Solo código, artefacto congelado con su especificación y registro del gate (ver .dockerignore):
# sin datasets, sin MLflow, sin secretos.
COPY src/ src/
COPY models/candidate/ models/candidate/
COPY reports/train/test_evaluation.json reports/train/test_evaluation.json

USER app
EXPOSE 8000

# Sin APP_MODE por defecto: si no se indica, el arranque se rechaza.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"]

# La fábrica verifica origen, congelación, registro del gate y SHA-256 antes de cargar el modelo.
CMD ["uvicorn", "hdfs_anomaly.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
