"""API REST de inferencia (FastAPI), SOLO en modo de demostración.

El único modelo disponible fue rechazado por el quality gate (precision en test por debajo del umbral), así que:
  - APP_MODE=demo        → carga el modelo y responde marcando cada salida como rechazada/demo;
  - APP_MODE=production  → el arranque se rechaza: no existe un modelo aprobado;
  - modo ausente u otro  → el arranque se rechaza (nunca se habilita la inferencia por defecto).

Arranque:  uvicorn hdfs_anomaly.api:create_app --factory --host 127.0.0.1 --port 8000
La entrada es una traza YA procesada (conteos E1–E29); no se procesan logs crudos.
"""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field, create_model

from hdfs_anomaly.config import REPO_ROOT
from hdfs_anomaly.data import EVENT_COLS

MODES = ("demo", "production")
FROZEN_PREFIX = "CONGELADA"
MAX_COUNT = 1_000_000  # límite de cordura por evento; evita desbordes de int64 con entradas absurdas
MODEL_NAME = "hdfs-anomaly-detector"


class StartupRefused(RuntimeError):
    """El servicio no debe arrancar (modo no permitido o artefacto no confiable)."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- Contrato de entrada ---------------------------------------------------------------
# Las 29 características son obligatorias, enteros >= 0 en modo estricto (sin coerción de
# "3", 3.0 ni true) y no se admite ningún campo adicional (BlockId, Label, Type...).
TraceFeatures = create_model(
    "TraceFeatures",
    __config__=ConfigDict(extra="forbid", strict=True,
                          json_schema_extra={"description": "Conteos de eventos E1–E29 de una traza de bloque"}),
    **{c: (Annotated[int, Field(ge=0, le=MAX_COUNT)], ...) for c in EVENT_COLS},
)


class Prediction(BaseModel):
    prediction: str
    is_anomaly: bool
    anomaly_score: float
    threshold: float
    model_version: str
    model_sha256: str
    model_status: str
    gate: str
    demo_only: bool
    mode: str


# --- Carga verificada del modelo --------------------------------------------------------
def load_verified(model_dir: Path, gate_record: Path, trusted_root: Path) -> dict:
    """Comprueba origen, congelación, registro del gate y SHA-256 ANTES de deserializar.
    El hash garantiza integridad, no confianza: por eso además solo se aceptan artefactos
    situados dentro de la carpeta de modelos del propio repositorio/imagen (trusted_root)."""
    model_dir, trusted_root = model_dir.resolve(), trusted_root.resolve()
    if not model_dir.is_relative_to(trusted_root):
        raise StartupRefused(f"{model_dir} está fuera de la carpeta de modelos confiable {trusted_root}")
    spec_path, model_path = model_dir / "spec.json", model_dir / "model.joblib"
    if not spec_path.is_file() or not model_path.is_file():
        raise StartupRefused(f"Faltan spec.json o model.joblib en {model_dir}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith(FROZEN_PREFIX):
        raise StartupRefused("La especificación del modelo no está congelada")
    if spec.get("features") != EVENT_COLS:
        raise StartupRefused("Las características de la especificación no coinciden con E1–E29 en orden")
    sha = _sha(model_path)
    if sha != spec["model_sha256"]:
        raise StartupRefused("El SHA-256 del artefacto no coincide con la especificación congelada")

    gate = {"gate": "not_evaluated", "model_status": "not_evaluated", "checks": None, "test": None,
            "freeze_commit": None}
    if gate_record.is_file():
        ev = json.loads(gate_record.read_text(encoding="utf-8"))
        if ev.get("status") != "EVALUADO" or ev["integrity_verified"]["model_sha256"] != sha \
                or ev["frozen_spec_sha256"] != _sha(spec_path):
            raise StartupRefused("El registro del gate no corresponde a este modelo congelado")
        passed = bool(ev["gate_passed"])
        gate = {"gate": "passed" if passed else "failed", "model_status": "gate_passed" if passed else "rejected",
                "checks": ev["gate_checks"], "freeze_commit": ev["freeze_commit"],
                "test": {k: ev["test"][k] for k in ("precision", "recall", "f1", "fpr", "pr_auc")}}

    model = joblib.load(model_path)  # solo tras verificar origen y SHA-256
    return {"model": model, "spec": spec, "sha": sha, **gate}


def _metrics(registry: CollectorRegistry, threshold: float) -> dict:
    """Métricas operativas. Etiquetas de cardinalidad acotada: plantilla de ruta
    (nunca la URL real), método, código y clase predicha. Sin BlockId, sin características,
    sin rutas de archivos ni datos de la petición."""
    score_buckets = sorted({0.001, 0.01, 0.05, round(threshold, 4), 0.25, 0.5, 0.75, 0.9, 0.99, 0.999})
    return {
        "requests": Counter("hdfs_api_requests_total", "Peticiones HTTP por endpoint, método y código",
                            ["endpoint", "method", "status"], registry=registry),
        "latency": Histogram("hdfs_api_request_duration_seconds", "Latencia de las peticiones por endpoint",
                             ["endpoint"], buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
                             registry=registry),
        "errors": Counter("hdfs_api_errors_total", "Respuestas de error por endpoint y tipo (client=4xx, server=5xx)",
                          ["endpoint", "kind"], registry=registry),
        "predictions": Counter("hdfs_predictions_total", "Predicciones emitidas por clase", ["prediction"],
                               registry=registry),
        "score": Histogram("hdfs_prediction_score", "Distribución de la puntuación de anomalía",
                           buckets=score_buckets, registry=registry),
        "model": Gauge("hdfs_model_info", "Modelo servido (valor constante 1)",
                       ["model_version", "model_status", "gate", "mode"], registry=registry),
    }


def create_app(mode: str | None = None, model_dir: Path | None = None, gate_record: Path | None = None,
               trusted_root: Path | None = None) -> FastAPI:
    """Fábrica de la aplicación. Sin argumentos lee APP_MODE y MODEL_DIR del entorno.
    trusted_root solo se puede cambiar desde código (pruebas), nunca por variable de entorno."""
    mode = os.environ.get("APP_MODE", "") if mode is None else mode
    if mode not in MODES:
        raise StartupRefused(f"APP_MODE={mode!r} no es válido; use 'demo' (único modo disponible) o 'production'")
    trusted_root = trusted_root or REPO_ROOT / "models"
    model_dir = model_dir or Path(os.environ.get("MODEL_DIR", trusted_root / "candidate"))
    gate_record = gate_record or REPO_ROOT / "reports" / "train" / "test_evaluation.json"
    m = load_verified(Path(model_dir), Path(gate_record), trusted_root)
    if mode == "production" and m["gate"] != "passed":
        raise StartupRefused("APP_MODE=production rechazado: no existe un modelo aprobado por el quality gate "
                             f"(estado del modelo: {m['model_status']}, gate: {m['gate']}). Use APP_MODE=demo.")

    spec, model, threshold = m["spec"], m["model"], float(m["spec"]["threshold"])
    demo_only = mode == "demo"
    version = f"{spec['family']}-frozen-{(m['freeze_commit'] or 'unknown')[:7]}"
    notice = {"model_status": m["model_status"], "gate": m["gate"], "demo_only": demo_only, "mode": mode}

    app = FastAPI(title="HDFS anomaly detector (demo)" if demo_only else "HDFS anomaly detector",
                  description="Clasifica una traza HDFS ya procesada (conteos E1–E29). "
                              + ("MODELO RECHAZADO POR EL QUALITY GATE: solo para demostración." if demo_only else ""),
                  version="0.6.0")
    registry = CollectorRegistry()
    met = _metrics(registry, threshold)
    for label in ("normal", "anomaly"):
        met["predictions"].labels(label)
    met["model"].labels(version, m["model_status"], m["gate"], mode).set(1)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start, status = time.perf_counter(), 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            endpoint = getattr(request.scope.get("route"), "path", "unmatched")  # plantilla, nunca la URL
            met["requests"].labels(endpoint, request.method, str(status)).inc()
            met["latency"].labels(endpoint).observe(time.perf_counter() - start)
            if status >= 400:
                met["errors"].labels(endpoint, "client" if status < 500 else "server").inc()

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health() -> dict:
        # Disponibilidad técnica del servicio; NO implica que el modelo esté aprobado.
        return {"status": "ok", "model_loaded": True, "mode": mode,
                "model_approved_for_production": m["gate"] == "passed",
                "note": "Disponibilidad técnica del servicio; no indica aprobación del modelo."}

    @app.get("/model/info")
    def model_info() -> dict:
        return {"model_name": MODEL_NAME, "model_version": version,
                "algorithm": {"family": spec["family"], "pipeline": "log1p -> StandardScaler -> LogisticRegression"
                              if spec["family"] == "lr" else spec["family"], "params": spec["params"]},
                "model_sha256": m["sha"], "threshold": threshold, "features": EVENT_COLS,
                "training_code_commit": spec["training_code_commit"], "freeze_commit": m["freeze_commit"],
                "gate_criteria": spec["gate"], "gate_checks": m["checks"], "test_metrics": m["test"], **notice}

    @app.post("/predict", response_model=Prediction)
    def predict(trace: TraceFeatures) -> Prediction:
        row = pd.DataFrame([[getattr(trace, c) for c in EVENT_COLS]], columns=EVENT_COLS)  # orden del modelo
        score = float(model.predict_proba(row)[0, 1])
        anomaly = score >= threshold
        met["predictions"].labels("anomaly" if anomaly else "normal").inc()
        met["score"].observe(score)
        return Prediction(prediction="anomaly" if anomaly else "normal", is_anomaly=anomaly, anomaly_score=score,
                          threshold=threshold, model_version=version, model_sha256=m["sha"], **notice)

    return app
