"""Pruebas de la API en modo demo. Usan el modelo congelado versionado
(models/candidate) y el registro de test versionado; las entradas son SINTÉTICAS o patrones
extraídos SOLO de la partición train real (nunca de validation ni de test)."""
import json
import os
import shutil
import subprocess
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hdfs_anomaly import api as A
from hdfs_anomaly.config import REPO_ROOT
from hdfs_anomaly.data import EVENT_COLS

MODELS = REPO_ROOT / "models"
SPEC = json.loads((MODELS / "candidate/spec.json").read_text(encoding="utf-8"))
RECORD = REPO_ROOT / "reports/train/test_evaluation.json"
# Estado esperado según la evidencia commiteada (no se supone): hoy el candidato está rechazado.
EV = json.loads(RECORD.read_text(encoding="utf-8"))
APPROVED = bool(EV["gate_passed"])
STATUS, GATE = ("gate_passed", "passed") if APPROVED else ("rejected", "failed")
VERSION = f"{SPEC['family']}-frozen-{EV['freeze_commit'][:7]}"

# Patrones más frecuentes de cada clase en la partición TRAIN real (153 124 y 2 276 bloques).
TRAIN_NORMAL = {**dict.fromkeys(EVENT_COLS, 0), "E5": 3, "E9": 3, "E11": 3, "E21": 3, "E22": 1, "E23": 3, "E26": 3}
TRAIN_ANOMALY = {**dict.fromkeys(EVENT_COLS, 0), "E5": 2, "E7": 1, "E22": 1}


@pytest.fixture(scope="module")
def client():
    return TestClient(A.create_app(mode="demo"))


def synthetic(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.poisson(1.0, size=(n, len(EVENT_COLS))) * rng.integers(0, 2, size=(n, len(EVENT_COLS))),
                        columns=EVENT_COLS)


# 1. Disponibilidad ------------------------------------------------------------------------
def test_health_is_technical_availability_not_approval(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["mode"] == "demo" and body["model_approved_for_production"] is APPROVED


# 2 y 9. Información y aviso de modelo rechazado ------------------------------------------------
def test_model_info_reports_real_gate_state(client):
    info = client.get("/model/info").json()
    assert info["model_sha256"] == SPEC["model_sha256"] and info["threshold"] == SPEC["threshold"]
    assert (info["model_status"], info["gate"], info["demo_only"], info["mode"]) == (STATUS, GATE, True, "demo")
    assert info["gate_checks"] == EV["gate_checks"]
    assert info["algorithm"]["params"] == SPEC["params"] and info["features"] == EVENT_COLS
    assert info["model_version"] == VERSION


# 3. Inferencia válida -------------------------------------------------------------------------
def test_predict_train_patterns(client):
    normal = client.post("/predict", json=TRAIN_NORMAL)
    anomaly = client.post("/predict", json=TRAIN_ANOMALY)
    assert normal.status_code == anomaly.status_code == 200
    assert normal.json()["prediction"] == "normal" and anomaly.json()["prediction"] == "anomaly"
    for body in (normal.json(), anomaly.json()):
        assert body["threshold"] == SPEC["threshold"] and 0 <= body["anomaly_score"] <= 1
        assert (body["model_status"], body["gate"], body["demo_only"]) == (STATUS, GATE, True)
        assert body["is_anomaly"] == (body["anomaly_score"] >= body["threshold"])


# 4. Entradas inválidas --------------------------------------------------------------------------
@pytest.mark.parametrize("change", [
    {"drop": "E7"}, {"E3": -1}, {"E3": 1.5}, {"E3": 2.0}, {"E3": "3"}, {"E3": True}, {"E3": None},
    {"E3": A.MAX_COUNT + 1}, {"Label": "Fail"}, {"Type": 1}, {"BlockId": "blk_1"}, {"e5": 3},
])
def test_invalid_inputs_are_rejected(client, change):
    body = dict(TRAIN_NORMAL)
    if "drop" in change:
        body.pop(change["drop"])
    else:
        body.update(change)
    assert client.post("/predict", json=body).status_code == 422


def test_malformed_requests_are_rejected(client):
    assert client.post("/predict", content=b"{no es json", headers={"content-type": "application/json"}).status_code == 422
    assert client.post("/predict", json=[TRAIN_NORMAL]).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
    assert client.get("/predict").status_code == 405


# 5. Coherencia con el modelo congelado ------------------------------------------------------------
def test_api_matches_frozen_model_exactly(client):
    X = synthetic()
    model = joblib.load(MODELS / "candidate/model.joblib")  # artefacto versionado del repositorio
    batch = model.predict_proba(X)[:, 1]
    for i, row in enumerate(X.to_dict("records")):
        shuffled = dict(reversed(list(row.items())))  # el orden del JSON no debe importar
        got = client.post("/predict", json={k: int(v) for k, v in shuffled.items()}).json()
        single = float(model.predict_proba(X.iloc[[i]])[0, 1])  # misma operación que la API: exacta
        assert got["anomaly_score"] == single
        assert got["anomaly_score"] == pytest.approx(float(batch[i]), abs=1e-12)  # en lote: redondeo de BLAS
        assert got["is_anomaly"] == (single >= SPEC["threshold"])


# 6. Artefacto alterado, origen no confiable o registro ajeno --------------------------------------
@pytest.fixture
def model_copy(tmp_path):
    shutil.copytree(MODELS / "candidate", tmp_path / "models/candidate")
    return tmp_path / "models"


def test_valid_copy_inside_trusted_root_starts(model_copy):
    A.create_app("demo", model_copy / "candidate", RECORD, trusted_root=model_copy)


def test_tampered_artifact_is_refused_before_loading(model_copy, monkeypatch):
    with open(model_copy / "candidate/model.joblib", "ab") as f:
        f.write(b"\0")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("no debe deserializar un artefacto alterado"))
    with pytest.raises(A.StartupRefused, match="SHA-256"):
        A.create_app("demo", model_copy / "candidate", RECORD, trusted_root=model_copy)


def test_wrong_hash_in_spec_is_refused(model_copy):
    p = model_copy / "candidate/spec.json"
    spec = json.loads(p.read_text(encoding="utf-8"))
    spec["model_sha256"] = "0" * 64
    p.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(A.StartupRefused):
        A.create_app("demo", model_copy / "candidate", RECORD, trusted_root=model_copy)


def test_untrusted_location_is_refused(model_copy, tmp_path):
    with pytest.raises(A.StartupRefused, match="confiable"):
        A.create_app("demo", model_copy / "candidate", RECORD, trusted_root=tmp_path / "otra")


def test_gate_record_of_another_model_is_refused(model_copy, tmp_path):
    ev = json.loads(RECORD.read_text(encoding="utf-8"))
    ev["integrity_verified"]["model_sha256"] = "f" * 64
    (tmp_path / "rec.json").write_text(json.dumps(ev), encoding="utf-8")
    with pytest.raises(A.StartupRefused, match="registro del gate"):
        A.create_app("demo", model_copy / "candidate", tmp_path / "rec.json", trusted_root=model_copy)


# 7 y 8. Bloqueo de producción y de modos desconocidos ---------------------------------------------
@pytest.mark.skipif(APPROVED, reason="con un modelo aprobado, production debe arrancar")
def test_production_mode_is_refused_without_approved_model():
    with pytest.raises(A.StartupRefused, match="no existe un modelo aprobado"):
        A.create_app(mode="production")


@pytest.mark.skipif(not APPROVED, reason="no hay modelo aprobado")
def test_production_mode_starts_with_approved_model():
    info = TestClient(A.create_app(mode="production")).get("/model/info").json()
    assert (info["gate"], info["demo_only"]) == ("passed", False)


@pytest.mark.parametrize("mode", ["", "Demo", "prod", "staging", " demo"])
def test_unknown_or_missing_mode_is_refused(mode):
    with pytest.raises(A.StartupRefused, match="APP_MODE"):
        A.create_app(mode=mode)


def test_missing_env_mode_is_refused(monkeypatch):
    monkeypatch.delenv("APP_MODE", raising=False)
    with pytest.raises(A.StartupRefused):
        A.create_app()


# /metrics (Prometheus) ---------------------------------------------------------------------
def _families(text):
    from prometheus_client.parser import text_string_to_metric_families
    return {f.name: f for f in text_string_to_metric_families(text)}


def _value(fams, name, sample, **labels):
    return sum(s.value for s in fams[name].samples if s.name == sample
               and all(s.labels.get(k) == v for k, v in labels.items()))


def test_metrics_count_requests_errors_predictions_and_latency():
    c = TestClient(A.create_app(mode="demo"))  # registro propio: contadores desde cero
    c.post("/predict", json=TRAIN_NORMAL)
    c.post("/predict", json=TRAIN_NORMAL)
    c.post("/predict", json=TRAIN_ANOMALY)
    c.post("/predict", json={**TRAIN_NORMAL, "Label": "Fail"})  # 422
    c.get("/no-existe/blk_123")  # 404 sin ruta: no debe crear una etiqueta con la URL
    r = c.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    fams = _families(r.text)
    req = "hdfs_api_requests"
    assert _value(fams, req, "hdfs_api_requests_total", endpoint="/predict", status="200") == 3
    assert _value(fams, req, "hdfs_api_requests_total", endpoint="/predict", status="422") == 1
    assert _value(fams, req, "hdfs_api_requests_total", endpoint="unmatched", status="404") == 1
    assert _value(fams, "hdfs_api_errors", "hdfs_api_errors_total", kind="client") == 2
    assert _value(fams, "hdfs_predictions", "hdfs_predictions_total", prediction="normal") == 2
    assert _value(fams, "hdfs_predictions", "hdfs_predictions_total", prediction="anomaly") == 1
    assert _value(fams, "hdfs_api_request_duration_seconds", "hdfs_api_request_duration_seconds_count",
                  endpoint="/predict") == 4
    assert _value(fams, "hdfs_prediction_score", "hdfs_prediction_score_count") == 3
    info = next(iter(fams["hdfs_model_info"].samples)).labels
    assert info == {"model_version": VERSION, "model_status": STATUS, "gate": GATE, "mode": "demo"}


def test_metrics_labels_have_no_identifiers_or_features():
    c = TestClient(A.create_app(mode="demo"))
    c.post("/predict", json=TRAIN_ANOMALY)
    c.get("/model/blk_-123456")
    text = c.get("/metrics").text
    labels = {k for f in _families(text).values() for s in f.samples for k in s.labels}
    assert labels <= {"endpoint", "method", "status", "kind", "prediction", "le",
                      "model_version", "model_status", "gate", "mode"}
    assert "blk_" not in text and "E7" not in text and str(REPO_ROOT) not in text and "sha256" not in text


# Imagen Docker y prueba de humo (solo si se pide: RUN_DOCKER_TESTS=1) ------------------------
IMAGE = os.environ.get("DOCKER_IMAGE", "hdfs-anomaly:demo-local")
docker_only = pytest.mark.skipif(os.environ.get("RUN_DOCKER_TESTS") != "1",
                                 reason="prueba de Docker: definir RUN_DOCKER_TESTS=1 con la imagen construida")


def _docker(*args, check=True):
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check, timeout=180)


@docker_only
def test_docker_image_smoke():
    port = "18080"
    cid = _docker("run", "-d", "--rm", "-e", "APP_MODE=demo", "-p", f"127.0.0.1:{port}:8000", IMAGE).stdout.strip()
    try:
        base = f"http://127.0.0.1:{port}"
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except OSError:
                time.sleep(1)
        info = json.loads(urllib.request.urlopen(f"{base}/model/info").read())
        assert info["model_sha256"] == SPEC["model_sha256"] and info["model_status"] == STATUS
        req = urllib.request.Request(f"{base}/predict", data=json.dumps(TRAIN_ANOMALY).encode(),
                                     headers={"content-type": "application/json"})
        body = json.loads(urllib.request.urlopen(req).read())
        assert body["prediction"] == "anomaly" and body["demo_only"] is True
        metrics = urllib.request.urlopen(f"{base}/metrics").read().decode()
        assert 'hdfs_predictions_total{prediction="anomaly"} 1.0' in metrics
        user = _docker("exec", cid, "id", "-u").stdout.strip()
        assert user != "0"  # usuario no privilegiado
    finally:
        _docker("stop", cid, check=False)


@docker_only
@pytest.mark.parametrize("env", [["-e", "APP_MODE=production"], []])
def test_docker_refuses_production_and_missing_mode(env):
    if env and APPROVED:
        pytest.skip("con un modelo aprobado, production debe arrancar")
    r = _docker("run", "--rm", *env, IMAGE, check=False)
    assert r.returncode != 0 and "StartupRefused" in (r.stdout + r.stderr)
