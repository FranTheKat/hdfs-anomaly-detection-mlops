"""Monitorización predictiva y simulación. Solo referencia de train y datos sintéticos."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hdfs_anomaly import api as A
from hdfs_anomaly import drift as D
from hdfs_anomaly import simulate as S
from hdfs_anomaly.data import EVENT_COLS

REF = json.loads(D.DEFAULT_REF.read_text(encoding="utf-8"))


def mix(n, seed=0):
    return S.train_mix(REF, n, np.random.default_rng(seed))


def test_reference_is_train_only_and_bound_to_frozen_artifacts():
    assert REF["source"] == "train" and REF["provenance"]["partition"] == "train"
    assert REF["rows"] == 345059  # tamaño de la partición train
    assert REF["provenance"]["model_sha256"] == "da81e6ba7be50c103887e78a5912340cd40f563d9c20368827f55402b79d1076"
    assert REF["threshold"] == 0.07222539011103471
    assert {c for c, f in REF["features"].items() if f["constant"] is not None} == {"E1", "E22"}


def test_psi_basics():
    g = [[0], [1], [2]]
    assert D.psi([100, 50, 50], [200, 100, 100], g) == pytest.approx(0, abs=1e-4)  # el suavizado deja un residuo mínimo
    assert D.psi([100, 50, 50], [10, 50, 140], g) > D.PSI_ALERT
    assert np.isfinite(D.psi([100, 0, 0], [0, 0, 100], g))  # ceros: suavizado, sin infinitos


def test_train_like_window_is_stable():
    r = D.report(mix(3000), REF)
    assert r["status"] == "ok", r["alerts"]
    assert r["novel_patterns"]["rate"] == 0
    assert r["features"]["E22"]["status"] == "constant_in_reference" and r["features"]["E22"]["psi"] is None


def test_small_window_never_alerts_on_distribution():
    df = S.shift(mix(300), np.random.default_rng(1), frac=1.0)
    r = D.report(df, REF)
    assert r["sample_size"] == "insufficient"
    # el esquema (contrato) sí se señala; las reglas estadísticas no alertan con n pequeño
    assert {a["rule"] for a in r["alerts"]} <= {"insufficient_sample", "schema"}
    assert D.report(mix(300), REF)["status"] == "insufficient_sample"


def test_controlled_shift_is_detected():
    r = D.report(S.shift(mix(3000), np.random.default_rng(1)), REF)
    rules = {(a["rule"], a["severity"]) for a in r["alerts"]}
    assert ("psi_feature", "alert") in rules and ("novel_patterns", "alert") in rules
    assert r["features"]["E2"]["psi"] >= D.PSI_ALERT
    assert "no una degradación" in r["interpretation"]


def test_schema_issues():
    df = mix(600)
    df.loc[0, "E5"] = -1
    df.loc[1, "E22"] = 0
    df["BlockId"] = "blk_x"
    r = D.report(df.drop(columns=["E3"]), REF)
    s = r["schema"]
    assert s["missing_columns"] == ["E3"] and s["forbidden_columns"] == ["BlockId"]
    assert s["negative"] == {"E5": 1} and s["constant_deviation"] == {"E22": 1}
    assert r["status"] == "alert" and r["features"] == {}


def test_report_is_byte_reproducible(tmp_path):
    csv = tmp_path / "w.csv"
    mix(1000).to_csv(csv, index=False)
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    D.report_file(csv, D.DEFAULT_REF, a)
    D.report_file(csv, D.DEFAULT_REF, b)
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.read_text(encoding="utf-8"))["window_sha256"]


def test_simulation_against_demo_api(tmp_path):
    """La simulación completa contra la API real (TestClient), sin dataset ni test."""
    c = TestClient(A.create_app(mode="demo"))

    def post(path, payload=None):
        r = c.get(path) if payload is None else c.post(path, json=payload)
        return r.status_code, r.text

    s = S.run(post, REF, out=tmp_path, n=600)  # ventanas reducidas para que la prueba sea rápida
    assert s["demo_only"] and s["invalid_request_status"] == [422, 422, 422]
    w = s["windows"]
    assert w["baseline"]["status"] == "ok", w["baseline"]
    assert w["small"]["status"] == "insufficient_sample"
    assert "alert:predicted_rate" in w["anomaly_burst"]["alerts"]
    assert "alert:psi_feature" in w["shifted"]["alerts"]
    metrics = (tmp_path / "metrics_snapshot.txt").read_text(encoding="utf-8")
    assert 'hdfs_api_requests_total{endpoint="/predict",method="POST",status="422"} 3.0' in metrics
    assert set(EVENT_COLS) <= set((tmp_path / "windows/baseline.csv").read_text(encoding="utf-8").split("\n")[0].split(","))
