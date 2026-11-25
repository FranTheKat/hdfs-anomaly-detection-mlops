"""Pruebas de datos: validación, preparación, partición y ejecución de la etapa."""
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hdfs_anomaly import data as D
from hdfs_anomaly.config import REPO_ROOT, load_config

FIXTURES = Path(__file__).parent / "fixtures"


def make_df(n_normal=8, n_anomaly=2):
    """Matriz sintética válida: E22=1, E5=3, anomalías con E7>0 y Type informado."""
    n = n_normal + n_anomaly
    rows = []
    for i in range(n):
        fail = i >= n_normal
        ev = {c: 0 for c in D.EVENT_COLS}
        ev.update(E5=3, E22=1, E7=1 if fail else 0)
        rows.append({"BlockId": f"blk_{i}", "Label": "Fail" if fail else "Success",
                     "Type": "5" if fail else None, **ev})
    return pd.DataFrame(rows, columns=D.EXPECTED_COLUMNS)


def labels_for(df):
    return pd.DataFrame({"BlockId": df["BlockId"],
                         "Label": df["Label"].map({"Success": "Normal", "Fail": "Anomaly"})})


def failed(report):
    return {c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "error"}


def warnings(report):
    return {c["name"] for c in report["checks"] if not c["passed"] and c["severity"] == "warning"}


# --- Validación -------------------------------------------------------------------
def test_valid_data_passes():
    df = make_df()
    r = D.validate(df, labels_for(df), min_rate=0.0, max_rate=1.0)
    assert r["passed"] and not failed(r) and not warnings(r)


def test_missing_and_extra_columns_fail():
    df = make_df()
    assert "schema_columns" in failed(D.validate(df.drop(columns=["E7"])))
    assert "schema_columns" in failed(D.validate(df.assign(Latency=1.0)))


def test_duplicate_and_malformed_block_ids_fail():
    df = make_df()
    df.loc[1, "BlockId"] = df.loc[0, "BlockId"]
    df.loc[2, "BlockId"] = "bloque-2"
    assert {"block_id_unique", "block_id_format"} <= failed(D.validate(df))


def test_invalid_label_fails():
    df = make_df()
    df.loc[0, "Label"] = "Normal"
    assert "label_values" in failed(D.validate(df))


def test_null_or_negative_events_fail():
    df = make_df()
    df["E3"] = df["E3"].astype(float)
    df.loc[0, "E3"] = np.nan
    assert "events_integer_no_nulls" in failed(D.validate(df))
    df = make_df()
    df.loc[0, "E4"] = -1
    assert "events_non_negative" in failed(D.validate(df))


def test_single_class_fails():
    assert "both_classes_present" in failed(D.validate(make_df(n_anomaly=0)))


def test_e22_and_rate_are_warnings_not_errors():
    df = make_df()
    df.loc[0, "E22"] = 2
    r = D.validate(df, min_rate=0.5, max_rate=0.9)
    assert r["passed"]
    assert {"e22_allocate_once", "anomaly_rate_in_range"} <= warnings(r)


def test_reference_labels_cross_check():
    df = make_df()
    ref = labels_for(df)
    ref.loc[0, "Label"] = "Anomaly"
    assert "labels_match_reference" in failed(D.validate(df, ref))
    assert "labels_same_block_ids" in failed(D.validate(df, ref.iloc[1:]))


def test_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="README"):
        D.load_matrix(tmp_path / "no_existe.csv")


# --- Preparación ----------------------------------------------------------------
def test_prepare_uses_only_event_columns():
    ids, X, y = D.prepare(make_df())
    assert list(X.columns) == D.EVENT_COLS
    assert not D.FORBIDDEN_FEATURES & set(X.columns)
    assert y.tolist() == [0] * 8 + [1] * 2
    assert ids.tolist()[0] == "blk_0"


def test_type_column_does_not_reach_features():
    df = make_df()
    _, X1, _ = D.prepare(df)
    _, X2, _ = D.prepare(df.assign(Type="filtrado"))
    pd.testing.assert_frame_equal(X1, X2)


# --- Partición --------------------------------------------------------------------
def test_first_seen_split_preserves_order_and_sizes():
    y = np.array([0] * 90 + [1] * 10)
    parts = D.split(y, "first_seen", 0.6, 0.2, 0.2)
    assert (parts[:60] == "train").all() and (parts[60:80] == "validation").all() and (parts[80:] == "test").all()


def test_stratified_split_is_deterministic_and_balanced():
    y = np.array([0] * 900 + [1] * 100)
    a = D.split(y, "stratified", seed=1)
    assert (a == D.split(y, "stratified", seed=1)).all()
    assert not (a == D.split(y, "stratified", seed=2)).all()
    for name, size in (("train", 600), ("validation", 200), ("test", 200)):
        assert (a == name).sum() == size
        assert y[a == name].mean() == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("fr", [(0.6, 0.2, 0.3), (0.8, 0.2, 0.0), (-0.1, 0.6, 0.5)])
def test_invalid_fractions_rejected(fr):
    with pytest.raises(ValueError):
        D.split(np.zeros(10), "first_seen", *fr)


def test_unknown_strategy_rejected():
    with pytest.raises(ValueError):
        D.split(np.zeros(10), "aleatoria")


def test_summary_detects_disjoint_splits():
    ids, X, y = D.prepare(make_df(16, 4))
    s = D.summarize(ids, X, y, D.split(y, "first_seen"))
    assert s["block_ids_disjoint"] and s["all_rows_assigned"]
    assert sum(s[k]["rows"] for k in D.SPLITS) == 20


def test_manifest_round_trip_and_stable_hash(tmp_path):
    ids, _, y = D.prepare(make_df())
    parts = D.split(y, "first_seen")
    h1 = D.write_manifest(tmp_path / "m1.csv.gz", ids, parts)
    h2 = D.write_manifest(tmp_path / "m2.csv.gz", ids, parts)
    assert h1 == h2
    assert (tmp_path / "m1.csv.gz").read_bytes() == (tmp_path / "m2.csv.gz").read_bytes()
    m = D.read_manifest(tmp_path / "m1.csv.gz")
    assert m["BlockId"].tolist() == ids.tolist() and m["split"].tolist() == parts.tolist()


# --- Etapa completa sobre la muestra real (misma ruta que usará la CI) -------------
@pytest.fixture
def sample_cfg(tmp_path):
    (tmp_path / "data" / "preprocessed").mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_matrix.csv", tmp_path / "data/preprocessed/Event_occurrence_matrix.csv")
    shutil.copy(FIXTURES / "sample_labels.csv", tmp_path / "data/preprocessed/anomaly_label.csv")
    shutil.copy(FIXTURES / "sample_block_times.csv", tmp_path / "block_times.csv")
    cfg_text = (REPO_ROOT / "config.toml").read_text(encoding="utf-8").replace(
        'dir = "data/HDFS_v1"', 'dir = "data"').replace(
        'block_times = "artifacts/data/block_times.csv.gz"', 'block_times = "block_times.csv"')
    (tmp_path / "config.toml").write_text(cfg_text, encoding="utf-8")
    return tmp_path


def test_stage_runs_on_real_sample(sample_cfg, monkeypatch):
    monkeypatch.delenv("HDFS_V1_DIR", raising=False)
    assert D.main(["--config", str(sample_cfg / "config.toml")]) == 0
    report = json.loads((sample_cfg / "reports/data/validation_report.json").read_text(encoding="utf-8"))
    summary = json.loads((sample_cfg / "reports/data/split_summary.json").read_text(encoding="utf-8"))
    assert report["passed"] and report["rows"] == 2000
    assert summary["splits"]["block_ids_disjoint"] and summary["temporal_checks"]["prospective"]
    rows = [summary["splits"][k]["rows"] for k in D.SPLITS]
    # Cortes en las posiciones 1200 y 1600; los empates en un corte van a la partición anterior.
    assert sum(rows) == 2000 and rows[0] >= 1200 and rows[0] + rows[1] >= 1600
    assert {c["name"] for c in report["checks"]} >= {"times_same_block_ids", "times_lines_match_events"}
    with gzip.open(sample_cfg / "artifacts/data/split_manifest.csv.gz", "rt") as f:
        assert f.readline().strip() == "BlockId,split"


def test_stage_stops_on_invalid_data(sample_cfg, monkeypatch, capsys):
    monkeypatch.delenv("HDFS_V1_DIR", raising=False)
    path = sample_cfg / "data/preprocessed/Event_occurrence_matrix.csv"
    df = pd.read_csv(path, dtype=str)
    df.loc[0, "E5"] = "-3"
    df.to_csv(path, index=False)
    assert D.main(["--config", str(sample_cfg / "config.toml")]) == 1
    assert "events_non_negative" in capsys.readouterr().err
    assert not (sample_cfg / "artifacts/data/split_manifest.csv.gz").exists()


def test_stage_requires_block_times_for_completion(sample_cfg, monkeypatch, capsys):
    monkeypatch.delenv("HDFS_V1_DIR", raising=False)
    (sample_cfg / "block_times.csv").unlink()
    assert D.main(["--config", str(sample_cfg / "config.toml")]) == 1
    assert "hdfs_anomaly.timeline" in capsys.readouterr().err


# --- Dataset completo (solo si está disponible localmente) -------------------------
FULL = load_config()


@pytest.mark.skipif(not (FULL["data"]["matrix_path"].is_file() and FULL["split"]["block_times_path"].is_file()),
                    reason="dataset HDFS_v1 completo o block_times no disponibles")
def test_full_dataset_validates_and_splits():
    from hdfs_anomaly.timeline import read_times
    df = D.load_matrix(FULL["data"]["matrix_path"])
    r = D.validate(df, D.load_labels(FULL["data"]["labels_path"]))
    assert r["passed"] and r["rows"] == 575061 and not warnings(r)
    ids, X, y = D.prepare(df)
    assert int(y.sum()) == 16838
    times = read_times(FULL["split"]["block_times_path"])
    assert all(c["passed"] for c in D.validate_times(ids, X, times))
    t = times.set_index("BlockId").loc[ids]
    first, last = t["first_ts"].to_numpy(), t["last_ts"].to_numpy()
    parts = D.split(y, "completion", last_ts=last)
    s = D.summarize(ids, X, y, parts)
    assert s["block_ids_disjoint"] and [s[k]["rows"] for k in D.SPLITS] == [345059, 114992, 115010]
    assert D.temporal_checks(parts, first, last)["prospective"]
    # La partición por primera aparición NO cumple el orden de finalización (motivo del cambio).
    assert not D.temporal_checks(D.split(y, "first_seen"), first, last)["prospective"]
