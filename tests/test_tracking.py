"""Pruebas de la importación a MLflow, integridad del artefacto, idempotencia y
tratamiento de versiones rechazadas. Todo sobre la muestra, en carpetas temporales."""
import json

import joblib
import pytest
from conftest import commit_all, import_into_new_repository

from hdfs_anomaly import release as R
from hdfs_anomaly import tracking as K
from hdfs_anomaly.config import load_config


def cfg_of(root):
    cfg = load_config(root / "config.toml")
    cfg["_path"] = root / "config.toml"
    return cfg


def test_gate_reason():
    assert K.gate_reason({"recall": True, "precision": False, "fpr": True}) == "precision_below_threshold"
    assert K.gate_reason({"recall": False, "precision": True, "fpr": False}) == \
        "recall_below_threshold,fpr_above_threshold"


def test_metrics_from_cm_matches_train_evaluate():
    import numpy as np

    from hdfs_anomaly import train as T
    y = np.array([1, 1, 0, 0, 0, 1])
    m = T.evaluate(y, np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.7]), 0.5)
    assert all(m[k] == v for k, v in R.metrics_from_cm(m).items())


def test_register_rejected_version_is_traceable_and_idempotent(rejected_project, tmp_path, monkeypatch):
    def no_unpickle(*a, **k):
        raise AssertionError("el registro no debe deserializar el modelo")
    monkeypatch.setattr(joblib, "load", no_unpickle)
    cfg, track = cfg_of(rejected_project), tmp_path / "mlflow"
    first = K.register(cfg, track)
    assert first["created_run"] and first["created_version"]
    assert (first["gate"], first["status"], first["reason"]) == ("failed", "rejected", "precision_below_threshold")
    second = K.register(cfg, track)  # idempotente
    assert not second["created_run"] and not second["created_version"]
    assert (second["run_id"], second["version"]) == (first["run_id"], first["version"])

    client = K.client_for(track)
    exp = client.get_experiment_by_name(K.EXPERIMENT)
    assert len(client.search_runs([exp.experiment_id])) == 1
    versions = client.search_model_versions(f"name = '{K.MODEL_NAME}'")
    assert len(versions) == 1 and versions[0].run_id == first["run_id"]
    tags = versions[0].tags
    assert tags["gate"] == "failed" and tags["status"] == "rejected" and tags["test_evaluated"] == "true"
    assert tags["promotion"] == "blocked"
    assert client.get_registered_model(K.MODEL_NAME).aliases == {}  # ni champion ni challenger

    run = client.get_run(first["run_id"])
    spec = json.loads((rejected_project / "models/candidate/spec.json").read_text(encoding="utf-8"))
    ev = json.loads((rejected_project / "reports/train/test_evaluation.json").read_text(encoding="utf-8"))
    assert run.data.metrics["test/precision"] == ev["test"]["precision"]
    assert run.data.metrics["test/fp"] == ev["test"]["confusion_matrix"]["fp"]
    assert run.data.metrics["validation/f1"] == spec["validation_metrics"]["f1"]
    assert run.data.params["threshold"] == repr(spec["threshold"]) and run.data.params["seed"] == "42"
    assert run.data.tags["sha256.model"] == spec["model_sha256"]
    assert run.data.tags["provenance.test"] == "reports/train/test_evaluation.json#test"
    assert run.data.tags["training_code_commit"] == spec["training_code_commit"]

    out = K.verify(cfg, track)
    assert out["model_sha256"] == spec["model_sha256"] and out["status"] == "rejected"


def test_interrupted_import_is_redone_without_duplicates(rejected_project, tmp_path):
    cfg, track = cfg_of(rejected_project), tmp_path / "mlflow"
    first = K.register(cfg, track)
    client = K.client_for(track)
    client.set_tag(first["run_id"], "import_complete", "false")  # simula una importación cortada
    client.delete_model_version(K.MODEL_NAME, first["version"])
    again = K.register(cfg, track)
    assert again["created_run"] and again["run_id"] != first["run_id"]
    exp = client.get_experiment_by_name(K.EXPERIMENT)
    assert len(client.search_runs([exp.experiment_id])) == 1
    assert len(client.search_model_versions(f"name = '{K.MODEL_NAME}'")) == 1
    K.verify(cfg, track)


@pytest.mark.parametrize("tamper", ["metric_uncommitted", "metric_committed", "cm_and_metric_committed",
                                    "model_committed"])
def test_register_rejects_edited_evidence(rejected_project, tmp_path, tamper):
    ev_path = rejected_project / "reports/train/test_evaluation.json"
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    if tamper.startswith("metric"):
        ev["test"]["precision"] = 0.95  # métrica inventada
    if tamper.startswith("cm_and_metric"):  # sustitución coherente: falla el registro único
        ev["test"]["confusion_matrix"]["fp"] = 0
        ev["test"]["precision"] = 1.0
    if tamper != "model_committed":
        ev_path.write_text(json.dumps(ev, indent=2), encoding="utf-8")
    else:
        with open(rejected_project / "models/candidate/model.joblib", "ab") as f:
            f.write(b"\0")
    if tamper.endswith("committed") and not tamper.endswith("uncommitted"):
        commit_all(rejected_project)
    with pytest.raises(K.IntegrityError):
        K.register(cfg_of(rejected_project), tmp_path / "mlflow")
    assert not (tmp_path / "mlflow" / "mlflow.db").exists()  # no se escribió nada en MLflow


def test_verify_detects_substituted_artifact_and_protected_alias(rejected_project, tmp_path):
    cfg, track = cfg_of(rejected_project), tmp_path / "mlflow"
    reg = K.register(cfg, track)
    client = K.client_for(track)
    client.set_registered_model_alias(K.MODEL_NAME, "champion", reg["version"])
    with pytest.raises(K.IntegrityError, match="Alias protegidos"):
        K.verify(cfg, track)
    client.delete_registered_model_alias(K.MODEL_NAME, "champion")
    K.verify(cfg, track)
    artifact = next((track / "mlartifacts").rglob("model.joblib"))
    artifact.write_bytes(artifact.read_bytes() + b"\0")  # sustitución del artefacto registrado
    with pytest.raises(K.IntegrityError, match="idéntico"):
        K.verify(cfg, track)


def test_rejected_version_cannot_receive_protected_alias(rejected_project, tmp_path):
    cfg, track = cfg_of(rejected_project), tmp_path / "mlflow"
    reg = K.register(cfg, track)
    for alias in K.PROTECTED_ALIASES:
        with pytest.raises(K.IntegrityError, match="no superó el quality gate"):
            K.set_alias(cfg, track, alias, reg["version"])
        assert K.main(["set-alias", "--alias", alias, "--version", reg["version"],
                       "--config", str(rejected_project / "config.toml"), "--tracking-dir", str(track)]) == 1
    assert K.client_for(track).get_registered_model(K.MODEL_NAME).aliases == {}


def _imported(rejected_project, tmp_path):
    """El proyecto rechazado copiado sin historial a un repositorio nuevo (como la distribución)."""
    files = [p.relative_to(rejected_project).as_posix() for d in ("models", "reports")
             for p in sorted((rejected_project / d).rglob("*")) if p.is_file()] + ["config.toml"]
    return import_into_new_repository(rejected_project, tmp_path / "nuevo", files)


def test_imported_candidate_registers_with_origin_provenance(rejected_project, tmp_path):
    root, track = _imported(rejected_project, tmp_path), tmp_path / "mlflow"
    ev = json.loads((root / "reports/train/test_evaluation.json").read_text(encoding="utf-8"))
    first = K.register(cfg_of(root), track)
    assert (first["gate"], first["status"], first["promotion"]) == ("failed", "rejected", "blocked")
    assert not K.register(cfg_of(root), track)["created_version"]  # idempotente
    run = K.client_for(track).get_run(first["run_id"])
    assert run.data.tags["provenance.history"] == "imported"
    assert run.data.tags["freeze_commit"] == ev["freeze_commit"]  # commit de origen, no uno nuevo
    origin = json.loads((root / R.PROVENANCE).read_text(encoding="utf-8"))["origin_commits"]
    assert run.data.tags["test_record_commit"] == origin["test_record_commit"]
    assert K.verify(cfg_of(root), track)["status"] == "rejected"
    for alias in K.PROTECTED_ALIASES:
        with pytest.raises(K.IntegrityError):
            K.set_alias(cfg_of(root), track, alias, first["version"])
    assert K.client_for(track).get_registered_model(K.MODEL_NAME).aliases == {}


def test_imported_candidate_without_manifest_is_not_registered(rejected_project, tmp_path):
    root = _imported(rejected_project, tmp_path)
    (root / R.PROVENANCE).unlink()
    commit_all(root)
    with pytest.raises(K.IntegrityError, match="manifiesto"):
        K.register(cfg_of(root), tmp_path / "mlflow")
    assert not (tmp_path / "mlflow" / "mlflow.db").exists()
