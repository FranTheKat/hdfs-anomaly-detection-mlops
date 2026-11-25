"""Pruebas del quality gate obligatorio del CD.

Los repositorios «aprobados» de estas pruebas son SINTÉTICOS: el «modelo» es un archivo de
bytes sin valor predictivo y las métricas son inventadas para ejercitar la lógica del CD.
No son modelos aprobados ni se publican; solo existen en carpetas temporales."""
import json
import shutil

import pytest
from conftest import commit_all, git_out, import_into_new_repository, is_repository_root

from hdfs_anomaly import release as R
from hdfs_anomaly.config import REPO_ROOT

GATE = {"min_recall": 0.9, "min_precision": 0.9, "max_fpr": 0.005}


def _metrics(tp, fp, fn, tn):
    m = {"confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}
    return {**R.metrics_from_cm(m), "pr_auc": 0.9, "roc_auc": 0.95, "mcc": 0.9, **m}


def make_repo(root, passed=True):
    """Repositorio sintético con historial propio: código → congelación → registro de test."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "models/candidate").mkdir(parents=True)
    (root / "reports/train").mkdir(parents=True)
    (root / "reports/data").mkdir(parents=True)
    (root / "config.toml").write_text("# SINTÉTICO\nx = 1\n", encoding="utf-8")
    (root / "models/candidate/model.joblib").write_bytes(b"SINTETICO: no es un modelo, solo prueba la logica del CD\n")
    test = _metrics(95, 5 if passed else 20, 5, 9895)
    summary = {"manifest_sha256_uncompressed": "m" * 64,
               "splits": {"test": {"rows": test["n"], "anomalies": test["positives"]}}}
    (root / "reports/data/split_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    commit_all(root)
    training = git_out(root, "rev-parse", "HEAD")
    spec = {"status": "CONGELADA — SINTÉTICA (solo pruebas del CD)", "family": "lr", "features": R.FEATURES,
            "model_sha256": R.sha256(root / "models/candidate/model.joblib"),
            "config_sha256": R.sha256(root / "config.toml"), "threshold": 0.5, "gate": GATE,
            "training_code_commit": training, "validation_metrics": _metrics(90, 3, 4, 9903)}
    (root / "models/candidate/spec.json").write_text(json.dumps(spec), encoding="utf-8")
    commit_all(root)
    freeze = git_out(root, "rev-parse", "HEAD")
    checks = R.gate_checks(test, GATE)
    record = {"status": "EVALUADO", "synthetic": True, "threshold": 0.5, "gate": GATE,
              "integrity_verified": {"model_sha256": spec["model_sha256"], "manifest_sha256": "m" * 64},
              "frozen_spec_sha256": R.sha256(root / "models/candidate/spec.json"),
              "training_code_commit": training, "freeze_commit": freeze,
              "evaluation_git": {"commit": freeze, "dirty": False}, "test": test,
              "gate_checks": checks, "gate_passed": all(checks.values())}
    (root / "reports/train/test_evaluation.json").write_text(json.dumps(record), encoding="utf-8")
    commit_all(root)
    return root


def make_imported_repo(tmp_path, passed=True):
    """Candidato sintético copiado SIN historial a un repositorio nuevo, con manifiesto de procedencia."""
    src = make_repo(tmp_path / "origen", passed=passed)
    return import_into_new_repository(src, tmp_path / "nuevo", R.FILES.values())


def run_cli(monkeypatch, root, *args):
    monkeypatch.setattr(R, "ROOT", root)
    return R.main(["decide", *args])


FROZEN_REJECTED_SHA = "da81e6ba7be50c103887e78a5912340cd40f563d9c20368827f55402b79d1076"
REAL_SPEC = json.loads((REPO_ROOT / "models/candidate/spec.json").read_text(encoding="utf-8"))
REAL_RECORD = json.loads((REPO_ROOT / "reports/train/test_evaluation.json").read_text(encoding="utf-8"))
IN_REPOSITORY = is_repository_root(REPO_ROOT)
needs_repository = pytest.mark.skipif(not IN_REPOSITORY,
                                      reason="la carpeta aún no es un repositorio Git (git init + commit)")


# --- Candidato real de la distribución --------------------------------------------------------
@pytest.mark.skipif(IN_REPOSITORY, reason="la carpeta ya es un repositorio Git")
def test_real_candidate_without_repository_fails_closed():
    """Sin historial no hay decisión posible: fallo de integridad, nunca aprobación."""
    with pytest.raises(R.IntegrityError, match="repositorio Git"):
        R.decide(REPO_ROOT)


@needs_repository
def test_real_decision_follows_committed_evidence():
    """La decisión depende de la evidencia, no de suponer un estado."""
    d = R.decide(REPO_ROOT)
    assert d["approved"] == d["aliases_allowed"]
    assert d["approved"] <= REAL_RECORD["gate_passed"]  # nunca aprueba lo que el gate rechazó
    assert d["decision"] == ("publish_allowed" if d["approved"] else "blocked")
    assert d["model_sha256"] == REAL_SPEC["model_sha256"]


@needs_repository
@pytest.mark.skipif(REAL_SPEC["model_sha256"] != FROZEN_REJECTED_SHA, reason="otro candidato congelado")
def test_frozen_lr_candidate_stays_blocked():
    """El candidato congelado da81… fue rechazado: su publicación debe seguir bloqueada."""
    d = R.decide(REPO_ROOT)
    assert d["decision"] == "blocked" and d["gate"] == "failed" and not d["aliases_allowed"]
    assert d["reason"] == "precision_below_threshold"
    assert d["provenance"] == "imported" and "origin_history_not_verifiable" in d["blockers"]
    assert d["freeze_commit"].startswith("feda273") and d["test_record_commit"].startswith("0f05be6")


# --- Candidatos con historial propio (native) -----------------------------------------------------
def test_synthetic_passed_model_is_allowed(tmp_path, monkeypatch):
    root = make_repo(tmp_path / "ok")
    out = tmp_path / "decision.json"
    assert run_cli(monkeypatch, root, "--output", str(out)) == R.EXIT_APPROVED
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["approved"] and d["gate"] == "passed" and d["reason"] == "none"
    assert d["provenance"] == "native" and d["blockers"] == []
    assert d["evidence_sha256"]["models/candidate/model.joblib"] == d["model_sha256"]


def test_synthetic_failed_model_is_blocked(tmp_path, monkeypatch):
    root = make_repo(tmp_path / "ko", passed=False)
    assert run_cli(monkeypatch, root) == R.EXIT_BLOCKED
    d = R.decide(root)
    assert d["reason"] == "precision_below_threshold" and d["blockers"] == ["gate_failed"]


@pytest.mark.parametrize("tamper", ["model", "metric", "gate_flag", "uncommitted", "second_record_commit",
                                    "unknown_freeze_commit"])
def test_integrity_failures_prevent_any_decision(tmp_path, monkeypatch, tamper):
    root = make_repo(tmp_path / "t")
    rec = root / "reports/train/test_evaluation.json"
    ev = json.loads(rec.read_text(encoding="utf-8"))
    if tamper == "model":
        (root / "models/candidate/model.joblib").write_bytes(b"otro artefacto")
    elif tamper == "metric":
        ev["test"]["precision"] = 0.99
    elif tamper == "gate_flag":  # declarar aprobado lo que las métricas no aprueban
        ev["test"] = _metrics(95, 20, 5, 9880)
    elif tamper == "second_record_commit":
        ev["note"] = "reescrito"
    if tamper in ("metric", "gate_flag", "second_record_commit", "uncommitted"):
        rec.write_text(json.dumps(ev if tamper != "uncommitted" else {**ev, "x": 1}), encoding="utf-8")
    if tamper == "unknown_freeze_commit":  # historial nuevo sin los commits de origen ni manifiesto
        shutil.copytree(root, tmp_path / "sin_historial", ignore=shutil.ignore_patterns(".git"))
        root = tmp_path / "sin_historial"
    if tamper != "uncommitted":
        commit_all(root)
    assert run_cli(monkeypatch, root) == R.EXIT_INTEGRITY
    with pytest.raises(R.IntegrityError):
        R.decide(root)


def test_folder_without_git_or_nested_in_another_repository_fails_closed(tmp_path, monkeypatch):
    src = make_repo(tmp_path / "ok")
    plain = tmp_path / "sin_git"
    shutil.copytree(src, plain, ignore=shutil.ignore_patterns(".git"))
    assert run_cli(monkeypatch, plain) == R.EXIT_INTEGRITY
    outer = tmp_path / "externo"
    shutil.copytree(src, outer / "proyecto", ignore=shutil.ignore_patterns(".git"))
    commit_all(outer)  # commiteados, pero en el repositorio de la carpeta superior
    with pytest.raises(R.IntegrityError, match="raíz"):
        R.decide(outer / "proyecto")


# --- Candidatos importados de otro repositorio (imported) -----------------------------------------
@pytest.mark.parametrize("passed", [True, False])
def test_imported_candidate_is_always_blocked(tmp_path, monkeypatch, passed):
    """Sin el historial de origen no se demuestra la evaluación única de test: se bloquea
    la publicación aunque el gate estuviera superado."""
    root = make_imported_repo(tmp_path, passed=passed)
    assert run_cli(monkeypatch, root) == R.EXIT_BLOCKED
    d = R.decide(root)
    assert d["provenance"] == "imported" and not d["approved"] and not d["aliases_allowed"]
    assert d["gate"] == ("passed" if passed else "failed")
    assert "origin_history_not_verifiable" in d["blockers"]
    assert d["test_record_commit"] != d["test_record_commit_in_this_repository"]


@pytest.mark.parametrize("tamper", ["no_manifest", "manifest_sha", "manifest_commit", "model_and_manifest"])
def test_imported_candidate_with_bad_manifest_fails_closed(tmp_path, monkeypatch, tamper):
    root = make_imported_repo(tmp_path)
    prov_path = root / R.PROVENANCE
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if tamper == "no_manifest":
        prov_path.unlink()
    elif tamper == "manifest_sha":
        prov["files_sha256"]["config.toml"] = "0" * 64
    elif tamper == "manifest_commit":
        prov["origin_commits"]["freeze_commit"] = "c" * 40
    elif tamper == "model_and_manifest":  # sustitución coherente del modelo: falla contra spec.json
        (root / R.FILES["model"]).write_bytes(b"otro artefacto")
        prov["files_sha256"][R.FILES["model"]] = R.sha256(root / R.FILES["model"])
    if prov_path.exists():
        prov_path.write_text(json.dumps(prov), encoding="utf-8")
    commit_all(root)
    assert run_cli(monkeypatch, root) == R.EXIT_INTEGRITY


def test_gate_cannot_be_skipped_by_env_or_arguments(tmp_path, monkeypatch):
    root = make_repo(tmp_path / "ko", passed=False)
    for var in ("FORCE_PUBLISH", "SKIP_GATE", "GATE", "APPROVED", "RELEASE_OVERRIDE", "CI"):
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("GATE", "passed")
    assert run_cli(monkeypatch, root) == R.EXIT_BLOCKED
    for extra in (["--force"], ["--skip-gate"], ["--root", str(root)]):
        with pytest.raises(SystemExit):
            R.main(["decide", *extra])
