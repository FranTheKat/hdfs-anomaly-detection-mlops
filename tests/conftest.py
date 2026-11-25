"""Proyectos de muestra compartidos en carpetas temporales: nunca tocan el
dataset real, los informes del repositorio ni su test."""
import shutil
import subprocess
from pathlib import Path

import pytest

from hdfs_anomaly import data as D
from hdfs_anomaly import train as T
from hdfs_anomaly.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"
CLEAN = {"commit": "a" * 40, "dirty": False}


def commit_all(root):
    for args in (["init", "-q"], ["add", "-A"],
                 ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"]):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def git_out(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def is_repository_root(root) -> bool:
    """True si `root` es la raíz de un repositorio Git (la distribución ya tiene su propio historial)."""
    r = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], check=False,
                       capture_output=True, text=True)
    return r.returncode == 0 and Path(r.stdout.strip()).resolve() == Path(root).resolve()


def import_into_new_repository(src, dest, files):
    """Copia un proyecto sin su historial a un repositorio nuevo, con un manifiesto de procedencia
    que registra los commits de origen y el SHA-256 de `files` (como la distribución publicable)."""
    import hashlib
    import json

    from hdfs_anomaly import release as R
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git"))
    spec = json.loads((dest / R.FILES["spec"]).read_text(encoding="utf-8"))
    ev = json.loads((dest / R.FILES["test_record"]).read_text(encoding="utf-8"))
    prov = {"origin_commits": {
                "training_code_commit": spec["training_code_commit"], "freeze_commit": ev["freeze_commit"],
                "evaluation_commit": ev["evaluation_git"]["commit"],
                "test_record_commit": git_out(src, "log", "-1", "--format=%H", "--", R.FILES["test_record"])},
            "files_sha256": {rel: hashlib.sha256((dest / rel).read_bytes()).hexdigest() for rel in files}}
    (dest / R.PROVENANCE).parent.mkdir(parents=True, exist_ok=True)
    (dest / R.PROVENANCE).write_text(json.dumps(prov, indent=2), encoding="utf-8")
    commit_all(dest)
    return dest


def _sample(tmp_path, extra=()):
    (tmp_path / "data/preprocessed").mkdir(parents=True)
    shutil.copy(FIXTURES / "sample_matrix.csv", tmp_path / "data/preprocessed/Event_occurrence_matrix.csv")
    shutil.copy(FIXTURES / "sample_labels.csv", tmp_path / "data/preprocessed/anomaly_label.csv")
    shutil.copy(FIXTURES / "sample_block_times.csv", tmp_path / "block_times.csv")
    text = (REPO_ROOT / "config.toml").read_text(encoding="utf-8")
    for a, b in [('dir = "data/HDFS_v1"', 'dir = "data"'),
                 ('block_times = "artifacts/data/block_times.csv.gz"', 'block_times = "block_times.csv"'),
                 ("C = [0.01, 0.1, 1.0, 10.0]", "C = [0.1, 1.0]"),
                 ('max_depth = [8, 16, "none"]', "max_depth = [4]"),
                 ("rf_n_estimators = 200", "rf_n_estimators = 10"),
                 ("n_jobs = 8 ", "n_jobs = 1 "),
                 ("cv_folds = 5", "cv_folds = 3"), *extra]:
        assert a in text, a
        text = text.replace(a, b)
    (tmp_path / "config.toml").write_text(text, encoding="utf-8")
    assert D.main(["--config", str(tmp_path / "config.toml")]) == 0
    return tmp_path


@pytest.fixture
def sample_project(tmp_path, monkeypatch):
    monkeypatch.delenv("HDFS_V1_DIR", raising=False)
    return _sample(tmp_path)


@pytest.fixture
def frozen_project(sample_project, monkeypatch):
    monkeypatch.setattr(D, "_git_state", lambda: CLEAN)
    cfg = str(sample_project / "config.toml")
    assert T.main(["select", "--config", cfg]) == 0
    assert T.main(["evaluate-test", "--config", cfg]) == 1  # sin congelar
    assert T.main(["freeze", "--config", cfg]) == 0
    return sample_project


@pytest.fixture
def rejected_project(tmp_path, monkeypatch):
    """Ciclo completo de entrenamiento y evaluación sobre la muestra, con un gate que solo falla por precision para
    reproducir el caso real (candidato rechazado), todo commiteado como en el repositorio."""
    monkeypatch.delenv("HDFS_V1_DIR", raising=False)
    monkeypatch.setattr(D, "_git_state", lambda: CLEAN)
    gate = [("min_recall = 0.90", "min_recall = 0.0"), ("min_precision = 0.90", "min_precision = 1.01"),
            ("max_fpr = 0.005\n\n# --- Fase 3", "max_fpr = 1.0\n\n# --- Fase 3")]
    root = _sample(tmp_path / "proj", gate)
    cfg = str(root / "config.toml")
    assert T.main(["select", "--config", cfg]) == 0
    assert T.main(["freeze", "--config", cfg]) == 0
    commit_all(root)
    assert T.main(["evaluate-test", "--config", cfg]) == 3  # gate no superado
    commit_all(root)
    return root
