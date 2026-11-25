"""Pruebas de entrenamiento: métricas, regla de umbral, selección, diagnósticos y que test
no interviene en la selección."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import CLEAN, commit_all

from hdfs_anomaly import data as D
from hdfs_anomaly import train as T
from hdfs_anomaly.config import load_config

FIXTURES = Path(__file__).parent / "fixtures"


# --- Umbral y métricas ---------------------------------------------------------------
def test_threshold_table_counts_ties_together():
    y = np.array([1, 0, 1, 0])
    score = np.array([0.9, 0.9, 0.5, 0.1])  # empate en 0.9: entran juntos
    tab = T.threshold_table(y, score)
    assert tab["tp"].tolist() == [1, 2, 2] and tab["fp"].tolist() == [1, 1, 2]
    assert tab["threshold"].iloc[0] == pytest.approx(0.7)  # a mitad de camino entre 0.9 y 0.5


def test_choose_threshold_respects_fpr_limit():
    # 10 normales y 4 anomalías; la anomalía de 0.4 solo se captura aceptando 2 falsos positivos.
    y = np.array([1, 1, 1, 1] + [0] * 10)
    score = np.array([0.95, 0.9, 0.85, 0.4] + [0.6, 0.5] + [0.1] * 8)
    free = T.choose_threshold(y, score, max_fpr=1.0)
    strict = T.choose_threshold(y, score, max_fpr=0.0)
    assert strict["constraint_met"]
    m = T.evaluate(y, score, strict["threshold"])
    assert m["fpr"] == 0 and m["recall"] == 0.75 and m["precision"] == 1.0
    assert free["unconstrained_best"]["f1"] == pytest.approx(m["f1"], abs=1e-6)  # aquí coinciden


def test_constraint_not_met_is_reported():
    y = np.array([1, 0, 0])
    score = np.array([0.1, 0.9, 0.8])  # cualquier umbral que capture la anomalía da FPR = 1
    r = T.choose_threshold(y, score, max_fpr=0.005)
    assert not r["constraint_met"]
    assert T.evaluate(y, score, r["threshold"])["recall"] == 0


def test_evaluate_confusion_matrix_and_fpr():
    y = np.array([1, 1, 0, 0, 0])
    m = T.evaluate(y, np.array([0.9, 0.2, 0.8, 0.1, 0.1]), 0.5)
    assert m["confusion_matrix"] == {"tn": 2, "fp": 1, "fn": 1, "tp": 1}
    assert m["fpr"] == pytest.approx(1 / 3, abs=1e-6) and m["precision"] == 0.5 and m["recall"] == 0.5


# --- Modelos -------------------------------------------------------------------------
def toy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.integers(0, 4, size=(n, len(D.EVENT_COLS))), columns=D.EVENT_COLS)
    y = ((X["E7"] > 2) | (X["E20"] > 2)).to_numpy(dtype=int)
    return X, y


def test_dummy_never_detects_anomalies():
    X, y = toy()
    row, _, _ = T.fit_and_score("dummy", {}, {"X": X, "y": y}, {"X": X, "y": y}, 0, 1, 0.005)
    assert not row["constraint_met"] and row["validation"]["recall"] == 0
    assert row["validation"]["pr_auc"] == pytest.approx(y.mean(), abs=1e-6)


def test_lr_transformations_are_fitted_only_on_train():
    X, y = toy()
    m = T.build_model("lr", {"C": 1.0, "class_weight": "balanced"}, 0).fit(X[:200], y[:200])
    assert np.allclose(m.named_steps["scale"].mean_, np.log1p(X[:200]).mean().to_numpy())


def test_training_is_deterministic():
    X, y = toy()
    p = {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 1, "class_weight": "none"}
    a = T.fit_model("rf", p, X, y, 7, 2).predict_proba(X)[:, 1]
    b = T.fit_model("rf", p, X, y, 7, 2).predict_proba(X)[:, 1]
    assert np.array_equal(a, b)


def test_grids_match_config():
    tcfg = load_config()["train"]
    assert len(T.grid("lr", tcfg)) == 8 and len(T.grid("rf", tcfg)) == 12 and len(T.grid("rf", tcfg, True)) == 2
    assert all(p["n_estimators"] == tcfg["rf_n_estimators"] for p in T.grid("rf", tcfg))


def test_tie_break_prefers_simpler_configuration():
    rows = [{"family": "rf", "params": {"max_depth": "none", "min_samples_leaf": 1, "class_weight": "none"},
             "validation": {"pr_auc": 0.99991}},
            {"family": "rf", "params": {"max_depth": 8, "min_samples_leaf": 5, "class_weight": "none"},
             "validation": {"pr_auc": 0.99994}}]
    assert T.pick_best(rows, "rf", 4)["params"]["max_depth"] == 8


@pytest.mark.parametrize("lr_f1,rf_f1,rf_auc,rf_ok,lr_ok,expected", [
    (0.90, 0.95, 0.96, True, True, "rf"),     # cumple las tres condiciones
    (0.90, 0.905, 0.96, True, True, "lr"),    # mejora de F1 insuficiente
    (0.90, 0.95, 0.94, True, True, "lr"),     # empeora la PR-AUC
    (0.90, 0.95, 0.96, False, True, "lr"),    # el RF no cumple FPR
    (0.90, 0.80, 0.96, True, False, "lr"),    # sin las tres condiciones se conserva la LR
])
def test_candidate_rule(lr_f1, rf_f1, rf_auc, rf_ok, lr_ok, expected):
    lr = {"validation": {"f1": lr_f1, "pr_auc": 0.95}, "constraint_met": lr_ok}
    rf = {"validation": {"f1": rf_f1, "pr_auc": rf_auc}, "constraint_met": rf_ok}
    assert T.decide_candidate(lr, rf, 0.01)[0] == expected


def test_grouped_cv_keeps_patterns_apart_and_checks_classes():
    X, y = toy(600)
    tr = {"X": X, "y": y}
    res = T.grouped_cv("lr", {"C": 1.0, "class_weight": "balanced"}, tr, 0, 1, 3, 0.5)
    assert res["folds_evaluated"] >= 1
    assert all("both_classes" in f and "train_has_both_classes" in f for f in res["folds"])
    keys = T.pattern_keys(X)
    from sklearn.model_selection import StratifiedGroupKFold
    for a, b in StratifiedGroupKFold(3, shuffle=True, random_state=0).split(X, y, keys):
        assert not set(keys[a]) & set(keys[b])


# --- Etapa completa sobre la muestra real ----------------------------------------------
def test_load_splits_excludes_test_by_default(sample_project):
    cfg = load_config(sample_project / "config.toml")
    s = T.load_splits(cfg)
    assert "test" not in s and {"train", "validation"} <= set(s)
    assert "test" in T.load_splits(cfg, include_test=True)


def test_manifest_tampering_is_detected(sample_project):
    cfg = load_config(sample_project / "config.toml")
    m = cfg["output"]["artifacts"] / "split_manifest.csv.gz"
    df = D.read_manifest(m)
    df.loc[0, "split"] = "test" if df.loc[0, "split"] != "test" else "train"
    D.write_manifest(m, df["BlockId"], df["split"].to_numpy())
    with pytest.raises(ValueError, match="manifiesto"):
        T.load_splits(cfg)


def test_select_end_to_end_without_test(sample_project):
    cfg_path = str(sample_project / "config.toml")
    assert T.main(["select", "--config", cfg_path]) == 0
    out = sample_project / "reports/train"
    sel = json.loads((out / "selection_results.json").read_text(encoding="utf-8"))
    assert sel["test_used"] is False and sel["candidate_family"] in ("lr", "rf")
    n_dev = sel["rows"]["train"] + sel["rows"]["validation"]
    assert n_dev < 2000  # el test de la muestra no se cargó
    sens = json.loads((out / "sensitivity.json").read_text(encoding="utf-8"))["results"]
    assert all(r["train_rows"] + r["eval_rows"] == n_dev for r in sens.values())  # test excluido
    spec = json.loads((out / "candidate_spec.json").read_text(encoding="utf-8"))
    assert spec["model_sha256"] == D.sha256_file(sample_project / spec["model_path"])
    assert "NO congelada" in spec["status"]
    assert (out / "comparativa_validacion.md").read_text(encoding="utf-8").startswith("# Comparativa")
    # Reproducible: una segunda ejecución da las mismas métricas.
    assert T.main(["select", "--config", cfg_path]) == 0
    sel2 = json.loads((out / "selection_results.json").read_text(encoding="utf-8"))
    assert [r["validation"] for r in sel["results"]] == [r["validation"] for r in sel2["results"]]


# --- Congelación -------------------------------------------------------------------------
def test_freeze_copies_candidate_and_refuses_overwrite(sample_project, monkeypatch):
    monkeypatch.setattr(D, "_git_state", lambda: CLEAN)
    cfg = str(sample_project / "config.toml")
    assert T.main(["select", "--config", cfg]) == 0
    assert T.main(["freeze", "--config", cfg]) == 0
    spec = json.loads((sample_project / "models/candidate/spec.json").read_text(encoding="utf-8"))
    proposed = json.loads((sample_project / "reports/train/candidate_spec.json").read_text(encoding="utf-8"))
    assert spec["status"].startswith("CONGELADA") and spec["training_code_commit"] == CLEAN["commit"]
    assert spec["model_sha256"] == proposed["model_sha256"] == D.sha256_file(sample_project / "models/candidate/model.joblib")
    assert spec["threshold"] == proposed["threshold"] and spec["gate"]["max_fpr"] == 0.005
    assert T.main(["freeze", "--config", cfg]) == 1  # no sobrescribe un candidato congelado


@pytest.mark.parametrize("tamper", ["dirty", "config", "model"])
def test_freeze_rejects_inconsistent_candidate(sample_project, monkeypatch, tamper):
    monkeypatch.setattr(D, "_git_state", lambda: {**CLEAN, "dirty": tamper == "dirty"})
    cfg = sample_project / "config.toml"
    assert T.main(["select", "--config", str(cfg)]) == 0
    if tamper == "config":
        cfg.write_text(cfg.read_text(encoding="utf-8") + "\n# cambio\n", encoding="utf-8")
    if tamper == "model":
        with open(sample_project / "artifacts/train/candidate.joblib", "ab") as f:
            f.write(b"x")
    assert T.main(["freeze", "--config", str(cfg)]) == 1
    assert not (sample_project / "models/candidate/spec.json").exists()


# --- Evaluación única en test ------------------------------------------------------------
def test_evaluate_test_runs_once(frozen_project, monkeypatch):
    cfg = str(frozen_project / "config.toml")
    out = frozen_project / "reports/train/test_evaluation.json"
    assert T.main(["evaluate-test", "--config", cfg]) == 1  # congelado pero sin commitear
    assert not out.exists()
    commit_all(frozen_project)
    freeze_commit = subprocess.run(["git", "-C", str(frozen_project), "rev-parse", "HEAD"],
                                   capture_output=True, text=True, check=True).stdout.strip()
    # Árbol con cambios sin commitear: se rechaza.
    (frozen_project / "nota.txt").write_text("x", encoding="utf-8")
    assert T.main(["evaluate-test", "--config", cfg]) == 1
    (frozen_project / "nota.txt").unlink()
    # Prevalidación: no carga test ni crea el registro.
    real_load, calls = T.load_splits, []
    monkeypatch.setattr(T, "load_splits", lambda c, include_test=False: calls.append(include_test) or real_load(c, include_test))
    assert T.main(["evaluate-test", "--check-only", "--config", cfg]) == 0
    assert calls == [False] and not out.exists()
    assert T.main(["evaluate-test", "--config", cfg]) in (0, 3)  # 3 = gate no superado
    assert calls == [False, True]
    res = json.loads(out.read_text(encoding="utf-8"))
    sel = json.loads((frozen_project / "reports/train/selection_results.json").read_text(encoding="utf-8"))
    assert res["status"] == "EVALUADO" and res["validation_reproduced"]
    assert res["test"]["n"] == 2000 - sel["rows"]["train"] - sel["rows"]["validation"]
    assert res["freeze_commit"] == freeze_commit and res["training_code_commit"] == CLEAN["commit"]
    assert res["gate_passed"] == all(res["gate_checks"].values())
    assert T.main(["evaluate-test", "--config", cfg]) == 1  # segunda ejecución rechazada
    commit_all(frozen_project)
    out.unlink()
    assert T.main(["evaluate-test", "--config", cfg]) == 1  # borrar el registro no basta: consta en Git


@pytest.mark.parametrize("tamper", ["model", "config", "labels"])
def test_evaluate_test_rejects_tampering_before_scoring(frozen_project, tamper):
    target = {"model": "models/candidate/model.joblib", "config": "config.toml",
              "labels": "data/preprocessed/anomaly_label.csv"}[tamper]
    with open(frozen_project / target, "ab") as f:
        f.write(b"\n# x\n" if tamper == "config" else b"\n")
    commit_all(frozen_project)  # commiteado: así falla el hash, no la comprobación de Git
    assert T.main(["evaluate-test", "--config", str(frozen_project / "config.toml")]) == 1
    assert not (frozen_project / "reports/train/test_evaluation.json").exists()
