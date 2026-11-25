"""Entrenamiento con train y selección con validation. Test NO se usa aquí.

Comandos:
    python -m hdfs_anomaly.train select --families dummy,lr   # referencia + baseline
    python -m hdfs_anomaly.train pilot                        # RF pequeño para medir tiempos
    python -m hdfs_anomaly.train select                       # selección completa + diagnósticos
    python -m hdfs_anomaly.train freeze                       # congela el candidato aprobado
    python -m hdfs_anomaly.train evaluate-test                # evaluación ÚNICA en test (congelado)

Salidas: reports/train/*.json, reports/train/comparativa_validacion.md (versionados),
artifacts/train/candidate.joblib (propuesto, no versionado) y, tras 'freeze',
models/candidate/{model.joblib,spec.json} (versionados).
"""
import argparse
import gzip
import hashlib
import itertools
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from hdfs_anomaly import data as D
from hdfs_anomaly.config import DEFAULT_CONFIG, load_config

FAMILIES = ("dummy", "lr", "rf")


# --- Datos ------------------------------------------------------------------------
def pattern_keys(X: pd.DataFrame) -> np.ndarray:
    """Identificador de patrón = hash del vector de conteos completo."""
    return pd.util.hash_pandas_object(X, index=False).to_numpy()


def load_splits(cfg: dict, include_test: bool = False) -> dict:
    """Carga train y validation según el manifiesto de partición. Test solo se carga
    si se pide explícitamente (evaluación final, no implementada en esta fase)."""
    summary_path = cfg["output"]["reports"] / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["strategy"] != "completion":
        raise ValueError(f"El entrenamiento exige la partición 'completion' (hay '{summary['strategy']}')")
    manifest_path = cfg["output"]["artifacts"] / "split_manifest.csv.gz"
    body = gzip.decompress(manifest_path.read_bytes())
    if hashlib.sha256(body).hexdigest() != summary["manifest_sha256_uncompressed"]:
        raise ValueError("El manifiesto de partición no coincide con split_summary.json; repite python -m hdfs_anomaly.data")
    manifest = D.read_manifest(manifest_path).set_index(D.ID_COL)["split"]

    ids, X, y = D.prepare(D.load_matrix(cfg["data"]["matrix_path"]))
    part = manifest.reindex(ids).to_numpy()
    if pd.isna(part).any():
        raise ValueError("Hay bloques sin partición en el manifiesto")
    wanted = ("train", "validation", "test") if include_test else ("train", "validation")
    out = {s: {"ids": ids[part == s].reset_index(drop=True), "X": X[part == s].reset_index(drop=True),
               "y": y[part == s], "pos": np.flatnonzero(part == s)} for s in wanted}
    out["manifest_sha256"] = summary["manifest_sha256_uncompressed"]
    out["sources"] = summary["sources"]
    return out


# --- Modelos ------------------------------------------------------------------------
def _none(v):
    return None if v == "none" else v


def build_model(family: str, params: dict, seed: int, n_jobs: int = 1) -> Pipeline:
    """Todas las transformaciones viven dentro del Pipeline y se ajustan solo con train."""
    if family == "dummy":
        clf = DummyClassifier(strategy="prior")
        return Pipeline([("clf", clf)])
    if family == "lr":
        return Pipeline([
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=params["C"], class_weight=_none(params["class_weight"]),
                                       max_iter=2000, random_state=seed)),
        ])
    if family == "rf":
        return Pipeline([("clf", RandomForestClassifier(
            n_estimators=params["n_estimators"], max_depth=_none(params["max_depth"]),
            min_samples_leaf=params["min_samples_leaf"], class_weight=_none(params["class_weight"]),
            n_jobs=n_jobs, random_state=seed))])
    raise ValueError(f"Familia desconocida: {family}")


def fit_model(family: str, params: dict, X, y, seed: int, n_jobs: int = 1) -> Pipeline:
    """Entrena en paralelo y deja el modelo en un solo hilo para predecir: con varios hilos,
    el RF suma las probabilidades de los árboles en orden variable y cambian los últimos bits."""
    model = build_model(family, params, seed, n_jobs).fit(X, y)
    if family == "rf":
        model.set_params(clf__n_jobs=1)
    return model


def grid(family: str, tcfg: dict, pilot: bool = False) -> list[dict]:
    if family == "dummy":
        return [{}]
    g = dict(tcfg["pilot"][family] if pilot else tcfg["grid"][family])
    if family == "rf":
        g["n_estimators"] = [g.pop("n_estimators", tcfg["rf_n_estimators"])]
    keys = sorted(g)
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(g[k] for k in keys))]


def simplicity_key(family: str, p: dict) -> tuple:
    """Menor = más sencillo (desempate entre PR-AUC iguales)."""
    if family == "lr":
        return (p["C"], p["class_weight"] != "none")
    if family == "rf":
        depth = float("inf") if p["max_depth"] == "none" else p["max_depth"]
        return (depth, -p["min_samples_leaf"], p["class_weight"] != "none")
    return ()


# --- Métricas y umbral ----------------------------------------------------------
def threshold_table(y: np.ndarray, score: np.ndarray) -> pd.DataFrame:
    """Para cada umbral distinto (predicción positiva si score >= umbral): TP, FP,
    precision, recall, F1 y FPR. El umbral se coloca a mitad de camino entre dos
    puntuaciones distintas consecutivas para que no dependa de igualdades exactas."""
    y = np.asarray(y, dtype=int)
    order = np.argsort(-score, kind="stable")
    s, t = score[order], y[order]
    last = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp, fp = np.cumsum(t)[last], np.cumsum(1 - t)[last]
    nxt = np.r_[s[last][1:], s[-1] - 1.0]
    thr = (s[last] + nxt) / 2
    P, N = t.sum(), len(t) - t.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = tp / P if P else np.zeros_like(tp, dtype=float)
        fpr = fp / N if N else np.zeros_like(fp, dtype=float)
        f1 = np.where(2 * tp + fp + (P - tp) > 0, 2 * tp / (2 * tp + fp + (P - tp)), 0.0)
    return pd.DataFrame({"threshold": thr, "tp": tp, "fp": fp, "precision": prec,
                         "recall": rec, "f1": f1, "fpr": fpr})


def choose_threshold(y: np.ndarray, score: np.ndarray, max_fpr: float) -> dict:
    """Umbral que maximiza F1 entre los que cumplen FPR <= max_fpr (desempate: menor FPR).
    'constraint_met' es False si ningún umbral con FPR permitido detecta alguna anomalía."""
    tab = threshold_table(y, score)
    ok = tab[tab["fpr"] <= max_fpr].sort_values(["f1", "fpr"], ascending=[False, True])
    free = tab.sort_values(["f1", "fpr"], ascending=[False, True]).iloc[0]
    if ok.empty or ok.iloc[0]["tp"] == 0:
        best = {"threshold": float(score.max()) + 1.0, "constraint_met": False}
    else:
        best = {"threshold": float(ok.iloc[0]["threshold"]), "constraint_met": True}
    best["unconstrained_best"] = {k: float(free[k]) for k in ("threshold", "f1", "fpr", "precision", "recall")}
    return best


def evaluate(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, dtype=int)
    pred = (score >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    both = 0 < y.sum() < len(y)
    return {
        "n": len(y), "positives": int(y.sum()),
        "pr_auc": round(float(average_precision_score(y, score)), 6) if both else None,
        "roc_auc": round(float(roc_auc_score(y, score)), 6) if both else None,
        "precision": round(tp / (tp + fp), 6) if tp + fp else 0.0,
        "recall": round(tp / (tp + fn), 6) if tp + fn else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 6) if tp + fp + fn else None,
        "fpr": round(fp / (fp + tn), 6) if fp + tn else None,
        "mcc": round(float(matthews_corrcoef(y, pred)), 6) if both else None,
        "accuracy": round((tp + tn) / len(y), 6),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# --- Selección --------------------------------------------------------------------
def fit_and_score(family, params, tr, va, seed, n_jobs, max_fpr) -> tuple[dict, Pipeline, np.ndarray]:
    t0 = time.perf_counter()
    model = fit_model(family, params, tr["X"], tr["y"], seed, n_jobs)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    score = model.predict_proba(va["X"])[:, 1]
    pred_s = time.perf_counter() - t0
    thr = choose_threshold(va["y"], score, max_fpr)
    train_score = model.predict_proba(tr["X"])[:, 1]
    row = {
        "family": family, "params": params, "seed": seed,
        "fit_seconds": round(fit_s, 3), "predict_seconds_validation": round(pred_s, 3),
        "threshold": thr["threshold"], "constraint_met": thr["constraint_met"],
        "unconstrained_best": thr["unconstrained_best"],
        "validation": evaluate(va["y"], score, thr["threshold"]),
        "train": evaluate(tr["y"], train_score, thr["threshold"]),
    }
    return row, model, score


def pick_best(rows: list[dict], family: str, decimals: int) -> dict:
    fam = [r for r in rows if r["family"] == family]
    return min(fam, key=lambda r: (-round(r["validation"]["pr_auc"] or 0, decimals),
                                   simplicity_key(family, r["params"])))


def decide_candidate(lr: dict, rf: dict, min_f1_gain: float) -> tuple[str, str]:
    """Regla única: el RF sustituye a la LR solo si mejora F1 en >= min_f1_gain,
    no empeora la PR-AUC y cumple la restricción de FPR. Si no, se conserva la LR."""
    lv, rv = lr["validation"], rf["validation"]
    gain = (rv["f1"] or 0) - (lv["f1"] or 0)
    if rf["constraint_met"] and gain >= min_f1_gain and (rv["pr_auc"] or 0) >= (lv["pr_auc"] or 0):
        return "rf", f"El RF mejora F1 en {gain:+.4f} (≥ {min_f1_gain}), no empeora la PR-AUC y cumple FPR."
    note = "" if lr["constraint_met"] else " La LR no cumple la restricción de FPR: candidato no aprobable."
    return "lr", (f"El RF no reúne las tres condiciones (ΔF1 = {gain:+.4f}, mínimo {min_f1_gain}; "
                  f"PR-AUC no inferior; FPR): se conserva la LR.{note}")


# --- Diagnósticos -------------------------------------------------------------------
def seen_unseen(tr: dict, va: dict, score: np.ndarray, threshold: float) -> dict:
    """Métricas en validation separando filas cuyo patrón aparece en train de las de patrón nuevo."""
    seen = np.isin(pattern_keys(va["X"]), np.unique(pattern_keys(tr["X"])))
    out = {}
    for name, m in (("seen_in_train", seen), ("unseen_in_train", ~seen)):
        r = evaluate(va["y"][m], score[m], threshold) if m.any() else {"n": 0}
        r["distinct_patterns"] = len(np.unique(pattern_keys(va["X"][m]))) if m.any() else 0
        out[name] = r
    return out


def grouped_cv(family, params, tr, seed, n_jobs, folds, threshold) -> dict:
    """StratifiedGroupKFold dentro de train agrupando por patrón: cada pliegue evalúa solo
    patrones que su modelo no ha visto. Comprueba que cada pliegue tiene ambas clases."""
    groups = pattern_keys(tr["X"])
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    res = []
    for k, (a, b) in enumerate(cv.split(tr["X"], tr["y"], groups)):
        yb = tr["y"][b]
        fold = {"fold": k, "rows": len(b), "positives": int(yb.sum()),
                "patterns": len(np.unique(groups[b])),
                "both_classes": bool(0 < yb.sum() < len(yb)),
                "train_has_both_classes": bool(0 < tr["y"][a].sum() < len(a))}
        if fold["both_classes"] and fold["train_has_both_classes"]:
            m = fit_model(family, params, tr["X"].iloc[a], tr["y"][a], seed, n_jobs)
            fold.update(evaluate(yb, m.predict_proba(tr["X"].iloc[b])[:, 1], threshold))
        res.append(fold)
    valid = [f for f in res if "pr_auc" in f]
    agg = {k: round(float(np.mean([f[k] for f in valid if f[k] is not None])), 6)
           for k in ("pr_auc", "precision", "recall", "f1", "fpr")} if valid else {}
    return {"family": family, "params": params, "threshold_applied": threshold, "folds": res,
            "folds_evaluated": len(valid), "mean": agg}


def sensitivity(best: dict, tr: dict, va: dict, seed: int, n_jobs: int, max_fpr: float) -> dict:
    """Diagnóstico independiente: se reparte SOLO el conjunto de desarrollo (train ∪ validation
    de 'completion'; test queda fuera) con otras dos definiciones, en 75/25, y cada protocolo
    elige su propio umbral en su propia parte de evaluación. No interviene en la selección."""
    X = pd.concat([tr["X"], va["X"]], ignore_index=True)
    y = np.r_[tr["y"], va["y"]]
    pos = np.r_[tr["pos"], va["pos"]]  # posición en la matriz = orden de primera aparición
    n = len(y)
    first_seen = np.zeros(n, bool)
    first_seen[np.argsort(pos, kind="stable")[: int(0.75 * n)]] = True
    stratified = np.zeros(n, bool)
    stratified[train_test_split(np.arange(n), train_size=0.75, stratify=y, random_state=seed)[0]] = True
    out = {}
    for name, is_train in (("first_seen_dev", first_seen), ("stratified_dev", stratified)):
        a = {"X": X[is_train], "y": y[is_train]}
        b = {"X": X[~is_train], "y": y[~is_train]}
        for fam in ("lr", "rf"):
            row, _, _ = fit_and_score(fam, best[fam]["params"], a, b, seed, n_jobs, max_fpr)
            out[f"{name}/{fam}"] = {"train_rows": int(is_train.sum()), "eval_rows": int((~is_train).sum()),
                                    "eval_positives": int(b["y"].sum()), "threshold": row["threshold"],
                                    "constraint_met": row["constraint_met"], "validation": row["validation"]}
    return out


def error_analysis(va: dict, tr: dict, score: np.ndarray, threshold: float, top: int = 10) -> dict:
    """Agrupa falsos positivos y falsos negativos de validation por patrón."""
    keys = pattern_keys(va["X"])
    seen = set(np.unique(pattern_keys(tr["X"])))
    pred = score >= threshold
    df = pd.DataFrame({"key": keys, "y": va["y"], "pred": pred})
    # Patrones ambiguos dentro de validation: error irreducible con estas variables.
    lab = df.groupby("key")["y"].agg(["sum", "count"])
    amb = lab[(lab["sum"] > 0) & (lab["sum"] < lab["count"])]
    out = {"irreducible_errors_validation": int(np.minimum(amb["sum"], amb["count"] - amb["sum"]).sum()),
           "ambiguous_patterns_validation": len(amb)}
    for name, m in (("false_positives", pred & (va["y"] == 0)), ("false_negatives", ~pred & (va["y"] == 1))):
        g = df[m].groupby("key").size().sort_values(ascending=False)
        rows = []
        for key, cnt in g.head(top).items():
            first = va["X"][keys == key].iloc[0]
            rows.append({"count": int(cnt), "pattern_seen_in_train": bool(key in seen),
                         "pattern_rows_in_validation": int(lab.loc[key, "count"]),
                         "pattern_anomaly_rate_validation": round(float(lab.loc[key, "sum"] / lab.loc[key, "count"]), 4),
                         "events": {c: int(v) for c, v in first.items() if v}})
        out[name] = {"total": int(m.sum()), "distinct_patterns": len(g), "top": rows}
    return out


# --- Informes -------------------------------------------------------------------------
def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8",
                    newline="\n")


def _sha(path: Path) -> str:
    return D.sha256_file(path)


def comparison_markdown(rows: list[dict], best: dict, candidate: str, reason: str, max_fpr: float) -> str:
    def fmt(v, p=4):
        return "—" if v is None else f"{v:.{p}f}"
    lines = [
        "# Comparativa en validation",
        "",
        f"Partición `completion`. Umbral elegido en validation: F1 máximo con FPR ≤ {max_fpr:.2%}.",
        "Test no se ha evaluado.",
        "",
        "## Mejor configuración por familia",
        "",
        "| Modelo | Parámetros | PR-AUC | Precision | Recall | F1 | FPR | TP | FP | FN | Umbral | ¿Cumple FPR? | Ajuste (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for fam in FAMILIES:
        if fam not in best:
            continue
        r = best[fam]
        v = r["validation"]
        cm = v["confusion_matrix"]
        lines.append(f"| {fam} | `{json.dumps(r['params'])}` | {fmt(v['pr_auc'])} | {fmt(v['precision'])} | "
                     f"{fmt(v['recall'])} | {fmt(v['f1'])} | {fmt(v['fpr'], 5)} | {cm['tp']} | {cm['fp']} | "
                     f"{cm['fn']} | {r['threshold']:.4f} | {'sí' if r['constraint_met'] else '**no**'} | "
                     f"{r['fit_seconds']:.1f} |")
    lines += ["", f"**Candidato:** `{candidate}`. {reason}", "",
              "## Todas las configuraciones", "",
              "| Modelo | Parámetros | PR-AUC val | F1 val | FPR val | PR-AUC train | ¿Cumple FPR? | Ajuste (s) |",
              "|---|---|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        v, t = r["validation"], r["train"]
        lines.append(f"| {r['family']} | `{json.dumps(r['params'])}` | {fmt(v['pr_auc'])} | {fmt(v['f1'])} | "
                     f"{fmt(v['fpr'], 5)} | {fmt(t['pr_auc'])} | {'sí' if r['constraint_met'] else 'no'} | "
                     f"{r['fit_seconds']:.1f} |")
    return "\n".join(lines) + "\n"


def run_select(cfg: dict, families: list[str], pilot: bool = False, diagnostics: bool = True) -> dict:
    tcfg, seed = cfg["train"], cfg["split"]["seed"]
    out_dir = cfg["output"]["reports"].parent / "train"
    art_dir = cfg["output"]["artifacts"].parent / "train"
    s = load_splits(cfg)
    tr, va = s["train"], s["validation"]
    rows, models, scores = [], {}, {}
    for fam in families:
        for params in grid(fam, tcfg, pilot and fam == "rf"):
            row, model, score = fit_and_score(fam, params, tr, va, seed, tcfg["n_jobs"], tcfg["max_fpr"])
            rows.append(row)
            key = (fam, json.dumps(params, sort_keys=True))
            models[key], scores[key] = model, score
            v = row["validation"]
            print(f"{fam:<5} {json.dumps(params):<85} PR-AUC={v['pr_auc']} F1={v['f1']} "
                  f"FPR={v['fpr']} fit={row['fit_seconds']}s", flush=True)

    meta = {"config_sha256": _sha(cfg["_path"]), "train_config": tcfg, "seed": seed,
            "partition": "completion", "manifest_sha256": s["manifest_sha256"], "sources": s["sources"],
            "rows": {"train": len(tr["y"]), "validation": len(va["y"])},
            "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "pandas": pd.__version__, "numpy": np.__version__},
            "git": D._git_state(), "test_used": False}
    if pilot:
        _write_json(out_dir / "rf_pilot.json", {**meta, "results": rows})
        return {"rows": rows}
    best = {f: pick_best(rows, f, tcfg["tie_decimals"]) for f in families}
    result = {**meta, "results": rows, "best_per_family": {f: best[f]["params"] for f in best}}
    if not set(FAMILIES) <= set(families) or not diagnostics:
        _write_json(out_dir / "selection_results.json", result)
        return result

    candidate, reason = decide_candidate(best["lr"], best["rf"], tcfg["rf_min_f1_gain"])
    result.update({"candidate_family": candidate, "candidate_reason": reason})
    _write_json(out_dir / "selection_results.json", result)
    (out_dir / "comparativa_validacion.md").write_text(
        comparison_markdown(rows, best, candidate, reason, tcfg["max_fpr"]), encoding="utf-8")

    def key(f):
        return (f, json.dumps(best[f]["params"], sort_keys=True))

    patterns = {f: {"threshold": best[f]["threshold"],
                    **seen_unseen(tr, va, scores[key(f)], best[f]["threshold"])} for f in FAMILIES}
    print("diagnóstico: validación cruzada agrupada por patrón...", flush=True)
    cv = {f: grouped_cv(f, best[f]["params"], tr, seed, tcfg["n_jobs"], tcfg["cv_folds"], best[f]["threshold"])
          for f in ("lr", "rf")}
    _write_json(out_dir / "pattern_diagnostics.json", {**meta, "validation_seen_unseen": patterns,
                                                       "grouped_cv_in_train": cv})
    _write_json(out_dir / "error_analysis.json",
                {**meta, "family": candidate, "threshold": best[candidate]["threshold"],
                 **error_analysis(va, tr, scores[key(candidate)], best[candidate]["threshold"])})
    print("diagnóstico: sensibilidad a la definición de partición...", flush=True)
    _write_json(out_dir / "sensitivity.json",
                {**meta, "note": "Solo train ∪ validation de 'completion'; test excluido. Diagnóstico.",
                 "results": sensitivity(best, tr, va, seed, tcfg["n_jobs"], tcfg["max_fpr"])})

    art_dir.mkdir(parents=True, exist_ok=True)
    model_path = art_dir / "candidate.joblib"
    joblib.dump(models[key(candidate)], model_path)
    spec = {"status": "PROPUESTA — pendiente de aprobación; NO congelada; test no evaluado",
            "family": candidate, "params": best[candidate]["params"], "seed": seed,
            "threshold": best[candidate]["threshold"], "max_fpr": tcfg["max_fpr"],
            "features": D.EVENT_COLS, "validation_metrics": best[candidate]["validation"],
            "baseline_lr_validation_metrics": best["lr"]["validation"],
            "model_sha256": _sha(model_path), "model_path": model_path.relative_to(model_path.parents[2]).as_posix(),
            **{k: meta[k] for k in ("config_sha256", "manifest_sha256", "sources", "environment", "git")}}
    _write_json(out_dir / "candidate_spec.json", spec)
    return result


# --- Congelación --------------------------------------------------------------------
FROZEN = "CONGELADA — aprobada para una única evaluación en test"


def freeze(cfg: dict) -> dict:
    """Copia el candidato propuesto a la carpeta versionada (output.models) con una
    especificación congelada. Exige que la selección se ejecutara desde un commit limpio,
    que config.toml y el modelo no hayan cambiado desde entonces y que no haya otro congelado."""
    spec = json.loads((cfg["output"]["reports"].parent / "train" / "candidate_spec.json").read_text(encoding="utf-8"))
    src = cfg["output"]["artifacts"].parent / "train" / "candidate.joblib"
    mdir = cfg["output"]["models"]
    if (mdir / "spec.json").exists():
        raise ValueError(f"Ya hay un candidato congelado en {mdir}; no se sobrescribe")
    if not spec["git"]["commit"] or spec["git"]["dirty"] is not False:
        raise ValueError("La selección no se ejecutó desde un commit limpio; commitea el código y repite 'select'")
    if _sha(cfg["_path"]) != spec["config_sha256"]:
        raise ValueError("config.toml ha cambiado desde la selección; repite 'select'")
    if _sha(src) != spec["model_sha256"]:
        raise ValueError("El modelo no coincide con el SHA-256 de candidate_spec.json")
    mdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, mdir / "model.joblib")
    frozen = {"status": FROZEN,
              **{k: v for k, v in spec.items() if k not in ("status", "model_path", "git", "max_fpr")},
              "model_file": "model.joblib", "model_sha256": _sha(mdir / "model.joblib"),
              "threshold_rule": f"F1 máximo en validation con FPR <= {spec['max_fpr']}",
              "gate": cfg["gate"], "trained_on": "train de 'completion' (sin reentrenar con validation)",
              "training_code_commit": spec["git"]["commit"],
              "note": "El commit que congela este archivo es posterior a training_code_commit y no se "
                      "registra aquí (evita una referencia circular); evaluate-test lo obtiene de Git."}
    _write_json(mdir / "spec.json", frozen)
    return frozen


# --- Evaluación final única en test ------------------------------------------------------
def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True, text=True)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def evaluate_test(cfg: dict, check_only: bool = False) -> dict:
    """Evaluación ÚNICA del candidato congelado en test. Todas las
    comprobaciones se hacen antes de deserializar el modelo y de puntuar test; el registro
    se crea en modo exclusivo antes de puntuar, así que una ejecución interrumpida también cuenta.
    Con check_only=True (prevalidación) hace las mismas comprobaciones sin cargar test ni
    crear el registro."""
    root = cfg["_path"].resolve().parent
    mdir = cfg["output"]["models"]
    spec_path, model_path = mdir / "spec.json", mdir / "model.joblib"
    out = cfg["output"]["reports"].parent / "train" / "test_evaluation.json"
    rel = {p: p.relative_to(root).as_posix() for p in (spec_path, model_path, out)}

    if out.exists() or _git(root, "log", "--all", "--format=%H", "--", rel[out]).stdout.strip():
        raise ValueError(f"Test ya se evaluó: {rel[out]} existe o existió en Git. Una nueva evaluación "
                         "exige un experimento nuevo con datos de test nuevos.")
    if not spec_path.is_file():
        raise FileNotFoundError("No hay candidato congelado: ejecuta 'freeze' y commitea models/candidate/")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("status") != FROZEN:
        raise ValueError("La especificación de models/candidate no está congelada")
    for p in (spec_path, model_path):
        if (_git(root, "ls-files", "--error-unmatch", rel[p]).returncode
                or _git(root, "status", "--porcelain", "--", rel[p]).stdout.strip()):
            raise ValueError(f"{rel[p]} no está commiteado o tiene cambios sin commitear")
    summary = json.loads((cfg["output"]["reports"] / "split_summary.json").read_text(encoding="utf-8"))
    checks = {
        "model_sha256": (_sha(model_path), spec["model_sha256"]),
        "config_sha256": (_sha(cfg["_path"]), spec["config_sha256"]),
        "matrix_sha256": (_sha(cfg["data"]["matrix_path"]),
                          spec["sources"][cfg["data"]["matrix_path"].name]["sha256"]),
        "labels_sha256": (_sha(cfg["data"]["labels_path"]),
                          spec["sources"][cfg["data"]["labels_path"].name]["sha256"]),
        "manifest_sha256": (summary["manifest_sha256_uncompressed"], spec["manifest_sha256"]),
    }
    bad = [k for k, (got, expected) in checks.items() if got != expected]
    if bad:
        raise ValueError(f"No coinciden con la especificación congelada: {', '.join(bad)}")
    evaluation_git = {"commit": _git(root, "rev-parse", "HEAD").stdout.strip(),
                      "dirty": bool(_git(root, "status", "--porcelain").stdout.strip())}
    if evaluation_git["dirty"]:
        raise ValueError("El árbol de trabajo tiene cambios sin commitear; la evaluación debe partir de un commit limpio")
    freeze_commit = _git(root, "log", "-1", "--format=%H", "--", rel[spec_path]).stdout.strip()

    s = load_splits(cfg, include_test=not check_only)  # vuelve a verificar el manifiesto contra split_summary
    model = joblib.load(model_path)  # solo después de verificar su SHA-256
    thr = spec["threshold"]
    tr, va = s["train"], s["validation"]
    if evaluate(va["y"], model.predict_proba(va["X"])[:, 1], thr) != spec["validation_metrics"]:
        raise ValueError("El modelo no reproduce las métricas de validation de la especificación")
    if check_only:
        return {"status": "PREVALIDACIÓN SUPERADA (test no cargado)", "threshold": thr, "gate": spec["gate"],
                "integrity_verified": {k: got for k, (got, _) in checks.items()}, "validation_reproduced": True,
                "training_code_commit": spec["training_code_commit"], "freeze_commit": freeze_commit,
                "evaluation_git": evaluation_git, "single_run_guard": f"{rel[out]} no existe ni consta en Git"}
    te = s["test"]

    record = {"status": "EN CURSO", "started_at": _now(), "frozen_spec_sha256": _sha(spec_path)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8", newline="\n") as f:
        json.dump(record, f)

    score = model.predict_proba(te["X"])[:, 1]
    m = evaluate(te["y"], score, thr)
    g = spec["gate"]
    gate = {"recall": (m["recall"] or 0) >= g["min_recall"],
            "precision": m["precision"] >= g["min_precision"],
            "fpr": m["fpr"] is not None and m["fpr"] <= g["max_fpr"]}
    result = {**record, "status": "EVALUADO", "finished_at": _now(),
              "family": spec["family"], "params": spec["params"], "threshold": thr,
              "gate": g, "gate_checks": gate, "gate_passed": all(gate.values()),
              "test": m, "test_seen_unseen": seen_unseen(tr, te, score, thr),
              "integrity_verified": {k: got for k, (got, _) in checks.items()}, "validation_reproduced": True,
              "training_code_commit": spec["training_code_commit"],
              "freeze_commit": freeze_commit, "evaluation_git": evaluation_git,
              "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "pandas": pd.__version__, "numpy": np.__version__}}
    _write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["select", "pilot", "freeze", "evaluate-test"])
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="evaluate-test: solo prevalidación, sin cargar test")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    cfg["_path"] = Path(args.config) if args.config else DEFAULT_CONFIG
    fams = ["rf"] if args.command == "pilot" else [f.strip() for f in args.families.split(",")]
    if not set(fams) <= set(FAMILIES):
        print(f"ERROR: familias válidas {FAMILIES}", file=sys.stderr)
        return 2
    try:
        if args.command == "freeze":
            f = freeze(cfg)
            print(f"Congelado: {f['family']} {f['params']} umbral={f['threshold']:.4f} sha={f['model_sha256']}")
        elif args.command == "evaluate-test" and args.check_only:
            print(json.dumps(evaluate_test(cfg, check_only=True), indent=2, ensure_ascii=False))
        elif args.command == "evaluate-test":
            r = evaluate_test(cfg)
            t = r["test"]
            print(f"Test: precision={t['precision']} recall={t['recall']} F1={t['f1']} FPR={t['fpr']} "
                  f"PR-AUC={t['pr_auc']} → gate {'SUPERADO' if r['gate_passed'] else 'NO superado'} {r['gate_checks']}")
            return 0 if r["gate_passed"] else 3
        else:
            run_select(cfg, fams, pilot=args.command == "pilot", diagnostics=not args.no_diagnostics)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
