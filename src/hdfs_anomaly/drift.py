"""Monitorización predictiva (cambios de distribución) con PSI.

    python -m hdfs_anomaly.drift reference                  # referencia desde TRAIN (requiere dataset)
    python -m hdfs_anomaly.drift report VENTANA.csv --output informe.json

Una ventana es un CSV con E1–E29 y, opcionalmente, las columnas anomaly_score y prediction
devueltas por la API. El informe es JSON determinista: mismas entradas → mismos bytes.

Por qué PSI (Population Stability Index): las variables son conteos discretos con muchos
ceros y empates. PSI compara proporciones por intervalos, no necesita etiquetas, es
comparable entre variables y tiene umbrales de lectura habituales (0,1 y 0,25). KS asume
continuidad y, con miles de filas, su p-valor marca diferencias irrelevantes; chi-cuadrado
tiene el mismo problema de tamaño. PSI con suavizado y agrupación de intervalos raros evita
divisiones por cero e inestabilidad en eventos poco frecuentes.

Un PSI alto indica un CAMBIO de distribución, no una degradación del modelo: precision y
recall en operación solo se pueden calcular cuando llegan etiquetas reales posteriores.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hdfs_anomaly.data import EVENT_COLS, FORBIDDEN_FEATURES

# Intervalos de conteo [a, b): 0, 1, 2, 3, 4, 5–9, 10–19, ≥20
COUNT_EDGES = [0, 1, 2, 3, 4, 5, 10, 20, math.inf]
SCORE_EDGES = [0.0, 0.001, 0.01, 0.05, None, 0.25, 0.5, 0.75, 0.9, 0.99, 1.000001]  # None = umbral del modelo
MIN_BIN = 0.005        # intervalos con <0,5 % en la referencia se agrupan en «raros»
SMOOTH = 0.5           # suavizado (pseudoconteo) para evitar log(0)
MIN_N, SMALL_N = 500, 2000   # tamaño mínimo para alertar / ventanas «pequeñas»
PSI_WARN, PSI_ALERT = 0.10, 0.25
RATE_Z, RATE_ABS = 4.0, 0.01       # cambio de tasa predicha: |z| ≥ 4 y |Δ| ≥ 1 punto
NOVEL_WARN, NOVEL_ALERT = 0.02, 0.05  # proporción de patrones no vistos en train
RULES = {
    "insufficient_sample": f"n < {MIN_N}: se calculan indicadores pero no se emiten alertas de distribución.",
    "psi_feature": f"PSI de una característica ≥ {PSI_WARN} → aviso; ≥ {PSI_ALERT} → alerta "
                   f"(si {MIN_N} ≤ n < {SMALL_N}, solo se emiten alertas ≥ {PSI_ALERT}).",
    "psi_score": f"PSI de la puntuación ≥ {PSI_WARN} → aviso; ≥ {PSI_ALERT} → alerta.",
    "predicted_rate": f"Tasa de anomalías predichas: |z| ≥ {RATE_Z} frente a la referencia y |Δ| ≥ {RATE_ABS:.0%}.",
    "novel_patterns": f"Patrones no vistos en train ≥ {NOVEL_WARN:.0%} → aviso; ≥ {NOVEL_ALERT:.0%} → alerta "
                      "(el modelo fue poco fiable con patrones nuevos).",
    "schema": "Columnas ausentes/extra o prohibidas, nulos, no enteros, negativos, valores por encima del "
              "máximo de train o desviaciones en columnas constantes → aviso; columnas ausentes, "
              "prohibidas o negativas → alerta.",
}
NOTE = ("Un PSI alto o un cambio de tasa indican un CAMBIO de distribución, no una degradación demostrada. "
        "Precision y recall en operación solo se pueden calcular con etiquetas reales posteriores. "
        "Las alertas sirven para decidir si investigar o reentrenar, no para reentrenar automáticamente.")


def pattern_hash(row) -> str:
    return hashlib.sha1(",".join(str(int(v)) for v in row).encode()).hexdigest()[:16]


def _hist(values: np.ndarray, edges: list) -> np.ndarray:
    idx = np.searchsorted(np.asarray(edges[1:], dtype=float), values, side="right")
    return np.bincount(np.clip(idx, 0, len(edges) - 2), minlength=len(edges) - 1)


def _groups(ref_props: list[float]) -> list[list[int]]:
    """Intervalos estables de la referencia + un grupo «raros» con los que quedan por debajo de MIN_BIN."""
    keep = [[i] for i, p in enumerate(ref_props) if p >= MIN_BIN]
    rare = [i for i, p in enumerate(ref_props) if p < MIN_BIN]
    return keep + ([rare] if rare else [])


def psi(ref_counts: list[int], cur_counts: list[int], groups: list[list[int]]) -> float:
    r = np.array([sum(ref_counts[i] for i in g) for g in groups], dtype=float)
    c = np.array([sum(cur_counts[i] for i in g) for g in groups], dtype=float)
    k = len(groups)
    p = (r + SMOOTH) / (r.sum() + SMOOTH * k)
    q = (c + SMOOTH) / (c.sum() + SMOOTH * k)
    return float(np.sum((q - p) * np.log(q / p)))


def _score_edges(threshold: float) -> list[float]:
    return sorted({threshold if e is None else e for e in SCORE_EDGES})


# --- Referencia (solo TRAIN) ----------------------------------------------------------------
def build_reference(X: pd.DataFrame, scores: np.ndarray, threshold: float, provenance: dict) -> dict:
    feats = {}
    for c in EVENT_COLS:
        v = X[c].to_numpy()
        counts = _hist(v, COUNT_EDGES).tolist()
        props = [n / len(v) for n in counts]
        feats[c] = {"counts": counts, "groups": _groups(props), "max": int(v.max()),
                    "constant": int(v[0]) if (v == v[0]).all() else None,
                    "nonzero_rate": round(float((v > 0).mean()), 6)}
    s_edges = _score_edges(threshold)
    s_counts = _hist(scores, s_edges).tolist()
    pats = X[EVENT_COLS].value_counts()
    return {"source": "train", "rows": len(X), "provenance": provenance, "threshold": threshold,
            "count_edges": [e if e != math.inf else "inf" for e in COUNT_EDGES], "features": feats,
            "score_edges": s_edges, "score_counts": s_counts,
            "score_groups": _groups([n / len(scores) for n in s_counts]),
            "predicted_anomaly_rate": round(float((scores >= threshold).mean()), 6),
            "pattern_hashes": sorted(pattern_hash(p) for p in pats.index),
            "top_patterns": [{"events": dict(zip(EVENT_COLS, map(int, p), strict=True)), "count": int(n)}
                             for p, n in pats.head(200).items()]}


# --- Informe de una ventana ----------------------------------------------------------------------
def schema_issues(df: pd.DataFrame, ref: dict) -> dict:
    cols = set(df.columns)
    allowed = set(EVENT_COLS) | {"anomaly_score", "prediction"}
    issues = {"missing_columns": sorted(set(EVENT_COLS) - cols),
              "forbidden_columns": sorted(cols & FORBIDDEN_FEATURES),
              "unexpected_columns": sorted(cols - allowed - FORBIDDEN_FEATURES)}
    present = [c for c in EVENT_COLS if c in cols]
    X = df[present].apply(pd.to_numeric, errors="coerce")
    issues["null_or_non_numeric"] = {c: int(n) for c, n in X.isna().sum().items() if n}
    issues["non_integer"] = {c: int(n) for c, n in ((X % 1) != 0).where(X.notna(), False).sum().items() if n}
    issues["negative"] = {c: int(n) for c, n in (X < 0).sum().items() if n}
    issues["above_train_max"] = {c: int(n) for c in present
                                 if (n := int((X[c] > ref["features"][c]["max"]).sum()))}
    issues["constant_deviation"] = {c: int(n) for c in present if ref["features"][c]["constant"] is not None
                                    and (n := int((X[c] != ref["features"][c]["constant"]).sum()))}
    issues["all_zero_rows"] = int((X.fillna(0) == 0).all(axis=1).sum()) if present else 0
    return issues


def report(df: pd.DataFrame, ref: dict, window: str = "window") -> dict:
    n = len(df)
    schema = schema_issues(df, ref)
    alerts = []

    def add(rule, severity, msg):
        alerts.append({"rule": rule, "severity": severity, "message": msg})

    if schema["missing_columns"] or schema["forbidden_columns"] or schema["negative"]:
        add("schema", "alert", "Columnas ausentes o prohibidas, o conteos negativos")
    elif any(schema[k] for k in ("unexpected_columns", "null_or_non_numeric", "non_integer",
                                 "above_train_max", "constant_deviation")):
        add("schema", "warning", "Valores fuera del contrato o del rango de train")

    size = "insufficient" if n < MIN_N else ("small" if n < SMALL_N else "ok")
    can_alert = size != "insufficient" and not schema["missing_columns"]
    if size == "insufficient":
        add("insufficient_sample", "info", f"Solo {n} filas: no se emiten alertas de distribución")

    def level(value):
        if not can_alert or value < PSI_WARN:
            return None
        if value >= PSI_ALERT:
            return "alert"
        return "warning" if size == "ok" else None

    feats = {}
    if not schema["missing_columns"]:
        X = df[EVENT_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        for c in EVENT_COLS:
            r = ref["features"][c]
            cur = _hist(X[c].to_numpy(), COUNT_EDGES).tolist()
            entry = {"nonzero_rate_ref": r["nonzero_rate"], "nonzero_rate": round(float((X[c] > 0).mean()), 6)}
            if r["constant"] is not None:
                entry.update(psi=None, status="constant_in_reference")
            else:
                entry["psi"] = round(psi(r["counts"], cur, r["groups"]), 6)
                entry["status"] = level(entry["psi"]) or "stable"
                if entry["status"] != "stable":
                    add("psi_feature", entry["status"], f"{c}: PSI = {entry['psi']:.3f}")
            feats[c] = entry
        keys = [pattern_hash(row) for row in X[EVENT_COLS].to_numpy()]
        known = set(ref["pattern_hashes"])
        novel = float(np.mean([k not in known for k in keys])) if keys else 0.0
        novel_info = {"rate": round(novel, 6), "distinct_patterns": len(set(keys))}
        if can_alert and novel >= NOVEL_WARN:
            add("novel_patterns", "alert" if novel >= NOVEL_ALERT else "warning",
                f"{novel:.1%} de filas con patrones no vistos en train")
    else:
        novel_info = None

    scores = None
    if "anomaly_score" in df.columns:
        s = pd.to_numeric(df["anomaly_score"], errors="coerce").dropna().to_numpy()
        cur = _hist(s, ref["score_edges"]).tolist()
        sp = round(psi(ref["score_counts"], cur, ref["score_groups"]), 6)
        rate = float((s >= ref["threshold"]).mean()) if len(s) else 0.0
        p0 = ref["predicted_anomaly_rate"]
        z = (rate - p0) / math.sqrt(p0 * (1 - p0) / max(len(s), 1)) if 0 < p0 < 1 else 0.0
        scores = {"psi": sp, "status": level(sp) or "stable", "score_counts": cur,
                  "predicted_anomaly_rate": round(rate, 6), "predicted_anomaly_rate_ref": p0, "z": round(z, 3),
                  "quantiles": {q: round(float(np.quantile(s, float(q))), 6) for q in ("0.5", "0.9", "0.99")} if len(s) else {}}
        if scores["status"] != "stable":
            add("psi_score", scores["status"], f"Puntuación: PSI = {sp:.3f}")
        if can_alert and abs(z) >= RATE_Z and abs(rate - p0) >= RATE_ABS:
            add("predicted_rate", "alert", f"Tasa predicha {rate:.2%} frente a {p0:.2%} en train (z = {z:.1f})")

    status = ("alert" if any(a["severity"] == "alert" for a in alerts)
              else "warning" if any(a["severity"] == "warning" for a in alerts)
              else "insufficient_sample" if size == "insufficient" else "ok")
    return {"window": window, "rows": n, "sample_size": size, "status": status, "alerts": alerts,
            "schema": schema, "features": feats, "novel_patterns": novel_info, "scores": scores,
            "reference": {"source": ref["source"], "rows": ref["rows"], **ref["provenance"]},
            "rules": RULES, "interpretation": NOTE}


def dumps(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def report_file(window_csv: Path, ref_path: Path | None, output: Path | None, ref: dict | None = None) -> dict:
    ref = ref or json.loads(Path(ref_path).read_text(encoding="utf-8"))
    df = pd.read_csv(window_csv)
    out = report(df, ref, window=Path(window_csv).name)
    out["window_sha256"] = hashlib.sha256(Path(window_csv).read_bytes()).hexdigest()
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(dumps(out), encoding="utf-8", newline="\n")
    return out


# --- CLI ------------------------------------------------------------------------------------
def _reference_from_train(output: Path) -> dict:
    import joblib

    from hdfs_anomaly import train as T
    from hdfs_anomaly.config import DEFAULT_CONFIG, load_config
    cfg = load_config()
    s = T.load_splits(cfg)  # solo train y validation; se usa únicamente train
    spec_path = cfg["output"]["models"] / "spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    model_path = cfg["output"]["models"] / "model.joblib"
    if T._sha(model_path) != spec["model_sha256"]:
        raise ValueError("El modelo congelado no coincide con su especificación")
    model = joblib.load(model_path)  # artefacto del repositorio, verificado por SHA-256
    tr = s["train"]
    scores = model.predict_proba(tr["X"])[:, 1]
    prov = {"partition": "train", "manifest_sha256": s["manifest_sha256"], "model_sha256": spec["model_sha256"],
            "config_sha256": T._sha(DEFAULT_CONFIG)}
    ref = build_reference(tr["X"], scores, float(spec["threshold"]), prov)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps(ref), encoding="utf-8", newline="\n")
    return ref


DEFAULT_REF = Path(__file__).resolve().parents[2] / "monitoring" / "reference_train.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--output", default=str(DEFAULT_REF))
    p = sub.add_parser("report")
    p.add_argument("window_csv")
    p.add_argument("--reference", default=str(DEFAULT_REF))
    p.add_argument("--output", default=None)
    args = ap.parse_args(argv)
    if args.command == "reference":
        ref = _reference_from_train(Path(args.output))
        print(f"referencia: {ref['rows']} filas de train, {len(ref['pattern_hashes'])} patrones")
        return 0
    out = report_file(Path(args.window_csv), Path(args.reference), Path(args.output) if args.output else None)
    print(json.dumps({k: out[k] for k in ("window", "rows", "status", "alerts")}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
