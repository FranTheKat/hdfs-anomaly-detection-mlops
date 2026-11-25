"""Ingesta, validación, preparación y partición de datos.

Los archivos originales solo se leen. Las salidas son:
- reports/data/validation_report.json  (informe de validación, versionado)
- reports/data/split_summary.json      (resumen de la partición, versionado)
- artifacts/data/split_manifest.csv.gz (BlockId -> partición, regenerable)

La partición por defecto (completion) necesita artifacts/data/block_times.csv.gz,
generado con `python -m hdfs_anomaly.timeline`.

Uso:
    python -m hdfs_anomaly.data [--config config.toml] [--data-dir RUTA]
"""
import argparse
import gzip
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from hdfs_anomaly.config import REPO_ROOT, load_config
from hdfs_anomaly.timeline import read_times

EVENT_COLS = [f"E{i}" for i in range(1, 30)]
ID_COL, LABEL_COL, TYPE_COL = "BlockId", "Label", "Type"
EXPECTED_COLUMNS = [ID_COL, LABEL_COL, TYPE_COL, *EVENT_COLS]
# Columnas que nunca pueden ser características (identificador, objetivo o fuga).
FORBIDDEN_FEATURES = {ID_COL, LABEL_COL, TYPE_COL, "Latency", "TimeInterval"}
LABEL_TO_TARGET = {"Success": 0, "Fail": 1}
REFERENCE_LABELS = {"Normal": "Success", "Anomaly": "Fail"}  # anomaly_label.csv -> matriz
SPLITS = ("train", "validation", "test")
BLOCK_ID_PATTERN = r"blk_-?\d+"


class DataValidationError(Exception):
    def __init__(self, report: dict):
        failed = [c["name"] for c in report["checks"] if c["severity"] == "error" and not c["passed"]]
        super().__init__(f"Validación de datos fallida: {', '.join(failed)}")
        self.report = report


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Ingesta ------------------------------------------------------------------
def _require(path: Path) -> Path:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No se encuentra {path}. Descarga HDFS_v1 según el README "
                                "o indica la ruta con HDFS_V1_DIR / --data-dir.")
    return Path(path)


def load_matrix(path: Path) -> pd.DataFrame:
    """Lee la matriz de ocurrencias. No fuerza tipos en los eventos para que la
    validación pueda detectar valores no enteros o vacíos (que pandas lee como float)."""
    return pd.read_csv(_require(path), dtype={ID_COL: str, LABEL_COL: str, TYPE_COL: str})


def load_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(_require(path), dtype=str)


# --- Validación ---------------------------------------------------------------
def validate(df: pd.DataFrame, labels: pd.DataFrame | None = None,
             min_rate: float = 0.005, max_rate: float = 0.10) -> dict:
    """Devuelve un informe con comprobaciones de severidad 'error' (bloquean el
    pipeline) o 'warning' (se registran y requieren revisión)."""
    checks = []

    def check(name, passed, detail="", severity="error"):
        checks.append({"name": name, "severity": severity, "passed": bool(passed), "detail": detail})
        return passed

    cols = list(df.columns)
    if not check("schema_columns", cols == EXPECTED_COLUMNS,
                 f"esperadas {len(EXPECTED_COLUMNS)} columnas en orden; "
                 f"faltan {sorted(set(EXPECTED_COLUMNS) - set(cols))}, "
                 f"sobran {sorted(set(cols) - set(EXPECTED_COLUMNS))}"):
        return _report(checks, df)
    check("non_empty", len(df) > 0, f"{len(df)} filas")

    ids = df[ID_COL]
    check("block_id_not_null", ids.notna().all(), f"{int(ids.isna().sum())} nulos")
    check("block_id_unique", ids.is_unique, f"{int(ids.duplicated().sum())} duplicados")
    bad_ids = int((~ids.fillna("").str.fullmatch(BLOCK_ID_PATTERN)).sum())
    check("block_id_format", bad_ids == 0, f"{bad_ids} con formato distinto de {BLOCK_ID_PATTERN}")

    labels_ok = df[LABEL_COL].isin(LABEL_TO_TARGET.keys())
    check("label_values", labels_ok.all(),
          f"{int((~labels_ok).sum())} etiquetas fuera de {sorted(LABEL_TO_TARGET)}")

    bad_type = [c for c in EVENT_COLS if not pd.api.types.is_integer_dtype(df[c])]
    check("events_integer_no_nulls", not bad_type,
          f"columnas no enteras o con vacíos: {bad_type}" if bad_type else "29 columnas enteras")
    if not bad_type:
        neg = [c for c in EVENT_COLS if (df[c] < 0).any()]
        check("events_non_negative", not neg, f"columnas con negativos: {neg}" if neg else "")
        e22 = int((df["E22"] != 1).sum())
        check("e22_allocate_once", e22 == 0,
              f"{e22} bloques con E22 != 1 (cada bloque debería asignarse una vez)", "warning")

    y = df[LABEL_COL].map(LABEL_TO_TARGET)
    check("both_classes_present", y.nunique() == 2, f"clases: {sorted(df[LABEL_COL].dropna().unique())}")
    rate = float(y.mean()) if len(df) else 0.0
    check("anomaly_rate_in_range", min_rate <= rate <= max_rate,
          f"tasa {rate:.4%}; rango esperado [{min_rate:.2%}, {max_rate:.2%}]", "warning")

    # Type solo existe en anomalías: es una fuga directa de la etiqueta y se excluye.
    leak = bool((df[TYPE_COL].notna() == (df[LABEL_COL] == "Fail")).all())
    check("type_column_is_label_leak", True,
          "Type informada exactamente en las filas Fail: excluida de las características"
          if leak else "Type no coincide exactamente con Fail; sigue excluida", "info")

    if labels is not None:
        ref = labels.set_index(ID_COL)[LABEL_COL].map(REFERENCE_LABELS)
        same_ids = set(ref.index) == set(ids)
        check("labels_same_block_ids", same_ids,
              f"{len(set(ids) - set(ref.index))} solo en matriz, {len(set(ref.index) - set(ids))} solo en etiquetas")
        if same_ids:
            mism = int((ref.reindex(ids).to_numpy() != df[LABEL_COL].to_numpy()).sum())
            check("labels_match_reference", mism == 0, f"{mism} etiquetas discordantes")
    return _report(checks, df)


def _report(checks: list, df: pd.DataFrame) -> dict:
    return {"passed": all(c["passed"] for c in checks if c["severity"] == "error"),
            "rows": len(df), "checks": checks}


# --- Preparación y partición ---------------------------------------------------
def prepare(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    """Separa identificador, características (solo E1..E29) y objetivo."""
    X = df[EVENT_COLS].astype("int64")
    assert not FORBIDDEN_FEATURES & set(X.columns), "columna prohibida en las características"
    y = df[LABEL_COL].map(LABEL_TO_TARGET).to_numpy(dtype="int8")
    return df[ID_COL].reset_index(drop=True), X.reset_index(drop=True), y


def validate_times(ids: pd.Series, X: pd.DataFrame, times: pd.DataFrame) -> list[dict]:
    """Comprueba que las marcas de tiempo reconstruidas corresponden a esta matriz."""
    same = times[ID_COL].is_unique and set(times[ID_COL]) == set(ids)
    checks = [{"name": "times_same_block_ids", "severity": "error", "passed": bool(same),
               "detail": f"{len(times)} bloques con marca de tiempo"}]
    if same:
        t = times.set_index(ID_COL).loc[ids]
        lines_ok = (t["n_lines"].to_numpy() == X.sum(axis=1).to_numpy()).all()
        order_ok = (t["first_ts"] <= t["last_ts"]).all()
        checks += [
            {"name": "times_lines_match_events", "severity": "error", "passed": bool(lines_ok),
             "detail": "líneas de log por bloque == suma de E1..E29"},
            {"name": "times_first_not_after_last", "severity": "error", "passed": bool(order_ok), "detail": ""},
        ]
    return checks


def completion_cuts(last_ts: np.ndarray, train: float, validation: float) -> tuple:
    """Instantes de corte: el bloque n·train-ésimo y el n·(train+validation)-ésimo por
    orden de finalización. Los empates en el corte quedan en la partición anterior."""
    s = np.sort(np.asarray(last_ts))
    n = len(s)
    c1, c2 = s[max(int(n * train), 1) - 1], s[max(int(n * (train + validation)), 1) - 1]
    if not (c1 < c2 < s[-1]):
        raise ValueError(f"Cortes temporales degenerados ({c1}, {c2}, fin {s[-1]}): demasiados empates")
    return c1, c2


def split(y: np.ndarray, strategy: str = "completion", train: float = 0.6, validation: float = 0.2,
          test: float = 0.2, seed: int = 42, last_ts: np.ndarray | None = None) -> np.ndarray:
    """Asigna 'train' / 'validation' / 'test' a cada fila.

    completion: cronológica por instante de finalización de la traza (``last_ts``).
        train = trazas terminadas en o antes del corte 1; validation = hasta el corte 2;
        test = después. Todo lo usado para entrenar existe completo en el corte 1.
    first_seen: orden de las filas (primera aparición del bloque). Solo comparación.
    stratified: aleatoria manteniendo la proporción de clases. Solo comparación.
    """
    fr = np.array([train, validation, test], dtype=float)
    if (fr <= 0).any() or not np.isclose(fr.sum(), 1.0):
        raise ValueError(f"Proporciones inválidas {fr.tolist()}: deben ser > 0 y sumar 1")
    n = len(y)
    out = np.empty(n, dtype=object)
    if strategy == "completion":
        if last_ts is None or len(last_ts) != n:
            raise ValueError("La partición 'completion' necesita last_ts para cada bloque")
        c1, c2 = completion_cuts(last_ts, train, validation)
        ts = np.asarray(last_ts)
        out[:] = np.where(ts <= c1, "train", np.where(ts <= c2, "validation", "test"))
    elif strategy == "first_seen":
        a, b = int(n * train), int(n * (train + validation))
        out[:a], out[a:b], out[b:] = SPLITS
    elif strategy == "stratified":
        idx = np.arange(n)
        tr, rest = train_test_split(idx, train_size=train, stratify=y, random_state=seed)
        va, te = train_test_split(rest, train_size=validation / (validation + test),
                                  stratify=y[rest], random_state=seed)
        out[tr], out[va], out[te] = SPLITS
    else:
        raise ValueError(f"Estrategia de partición desconocida: {strategy!r}")
    return out


def summarize(ids: pd.Series, X: pd.DataFrame, y: np.ndarray, parts: np.ndarray) -> dict:
    """Tamaño, prevalencia y solapamiento de patrones de cada partición."""
    vectors = pd.util.hash_pandas_object(X, index=False).to_numpy()
    train_vectors = set(vectors[parts == "train"])
    out = {}
    for name in SPLITS:
        m = parts == name
        v = vectors[m]
        out[name] = {
            "rows": int(m.sum()),
            "anomalies": int(y[m].sum()),
            "anomaly_rate": round(float(y[m].mean()), 6),
            "distinct_vectors": len(set(v)),
            "rows_with_vector_seen_in_train": round(float(np.isin(v, list(train_vectors)).mean()), 6),
            "distinct_vectors_unseen_in_train": len(set(v) - train_vectors),
        }
    sets = [set(ids[parts == s]) for s in SPLITS]
    out["block_ids_disjoint"] = not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    out["all_rows_assigned"] = int(sum(len(s) for s in sets)) == len(ids)
    out["every_split_has_both_classes"] = all(0 < out[s]["anomalies"] < out[s]["rows"] for s in SPLITS)
    return out


def temporal_checks(parts: np.ndarray, first_ts: np.ndarray, last_ts: np.ndarray) -> dict:
    """Condiciones de una simulación prospectiva por finalización de traza:
    toda traza de train termina antes de que termine cualquiera de validation, y lo
    mismo entre validation y test. Así, en el instante de corte, las características
    (conteos completos) y la etiqueta de cada bloque de train ya existen, y ningún
    bloque de validation/test ha terminado todavía."""
    first_ts, last_ts = pd.Series(first_ts), pd.Series(last_ts)
    rng = {s: (last_ts[parts == s].min(), last_ts[parts == s].max()) for s in SPLITS}
    out = {f"{s}_completion_range": [str(a), str(b)] for s, (a, b) in rng.items()}
    out["train_complete_before_validation"] = bool(rng["train"][1] < rng["validation"][0])
    out["validation_complete_before_test"] = bool(rng["validation"][1] < rng["test"][0])
    # Bloques en curso en el corte anterior: no se usan para entrenar, solo se evalúan al terminar.
    for s, prev in (("validation", "train"), ("test", "validation")):
        started = (first_ts[parts == s] <= rng[prev][1]).mean()
        out[f"{s}_started_before_previous_cut"] = round(float(started), 6)
    out["prospective"] = out["train_complete_before_validation"] and out["validation_complete_before_test"]
    return out


def write_manifest(path: Path, ids: pd.Series, parts: np.ndarray) -> str:
    """Escribe BlockId,split en CSV comprimido. Devuelve el SHA-256 del CSV sin
    comprimir (estable entre versiones de zlib)."""
    body = pd.DataFrame({ID_COL: ids, "split": parts}).to_csv(index=False, lineterminator="\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    # filename="" y mtime=0: la cabecera gzip no depende del nombre ni de la fecha.
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(body)
    return hashlib.sha256(body).hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


# --- Orquestación de la etapa ---------------------------------------------------
def _git_state() -> dict:
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True,
                                        text=True, check=True).stdout.strip()
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def run(cfg: dict) -> dict:
    d, s, v, o = cfg["data"], cfg["split"], cfg["validation"], cfg["output"]
    matrix_path, labels_path = d["matrix_path"], d["labels_path"]
    df = load_matrix(matrix_path)
    labels = load_labels(labels_path)

    times_path = s["block_times_path"]
    sources = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
               for p in (matrix_path, labels_path, times_path) if p.is_file()}
    report = validate(df, labels, v["min_anomaly_rate"], v["max_anomaly_rate"])
    report["sources"] = sources
    ids, X, y = prepare(df) if report["passed"] else (None, None, None)
    times = None
    if report["passed"] and (s["strategy"] == "completion" or times_path.is_file()):
        times = read_times(times_path)
        report["checks"] += validate_times(ids, X, times)
        report["passed"] = all(c["passed"] for c in report["checks"] if c["severity"] == "error")
    _write_json(o["reports"] / "validation_report.json", report)
    if not report["passed"]:
        raise DataValidationError(report)

    if times is not None:
        times = times.set_index(ID_COL).loc[ids]
    last_ts = times["last_ts"].to_numpy() if times is not None else None
    parts = split(y, s["strategy"], s["train"], s["validation"], s["test"], s["seed"], last_ts)
    splits = summarize(ids, X, y, parts)
    temporal = temporal_checks(parts, times["first_ts"].to_numpy(), last_ts) if times is not None else None
    if s["strategy"] == "completion" and not temporal["prospective"]:
        raise RuntimeError(f"La partición por finalización no cumple las condiciones temporales: {temporal}")
    if not (splits["block_ids_disjoint"] and splits["all_rows_assigned"] and splits["every_split_has_both_classes"]):
        raise RuntimeError(f"Partición inválida: {splits}")
    manifest_sha = write_manifest(o["artifacts"] / "split_manifest.csv.gz", ids, parts)
    summary = {
        "strategy": s["strategy"],
        "fractions": {k: s[k] for k in SPLITS},
        "seed": s["seed"],
        "features": EVENT_COLS,
        "excluded_columns": sorted(FORBIDDEN_FEATURES),
        "target": "Label == 'Fail' -> 1",
        "sources": sources,
        "manifest_sha256_uncompressed": manifest_sha,
        "temporal_checks": temporal,
        "splits": splits,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__,
                        "numpy": np.__version__},
        "git": _git_state(),
    }
    _write_json(o["reports"] / "split_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    try:
        summary = run(cfg)
    except (FileNotFoundError, DataValidationError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    for name in SPLITS:
        p = summary["splits"][name]
        print(f"{name:<10} filas={p['rows']:>7} anomalías={p['anomalies']:>6} ({p['anomaly_rate']:.2%})")
    print(f"informes en {cfg['output']['reports']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
