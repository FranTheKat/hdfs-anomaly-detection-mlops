"""Simulación local de tráfico contra la API de DEMOSTRACIÓN.

    python -m hdfs_anomaly.simulate --base-url http://127.0.0.1:8000

Genera ventanas a partir de las frecuencias de patrones de TRAIN guardadas en la referencia
(monitoring/reference_train.json) y de perturbaciones sintéticas; nunca usa el holdout de test.
Envía cada traza a POST /predict, guarda la ventana con la puntuación devuelta, calcula su
informe de deriva y una instantánea de GET /metrics en reports/monitoring/.

Ventanas:
- baseline:      2000 trazas con la mezcla de train (incluye las anomalías que el modelo marca).
- anomaly_burst: 2000 trazas; 15 % repetidas de las que el modelo marcó como anómalas.
- shifted:       2000 trazas; en el 40 % se suman 1–3 eventos E2/E4 (cambio controlado sintético).
- small:         200 trazas de train → muestra insuficiente, sin alertas.
Además, unas pocas peticiones inválidas (campo BlockId, negativo, texto) para ver errores 4xx.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from hdfs_anomaly import drift
from hdfs_anomaly.data import EVENT_COLS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "monitoring"
SEED = 42
INVALID = [{**{c: 0 for c in EVENT_COLS}, "BlockId": "blk_sintetico"},
           {**{c: 0 for c in EVENT_COLS}, "E5": -1},
           {**{c: 0 for c in EVENT_COLS}, "E5": "3"}]


def http_post(base_url: str):
    def post(path: str, payload: dict | None = None) -> tuple[int, str]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base_url.rstrip("/") + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as r:  # URL local indicada por quien ejecuta
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()
    return post


def train_mix(ref: dict, n: int, rng: np.random.Generator) -> pd.DataFrame:
    pats = ref["top_patterns"]
    w = np.array([p["count"] for p in pats], dtype=float)
    idx = rng.choice(len(pats), size=n, p=w / w.sum())
    return pd.DataFrame([pats[i]["events"] for i in idx], columns=EVENT_COLS)


def shift(df: pd.DataFrame, rng: np.random.Generator, frac=0.4) -> pd.DataFrame:
    out = df.copy()
    rows = rng.random(len(out)) < frac
    for c in ("E2", "E4"):
        out.loc[rows, c] += rng.integers(1, 4, size=rows.sum())
    return out


def score(df: pd.DataFrame, post) -> pd.DataFrame:
    res = []
    for row in df[EVENT_COLS].astype(int).to_dict("records"):
        status, body = post("/predict", row)
        if status != 200:
            raise RuntimeError(f"/predict devolvió {status}: {body[:200]}")
        d = json.loads(body)
        if not d["demo_only"]:
            raise RuntimeError("La simulación solo se ejecuta contra la API en modo demostración")
        res.append((d["anomaly_score"], d["prediction"]))
    return df.assign(anomaly_score=[s for s, _ in res], prediction=[p for _, p in res])


def run(post, ref: dict, out: Path = OUT, seed: int = SEED, n: int = 2000) -> dict:
    rng = np.random.default_rng(seed)
    (out / "windows").mkdir(parents=True, exist_ok=True)
    baseline = score(train_mix(ref, n, rng), post)
    flagged = baseline[baseline["prediction"] == "anomaly"][EVENT_COLS]
    if flagged.empty:
        raise RuntimeError("La ventana base no contiene anomalías predichas")
    k = int(0.15 * n)
    burst = pd.concat([train_mix(ref, n - k, rng), flagged.sample(k, replace=True, random_state=seed)],
                      ignore_index=True)
    windows = {"baseline": baseline,
               "anomaly_burst": score(burst.sample(frac=1, random_state=seed).reset_index(drop=True), post),
               "shifted": score(shift(train_mix(ref, n, rng), rng), post),
               "small": score(train_mix(ref, n // 10, rng), post)}
    invalid = [post("/predict", p)[0] for p in INVALID]
    summary = {"seed": seed, "window_rows": n, "demo_only": True, "invalid_request_status": invalid, "windows": {}}
    for name, df in windows.items():
        csv = out / "windows" / f"{name}.csv"
        df.to_csv(csv, index=False, lineterminator="\n", float_format="%.12g")
        rep = drift.report_file(csv, ref_path=None, output=out / f"drift_{name}.json", ref=ref)
        summary["windows"][name] = {"rows": rep["rows"], "status": rep["status"],
                                    "alerts": [f"{a['severity']}:{a['rule']}" for a in rep["alerts"]],
                                    "predicted_anomaly_rate": rep["scores"]["predicted_anomaly_rate"]}
    status, metrics = post("/metrics")
    if status != 200:
        raise RuntimeError(f"/metrics devolvió {status}")
    (out / "metrics_snapshot.txt").write_text(metrics, encoding="utf-8", newline="\n")
    (out / "simulation_summary.json").write_text(drift.dumps(summary), encoding="utf-8", newline="\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--reference", default=str(drift.DEFAULT_REF))
    args = ap.parse_args(argv)
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    print(drift.dumps(run(http_post(args.base_url), ref)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
