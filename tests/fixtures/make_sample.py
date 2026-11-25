"""Genera la muestra pequeña de HDFS_v1 usada por las pruebas (y por la CI).

La muestra procede EXCLUSIVAMENTE de la partición train real (manifiesto de partición), para no introducir bloques del holdout test en ninguna prueba automatizada.
Selecciona 1500 bloques normales y 500 anómalos de train al azar (semilla fija) y conserva
su orden original. Incluye las marcas de tiempo de cada bloque (requiere haber ejecutado
`python -m hdfs_anomaly.timeline` y `python -m hdfs_anomaly.data`).
Datos de Loghub (CC BY 4.0): https://github.com/logpai/loghub

Uso:
    python tests/fixtures/make_sample.py [ruta_HDFS_v1]
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from hdfs_anomaly import data as D
from hdfs_anomaly.config import load_config
from hdfs_anomaly.timeline import read_times, times_path

cfg = load_config(data_dir=sys.argv[1] if len(sys.argv) > 1 else None)
summary = json.loads((cfg["output"]["reports"] / "split_summary.json").read_text(encoding="utf-8"))
manifest_path = cfg["output"]["artifacts"] / "split_manifest.csv.gz"
if hashlib.sha256(gzip.decompress(manifest_path.read_bytes())).hexdigest() != summary["manifest_sha256_uncompressed"]:
    sys.exit("El manifiesto no coincide con split_summary.json; repite python -m hdfs_anomaly.data")
split = D.read_manifest(manifest_path).set_index(D.ID_COL)["split"]

matrix = pd.read_csv(cfg["data"]["matrix_path"], dtype=str)
labels = pd.read_csv(cfg["data"]["labels_path"], dtype=str)
in_train = (split.reindex(matrix[D.ID_COL]) == "train").to_numpy()

rng = np.random.default_rng(42)
normal = np.flatnonzero(in_train & (matrix["Label"] == "Success").to_numpy())
anomaly = np.flatnonzero(in_train & (matrix["Label"] == "Fail").to_numpy())
rows = np.sort(np.concatenate([rng.choice(normal, 1500, replace=False), rng.choice(anomaly, 500, replace=False)]))

sample = matrix.iloc[rows]
assert (split.reindex(sample[D.ID_COL]) == "train").all()
sample.to_csv(HERE / "sample_matrix.csv", index=False, lineterminator="\n")
labels.set_index("BlockId").loc[sample["BlockId"]].reset_index().to_csv(
    HERE / "sample_labels.csv", index=False, lineterminator="\n")
times = read_times(times_path(cfg)).set_index("BlockId").loc[sample["BlockId"]].reset_index()
times.to_csv(HERE / "sample_block_times.csv", index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S")
provenance = {"source_partition": "train", "strategy": summary["strategy"],
              "manifest_sha256_uncompressed": summary["manifest_sha256_uncompressed"],
              "rows": len(sample), "anomalies": int((sample["Label"] == "Fail").sum()), "seed": 42,
              "sample_matrix_sha256": hashlib.sha256((HERE / "sample_matrix.csv").read_bytes()).hexdigest()}
(HERE / "sample_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"muestra: {len(sample)} filas de train ({provenance['anomalies']} anómalas)")
