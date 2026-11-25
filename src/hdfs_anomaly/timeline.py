"""Reconstrucción de la línea temporal de cada bloque a partir de HDFS.log.

Recorre el log una sola vez, en streaming y sin modificarlo, y obtiene por
BlockId la marca de tiempo de su primera y última línea y el número de líneas.
Resultado: artifacts/data/block_times.csv.gz (regenerable, no versionado).

Uso (≈20-30 s):
    python -m hdfs_anomaly.timeline [--config config.toml] [--data-dir RUTA]
"""
import argparse
import gzip
import hashlib
import re
import sys
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import pandas as pd

from hdfs_anomaly.config import load_config

BLOCK_RE = re.compile(rb"blk_-?\d+")
TS_FORMAT = "%y%m%d %H%M%S"  # cabecera de cada línea: "081109 203518"
TIMES_COLUMNS = ["BlockId", "first_ts", "last_ts", "n_lines"]


def extract_block_times(lines: Iterable[bytes]) -> pd.DataFrame:
    """Devuelve BlockId, first_ts, last_ts (datetime) y n_lines, en orden de primera
    aparición. Exige que las marcas de tiempo del log sean no decrecientes, porque
    "última línea" solo equivale a "instante de finalización" si el log está ordenado."""
    first, last, count = {}, {}, {}
    prev = b""
    for i, line in enumerate(lines, 1):
        ts = line[:13]
        if ts < prev:
            raise ValueError(f"Marca de tiempo decreciente en la línea {i}: {ts!r} < {prev!r}")
        prev = ts
        for b in set(BLOCK_RE.findall(line)):
            if b not in first:
                first[b] = ts
            last[b] = ts
            count[b] = count.get(b, 0) + 1

    def conv(d):
        return pd.to_datetime([v.decode() for v in d.values()], format=TS_FORMAT)

    return pd.DataFrame({"BlockId": [b.decode() for b in first], "first_ts": conv(first),
                         "last_ts": conv(last), "n_lines": list(count.values())})


def write_times(path: Path, times: pd.DataFrame) -> str:
    body = times.to_csv(index=False, lineterminator="\n", date_format="%Y-%m-%d %H:%M:%S").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write(body)
    return hashlib.sha256(body).hexdigest()


def read_times(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No se encuentra {path}. Genera las marcas de tiempo con "
                                "`python -m hdfs_anomaly.timeline` (requiere HDFS.log).")
    return pd.read_csv(path, dtype={"BlockId": str}, parse_dates=["first_ts", "last_ts"])


def times_path(cfg: dict) -> Path:
    return cfg["output"]["artifacts"] / "block_times.csv.gz"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.data_dir)
    log = cfg["data"]["dir"] / "HDFS.log"
    if not log.is_file():
        print(f"ERROR: no se encuentra {log}", file=sys.stderr)
        return 1
    with open(log, "rb") as fh:
        times = extract_block_times(fh)
    sha = write_times(times_path(cfg), times)
    print(f"{len(times)} bloques; {times['first_ts'].min()} → {times['last_ts'].max()}; "
          f"sha256 (sin comprimir) {sha}; {datetime.now():%H:%M:%S}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
