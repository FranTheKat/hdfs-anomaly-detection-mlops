"""Carga de la configuración central (config.toml)."""
import os
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config.toml"


def load_config(path: str | Path | None = None, data_dir: str | Path | None = None) -> dict:
    """Lee config.toml y resuelve rutas.

    Prioridad del directorio de datos: argumento ``data_dir`` > variable de entorno
    HDFS_V1_DIR > ``[data].dir`` del archivo. Las rutas relativas se resuelven
    desde la raíz del repositorio para que el resultado no dependa del directorio actual.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    root = path.resolve().parent
    data = Path(data_dir or os.environ.get("HDFS_V1_DIR") or cfg["data"]["dir"])
    data = data if data.is_absolute() else (root / data)
    cfg["data"]["dir"] = data.resolve()
    cfg["data"]["matrix_path"] = cfg["data"]["dir"] / cfg["data"]["matrix"]
    cfg["data"]["labels_path"] = cfg["data"]["dir"] / cfg["data"]["labels"]
    for key in ("artifacts", "reports", "models"):
        cfg["output"][key] = (root / cfg["output"][key]).resolve()
    cfg["split"]["block_times_path"] = (root / cfg["split"]["block_times"]).resolve()
    return cfg
