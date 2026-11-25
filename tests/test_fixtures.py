"""Aislamiento del holdout: la muestra de pruebas procede solo de la partición train real."""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from hdfs_anomaly import data as D
from hdfs_anomaly.config import REPO_ROOT

FIXTURES = Path(__file__).parent / "fixtures"
PROV = json.loads((FIXTURES / "sample_provenance.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((REPO_ROOT / "reports/data/split_summary.json").read_text(encoding="utf-8"))
MANIFEST = REPO_ROOT / "artifacts/data/split_manifest.csv.gz"


def test_sample_provenance_is_train_only_and_unchanged():
    assert PROV["source_partition"] == "train" and PROV["strategy"] == "completion"
    assert PROV["manifest_sha256_uncompressed"] == SUMMARY["manifest_sha256_uncompressed"]
    assert hashlib.sha256((FIXTURES / "sample_matrix.csv").read_bytes()).hexdigest() == PROV["sample_matrix_sha256"]
    sample = pd.read_csv(FIXTURES / "sample_matrix.csv")
    assert len(sample) == PROV["rows"] and int((sample["Label"] == "Fail").sum()) == PROV["anomalies"]


@pytest.mark.skipif(not MANIFEST.is_file(), reason="manifiesto real no disponible (se regenera con python -m hdfs_anomaly.data)")
def test_no_sample_block_belongs_to_validation_or_test():
    body = gzip.decompress(MANIFEST.read_bytes())
    assert hashlib.sha256(body).hexdigest() == SUMMARY["manifest_sha256_uncompressed"]
    split = D.read_manifest(MANIFEST).set_index(D.ID_COL)["split"]
    for name in ("sample_matrix.csv", "sample_labels.csv", "sample_block_times.csv"):
        ids = pd.read_csv(FIXTURES / name)[D.ID_COL]
        assert (split.reindex(ids) == "train").all(), name
