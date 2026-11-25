"""Pruebas de las condiciones temporales: reconstrucción de la línea de tiempo y
partición por instante de finalización de traza."""
import numpy as np
import pandas as pd
import pytest

from hdfs_anomaly import data as D
from hdfs_anomaly import timeline as T

LOG = [
    b"081109 200000 1 INFO a: Receiving block blk_1 src: x dest: y",
    b"081109 200001 1 INFO a: Receiving block blk_-2 src: x dest: y",
    b"081109 200005 1 INFO a: Transmitted block blk_1 to blk_-2 blk_1",  # dos bloques, blk_1 dos veces
    b"081109 210000 1 INFO a: Deleting block blk_-2 file /x",
    b"081109 230000 1 INFO a: Deleting block blk_1 file /x",
]


def ts(*hours):
    return pd.to_datetime("2008-11-10") + pd.to_timedelta(list(hours), unit="h")


# --- Reconstrucción desde el log ----------------------------------------------------
def test_extract_first_last_and_line_counts():
    t = T.extract_block_times(LOG).set_index("BlockId")
    assert list(t.index) == ["blk_1", "blk_-2"]  # orden de primera aparición
    assert t.loc["blk_1", "first_ts"] == pd.Timestamp("2008-11-09 20:00:00")
    assert t.loc["blk_1", "last_ts"] == pd.Timestamp("2008-11-09 23:00:00")
    assert t.loc["blk_-2", "last_ts"] == pd.Timestamp("2008-11-09 21:00:00")
    # Una línea cuenta una vez por bloque aunque lo mencione dos veces (como la matriz de eventos).
    assert t.loc["blk_1", "n_lines"] == 3 and t.loc["blk_-2", "n_lines"] == 3


def test_unordered_log_is_rejected():
    with pytest.raises(ValueError, match="decreciente"):
        T.extract_block_times([LOG[3], LOG[0]])


def test_times_round_trip(tmp_path):
    t = T.extract_block_times(LOG)
    h1 = T.write_times(tmp_path / "t.csv.gz", t)
    assert h1 == T.write_times(tmp_path / "t2.csv.gz", t)
    pd.testing.assert_frame_equal(T.read_times(tmp_path / "t.csv.gz"), t, check_dtype=False)


def test_missing_times_file_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="hdfs_anomaly.timeline"):
        T.read_times(tmp_path / "no.csv.gz")


# --- Partición por finalización -------------------------------------------------------
def test_completion_split_orders_by_end_not_by_start():
    # El bloque 0 empieza el primero pero termina el último: debe ir a test.
    last = ts(9, 1, 2, 3, 4, 5, 6, 7, 8, 0.5)
    first = ts(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.05)
    y = np.array([1, 0, 1, 0, 0, 1, 0, 1, 0, 1])
    parts = D.split(y, "completion", 0.6, 0.2, 0.2, last_ts=last.to_numpy())
    assert parts[0] == "test" and parts[9] == "train"
    chk = D.temporal_checks(parts, first.to_numpy(), last.to_numpy())
    assert chk["prospective"]
    # Todo lo de train está completo antes de que termine cualquier traza de validación o test.
    assert last[parts == "train"].max() < last[parts == "validation"].min()
    assert last[parts == "validation"].max() < last[parts == "test"].min()


def test_ties_at_cut_stay_in_earlier_split():
    last = ts(1, 1, 1, 1, 1, 1, 1, 2, 3, 4)  # 7 empates en el primer corte
    parts = D.split(np.zeros(10), "completion", 0.6, 0.2, 0.2, last_ts=last.to_numpy())
    assert (parts[:7] == "train").all()
    assert D.temporal_checks(parts, last.to_numpy(), last.to_numpy())["prospective"]


def test_degenerate_cuts_rejected():
    with pytest.raises(ValueError, match="degenerados"):
        D.split(np.zeros(5), "completion", last_ts=ts(1, 1, 1, 1, 1).to_numpy())


def test_completion_requires_timestamps():
    with pytest.raises(ValueError, match="last_ts"):
        D.split(np.zeros(5), "completion")


def test_first_seen_split_can_violate_completion_order():
    last = ts(9, 1, 2, 3, 4, 5, 6, 7, 8, 0.5)  # el bloque 0 (train) termina después que todos
    parts = D.split(np.zeros(10), "first_seen", 0.6, 0.2, 0.2)
    assert not D.temporal_checks(parts, last.to_numpy(), last.to_numpy())["prospective"]


def test_times_must_match_matrix():
    df = pd.DataFrame({"BlockId": ["blk_1", "blk_-2"], "Label": ["Success", "Fail"], "Type": [None, "1"],
                       **{c: [0, 0] for c in D.EVENT_COLS}})
    df["E22"] = 1
    df["E5"] = [2, 2]  # 2 + E22 = 3 eventos = 3 líneas en LOG para cada bloque
    ids, X, _ = D.prepare(df)
    times = T.extract_block_times(LOG)
    assert all(c["passed"] for c in D.validate_times(ids, X, times))
    df.loc[1, "E5"] = 5  # ya no coincide con las líneas del log
    _, X_bad, _ = D.prepare(df)
    by = {c["name"]: c["passed"] for c in D.validate_times(ids, X_bad, times)}
    assert by["times_same_block_ids"] and not by["times_lines_match_events"]
    other = times.assign(BlockId=["blk_1", "blk_9"])
    assert not D.validate_times(ids, X, other)[0]["passed"]
