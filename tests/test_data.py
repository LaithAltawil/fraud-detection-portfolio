import json
from pathlib import Path

from fraud_detection.data import _iter_label_pairs, load_labels


def test_label_parser_maps_yes_no(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"target": {"10": "No", "20": "Yes", "30": "No"}}), encoding="utf-8")
    pairs = dict(_iter_label_pairs(path))
    assert pairs == {10: 0, 20: 1, 30: 0}


def test_load_labels_caches_parquet(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"target": {"1": "Yes", "2": "No"}}), encoding="utf-8")
    cache = tmp_path / "labels.parquet"
    table = load_labels(path, cache)
    assert cache.exists()
    lookups = dict(zip(table["id"].to_pylist(), table["is_fraud"].to_pylist(), strict=True))
    assert lookups == {1: 1, 2: 0}
