"""Export a small, shareable dashboard bundle.

The full dataset (1.2 GB) and feature table (785 MB) are far too large to ship
with a public dashboard. This module distils everything the app needs into
``dashboard_data/`` (a few MB) so the Streamlit app can run standalone and be
deployed to Streamlit Community Cloud with no raw data.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import precision_recall_curve

from . import data
from .config import Config
from .unsupervised import score_unsupervised

BUNDLE_NAME = "dashboard_data"


def _write_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")


def _records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _downsample(precision, recall, n: int = 300) -> dict:
    idx = np.linspace(0, len(recall) - 1, min(n, len(recall))).astype(int)
    return {
        "recall": [float(recall[i]) for i in idx],
        "precision": [float(precision[i]) for i in idx],
    }


def _cost_curve(scores: np.ndarray, labels: np.ndarray, cfg: Config, n: int = 300) -> dict:
    order = np.argsort(-scores)
    y = labels[order]
    n_pos = int(labels.sum())
    tp = np.cumsum(y)
    flagged = np.arange(1, len(y) + 1)
    fp = flagged - tp
    fn = n_pos - tp
    cost = fn * cfg.evaluation.cost_false_negative + fp * cfg.evaluation.cost_false_positive
    best = int(np.argmin(cost))
    idx = np.linspace(0, len(y) - 1, min(n, len(y))).astype(int)
    return {
        "flagged_frac": [float(flagged[i] / len(y)) for i in idx],
        "cost_per_txn": [float(cost[i] / len(y)) for i in idx],
        "optimal": {
            "flagged_frac": float(flagged[best] / len(y)),
            "threshold": float(scores[order][best]),
            "cost_per_txn": float(cost[best] / len(y)),
        },
    }


def export_dashboard_data(cfg: Config, force: bool = False) -> Path:
    """Build every aggregate the dashboard needs into ``dashboard_data/``."""
    out = cfg.project_root / BUNDLE_NAME
    out.mkdir(parents=True, exist_ok=True)

    con = data.prepare(cfg)
    labeled = "WHERE is_fraud IS NOT NULL"

    # --- summary -----------------------------------------------------------
    s = data.summary(con)
    row = con.execute(
        """SELECT
               count(*) FILTER (WHERE is_fraud IS NOT NULL) AS labeled,
               count(*) FILTER (WHERE is_fraud IS NULL)     AS unlabeled,
               min(ts) AS date_min, max(ts) AS date_max
            FROM tx_enriched"""
    ).fetchone() or (0, 0, None, None)
    summary = {
        **s,
        "labeled": int(row[0]),
        "unlabeled": int(row[1]),
        "date_min": str(row[2]),
        "date_max": str(row[3]),
    }
    _write_json(summary, out / "summary.json")

    # --- EDA aggregates ----------------------------------------------------
    def grouped(sql: str) -> list[dict]:
        return _records(con.execute(sql).df())

    _write_json(
        grouped(
            f"""SELECT date_part('hour', ts) AS hour, avg(is_fraud) AS fraud_rate, count(*) AS n
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_by_hour.json",
    )
    _write_json(
        grouped(
            f"""SELECT date_part('dow', ts) AS dow, avg(is_fraud) AS fraud_rate, count(*) AS n
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_by_dow.json",
    )
    _write_json(
        grouped(
            f"""SELECT use_chip, avg(is_fraud) AS fraud_rate, count(*) AS n
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 2 DESC"""
        ),
        out / "eda_by_chip.json",
    )
    _write_json(
        grouped(
            f"""SELECT mcc_desc, count(*) AS frauds, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} GROUP BY 1
                ORDER BY 2 DESC LIMIT 15"""
        ),
        out / "eda_by_category.json",
    )
    _write_json(
        grouped(
            f"""SELECT date_trunc('month', ts) AS month, avg(is_fraud) AS fraud_rate, count(*) AS n
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_monthly.json",
    )
    con.execute(
        f"""COPY (
                SELECT is_fraud, amount FROM tx_enriched {labeled}
                USING SAMPLE 200000 ROWS
            ) TO '{(out / 'amount_sample.parquet').as_posix()}'
            (FORMAT PARQUET, COMPRESSION ZSTD)"""
    )

    # --- supervised metrics / curves --------------------------------------
    metrics = json.loads(cfg.artifacts.metrics.read_text(encoding="utf-8"))
    _write_json(metrics, out / "metrics.json")

    preds = pl.read_parquet(cfg.artifacts.predictions)
    y_test = preds["is_fraud"].to_numpy().astype(int)
    scores = preds["score"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_test, scores)
    _write_json(_downsample(precision, recall), out / "pr_curve.json")
    _write_json(_cost_curve(scores, y_test, cfg), out / "cost_curve.json")

    top = (
        preds.sort("score", descending=True)
        .head(500)
        .with_columns(pl.col("ts").cast(pl.Utf8))
    )
    top.write_parquet(out / "top_risk.parquet")

    model = joblib.load(cfg.artifacts.model)
    gains = model.booster_.feature_importance(importance_type="gain")
    names = model.booster_.feature_name()
    importance = sorted(
        ({"feature": n, "gain": float(g)} for n, g in zip(names, gains, strict=True)),
        key=lambda d: d["gain"],
        reverse=True,
    )[:20]
    _write_json(importance, out / "feature_importance.json")

    # --- unsupervised ------------------------------------------------------
    _write_json(score_unsupervised(cfg), out / "unsupervised_metrics.json")
    anom = pl.read_parquet(cfg.artifacts.dir / "unsupervised_scores.parquet")
    if anom.height > 50_000:
        anom = anom.sample(50_000, seed=cfg.seed)
    anom.select("anomaly_score", "is_fraud").write_parquet(out / "anomaly_sample.parquet")

    return out
