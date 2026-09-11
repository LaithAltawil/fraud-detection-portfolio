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
from sklearn.metrics import precision_recall_curve, roc_curve

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


def _roc_curve(scores: np.ndarray, labels: np.ndarray, n: int = 300) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.linspace(0, len(fpr) - 1, min(n, len(fpr))).astype(int)
    return {"fpr": [float(fpr[i]) for i in idx], "tpr": [float(tpr[i]) for i in idx]}


def _results_bundle(scores: np.ndarray, labels: np.ndarray, threshold: float, n: int = 60) -> dict:
    """Decile lift, confusion matrix at the optimal threshold and a threshold scan."""
    order = np.argsort(-scores)
    y = labels[order]
    n_pos = int(labels.sum())
    base = n_pos / len(y)

    # Decile lift.
    deciles = np.array_split(np.arange(len(y)), 10)
    cumulative = 0
    decile_rows = []
    for i, idx in enumerate(deciles, 1):
        frauds = int(y[idx].sum())
        cumulative += frauds
        rate = frauds / len(idx)
        decile_rows.append(
            {
                "decile": i,
                "n": int(len(idx)),
                "frauds": frauds,
                "fraud_rate": rate,
                "lift": rate / base,
                "cumulative_recall": cumulative / n_pos,
            }
        )

    # Confusion matrix at the cost-optimal threshold.
    pred = scores >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    confusion = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "flagged_frac": (tp + fp) / len(labels),
    }

    # Threshold scan: precision/recall/FPR as the review budget grows.
    scan = []
    for frac in np.linspace(0.0005, 0.1, n):
        k = max(1, int(len(y) * frac))
        flagged = y[:k]
        tp_k = int(flagged.sum())
        scan.append(
            {
                "budget": float(k / len(y)),
                "recall": tp_k / n_pos,
                "precision": tp_k / k,
                "lift": (tp_k / k) / base,
            }
        )

    return {"deciles": decile_rows, "confusion": confusion, "threshold_scan": scan}


def _split_summary(cfg: Config) -> dict:
    """Chronological split sizes and boundary timestamps (labeled rows only)."""
    ts = (
        pl.scan_parquet(cfg.artifacts.features)
        .select("ts", "is_fraud")
        .filter(pl.col("is_fraud").is_not_null())
        .select("ts")
        .collect()["ts"]
        .sort()
    )
    n = len(ts)
    t = cfg.split
    i1 = int(n * t.train_frac)
    i2 = int(n * (t.train_frac + t.valid_frac))
    train_end, valid_end = ts[i1], ts[i2]
    return {
        "train": int((ts <= train_end).sum()),
        "valid": int(((ts > train_end) & (ts <= valid_end)).sum()),
        "test": int((ts > valid_end).sum()),
        "train_frac": t.train_frac,
        "valid_frac": t.valid_frac,
        "test_frac": t.test_frac,
        "train_end": str(train_end),
        "valid_end": str(valid_end),
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
    _write_json(_split_summary(cfg), out / "split.json")

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
    # Additional EDA cuts.
    # Geography: ``merchant_state`` mixes US state codes with country names, so
    # the US and international cuts are exported separately (otherwise a single
    # high-fraud country like Italy dominates the US view).
    _write_json(
        grouped(
            f"""SELECT merchant_state AS state, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} AND length(merchant_state) = 2
                GROUP BY 1 HAVING count(*) > 2000
                ORDER BY fraud_rate DESC LIMIT 15"""
        ),
        out / "eda_by_state.json",
    )
    _write_json(
        grouped(
            f"""SELECT merchant_state AS country, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} AND length(merchant_state) > 2
                GROUP BY 1 HAVING count(*) > 200
                ORDER BY fraud_rate DESC LIMIT 15"""
        ),
        out / "eda_by_country.json",
    )
    _write_json(
        grouped(
            f"""SELECT CASE WHEN merchant_state IS NULL THEN 'unknown'
                             WHEN length(merchant_state) = 2 THEN 'US state'
                             ELSE 'international' END AS region,
                       count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 3 DESC"""
        ),
        out / "eda_by_region.json",
    )
    _write_json(
        grouped(
            f"""SELECT card_brand, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 2 DESC"""
        ),
        out / "eda_by_brand.json",
    )
    _write_json(
        grouped(
            f"""SELECT card_type, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 2 DESC"""
        ),
        out / "eda_by_type.json",
    )
    _write_json(
        grouped(
            f"""SELECT year(ts) AS year, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_by_year.json",
    )
    _write_json(
        grouped(
            f"""SELECT merchant_id, any_value(merchant_city) AS city,
                       any_value(mcc_desc) AS category, count(*) AS frauds
                FROM tx_enriched {labeled} AND is_fraud = 1
                GROUP BY 1 ORDER BY 4 DESC LIMIT 15"""
        ),
        out / "eda_top_merchants.json",
    )
    _write_json(
        grouped(
            f"""SELECT ((credit_score - 300) / 69)::INT AS bucket,
                       min(credit_score) AS lo, max(credit_score) AS hi,
                       count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM tx_enriched {labeled} AND credit_score IS NOT NULL
                GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_by_credit.json",
    )
    _write_json(
        grouped(
            f"""SELECT is_night, is_weekend, count(*) AS n, avg(is_fraud) AS fraud_rate
                FROM (
                    SELECT (date_part('hour', ts) BETWEEN 0 AND 5)::INT AS is_night,
                           (date_part('dow', ts) IN (0, 6))::INT AS is_weekend,
                           is_fraud
                    FROM tx_enriched {labeled}
                ) GROUP BY 1, 2 ORDER BY 1, 2"""
        ),
        out / "eda_time_flags.json",
    )
    _write_json(
        grouped(
            f"""SELECT is_fraud, count(*) AS n, avg(amount) AS mean,
                       median(amount) AS median,
                       quantile_cont(amount, 0.25) AS q25,
                       quantile_cont(amount, 0.75) AS q75,
                       quantile_cont(amount, 0.95) AS q95
                FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
        ),
        out / "eda_amount_stats.json",
    )

    # --- supervised metrics / curves --------------------------------------
    metrics = json.loads(cfg.artifacts.metrics.read_text(encoding="utf-8"))
    _write_json(metrics, out / "metrics.json")

    preds = pl.read_parquet(cfg.artifacts.predictions)
    y_test = preds["is_fraud"].to_numpy().astype(int)
    scores = preds["score"].to_numpy()
    precision, recall, _ = precision_recall_curve(y_test, scores)
    _write_json(_downsample(precision, recall), out / "pr_curve.json")
    cost = _cost_curve(scores, y_test, cfg)
    _write_json(cost, out / "cost_curve.json")
    _write_json(_roc_curve(scores, y_test), out / "roc_curve.json")
    _write_json(
        _results_bundle(scores, y_test, cost["optimal"]["threshold"]),
        out / "results.json",
    )

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
    unsup_path = out / "unsupervised_metrics.json"
    if force or not unsup_path.exists():
        _write_json(score_unsupervised(cfg), unsup_path)
    anom_path = out / "anomaly_sample.parquet"
    if force or not anom_path.exists():
        anom = pl.read_parquet(cfg.artifacts.dir / "unsupervised_scores.parquet")
        if anom.height > 50_000:
            anom = anom.sample(50_000, seed=cfg.seed)
        anom.select("anomaly_score", "is_fraud").write_parquet(anom_path)

    return out
