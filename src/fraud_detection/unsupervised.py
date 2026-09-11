"""Unsupervised anomaly-detection extension.

Labels are not always available (new fraud patterns, cold-start, privacy). This
module fits detectors on **legitimate-looking training data only** and scores the
test window, evaluating against the held-out labels purely as ground truth.

We compare:

* Isolation Forest - tree-based, scales well, ``contamination`` set near base rate.
* Local Outlier Factor - density-based, strong on local structure but O(n^2),
  so it runs on a subsample.

The point of the exercise is to show how far unsupervised methods get *without*
labels, and why they are usually a triage/screening layer rather than a final
decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import Config
from .evaluate import evaluate
from .split import time_split
from .train import frame_to_model_input


@dataclass
class UnsupervisedResult:
    scores: np.ndarray
    metrics: dict


def _numeric_matrix(df: pl.DataFrame) -> np.ndarray:
    """Numeric feature matrix with NaNs/infs imputed to 0.

    LightGBM handles missing values natively, but the scikit-learn detectors here
    (Isolation Forest, LOF) do not, so we impute once at the boundary.
    """
    pdf = frame_to_model_input(df, use_categoricals=False)
    return np.nan_to_num(
        pdf.to_numpy(dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0
    )


def score_unsupervised(
    cfg: Config, max_fit_rows: int = 400_000, max_lof_rows: int = 50_000
) -> dict:
    """Fit detectors on the training window and score the test window."""
    splits = time_split(cfg.artifacts.features.as_posix(), cfg)
    rng = np.random.default_rng(cfg.seed)

    def subsample(df: pl.DataFrame, n: int) -> pl.DataFrame:
        if df.height <= n:
            return df
        idx = rng.choice(df.height, size=n, replace=False)
        return df[idx]

    fit_df = subsample(splits.train, max_fit_rows)
    X_fit = _numeric_matrix(fit_df)
    X_test = _numeric_matrix(splits.test)
    y_test = splits.test.select("is_fraud").to_numpy().ravel().astype(int)

    scaler = StandardScaler().fit(X_fit)
    X_fit_s = scaler.transform(X_fit)
    X_test_s = scaler.transform(X_test)

    results: dict[str, dict] = {}

    # 1) Isolation Forest
    iso = IsolationForest(
        n_estimators=200, contamination=float(y_test.mean()), random_state=cfg.seed, n_jobs=-1
    )
    iso.fit(X_fit_s)
    iso_scores = -iso.score_samples(X_test_s)  # higher = more anomalous
    results["isolation_forest"] = evaluate(
        "isolation_forest", y_test, iso_scores, cfg.evaluation
    ).to_dict()

    # 2) Local Outlier Factor on a subsample of the test window
    lof_idx = rng.choice(len(X_test_s), size=min(max_lof_rows, len(X_test_s)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=False, contamination=float(y_test.mean()))
    lof.fit(X_test_s[lof_idx])
    lof_scores = -lof.negative_outlier_factor_
    results["lof"] = evaluate(
        "lof", y_test[lof_idx], lof_scores, cfg.evaluation
    ).to_dict()

    return results


def write_unsupervised_predictions(cfg: Config, max_fit_rows: int = 400_000) -> None:
    """Persist Isolation Forest test scores for the dashboard."""
    splits = time_split(cfg.artifacts.features.as_posix(), cfg)
    rng = np.random.default_rng(cfg.seed)
    fit_df = splits.train
    if fit_df.height > max_fit_rows:
        fit_df = fit_df[rng.choice(fit_df.height, size=max_fit_rows, replace=False)]

    scaler = StandardScaler().fit(_numeric_matrix(fit_df))
    X_test = _numeric_matrix(splits.test)
    contamination = float(splits.test.select(pl.col("is_fraud").mean()).item())
    iso = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=cfg.seed, n_jobs=-1
    )
    iso.fit(scaler.transform(_numeric_matrix(fit_df)))
    scores = -iso.score_samples(scaler.transform(X_test))

    out = splits.test.select("transaction_id", "ts", "is_fraud").to_pandas()
    out["anomaly_score"] = scores
    out.to_parquet(cfg.artifacts.dir / "unsupervised_scores.parquet", index=False)
