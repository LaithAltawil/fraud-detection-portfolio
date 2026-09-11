"""Supervised fraud model.

Two models are trained for the report:

1. **Logistic regression baseline** on a stratified subsample of the training
   window - fast, interpretable, establishes a floor.
2. **LightGBM** on the full training window, early-stopped on validation
   average precision.

The decision threshold is later chosen by business cost (see ``evaluate``), not
accuracy. Note that class weighting (``scale_pos_weight``) was tested and
rejected - it degraded ranking on this dataset.
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config
from .evaluate import evaluate, plot_pr_curve
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN
from .split import time_split


def frame_to_model_input(df: pl.DataFrame, use_categoricals: bool = True) -> pd.DataFrame:
    """Convert a Polars feature slice into a LightGBM-friendly pandas frame."""
    cols = NUMERIC_FEATURES + (CATEGORICAL_FEATURES if use_categoricals else [])
    pdf = df.select(cols).to_pandas()
    for col in NUMERIC_FEATURES:
        pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("float32")
    if use_categoricals:
        for col in CATEGORICAL_FEATURES:
            pdf[col] = pdf[col].astype("string").fillna("__missing__").astype("category")
    return pdf


def _labels(df: pl.DataFrame) -> np.ndarray:
    return df.select(TARGET_COLUMN).to_numpy().ravel().astype(int)


def train_baseline(
    train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, cfg: Config, max_rows: int = 500_000
) -> tuple[Pipeline, dict]:
    """Stratified-subsample logistic regression baseline on numeric features."""
    rng = np.random.default_rng(cfg.seed)
    y_all = _labels(train)
    pos_idx = np.flatnonzero(y_all == 1)
    neg_idx = np.flatnonzero(y_all == 0)
    n_neg = min(len(neg_idx), max_rows)
    neg_idx = rng.choice(neg_idx, size=n_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)

    X = train.select(NUMERIC_FEATURES).to_pandas().iloc[idx]
    y = y_all[idx]
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced")),
        ]
    )
    pipe.fit(X, y)

    def score(frame: pl.DataFrame) -> np.ndarray:
        Xf = frame.select(NUMERIC_FEATURES).to_pandas()
        Xf = Xf.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
        return pipe.predict_proba(Xf)[:, 1]

    metrics = {
        "test": evaluate("baseline", _labels(test), score(test), cfg.evaluation).to_dict()
    }
    return pipe, metrics


def train_lightgbm(
    train: pl.DataFrame, valid: pl.DataFrame, test: pl.DataFrame, cfg: Config
) -> tuple[lgb.LGBMClassifier, dict, pd.DataFrame]:
    """Train LightGBM with early stopping on the validation metric.

    We intentionally omit ``scale_pos_weight``: on this dataset it collapsed
    ranking quality (ROC 0.93 -> 0.60). Imbalance is handled downstream by the
    cost-based threshold.
    """
    use_cat = cfg.features.use_categoricals
    X_train = frame_to_model_input(train, use_cat)
    y_train = _labels(train)
    X_valid = frame_to_model_input(valid, use_cat)
    y_valid = _labels(valid)
    X_test = frame_to_model_input(test, use_cat)
    y_test = _labels(test)

    model = lgb.LGBMClassifier(
        **cfg.model.params,
        random_state=cfg.model.random_state,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=cfg.model.metric,
        callbacks=[
            lgb.early_stopping(cfg.model.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    valid_score = np.asarray(model.predict_proba(X_valid))[:, 1]
    test_score = np.asarray(model.predict_proba(X_test))[:, 1]

    metrics = {
        "valid": evaluate("lgbm", y_valid, valid_score, cfg.evaluation).to_dict(),
        "test": evaluate("lgbm", y_test, test_score, cfg.evaluation).to_dict(),
        "best_iteration": int(model.best_iteration_ or model.n_estimators_),
        "n_features": int(X_train.shape[1]),
        "use_categoricals": use_cat,
    }

    preds = test.select("transaction_id", "ts", "is_fraud").to_pandas()
    preds["score"] = test_score
    return model, metrics, preds


def save_artifacts(
    cfg: Config, model: lgb.LGBMClassifier, metrics: dict, preds: pd.DataFrame
) -> None:
    cfg.artifacts.dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg.artifacts.model)
    cfg.artifacts.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preds.to_parquet(cfg.artifacts.predictions, index=False)


def save_feature_importance(model: lgb.LGBMClassifier, cfg: Config, top_n: int = 20) -> None:
    """Save a LightGBM gain-importance bar chart for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(model.booster_.feature_name())
    gains = model.booster_.feature_importance(importance_type="gain")
    order = sorted(range(len(gains)), key=lambda i: gains[i], reverse=True)[:top_n]
    labels = [names[i] for i in order][::-1]
    values = [gains[i] for i in order][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, values)
    ax.set_title(f"LightGBM feature importance (gain, top {top_n})")
    fig.tight_layout()
    fig.savefig(cfg.project_root / "reports" / "figures" / "feature_importance.png", dpi=140)
    plt.close(fig)


def train(cfg: Config) -> dict:
    """Full supervised training entry point."""
    splits = time_split(cfg.artifacts.features.as_posix(), cfg)
    print(f"split sizes: {splits.sizes()}  train_end={splits.train_end}  valid_end={splits.valid_end}")

    _, baseline_metrics = train_baseline(splits.train, splits.valid, splits.test, cfg)
    model, lgbm_metrics, preds = train_lightgbm(splits.train, splits.valid, splits.test, cfg)

    metrics = {"baseline": baseline_metrics, "lightgbm": lgbm_metrics}
    save_artifacts(cfg, model, metrics, preds)
    save_feature_importance(model, cfg)

    plot_pr_curve(
        splits.test.select(TARGET_COLUMN).to_numpy().ravel().astype(int),
        preds["score"].to_numpy(),
        str(cfg.project_root / "reports" / "figures" / "pr_curve_test.png"),
        title="LightGBM - test precision-recall",
    )
    return metrics
