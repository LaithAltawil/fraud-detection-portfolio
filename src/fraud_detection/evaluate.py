"""Evaluation for extreme class imbalance.

Accuracy is useless at a 0.10% base rate (predicting "no fraud" always scores
99.9%). We report instead:

* **PR-AUC** (average precision) - the headline ranking metric for rare positives.
* **ROC-AUC** - kept for comparability, but optimistic under imbalance.
* **Recall @ FPR** - how much fraud we catch at review budgets of 0.1-5%.
* **Precision@k** - accuracy of the top-k riskiest transactions.
* **Cost-optimal threshold** - the decision threshold that minimises
  ``FN x cost_FN + FP x cost_FP``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .config import EvaluationConfig


@dataclass
class ModelMetrics:
    pr_auc: float
    roc_auc: float
    recall_at_fpr: dict[str, float]
    precision_at_k: dict[str, float]
    best_threshold: float
    total_cost: float
    cost_per_txn: float
    n: int
    n_pos: int

    def to_dict(self) -> dict:
        return asdict(self)


def recall_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_targets: list[float]) -> dict[str, float]:
    """Recall (TPR) achieved while holding the false-positive rate <= target."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    out: dict[str, float] = {}
    for target in fpr_targets:
        out[f"recall@fpr={target:.3f}"] = float(np.interp(target, fpr, tpr))
    return out


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, fractions: list[float]) -> dict[str, float]:
    """Precision among the top-k highest-scored transactions."""
    order = np.argsort(-y_score)
    ranked = y_true[order]
    n = len(ranked)
    out: dict[str, float] = {}
    for frac in fractions:
        k = max(1, int(n * frac))
        out[f"precision@top{frac * 100:g}%"] = float(ranked[:k].mean())
    return out


def cost_optimal_threshold(
    y_true: np.ndarray, y_score: np.ndarray, cfg: EvaluationConfig
) -> tuple[float, float]:
    """Return the (threshold, total_cost) that minimises the business cost."""
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = int(y_true.sum())
    tp = np.cumsum(y_sorted)                 # TP if we flag top-i+1
    flagged = np.arange(1, len(y_sorted) + 1)
    fp = flagged - tp
    fn = n_pos - tp
    cost = fn * cfg.cost_false_negative + fp * cfg.cost_false_positive
    best = int(np.argmin(cost))
    threshold = float(y_score[order][best])
    return threshold, float(cost[best])


def evaluate(
    name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    cfg: EvaluationConfig,
) -> ModelMetrics:
    """Compute the full metric suite for a set of scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    threshold, total_cost = cost_optimal_threshold(y_true, y_score, cfg)

    metrics = ModelMetrics(
        pr_auc=float(average_precision_score(y_true, y_score)),
        roc_auc=float(roc_auc_score(y_true, y_score)),
        recall_at_fpr=recall_at_fpr(y_true, y_score, cfg.fpr_targets),
        precision_at_k=precision_at_k(y_true, y_score, [0.001, 0.01]),
        best_threshold=threshold,
        total_cost=total_cost,
        cost_per_txn=total_cost / len(y_true),
        n=int(len(y_true)),
        n_pos=int(y_true.sum()),
    )
    return metrics


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: str, title: str = "") -> None:
    """Save a precision-recall curve for the report."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    base = float(np.mean(y_true))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.axhline(base, ls="--", color="grey", label=f"baseline = {base:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title or "Precision-Recall")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
