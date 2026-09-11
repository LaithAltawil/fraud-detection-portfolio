import numpy as np

from fraud_detection.config import EvaluationConfig
from fraud_detection.evaluate import cost_optimal_threshold, evaluate


def test_perfect_ranking_scores_one() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9, 0.95])
    m = evaluate("t", y, scores, EvaluationConfig())
    assert m.roc_auc == 1.0
    assert m.pr_auc > 0.99


def test_cost_threshold_beats_flagging_nothing() -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 0.9])
    cfg = EvaluationConfig(cost_false_negative=100.0, cost_false_positive=1.0)
    _, total = cost_optimal_threshold(y, scores, cfg)
    flag_nothing = y.sum() * cfg.cost_false_negative
    assert total <= flag_nothing


def test_recall_at_fpr_reported() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(1000) < 0.1).astype(int)
    scores = rng.random(1000) + y * 0.5
    m = evaluate("t", y, scores, EvaluationConfig())
    assert set(m.recall_at_fpr) == {
        "recall@fpr=0.001",
        "recall@fpr=0.005",
        "recall@fpr=0.010",
        "recall@fpr=0.050",
    }
