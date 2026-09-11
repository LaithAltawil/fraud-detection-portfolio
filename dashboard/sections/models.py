"""What each of the four models is for and how to run it."""

from __future__ import annotations

import streamlit as st
from common import page_header

MODELS = [
    {
        "name": "Logistic Regression",
        "tag": "Supervised · baseline",
        "rows": [
            ("Purpose", "An interpretable floor before reaching for boosting."),
            (
                "How to use",
                "Trains on a stratified 500k subsample of the numeric features "
                "(standardised), `class_weight='balanced'`.",
            ),
            ("Output", "Calibrated fraud probability with signed coefficients."),
            ("Command", "`uv run python scripts/03_train.py`"),
            ("Result", "ROC-AUC 0.856 · PR-AUC 0.0077"),
        ],
    },
    {
        "name": "LightGBM",
        "tag": "Supervised · final model",
        "rows": [
            ("Purpose", "Maximise ranking quality on imbalanced tabular data."),
            (
                "How to use",
                "Full training window, early-stops on validation average precision; "
                "the decision threshold comes from the cost curve, not 0.5.",
            ),
            ("Output", "A risk score per transaction for the review queue."),
            ("Command", "`uv run python scripts/03_train.py`"),
            ("Result", "ROC-AUC 0.943 · PR-AUC 0.0555 · 78× lift"),
        ],
    },
    {
        "name": "Isolation Forest",
        "tag": "Unsupervised · screening",
        "rows": [
            ("Purpose", "Flag fraud-like anomalies with no labels at all."),
            (
                "How to use",
                "Fit on legitimate training data, score new transactions; higher "
                "score = more anomalous. NaNs imputed to 0 (sklearn rejects them).",
            ),
            ("Output", "An anomaly score for triage, not an automatic block."),
            ("Command", "`uv run python scripts/04_unsupervised.py`"),
            ("Result", "ROC-AUC 0.396 — negative result"),
        ],
    },
    {
        "name": "Local Outlier Factor",
        "tag": "Unsupervised · density",
        "rows": [
            ("Purpose", "Density-based anomaly detection on local structure."),
            ("How to use", "Runs on a 50k subsample with `n_neighbors=20`."),
            ("Output", "A local-density outlier score."),
            ("Command", "`uv run python scripts/04_unsupervised.py`"),
            ("Result", "ROC-AUC 0.549 — weak"),
        ],
    },
]


def render() -> None:
    page_header(
        "The Four Models & How to Use Them",
        "Two supervised (production path) and two unsupervised (research extension).",
    )
    st.markdown(
        """
All four are evaluated on the **same chronological test window**. The supervised
models are the production path; the unsupervised models exist to answer "what if
we had no labels?".
        """
    )
    cols = st.columns(2)
    for i, model in enumerate(MODELS):
        with cols[i % 2]:
            card = st.container(border=True)
            card.markdown(f"### {model['name']}")
            card.caption(model["tag"])
            for key, value in model["rows"]:
                card.markdown(f"**{key}:** {value}")

    st.markdown("### When to reach for which")
    st.markdown(
        """
- **Have labels and want the best ranking** → LightGBM.
- **Need explainability / a monitored baseline** → logistic regression.
- **No labels yet, want to triage** → Isolation Forest (or LOF on smaller data).
- **Benchmarking a new dataset** → run the unsupervised pair first; if they score
  well, labels may be less critical — here they did not, which told us the signal
  lives in supervised behavioural features.
        """
    )
