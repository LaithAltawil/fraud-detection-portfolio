"""Landing page: problem, headline results and the end-to-end pipeline."""

from __future__ import annotations

import streamlit as st
from common import callout, hero, load_json


def render() -> None:
    hero()
    s = load_json("summary.json")
    m = load_json("metrics.json")
    test, base = m["lightgbm"]["test"], m["baseline"]["test"]
    lift = test["precision_at_k"]["precision@top0.1%"] / s["fraud_rate"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{s['transactions']:,}")
    c2.metric("Fraud rate", f"{s['fraud_rate'] * 100:.3f}%")
    c3.metric("ROC-AUC", f"{test['roc_auc']:.3f}", f"baseline {base['roc_auc']:.3f}")
    c4.metric("Lift @ top 0.1%", f"{lift:.0f}×")

    st.markdown("### The problem")
    st.markdown(
        """
Fraud is a **1-in-1,000** problem. Accuracy is meaningless — always predicting
"legitimate" scores 99.9%. What matters is **ranking** transactions by risk and
choosing a cut-off from the **business cost** of a missed fraud versus a false
alarm. This project builds the full pipeline a fraud team would own — and
documents honestly what worked and what did not.
        """
    )

    st.markdown("### Headline results (chronological hold-out)")
    r1, r2, r3 = st.columns(3)
    r1.metric("PR-AUC", f"{test['pr_auc']:.4f}", f"baseline {base['pr_auc']:.4f}")
    r2.metric("Recall @ 1% FPR", f"{test['recall_at_fpr']['recall@fpr=0.010']:.3f}")
    r3.metric("Precision @ top 0.1%", f"{test['precision_at_k']['precision@top0.1%']:.3f}")

    callout(
        f"<b>LightGBM catches ~30% of fraud at a 1% false-positive budget</b> and "
        f"finds fraud in the riskiest 0.1% of transactions at <b>{lift:.0f}×</b> the base rate."
    )
    callout(
        "<b>Honest negative result:</b> unsupervised anomaly detection (Isolation "
        "Forest, LOF) failed here — fraud is not an outlier, it looks statistically "
        "ordinary.",
        "warn",
    )

    st.markdown("### How the project fits together")
    steps = [
        ("1 · Data layer", "five raw files exposed as DuckDB views and joined into one enriched fact table (13.3M rows, out-of-core)."),
        ("2 · Exploratory analysis", "fraud by time, hour, weekday, category and channel on the 8.9M labeled transactions."),
        ("3 · Causal features", "prior-only behavioural signals (velocity, amount z-score, merchant novelty), split chronologically 70/15/15."),
        ("4 · Supervised model", "logistic baseline vs LightGBM, evaluated with PR-AUC, recall@FPR and a business cost curve."),
        ("5 · Unsupervised extension", "Isolation Forest and LOF benchmarked against the labels (negative result)."),
    ]
    for title, description in steps:
        st.markdown(
            f'<div class="step"><b>{title}</b> — {description}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("### Data at a glance")
    st.markdown(
        f"""
| | |
|---|---|
| Transactions | **{s['transactions']:,}** ({s['fraud_rate'] * 100:.3f}% fraud) |
| Labeled / unlabeled | **{s['labeled']:,}** / {s['unlabeled']:,} |
| Cardholders | {s['users']:,} |
| Cards | {s['cards']:,} |
| Period | {s['date_min'][:10]} → {s['date_max'][:10]} |
        """
    )
    callout(
        "Only <b>8.9M of 13.3M</b> transactions carry a label — supervised training "
        "restricts to those and keeps the rest as NULL, never as 0."
    )
