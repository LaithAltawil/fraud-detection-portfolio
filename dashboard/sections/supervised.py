"""Supervised model: baseline vs LightGBM, evaluation and risk ranking."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from common import (
    ACCENT,
    INDIGO,
    MUTED,
    RED,
    TEAL,
    callout,
    load_json,
    load_parquet,
    metrics_glossary,
    page_header,
    style_fig,
)


def render() -> None:
    page_header(
        "Supervised Model",
        "Chronological hold-out · the decision threshold is chosen by business cost, not 0.5.",
    )
    m = load_json("metrics.json")
    test, base = m["lightgbm"]["test"], m["baseline"]["test"]

    st.markdown("### Baseline vs final model")
    table = pd.DataFrame(
        [
            [
                "Logistic regression (baseline)",
                base["pr_auc"],
                base["roc_auc"],
                base["recall_at_fpr"]["recall@fpr=0.010"],
                base["precision_at_k"]["precision@top0.1%"],
            ],
            [
                "LightGBM (final)",
                test["pr_auc"],
                test["roc_auc"],
                test["recall_at_fpr"]["recall@fpr=0.010"],
                test["precision_at_k"]["precision@top0.1%"],
            ],
        ],
        columns=["Model", "PR-AUC", "ROC-AUC", "Recall @ 1% FPR", "Precision @ top 0.1%"],
    )
    st.dataframe(
        table.style.format({c: "{:.4f}" for c in table.columns[1:]}),
        width="stretch",
        hide_index=True,
    )

    with st.expander("What these metrics mean & why we chose them", expanded=False):
        metrics_glossary()
        st.markdown(
            "**Why this matters:** with 0.176% positives, accuracy would be ~99.9% for a "
            "model that predicts nothing. We therefore rank with PR-AUC and pick the cut-off "
            "from a dollar-cost curve."
        )

    left, right = st.columns(2)
    pr = pd.DataFrame(load_json("pr_curve.json"))
    fig = go.Figure()
    fig.add_scatter(x=pr["recall"], y=pr["precision"], name="LightGBM", line=dict(color=ACCENT, width=3))
    fig.add_hline(
        y=load_json("summary.json")["fraud_rate"],
        line_dash="dash",
        line_color=MUTED,
        annotation_text="base rate",
    )
    left.plotly_chart(style_fig(fig, 360, title="Precision-Recall curve"), width="stretch")

    imp = pd.DataFrame(load_json("feature_importance.json")).sort_values("gain")
    fig = px_bar(imp)
    right.plotly_chart(style_fig(fig, 360), width="stretch")
    st.caption(
        "Gain importance confirms the story: the model ranks on **how unusual** a "
        "transaction is for that card, not on any single static attribute."
    )

    st.markdown("### Choosing the threshold from business cost")
    cost = load_json("cost_curve.json")
    cc = pd.DataFrame({"flagged_frac": cost["flagged_frac"], "cost_per_txn": cost["cost_per_txn"]})
    fig = go.Figure()
    fig.add_scatter(
        x=cc["flagged_frac"], y=cc["cost_per_txn"], mode="lines",
        name="expected cost", line=dict(color=INDIGO, width=3),
    )
    opt = cost["optimal"]
    fig.add_scatter(
        x=[opt["flagged_frac"]], y=[opt["cost_per_txn"]], mode="markers",
        name="cost-optimal", marker=dict(color=RED, size=13, symbol="star"),
    )
    fig.update_xaxes(title="fraction of transactions flagged")
    fig.update_yaxes(title="cost per transaction ($)")
    st.plotly_chart(style_fig(fig, 380, title="Business cost vs review budget"), width="stretch")
    callout(
        f"The optimal cut-off flags <b>{opt['flagged_frac'] * 100:.2f}%</b> of "
        f"transactions at <b>${opt['cost_per_txn']:.3f}</b> per transaction "
        f"(cost matrix: missed fraud = $200, false alarm = $5).",
        "good",
    )

    st.markdown("### Riskiest transactions in the test window")
    top = load_parquet("top_risk.parquet").rename(
        columns={
            "transaction_id": "Transaction ID",
            "ts": "Timestamp",
            "is_fraud": "Actually fraud",
            "score": "Risk score",
        }
    )
    top["Actually fraud"] = top["Actually fraud"].map({0: "No", 1: "Yes"})
    st.dataframe(
        top.head(25).style.format({"Risk score": "{:.5f}"}), width="stretch", hide_index=True
    )

    with st.expander("Training details"):
        st.markdown(
            f"""
- **Baseline:** logistic regression on a stratified 500k subsample, numeric
  features standardised, `class_weight='balanced'`.
- **LightGBM:** full training window ({m['lightgbm'].get('best_iteration', '—')} best
  iteration), `n_estimators=1500`, `learning_rate=0.05`, `num_leaves=63`,
  early-stopping on validation **average precision** ({m['lightgbm']['n_features']} features).
- **No `scale_pos_weight`.** Class weighting collapsed ranking here
  (ROC 0.93 → 0.60); imbalance is handled by PR metrics + the cost threshold.
            """
        )
        st.caption("PR-AUC on validation: " + str(round(m["lightgbm"]["valid"]["pr_auc"], 4)))


def px_bar(df: pd.DataFrame) -> go.Figure:
    import plotly.express as px

    fig = px.bar(df, x="gain", y="feature", orientation="h", title="Feature importance (gain)")
    fig.update_traces(marker_color=TEAL)
    return fig
