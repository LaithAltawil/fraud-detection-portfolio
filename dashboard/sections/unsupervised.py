"""Unsupervised anomaly detection extension and its negative result."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import RED, callout, load_json, load_parquet, page_header, style_fig


def render() -> None:
    page_header(
        "Unsupervised Extension",
        "Can anomaly detection find fraud without labels? Isolation Forest and LOF, benchmarked.",
    )
    st.markdown(
        """
Labels are not always available — new fraud patterns, cold-start, privacy. I
fitted **Isolation Forest** and **Local Outlier Factor** on legitimate training
transactions and scored the test window, using the held-out labels only as ground
truth for evaluation.
        """
    )

    u = load_json("unsupervised_metrics.json")
    m = load_json("metrics.json")["lightgbm"]["test"]
    comp = pd.DataFrame(
        {
            "Model": ["LightGBM (supervised)", "Isolation Forest", "Local Outlier Factor"],
            "ROC-AUC": [m["roc_auc"], u["isolation_forest"]["roc_auc"], u["lof"]["roc_auc"]],
            "PR-AUC": [m["pr_auc"], u["isolation_forest"]["pr_auc"], u["lof"]["pr_auc"]],
        }
    )
    fig = px.bar(
        comp.melt(id_vars="Model", var_name="metric", value_name="value"),
        x="Model",
        y="value",
        color="metric",
        barmode="group",
        title="Ranking quality by approach",
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    anom = load_parquet("anomaly_sample.parquet")
    anom["label"] = anom["is_fraud"].map({0: "legitimate", 1: "fraud"})
    fig = px.histogram(
        anom,
        x="anomaly_score",
        color="label",
        barmode="overlay",
        histnorm="probability density",
        nbins=60,
        title="Isolation Forest anomaly score distribution",
        color_discrete_map={"legitimate": "#CBD5E1", "fraud": RED},
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    callout(
        "<b>Negative result.</b> Isolation Forest (ROC 0.40) barely beats random "
        "and LOF (ROC 0.55) is weak. Fraud here is <b>not an outlier</b> — its "
        "amount and timing look statistically ordinary. Only labels teach the model "
        "which subtle behavioural deviations matter. Unsupervised detection is a "
        "screening layer, not a decision-maker.",
        "warn",
    )
    st.markdown(
        """
### Why this is a useful finding
A weak-but-honest unsupervised baseline **justifies** the supervised investment:
it shows that the performance gap comes from *labels and behavioural features*,
not from an arbitrary modelling choice. In production, an anomaly score could
still triage unseen merchants at the front of a review queue — but it cannot be
the final decision.
        """
    )
