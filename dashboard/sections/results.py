"""A deep, plain-language explanation of every result."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
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
    metrics_glossary,
    page_header,
    style_fig,
)


def _metric_table(test: dict, base: dict) -> pd.DataFrame:
    rows = []
    for name, m in [("Logistic (baseline)", base), ("LightGBM (final)", test)]:
        rows.append(
            {
                "Model": name,
                "PR-AUC": m["pr_auc"],
                "ROC-AUC": m["roc_auc"],
                "Recall@0.1% FPR": m["recall_at_fpr"]["recall@fpr=0.001"],
                "Recall@1% FPR": m["recall_at_fpr"]["recall@fpr=0.010"],
                "Recall@5% FPR": m["recall_at_fpr"]["recall@fpr=0.050"],
                "Precision@top0.1%": m["precision_at_k"]["precision@top0.1%"],
                "Cost / txn ($)": m["cost_per_txn"],
            }
        )
    return pd.DataFrame(rows)


def render() -> None:
    page_header(
        "Results, Explained",
        "What every number means, what the model actually gets right, and how the decision changes with the budget.",
    )

    m = load_json("metrics.json")
    test, base = m["lightgbm"]["test"], m["baseline"]["test"]
    summary = load_json("summary.json")

    st.markdown("### What the metrics mean — and why we selected them")
    metrics_glossary()

    st.markdown("### Model comparison (test window)")
    table = _metric_table(test, base)
    st.dataframe(
        table.style.format(
            {
                c: "{:.4f}"
                for c in table.columns
                if c not in ("Model", "Cost / txn ($)", "Recall@0.1% FPR", "Recall@1% FPR", "Recall@5% FPR")
            }
        ).format({"Cost / txn ($)": "${:.3f}"}),
        width="stretch",
        hide_index=True,
    )
    callout(
        f"LightGBM lifts PR-AUC from <b>{base['pr_auc']:.4f}</b> to "
        f"<b>{test['pr_auc']:.4f}</b> (~{test['pr_auc'] / base['pr_auc']:.0f}×) and ROC-AUC "
        f"from <b>{base['roc_auc']:.3f}</b> to <b>{test['roc_auc']:.3f}</b>.",
        "good",
    )

    # --- Ranking curves ----------------------------------------------------
    st.markdown("### How well does it rank?")
    left, right = st.columns(2)
    roc = pd.DataFrame(load_json("roc_curve.json"))
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], name="LightGBM", line=dict(color=ACCENT, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="random", line=dict(color=MUTED, dash="dash"))
    fig.update_xaxes(title="false positive rate")
    fig.update_yaxes(title="true positive rate")
    left.plotly_chart(style_fig(fig, 360, title="ROC curve"), width="stretch")

    pr = pd.DataFrame(load_json("pr_curve.json"))
    fig = go.Figure()
    fig.add_scatter(x=pr["recall"], y=pr["precision"], name="LightGBM", line=dict(color=TEAL, width=3))
    fig.add_hline(y=summary["fraud_rate"], line_dash="dash", line_color=MUTED, annotation_text="base rate")
    fig.update_xaxes(title="recall")
    fig.update_yaxes(title="precision")
    right.plotly_chart(style_fig(fig, 360, title="Precision-Recall curve"), width="stretch")

    # --- Confusion at chosen threshold ------------------------------------
    results = load_json("results.json")
    conf = results["confusion"]
    st.markdown("### The confusion matrix at the cost-optimal threshold")
    cm = pd.DataFrame(
        [[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]],
        index=["Actually legitimate", "Actually fraud"],
        columns=["Predicted legitimate", "Predicted fraud"],
    )
    left, right = st.columns([3, 2])
    left.dataframe(
        cm.style.format("{:,.0f}").set_caption("Counts on the test window"), width="stretch"
    )
    right.markdown(
        f"""
- **Precision:** `{conf['precision']:.1%}` of alerts are real fraud
- **Recall:** `{conf['recall']:.1%}` of all fraud is caught
- **False-positive rate:** `{conf['fpr']:.2%}`
- **Review budget:** `{conf['flagged_frac']:.2%}` of transactions flagged
        """
    )
    callout(
        f"At the optimal cut-off we flag <b>{conf['flagged_frac']:.2%}</b> of transactions, "
        f"catch <b>{conf['recall']:.1%}</b> of fraud, and every alert is right "
        f"<b>{conf['precision']:.0%}</b> of the time — a workable review queue.",
        "good",
    )

    # --- Decile lift -------------------------------------------------------
    st.markdown("### Decile lift: is the ranking usable?")
    deciles = pd.DataFrame(results["deciles"])
    fig = px.bar(deciles, x="decile", y="lift", text="frauds", title="Lift over base rate, by risk decile")
    fig.update_traces(marker_color=INDIGO)
    fig.update_xaxes(title="risk decile (1 = riskiest)")
    fig.update_yaxes(title="lift vs base rate")
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    top10 = deciles.iloc[0]
    callout(
        f"The riskiest decile contains <b>{top10['frauds']:,}</b> frauds "
        f"({top10['cumulative_recall']:.0%} of all fraud) at a rate <b>{top10['lift']:.0f}×</b> "
        f"the base rate — the score cleanly separates a small, actionable high-risk group."
    )

    # --- Budget scan -------------------------------------------------------
    st.markdown("### How results change with the review budget")
    scan = pd.DataFrame(results["threshold_scan"])
    fig = go.Figure()
    fig.add_scatter(x=scan["budget"], y=scan["recall"], name="recall", line=dict(color=RED, width=3))
    fig.add_scatter(x=scan["budget"], y=scan["precision"], name="precision", line=dict(color=ACCENT, width=3))
    fig.update_xaxes(title="fraction of transactions reviewed", tickformat=".0%")
    fig.update_yaxes(title="rate", tickformat=".0%")
    st.plotly_chart(style_fig(fig, 360, title="Recall & precision vs review budget"), width="stretch")
    st.caption(
        "This is the core trade-off: reviewing more transactions catches more fraud but "
        "lowers precision. The cost-optimal point balances the two against the dollar matrix."
    )

    # --- Cost --------------------------------------------------------------
    st.markdown("### What it saves")
    savings = base["total_cost"] - test["total_cost"]
    do_nothing = test["n_pos"] * 200
    c1, c2, c3 = st.columns(3)
    c1.metric("Cost / txn — baseline", f"${base['cost_per_txn']:.3f}")
    c2.metric("Cost / txn — LightGBM", f"${test['cost_per_txn']:.3f}", f"-${base['cost_per_txn'] - test['cost_per_txn']:.3f}")
    c3.metric("Test-window saving", f"${savings:,.0f}", "vs baseline")
    callout(
        f"Against the baseline, LightGBM saves <b>${savings:,.0f}</b> on the test window; "
        f"doing nothing (missing every fraud at $200) would cost "
        f"<b>${do_nothing:,.0f}</b>. Costs assume $200 per missed fraud and $5 per manual review."
    )

    # --- Unsupervised summary ---------------------------------------------
    u = load_json("unsupervised_metrics.json")
    st.markdown("### Unsupervised comparison")
    comp = pd.DataFrame(
        {
            "Approach": ["LightGBM (supervised)", "Isolation Forest", "Local Outlier Factor"],
            "ROC-AUC": [test["roc_auc"], u["isolation_forest"]["roc_auc"], u["lof"]["roc_auc"]],
            "PR-AUC": [test["pr_auc"], u["isolation_forest"]["pr_auc"], u["lof"]["pr_auc"]],
        }
    )
    st.dataframe(comp.style.format({"ROC-AUC": "{:.3f}", "PR-AUC": "{:.4f}"}), width="stretch", hide_index=True)
    st.caption(
        "Unsupervised detectors barely beat random — fraud is not an outlier in this feature "
        "space. The supervised model is not just 'more complex'; it is enabled by labels."
    )
