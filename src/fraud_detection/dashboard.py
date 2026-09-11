"""Unified fraud-analytics dashboard (Streamlit).

One app, five deep sections:
    1. Executive summary   2. Data & EDA   3. Supervised model
    4. Unsupervised extension   5. Model usage & insights

It reads the small, committed ``dashboard_data/`` bundle, so it runs with no raw
data and deploys cleanly to Streamlit Community Cloud.

Run:
    uv run streamlit run src/fraud_detection/dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "dashboard_data"
REPO_URL = "https://github.com/LaithAltawil/fraud-detection-portfolio"

ACCENT = "#0EA5E9"
TEAL = "#14B8A6"
INDIGO = "#6366F1"
AMBER = "#F59E0B"
RED = "#EF4444"
INK = "#0F172A"
MUTED = "#64748B"
PALETTE = [ACCENT, TEAL, INDIGO, AMBER, RED]

st.set_page_config(
    page_title="Credit-Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1150px; }}
      html, body, [class*="css"] {{ font-family: 'Inter', 'Segoe UI', sans-serif; }}
      h1, h2, h3 {{ color: {INK}; letter-spacing: -0.02em; }}
      [data-testid="stMetric"] {{
          background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px;
          padding: 16px 18px; box-shadow: 0 1px 2px rgba(15,23,42,.04);
      }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; font-weight: 600; }}
      [data-testid="stMetricValue"] {{ color: {INK}; font-weight: 700; }}
      .hero {{
          background: linear-gradient(120deg, #0F172A 0%, #0E7490 100%);
          border-radius: 18px; padding: 30px 34px; color: white; margin-bottom: 8px;
      }}
      .hero h1 {{ color: white; margin: 0 0 6px 0; font-size: 2.1rem; }}
      .hero p {{ color: #CBD5E1; margin: 0; font-size: 1.02rem; }}
      .pill {{
          display: inline-block; background: rgba(255,255,255,.14); color: #E2E8F0;
          border-radius: 999px; padding: 3px 12px; margin: 10px 6px 0 0; font-size: .78rem;
      }}
      .callout {{
          border-left: 4px solid {ACCENT}; background: #F0F9FF; border-radius: 8px;
          padding: 12px 16px; margin: 8px 0; color: #0C4A6E;
      }}
      .callout.warn {{ border-color: {AMBER}; background: #FFFBEB; color: #78350F; }}
      .small {{ color: {MUTED}; font-size: .85rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(BUNDLE / name, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / name)


def style_fig(fig: go.Figure, height: int = 360, title: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=title,
        colorway=PALETTE,
        margin=dict(l=10, r=10, t=48, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", color=INK),
        title_font=dict(size=16),
    )
    return fig


def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>Credit-Card Fraud Detection</h1>
          <p>An end-to-end machine learning system on 13.3M card transactions (0.10% fraud).</p>
          <span class="pill">DuckDB</span><span class="pill">Polars</span>
          <span class="pill">LightGBM</span><span class="pill">scikit-learn</span>
          <span class="pill">Streamlit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def section_overview() -> None:
    hero()
    s = load_json("summary.json")
    m = load_json("metrics.json")
    test = m["lightgbm"]["test"]
    base = m["baseline"]["test"]
    lift = test["precision_at_k"]["precision@top0.1%"] / s["fraud_rate"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{s['transactions']:,}")
    c2.metric("Fraud rate", f"{s['fraud_rate'] * 100:.3f}%")
    c3.metric("ROC-AUC", f"{test['roc_auc']:.3f}", f"baseline {base['roc_auc']:.3f}")
    c4.metric("Lift @ top 0.1%", f"{lift:.0f}×")

    st.markdown("### The problem")
    st.markdown(
        """
Fraud is a **1-in-1,000** problem. Accuracy is meaningless here — always
predicting "legitimate" scores 99.9%. What matters is **ranking** transactions
by risk and choosing a cut-off from the **business cost** of a miss versus a
false alarm. This project builds the full pipeline a fraud team would own, and
documents honestly what worked.
        """
    )

    st.markdown("### Headline results (chronological hold-out)")
    r1, r2, r3 = st.columns(3)
    r1.metric("PR-AUC", f"{test['pr_auc']:.4f}", f"baseline {base['pr_auc']:.4f}")
    r2.metric("Recall @ 1% FPR", f"{test['recall_at_fpr']['recall@fpr=0.010']:.3f}")
    r3.metric("Precision @ top 0.1%", f"{test['precision_at_k']['precision@top0.1%']:.3f}")

    st.markdown(
        f"""
<div class="callout">
<b>LightGBM catches ~30% of fraud at a 1% false-positive budget</b> and finds
fraud in the riskiest 0.1% of transactions at <b>{lift:.0f}×</b> the base rate.
</div>
<div class="callout warn">
<b>Honest negative result:</b> unsupervised anomaly detection (Isolation Forest,
LOF) failed here — fraud is not an outlier, it looks statistically ordinary.
</div>
        """,
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


def section_eda() -> None:
    st.markdown("## Data & Exploratory Analysis")
    st.caption("All rates computed on the 8.9M labeled transactions.")

    monthly = pd.DataFrame(load_json("eda_monthly.json"))
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig = px.line(monthly, x="month", y="fraud_rate", title="Monthly fraud rate")
    fig.update_traces(line_color=ACCENT, line_width=2.5)
    st.plotly_chart(style_fig(fig, 340), width="stretch")

    left, right = st.columns(2)
    hour = pd.DataFrame(load_json("eda_by_hour.json"))
    fig = px.bar(hour, x="hour", y="fraud_rate", title="Fraud rate by hour of day")
    fig.update_traces(marker_color=ACCENT)
    left.plotly_chart(style_fig(fig, 320), width="stretch")

    dow_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    dow = pd.DataFrame(load_json("eda_by_dow.json"))
    dow["day"] = dow["dow"].astype(int).map(lambda i: dow_names[i])
    fig = px.bar(dow, x="day", y="fraud_rate", title="Fraud rate by day of week")
    fig.update_traces(marker_color=TEAL)
    right.plotly_chart(style_fig(fig, 320), width="stretch")

    left2, right2 = st.columns(2)
    amt = load_parquet("amount_sample.parquet")
    amt = amt[(amt["amount"] > 0) & (amt["amount"] < 1500)]
    amt["label"] = amt["is_fraud"].map({0: "legitimate", 1: "fraud"})
    fig = px.histogram(
        amt, x="amount", color="label", barmode="overlay", histnorm="probability density",
        nbins=60, title="Transaction amount distribution",
        color_discrete_map={"legitimate": "#CBD5E1", "fraud": RED},
    )
    left2.plotly_chart(style_fig(fig, 340), width="stretch")

    cats = pd.DataFrame(load_json("eda_by_category.json")).sort_values("frauds")
    fig = px.bar(
        cats, x="frauds", y="mcc_desc", orientation="h", title="Top categories by fraud count"
    )
    fig.update_traces(marker_color=INDIGO)
    right2.plotly_chart(style_fig(fig, 340), width="stretch")

    chip = pd.DataFrame(load_json("eda_by_chip.json"))
    st.markdown("### Fraud rate by transaction channel")
    fig = px.bar(chip, x="use_chip", y="fraud_rate", text="n", title="")
    fig.update_traces(marker_color=AMBER)
    st.plotly_chart(style_fig(fig, 300), width="stretch")


def section_supervised() -> None:
    st.markdown("## Supervised Model")
    st.caption("Chronological 70/15/15 split · threshold chosen by business cost.")

    m = load_json("metrics.json")
    test, base = m["lightgbm"]["test"], m["baseline"]["test"]

    table = pd.DataFrame(
        [
            ["Logistic regression (baseline)", base["pr_auc"], base["roc_auc"],
             base["recall_at_fpr"]["recall@fpr=0.010"], base["precision_at_k"]["precision@top0.1%"]],
            ["LightGBM (final)", test["pr_auc"], test["roc_auc"],
             test["recall_at_fpr"]["recall@fpr=0.010"], test["precision_at_k"]["precision@top0.1%"]],
        ],
        columns=["Model", "PR-AUC", "ROC-AUC", "Recall @ 1% FPR", "Precision @ top 0.1%"],
    )
    st.dataframe(table.style.format({c: "{:.4f}" for c in table.columns[1:]}), width="stretch", hide_index=True)

    left, right = st.columns(2)
    pr = pd.DataFrame(load_json("pr_curve.json"))
    fig = go.Figure()
    fig.add_scatter(x=pr["recall"], y=pr["precision"], name="LightGBM", line=dict(color=ACCENT, width=3))
    fig.add_hline(y=load_json("summary.json")["fraud_rate"], line_dash="dash", line_color=MUTED,
                  annotation_text="base rate")
    left.plotly_chart(style_fig(fig, 360, title="Precision-Recall curve"), width="stretch")

    imp = pd.DataFrame(load_json("feature_importance.json")).sort_values("gain")
    fig = px.bar(imp, x="gain", y="feature", orientation="h", title="Feature importance (gain)")
    fig.update_traces(marker_color=TEAL)
    right.plotly_chart(style_fig(fig, 360), width="stretch")

    cost = load_json("cost_curve.json")
    cc = pd.DataFrame(
        {"flagged_frac": cost["flagged_frac"], "cost_per_txn": cost["cost_per_txn"]}
    )
    fig = go.Figure()
    fig.add_scatter(x=cc["flagged_frac"], y=cc["cost_per_txn"], mode="lines",
                    name="expected cost", line=dict(color=INDIGO, width=3))
    opt = cost["optimal"]
    fig.add_scatter(x=[opt["flagged_frac"]], y=[opt["cost_per_txn"]], mode="markers",
                    name="cost-optimal threshold", marker=dict(color=RED, size=13, symbol="star"))
    fig.update_xaxes(title="fraction of transactions flagged")
    fig.update_yaxes(title="cost per transaction ($)")
    st.plotly_chart(style_fig(fig, 380, title="Business cost vs review budget"), width="stretch")
    st.markdown(
        f"<div class='callout'>The optimal cut-off flags <b>{opt['flagged_frac'] * 100:.2f}%</b> "
        f"of transactions at a cost of <b>${opt['cost_per_txn']:.3f}</b> per transaction "
        f"(cost matrix: missed fraud = $200, false alarm = $5).</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Riskiest transactions in the test window")
    top = load_parquet("top_risk.parquet")
    top = top.rename(
        columns={"transaction_id": "Transaction ID", "ts": "Timestamp", "is_fraud": "Actually fraud", "score": "Risk score"}
    )
    top["Actually fraud"] = top["Actually fraud"].map({0: "No", 1: "Yes"})
    st.dataframe(top.head(25).style.format({"Risk score": "{:.5f}"}), width="stretch", hide_index=True)


def section_unsupervised() -> None:
    st.markdown("## Unsupervised Extension")
    st.markdown(
        """
Labels are not always available. Can anomaly detection find fraud **without**
labels? I fitted Isolation Forest and LOF on legitimate training transactions
and scored the test window.
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
        x="Model", y="value", color="metric", barmode="group", title="Ranking quality by approach",
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    anom = load_parquet("anomaly_sample.parquet")
    anom["label"] = anom["is_fraud"].map({0: "legitimate", 1: "fraud"})
    fig = px.histogram(
        anom, x="anomaly_score", color="label", barmode="overlay", histnorm="probability density",
        nbins=60, title="Isolation Forest anomaly score distribution",
        color_discrete_map={"legitimate": "#CBD5E1", "fraud": RED},
    )
    st.plotly_chart(style_fig(fig, 360), width="stretch")

    st.markdown(
        """
<div class="callout warn">
<b>Negative result.</b> Isolation Forest (ROC 0.40) barely beats random and LOF
(ROC 0.55) is weak. Fraud here is <b>not an outlier</b> — its amount and timing
look statistically ordinary. Only labels teach the model which subtle behavioural
deviations matter. Unsupervised detection is a screening layer, not a decision-maker.
</div>
        """,
        unsafe_allow_html=True,
    )


def section_models() -> None:
    hero_models = [
        ("Logistic Regression", "Supervised · baseline", [
            ("Purpose", "Interpretable floor before reaching for boosting."),
            ("How to use", "Trains on a stratified 500k subsample of numeric features "
                           "(standardised), class_weight='balanced'."),
            ("Run", "`scripts/03_train.py`"),
            ("Result", "ROC-AUC 0.856 · PR-AUC 0.0077"),
        ]),
        ("LightGBM", "Supervised · final model", [
            ("Purpose", "Maximise ranking quality on imbalanced tabular data."),
            ("How to use", "Full training window, early-stops on validation average "
                           "precision; threshold set by cost, not 0.5."),
            ("Run", "`scripts/03_train.py`"),
            ("Result", "ROC-AUC 0.943 · PR-AUC 0.0555 · 78× lift"),
        ]),
        ("Isolation Forest", "Unsupervised · screening", [
            ("Purpose", "Flag fraud-like anomalies with no labels."),
            ("How to use", "Fit on legitimate training data, score new transactions; "
                           "higher = more anomalous."),
            ("Run", "`scripts/04_unsupervised.py`"),
            ("Result", "ROC-AUC 0.396 — negative result"),
        ]),
        ("Local Outlier Factor", "Unsupervised · density", [
            ("Purpose", "Density-based anomaly detection on local structure."),
            ("How to use", "Runs on a 50k subsample, n_neighbors=20."),
            ("Run", "`scripts/04_unsupervised.py`"),
            ("Result", "ROC-AUC 0.549 — weak"),
        ]),
    ]
    st.markdown("## The Four Models & How to Use Them")
    cols = st.columns(2)
    for i, (name, tag, rows) in enumerate(hero_models):
        with cols[i % 2]:
            st.markdown(f"### {name}")
            st.caption(tag)
            for k, v in rows:
                st.markdown(f"**{k}:** {v}")
            st.markdown("---")


def section_insights() -> None:
    st.markdown("## Insights & Next Steps")
    st.markdown(
        """
#### What drives detection
Behavioural, label-supervised features do the work: **how unusual an amount is for
that card** (`amount_z`), **how novel the merchant is** (`card_merchant_prior_count`),
and **how long since the card last transacted** (`sec_since_prev_txn`). Static
attributes (category, merchant state) added noise and were dropped.

#### Engineering decisions that mattered
- **Causal features** — every behavioural signal uses only *past* transactions.
- **Chronological split** — fraud clusters in time; random splits leak.
- **Cost-based threshold** — $200 per missed fraud vs $5 per false alarm.
- **Rejected `scale_pos_weight`** — it collapsed ranking (ROC 0.93 → 0.60).

#### Next steps
- Add SHAP explanations for per-transaction reasoning.
- Wrap the model in a FastAPI `/score` endpoint with a Docker image.
- Add drift monitoring on feature distributions over time.
        """
    )
    st.markdown(
        f"<p class='small'>Built with DuckDB · Polars · LightGBM · scikit-learn · Streamlit. "
        f"<a href='{REPO_URL}'>View the code on GitHub</a>.</p>",
        unsafe_allow_html=True,
    )


def main() -> None:
    if not BUNDLE.exists():
        st.error(
            "Dashboard bundle not found. Run `uv run python scripts/06_export_dashboard.py` first."
        )
        return

    st.sidebar.markdown("### 💳 Fraud Analytics")
    page = st.sidebar.radio(
        "Section",
        [
            "Executive summary",
            "Data & EDA",
            "Supervised model",
            "Unsupervised extension",
            "Models & usage",
            "Insights",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "An end-to-end ML project on **13.3M** transactions.\n\n"
        f"[GitHub repo]({REPO_URL})"
    )

    {
        "Executive summary": section_overview,
        "Data & EDA": section_eda,
        "Supervised model": section_supervised,
        "Unsupervised extension": section_unsupervised,
        "Models & usage": section_models,
        "Insights": section_insights,
    }[page]()


if __name__ == "__main__":
    main()
