"""Interactive analytics dashboard (Streamlit).

Turns the 13.3M-row transaction table into a business narrative and surfaces the
supervised model's riskiest transactions plus the unsupervised anomaly scores.

Run:
    uv run streamlit run src/fraud_detection/dashboard.py
"""

from __future__ import annotations

import json
from typing import cast

import plotly.express as px
import streamlit as st

from fraud_detection import data
from fraud_detection.config import load_config

st.set_page_config(page_title="Credit-Card Fraud Analytics", layout="wide")


@st.cache_resource
def get_connection():
    cfg = load_config()
    return cfg, data.prepare(cfg)


@st.cache_data(show_spinner=False)
def q(_con, sql: str):
    return _con.execute(sql).df()


def main() -> None:
    cfg, con = get_connection()

    st.title("Credit-Card Fraud Analytics")
    st.caption("13.3M transactions · 2,000 cardholders · 0.10% fraud — DuckDB + LightGBM")

    s = data.summary(con)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions", f"{s['transactions']:,}")
    c2.metric("Cardholders", f"{s['users']:,}")
    c3.metric("Frauds", f"{s['frauds']:,}")
    c4.metric("Fraud rate", f"{cast(float, s['fraud_rate']) * 100:.3f}%")

    tab_overview, tab_model, tab_risk = st.tabs(["Fraud overview", "Model performance", "Top risk"])

    with tab_overview:
        monthly = q(
            con,
            """SELECT date_trunc('month', ts) AS month, avg(is_fraud) AS fraud_rate, count(*) AS n
               FROM tx_enriched WHERE is_fraud IS NOT NULL GROUP BY 1 ORDER BY 1""",
        )
        st.plotly_chart(
            px.line(monthly, x="month", y="fraud_rate", title="Monthly fraud rate"), width="stretch"
        )

        left, right = st.columns(2)
        hourly = q(
            con,
            """SELECT date_part('hour', ts) AS hour, avg(is_fraud) AS fraud_rate
               FROM tx_enriched WHERE is_fraud IS NOT NULL GROUP BY 1 ORDER BY 1""",
        )
        left.plotly_chart(px.bar(hourly, x="hour", y="fraud_rate", title="Fraud rate by hour"),
                          width="stretch")

        cats = q(
            con,
            """SELECT mcc_desc, count(*) AS frauds FROM tx_enriched
               WHERE is_fraud = 1 GROUP BY 1 ORDER BY 2 DESC LIMIT 15""",
        )
        right.plotly_chart(
            px.bar(cats, x="frauds", y="mcc_desc", orientation="h", title="Top categories by fraud count"),
            width="stretch",
        )

    with tab_model:
        if cfg.artifacts.metrics.exists():
            metrics = json.loads(cfg.artifacts.metrics.read_text(encoding="utf-8"))
            test = metrics["lightgbm"]["test"]
            base = metrics["baseline"]["test"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PR-AUC", f"{test['pr_auc']:.4f}", help=f"baseline {base['pr_auc']:.4f}")
            c2.metric("ROC-AUC", f"{test['roc_auc']:.4f}", help=f"baseline {base['roc_auc']:.4f}")
            c3.metric("Recall @ 1% FPR", f"{test['recall_at_fpr']['recall@fpr=0.010']:.3f}")
            c4.metric("Precision @ top 0.1%", f"{test['precision_at_k']['precision@top0.1%']:.3f}")
            st.json(test)
        else:
            st.info("Train the model first: scripts/03_train.py")

    with tab_risk:
        if cfg.artifacts.predictions.exists():
            preds = q(
                con,
                f"""SELECT p.transaction_id, p.ts, p.is_fraud, p.score
                    FROM read_parquet('{cfg.artifacts.predictions.as_posix()}') p
                    ORDER BY p.score DESC LIMIT 200""",
            )
            st.plotly_chart(
                px.histogram(preds, x="score", color="is_fraud", nbins=50,
                             title="Model score distribution (test window)"),
                width="stretch",
            )
            st.dataframe(preds.head(50), width="stretch")
        else:
            st.info("Train the model first: scripts/03_train.py")


if __name__ == "__main__":
    main()
