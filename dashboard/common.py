"""Shared helpers for the multipage fraud dashboard.

The app is intentionally **standalone** - it imports only streamlit, plotly,
pandas and pyarrow, and reads the small committed ``dashboard_data/`` bundle.
That keeps it deployable to Streamlit Community Cloud with no raw data and no
dependency on the training package.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
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


def setup_page() -> None:
    """Set the page config and inject the shared theme. Call before any st call."""
    st.set_page_config(
        page_title="Credit-Card Fraud Detection",
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
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
              border-radius: 18px; padding: 30px 34px; color: white; margin-bottom: 10px;
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
          .callout.good {{ border-color: #10B981; background: #ECFDF5; color: #065F46; }}
          .small {{ color: {MUTED}; font-size: .85rem; }}
          .step {{
              background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px;
              padding:14px 16px; margin:6px 0;
          }}
          .step b {{ color: {INK}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "An end-to-end ML project on **13.3M** card transactions.\n\n"
        f"[Source on GitHub]({REPO_URL})"
    )


@st.cache_data(show_spinner=False)
def load_json(name: str):
    with open(BUNDLE / name, encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(BUNDLE / name)


def bundle_ready() -> bool:
    if not BUNDLE.exists():
        st.error(
            "Dashboard bundle not found. Run "
            "`uv run python scripts/06_export_dashboard.py` first."
        )
        return False
    return True


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
    html = (
        '<div class="hero">'
        "<h1>Credit-Card Fraud Detection</h1>"
        "<p>An end-to-end machine learning system on 13.3M card transactions (0.10% fraud).</p>"
        '<span class="pill">DuckDB</span><span class="pill">Polars</span>'
        '<span class="pill">LightGBM</span><span class="pill">scikit-learn</span>'
        '<span class="pill">Streamlit</span>'
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def callout(text: str, kind: str = "") -> None:
    st.markdown(f"<div class='callout {kind}'>{text}</div>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)


def metrics_glossary() -> None:
    """Explain every metric and why it was (or was not) chosen."""
    st.markdown(
        """
| Metric | What it measures | Why we use it |
|---|---|---|
| **Accuracy** | Share of all predictions that are correct | **Rejected.** At 0.1% fraud, "never fraud" scores 99.9% and catches nothing. |
| **ROC-AUC** | Chance a random fraud ranks above a random legit transaction | Familiar summary, but **optimistic** when positives are rare. |
| **PR-AUC** (average precision) | Area under the precision-recall curve | **Headline metric** — it only rewards performance on the rare positive class. |
| **Recall @ x% FPR** | Share of fraud caught while false alarms stay ≤ x% | Ties performance to a **fixed review budget** a team can staff. |
| **Precision @ top k%** | Accuracy among the riskiest k% of transactions | Tells you whether an **alert queue is actually usable**. |
| **Cost / transaction** | `missed_fraud × $200 + false_alarm × $5`, per transaction | Puts a **dollar value** on the exact errors a fraud team trades off. |
        """
    )
    st.caption(
        "The decision threshold is not 0.5 — it is chosen to minimise the cost above. "
        "$200 per missed fraud and $5 per manual review are explicit assumptions exposed "
        "in `config/config.yaml` so they can be re-tuned."
    )
