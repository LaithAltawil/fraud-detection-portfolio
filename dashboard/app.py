"""Multipage fraud-analytics dashboard (entrypoint).

Run locally:
    uv run streamlit run dashboard/app.py

Deploy (Streamlit Community Cloud):
    main file path: dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import streamlit as st  # noqa: E402
from common import bundle_ready, setup_page, sidebar_footer  # noqa: E402
from sections import (  # noqa: E402
    data_pipeline,
    eda,
    features,
    insights,
    models,
    overview,
    results,
    supervised,
    unsupervised,
)

setup_page()

if not bundle_ready():
    st.stop()

st.sidebar.markdown("### 💳 Fraud Analytics")

nav = st.navigation(
    [
        st.Page(overview.render, title="Executive summary", icon="📊", url_path="overview", default=True),
        st.Page(data_pipeline.render, title="Data & pipeline", icon="🗂️", url_path="data"),
        st.Page(eda.render, title="Exploratory analysis", icon="🔎", url_path="eda"),
        st.Page(features.render, title="Features & split", icon="🧮", url_path="features"),
        st.Page(supervised.render, title="Supervised model", icon="🏆", url_path="supervised"),
        st.Page(results.render, title="Results, explained", icon="📈", url_path="results"),
        st.Page(unsupervised.render, title="Unsupervised extension", icon="🧪", url_path="unsupervised"),
        st.Page(models.render, title="Models & usage", icon="🤖", url_path="models"),
        st.Page(insights.render, title="Insights", icon="💡", url_path="insights"),
    ]
)
sidebar_footer()
nav.run()
