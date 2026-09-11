"""Exploratory analysis of the labeled transactions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from common import (
    ACCENT,
    AMBER,
    INDIGO,
    RED,
    TEAL,
    load_json,
    load_parquet,
    page_header,
    style_fig,
)


def render() -> None:
    page_header(
        "Exploratory Analysis",
        "All rates computed on the 8.9M labeled transactions.",
    )

    monthly = pd.DataFrame(load_json("eda_monthly.json"))
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig = px.line(monthly, x="month", y="fraud_rate", title="Monthly fraud rate")
    fig.update_traces(line_color=ACCENT, line_width=2.5)
    st.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(
        "Fraud is a temporal process — the rate drifts year to year, which is exactly "
        "why the train/test split must be chronological."
    )

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
    st.caption(
        "Fraud concentrates at night and on particular weekdays — this motivates the "
        "calendar features (`txn_hour`, `txn_dow`, `is_night`, `is_weekend`)."
    )

    left2, right2 = st.columns(2)
    amt = load_parquet("amount_sample.parquet")
    amt = amt[(amt["amount"] > 0) & (amt["amount"] < 1500)]
    amt["label"] = amt["is_fraud"].map({0: "legitimate", 1: "fraud"})
    fig = px.histogram(
        amt,
        x="amount",
        color="label",
        barmode="overlay",
        histnorm="probability density",
        nbins=60,
        title="Transaction amount distribution",
        color_discrete_map={"legitimate": "#CBD5E1", "fraud": RED},
    )
    left2.plotly_chart(style_fig(fig, 340), width="stretch")

    cats = pd.DataFrame(load_json("eda_by_category.json")).sort_values("frauds")
    fig = px.bar(
        cats, x="frauds", y="mcc_desc", orientation="h", title="Top categories by fraud count"
    )
    fig.update_traces(marker_color=INDIGO)
    right2.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(
        "Fraud is spread across ordinary categories (groceries, gas, restaurants) — "
        "there is no single 'fraud category' to filter on, so the model needs "
        "behavioural signals."
    )

    chip = pd.DataFrame(load_json("eda_by_chip.json"))
    fig = px.bar(chip, x="use_chip", y="fraud_rate", text="n", title="Fraud rate by channel")
    fig.update_traces(marker_color=AMBER)
    st.plotly_chart(style_fig(fig, 300), width="stretch")

    st.markdown("### What this told us")
    st.markdown(
        """
- Fraud is **time-dependent** and **behavioural**, not a fixed property of an
  amount or a category.
- Raw amount alone separates classes only weakly (ROC ≈ 0.56) — features must be
  **relative to each card's history**.
- These findings shaped the feature set: calendar features + per-card velocity
  and deviation, rather than one-hot encoding the merchant universe.
        """
    )
