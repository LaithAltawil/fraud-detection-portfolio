"""Exploratory analysis: the full set of fraud cuts with interpretation."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import (
    ACCENT,
    AMBER,
    INDIGO,
    RED,
    TEAL,
    callout,
    load_json,
    load_parquet,
    page_header,
    style_fig,
)

DOW = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def _rate_bar(df: pd.DataFrame, x: str, title: str, color: str = ACCENT) -> go.Figure:
    fig = px.bar(df, x=x, y="fraud_rate", title=title)
    fig.update_traces(marker_color=color)
    return style_fig(fig, 320)


def render() -> None:
    page_header(
        "Exploratory Analysis",
        "How fraud behaves across time, amount, category, channel and customer profile — on the 8.9M labeled transactions.",
    )
    s = load_json("summary.json")
    callout(
        f"Base rate to beat: <b>{s['fraud_rate'] * 100:.3f}%</b> "
        f"({s['frauds']:,} frauds in {s['transactions']:,} transactions). "
        "Every chart below is a hunt for where fraud concentrates."
    )

    # --- Time --------------------------------------------------------------
    st.markdown("### 1 · Fraud over time")
    monthly = pd.DataFrame(load_json("eda_monthly.json"))
    monthly["month"] = pd.to_datetime(monthly["month"])
    fig = px.line(monthly, x="month", y="fraud_rate", title="Monthly fraud rate")
    fig.update_traces(line_color=ACCENT, line_width=2.5)
    st.plotly_chart(style_fig(fig, 320), width="stretch")

    year = pd.DataFrame(load_json("eda_by_year.json"))
    fig = px.bar(year, x="year", y="fraud_rate", title="Fraud rate by year")
    fig.update_traces(marker_color=ACCENT)
    st.plotly_chart(style_fig(fig, 300), width="stretch")
    st.caption(
        "The rate drifts year to year. This non-stationarity is the whole reason the "
        "train/test split is **chronological** — a random split would score the model "
        "on years it effectively already saw."
    )

    # --- Hour / weekday ----------------------------------------------------
    st.markdown("### 2 · When does fraud happen?")
    hour = pd.DataFrame(load_json("eda_by_hour.json"))
    fig = go.Figure()
    fig.add_bar(x=hour["hour"], y=hour["n"], name="transactions", marker_color="#E2E8F0", yaxis="y2")
    fig.add_scatter(x=hour["hour"], y=hour["fraud_rate"], name="fraud rate", line=dict(color=RED, width=3))
    fig.update_layout(
        yaxis=dict(title="fraud rate"),
        yaxis2=dict(title="volume", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(style_fig(fig, 340, title="Fraud rate vs volume by hour"), width="stretch")

    left, right = st.columns(2)
    dow = pd.DataFrame(load_json("eda_by_dow.json"))
    dow["day"] = dow["dow"].astype(int).map(lambda i: DOW[i])
    left.plotly_chart(_rate_bar(dow, "day", "Fraud rate by day of week", TEAL), width="stretch")

    flags = pd.DataFrame(load_json("eda_time_flags.json"))
    flags["label"] = flags.apply(
        lambda r: ("night" if r["is_night"] else "day") + " · " + ("weekend" if r["is_weekend"] else "weekday"),
        axis=1,
    )
    right.plotly_chart(_rate_bar(flags, "label", "Fraud rate: night/weekend", AMBER), width="stretch")
    st.caption(
        "Fraud concentrates at night and on weekends — this motivates the calendar "
        "features (`txn_hour`, `txn_dow`, `is_night`, `is_weekend`)."
    )

    # --- Amount ------------------------------------------------------------
    st.markdown("### 3 · Does the amount give it away?")
    left, right = st.columns([3, 2])
    amt = load_parquet("amount_sample.parquet")
    amt = amt[(amt["amount"] > 0) & (amt["amount"] < 1500)]
    amt["label"] = amt["is_fraud"].map({0: "legitimate", 1: "fraud"})
    fig = px.histogram(
        amt, x="amount", color="label", barmode="overlay", histnorm="probability density",
        nbins=60, title="Amount distribution by class",
        color_discrete_map={"legitimate": "#CBD5E1", "fraud": RED},
    )
    left.plotly_chart(style_fig(fig, 360), width="stretch")

    stats = pd.DataFrame(load_json("eda_amount_stats.json"))
    stats["class"] = stats["is_fraud"].map({0: "legitimate", 1: "fraud"})
    stats = stats[["class", "n", "mean", "median", "q25", "q75", "q95"]]
    right.markdown("**Amount statistics**")
    right.dataframe(
        stats.style.format({"n": "{:,.0f}", "mean": "${:,.0f}", "median": "${:,.0f}",
                            "q25": "${:,.0f}", "q75": "${:,.0f}", "q95": "${:,.0f}"}),
        width="stretch", hide_index=True,
    )
    st.caption(
        "The distributions overlap heavily — raw amount alone separates the classes only "
        "weakly (ROC ≈ 0.56). Fraud must be measured **relative to each card's history**, "
        "which is exactly what `amount_z` and `amount_dev_from_prior_avg` do."
    )

    # --- Category / channel ------------------------------------------------
    st.markdown("### 4 · Where does fraud happen?")
    left, right = st.columns(2)
    cats = pd.DataFrame(load_json("eda_by_category.json")).sort_values("frauds")
    fig = px.bar(cats, x="frauds", y="mcc_desc", orientation="h", title="Top categories by fraud count")
    fig.update_traces(marker_color=INDIGO)
    left.plotly_chart(style_fig(fig, 380), width="stretch")

    chip = pd.DataFrame(load_json("eda_by_chip.json"))
    fig = px.bar(chip, x="use_chip", y="fraud_rate", text="n", title="Fraud rate by channel")
    fig.update_traces(marker_color=AMBER)
    right.plotly_chart(style_fig(fig, 380), width="stretch")
    st.caption(
        "Fraud hides in ordinary categories (groceries, gas, restaurants) — there is no "
        "'fraud category' to filter on. Online/`use_chip` channels carry a visibly higher rate."
    )

    # --- Card / customer profile ------------------------------------------
    st.markdown("### 5 · Which cards and customers are targeted?")
    left, right = st.columns(2)
    brand = pd.DataFrame(load_json("eda_by_brand.json"))
    left.plotly_chart(_rate_bar(brand, "card_brand", "Fraud rate by card brand", ACCENT), width="stretch")
    ctype = pd.DataFrame(load_json("eda_by_type.json"))
    right.plotly_chart(_rate_bar(ctype, "card_type", "Fraud rate by card type", TEAL), width="stretch")

    credit = pd.DataFrame(load_json("eda_by_credit.json"))
    credit["range"] = credit["lo"].astype(str) + "–" + credit["hi"].astype(str)
    fig = px.bar(credit, x="range", y="fraud_rate", title="Fraud rate by credit-score band")
    fig.update_traces(marker_color=INDIGO)
    st.plotly_chart(style_fig(fig, 320), width="stretch")
    st.caption(
        "Fraud is fairly flat across credit-score bands — customer credit quality is a weak "
        "predictor here, consistent with the feature-importance results."
    )

    # --- Geography / merchants --------------------------------------------
    st.markdown("### 6 · Geography and merchants")
    left, right = st.columns(2)
    state = pd.DataFrame(load_json("eda_by_state.json")).sort_values("fraud_rate")
    fig = px.bar(state, x="fraud_rate", y="state", orientation="h", title="Top states by fraud rate")
    fig.update_traces(marker_color=RED)
    left.plotly_chart(style_fig(fig, 460), width="stretch")

    merchants = pd.DataFrame(load_json("eda_top_merchants.json")).sort_values("frauds")
    merchants["label"] = merchants["city"] + " · " + merchants["category"].astype(str).str[:22]
    fig = px.bar(merchants, x="frauds", y="label", orientation="h", title="Merchants with most frauds")
    fig.update_traces(marker_color=TEAL)
    right.plotly_chart(style_fig(fig, 460), width="stretch")

    st.markdown("### What this told us")
    st.markdown(
        """
- Fraud is **time-dependent** and **behavioural**, not a fixed property of an
  amount, category or customer band.
- No single raw field separates the classes — the signal lives in **deviation from
  each card's own history**.
- These findings directly shaped the feature set: calendar features plus per-card
  velocity, merchant-novelty and amount-deviation features.
        """
    )
