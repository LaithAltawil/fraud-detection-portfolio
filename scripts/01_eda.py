"""Exploratory data analysis: prints a summary and saves report figures.

Usage:
    uv run python scripts/01_eda.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fraud_detection import data
from fraud_detection.config import load_config

FIG_DIR = None


def _bar(series: pd.Series, title: str, xlabel: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraud rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=140)
    plt.close(fig)


def main() -> None:
    global FIG_DIR
    cfg = load_config()
    FIG_DIR = cfg.project_root / "reports" / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    con = data.prepare(cfg)
    print("Dataset summary:")
    for k, v in data.summary(con).items():
        print(f"  {k}: {v}")

    labeled = "WHERE is_fraud IS NOT NULL"

    # 1) Fraud rate by hour of day
    hourly = con.execute(
        f"""SELECT date_part('hour', ts) AS hr, avg(is_fraud) AS fraud_rate
            FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
    ).df()
    _bar(hourly.set_index("hr")["fraud_rate"], "Fraud rate by hour", "hour", "fraud_by_hour.png")

    # 2) Fraud rate by day of week
    dow = con.execute(
        f"""SELECT date_part('dow', ts) AS dow, avg(is_fraud) AS fraud_rate
            FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
    ).df()
    _bar(dow.set_index("dow")["fraud_rate"], "Fraud rate by day of week", "dow", "fraud_by_dow.png")

    # 3) Amount distribution, fraud vs legitimate (log-ish x)
    amounts = con.execute(
        f"""SELECT is_fraud, amount FROM tx_enriched {labeled} USING SAMPLE 500000 ROWS"""
    ).df()
    fig, ax = plt.subplots(figsize=(8, 4))
    for flag, label in [(0, "legitimate"), (1, "fraud")]:
        vals = amounts.loc[amounts["is_fraud"] == flag, "amount"]
        vals = vals[(vals > 0) & (vals < 2000)]
        ax.hist(vals, bins=60, alpha=0.6, density=True, label=label)
    ax.set_title("Transaction amount distribution")
    ax.set_xlabel("amount ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "amount_distribution.png", dpi=140)
    plt.close(fig)

    # 4) Top-15 categories by fraud count
    cats = con.execute(
        f"""SELECT mcc_desc, count(*) AS frauds
            FROM tx_enriched {labeled} AND is_fraud = 1
            GROUP BY 1 ORDER BY 2 DESC LIMIT 15"""
    ).df()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(cats["mcc_desc"][::-1], cats["frauds"][::-1])
    ax.set_title("Top categories by fraud count")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_by_category.png", dpi=140)
    plt.close(fig)

    # 5) Fraud rate over time (monthly)
    monthly = con.execute(
        f"""SELECT date_trunc('month', ts) AS month, avg(is_fraud) AS fraud_rate, count(*) AS n
            FROM tx_enriched {labeled} GROUP BY 1 ORDER BY 1"""
    ).df()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["month"], monthly["fraud_rate"])
    ax.set_title("Monthly fraud rate")
    ax.set_ylabel("fraud rate")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fraud_over_time.png", dpi=140)
    plt.close(fig)

    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
