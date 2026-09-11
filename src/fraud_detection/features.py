"""Causal feature engineering.

Features are built in DuckDB over the full 13.3M-row enriched view and written to
Parquet. The important design choice is **causality**: every behavioural feature
uses only transactions that happened *before* the current one, via window
functions bounded with ``ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING``.
That way a prediction can never peek at the current row's label or at the future.

Feature families
----------------
* **Static** - amount, merchant category, card attributes, cardholder profile.
* **Calendar** - hour, day-of-week, month, weekend/night flags.
* **Behavioural (causal)** - prior transaction count, prior average/std/max
  amount on the card, deviation & z-score of the current amount versus that
  history, seconds since the previous card transaction, prior count at the same
  merchant.
* **Derived** - amount-to-credit-limit ratio, card age in years.
"""

from __future__ import annotations

import duckdb

from .config import Config

# Feature columns consumed by the model (order matters for reproducibility).
NUMERIC_FEATURES = [
    "amount",
    "txn_hour",
    "txn_dow",
    "txn_month",
    "is_weekend",
    "is_night",
    "current_age",
    "credit_score",
    "num_credit_cards",
    "per_capita_income",
    "yearly_income",
    "total_debt",
    "credit_limit",
    "amount_to_limit",
    "card_prior_txn_count",
    "card_merchant_prior_count",
    "card_prior_avg_amount",
    "card_prior_std_amount",
    "card_prior_max_amount",
    "sec_since_prev_txn",
    "amount_dev_from_prior_avg",
    "amount_z",
    "is_first_txn",
    "card_age_years",
    "merchant_prior_count",
]

CATEGORICAL_FEATURES = [
    "use_chip",
    "errors",
    "merchant_state",
    "mcc",
    "mcc_desc",
    "card_brand",
    "card_type",
    "has_chip",
    "card_on_dark_web",
    "gender",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "is_fraud"


def feature_sql(sample_rows: int | None = None) -> str:
    """Return the SQL that produces the modelling table.

    ``sample_rows`` limits the base scan for fast local smoke tests; ``None``
    uses all 13.3M transactions (the default for the real run).
    """
    limit = f"LIMIT {int(sample_rows)}" if sample_rows else ""
    return f"""
    WITH base AS (
        SELECT * FROM tx_enriched
        {limit}
    ),
    windowed AS (
        SELECT
            *,
            count(*)        OVER card_hist           AS card_prior_txn_count,
            avg(amount)     OVER card_hist           AS card_prior_avg_amount,
            stddev_pop(amount) OVER card_hist        AS card_prior_std_amount,
            max(amount)     OVER card_hist           AS card_prior_max_amount,
            lag(ts)         OVER card_hist           AS card_prev_ts,
            count(*)        OVER card_merchant_hist  AS card_merchant_prior_count,
            count(*)        OVER merchant_hist       AS merchant_prior_count
        FROM base
        WINDOW
            card_hist AS (
                PARTITION BY card_id
                ORDER BY ts, transaction_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            card_merchant_hist AS (
                PARTITION BY card_id, merchant_id
                ORDER BY ts, transaction_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            ),
            merchant_hist AS (
                PARTITION BY merchant_id
                ORDER BY ts, transaction_id
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
    )
    SELECT
        transaction_id,
        ts,
        is_fraud,
        amount,
        date_part('hour', ts)                    AS txn_hour,
        date_part('dow', ts)                     AS txn_dow,
        date_part('month', ts)                   AS txn_month,
        (date_part('dow', ts) IN (0, 6))::INTEGER AS is_weekend,
        (date_part('hour', ts) BETWEEN 0 AND 5)::INTEGER AS is_night,
        use_chip,
        errors,
        merchant_state,
        mcc,
        mcc_desc,
        card_brand,
        card_type,
        has_chip,
        card_on_dark_web,
        current_age,
        gender,
        credit_score,
        num_credit_cards,
        per_capita_income,
        yearly_income,
        total_debt,
        credit_limit,
        amount / nullif(credit_limit, 0)         AS amount_to_limit,
        card_prior_txn_count,
        card_merchant_prior_count,
        merchant_prior_count,
        coalesce(card_prior_avg_amount, 0)       AS card_prior_avg_amount,
        coalesce(card_prior_std_amount, 0)       AS card_prior_std_amount,
        coalesce(card_prior_max_amount, amount)  AS card_prior_max_amount,
        epoch(ts - card_prev_ts)                 AS sec_since_prev_txn,
        amount - card_prior_avg_amount           AS amount_dev_from_prior_avg,
        (amount - card_prior_avg_amount) / nullif(card_prior_std_amount, 0) AS amount_z,
        (card_prev_ts IS NULL)::INTEGER          AS is_first_txn,
        year(ts) - year(acct_open_date)          AS card_age_years
    FROM windowed
    """


def build_features(con: duckdb.DuckDBPyConnection, cfg: Config, force: bool = False) -> str:
    """Materialise the feature table to Parquet and return its path (as string)."""
    out = cfg.artifacts.features
    if out.exists() and not force:
        return out.as_posix()

    out.parent.mkdir(parents=True, exist_ok=True)
    sql = feature_sql(cfg.features.sample_rows)
    con.execute(
        f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return out.as_posix()
