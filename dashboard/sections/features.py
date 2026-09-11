"""The causal feature engineering and the chronological split."""

from __future__ import annotations

import streamlit as st
from common import callout, load_json, page_header


def render() -> None:
    page_header(
        "Feature Engineering & Split",
        "Every behavioural signal uses only transactions that happened before the current one.",
    )

    st.markdown("### Why causality matters")
    st.markdown(
        """
A fraud model is deployed to score a transaction **at the moment it happens**. If
a feature accidentally uses the current row's label, or information from the
future, the offline score is a lie. To prevent that, all rolling features are
computed with DuckDB window functions bounded to **strictly-prior rows**:

```sql
count(*)    OVER card_hist AS card_prior_txn_count,
avg(amount) OVER card_hist AS card_prior_avg_amount,
lag(ts)     OVER card_hist AS card_prev_ts
...
WINDOW card_hist AS (
    PARTITION BY card_id ORDER BY ts, transaction_id
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING   -- past only
)
```
        """
    )

    st.markdown("### The feature set (25 numeric features)")
    st.markdown(
        """
**Behavioural — the signal carriers**

| Feature | Meaning |
|---|---|
| `card_prior_txn_count` | Card velocity so far (how active the card is) |
| `card_merchant_prior_count` | How often this card used this merchant before (novelty) |
| `merchant_prior_count` | Merchant's overall transaction count so far |
| `card_prior_avg_amount` / `_std_amount` / `_max_amount` | The card's spending history |
| `sec_since_prev_txn` | Seconds since this card's previous transaction |
| `amount_dev_from_prior_avg` | Current amount minus the card's prior average |
| `amount_z` | Deviation in standard deviations (how unusual the amount is) |
| `is_first_txn` | First transaction on the card |
| `amount_to_limit` | Amount as a fraction of the credit limit |

**Calendar** — `txn_hour`, `txn_dow`, `txn_month`, `is_weekend`, `is_night`

**Static profile** — `amount`, `current_age`, `credit_score`, `num_credit_cards`,
`per_capita_income`, `yearly_income`, `total_debt`, `credit_limit`, `card_age_years`
        """
    )

    st.markdown("### Feature ablation: more is not always better")
    callout(
        "Adding merchant/category **categorical** attributes (state, MCC, brand) "
        "<b>reduced</b> ROC from 0.92 to 0.86 on this synthetic data — they added "
        "noise. The final model trains on the behavioural numeric features only "
        "(toggle: <code>features.use_categoricals</code>).",
        "warn",
    )

    split = load_json("split.json")
    st.markdown("### Chronological split (70 / 15 / 15)")
    st.markdown(
        f"""
Data is sorted by timestamp and cut in time — **never randomly** — because the
2,000 cardholders transact repeatedly and a random split would leak near-duplicate
behaviour across folds.

| Split | Rows | Ends |
|---|---|---|
| Train | {split['train']:,} | {split['train_end'][:16]} |
| Validation | {split['valid']:,} | {split['valid_end'][:16]} |
| Test | {split['test']:,} | (latest) |
        """
    )
    callout(
        "The test window is the most recent 15% of the labeled timeline, so reported "
        "metrics are true <b>future-performance</b> numbers.",
        "good",
    )
