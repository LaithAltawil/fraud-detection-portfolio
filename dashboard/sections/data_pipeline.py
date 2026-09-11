"""How the raw data was loaded, cleaned and joined."""

from __future__ import annotations

import streamlit as st
from common import callout, load_json, page_header


def render() -> None:
    page_header(
        "Data & Pipeline",
        "Five raw files · 13.3M transactions · loaded and joined with DuckDB, out-of-core.",
    )
    s = load_json("summary.json")

    st.markdown("### The raw files")
    st.markdown(
        f"""
| File | Rows | What it holds |
|---|---|---|
| `transactions_data.csv` | **{s['transactions']:,}** | id, timestamp, client/card ids, amount, channel, merchant id/city/state, zip, MCC code, errors |
| `users_data.csv` | {s['users']:,} | cardholder demographics, address, income, debt, credit score |
| `cards_data.csv` | {s['cards']:,} | card brand/type, number, expiry, chip, credit limit, dark-web flag |
| `mcc_codes.json` | 109 | merchant category code → human-readable description |
| `train_fraud_labels.json` | **{s['labeled']:,}** | transaction id → fraud Yes/No |
        """
    )

    st.markdown("### Data-quality quirks handled once, in the data layer")
    st.markdown(
        """
- **Currency strings.** `amount` arrives as `"$-77.00"` and incomes/debts/limits
  as `"$59,696"`. A single `money()` SQL helper strips `$` and `,` and casts to
  `DOUBLE`. Negative amounts are refunds/credits and are kept as-is.
- **Sparse `errors` column.** Mostly empty, but treated as a **signal**, not junk.
- **Nullable `zip`.** Parsed as `DOUBLE` so it can be modelled or ignored cleanly.
- **Only 8.9M of 13.3M rows are labeled.** Unlabeled rows are kept as `NULL` (not
  coerced to 0) so the supervised model never learns from fake negatives.
- **A 151 MB nested label JSON.** DuckDB's JSON reader *hung* building a
  13.3M-key object; a streaming `JSONDecoder.raw_decode` parser extracts
  `(id, 0/1)` pairs and caches them to Parquet instead.
        """
    )
    callout(
        "Verified facts: <b>13,305,915</b> transactions, <b>13,332</b> frauds "
        "(0.1002%), 8,915,963 labeled rows."
    )

    st.markdown("### From files to one enriched fact table")
    st.markdown(
        """
Five raw files become a single denormalised view, `tx_enriched`, so every
downstream number (EDA, features, dashboard) is definitionally consistent:

```sql
transactions t
  LEFT JOIN cards  c ON t.card_id  = c.id
  LEFT JOIN users  u ON t.client_id = u.id
  LEFT JOIN mcc    m ON t.mcc       = m.mcc
  LEFT JOIN labels l ON t.id        = l.id
```

It exposes the transaction plus card attributes, cardholder profile, category
description and the fraud label in one place — **13.3M rows would not fit
comfortably in pandas**, so it stays behind DuckDB.
        """
    )

    st.markdown("### Why DuckDB")
    st.markdown(
        """
- Queries the 1.2 GB transaction CSV **without loading it into memory**.
- Window functions compute the causal behavioural features in pure SQL.
- Spills to disk (`temp_directory`) so 13.3M-row joins stay within a few GB of RAM.
- Columnar + predicate pushdown makes the EDA aggregates fast (< 1 second each).

The whole data layer lives in `src/fraud_detection/data.py` (`register_views`,
`create_enriched_view`, `_iter_label_pairs`).
        """
    )

    with st.expander("Show the per-file schema used"):
        st.code(
            """transactions: id, date, client_id, card_id, amount, use_chip,
              merchant_id, merchant_city, merchant_state, zip, mcc, errors
users:        id, current_age, retirement_age, birth_year, birth_month, gender,
              address, latitude, longitude, per_capita_income, yearly_income,
              total_debt, credit_score, num_credit_cards
cards:        id, client_id, card_brand, card_type, card_number, expires, cvv,
              has_chip, num_cards_issued, credit_limit, acct_open_date,
              year_pin_last_changed, card_on_dark_web""",
            language="text",
        )
