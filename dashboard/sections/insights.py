"""Business insights, engineering decisions and next steps."""

from __future__ import annotations

import streamlit as st
from common import REPO_URL, callout, page_header


def render() -> None:
    page_header("Insights & Next Steps", "What the project proved and where it goes next.")

    st.markdown("### What drives detection")
    st.markdown(
        """
Behavioural, label-supervised features do the work: **how unusual an amount is
for that card** (`amount_z`), **how novel the merchant is**
(`card_merchant_prior_count`), and **how long since the card last transacted**
(`sec_since_prev_txn`). Static attributes (category, merchant state) added noise
and were dropped.
        """
    )

    st.markdown("### Engineering decisions that mattered")
    st.markdown(
        """
- **Causal features** — every behavioural signal uses only *past* transactions.
- **Chronological split** — fraud clusters in time; random splits leak.
- **Cost-based threshold** — $200 per missed fraud vs $5 per false alarm.
- **Rejected `scale_pos_weight`** — it collapsed ranking (ROC 0.93 → 0.60).
- **Feature ablation** — categoricals hurt; behavioural numerics carried the signal.
        """
    )
    callout(
        "LightGBM catches ~30% of fraud at a 1% false-positive budget and finds "
        "fraud in the riskiest 0.1% of transactions at 78× the base rate.",
        "good",
    )

    st.markdown("### Next steps")
    st.markdown(
        """
1. **SHAP explanations** for per-transaction reasoning in the review UI.
2. **FastAPI `/score` endpoint** + Docker image for real-time serving.
3. **Drift monitoring** on feature distributions over time.
4. **Cost tuning** once real fraud-loss figures replace the assumed $200/$5.
        """
    )

    st.markdown("---")
    st.markdown(
        f"""
**Stack:** DuckDB · Polars · LightGBM · scikit-learn · Streamlit

**Code:** [github.com/LaithAltawil/fraud-detection-portfolio]({REPO_URL})
        """
    )
