# Credit-Card Fraud Detection

An end-to-end fraud detection project on **13,305,915 synthetic card transactions**
(0.10% fraud) — from raw CSV to a cost-aware supervised model, an unsupervised
anomaly-detection comparison, and an interactive analytics dashboard.

> **One repo, three angles on the same problem.** A supervised centerpiece for
> accuracy-optimised detection, an unsupervised extension for when labels are
> scarce, and a dashboard that turns 13M rows into a business narrative.

---

## Why this dataset is interesting

| Property | Value | Consequence |
|---|---|---|
| Transactions | 13.3M rows / 1.2 GB | Needs columnar/out-of-core tooling (DuckDB + Parquet), not in-memory pandas |
| Fraud rate | ~0.10% (1:1000) | Accuracy is meaningless — use PR-AUC, recall @ FPR, precision@k, cost |
| Cardholders | 2,000 users → 6,146 cards | Heavy repeat behaviour; splits must be time/group-aware to avoid leakage |
| Signals | time, geo, merchant, MCC, card flags, dark-web | Rich relational joins across 5 files |

## Architecture

```
raw CSVs/JSON ──▶ DuckDB views ──▶ causal feature table (Parquet)
                     │                        │
                     │                        ├─▶ supervised  (LightGBM, cost-aware threshold)
                     │                        ├─▶ unsupervised (Isolation Forest / LOF)
                     │                        └─▶ dashboard   (Streamlit + DuckDB SQL)
                     └─▶ EDA
```

Every stage writes a reproducible artifact to `artifacts/`:

| Stage | Script | Output |
|---|---|---|
| EDA | `scripts/01_eda.py` | `reports/figures/*.png`, printed summary |
| Features | `scripts/02_build_features.py` | `artifacts/features.parquet` |
| Supervised | `scripts/03_train.py` | `artifacts/lightgbm_fraud.joblib`, `artifacts/metrics.json` |
| Unsupervised | `scripts/04_unsupervised.py` | `artifacts/unsupervised_scores.parquet` |
| Dashboard bundle | `scripts/06_export_dashboard.py` | `dashboard_data/` (small, committed) |
| Dashboard | `scripts/05_dashboard.py` | Streamlit app |

## Quickstart

```bash
uv sync --extra dashboard    # or: pip install -r requirements.txt
uv run python scripts/01_eda.py
uv run python scripts/02_build_features.py
uv run python scripts/03_train.py
uv run python scripts/04_unsupervised.py
uv run python scripts/06_export_dashboard.py   # build dashboard_data/ bundle
uv run python scripts/05_dashboard.py          # launch the app
```

The raw data is expected at `../archive` (see `config/config.yaml`). Set
`features.sample_rows` in the config for a fast local smoke test.

## Modeling notes

- **Leakage-safe validation.** Data is sorted by timestamp and split
  chronologically (70/15/15); the model never trains on the future.
- **Causal features.** Card-level rolling statistics use only *prior*
  transactions (`ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`), so a
  prediction never peeks at its own label or future events.
- **Cost-aware threshold.** The decision threshold is chosen to minimise
  `FN × $200 + FP × $5`, not to maximise accuracy.
- **Class imbalance.** Deliberately **not** handled with `scale_pos_weight` —
  it collapsed ranking (ROC 0.93 → 0.60). Imbalance is handled by ranking
  metrics plus the cost-based threshold.

## Results

Test window = the most recent 15% of the labeled timeline (1,337,243 transactions,
2,353 frauds, base rate 0.176%). Chronological split, so these are true
future-performance numbers.

| Model | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ top 0.1% | Cost / txn |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.0077 | 0.856 | 0.053 | 0.017 | $0.351 |
| **LightGBM (final)** | **0.0555** | **0.943** | **0.297** | **0.137** | **$0.295** |

LightGBM lifts PR-AUC **7x** over the baseline and finds **13.7%** precision in
the riskiest 0.1% of transactions — a **78x** lift over the 0.176% base rate —
while catching ~30% of fraud at a 1% false-positive budget.

### Unsupervised extension (negative result)

Fitting Isolation Forest and LOF on legitimate transactions and scoring the test
window did **not** work:

| Detector | PR-AUC | ROC-AUC |
|---|---|---|
| Isolation Forest | 0.0014 | 0.396 |
| Local Outlier Factor | 0.0025 | 0.549 |

Fraud here is **not an outlier** in behavioural feature space — it looks
statistically ordinary and is only separable once labels teach the model which
subtle deviations matter. This is a genuine finding, not a bug: it shows why
label-supervised methods dominate and frames unsupervised detection as a
screening layer, not a decision-maker.

## Interactive dashboard

A single Streamlit app tells the whole story in six sections:

1. **Executive summary** — KPIs, headline results, honest negative result.
2. **Data & EDA** — dataset facts, fraud by time / hour / weekday / category / channel.
3. **Supervised model** — baseline vs LightGBM, PR curve, feature importance,
   business-cost curve, riskiest transactions.
4. **Unsupervised extension** — Isolation Forest / LOF vs supervised, anomaly scores.
5. **Models & usage** — what each of the four models is for and how to run it.
6. **Insights** — what drives detection and next steps.

The app reads a small, committed `dashboard_data/` bundle (a few hundred KB)
produced by `scripts/06_export_dashboard.py`, so it runs with **no raw data** and
deploys without shipping the 1.2 GB dataset.

### Deploy to Streamlit Community Cloud (shareable link)

1. Push this repo (done).
2. Go to <https://share.streamlit.io> → **New app** → pick this repo.
3. **Main file path:** `src/fraud_detection/dashboard.py`
4. Deploy. Streamlit installs `requirements.txt` and serves the app from the
   committed `dashboard_data/` bundle — no data upload needed.

The `.streamlit/config.toml` theme is applied automatically.

## What I learned (interview talking points)

- **`scale_pos_weight` backfired.** Naively weighting the 1:1000 imbalance
  collapsed ranking (ROC 0.93 → 0.60). Imbalance was handled better by ranking
  metrics + a cost-based threshold than by reweighting the loss.
- **More features ≠ better.** Adding merchant/category categoricals *reduced*
  ROC from 0.92 to 0.86 on this synthetic data; behavioural velocity features
  (`card_merchant_prior_count`, `sec_since_prev_txn`, `amount_z`) carried the
  signal. Feature ablation is configurable (`features.use_categoricals`).
- **Only 8.9M of 13.3M transactions are labeled** — supervised training must
  restrict to labeled rows, and unlabeled rows must stay NULL, not 0.
- **Time-based splits matter.** A 2M-row early sample had *zero* fraud in the
  validation window, showing fraud clusters in time; the full chronological
  split is the only honest evaluation.

## Project layout

```
config/config.yaml          # single source of truth for paths + hyperparameters
src/fraud_detection/
  config.py                 # typed config loader
  data.py                   # DuckDB views + enriched fact table + label parser
  features.py               # causal feature engineering
  split.py                  # time-based split
  evaluate.py               # PR-AUC, recall@FPR, precision@k, cost curve
  train.py                  # logistic baseline + LightGBM
  unsupervised.py           # Isolation Forest / LOF extension
  export_dashboard.py       # build the small dashboard_data/ bundle
  dashboard.py              # unified Streamlit analytics app
scripts/
  01_eda.py                 # summary + report figures
  02_build_features.py      # build artifacts/features.parquet
  03_train.py               # train + evaluate + save artifacts
  04_unsupervised.py        # anomaly-detection extension
  05_dashboard.py           # launch Streamlit
  06_export_dashboard.py    # build dashboard_data/ for the deployed app
dashboard_data/             # small, committed bundle that powers the dashboard
.streamlit/config.toml      # dashboard theme
tests/                      # unit tests for the label parser, features, metrics
```
