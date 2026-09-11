# Credit-Card Fraud Detection

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Model](https://img.shields.io/badge/model-LightGBM-brightgreen)
![App](https://img.shields.io/badge/app-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-8%20passing-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

An end-to-end fraud detection system on **13,305,915 synthetic card transactions**
(**0.10% fraud**) — a DuckDB data layer, leakage-safe causal features, a
cost-aware LightGBM model, an honest unsupervised benchmark, and a nine-page
interactive dashboard.

> **The one-sentence pitch for an interviewer:** I built the full pipeline a fraud
> team would own — raw files → SQL data layer → causal features → cost-aware
> ranking model → dashboard — and it catches ~30% of fraud at a 1% false-positive
> budget while documenting, honestly, where the unsupervised approach fails.

---

## Contents
- [Highlights](#highlights)
- [The problem](#the-problem)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Results](#results)
- [Metrics & why they were chosen](#metrics--why-they-were-chosen)
- [Interactive dashboard](#interactive-dashboard-multipage)
- [What I learned](#what-i-learned-interview-talking-points)
- [Project layout](#project-layout)
- [Limitations & next steps](#limitations--next-steps)

## Highlights
- **13.3M rows / 1.2 GB** processed out-of-core with **DuckDB** — no in-memory pandas.
- **Leakage-safe features** computed with prior-only window functions, validated on a
  **chronological** 70/15/15 split.
- **LightGBM** reaches **ROC-AUC 0.943** and **PR-AUC 0.055** (7× the logistic
  baseline), catching **~30% of fraud at a 1% FPR** and finding fraud in the
  riskiest 0.1% at a **78× lift** over the base rate.
- **Cost-aware threshold** chosen from a dollar matrix (`missed × $200 + review × $5`).
- **Honest negative result:** Isolation Forest (ROC 0.40) and LOF (ROC 0.55) fail —
  fraud is *not* an outlier, so labels and behavioural features are what matter.
- **Nine-page Streamlit dashboard** that runs with no raw data via a small committed bundle.

## The problem
Fraud is the canonical **extreme class-imbalance** problem: 1 fraud per ~1,000
transactions. **Accuracy is meaningless** — a model that always predicts
"legitimate" scores 99.9% and catches nothing. What actually matters is:

1. **Ranking** transactions by risk (who should a human look at first?), and
2. choosing a decision threshold from the **business cost** of a missed fraud
   versus a false alarm.

This project builds that whole system and reports the metrics a fraud team owns.

## Dataset
Five synthetic files. `transactions_data.csv` is 1.2 GB / 13.3M rows.

| File | Rows | Key fields |
|---|---|---|
| `transactions_data.csv` | 13,305,915 | id, timestamp, client/card id, amount, channel, merchant id/city/state, zip, MCC, errors |
| `users_data.csv` | 2,000 | demographics, address, income, debt, credit score |
| `cards_data.csv` | 6,146 | brand, type, number, expiry, chip, credit limit, dark-web flag |
| `mcc_codes.json` | 109 | merchant category code → description |
| `train_fraud_labels.json` | 8,915,963 | transaction id → fraud Yes/No |

**Data quirks handled once, in the data layer (`data.py`):**
- `amount` / incomes / limits are **currency strings** (`"$-77.00"`) → parsed via a `money()` SQL helper.
- Only **8.9M of 13.3M** rows are labeled → unlabeled rows stay `NULL`, never 0.
- The 151 MB nested label JSON **stalled DuckDB's JSON reader** → replaced with a streaming `JSONDecoder.raw_decode` parser cached to Parquet.
- `merchant_state` mixes **US state codes with country names** (Italy, Canada, …).

## Architecture
```
raw CSVs/JSON ──▶ DuckDB views ──▶ causal feature table (Parquet)
                     │                        │
                     │                        ├─▶ supervised  (LightGBM, cost-aware threshold)
                     │                        ├─▶ unsupervised (Isolation Forest / LOF)
                     │                        └─▶ dashboard   (multipage Streamlit)
                     └─▶ EDA
```

## Quickstart
The raw data is expected at `../archive` (see `config/config.yaml`).

```bash
uv sync --extra dashboard --extra dev     # or: pip install -r requirements.txt

uv run python scripts/01_eda.py             # summary + report figures
uv run python scripts/02_build_features.py  # artifacts/features.parquet
uv run python scripts/03_train.py           # baseline + LightGBM + metrics
uv run python scripts/04_unsupervised.py    # Isolation Forest + LOF
uv run python scripts/06_export_dashboard.py# small dashboard_data/ bundle
uv run python scripts/05_dashboard.py       # launch the multipage app

uv run pytest -q && uv run ruff check . && uv run mypy src
```

Set `features.sample_rows` in `config/config.yaml` for a fast local smoke test.

## Results
Test window = the most recent 15% of the labeled timeline (**1,337,243** transactions,
**2,353** frauds, base rate **0.176%**). Chronological, so these are true
future-performance numbers.

| Model | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ top 0.1% | Cost / txn |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.0077 | 0.856 | 0.053 | 0.017 | $0.351 |
| **LightGBM (final)** | **0.0555** | **0.943** | **0.297** | **0.137** | **$0.295** |

LightGBM lifts **PR-AUC 7×** over the baseline and finds **13.7%** precision in
the riskiest 0.1% of transactions — a **78× lift** over the 0.176% base rate —
while catching ~30% of fraud at a 1% false-positive budget.

### Unsupervised extension (negative result)
Fitting Isolation Forest and LOF on legitimate transactions and scoring the test
window did **not** work:

| Detector | PR-AUC | ROC-AUC |
|---|---|---|
| Isolation Forest | 0.0014 | 0.396 |
| Local Outlier Factor | 0.0025 | 0.549 |

Fraud here is **not an outlier** in behavioural feature space — it looks
statistically ordinary. This is a genuine finding, not a bug: it shows why
label-supervised methods dominate and frames unsupervised detection as a
screening layer, not a decision-maker.

### What drives detection
Behavioural, label-supervised features do the work: **how unusual an amount is for
that card** (`amount_z`), **how novel the merchant is**
(`card_merchant_prior_count`), and **how long since the card last transacted**
(`sec_since_prev_txn`). Conversely, **international merchant locations carry a
5.6% fraud rate** versus ~0.016% for US states — a strong geographic signal.

## Metrics & why they were chosen
At a 0.176% base rate the choice of metric *is* the project. This is what the
dashboard reports and why:

| Metric | What it measures | Why we use it |
|---|---|---|
| **Accuracy** | Correct predictions / all | **Rejected** — 99.9% for a model that predicts nothing |
| **ROC-AUC** | Chance a random fraud outranks a random legit txn | Familiar, but **optimistic** under extreme imbalance |
| **PR-AUC** (average precision) | Area under the precision-recall curve | **Headline** — rewards the rare positive class only |
| **Recall @ x% FPR** | Fraud caught while false alarms ≤ x% | Maps to a **fixed review budget** |
| **Precision @ top k%** | Accuracy among the riskiest k% | Is the **alert queue usable**? |
| **Cost / transaction** | `missed × $200 + review × $5`, per txn | Puts a **dollar value** on the trade-off |

The decision **threshold is not 0.5** — it is chosen to minimise the cost above.
The `$200 / $5` matrix is an explicit assumption in `config/config.yaml` so it can
be re-tuned to a real cost structure.

## Interactive dashboard (multipage)
A nine-page Streamlit app explains **everything done with the data**:

| Page | What it explains |
|---|---|
| **Executive summary** | Problem, KPIs, headline results, negative result, pipeline |
| **Data & pipeline** | The five files, quirks handled, DuckDB joins, why out-of-core |
| **Exploratory analysis** | 15 charts: time, hour, weekday, night/weekend, amount, category, channel, card brand/type, credit band, geography (US vs international), merchants |
| **Features & split** | Causal window-function features and the 70/15/15 chronological split |
| **Supervised model** | Baseline vs LightGBM, PR curve, feature importance, cost curve, top risk |
| **Results, explained** | Metric glossary, ROC/PR curves, confusion matrix at the optimal threshold, decile lift, budget trade-offs, savings |
| **Unsupervised extension** | Isolation Forest / LOF vs supervised, why it failed |
| **Models & usage** | What each of the four models is for and how to run it |
| **Insights** | Drivers, decisions, next steps |

It reads a small committed `dashboard_data/` bundle (~1 MB), so it runs with **no
raw data**.

### Deploy to Streamlit Community Cloud
1. Open <https://share.streamlit.io> → **New app** → select this repo.
2. **Main file path:** `dashboard/app.py`
3. Deploy. Streamlit installs `requirements.txt` and serves the app from the committed bundle.

## What I learned (interview talking points)
- **`scale_pos_weight` backfired.** Weighting the 1:1000 imbalance collapsed ranking
  (ROC 0.93 → 0.60). Ranking metrics + a cost threshold beat reweighting the loss.
- **More features ≠ better.** Merchant/category categoricals *reduced* ROC from
  0.92 to 0.86; behavioural velocity features carried the signal (ablation is
  configurable via `features.use_categoricals`).
- **Only 8.9M of 13.3M rows are labeled** — training must restrict to labeled rows.
- **Time-based splits matter.** A 2M-row early sample had *zero* fraud in its
  validation window, because fraud clusters in time.
- **Not all "state" values are states.** `merchant_state` mixes countries; splitting
  them revealed international merchants as a strong fraud signal.

## Project layout
```
config/config.yaml          # paths, split fractions, hyperparameters, cost matrix
src/fraud_detection/
  config.py                 # typed config loader
  data.py                   # DuckDB views + enriched fact table + label parser
  features.py               # causal feature engineering
  split.py                  # chronological train/valid/test split
  evaluate.py               # PR-AUC, recall@FPR, precision@k, cost curve
  train.py                  # logistic baseline + LightGBM + feature importance
  unsupervised.py           # Isolation Forest / LOF extension
  export_dashboard.py       # builds the small dashboard_data/ bundle
scripts/                    # 01_eda → 06_export_dashboard CLI entry points
dashboard/                  # multipage Streamlit app (deploy: dashboard/app.py)
  app.py                    # entrypoint + navigation
  common.py                 # theme, bundle loaders, metric glossary
  sections/                 # one module per page
dashboard_data/             # small, committed bundle that powers the dashboard
.streamlit/config.toml      # dashboard theme
tests/                      # unit tests (label parser, features, metrics)
```

## Limitations & next steps
- **Synthetic labels** — absolute precision/recall depend on the generator; the
  methodology, not the exact number, is the point.
- **Cost constants are assumed** ($200/$5) — tune with real fraud-loss figures.
- **Next:** SHAP per-transaction explanations, a FastAPI `/score` endpoint with
  Docker, and drift monitoring on feature distributions over time.

## References
- Repo: https://github.com/LaithAltawil/fraud-detection-portfolio
