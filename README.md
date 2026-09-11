# Credit-Card Fraud Detection

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Model](https://img.shields.io/badge/model-LightGBM-brightgreen)
![App](https://img.shields.io/badge/app-Streamlit-ff4b4b)
![Tests](https://img.shields.io/badge/tests-8%20passing-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A production-style fraud detection system built on **13,305,915 card
transactions** (**0.10% fraud**). It covers the full delivery lifecycle — data
engineering, exploratory analysis, leakage-safe feature development, a
cost-aware machine learning model, an evaluation framework aligned to business
economics, and an interactive analytics dashboard.

The model catches **~30% of fraud at a 1% false-positive rate** and identifies
the riskiest 0.1% of transactions with a **78× lift** over the base rate, while
supporting a decision threshold driven by the financial cost of fraud versus
manual review.

---

## Contents
- [At a glance](#at-a-glance)
- [Business context](#business-context)
- [Dataset](#dataset)
- [Approach](#approach)
- [Results](#results)
- [Metrics & why they were chosen](#metrics--why-they-were-chosen)
- [Interactive dashboard](#interactive-dashboard)
- [Technology](#technology)
- [Project structure](#project-structure)
- [Running the project](#running-the-project)
- [Limitations & future work](#limitations--future-work)

## At a glance
- **Scale:** 13.3M transactions / 1.2 GB processed out-of-core — no sampling required.
- **Performance:** LightGBM reaches **ROC-AUC 0.943** and **PR-AUC 0.055**, a **7×**
  improvement over a logistic-regression baseline.
- **Business value:** catches **~30% of fraud at a 1% false-positive rate** and
  **13.7% precision in the riskiest 0.1%** of transactions.
- **Decision economics:** the alert threshold is selected to minimise
  `missed_fraud × cost + false_alarm × cost`, not to maximise accuracy.
- **Interpretability:** every prediction rests on transparent, per-card
  behavioural features (amount deviation, merchant novelty, transaction velocity).

## Business context
Fraud detection is a high-stakes, extreme-imbalance problem: roughly **1 fraud per
1,000 transactions**. In this setting **accuracy is meaningless** — a system that
approves every transaction is 99.9% accurate and prevents zero fraud.

What matters to a fraud operations team is:
1. **Prioritisation** — which transactions should a human review first?
2. **Economics** — how many frauds are caught, and at what cost in false alarms?

This system is designed around those two questions, and reports the metrics that
reflect them.

## Dataset
Five related files are joined into a single transaction fact table.

| File | Rows | Key fields |
|---|---|---|
| `transactions_data.csv` | 13,305,915 | id, timestamp, client/card id, amount, channel, merchant id/city/state, zip, MCC code, errors |
| `users_data.csv` | 2,000 | demographics, address, income, debt, credit score |
| `cards_data.csv` | 6,146 | brand, type, card number, expiry, chip, credit limit, dark-web flag |
| `mcc_codes.json` | 109 | merchant category code → description |
| `train_fraud_labels.json` | 8,915,963 | transaction id → fraud Yes/No |

**Data quality is handled in one place (the data layer):**
- Monetary fields arrive as **currency strings** (`"$-77.00"`, `"$59,696"`) and are parsed centrally.
- Only **8.9M of 13.3M** transactions carry a label; unlabeled rows are preserved as
  `NULL` so the model never trains on fabricated negatives.
- A 151 MB nested label file that stalled the default JSON reader is parsed with a
  streaming decoder and cached to a columnar format.
- `merchant_state` mixes **US states with country names**, which are separated for analysis.

## Approach
```
Raw CSVs / JSON ──▶ SQL data layer ──▶ causal feature table (Parquet)
                         │                         │
                         │                         ├─▶ Supervised model (LightGBM, cost-aware threshold)
                         │                         ├─▶ Unsupervised benchmark (Isolation Forest / LOF)
                         │                         └─▶ Interactive dashboard
                         └─▶ Exploratory analysis
```

Highlights of the engineering:
- **Out-of-core processing.** A columnar SQL engine (DuckDB) queries the full
  1.2 GB dataset directly, keeping memory usage predictable.
- **Leakage-safe features.** Every behavioural feature uses only transactions that
  occurred *before* the one being scored (prior-only window functions). This
  mirrors how a deployed model actually sees data.
- **Chronological validation.** Data is split **70/15/15 by time**, not at random,
  because cardholder behaviour repeats and fraud trends shift over time.
- **Cost-aware decisions.** The classification threshold is chosen from a business
  cost curve rather than a default 0.5.

## Results
Test window = the most recent 15% of the timeline (**1,337,243** transactions,
**2,353** frauds, base rate **0.176%**).

| Model | PR-AUC | ROC-AUC | Recall @ 1% FPR | Precision @ top 0.1% | Cost / transaction |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.0077 | 0.856 | 0.053 | 0.017 | $0.351 |
| **LightGBM (final)** | **0.0555** | **0.943** | **0.297** | **0.137** | **$0.295** |

The final model provides a **7× PR-AUC improvement** over the baseline and finds
fraud in the riskiest 0.1% of transactions at **78× the base rate**.

### Unsupervised benchmark
As a comparison, anomaly-detection methods were fitted without labels and scored
against the held-out ground truth:

| Detector | PR-AUC | ROC-AUC |
|---|---|---|
| Isolation Forest | 0.0014 | 0.396 |
| Local Outlier Factor | 0.0025 | 0.549 |

The result is clear and useful: fraud in this dataset is **not an outlier** purely
by statistical distribution — it looks ordinary. The supervised, label-driven
behavioural features are what make detection possible. This establishes that
unsupervised methods serve as a **screening layer**, not a final decision-maker.

### What drives detection
Behavioural features do the heavy lifting:
- **`amount_z`** — how unusual a transaction amount is for that specific card.
- **`card_merchant_prior_count`** — how novel the merchant is to the card.
- **`sec_since_prev_txn`** — the time since the card last transacted.

Geography is another strong signal: **international merchants carry a 5.6% fraud
rate versus ~0.016% for US states.**

## Metrics & why they were chosen
With a 0.176% base rate, the choice of metric is central to the project.

| Metric | What it measures | Why it is used |
|---|---|---|
| **Accuracy** | Correct predictions / all predictions | **Rejected** — 99.9% for a model that blocks nothing |
| **ROC-AUC** | Chance a random fraud ranks above a random legitimate transaction | Familiar summary, but optimistic under extreme imbalance |
| **PR-AUC** (average precision) | Area under the precision-recall curve | **Headline metric** — focuses on the rare positive class |
| **Recall @ x% FPR** | Fraud caught while false alarms stay ≤ x% | Directly maps to a review-team budget |
| **Precision @ top k%** | Accuracy among the riskiest k% | Indicates whether the alert queue is actionable |
| **Cost / transaction** | Missed fraud and false alarms priced together | Expresses performance in financial terms |

The decision threshold is not fixed at 0.5. It is selected to **minimise total
cost** (`$200` per missed fraud, `$5` per manual review) — an assumption kept in
configuration so it can be tuned to any organisation's cost structure.

## Interactive dashboard
A nine-page Streamlit application presents the full analysis for both technical
and business audiences:

| Page | Contents |
|---|---|
| **Executive summary** | Key performance indicators, headline results, and the analysis pipeline |
| **Data & pipeline** | Source files, data-quality handling, table joins, and out-of-core processing |
| **Exploratory analysis** | 15 charts: fraud over time, hour, weekday, amount, category, channel, card type, credit band, geography, and merchants |
| **Features & split** | The causal feature set and the chronological train/validation/test split |
| **Supervised model** | Baseline vs final model, precision-recall curve, feature importance, and cost curve |
| **Results, explained** | Metric definitions, ROC/PR curves, confusion matrix at the chosen threshold, decile lift, and review-budget trade-offs |
| **Unsupervised extension** | Isolation Forest / LOF compared against the supervised model |
| **Models & usage** | Purpose, inputs/outputs, and commands for each model |
| **Insights** | Key drivers, engineering decisions, and next steps |

The dashboard runs from a small, self-contained data bundle, so it presents
without requiring access to the full dataset.

## Technology
| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Data engine | DuckDB, Polars, pandas, PyArrow |
| Machine learning | LightGBM, scikit-learn |
| Visualisation | Streamlit, Plotly, Matplotlib |
| Quality | pytest, Ruff, mypy |
| Configuration | YAML, `uv` |

## Project structure
```
config/config.yaml          # paths, split fractions, hyperparameters, cost matrix
src/fraud_detection/
  config.py                 # typed configuration loader
  data.py                   # SQL views, fact table, and label parser
  features.py               # causal feature engineering
  split.py                  # chronological train/validation/test split
  evaluate.py               # PR-AUC, recall@FPR, precision@k, cost curve
  train.py                  # logistic baseline + LightGBM + feature importance
  unsupervised.py           # Isolation Forest / LOF benchmark
  export_dashboard.py       # builds the dashboard data bundle
scripts/                    # 01_eda → 06_export_dashboard command-line entry points
dashboard/                  # multipage Streamlit application
dashboard_data/             # small, self-contained data bundle for the dashboard
tests/                      # unit tests for the parser, features, and metrics
```

## Running the project
The raw data is expected at `../archive` (see `config/config.yaml`).

```bash
uv sync --extra dashboard --extra dev      # or: pip install -r requirements.txt

uv run python scripts/01_eda.py             # summary and report figures
uv run python scripts/02_build_features.py  # build the feature table
uv run python scripts/03_train.py           # train and evaluate the models
uv run python scripts/04_unsupervised.py    # unsupervised benchmark
uv run python scripts/06_export_dashboard.py# build the dashboard bundle
uv run python scripts/05_dashboard.py       # launch the dashboard

uv run pytest -q && uv run ruff check . && uv run mypy src   # quality checks
```

A fast local smoke test is available by setting `features.sample_rows` in the
configuration file.

## Limitations & future work
- **Synthetic data.** Absolute precision and recall depend on the data generator;
  the methodology, not the exact figure, is the deliverable.
- **Assumed costs.** The `$200` / `$5` cost constants should be replaced with an
  organisation's real fraud-loss figures.
- **Next steps.** Per-transaction explanations (SHAP), a real-time scoring API,
  and drift monitoring on feature distributions over time.
