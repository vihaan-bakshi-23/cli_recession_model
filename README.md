# US Recession Probability Model

A recession forecasting pipeline built on 8 macroeconomic indicators, using methodology inspired by the OECD's Composite Leading Indicator (CLI) framework — with logistic regression validated against NBER recession dates since 1986.

**[Live Dashboard](https://vihaan-us-recession-model.streamlit.app)**

---

## Overview

This project estimates the probability that the US economy is currently in or entering a recession, using publicly available macroeconomic data. It replicates the spirit of the OECD's CLI methodology (extracting cyclical signal from leading indicators via HP-filter decomposition) while also testing a more direct approach: feeding all indicators straight into a classification model.

**Key finding:** the direct multivariate approach substantially outperforms the traditional composite-CLI approach for this dataset (walk-forward OOS AUC of ~0.84 vs. ~0.67), suggesting that compressing 8 indicators into a single composite score loses predictive information that a multivariate model can otherwise exploit.

## Data

8 indicators, monthly frequency, 1986–present:

| Indicator | Source |
|---|---|
| Unemployment rate | FRED |
| Housing starts | FRED |
| Consumer sentiment | FRED |
| Industrial production | FRED |
| Initial jobless claims | FRED |
| 10Y–3M yield spread | FRED |
| Credit spread (Baa–10Y Treasury) | FRED |
| S&P 500 | Yahoo Finance |

Target variable: NBER recession dates.

## Methodology

1. **Data collection** — pull raw series from FRED API and Yahoo Finance
2. **Cleaning & alignment** — resample to monthly frequency, align to a common timeline
3. **HP filter (λ=1600)** — isolate cyclical components, strip long-run trends
4. **Normalization & lead-lag optimization** — z-score normalize; find each indicator's optimal predictive lag (1–18 months) via absolute correlation against recession periods
5. **Composite index construction** — combine lag-adjusted series into equal-weighted, PCA-weighted, and correlation-weighted CLI scores
6. **Recession probability modeling** — logistic regression on both CLI scores and raw multivariate features
7. **Backtesting** — walk-forward out-of-sample validation, persistence filtering, lead-time analysis
8. **Visualization** — interactive dashboard with NBER recession shading and probability traces

## Results

| Model | In-Sample AUC | Walk-Forward OOS AUC |
|---|---|---|
| Multivariate (all 8 indicators) | 0.953 | **0.838** |
| CLI equal-weighted composite | — | ~0.67 |
| CLI PCA-weighted composite | — | ~0.67 |

- **Persistence filtering** (requiring 3 consecutive months of signal) reduced the false-alarm rate from 0.594 to 0.396 with no loss in recall.
- The model predicts **current-month** recession status (not a future date) — the ~12-month average lead time emerges naturally from the indicators' own lag structure, not from a forward-shifted target.

## What didn't work (and why that's informative)

- **XGBoost**: tested as an alternative to logistic regression; walk-forward OOS AUC of 0.922 vs. logistic regression's 0.934. With only ~460 months of data and 4 recession events, there isn't enough positive-class variety for a more flexible tree-based model to outperform a simpler linear one — a useful reminder that model complexity should match data volume, not just be maximized by default.
- **CLI composite scores**: methodologically faithful to the OECD's approach, but underperform a direct multivariate model on this dataset — compressing 8 indicators into one score loses information a logistic regression can otherwise use directly.

## Live Dashboard

The [Streamlit app](https://YOUR-APP-NAME.streamlit.app) surfaces the two most trustworthy metrics — the multivariate model and the walk-forward OOS estimate — front and center, with the CLI-based scores available for reference. Data refreshes monthly via an automated pipeline (FRED/Yahoo pull → full model re-run → dashboard regeneration), scheduled locally via Windows Task Scheduler.

## Tech Stack

- **Data**: `fredapi`, `yfinance`
- **Modeling**: `scikit-learn`, `xgboost`
- **Visualization**: `plotly`, `streamlit`
- **Automation**: Python `subprocess` orchestration, Windows Task Scheduler

## Project Structure
cli_recession_model/
├── src/
│   ├── 1_data_pull.py
│   ├── 2_preprocessing.py
│   ├── 3_hp_filter.py
│   ├── 4_normalize_leads.py
│   ├── 5_build_cli.py
│   ├── 6_recession_probability.py
│   ├── 7_backtesting.py
│   ├── 7b_walkforward.py
│   ├── 8_dashboard.py
│   ├── 9_xgboost_model.py
│   └── 10_monthly_refresh.py
├── data/processed/
├── outputs/
├── streamlit_app.py
├── run_refresh.bat
└── requirements.txt

## Running Locally

```bash
git clone https://github.com/vihaan-bakshi-23/cli_recession_model.git
cd cli_recession_model
pip install -r requirements.txt
# Add your FRED API key to a .env file: FRED_API_KEY=your_key_here
python src/10_monthly_refresh.py --full   # runs the full pipeline
streamlit run streamlit_app.py            # launches the local dashboard
```

## Limitations

- Small positive-class sample (4 US recessions in the 1986–2024 training window) limits how far model complexity can be pushed before overfitting risk outweighs any gain.
- HP filter is known to exhibit endpoint bias — cyclical estimates near the most recent data points are less stable than those in the middle of the series.
- Data reflects revised (not real-time-as-published) values; performance against real-time vintage data has not yet been tested.

## Author

**Vihaan Bakshi**
[GitHub](https://github.com/YOUR_GITHUB_USERNAME) · [LinkedIn](#)