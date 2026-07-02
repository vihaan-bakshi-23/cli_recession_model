# src/normalize_leads.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- load cyclical components csv ---
df = pd.read_csv("C:\\Users\\vihaa\\Projects\\cli_recession_model\\data\\processed\\cyclical_components.csv", index_col=0, parse_dates=True)

RECESSION_COL = 'recession_indicator'
indicators = [col for col in df.columns if col!=RECESSION_COL]

# --- 1. normalise ---
scaler = StandardScaler()
normalised = pd.DataFrame(
    scaler.fit_transform(df[indicators]), 
    index=df.index, 
    columns=indicators
    )

# --- 2. optimise lead-lag ---
MAX_LAG = 18
optimal_lags = {}
best_corrs = {}

for col in indicators:
    best_lag = 0
    best_corr = 0
    for lag in range(1, MAX_LAG+1):
        shifted = normalised[col].shift(lag)
        corr = shifted.corr(df[RECESSION_COL])
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    optimal_lags[col] = best_lag
    best_corrs[col] = abs(best_corr)
    print(f"{col}: optimal lag = {best_lag} months (corr = {best_corr:.3f})")

# --- 3. optimal lags ---
lagged = pd.DataFrame(index=df.index)
for col in indicators:
    lagged[col] = normalised[col].shift(optimal_lags[col])

lagged[RECESSION_COL] = df[RECESSION_COL]

cols = [RECESSION_COL] + indicators
lagged = lagged[cols]

lagged.to_csv("data/processed/normalized_lagged.csv")
print(f"\nSaved normalized + lagged data: {lagged.shape}")
print(f"\nNull counts:\n{lagged.isnull().sum()}")

# --- 4. save info on lag + correlations --- 
lag_df = pd.DataFrame([
    {"indicator": col, "optimal_lag": optimal_lags[col], "abs_correlation": best_corrs[col]}
    for col in indicators
])
lag_df.to_csv("data/processed/lag_correlations.csv", index=False)
print("\nSaved lag_correlations.csv")