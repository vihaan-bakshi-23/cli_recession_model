# src/5_build_cli.py  (full file - replace existing)

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# --- 1. Load normalized + lagged data ---
df = pd.read_csv("data/processed/normalized_lagged.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")

# --- 2. Drop NaN rows introduced by lagging ---
df = df.dropna()
print(f"Rows after dropping NaNs: {len(df)}")

# --- 3. Sign adjustments ---
indicator_cols = [
    "unemployment_rate",
    "housing_starts",
    "consumer_sentiment",
    "industrial_production",
    "jobless_claims",
    "yield_spread",
    "credit_spread",
    "sp500",
]

sign_map = {
    "unemployment_rate":     -1,
    "housing_starts":        +1,
    "consumer_sentiment":    +1,
    "industrial_production": +1,
    "jobless_claims":        -1,
    "yield_spread":          +1,
    "credit_spread":         -1,
    "sp500":                 +1,
}

X = df[indicator_cols].copy()
for col, sign in sign_map.items():
    X[col] = X[col] * sign

# --- 4. Equal-weighted CLI ---
df["cli_equal"] = X.mean(axis=1)

# --- 5. PCA-weighted CLI (PC1 scores) ---
pca = PCA(n_components=1)
df["cli_pca"] = pca.fit_transform(X)

print(f"\nPC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
loadings = pd.Series(pca.components_[0], index=indicator_cols)
print("\nPC1 loadings:")
print(loadings.round(3).to_string())

# --- 6. Correlation-weighted CLI ---
lag_df = pd.read_csv("data/processed/lag_correlations.csv").set_index("indicator")

weights = lag_df.loc[indicator_cols, "abs_correlation"]
weights = weights / weights.sum()

print("\nCorrelation weights (normalized):")
print(weights.round(3).to_string())

df["cli_weighted"] = X[indicator_cols].multiply(weights, axis=1).sum(axis=1)

# --- 7. Save output ---
out = df[["recession_indicator", "cli_equal", "cli_pca", "cli_weighted"]]
out.to_csv("data/processed/cli_composite.csv")
print(f"\nSaved cli_composite.csv: {out.shape}")
print(out.tail())