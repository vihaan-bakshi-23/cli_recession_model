# src/5_build_cli.py

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# --- 1. load normalized + lagged data ---
df = pd.read_csv("C:\\Users\\vihaa\\Projects\\cli_recession_model\\data\\processed\\normalized_lagged.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")

# --- 2. drop NaN rows due to lagging ---
df = df.dropna()
print(f"Rows after dropping NaNs: {len(df)}")

# --- 3. sign adjustment ---
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
    "unemployment_rate"      :-1,
    "housing_starts"         :+1,
    "consumer_sentiment"     :+1,
    "industrial_production"  :+1,
    "jobless_claims"         :-1,
    "yield_spread"           :+1,
    "credit_spread"          :-1,
    "sp500"                  :+1,
}

X = df[indicator_cols].copy()
for col, sign in sign_map.items():
    X[col] = X[col]*sign

# --- 4. equal-weighted CLI ---
df['cli_equal'] = X.mean(axis=1)

# --- 5. PCA-weighted CLI ---
pca = PCA(n_components=1)
df['cli_pca'] = pca.fit_transform(X)

print(f"\nPC1 explained variance: {pca.explained_variance_ratio_[0]:.3f}")
loadings = pd.Series(pca.components_[0], index=indicator_cols)
print(f"\nPC1 loadings:")
print(loadings.round(3).to_string())

# --- 6. save to csv ---
out = df[["recession_indicator", "cli_equal", "cli_pca"]]
out.to_csv("C:\\Users\\vihaa\\Projects\\cli_recession_model\\data\\processed\\cli_composite.csv")
print(f"\nSaved cli_composite.csv: {out.shape}")
print(out.tail())