# src/6_recession_probability.py  (full file - replace existing)

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- 1. Load CLI composite data ---
df = pd.read_csv("data/processed/cli_composite.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")

y = df["recession_indicator"].values

# --- 2. Helper: fit, evaluate, and print results ---
def evaluate_cli(df, cli_col, y):
    X = df[[cli_col]].values
    model = LogisticRegression(class_weight="balanced", random_state=42)
    model.fit(X, y)

    df[f"recession_prob_{cli_col}"] = model.predict_proba(X)[:, 1]
    probs = df[f"recession_prob_{cli_col}"]

    auc = roc_auc_score(y, probs)
    rec_avg  = probs[y == 1].mean()
    non_avg  = probs[y == 0].mean()

    print(f"\n{'─'*45}")
    print(f"  CLI column: {cli_col}")
    print(f"  Coefficient: {model.coef_[0][0]:.4f}")
    print(f"  AUC: {auc:.4f}")
    print(f"  Avg prob during recessions:     {rec_avg:.3f}")
    print(f"  Avg prob during non-recessions: {non_avg:.3f}")

    for threshold in [0.3, 0.4, 0.5]:
        preds = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
        recall    = tp / (tp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        print(f"  Threshold {threshold} → Recall: {recall:.3f} | Precision: {precision:.3f} | FN: {fn} | FP: {fp}")

    return model

# --- 3. Evaluate both CLI versions ---
evaluate_cli(df, "cli_equal", y)
evaluate_cli(df, "cli_weighted", y)

# --- 4. All-indicators model ---
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

# Load normalized + lagged data (has all 8 columns)
lagged = pd.read_csv("data/processed/normalized_lagged.csv", index_col=0, parse_dates=True)
lagged = lagged.dropna()

# Align index with df (cli_composite may have fewer rows)
common_idx = df.index.intersection(lagged.index)
X_multi = lagged.loc[common_idx, indicator_cols].values
y_multi = lagged.loc[common_idx, "recession_indicator"].values

model_multi = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
model_multi.fit(X_multi, y_multi)

probs_multi = model_multi.predict_proba(X_multi)[:, 1]
auc_multi = roc_auc_score(y_multi, probs_multi)

print(f"\n{'─'*45}")
print(f"  Model: all 8 indicators (multivariate)")
print(f"  AUC: {auc_multi:.4f}")
print(f"  Avg prob during recessions:     {probs_multi[y_multi == 1].mean():.3f}")
print(f"  Avg prob during non-recessions: {probs_multi[y_multi == 0].mean():.3f}")

print("\n  Per-indicator coefficients:")
for col, coef in zip(indicator_cols, model_multi.coef_[0]):
    print(f"    {col:<25} {coef:+.4f}")

for threshold in [0.3, 0.4, 0.5]:
    preds = (probs_multi >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_multi, preds).ravel()
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  Threshold {threshold} → Recall: {recall:.3f} | Precision: {precision:.3f} | FN: {fn} | FP: {fp}")

prob_multi_series = pd.Series(probs_multi, index=lagged.loc[common_idx].index, name="recession_prob_multi")
df = df.join(prob_multi_series, how="left")

# --- 5. Save using whichever performs better ---
# We'll decide after seeing the output — for now save both probability columns
df.to_csv("data/processed/recession_probabilities.csv")
print(f"\nSaved recession_probabilities.csv: {df.shape}")