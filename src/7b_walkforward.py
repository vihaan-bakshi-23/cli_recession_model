# src/7b_walkforward.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix

# --- 1. Load data --- 
lagged = pd.read_csv("data/processed/normalized_lagged.csv", index_col=0, parse_dates=True)
lagged = lagged.dropna()

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

X = lagged[indicator_cols]
y = lagged["recession_indicator"]

# --- 2. Define expanding windows --- 
# Each entry is the cutoff year — train on everything before, test on everything after
cutoffs = [2000, 2005, 2010, 2015]

# --- 3. Walk-forward validation --- 
all_preds = []

for cutoff in cutoffs:
    cutoff_date = pd.Timestamp(f"{cutoff}-12-31")

    X_train = X[X.index <= cutoff_date]
    y_train = y[y.index <= cutoff_date]
    X_test  = X[X.index > cutoff_date]
    y_test  = y[y.index > cutoff_date]

    if y_train.sum() < 1:
        print(f"Cutoff {cutoff}: not enough recession months to train — skipping")
        continue

    model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    fold_auc = roc_auc_score(y_test, probs) if y_test.sum() > 0 else None

    print(f"\nCutoff: {cutoff}")
    print(f"  Train: {X_train.index[0].strftime('%Y-%m')} → {X_train.index[-1].strftime('%Y-%m')} ({len(X_train)} months, {int(y_train.sum())} recession months)")
    print(f"  Test:  {X_test.index[0].strftime('%Y-%m')} → {X_test.index[-1].strftime('%Y-%m')} ({len(X_test)} months, {int(y_test.sum())} recession months)")
    print(f"  Out-of-sample AUC: {fold_auc:.4f}" if fold_auc else "  AUC: N/A (no recession months in test set)")

    all_preds.append(pd.Series(probs, index=X_test.index, name="prob"))

# --- 4. Stitch predictions together --- 
# Use only the first prediction for each date (earliest cutoff that covers it)
combined = pd.concat(all_preds)
combined = combined[~combined.index.duplicated(keep="first")]
combined = combined.sort_index()

# Align with actual recession indicator
y_combined = y.loc[combined.index]

overall_auc = roc_auc_score(y_combined, combined)
print(f"\n── Overall Walk-Forward Results ──")
print(f"  Out-of-sample months: {len(combined)}")
print(f"  Recession months in test set: {int(y_combined.sum())}")
print(f"  Overall out-of-sample AUC: {overall_auc:.4f}")
print(f"  (In-sample AUC was: 0.9534)")

# --- 5. Threshold analysis on stitched predictions --- 
THRESHOLD = 0.4
print(f"\n── Threshold Analysis (threshold = {THRESHOLD}) ──")
for threshold in [0.3, 0.4, 0.5]:
    preds = (combined >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_combined, preds).ravel()
    recall    = tp / (tp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    print(f"  Threshold {threshold} → Recall: {recall:.3f} | Precision: {precision:.3f} | FN: {fn} | FP: {fp}")

# --- 6. Save --- 
out = pd.DataFrame({
    "recession_indicator": y_combined,
    "recession_prob_walkforward": combined
})
out.to_csv("data/processed/walkforward_probabilities.csv")
print(f"\nSaved walkforward_probabilities.csv: {out.shape}")