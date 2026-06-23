import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

# --- Load cleaned data ---
df = pd.read_csv("data/processed/monthly_indicators.csv", index_col=0, parse_dates=True)

RECESSION_COL = "recession_indicator"
indicators = [col for col in df.columns if col != RECESSION_COL]

cycles = {}

for col in indicators:
    series = df[col].dropna()
    cycle, trend = hpfilter(series, lamb=1600)
    cycles[col] = cycle

cycle_df = pd.DataFrame(cycles)

# Re-attach NBER column, aligned by index
cycle_df[RECESSION_COL] = df[RECESSION_COL]

# Reorder - recession_col is first
cols = [RECESSION_COL] + [c for c in cycle_df.columns if c != RECESSION_COL]
cycle_df = cycle_df[cols]

cycle_df.to_csv("data/processed/cyclical_components.csv")
print(f"Saved cyclical components: {cycle_df.shape}")
print(cycle_df.head())
print(f"\nNull counts:\n{cycle_df.isnull().sum()}")
