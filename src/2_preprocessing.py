# src/preprocessing.py

import pandas as pd
import numpy as np
import os

# --- Load raw data ---
df_raw = pd.read_csv('data/raw/raw_indicators.csv', index_col=0, parse_dates=True)

# --- Resample each series to monthly frequency ---
monthly = pd.DataFrame()

# Monthly series — first value of each month
for col in ['unemployment_rate', 'housing_starts', 'consumer_sentiment',
            'industrial_production', 'recession_indicator']:
    monthly[col] = df_raw[col].resample('MS').first()

# Weekly series — monthly mean
monthly['jobless_claims'] = df_raw['jobless_claims'].resample('MS').mean()

# Daily series — monthly mean
monthly['yield_spread']  = df_raw['yield_spread'].resample('MS').mean()
monthly['credit_spread'] = df_raw['credit_spread'].resample('MS').mean()

# S&P 500 — last trading day of each month
monthly['sp500'] = df_raw['sp500'].resample('MS').last()

# --- common/overlapping date range ---
monthly = monthly.loc['1986-01-01':'2024-12-01']

# --- remaining NaNs ---
monthly = monthly.ffill(limit=2)
monthly = monthly.dropna()

# --- output ---
print(monthly.shape)
print(monthly.head())
print(monthly.tail())
print("\n--- Null Counts ---")
print(monthly.isnull().sum())

# --- Save to processed ---
os.makedirs('data/processed', exist_ok=True)
monthly.to_csv('data/processed/monthly_indicators.csv')
print("\nSaved to data/processed/monthly_indicators.csv")