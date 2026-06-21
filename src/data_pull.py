# src/data_pull.py

import pandas as pd
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv
import os

load_dotenv()
fred = Fred(api_key = os.getenv("FRED_API_KEY"))

# --- Series to pull ---
fred_series = {
    "UNRATE" : "unemployment_rate",
    "ICSA"   : "jobless_claims",
    "HOUST"  : "housing_starts",
    "UMCSENT": "consumer_sentiment",
    "T10Y2Y" : "yield_spread",
    "BAA10Y" : "credit_spread",
    "INDPRO" : "industrial_production",
    "USREC"  : "recession_indicator"
}

# --- pull series from FRED ---
start_date = '1980-01-01'
end_date   = '2025-01-01'

raw_data = {}
for code, name in fred_series.items():
    print(f"Pulling {name} ...")
    raw_data[name] = fred.get_series(code, start_date, end_date)

# --- pull S&P 500 from Yahoo Finance ---
print("Pulling S&P 500 ...")
sp500 = yf.download("^GSPC", start=start_date, end=end_date, auto_adjust=True)['Close'].squeeze()
sp500_monthly = sp500.resample('MS').last()
raw_data['sp500'] = sp500_monthly

# --- Combine into one dataframe ---
df_raw = pd.DataFrame(raw_data)
df_raw.index = pd.to_datetime(df_raw.index)

print(f'\n{df_raw.shape}')
print(df_raw.head())
print(df_raw.tail())

print("\n--- Null Counts ---")
print(df_raw.isnull().sum())

print("\n--- Date Ranges ---")
for col in df_raw.columns:
    valid = df_raw[col].dropna()
    print(f"{col}: {valid.index[0].date()} → {valid.index[-1].date()}")