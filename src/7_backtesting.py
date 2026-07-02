# src/7_backtesting.py

import pandas as pd
import numpy as np

THRESHOLD = 0.4

# --- 1a. Load data ---
df = pd.read_csv("data/processed/recession_probabilities.csv", index_col=0, parse_dates=True)
print(f"Loaded: {df.shape}")

prob_col = "recession_prob_multi"
recession_col = "recession_indicator"

# --- 1b. Apply persistence filter ---
# Only trigger alarm if probability stays above threshold for 3 consecutive months
PERSISTENCE = 3
df["signal_raw"] = (df[prob_col] >= THRESHOLD).astype(int)
df["signal"] = (
    df["signal_raw"]
    .rolling(PERSISTENCE)
    .sum()
    .ge(PERSISTENCE)
    .astype(int)
)
print(f"\nSignal months before persistence filter: {df['signal_raw'].sum()}")
print(f"Signal months after persistence filter:  {df['signal'].sum()}")

# --- 2. Identify individual recessions ---
# Each contiguous block of recession_indicator == 1 is one recession
df["recession_id"] = (df[recession_col] != df[recession_col].shift()).cumsum()
recession_ids = df[(df[recession_col] == 1)]["recession_id"].unique()

recessions = []
for rid in recession_ids:
    block = df[df["recession_id"] == rid]
    recessions.append({
        "recession_id": rid,
        "start": block.index[0],
        "end": block.index[-1],
        "duration_months": len(block)
    })
recessions_df = pd.DataFrame(recessions)
print(f"\nRecessions identified: {len(recessions_df)}")
print(recessions_df.to_string(index=False))

# --- 3. Lead time per recession ---
# For each recession, find the first month BEFORE start where prob crossed threshold
lead_times = []

for _, rec in recessions_df.iterrows():
    rec_start = rec["start"]
    # Look at the 24 months before recession start
    window = df[df.index < rec_start].tail(24)
    signals = window[window[prob_col] >= THRESHOLD]
    if len(signals) > 0:
        first_signal = signals.index[0]
        lead = (rec_start.year - first_signal.year) * 12 + (rec_start.month - first_signal.month)
        lead_times.append({
            "recession_start": rec_start,
            "first_signal": first_signal,
            "lead_months": lead,
            "caught": True
        })
    else:
        lead_times.append({
            "recession_start": rec_start,
            "first_signal": None,
            "lead_months": None,
            "caught": False
        })

lead_df = pd.DataFrame(lead_times)
print(f"\n── Lead Times (threshold = {THRESHOLD}) ──")
print(lead_df.to_string(index=False))
print(f"\nAvg lead time (caught recessions): {lead_df['lead_months'].mean():.1f} months")
print(f"Recessions caught: {lead_df['caught'].sum()} / {len(lead_df)}")

# --- 4. False alarm rate ---
# A false alarm is a month where prob >= threshold but no recession
# starts within the next 6 months
df["signal"] = (df["signal"] == 1).astype(int)
df["is_recession"] = df[recession_col].astype(int)

false_alarms = 0
true_alarms = 0

signal_months = df[df["signal"] == 1].index
for date in signal_months:
    # check if a recession starts within 6 months after this signal
    window_end = date + pd.DateOffset(months=6)
    upcoming = df.loc[date:window_end, recession_col]
    if upcoming.sum() > 0:
        true_alarms += 1
    else:
        false_alarms += 1

print(f"\n── False Alarm Analysis ──")
print(f"Total signal months:  {len(signal_months)}")
print(f"True alarms:          {true_alarms}")
print(f"False alarms:         {false_alarms}")
print(f"False alarm rate:     {false_alarms / len(signal_months):.3f}" if len(signal_months) > 0 else "No signals")

# --- 5. Recall by recession ---
print(f"\n── Recall by Recession ──")
for _, rec in recessions_df.iterrows():
    block = df[(df.index >= rec["start"]) & (df.index <= rec["end"])]
    months_flagged = (block[prob_col] >= THRESHOLD).sum()
    recall = months_flagged / len(block)
    print(f"  {rec['start'].strftime('%Y-%m')} to {rec['end'].strftime('%Y-%m')} "
          f"({rec['duration_months']} months) → "
          f"flagged {months_flagged}/{rec['duration_months']} months | recall: {recall:.2f}")

# --- 6. Save backtesting summary ---
lead_df.to_csv("data/processed/backtest_lead_times.csv", index=False)
print(f"\nSaved backtest_lead_times.csv")