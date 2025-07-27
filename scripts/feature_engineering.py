# scripts/feature_engineering.py

import pandas as pd
import os

RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

def add_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(window=5).mean()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["Volatility"] = df["Return"].rolling(window=5).std()
    df["Direction"] = (df["Return"] > 0).astype(int)
    return df.dropna()

for filename in os.listdir(RAW_PATH):
    if filename.endswith(".csv"):
        ticker = filename.replace(".csv", "")
        print(f"Processing {ticker}...")

        df = pd.read_csv(os.path.join(RAW_PATH, filename), index_col=0, parse_dates=True)

        # Convert only columns that exist
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'Close' not in df.columns:
            print(f"❌ Skipping {ticker}: no 'Close' column found.\n")
            continue

        df = df.dropna(subset=['Close'])

        try:
            df_feat = add_features(df)
            df_feat.to_csv(os.path.join(PROCESSED_PATH, f"{ticker}_features.csv"))
            print(f"✅ Saved: {ticker}_features.csv\n")
        except Exception as e:
            print(f"❌ Failed to process {ticker}: {e}\n")