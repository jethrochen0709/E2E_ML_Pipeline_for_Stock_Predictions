# scripts/clean_data.py

import yfinance as yf
import pandas as pd
import os
from time import sleep
from concurrent.futures import ThreadPoolExecutor

# Load tickers
with open("tickers.txt") as f:
    tickers = [t.strip() for t in f if t.strip()]

START, END = "2018-01-01", "2024-12-31"
OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)

def fetch_and_save(ticker):
    try:
        df = yf.download(ticker, start=START, end=END, progress=False)
        if not df.empty:
            df.to_csv(f"{OUT}/{ticker}.csv")
            print(f"✔ {ticker}")
        else:
            print(f"⚠️  No data for {ticker}")
    except Exception as e:
        print(f"❌ {ticker} failed: {e}")

# download in parallel (5 at a time)
with ThreadPoolExecutor(max_workers=5) as exe:
    exe.map(fetch_and_save, tickers)