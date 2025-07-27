# scripts/get_sp500_tickers.py

import pandas as pd

# 1. Read the first table on the Wikipedia page
url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
df = pd.read_html(url, header=0)[0]

# 2. Extract the “Symbol” column
tickers = df["Symbol"].tolist()

# 3. Write to tickers.txt in the project root
with open("tickers.txt", "w") as f:
    for t in tickers:
        f.write(f"{t}\n")

print(f"Saved {len(tickers)} tickers to tickers.txt")