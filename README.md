# 📈 Stock Market ML Pipeline

An end-to-end machine learning pipeline for predicting stock price direction across all S&P 500 tickers.  
Built in **Python** with **Scikit-Learn** and **PyTorch** for modeling, and **R Shiny** for interactive visualization.

---

## 🧠 Features

- ✅ Modular ETL pipeline (`clean_data.py`, `feature_engineering.py`)
- ✅ Feature engineering: moving averages, returns, volatility
- ✅ Random Forest and PyTorch neural network with time-series validation
- ✅ Supports 500+ tickers (from `tickers.txt`)
- ✅ Predictions exported to CSVs
- ✅ Interactive R Shiny app for real-time model insights

---

## 📂 Project Structure <br/>

stock-market-ml/ <br/>
├── data/<br/>
│   ├── raw/                # Raw downloaded stock data<br/>
│   └── processed/          # Feature-engineered data<br/>
├── output/                 # Model predictions and saved models<br/>
├── scripts/<br/>
│   ├── clean_data.py <br/>
│   ├── feature_engineering.py <br/>
│   ├── train_model.py <br/>
│   └── get_sp500_tickers.py <br/>
├── R/ <br/>
│   └── app.R               # Shiny dashboard <br/>
└── tickers.txt             # Ticker list (S&P 500) <br/>

---

## 🚀 Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/your_username/stock-market-ml.git
cd stock-market-ml
pip install yfinance pandas numpy scikit-learn matplotlib torch

2. Generate tickers and download data

python scripts/get_sp500_tickers.py
python scripts/clean_data.py
python scripts/feature_engineering.py

3. Train models

python scripts/train_model.py

4. Launch Shiny App (in R)

setwd("~/Documents/stock-market-ml")
shiny::runApp("R/app.R")


⸻

📊 Sample Output
	•	Directional accuracy: 55–60% (PyTorch)
	•	Real-time predictions shown in R Shiny dashboard
	•	Supports model, ticker, and date selection

⸻

🛠️ Future Improvements
	•	Add backtesting + P&L simulation
	•	Integrate SHAP for feature explainability
	•	Deploy app via shinyapps.io
