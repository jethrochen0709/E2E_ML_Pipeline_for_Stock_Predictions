# scripts/train_model.py

import os
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

RAW_DIR = "data/processed"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def train_and_save_for_ticker(fn: str) -> None:
    ticker = fn.replace("_features.csv", "")
    print(f"\n▶ Processing {ticker}")

    # Load features
    df = pd.read_csv(os.path.join(RAW_DIR, fn))
    features = ["MA5", "MA10", "Volatility"]
    target = "Direction"

    # Drop any rows with missing data
    df = df.dropna(subset=features + [target])

    X = df[features].values
    y = df[target].values

    # Time-series split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False
    )

    # --- 1) Scikit-Learn Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    sk_preds = rf.predict(X_test)
    sk_acc = accuracy_score(y_test, sk_preds)
    print(f"RandomForest accuracy for {ticker}: {sk_acc:.4f}")

    # Save predictions
    df_rf = df.iloc[-len(sk_preds) :].copy()
    df_rf["Prediction"] = sk_preds
    rf_out = f"predictions_sklearn_{ticker}.csv"
    df_rf.to_csv(os.path.join(OUT_DIR, rf_out), index=False)

    # Persist the model
    joblib.dump(rf, os.path.join(OUT_DIR, f"rf_model_{ticker}.pkl"))

    # --- 2) PyTorch Feed-Forward Net ---
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = Net(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).squeeze().numpy()
        pt_preds = (probs > 0.5).astype(int)

    pt_acc = accuracy_score(y_test, pt_preds)
    print(f"PyTorch NN accuracy for {ticker}: {pt_acc:.4f}")

    df_pt = df.iloc[-len(pt_preds) :].copy()
    df_pt["Prediction"] = pt_preds
    pt_out = f"predictions_pytorch_{ticker}.csv"
    df_pt.to_csv(os.path.join(OUT_DIR, pt_out), index=False)


if __name__ == "__main__":
    for fname in os.listdir(RAW_DIR):
        if fname.endswith("_features.csv"):
            train_and_save_for_ticker(fname)

    print("\n🎉 All tickers processed!")