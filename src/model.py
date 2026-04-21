import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import numpy as np

FEATURES = [
    "daily_return", "ma_5", "ma_20", "ma_50", "ma_200",
    "momentum_5", "momentum_10", "rsi",
    "volume_change", "close_to_ma5", "close_to_ma20", "volatility"
]

# ── PyTorch Model Definition ──────────────────────────────────────────────────
class StockNet(nn.Module):
    def __init__(self, input_size):
        super(StockNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_model(df):
    X = df[FEATURES].values
    y = df["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Last row = today, used to predict next trading day
    last_row = X_scaled[[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    results = {}

    # ── 1. Logistic Regression ────────────────────────────────────────────────
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train, y_train)
    results["Logistic Regression"] = {
        "predictions":      lr_model.predict(X_test),
        "y_test":           y_test,
        "last_prediction":  int(lr_model.predict(last_row)[0])
    }

    # ── 2. XGBoost ────────────────────────────────────────────────────────────
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train, y_train)
    results["XGBoost"] = {
        "predictions":      xgb_model.predict(X_test),
        "y_test":           y_test,
        "last_prediction":  int(xgb_model.predict(last_row)[0])
    }

    # ── 3. PyTorch Neural Network ─────────────────────────────────────────────
    X_train_t  = torch.tensor(X_train,  dtype=torch.float32)
    y_train_t  = torch.tensor(y_train,  dtype=torch.float32).unsqueeze(1)
    X_test_t   = torch.tensor(X_test,   dtype=torch.float32)
    last_row_t = torch.tensor(last_row, dtype=torch.float32)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    nn_model  = StockNet(input_size=len(FEATURES))
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    nn_model.train()
    for epoch in range(100):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(nn_model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    nn_model.eval()
    with torch.no_grad():
        preds        = (nn_model(X_test_t)   >= 0.5).int().numpy().flatten()
        last_pred_nn = (nn_model(last_row_t) >= 0.5).int().item()

    results["PyTorch"] = {
        "predictions":      preds,
        "y_test":           y_test,
        "last_prediction":  last_pred_nn
    }

    return results