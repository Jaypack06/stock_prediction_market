import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import os
import torch
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from preprocess import load_and_clean_data
from features import add_features
from model import train_model, FEATURES

# ── Setup output folder ───────────────────────────────────────────────────────
base_dir   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
charts_dir = os.path.join(base_dir, "results", "charts")
os.makedirs(charts_dir, exist_ok=True)

# ── Load and prepare data ─────────────────────────────────────────────────────
df = load_and_clean_data()
df = add_features(df)

print("Generating charts...")

# ── 1. Closing Price History ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index, df["close"], color="#2196F3", linewidth=1.2, label="Close Price")
ax.plot(df.index, df["ma_20"],  color="#FF9800", linewidth=1,   linestyle="--", label="20-Day MA")
ax.plot(df.index, df["ma_50"],  color="#9C27B0", linewidth=1,   linestyle="--", label="50-Day MA")
ax.plot(df.index, df["ma_200"], color="#F44336", linewidth=1,   linestyle="--", label="200-Day MA")
ax.set_title("IBM Closing Price History with Moving Averages", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "1_closing_price_history.png"), dpi=150)
plt.close()
print("  [1/6] Closing price history saved")

# ── 2. Daily Returns Distribution ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["daily_return"].dropna(), bins=80, color="#2196F3", edgecolor="white", alpha=0.85)
ax.axvline(0, color="#F44336", linewidth=1.5, linestyle="--", label="Zero Return")
ax.axvline(df["daily_return"].mean(), color="#FF9800", linewidth=1.5, linestyle="--", label=f'Mean: {df["daily_return"].mean():.4f}')
ax.set_title("Distribution of Daily Returns", fontsize=14, fontweight="bold")
ax.set_xlabel("Daily Return")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "2_daily_returns_distribution.png"), dpi=150)
plt.close()
print("  [2/6] Daily returns distribution saved")

# ── 3. RSI Over Time ──────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

ax1.plot(df.index, df["close"], color="#2196F3", linewidth=1.2)
ax1.set_title("IBM Closing Price", fontsize=12, fontweight="bold")
ax1.set_ylabel("Price (USD)")
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df["rsi"], color="#9C27B0", linewidth=1)
ax2.axhline(70, color="#F44336", linewidth=1.2, linestyle="--", label="Overbought (70)")
ax2.axhline(30, color="#4CAF50", linewidth=1.2, linestyle="--", label="Oversold (30)")
ax2.fill_between(df.index, df["rsi"], 70, where=(df["rsi"] >= 70), alpha=0.3, color="#F44336")
ax2.fill_between(df.index, df["rsi"], 30, where=(df["rsi"] <= 30), alpha=0.3, color="#4CAF50")
ax2.set_title("Relative Strength Index (RSI)", fontsize=12, fontweight="bold")
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "3_rsi.png"), dpi=150)
plt.close()
print("  [3/6] RSI chart saved")

# ── 4. Volume Over Time ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(df.index, df["volume"], color="#2196F3", alpha=0.6, width=1.5)
ax.plot(df.index, df["volume"].rolling(20).mean(), color="#F44336", linewidth=1.5, label="20-Day Avg Volume")
ax.set_title("IBM Trading Volume Over Time", fontsize=14, fontweight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "4_volume.png"), dpi=150)
plt.close()
print("  [4/6] Volume chart saved")

# ── 5. XGBoost Feature Importance ────────────────────────────────────────────
X = df[FEATURES].values
y = df["target"].values
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

importance = xgb_model.feature_importances_
sorted_idx = np.argsort(importance)
colors     = ["#2196F3" if i % 2 == 0 else "#42A5F5" for i in range(len(FEATURES))]

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    [FEATURES[i] for i in sorted_idx],
    importance[sorted_idx],
    color=[colors[i] for i in sorted_idx],
    edgecolor="white"
)
ax.set_title("XGBoost Feature Importance", fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.grid(True, alpha=0.3, axis="x")
for bar, val in zip(bars, importance[sorted_idx]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "5_xgboost_feature_importance.png"), dpi=150)
plt.close()
print("  [5/6] XGBoost feature importance saved")

# ── 6. Model Accuracy Comparison ─────────────────────────────────────────────
results = train_model(df)

model_names  = list(results.keys())
accuracies   = [accuracy_score(results[m]["y_test"], results[m]["predictions"]) for m in model_names]
bar_colors   = ["#2196F3", "#FF9800", "#4CAF50"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(model_names, accuracies, color=bar_colors, edgecolor="white", width=0.5)
ax.axhline(0.5, color="#F44336", linewidth=1.5, linestyle="--", label="Baseline (50%)")
ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.4, 0.7)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(charts_dir, "6_model_accuracy_comparison.png"), dpi=150)
plt.close()
print("  [6/6] Model accuracy comparison saved")

print(f"\nAll charts saved to: {charts_dir}")