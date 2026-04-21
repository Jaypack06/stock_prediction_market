from fetch_data import fetch_stock_data
from preprocess import load_and_clean_data
from features import add_features
from model import train_model
from evaluate import evaluate_model
from datetime import datetime, timedelta
import os

# Step 1: Fetch data up to today
fetch_stock_data()

# Step 2: Load and preprocess
df = load_and_clean_data()

# Step 3: Feature engineering
df = add_features(df)

# Step 4: Train all models
results = train_model(df)

# Step 5: Evaluate all models
evaluations = evaluate_model(results)

# Step 6: Figure out next trading day (skip weekends)
today = datetime.today()
days_ahead = 1
while (today + timedelta(days=days_ahead)).weekday() >= 5:
    days_ahead += 1
next_trading_day = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

# Step 7: Save results to repo root /results folder
base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename  = os.path.join(results_dir, f"metrics_{timestamp}.txt")

with open(filename, "w", encoding="utf-8") as f:
    f.write(f"Model Evaluation\nTimestamp: {datetime.now()}\n\n")

    for model_name, metrics in evaluations.items():
        f.write(f"-- {model_name} --\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Confusion Matrix:\n{metrics['matrix']}\n\n")
        print(f"{model_name} Accuracy: {metrics['accuracy']:.4f}")

    f.write(f"\n-- Next Trading Day Prediction: {next_trading_day} --\n")
    print(f"\n-- Next Trading Day Prediction: {next_trading_day} --")

    for model_name, data in results.items():
        pred = data.get("last_prediction")
        if pred is not None:
            direction = "UP ^" if pred == 1 else "DOWN v"
            f.write(f"{model_name}: {direction}\n")
            print(f"{model_name}: {direction}")

print(f"\nResults saved to {filename}")