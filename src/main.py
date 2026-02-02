from fetch_data import fetch_stock_data
from preprocess import load_and_clean_data
from features import add_features
from model import train_model
from evaluate import evaluate_model
from datetime import datetime
import os

# Step 1: Fetch data
fetch_stock_data()

# Step 2: Load and preprocess
df = load_and_clean_data()

# Step 3: Feature engineering
df = add_features(df)

# Step 4: Train model
model, X_test, y_test = train_model(df)

# Step 5: Evaluate
accuracy, matrix = evaluate_model(model, X_test, y_test)

# Step 6: Save results
os.makedirs("./results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./results/metrics_{timestamp}.txt"

with open(filename, "w") as f:
    f.write(f"Model Evaluation\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{matrix}\n")

print("Pipeline complete!")
print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{matrix}")
print(f"Results saved to {filename}")
