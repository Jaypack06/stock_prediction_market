import os
from fetch_data import fetch_stock_data
from preprocess import load_and_clean_data
from features import add_features
from model import train_model
from evaluate import evaluate_model

fetch_stock_data()
df = load_and_clean_data()
df = add_features(df)

model, X_test, y_test = train_model(df)
accuracy, matrix = evaluate_model(model, X_test, y_test)

os.makedirs("../results", exist_ok=True)

with open("../results/metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Confusion Matrix:\n{matrix}")
    
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", matrix)