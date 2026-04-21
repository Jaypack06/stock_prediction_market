import pandas as pd
import os

def load_and_clean_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_stock_data.csv")

    df = pd.read_csv(data_path, header=0, skiprows=[1, 2])

    df.columns = [col.strip().lower() for col in df.columns]

    if "price" in df.columns:
        df = df.rename(columns={"price": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df.dropna(subset=["date"], inplace=True)
    df.set_index("date", inplace=True)

    df = df[["open", "high", "low", "close", "volume"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.dropna(inplace=True)

    # Sort ascending so features/model work correctly
    df = df.sort_index(ascending=True)

    return df