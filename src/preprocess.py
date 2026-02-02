import pandas as pd

def load_and_clean_data():
    df = pd.read_json("data/raw_data.json")

    time_series = df["Time Series (Daily)"].T
    time_series.columns = ["open", "high", "low", "close", "volume"]

    time_series = time_series.astype(float)
    time_series.index = pd.to_datetime(time_series.index)
    time_series.sort_index(inplace=True)

    return time_series
