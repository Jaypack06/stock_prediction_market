import pandas as pd
import json

def load_and_clean_data():
    with open("../data/raw_data.json") as f:
        data = json.load(f)
    
    ts = data["Time Series (Daily)"]
    
    df = pd.DataFrame.from_dict(ts, orient='index')
    df = df[["1. open", "2. high", "3. low", "4. close", "5. volume"]]
    df.columns = ["open", "high", "low", "close", "volume"]
    
    # Convert to numeric safely
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    
    return df
