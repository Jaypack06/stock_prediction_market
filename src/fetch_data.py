import yfinance as yf
import os
from datetime import datetime, timedelta

SYMBOL = "IBM"

def fetch_stock_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw_stock_data.csv")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    today    = datetime.today().strftime("%Y-%m-%d")
    tomorrow = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        SYMBOL,
        start="2015-01-01",
        end=tomorrow,
        interval="1d"
    )

    # Sort most recent dates to the top
    df = df.sort_index(ascending=False)

    df.to_csv(data_path)
    print(f"Stock data fetched up to {today} and saved to {data_path}")
    return df