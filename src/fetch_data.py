import requests
import json

API_KEY = "YOUR_API_KEY"
SYMBOL = "IBM"
URL = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=OCXQ78XIZ2FCGO3J"

def fetch_stock_data():
    response = requests.get(URL)
    data = response.json()

    with open("data/raw_data.json", "w") as f:
        json.dump(data, f, indent=2)

    return data
