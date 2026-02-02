import requests
import json
import os

API_KEY = "OCXQ78XIZ2FCGO3J"
SYMBOL = "IBM"

def fetch_stock_data():
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=OCXQ78XIZ2FCGO3J'
    response = requests.get(url)
    data = response.json()
    
    os.makedirs("../data", exist_ok=True)
    with open("../data/raw_data.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print("Stock data fetched and saved.")
    return data




