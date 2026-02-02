def add_features(df):
    df["daily_return"] = df["close"].pct_change()
    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["volatility"] = df["daily_return"].rolling(window=5).std()
    
    # Target: 1 if next day price goes up, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    
    df.dropna(inplace=True)
    return df
