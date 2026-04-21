def add_features(df):
    # Existing features
    df["daily_return"]  = df["close"].pct_change()
    df["ma_5"]          = df["close"].rolling(window=5).mean()
    df["ma_20"]         = df["close"].rolling(window=20).mean()
    df["volatility"]    = df["daily_return"].rolling(window=5).std()

    # New features
    df["ma_50"]         = df["close"].rolling(window=50).mean()
    df["ma_200"]        = df["close"].rolling(window=200).mean()

    # Momentum
    df["momentum_5"]    = df["close"] - df["close"].shift(5)
    df["momentum_10"]   = df["close"] - df["close"].shift(10)

    # RSI
    delta     = df["close"].diff()
    gain      = delta.clip(lower=0).rolling(window=14).mean()
    loss      = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs        = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # Volume change
    df["volume_change"] = df["volume"].pct_change()

    # Price distance from moving averages
    df["close_to_ma5"]  = (df["close"] - df["ma_5"])  / df["ma_5"]
    df["close_to_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]

    # Target: 1 if next day price goes up, else 0
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    return df