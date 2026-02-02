from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train_model(df):
    X = df[["daily_return", "ma_5", "ma_20", "volatility"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test
