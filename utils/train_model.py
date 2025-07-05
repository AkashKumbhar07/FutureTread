# utils/train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.indicators import add_indicators


def retrain_model(csv_path="data/AAPL.csv", model_dir="models"):
    df = pd.read_csv(csv_path, index_col=0)
    df = add_indicators(df)
    df.dropna(inplace=True)

    # df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df['target'] = 0  # HOLD by default
    df.loc[df['Close'].shift(-1) > df['Close'], 'target'] = 1  # BUY
    df.loc[df['Close'].shift(-1) < df['Close'], 'target'] = -1  # SELL

    features = ['rsi', 'ema', 'macd']
    X = df[features]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # === Train Random Forest ===
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, f"{model_dir}/rf_model.pkl")

    # === Train Logistic Regression ===
    log_model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    log_model.fit(X_train, y_train)
    joblib.dump(log_model, f"{model_dir}/log_model.pkl")

    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report

if __name__ == "__main__":
    report = retrain_model()
    print("✅ Models trained and saved.")
    print(report)
