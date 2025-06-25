import yfinance as yf
import pandas as pd
import joblib
from datetime import datetime
from utils.indicators import add_indicators
import os

def get_latest_data(symbol="AAPL", days=100):
    df = yf.download(symbol, period=f"{days}d", interval="1d", group_by="ticker")
    df = df.dropna()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df



def make_prediction(df):
    model = joblib.load("models/rf_model.pkl")
    df = add_indicators(df)

    features = ['rsi', 'ema', 'macd']
    X = df[features].tail(1)  # Only the last/latest row
    pred = model.predict(X)[0]
    return pred, df.iloc[-1]['Close']

def log_signal(symbol, signal, close_price):
    log_path = "logs/live_signals.csv"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = pd.DataFrame([[now, symbol, close_price, "BUY" if signal else "HOLD"]],
                         columns=["Timestamp", "Symbol", "Close", "Signal"])
    
    if not os.path.exists(log_path):
        entry.to_csv(log_path, index=False)
    else:
        entry.to_csv(log_path, mode='a', header=False, index=False)

if __name__ == "__main__":
    symbol = "AAPL"
    df = get_latest_data(symbol)
    signal, close = make_prediction(df)
    log_signal(symbol, signal, close)
    print(f"✅ {symbol} Signal: {'BUY' if signal else 'HOLD'} @ {close}")
