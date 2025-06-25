import pandas as pd
import ta

def add_indicators(df):
    # Ensure numeric types
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    # Drop rows with any NaNs
    df.dropna(inplace=True)

    # Technical indicators
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    
    # Drop NaNs again after indicators
    df.dropna(inplace=True)
    return df
