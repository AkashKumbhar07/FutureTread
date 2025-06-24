import pandas as pd
import ta

def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ema'] = ta.trend.EMAIndicator(df['Close'], window=14).ema_indicator()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    df = df.dropna()
    return df
