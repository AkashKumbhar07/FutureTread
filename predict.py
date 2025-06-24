import pandas as pd
import joblib
from utils.indicators import add_indicators

df = pd.read_csv("data/AAPL.csv", index_col=0)
df = add_indicators(df)

features = ['rsi', 'ema', 'macd']
X = df[features]

model = joblib.load("models/rf_model.pkl")
df['prediction'] = model.predict(X)

print(df[['Close', 'prediction']].tail(10))
