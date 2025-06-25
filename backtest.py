import pandas as pd
import joblib
from utils.indicators import add_indicators
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("data/AAPL.csv", index_col=0)
df = add_indicators(df)

# Load trained model
model = joblib.load("models/rf_model.pkl")

# Predict
features = ['rsi', 'ema', 'macd']
df = df.dropna()
df['prediction'] = model.predict(df[features])

# Simulate strategy: Buy if prediction == 1, Sell (or stay in cash) if 0
df['position'] = df['prediction'].shift(1)  # Act on previous day's prediction
df['position'] = df['position'].fillna(0)

# Calculate returns
df['market_return'] = df['Close'].pct_change()
df['strategy_return'] = df['market_return'] * df['position']

# Cumulative returns
df['cumulative_market'] = (1 + df['market_return']).cumprod()
df['cumulative_strategy'] = (1 + df['strategy_return']).cumprod()

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df['cumulative_market'], label='Market Return')
plt.plot(df['cumulative_strategy'], label='Strategy Return')
plt.legend()
plt.title("Backtest: AI Strategy vs Market")
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.grid(True)
plt.show()

# Metrics
total_return = df['cumulative_strategy'].iloc[-1] - 1
win_rate = (df['strategy_return'] > 0).sum() / df['strategy_return'].count()
print(f"\n🧾 Strategy Total Return: {total_return*100:.2f}%")
print(f"✅ Win Rate: {win_rate*100:.2f}%")

# Create signal log where position changes (Buy/Sell signals)
df['signal'] = df['position'].diff()

# Filter only actual Buy/Sell signals
signal_df = df[df['signal'] != 0].copy()

# Label signal type
signal_df['signal_type'] = signal_df['signal'].apply(lambda x: "BUY" if x == 1 else "SELL")

# Keep only useful columns
signal_df = signal_df[['Close', 'position', 'signal_type']]
signal_df.index.name = 'Date'

# Save to CSV
signal_df.to_csv("logs/signals.csv")

print(f"\n📄 Signals exported to logs/signals.csv — {len(signal_df)} entries.")
