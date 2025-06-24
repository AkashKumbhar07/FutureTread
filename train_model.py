import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

from utils.indicators import add_indicators

# Load data
df = pd.read_csv("data/AAPL.csv", index_col=0)
df = add_indicators(df)

# Target: if tomorrow's close > today's close → 1, else 0
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Features
features = ['rsi', 'ema', 'macd']
X = df[features]
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/rf_model.pkl")
