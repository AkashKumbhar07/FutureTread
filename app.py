import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import plotly.graph_objects as go
from utils.indicators import add_indicators
from datetime import datetime
from utils.telegram import send_telegram_message


# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load("models/rf_model.pkl")

# === Get stock data ===
def get_data(symbol="AAPL", start="2023-01-01", end=None):
    df = yf.download(symbol, start=start, end=end, interval="1d", group_by="ticker")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(1)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

# === Predict using model ===
def predict(df, model):
    df = add_indicators(df)
    df.dropna(inplace=True)
    X = df[['rsi', 'ema', 'macd']]
    df['Signal'] = model.predict(X)
    df['Buy'] = df['Signal'].diff() == 1
    df['Sell'] = df['Signal'].diff() == -1
    return df

# === Backtest stats ===
def calculate_stats(df):
    df['Returns'] = df['Close'].pct_change()
    df['Strategy'] = df['Signal'].shift(1) * df['Returns']
    df.dropna(inplace=True)
    total_return = (df['Strategy'] + 1).prod() - 1
    win_rate = (df['Strategy'] > 0).sum() / len(df)
    return round(total_return * 100, 2), round(win_rate * 100, 2)

# === Streamlit UI ===
st.set_page_config(page_title="FutureTread AI", layout="centered")
st.title("📈 FutureTread: AI Trading Signal")

# --- Inputs ---
ticker_list = ["AAPL", "GOOGL", "TSLA", "MSFT", "BTC-USD", "ETH-USD"]
symbol = st.selectbox("Select Ticker", ticker_list)

col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start Date", pd.to_datetime("2023-01-01"))
with col2:
    end_date = st.date_input("End Date", datetime.now().date())

st.markdown("### 🔔 Telegram Alerts (Optional)")
telegram_token = st.text_input("Telegram Bot Token")
telegram_chat_id = st.text_input("Telegram Chat ID")


# --- Predict button ---
if st.button("🔮 Predict Signal"):
    with st.spinner("Running predictions..."):
        model = load_model()
        df = get_data(symbol, str(start_date), str(end_date))
        result_df = predict(df, model)

        # Latest prediction
        latest = result_df.iloc[-1]
        signal = "BUY 🟢" if latest['Signal'] == 1 else "HOLD ⚪"

        st.subheader(f"📊 Prediction for {symbol}")
        st.metric("Latest Signal", signal)
        st.metric("Latest Price", f"${latest['Close']:.2f}")
        if signal.startswith("BUY") and telegram_token and telegram_chat_id:
            sent = send_telegram_message(
                telegram_token,
                telegram_chat_id,
                f"🚀 {symbol} Signal: {signal} @ ${latest['Close']:.2f}"
            )
            if sent:
                st.success("✅ Telegram alert sent!")
            else:
                st.warning("⚠️ Failed to send Telegram alert.")

        # --- Backtest stats ---
        total_return, win_rate = calculate_stats(result_df)
        col1, col2 = st.columns(2)
        col1.metric("📈 Total Return", f"{total_return}%")
        col2.metric("🎯 Win Rate", f"{win_rate}%")

        # --- Plot chart ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result_df.index, y=result_df['Close'], name='Close Price'))

        # Plot Buy markers
        fig.add_trace(go.Scatter(
            x=result_df[result_df['Buy']].index,
            y=result_df[result_df['Buy']]['Close'],
            mode='markers',
            marker_symbol='triangle-up',
            marker_color='green',
            marker_size=10,
            name='BUY'
        ))

        # Plot Sell markers
        fig.add_trace(go.Scatter(
            x=result_df[result_df['Sell']].index,
            y=result_df[result_df['Sell']]['Close'],
            mode='markers',
            marker_symbol='triangle-down',
            marker_color='red',
            marker_size=10,
            name='SELL'
        ))

        fig.update_layout(title=f"{symbol} Buy/Sell Signals", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # --- Data view & export ---
        with st.expander("📄 View Signal Data"):
            st.dataframe(result_df.tail(10))

        st.download_button("⬇️ Download All Signals", result_df.to_csv(index=True), file_name=f"{symbol}_signals.csv")
        
