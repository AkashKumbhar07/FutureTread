import yfinance as yf
import pandas as pd

def download_data(symbol="AAPL", start="2022-01-01", end="2024-01-01"):
    df = yf.download(symbol, start=start, end=end)
    df.to_csv(f"data/{symbol}.csv")
    print("Data saved.")
    return df

if __name__ == "__main__":
    download_data()
