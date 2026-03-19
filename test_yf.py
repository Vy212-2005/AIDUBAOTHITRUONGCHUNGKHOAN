import yfinance as yf
ticker = "AAPL"
df = yf.download(ticker, period='2y')
print(f"Columns: {df.columns}")
print(f"Shape of df['Close']: {df['Close'].shape if 'Close' in df else 'No Close'}")
