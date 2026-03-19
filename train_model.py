import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import joblib
import os
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def create_features(df):
    """Add technical indicators as features."""
    df = df.copy()
    # SMA 20 and 50
    df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    # EMA 20
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    # RSI
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    
    # Drop rows with NaN values resulting from indicators
    df.dropna(inplace=True)
    return df

def prepare_data(df, feature_cols, target_col='Close', n_steps=60):
    """Scale data and create sequences for LSTM."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    X, y = [], []
    for i in range(n_steps, len(scaled_data)):
        X.append(scaled_data[i-n_steps:i])
        # We target the 'Close' price which is at a specific index in feature_cols
        target_idx = feature_cols.index(target_col)
        y.append(scaled_data[i, target_idx])
        
    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    """Build and compile LSTM model."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_save(ticker, period='5y'):
    """Main pipeline to download, train, and save model/scaler."""
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if df.empty:
        print(f"No data found for {ticker}")
        return
    
    df = create_features(df)
    feature_cols = ['Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'Volume']
    
    n_steps = 60
    X, y, scaler = prepare_data(df, feature_cols, n_steps=n_steps)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]
    
    print(f"Training model for {ticker}...")
    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test), verbose=1)
    
    # Save assets
    model_path = f'model_{ticker.lower()}.keras'
    scaler_path = f'scaler_{ticker.lower()}.pkl'
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    train_and_save(ticker)
