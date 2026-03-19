import yfinance as yf
import pandas as pd
import tensorflow as tf
from ta.trend import SMAIndicator
import os

def test_imports():
    print("Testing imports...")
    try:
        import streamlit
        import joblib
        import plotly
        import sklearn
        print("OK: All libraries imported successfully.")
    except ImportError as e:
        print(f"FAIL: Import failed: {e}")
        return False
    return True

def test_yfinance():
    print("Testing yfinance data download...")
    try:
        df = yf.download('AAPL', period='5d')
        if not df.empty:
            print("OK: yfinance is working.")
        else:
            print("FAIL: yfinance returned empty data.")
            return False
    except Exception as e:
        print(f"FAIL: yfinance failed: {e}")
        return False
    return True

def test_tensorflow():
    print("Testing TensorFlow...")
    try:
        ver = tf.__version__
        print(f"OK: TensorFlow version: {ver}")
    except Exception as e:
        print(f"FAIL: TensorFlow failed: {e}")
        return False
    return True

if __name__ == "__main__":
    if test_imports() and test_yfinance() and test_tensorflow():
        print("\nREADY: Environment is ready!")
    else:
        print("\nISSUE: Environment has issues.")
