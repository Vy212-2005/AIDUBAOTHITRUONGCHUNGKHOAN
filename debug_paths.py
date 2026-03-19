import os
import sys
from pathlib import Path

def check_files():
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"File Location: {__file__}")
    
    ticker = "aapl"
    model_path = f"model_{ticker}.keras"
    scaler_path = f"scaler_{ticker}.pkl"
    
    print(f"Checking relative: {model_path} -> {os.path.exists(model_path)}")
    
    abs_dir = Path(__file__).parent
    abs_model = abs_dir / model_path
    print(f"Checking absolute: {abs_model} -> {abs_model.exists()}")
    
    print("Files in directory:")
    for f in os.listdir(abs_dir):
        if f.endswith('.keras') or f.endswith('.pkl'):
            print(f"- {f}")

if __name__ == "__main__":
    check_files()
