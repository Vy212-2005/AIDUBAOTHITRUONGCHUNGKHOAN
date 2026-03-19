import streamlit as st
import os
import joblib
from tensorflow.keras.models import load_model

def tai_tai_san_model(ticker):
    """Tải file .keras và .pkl (scaler) cho mã tương ứng."""
    try:
        m_path = f'model_{ticker.lower()}.keras'
        s_path = f'scaler_{ticker.lower()}.pkl'
        if os.path.exists(m_path) and os.path.exists(s_path):
            m = load_model(m_path)
            s = joblib.load(s_path)
            return m, s
    except Exception as e:
        st.warning(f"Không tìm thấy model cho {ticker}: {e}")
    return None, None

def tinh_bien_dong_ky(df, ngay):
    """Tính % thay đổi giá trong X ngày."""
    try:
        dong_cua = df['Close']
        if len(dong_cua) > ngay:
            dau = dong_cua.iloc[-(ngay + 1)]
            cuoi = dong_cua.iloc[-1]
            return ((cuoi - dau) / dau) * 100
        return 0
    except: return 0
