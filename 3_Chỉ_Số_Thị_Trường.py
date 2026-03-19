import streamlit as st
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator

def ve_the_hieu_suat(col, nhan, gia_tri):
    """Vẽ thẻ biến động phần trăm với phong cách hiện đại."""
    mau = "#10b981" if gia_tri >= 0 else "#ef4444"
    bg_gradient = "linear-gradient(135deg, rgba(16,185,129,0.1), rgba(16,185,129,0.02))" if gia_tri >= 0 else "linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.02))"
    icon = "📈" if gia_tri >= 0 else "📉"
    col.markdown(f"""
        <div style="background: {bg_gradient}; border: 1px solid {mau}44; border-radius: 16px; padding: 20px; text-align: center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="color: #94a3b8; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px;">{nhan}</div>
            <div style="color: {mau}; font-size: 1.8rem; font-weight: 900; filter: drop-shadow(0 0 8px {mau}44);">{icon} {gia_tri:+.2f}%</div>
        </div>
    """, unsafe_allow_html=True)

def ve_the_chi_so(col, nhan, gia_tri, delta=None, delta_text=""):
    """Vẽ thẻ chỉ số tài chính cơ bản."""
    delta_class = "metric-delta-up" if (delta and delta >= 0) else "metric-delta-down"
    delta_val = f"{delta:+.2f}%" if delta is not None else delta_text
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{nhan}</div>
            <div class="metric-value">{gia_tri}</div>
            <div class="{delta_class}">{delta_val}</div>
        </div>
    """, unsafe_allow_html=True)

def tinh_toan_chi_bao(df):
    """Tính toán các chỉ số kỹ thuật từ dataframe."""
    df = df.copy()
    df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
    df['SMA100'] = SMAIndicator(close=df['Close'], window=100).sma_indicator()
    df['SMA200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
    df['EMA20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    return df
