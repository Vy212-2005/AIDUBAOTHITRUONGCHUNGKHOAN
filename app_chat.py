import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# --- 1. CẤU HÌNH API & MODEL ---
API_KEY = "AIzaSyCi0PkcrE5rvpU1DHlkw91JlaqyhbnELOo" 
genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# --- 2. TẢI TÀI NGUYÊN (CACHE) ---
@st.cache_resource
def load_assets():
    try:
        m = load_model('stock_model_aapl.h5')
        s = joblib.load('scaler_aapl.pkl')
        return m, s
    except: return None, None

model, scaler = load_assets()

# --- 3. GIAO DIỆN CHÍNH ---
st.set_page_config(page_title="AAPL AI Dashboard", layout="wide")
st.title("AI Dự Báo Thị Trường Chứng Khoán")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Hỏi AI về cổ phiếu AAPL..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Lấy dữ liệu 100 ngày gần nhất
            df = yf.download('AAPL', period='100d')
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # --- TÍNH TOÁN THÔNG SỐ OHLC ---
            o_p, h_p, l_p, c_p = df['Open'].iloc[-1], df['High'].iloc[-1], df['Low'].iloc[-1], df['Close'].iloc[-1]
            prev_c = df['Close'].iloc[-2]
            change = ((c_p - prev_c) / prev_c) * 100
            ma20 = df['Close'].rolling(20).mean().iloc[-1]

            # Dự báo AI (LSTM)
            pred_p = 0
            if model and scaler:
                last_60 = df['Close'].tail(60).values.reshape(-1, 1)
                scaled = scaler.transform(last_60)
                pred = model.predict(np.reshape(scaled, (1, 60, 1)), verbose=0)
                pred_p = float(scaler.inverse_transform(pred)[0][0])

            # --- HIỂN THỊ DASHBOARD (CHỮ TRẮNG) ---
            trend_color = "#26a69a" if c_p > ma20 else "#ef5350"
            change_color = "#26a69a" if change >= 0 else "#ef5350"

            st.markdown(f"""
                <div style="background-color: #1e1e1e; padding: 20px; border-radius: 10px; border: 1px solid #333; margin-bottom: 20px;">
                    <div style="display: flex; justify-content: space-between; border-bottom: 1px solid #444; padding-bottom: 15px; margin-bottom: 15px;">
                        <div style="text-align: center;"><p style="margin:0; color: #aaa; font-size: 14px;">Mở cửa</p><p style="margin:0; color: white; font-size: 18px; font-weight: bold;">${o_p:.2f}</p></div>
                        <div style="text-align: center;"><p style="margin:0; color: #aaa; font-size: 14px;">Cao nhất</p><p style="margin:0; color: white; font-size: 18px; font-weight: bold;">${h_p:.2f}</p></div>
                        <div style="text-align: center;"><p style="margin:0; color: #aaa; font-size: 14px;">Thấp nhất</p><p style="margin:0; color: white; font-size: 18px; font-weight: bold;">${l_p:.2f}</p></div>
                        <div style="text-align: center;"><p style="margin:0; color: #aaa; font-size: 14px;">Đóng cửa</p><p style="margin:0; color: white; font-size: 18px; font-weight: bold;">${c_p:.2f} <span style="color:{change_color};">({change:+.2f}%)</span></p></div>
                    </div>
                    <p style="margin:0; color: white;">📈 <b>Xu hướng (MA20):</b> <span style="color:{trend_color}; font-weight:bold;">{ 'TĂNG' if c_p > ma20 else 'GIẢM' }</span></p>
                    <p style="margin:5px 0 0 0; color: white;">🤖 <b>Dự báo AI ngày mai:</b> <span style="color:white; font-weight:bold;">${pred_p:.2f}</span></p>
                </div>
            """, unsafe_allow_html=True)

            # --- VẼ BIỂU ĐỒ NẾN + VOLUME ---
            df_plot = df.tail(60)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
            # Nến & MA20
            fig.add_trace(go.Candlestick(x=df_plot.index, open=df_plot['Open'], high=df_plot['High'], low=df_plot['Low'], close=df_plot['Close'], name='Giá'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'].rolling(20).mean(), name='MA20', line=dict(color='orange')), row=1, col=1)
            # Volume
            vol_colors = ['#26a69a' if c >= o else '#ef5350' for o, c in zip(df_plot['Open'], df_plot['Close'])]
            fig.add_trace(go.Bar(x=df_plot.index, y=df_plot['Volume'], marker_color=vol_colors, name='Volume'), row=2, col=1)
            
            fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

            # --- GỌI AI TƯ VẤN (RETRY LOGIC) ---
            with st.spinner("Đang kết nối chuyên gia AI..."):
                final_res = ""
                for i in range(3): 
                    try:
                        res = gemini_model.generate_content(f"AAPL: Open {o_p:.2f}, Close {c_p:.2f}, MA20 {ma20:.2f}, Pred {pred_p:.2f}. Q: {prompt}")
                        final_res = res.text
                        break
                    except Exception as e:
                        if "429" in str(e): time.sleep(15); continue
                        else: final_res = f"Lỗi AI: {e}"; break
                
                st.info(final_res if final_res else "⚠️ Hệ thống bận, hãy thử lại.")
                st.session_state.messages.append({"role": "assistant", "content": final_res})

        except Exception as e:

            st.error(f"Lỗi: {e}")
