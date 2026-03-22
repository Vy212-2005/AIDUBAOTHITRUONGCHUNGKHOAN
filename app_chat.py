import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import importlib
import os
import time
from dotenv import load_dotenv
from PIL import Image

# Load environment variables with explicit path
from pathlib import Path
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)

# Nhập các module chuyên ngành Chứng khoán 1-5
PhanTichKT = importlib.import_module("1_Phân_Tích_Kỹ_Thuật")
DuLieuLS = importlib.import_module("2_Dữ_Liệu_Lịch_Sử")
ChiSoTT = importlib.import_module("3_Chỉ_Số_Thị_Trường")
TinTucSK = importlib.import_module("4_Tin_Tức_Sự_Kiện")
MoHinhDB = importlib.import_module("5_Mô_Hình_Dự_Báo")
ThongKe = importlib.import_module("6_Thống_Kê_Lịch_Sử")

from hang_so import CSS_CHUNG_KHOAN

# --- 1. SETUP ---
st.set_page_config(page_title="AI Dự Báo Chứng Khoán", layout="wide")
st.markdown(CSS_CHUNG_KHOAN, unsafe_allow_html=True)

if "ticker" not in st.session_state: st.session_state.ticker = "AAPL"
if "chat" not in st.session_state: st.session_state.chat = []
ai_model = None

# --- 2. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #4facfe;'>📊 Stock AI</h1>", unsafe_allow_html=True)
    
    st.subheader("📂 Menu Điều hướng")
    menu = st.radio(
        "Chọn tính năng:",
        ["📈 Phân Tích Kỹ Thuật", "🕒 Lịch Sử & Hiệu Suất", "🎯 Chỉ Số Thị Trường", "📰 Tin Tức & Sự Kiện", "📊 Thống Kê & Lịch Sử"]
    )
    
    st.divider()
    st.subheader("⚙️ Cài đặt")
    
    # Gợi ý mã phổ biến
    goi_y = {
        "--- Chọn nhanh ---": None,
        "Cổ phiếu VN": ["HPG.VN", "VNM.VN", "VCB.VN", "FPT.VN", "VIC.VN"],
        "Cổ phiếu US": ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL"],
        "Tiền tệ & Vàng": ["BTC-USD", "ETH-USD", "VND=X", "XAU=F"]
    }
    
    selected_group = st.selectbox("Gợi ý nhanh:", list(goi_y.keys()))
    if selected_group and goi_y[selected_group]:
        # Cập nhật ngay lập tức khi người dùng chọn mã mới
        selected_ticker = st.selectbox("Chọn mã:", goi_y[selected_group], index=None, placeholder="Chọn một mã...")
        if selected_ticker and selected_ticker != st.session_state.ticker:
            st.toast(f"🚀 Đang tải dữ liệu cho {selected_ticker}...", icon="📈")
            st.session_state.ticker = selected_ticker
            time.sleep(0.5)
            st.rerun()

    st.divider()
    ticker_in = st.text_input("Nhập mã thủ công (ví dụ: AAPL, HPG.VN...)", value=st.session_state.ticker).upper()
    if ticker_in != st.session_state.ticker:
        st.toast(f"🔍 Đang tìm kiếm mã {ticker_in}...", icon="🔎")
        st.session_state.ticker = ticker_in
        time.sleep(0.5)
        st.rerun()
    ticker = st.session_state.ticker
    
    # Tự động lấy API key từ biến môi trường hoặc Streamlit Secrets
    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    secrets_key = ""
    try:
        if "GEMINI_KEYS" in st.secrets:
            keys_secret = st.secrets["GEMINI_KEYS"]
            if isinstance(keys_secret, list):
                secrets_key = ",".join(keys_secret)
            else:
                secrets_key = str(keys_secret)
        elif "GEMINI_API_KEY" in st.secrets:
            secrets_key = str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    default_key = secrets_key if secrets_key else env_key
    
    key = st.text_input("Gemini API Key", value=default_key, type="password")
    
    if not key:
        st.caption("⚠️ Chưa tìm thấy key trong secrets, file .env hoặc sidebar")
    elif secrets_key and key == secrets_key:
        st.sidebar.caption("✅ Đã tải key từ Streamlit Secrets")
    elif env_key and key == env_key:
        st.sidebar.caption("✅ Đã tải key từ .env")
    
    if key: 
        ai_model = MoHinhDB.cau_hinh_gemini(key)

# --- 3. MAIN APP ---
main_container = st.container()
with main_container:
    st.markdown(f"<div class='fade-in'>", unsafe_allow_html=True)
    st.markdown(f"<h1 class='main-title'>Dự Báo Thị Trường: {ticker}</h1>", unsafe_allow_html=True)

try:
    df = yf.download(ticker, period='2y')
    if df.empty: st.error("Không có dữ liệu"); st.stop()
    
    # Làm phẳng MultiIndex nếu yfinance trả về nhiều cấp
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ghi log lượt truy cập
    ThongKe.ghi_log_truy_cap(ticker, menu)

    df = ChiSoTT.tinh_toan_chi_bao(df)
    # Tính toán các chỉ số chung cho toàn bộ ứng dụng (bao gồm cả AI)
    ma_vals = {
        "SMA20": df['SMA20'].iloc[-1],
        "SMA50": df['SMA50'].iloc[-1],
        "SMA100": df['SMA100'].iloc[-1],
        "SMA200": df['SMA200'].iloc[-1],
        "EMA20": df['EMA20'].iloc[-1]
    }
    cp = df['Close'].iloc[-1]; op = df['Open'].iloc[-1]
    hp = df['High'].iloc[-1]; lp = df['Low'].iloc[-1]
    vol = df['Volume'].iloc[-1]; avg_v = df['Volume'].tail(30).mean()
    ma20 = ma_vals['SMA20']; rsi = df['RSI'].iloc[-1]
    bien_dong = ((cp - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100

    if menu == "📈 Phân Tích Kỹ Thuật":
        st.plotly_chart(PhanTichKT.tao_bieu_do_ky_thuat(df, ticker, cp, op, hp, lp, vol, avg_v, ma_vals), use_container_width=True)

    elif menu == "🕒 Lịch Sử & Hiệu Suất":
        st.subheader(f"📈 Hiệu suất lịch sử của {ticker}")
        c1, c2, c3, c4 = st.columns(4)
        ChiSoTT.ve_the_hieu_suat(c1, "1 Ngày", bien_dong)
        ChiSoTT.ve_the_hieu_suat(c2, "1 Tuần", DuLieuLS.tinh_bien_dong_ky(df, 5))
        ChiSoTT.ve_the_hieu_suat(c3, "1 Tháng", DuLieuLS.tinh_bien_dong_ky(df, 21))
        ChiSoTT.ve_the_hieu_suat(c4, "1 Năm", DuLieuLS.tinh_bien_dong_ky(df, 252))
        
        st.divider()
        st.subheader("🔥 Xu Hướng Thị Trường Toàn Cầu")
        ThongKe.hien_thi_heatmap_the_gioi()

    elif menu == "🎯 Chỉ Số Thị Trường":
        m1, m2, m3, m4 = st.columns(4)
        sym = "$" # Simplified
        # Gọi mô hình Keras để dự báo
        gia_du_bao, err_db = MoHinhDB.du_bao_gia_keras(df, ticker)
        if gia_du_bao:
            du_bao_txt = f"{sym}{gia_du_bao:,.2f}"
            delta_db = ((gia_du_bao - cp) / cp) * 100
            delta_db_txt = f"{'TĂNG' if delta_db > 0 else 'GIẢM'} ({delta_db:+.2f}%)"
        else:
            du_bao_txt = "N/A"
            delta_db = None
            delta_db_txt = err_db if err_db else "Cần Load Model"

        ChiSoTT.ve_the_chi_so(m1, "Giá Hiện Tại", f"{sym}{cp:,.2f}", bien_dong)
        ChiSoTT.ve_the_chi_so(m2, "Xu Hướng (MA20)", "TĂNG" if cp > ma20 else "GIẢM", delta=0 if cp > ma20 else -1)
        ChiSoTT.ve_the_chi_so(m3, "RSI (14)", f"{rsi:.1f}", delta_text="Ổn Định" if 30<=rsi<=70 else "Biến Động")
        ChiSoTT.ve_the_chi_so(m4, "Dự Báo (Mô Hình)", du_bao_txt, delta=delta_db, delta_text=delta_db_txt)

        st.markdown("<br>", unsafe_allow_html=True)
        o1, o2, o3, o4 = st.columns(4)
        ChiSoTT.ve_the_chi_so(o1, "Giá Mở", f"{sym}{op:,.2f}")
        ChiSoTT.ve_the_chi_so(o2, "Giá Cao Nhất", f"{sym}{hp:,.2f}")
        ChiSoTT.ve_the_chi_so(o3, "Giá Thấp Nhất", f"{sym}{lp:,.2f}")
        ChiSoTT.ve_the_chi_so(o4, "Giá Đóng (Phiên Trước)", f"{sym}{df['Close'].iloc[-2]:,.2f}")

    elif menu == "📰 Tin Tức & Sự Kiện":
        TinTucSK.hien_thi_tin_tuc(ticker, f"Tin tức {ticker}")

    elif menu == "📊 Thống Kê & Lịch Sử":
        ThongKe.hien_thi_thong_ke()


    # --- 4. AI CHAT PERSISTENT ---
    st.divider()
    st.markdown("### 💬 Tư Vấn Chiến Lược AI")
    
    # Display chat history
    for m in st.session_state.chat:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    
    # Optional image upload for analysis
    anh_tai = st.file_uploader("Tải biểu đồ phân tích", type=["jpg","png"])
    anh = Image.open(anh_tai) if anh_tai else None
    
    # Chat Input
    if p := st.chat_input("Hỏi AI về chiến lược mua/bán..."):
        st.session_state.chat.append({"role": "user", "content": p})
        # Note: In Streamlit, calling rerun here would reset the current execution 
        # but we want to process the message below. If we want immediate update, 
        # we can use st.rerun(), but since logic is below it will catch it too.
        # However, to be safe and responsive:
        st.rerun()

    # Process AI Response if last message is from user
    if st.session_state.chat and st.session_state.chat[-1]["role"] == "user":
        p_last = st.session_state.chat[-1]["content"]
        with st.chat_message("assistant"):
            if ai_model:
                # Tạo ngữ cảnh chi tiết hơn cho AI
                ctx = (
                    f"Mã: {ticker}, Giá hiện tại: {cp:,.2f}, Mở: {op:,.2f}, Cao: {hp:,.2f}, Thấp: {lp:,.2f}, "
                    f"Khối lượng: {vol:,.0f} (Trung bình 30 ngày: {avg_v:,.0f}), RSI: {rsi:.1f}. "
                    f"Các đường MA: " + ", ".join([f"{k}: {v:,.2f}" for k, v in ma_vals.items()]) + ". "
                    f"Dựa trên các dữ liệu kỹ thuật trên, hãy đóng vai trò là một chuyên gia phân tích kỹ thuật tài chính lão luyện để trả lời câu hỏi của người dùng. "
                    f"YÊU CẦU QUAN TRỌNG: "
                    f"1. Phân tích CHUYÊN SÂU, BÓC TÁCH CHI TIẾT các tín hiệu từ MA, RSI, Khối lượng và biến động giá để trả lời câu hỏi. "
                    f"2. Đưa ra lập luận chặt chẽ, chuyên nghiệp, có chiều sâu về tâm lý thị trường (Market Sentiment) và hành vi giá (Price Action) tại vùng giá hiện tại. "
                    f"3. TUYỆT ĐỐI KHÔNG giải thích vòng vo các khái niệm cơ bản. Phải ĐÚNG TRỌNG TÂM câu hỏi được giao, đi sâu vào bản chất kỹ thuật thay vì liệt kê dữ liệu đơn thuần. "
                    f"Câu hỏi của người dùng: {p_last}"
                )
                with st.spinner("🤖 Đang phân tích dữ liệu và suy nghĩ chiến lược..."):
                    res, err = MoHinhDB.goi_phan_tich_ai(ai_model, ctx, anh)
                
                if res: 
                    st.markdown(res)
                    st.session_state.chat.append({"role": "assistant", "content": res})
                else: st.error(err)
            else: st.warning("Hãy nhập API Key ở Sidebar")

    st.markdown("</div>", unsafe_allow_html=True)

except Exception as e:
    st.error(f"Hệ thống lỗi: {e}")
