import streamlit as st

# --- CẤU HÌNH HỆ THỐNG ---
DUONG_DAN_PYTHON = r"C:\Users\LOQ\anaconda3\envs\stock_ai\python.exe"

# --- GIAO DIỆN PREMIUM (CSS) ---
CSS_CHUNG_KHOAN = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #06080c;
    }
    
    .main-title {
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #4facfe 0%, #0099ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .metric-card {
        background: rgba(16, 18, 22, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5);
    }
    
    .metric-label { color: #94a3b8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
    .metric-value { color: #f8fafc; font-size: 1.8rem; font-weight: 700; }
    .metric-delta-up { color: #10b981; font-size: 0.85rem; font-weight: 600; }
    .metric-delta-down { color: #ef4444; font-size: 0.85rem; font-weight: 600; }

    /* --- HIỆU ỨNG CHUYỂN CẢNH --- */
    .fade-in {
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* Tinh chỉnh sidebar và nút bấm */
    .stSelectbox, .stTextInput {
        transition: all 0.3s ease;
    }
    .stSelectbox:hover, .stTextInput:hover {
        transform: translateY(-2px);
    }
</style>
"""
