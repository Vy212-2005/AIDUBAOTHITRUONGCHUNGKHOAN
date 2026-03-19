import streamlit as st
import google.generativeai as genai
import time
import os
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

def cau_hinh_gemini(keys_string):
    """
    Thiết lập AI Gemini cho việc dự báo.
    keys_string: Có thể là một key đơn lẻ hoặc danh sách các keys cách nhau bởi dấu phẩy.
    """
    if not keys_string: return None
    
    # Chuyển đổi chuỗi keys thành danh sách (list)
    if isinstance(keys_string, str):
        api_keys = [k.strip() for k in keys_string.split(',') if k.strip()]
    else:
        api_keys = keys_string

    if not api_keys: return None

    # Lưu danh sách keys vào session_state để quản lý xoay vòng
    if "api_keys_list" not in st.session_state or st.session_state.api_keys_list != api_keys:
        st.session_state.api_keys_list = api_keys
        st.session_state.current_key_index = 0

    current_index = st.session_state.current_key_index
    key = api_keys[current_index]

    try:
        genai.configure(api_key=key)
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        name = next((m for m in ['models/gemini-1.5-flash', 'models/gemini-pro'] if m in models), models[0])
        model = genai.GenerativeModel(name)
        
        # Hiển thị thông báo sử dụng key số mấy (để người dùng biết)
        key_masked = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
        st.sidebar.success(f"AI sẵn sàng (Key #{current_index + 1}: {key_masked})")
        return model
    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "429" in error_str:
            # Nếu còn key khác, thử xoay sang key tiếp theo
            if len(api_keys) > 1:
                next_index = (current_index + 1) % len(api_keys)
                if next_index != current_index:
                    st.session_state.current_key_index = next_index
                    st.sidebar.warning(f"Key #{current_index + 1} lỗi ({error_str[:30]}...). Đang đổi sang Key #{next_index + 1}")
                    st.rerun()
        
        st.sidebar.error(f"Lỗi AI: {e}")
    return None

def goi_phan_tich_ai(model, context, anh=None):
    """Gửi dữ liệu thị trường cho AI phân tích, có hỗ trợ retry logic."""
    for attempt in range(2):
        try:
            dau_vao = [context]
            if anh: dau_vao.append(anh)
            res = model.generate_content(dau_vao)
            return res.text, None
        except Exception as e:
            err_msg = str(e)
            # Nếu lỗi Quota (429) hoặc Quyền truy cập (403), thử đổi key ở lần gọi sau
            if "429" in err_msg or "403" in err_msg:
                if "api_keys_list" in st.session_state and len(st.session_state.api_keys_list) > 1:
                    st.session_state.current_key_index = (st.session_state.current_key_index + 1) % len(st.session_state.api_keys_list)
                    st.warning("⚠️ Lỗi API: Đang tự động đổi sang Key dự phòng...")
                    time.sleep(1)
                    st.rerun()
            return None, f"Lỗi AI: {err_msg}"
    return None, "Không thể kết nối với AI sau nhiều lần thử."

def du_bao_gia_keras(df, ticker):
    """Sử dụng mô hình LSTM đã train (nếu có) để dự báo giá ngày kế tiếp."""
    ticker_low = ticker.lower()
    
    # Sử dụng đường dẫn tuyệt đối dựa trên vị trí file này
    base_dir = Path(__file__).parent
    model_path = base_dir / f'model_{ticker_low}.keras'
    scaler_path = base_dir / f'scaler_{ticker_low}.pkl'
    
    if not model_path.exists() or not scaler_path.exists():
        # st.sidebar.info(f"DEBUG: Missing {model_path.name}") # Optional debug
        return None, "Chưa có Model cho mã này"
    
    try:
        # Load model và scaler (chuyển sang string cho Keras/Joblib nếu cần)
        model = load_model(str(model_path))
        scaler = joblib.load(str(scaler_path))
        
        # Đặc trưng giống như trong train_model.py
        feature_cols = ['Open', 'High', 'Low', 'Close', 'SMA20', 'SMA50', 'EMA20', 'RSI', 'Volume']
        
        # Lấy 60 ngày gần nhất
        if len(df) < 60:
            return None, "Cần tối thiểu 60 ngày dữ liệu"
            
        last_60_days = df[feature_cols].tail(60).values
        
        # Scale
        scaled_data = scaler.transform(last_60_days)
        
        # Reshape cho LSTM: (1, 60, n_features)
        X_input = np.array([scaled_data])
        
        # Predict
        prediction_scaled = model.predict(X_input, verbose=0)
        
        # Inverse Scale (để lấy giá thực tế)
        # Vì scaler fit trên toàn bộ features, ta cần tạo 1 array dummy để inverse
        dummy = np.zeros((1, len(feature_cols)))
        target_idx = feature_cols.index('Close')
        dummy[0, target_idx] = prediction_scaled[0, 0]
        
        prediction_final = scaler.inverse_transform(dummy)[0, target_idx]
        
        return prediction_final, None
    except Exception as e:
        return None, f"Lỗi dự báo: {str(e)}"
