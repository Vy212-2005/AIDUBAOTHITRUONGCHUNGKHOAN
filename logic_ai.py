import streamlit as st
import google.generativeai as genai
import time

def configure_gemini(api_key):
    """Cấu hình API Key và tự động tìm kiếm model Gemini khả dụng."""
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if available_models:
            # Ưu tiên các model mới nhất
            preferred = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
            selected_model_name = next((m for m in preferred if m in available_models), available_models[0])
            model = genai.GenerativeModel(selected_model_name)
            st.sidebar.success(f"AI đã sẵn sàng: {selected_model_name.split('/')[-1]}")
            return model
        else:
            st.sidebar.warning("Không tìm thấy model hỗ trợ tạo nội dung.")
    except Exception as e:
        st.sidebar.error(f"Lỗi API: {str(e).split(']')[0]}")
    return None

def get_ai_response(model, context, chart_image=None):
    """Gửi yêu cầu phân tích tới AI (hỗ trợ cả văn bản và hình ảnh)."""
    success = False
    err_msg = ""
    for attempt in range(2):
        try:
            inputs = [context]
            if chart_image:
                inputs.append(chart_image)
            
            res = model.generate_content(inputs)
            return res.text, None
        except Exception as e_gen:
            err_msg = str(e_gen)
            if "429" in err_msg:
                st.info("⏳ Quá tải (Quota 429), đang thử lại sau 10 giây...")
                time.sleep(10)
            else:
                return None, err_msg
    return None, err_msg
