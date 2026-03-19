import streamlit as st
import pandas as pd
import datetime
import os
import plotly.express as px
import yfinance as yf

LOG_FILE = "truy_cap_log.csv"
# Danh sách mã được quan tâm toàn cầu
GLOBAL_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "GOOGL", "BTC-USD", "ETH-USD", "VND=X", "XAU=F", "META"]

def ghi_log_truy_cap(ticker, hanh_dong):
    """Ghi lại lịch sử truy cập mã cổ phiếu và tính năng."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([[now, ticker, hanh_dong]], columns=["Thời gian", "Mã", "Tính năng"])
    
    if not os.path.isfile(LOG_FILE):
        new_data.to_csv(LOG_FILE, index=False, encoding='utf-8-sig')
    else:
        new_data.to_csv(LOG_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')

def doc_log_truy_cap():
    """Đọc dữ liệu từ file log."""
    if not os.path.isfile(LOG_FILE):
        return pd.DataFrame(columns=["Thời gian", "Mã", "Tính năng"])
    return pd.read_csv(LOG_FILE)

@st.cache_data(ttl=3600)
def lay_du_lieu_the_gioi():
    """Lấy dữ liệu biến động giá của các mã phổ biến trên thế giới."""
    data = []
    try:
        # Tải dữ liệu 2 ngày gần nhất để tính % biến động
        df = yf.download(GLOBAL_TICKERS, period="2d", group_by='ticker', progress=False)
        
        for t in GLOBAL_TICKERS:
            try:
                # Trích xuất dữ liệu cho từng mã từ MultiIndex DataFrame
                if len(GLOBAL_TICKERS) > 1:
                    ticker_df = df[t]
                else:
                    ticker_df = df
                
                # Bỏ qua nếu không có dữ liệu hoặc toàn giá trị NaN
                if ticker_df.empty or ticker_df['Close'].isnull().all():
                    continue

                # Lấy 2 giá đóng cửa gần nhất (loại bỏ NaN)
                prices = ticker_df['Close'].dropna()
                if len(prices) >= 2:
                    c = prices.iloc[-1]
                    p = prices.iloc[-2]
                    change = ((c - p) / p) * 100
                    data.append({"Mã": t, "Giá Hiện Tại": c, "Biến động (%)": change})
            except Exception:
                continue
    except Exception as e:
        st.error(f"Lỗi tải dữ liệu thế giới: {e}")
    
    res_df = pd.DataFrame(data)
    if not res_df.empty:
        return res_df.sort_values(by="Biến động (%)", ascending=False)
    return res_df

def hien_thi_heatmap_the_gioi():
    """Hiển thị Heatmap thị trường toàn cầu với chú thích rõ ràng."""
    st.info("""
        💡 **Hướng dẫn đọc biểu đồ:**
        - **Kích thước ô**: Thể hiện thị giá (Giá cao hơn = Ô to hơn).
        - **Màu sắc**: Thể hiện biến động trong ngày (Xanh = Tăng, Đỏ = Giảm, Xám = Đứng giá).
        - **Đậm nhạt**: Màu càng đậm thì mức tăng/giảm càng mạnh.
    """)
    
    with st.spinner("Đang tải dữ liệu thế giới..."):
        df_world = lay_du_lieu_the_gioi()
    
    if not df_world.empty:
        # Làm sạch dữ liệu hiển thị (Thêm cột text cho label)
        df_world['Label'] = df_world.apply(lambda r: f"{r['Mã']}<br>{r['Giá Hiện Tại']:,.1f}<br>{r['Biến động (%)']:+.2f}%", axis=1)
        
        # Heatmap Treemap chuyên nghiệp hơn
        fig_tree = px.treemap(df_world, 
                             path=[px.Constant("Thị Trường Toàn Cầu"), 'Mã'], 
                             values='Giá Hiện Tại',
                             color='Biến động (%)',
                             color_continuous_scale='RdYlGn',
                             color_continuous_midpoint=0,
                             custom_data=['Giá Hiện Tại', 'Biến động (%)'],
                             title="Bản đồ nhiệt: Tương quan Giá & Biến động")
        
        fig_tree.update_traces(
            texttemplate="%{label}<br>%{customdata[1]:+.2f}%",
            textposition="middle center",
            hovertemplate="<b>Mã:</b> %{label}<br><b>Giá:</b> %{customdata[0]:,.2f}<br><b>Biến động:</b> %{customdata[1]:+.2f}%"
        )
        
        fig_tree.update_layout(
            margin=dict(t=50, l=10, r=10, b=10),
            coloraxis_colorbar=dict(
                title="Biến động (%)",
                thicknessmode="pixels", thickness=15,
                lenmode="pixels", len=200,
                yanchor="top", y=1,
                ticks="outside"
            )
        )
        
        st.plotly_chart(fig_tree, use_container_width=True)
        return df_world
    return None

def hien_thi_thong_ke():
    """Giao diện hiển thị thống kê và lịch sử cá nhân."""
    st.markdown("## 👤 Quản lý Lịch sử & Hoạt động")
    
    df_log = doc_log_truy_cap()
    
    # 1. Thống kê tổng quan
    if not df_log.empty:
        c1, c2, c3, c4 = st.columns(4)
        total_views = len(df_log)
        unique_tickers = df_log['Mã'].nunique()
        most_active_day = pd.to_datetime(df_log['Thời gian']).dt.date.mode()[0]
        fav_feature = df_log['Tính năng'].mode()[0] if not df_log['Tính năng'].empty else "N/A"
        
        c1.metric("Tổng lượt truy cập", f"{total_views:,}")
        c2.metric("Số mã đã tra", unique_tickers)
        c3.metric("Ngày hoạt động nhất", str(most_active_day))
        c4.metric("Tính năng yêu thích", fav_feature)
        st.divider()

    # 2. Tabs phân chia lịch sử cá nhân và phân tích
    tab_personal, tab_analysis = st.tabs(["👤 Lịch Sử Cá Nhân", "📈 Phân Tích Hoạt Động"])

    with tab_personal:
        if df_log.empty:
            st.info("Chưa có dữ liệu lịch sử cá nhân.")
        else:
            col_p_chart, col_p_table = st.columns([0.6, 0.4])
            with col_p_chart:
                st.subheader("📈 Mã bạn quan tâm nhất")
                top_p = df_log['Mã'].value_counts().head(10).reset_index()
                top_p.columns = ['Mã', 'Lượt xem']
                fig_p = px.pie(top_p, values='Lượt xem', names='Mã', hole=0.4,
                               template='plotly_dark', color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_p, use_container_width=True)
            
            with col_p_table:
                st.subheader("📋 Xếp hạng cá nhân")
                st.dataframe(top_p, hide_index=True, use_container_width=True)
            
            st.divider()
            st.subheader("🕒 Nhật ký tra cứu gần đây")
            st.dataframe(df_log.iloc[::-1], use_container_width=True, hide_index=True)

    with tab_analysis:
        if not df_log.empty:
            st.subheader("📅 Tần suất tra cứu hàng ngày")
            df_log['Ngày'] = pd.to_datetime(df_log['Thời gian']).dt.date
            df_trend = df_log.groupby('Ngày').size().reset_index(name='Lượt truy cập')
            fig_trend = px.line(df_trend, x='Ngày', y='Lượt truy cập', markers=True,
                               template='plotly_dark', color_discrete_sequence=['#4facfe'])
            st.plotly_chart(fig_trend, use_container_width=True)
            
            st.subheader("🛠️ Phân bổ tính năng")
            feat_count = df_log['Tính năng'].value_counts().reset_index()
            feat_count.columns = ['Tính năng', 'Số lần dùng']
            fig_feat = px.bar(feat_count, x='Số lần dùng', y='Tính năng', orientation='h',
                             color='Số lần dùng', color_continuous_scale='Blues')
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Cần thêm dữ liệu để phân tích xu hướng.")

    # Nút xóa lịch sử (Vẫn giữ ở cuối)
    if not df_log.empty:
        if st.button("🗑️ Xóa lịch sử cá nhân"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
                st.success("Đã xóa lịch sử thành công!")
                st.rerun()
