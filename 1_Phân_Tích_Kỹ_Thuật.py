import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def tao_bieu_do_ky_thuat(df, ticker, c_p, o_p, h_p, l_p, v_p, avg_v, ma_vals=None):
    """Tạo biểu đồ nến chuyên sâu với các đường chỉ báo kỹ thuật."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    
    # Nến Nhật
    fig.add_trace(go.Candlestick(
        x=df.index[-100:], 
        open=df['Open'][-100:], high=df['High'][-100:], low=df['Low'][-100:], close=df['Close'][-100:],
        name='Giá nến',
        increasing_line_color='#00ff88', decreasing_line_color='#ff4b4b'
    ), row=1, col=1)
    
    # Chỉ báo SMA/EMA
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA20'][-100:], name='SMA 20', line=dict(color='#ff9f43', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA50'][-100:], name='SMA 50', line=dict(color='#f1c40f', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA100'][-100:], name='SMA 100', line=dict(color='#9b59b6', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['SMA200'][-100:], name='SMA 200', line=dict(color='#e91e63', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-100:], y=df['EMA20'][-100:], name='EMA 20', line=dict(color='#4facfe', width=1.2, dash='dot')), row=1, col=1)
    
    # Khối lượng giao dịch
    vol_colors = ['#00ff88' if c >= o else '#ff4b4b' for o, c in zip(df['Open'][-100:], df['Close'][-100:])]
    fig.add_trace(go.Bar(x=df.index[-100:], y=df['Volume'][-100:], marker_color=vol_colors, name='Volume'), row=2, col=1)
    
    # Hiển thị thông số OHLC và MA hiện tại trên biểu đồ
    mau_gia = '#00ff88' if c_p >= o_p else '#ff4b4b'
    ohlc_text = f"<b>Giá Mở: {o_p:,.2f} | Giá Cao: {h_p:,.2f} | Giá Thấp: {l_p:,.2f} | Giá Đóng: {c_p:,.2f}</b>"
    
    ma_text = ""
    if ma_vals:
        ma_text = "<br>" + " | ".join([f"{k}: {v:,.2f}" for k,v in ma_vals.items()])
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.98,
        text=ohlc_text + ma_text,
        showarrow=False,
        font=dict(size=12, color="white", family="Inter, sans-serif"),
        align="left",
        bgcolor="rgba(0,0,0,0.6)",
        bordercolor=mau_gia,
        borderwidth=1,
        borderpad=6,
    )
    
    fig.update_layout(template='plotly_dark', xaxis_rangeslider_visible=False, height=600, margin=dict(t=30, b=0, l=10, r=10))
    return fig
