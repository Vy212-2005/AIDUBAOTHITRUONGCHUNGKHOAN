import streamlit as st
import yfinance as yf
import time

def hien_thi_tin_tuc(ticker, che_do):
    """Lấy và trình bày tin tức tài chính thế giới hoặc mã ticker."""
    tin_tuc = []
    if che_do == "🌍 Tài chính Thế giới":
        for chi_so in ["^GSPC", "^DJI", "^IXIC"]:
            try: tin_tuc.extend(yf.Ticker(chi_so).news)
            except: continue
        tin_tuc = sorted({t['uuid']: t for t in tin_tuc if 'uuid' in t}.values(), key=lambda x: x.get('provider_publish_time') or 0, reverse=True)
    else:
        try: tin_tuc = yf.Ticker(ticker).news
        except: tin_tuc = []

    if not tin_tuc:
        st.info("Chưa có tin tức mới.")
    else:
        for t in tin_tuc[:12]:
            with st.container():
                c1, c2 = st.columns([1, 3])
                nd = t.get('content') or {}
                tieu_de = nd.get('title') or t.get('title') or 'Tin Tài Chính'
                link = (nd.get('canonicalUrl') or {}).get('url') or t.get('link') or '#'
                nguon = (nd.get('provider') or {}).get('displayName') or t.get('publisher') or 'Reuters'
                hinh_anh = (nd.get('thumbnail') or {}).get('originalUrl')
                
                with c1:
                    if hinh_anh:
                        st.image(hinh_anh, use_container_width=True)
                    else:
                        st.image("https://via.placeholder.com/150?text=No+Image", use_container_width=True)
                with c2:
                    st.markdown(f"**[{tieu_de}]({link})**")
                    st.caption(f"Nguồn: {nguon}")
                st.divider()
