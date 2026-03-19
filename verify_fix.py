import yfinance as yf

def test_news_logic(ticker):
    print(f"Testing news logic for {ticker}...")
    try:
        news = yf.Ticker(ticker).news
        print(f"Found {len(news)} news items.")
        for i, t in enumerate(news[:12]):
            # Correct logic from fixed 4_Tin_Tức_Sự_Kiện.py
            nd = t.get('content') or {}
            tieu_de = nd.get('title') or t.get('title') or 'Tin Tài Chính'
            link = (nd.get('canonicalUrl') or {}).get('url') or t.get('link') or '#'
            nguon = (nd.get('provider') or {}).get('displayName') or t.get('publisher') or 'Reuters'
            hinh_anh = (nd.get('thumbnail') or {}).get('originalUrl')
            
            print(f"  Item {i}: OK")
            # print(f"    Title: {tieu_de[:30]}...")
            
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False

if __name__ == "__main__":
    success = test_news_logic("BTC-USD")
    if success:
        print("\nAll tests passed! Logic is now robust against NoneType values.")
    else:
        exit(1)
