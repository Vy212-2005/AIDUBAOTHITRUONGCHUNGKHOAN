import yfinance as yf
import json

ticker = "BTC-USD"
print(f"Fetching news for {ticker}...")
try:
    news = yf.Ticker(ticker).news
    print(f"Found {len(news)} news items.")
    for i, t in enumerate(news[:12]):
        print(f"\nItem {i}:")
        nd = t.get('content', {})
        print(f"  nd type: {type(nd)}")
        if nd is None:
            print("  WARNING: nd is None!")
        else:
            try:
                tieu_de = nd.get('title', t.get('title', 'Tin Tài Chính'))
                print(f"  Title: {tieu_de}")
            except Exception as e:
                print(f"  Error getting title: {e}")
                
            try:
                # Potential crash site 1
                can_url = nd.get('canonicalUrl', {})
                print(f"  can_url type: {type(can_url)}")
                if can_url is None:
                    print("  WARNING: can_url is None!")
                link = can_url.get('url', t.get('link', '#')) if can_url else t.get('link', '#')
                print(f"  Link: {link}")
            except Exception as e:
                print(f"  Error getting link: {e}")

            try:
                # Potential crash site 2
                prov = nd.get('provider', {})
                print(f"  prov type: {type(prov)}")
                if prov is None:
                    print("  WARNING: prov is None!")
                nguon = prov.get('displayName', t.get('publisher', 'Reuters')) if prov else t.get('publisher', 'Reuters')
                print(f"  Source: {nguon}")
            except Exception as e:
                print(f"  Error getting provider: {e}")

except Exception as e:
    print(f"Global error: {e}")
