import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA

# ---------------- Prophet (Optional) ----------------
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception as e:
    PROPHET_AVAILABLE = False
    st.warning(f"Prophet not available: {e}. Falling back to ARIMA.")

# ---------------- Gemini ----------------
import google.generativeai as genai

# ---------------- Wikipedia (Optional) ----------------
try:
    import wikipedia
    WIKI_AVAILABLE = True
except Exception:
    WIKI_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================
st.set_page_config(page_title="Infosys InsightSphere", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.metric-card {background:#0f172a; padding:16px; border-radius:14px; text-align:center;}
.metric-title {color:#94a3b8; font-size:0.9rem;}
.metric-value {color:#e5e7eb; font-size:1.4rem; font-weight:700;}
.badge {padding:8px 16px; border-radius:12px; font-weight:700; display:inline-block; font-size:1.1rem;}
.section {background:#020617; padding:20px; border-radius:18px; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is required in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")  # More reliable than 2.0-flash

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🔎 Asset Selection")
    user_input = st.text_input("Company / Crypto / Ticker", value="Tesla", help="e.g., Tesla, Apple, BTC, TSLA")
    horizon = st.slider("Forecast Horizon (Days)", 3, 14, 7)
    debug_mode = st.checkbox("Enable Debug Logs", False)
    run_btn = st.button("🚀 Run Analysis", use_container_width=True, type="primary")

if not run_btn:
    st.info("Enter an asset and click 'Run Analysis' to begin.")
    st.stop()

# ============================================================
# SYMBOL RESOLUTION
# ============================================================
def resolve_symbol(query: str):
    query = query.strip()
    crypto_map = {"bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "eth": "ETH-USD"}
    lower = query.lower()
    if lower in crypto_map:
        return crypto_map[lower], "Crypto"

    # Direct ticker
    try:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        test_df = yf.download(query.upper(), period="5d", progress=False, session=session)
        if not test_df.empty:
            return query.upper(), "Equity"
    except:
        pass

    # Yahoo search
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            for item in data.get("quotes", []):
                if item.get("quoteType") in ["EQUITY", "CRYPTOCURRENCY"]:
                    return item.get("symbol"), "Equity" if item.get("quoteType") == "EQUITY" else "Crypto"
    except Exception as e:
        if debug_mode: st.warning(f"Search API error: {e}")

    return None, None

symbol, asset_type = resolve_symbol(user_input)
if not symbol:
    st.error(f"❌ Could not resolve '{user_input}'. Try a valid ticker or company name.")
    st.stop()

st.success(f"Resolved: **{symbol}** ({asset_type})")

# ============================================================
# ROBUST MARKET DATA FETCH
# ============================================================
@st.cache_data(ttl=600, show_spinner="Fetching market data...")
def fetch_market(symbol: str):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    session = requests.Session()
    session.headers.update(headers)

    # Method 1: yf.download with session + retries
    for attempt in range(3):
        try:
            df = yf.download(
                symbol,
                period="1y",
                auto_adjust=True,
                progress=False,
                session=session
            )
            if not df.empty and len(df) > 50:
                df = df.reset_index()[["Date", "Close"]].copy()
                df.columns = ["ds", "y"]
                df["ds"] = pd.to_datetime(df["ds"]).dt.date
                return df, "Yahoo Finance (download)"
        except Exception as e:
            if debug_mode: st.warning(f"Attempt {attempt+1} download failed: {e}")
            time.sleep(1)

    # Method 2: Ticker.history
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", auto_adjust=True)
        if not hist.empty and len(hist) > 50:
            hist = hist.reset_index()[["Date", "Close"]].copy()
            hist.columns = ["ds", "y"]
            hist["ds"] = pd.to_datetime(hist["ds"]).dt.date
            return hist, "Yahoo Finance (history)"
    except Exception as e:
        if debug_mode: st.warning(f"Ticker.history failed: {e}")

    return None, "Failed"

market_df, source = fetch_market(symbol)
if market_df is None:
    st.error("❌ Failed to fetch market data after multiple attempts. Try again later or check the ticker.")
    if debug_mode: st.info("Tip: Use yfinance==0.2.38 in requirements.txt for maximum stability.")
    st.stop()

current_price = float(market_df["y"].iloc[-1])
st.caption(f"Data source: {source} • {len(market_df)} days")

# ============================================================
# COMPANY PROFILE (Equity only)
# ============================================================
company_info = {"name": user_input}
wiki_summary = ""
if asset_type == "Equity":
    try:
        info = yf.Ticker(symbol).info
        company_info.update({
            "name": info.get("longName", user_input),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "website": info.get("website", "N/A")
        })
    except:
        pass

    if WIKI_AVAILABLE:
        try:
            wiki_summary = wikipedia.summary(company_info["name"], sentences=4)
        except:
            wiki_summary = "Wikipedia summary unavailable."

# ============================================================
# NEWS & SENTIMENT
# ============================================================
@st.cache_data(ttl=1800)
def get_sentiment(query: str):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        import feedparser
        feed = feedparser.parse(feed)
        texts = [f"{e.title} {getattr(e, 'summary', '')}" for e in feed.entries[:10]]
        
        scores = []
        for text in texts[:6]:  # Limit to avoid quota
            try:
                prompt = f"On a scale of -100 (very negative) to +100 (very positive), rate the sentiment of this financial news strictly as an integer only:\n\n{text[:1000]}"
                response = GEMINI_MODEL.generate_content(prompt)
                match = re.search(r"-?\d+", response.text)
                scores.append(int(match.group()) if match else 0)
            except:
                scores.append(0)
        return np.mean(scores) if scores else 0.0
    except:
        return 0.0

sentiment = get_sentiment(f"{user_input} stock" if asset_type == "Equity" else user_input)

# ============================================================
# FORECASTING
# ============================================================
try:
    if PROPHET_AVAILABLE:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(market_df)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        forecast_price = forecast["yhat"].tail(horizon).mean()
        model_used = "Prophet"
    else:
        raise Exception("Prophet unavailable")
except Exception as e:
    if debug_mode: st.warning(f"Prophet failed: {e}")
    series = market_df["y"]
    try:
        model = ARIMA(series, order=(5,1,0))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=horizon)
        forecast_price = float(pred.mean())
        model_used = "ARIMA"
    except:
        forecast_price = current_price
        model_used = "Static (fallback)"

pct_change = ((forecast_price - current_price) / current_price) * 100

# ============================================================
# SIGNAL
# ============================================================
if pct_change > 3 and sentiment > 15:
    signal, color = "STRONG BUY", "#16a34a"
elif pct_change > 1.5:
    signal, color = "BUY", "#22c55e"
elif pct_change < -3 and sentiment < -15:
    signal, color = "STRONG SELL", "#dc2626"
elif pct_change < -1.5:
    signal, color = "SELL", "#ef4444"
else:
    signal, color = "HOLD", "#eab308"

# ============================================================
# DASHBOARD
# ============================================================
st.title("📊 Infosys InsightSphere")
st.markdown(f"### {company_info['name']} ({symbol}) • {asset_type}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Forecast Avg", f"${forecast_price:.2f}")
col3.metric("Projected Change", f"{pct_change:+.2f}%")
col4.metric("News Sentiment", f"{sentiment:+.1f}")
col5.metric("Model Used", model_used)

st.markdown(f"<div class='badge' style='background:{color}; color:white;'>{signal}</div>", unsafe_allow_html=True)

# Chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=market_df["ds"], y=market_df["y"], name="Historical", line=dict(color="#3b82f6")))
fig.update_layout(height=500, template="plotly_dark", title="Price History & Forecast Context")
st.plotly_chart(fig, use_container_width=True)

if asset_type == "Equity":
    st.subheader("🏢 Company Overview")
    st.write(f"**Sector:** {company_info['sector']} | **Industry:** {company_info['industry']} | **Country:** {company_info['country']}")
    if company_info.get("website"):
        st.write(f"**Website:** {company_info['website']}")
    if wiki_summary:
        st.info(wiki_summary)

st.subheader("📝 Executive Strategic Summary")
with st.spinner("Generating AI-powered insights..."):
    try:
        prompt = f"""
        Provide a concise executive summary for {company_info['name']} ({symbol}):
        - Current price: ${current_price:.2f}
        - {horizon}-day forecast: ${forecast_price:.2f} ({pct_change:+.2f}%)
        - Market sentiment: {sentiment:+.1f}
        - Signal: {signal}
        
        Include key drivers, risks, opportunities, and final recommendation.
        """
        summary = GEMINI_MODEL.generate_content(prompt).text
        st.write(summary)
    except Exception as e:
        st.error("AI summary failed (quota or network). Try again later.")

if st.button("📤 Send Slack Alert"):
    if "SLACK_WEBHOOK_URL" in st.secrets:
        payload = {"text": f"*InsightSphere Alert*\n*{user_input} ({symbol})*\nSignal: {signal}\nPrice: ${current_price:.2f} → ${forecast_price:.2f} ({pct_change:+.2f}%)\nSentiment: {sentiment:+.1f}"}
        try:
            requests.post(st.secrets["SLACK_WEBHOOK_URL"], json=payload)
            st.success("Alert sent to Slack!")
        except:
            st.error("Slack send failed.")
    else:
        st.warning("SLACK_WEBHOOK_URL not configured.")

st.caption("© 2025 Infosys Springboard Internship Project — Real-Time Strategic Intelligence System")
