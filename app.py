import re
import requests
import feedparser
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.arima.model import ARIMA

# ---------------- Prophet (Primary Forecast Model) ----------------
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------------- Gemini (MANDATORY) ----------------
import google.generativeai as genai

# ---------------- Wikipedia ----------------
try:
    import wikipedia
    WIKI_AVAILABLE = True
except Exception:
    WIKI_AVAILABLE = False

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- UI Styling ----------------
st.markdown("""
<style>
.metric-card {
    background:#0f172a;
    padding:16px;
    border-radius:14px;
    text-align:center;
}
.metric-title { color:#94a3b8; font-size:0.9rem; }
.metric-value { color:#e5e7eb; font-size:1.4rem; font-weight:700; }
.badge {
    padding:8px 14px;
    border-radius:12px;
    font-weight:700;
}
.section {
    background:#020617;
    padding:18px;
    border-radius:18px;
    margin-bottom:18px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is mandatory. Add it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ============================================================
# SIDEBAR – USER INPUT
# ============================================================
with st.sidebar:
    st.header("🔎 Asset Selection")
    user_input = st.text_input(
        "Company / Crypto / Ticker",
        value="Tesla",
        help="Examples: Tesla, Apple, Infosys, Bitcoin, TSLA, BTC"
    )
    horizon = st.slider("Forecast Horizon (Days)", 3, 14, 7)
    run_btn = st.button("Run Analysis", use_container_width=True)

if not run_btn:
    st.stop()

# ============================================================
# SMART SYMBOL RESOLUTION
# ============================================================
def resolve_symbol(query: str):
    crypto_map = {
        "bitcoin": "BTC-USD",
        "btc": "BTC-USD",
        "ethereum": "ETH-USD",
        "eth": "ETH-USD",
        "solana": "SOL-USD",
        "sol": "SOL-USD"
    }
    q = query.strip().lower()
    if q in crypto_map:
        return crypto_map[q], "Crypto"

    # Direct ticker check
    try:
        t = yf.Ticker(query.upper())
        if not t.history(period="5d").empty:
            return query.upper(), "Equity"
    except Exception:
        pass

    # Yahoo search API
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.replace(' ', '+')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        data = r.json()
        for item in data.get("quotes", []):
            if item.get("quoteType") == "EQUITY":
                return item.get("symbol"), "Equity"
    except Exception:
        pass

    return None, None

symbol, asset_type = resolve_symbol(user_input)

if not symbol:
    st.error("❌ Could not resolve asset. Try a different name.")
    st.stop()

# ============================================================
# MARKET DATA
# ============================================================
@st.cache_data(ttl=600)
def fetch_market(symbol):
    df = yf.download(symbol, period="1y", progress=False)
    if df is None or df.empty:
        return None
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    return df

market_df = fetch_market(symbol)

if market_df is None:
    st.error("❌ No historical data available.")
    st.stop()

current_price = float(market_df["y"].iloc[-1])

# ============================================================
# COMPANY PROFILE (EQUITY ONLY)
# ============================================================
company_info = {}
wiki_summary = ""

if asset_type == "Equity":
    try:
        info = yf.Ticker(symbol).info
        company_info = {
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website")
        }
    except Exception:
        company_info = {}

    if WIKI_AVAILABLE:
        try:
            results = wikipedia.search(company_info.get("name", user_input))
            if results:
                wiki_summary = wikipedia.summary(results[0], sentences=3)
        except Exception:
            wiki_summary = "Wikipedia summary not available."

# ============================================================
# NEWS + GEMINI SENTIMENT
# ============================================================
@st.cache_data(ttl=900)
def gemini_sentiment(texts):
    scores = []
    for text in texts[:5]:
        try:
            prompt = f"Return sentiment between -100 and 100 as integer only:\n{text[:1200]}"
            r = GEMINI_MODEL.generate_content(prompt)
            m = re.search(r"-?\d+", r.text)
            scores.append(int(m.group()) if m else 0)
        except Exception:
            scores.append(0)
    return float(np.mean(scores)) if scores else 0.0

news_query = f"{user_input} stock" if asset_type == "Equity" else user_input
news_url = f"https://news.google.com/rss/search?q={news_query.replace(' ', '+')}"
feed = feedparser.parse(news_url)
news_texts = [f"{e.title}. {getattr(e,'summary','')}" for e in feed.entries[:8]]

sentiment = gemini_sentiment(news_texts)

# ============================================================
# FORECASTING (PROPHET → ARIMA)
# ============================================================
forecast_price = None
model_used = None

if PROPHET_AVAILABLE:
    try:
        m = Prophet(daily_seasonality=True)
        m.fit(market_df)
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future).tail(horizon)
        forecast_price = float(fc["yhat"].mean())
        model_used = "Prophet"
    except Exception:
        PROPHET_AVAILABLE = False

if not PROPHET_AVAILABLE:
    series = market_df["y"]
    arima = ARIMA(series, order=(1,1,1)).fit()
    pred = arima.forecast(horizon)
    forecast_price = float(pred.mean())
    model_used = "ARIMA"

pct_change = ((forecast_price - current_price) / current_price) * 100

# ============================================================
# SIGNAL ENGINE
# ============================================================
if pct_change > 3 and sentiment > 20:
    signal, color = "STRONG BUY", "#16a34a"
elif pct_change > 1:
    signal, color = "BUY", "#22c55e"
elif pct_change < -3 and sentiment < -20:
    signal, color = "STRONG SELL", "#dc2626"
elif pct_change < -1:
    signal, color = "SELL", "#ef4444"
else:
    signal, color = "HOLD", "#eab308"

# ============================================================
# DASHBOARD UI
# ============================================================
st.title("Infosys InsightSphere")
st.caption("Real-Time Industry Insight & Strategic Intelligence Dashboard")

st.markdown(f"Viewing **{user_input} ({symbol})** — Asset Type: **{asset_type}**")

c1,c2,c3,c4,c5 = st.columns(5)
c1.markdown(f"<div class='metric-card'><div class='metric-title'>Current Price</div><div class='metric-value'>${current_price:.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric-card'><div class='metric-title'>Forecast Avg</div><div class='metric-value'>${forecast_price:.2f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric-card'><div class='metric-title'>Projected Move</div><div class='metric-value'>{pct_change:+.2f}%</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric-card'><div class='metric-title'>Sentiment</div><div class='metric-value'>{sentiment:+.1f}</div></div>", unsafe_allow_html=True)
c5.markdown(f"<div class='metric-card'><div class='metric-title'>Model</div><div class='metric-value'>{model_used}</div></div>", unsafe_allow_html=True)

st.markdown(f"<div class='badge' style='background:{color}'>Signal: {signal}</div>", unsafe_allow_html=True)

# ---------------- Chart ----------------
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=market_df["ds"], y=market_df["y"], name="Price"))
fig.update_layout(height=420, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ---------------- Company Overview ----------------
if asset_type == "Equity":
    st.subheader("Company Overview")
    st.markdown(f"**Sector:** {company_info.get('sector','N/A')}")
    st.markdown(f"**Industry:** {company_info.get('industry','N/A')}")
    st.markdown(f"**Country:** {company_info.get('country','N/A')}")
    st.markdown(f"**Website:** {company_info.get('website','N/A')}")
    st.info(wiki_summary)

# ---------------- Executive Summary (Gemini Safe) ----------------
st.subheader("Executive Strategic Summary")
try:
    prompt = f"""
    Company: {user_input}
    Asset Type: {asset_type}
    Current Price: {current_price}
    Forecast Avg: {forecast_price}
    Projected Change: {pct_change:+.2f}%
    Sentiment: {sentiment:+.1f}
    Signal: {signal}

    Provide executive summary, key drivers, risks, opportunities and recommendation.
    """
    exec_summary = GEMINI_MODEL.generate_content(prompt).text
except Exception:
    exec_summary = "AI summary temporarily unavailable due to quota limits."

st.write(exec_summary)

# ---------------- Slack Alert ----------------
if SLACK_WEBHOOK_URL and st.button("Send Slack Alert"):
    msg = (
        f"InsightSphere Alert\n"
        f"{user_input} ({symbol})\n"
        f"Signal: {signal}\n"
        f"Price: {current_price:.2f} → {forecast_price:.2f}\n"
        f"Move: {pct_change:+.2f}%"
    )
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=10)
        st.success("Slack alert sent")
    except Exception as e:
        st.error(f"Slack failed: {e}")

st.caption("© 2025 Infosys Springboard Internship — Real time Industrial Insights")
