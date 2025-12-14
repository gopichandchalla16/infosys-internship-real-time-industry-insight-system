import re
import requests
import feedparser
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

import yfinance as yf
from pandas_datareader import data as pdr

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

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "")
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is mandatory")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", value="TSLA").upper().strip()
    company_name = st.text_input("Company Name", value="Tesla, Inc.")
    horizon = st.slider("Forecast Horizon (Days)", 3, 14, 7)
    run_btn = st.button("Run Analysis")

if not run_btn:
    st.stop()

# ============================================================
# MARKET DATA (yfinance → Alpha Vantage → STOOQ)
# ============================================================
@st.cache_data(ttl=900)
def fetch_market_data(symbol: str):
    # ---- Method 1: yfinance.download ----
    try:
        df = yf.download(
            symbol,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if df is not None and not df.empty:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["ds", "y"]
            return df
    except Exception:
        pass

    # ---- Method 2: yfinance Ticker().history ----
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="1y", auto_adjust=True)
        if hist is not None and not hist.empty:
            hist = hist.reset_index()[["Date", "Close"]]
            hist.columns = ["ds", "y"]
            return hist
    except Exception:
        pass

    # ---- Method 3: Alpha Vantage ----
    if ALPHA_VANTAGE_API_KEY:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "compact",
                "apikey": ALPHA_VANTAGE_API_KEY
            }
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            ts = data.get("Time Series (Daily)", {})
            if ts:
                rows = [
                    {"ds": pd.to_datetime(d), "y": float(v["5. adjusted close"])}
                    for d, v in ts.items()
                ]
                return pd.DataFrame(rows).sort_values("ds")
        except Exception:
            pass

    # ---- Method 4: STOOQ (guaranteed fallback) ----
    try:
        df = pdr.DataReader(symbol.lower(), "stooq")
        if df is not None and not df.empty:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["ds", "y"]
            return df.sort_values("ds")
    except Exception:
        pass

    return None

with st.spinner("Fetching historical market data..."):
    market_df = fetch_market_data(ticker)

if market_df is None or market_df.empty:
    st.error(f"❌ Could not fetch historical data for '{ticker}'.")
    st.stop()

current_price = float(market_df["y"].iloc[-1])

# ============================================================
# NEWS COLLECTION (GOOGLE NEWS RSS)
# ============================================================
@st.cache_data(ttl=900)
def fetch_news(company, ticker):
    query = re.sub(r"[^A-Za-z0-9 ]", "", f"{company} {ticker} stock")
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = feedparser.parse(url)
    texts = []
    for e in feed.entries[:10]:
        texts.append(
            f"{getattr(e,'title','')}. {getattr(e,'summary','')}"
        )
    return texts

news_texts = fetch_news(company_name, ticker)

# ============================================================
# GEMINI SENTIMENT ENGINE (MANDATORY)
# ============================================================
@st.cache_data(ttl=900)
def gemini_sentiment(texts):
    scores = []
    for text in texts[:6]:
        try:
            prompt = (
                "Return ONLY one integer between -100 and 100.\n\n"
                f"{text[:1500]}"
            )
            resp = GEMINI_MODEL.generate_content(prompt)
            m = re.search(r"-?\d+", resp.text)
            scores.append(int(m.group()) if m else 0)
        except Exception:
            scores.append(0)
    return float(np.mean(scores)) if scores else 0.0

with st.spinner("Analyzing sentiment using Gemini..."):
    sentiment_score = gemini_sentiment(news_texts)

# ============================================================
# FORECASTING (PROPHET → ARIMA)
# ============================================================
with st.spinner("Building forecast model..."):
    forecast_price = None
    model_used = None

    if PROPHET_AVAILABLE:
        try:
            p = Prophet(daily_seasonality=True)
            p.fit(market_df)
            future = p.make_future_dataframe(periods=horizon)
            fc = p.predict(future).tail(horizon)
            forecast_price = float(fc["yhat"].mean())
            model_used = "Prophet"
        except Exception:
            PROPHET_AVAILABLE = False

    if not PROPHET_AVAILABLE:
        series = market_df["y"]
        arima = ARIMA(series, order=(1, 1, 1))
        fit = arima.fit()
        pred = fit.forecast(horizon)
        forecast_price = float(pred.mean())
        model_used = "ARIMA"

pct_change = ((forecast_price - current_price) / current_price) * 100

# ============================================================
# SIGNAL ENGINE
# ============================================================
if pct_change > 3 and sentiment_score > 20:
    signal = "STRONG BUY"
elif pct_change > 1:
    signal = "BUY"
elif pct_change < -3 and sentiment_score < -20:
    signal = "STRONG SELL"
elif pct_change < -1:
    signal = "SELL"
else:
    signal = "HOLD"

# ============================================================
# DASHBOARD
# ============================================================
st.title("Infosys InsightSphere")
st.caption("Real-Time Industry Insight & Strategic Intelligence Dashboard")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${current_price:.2f}")
c2.metric("Forecast Avg", f"${forecast_price:.2f}")
c3.metric("Projected Move", f"{pct_change:+.2f}%")
c4.metric("Sentiment", f"{sentiment_score:+.1f}")
c5.metric("Model Used", model_used)

st.subheader(f"Signal: {signal}")

fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=market_df["ds"], y=market_df["y"], name="Historical Price"))
fig.update_layout(height=420)
st.plotly_chart(fig, use_container_width=True)

# ============================================================
# EXECUTIVE SUMMARY (GEMINI)
# ============================================================
summary_prompt = f"""
You are an enterprise strategic intelligence analyst.

Company: {company_name}
Ticker: {ticker}
Current Price: {current_price}
Forecast Price: {forecast_price}
Projected Change: {pct_change:+.2f}%
Sentiment Score: {sentiment_score:+.1f}
Signal: {signal}

Provide:
1) Executive summary (2 sentences)
2) Key drivers (3 bullets)
3) Risks (3 bullets)
4) Opportunities (3 bullets)
5) Recommended action (1 paragraph)
"""

with st.spinner("Generating executive insights..."):
    executive_summary = GEMINI_MODEL.generate_content(summary_prompt).text

st.subheader("Executive Strategic Summary")
st.write(executive_summary)

# ============================================================
# SLACK ALERT
# ============================================================
if SLACK_WEBHOOK_URL and st.button("Send Slack Alert"):
    msg = (
        f"*InsightSphere Alert*\n"
        f"{company_name} ({ticker})\n"
        f"Signal: {signal}\n"
        f"Price: ${current_price:.2f} → ${forecast_price:.2f}\n"
        f"Move: {pct_change:+.2f}%\n"
        f"Sentiment: {sentiment_score:+.1f}"
    )
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": msg}, timeout=10)
        st.success("✅ Slack alert sent successfully.")
    except Exception as e:
        st.error(f"Slack alert failed: {e}")

st.caption("© 2025 Infosys Springboard Internship — Real-Time Strategic Intelligence System")
