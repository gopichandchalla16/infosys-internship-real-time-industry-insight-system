import os
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

# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# SECRETS (MANDATORY)
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is mandatory. Add it in Streamlit Secrets.")
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
# MARKET DATA (YFINANCE – SAFE)
# ============================================================
@st.cache_data(ttl=600)
def fetch_market_data(symbol):
    df = yf.download(symbol, period="1y", progress=False)
    if df is None or df.empty:
        return None
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["ds", "y"]
    return df

with st.spinner("Fetching market data..."):
    market_df = fetch_market_data(ticker)

if market_df is None:
    st.error(f"❌ No historical data found for ticker '{ticker}'.")
    st.stop()

current_price = float(market_df["y"].iloc[-1])

# ============================================================
# NEWS COLLECTION (GOOGLE NEWS RSS – SAFE)
# ============================================================
@st.cache_data(ttl=900)
def fetch_news(company, ticker):
    query = re.sub(r"[^A-Za-z0-9 ]", "", f"{company} {ticker} stock")
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    feed = feedparser.parse(url)
    texts = []
    for e in feed.entries[:10]:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        texts.append(f"{title}. {summary}")
    return texts

news_texts = fetch_news(company_name, ticker)

# ============================================================
# GEMINI SENTIMENT (MANDATORY)
# ============================================================
@st.cache_data(ttl=900)
def gemini_sentiment(texts):
    scores = []
    for text in texts[:6]:
        try:
            prompt = (
                "You are a financial sentiment model. "
                "Return ONLY an integer between -100 and 100.\n\n"
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
forecast_price = None
model_used = None

with st.spinner("Building forecast..."):
    if PROPHET_AVAILABLE:
        try:
            model = Prophet(daily_seasonality=True)
            model.fit(market_df)
            future = model.make_future_dataframe(periods=horizon)
            forecast = model.predict(future).tail(horizon)
            forecast_price = float(forecast["yhat"].mean())
            model_used = "Prophet"
        except Exception:
            PROPHET_AVAILABLE = False

    if not PROPHET_AVAILABLE:
        try:
            series = market_df["y"]
            arima = ARIMA(series, order=(1, 1, 1))
            fit = arima.fit()
            pred = fit.forecast(horizon)
            forecast_price = float(pred.mean())
            model_used = "ARIMA"
        except Exception:
            st.error("❌ Forecasting failed.")
            st.stop()

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
# DASHBOARD UI
# ============================================================
st.title("Infosys InsightSphere")
st.caption("Real-Time Industry Insight & Strategic Intelligence Dashboard")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Current Price", f"${current_price:.2f}")
k2.metric("Forecast Avg", f"${forecast_price:.2f}")
k3.metric("Projected Move", f"{pct_change:+.2f}%")
k4.metric("Sentiment", f"{sentiment_score:+.1f}")
k5.metric("Model", model_used)

st.subheader(f"Signal: {signal}")

# ============================================================
# PRICE CHART
# ============================================================
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(
    x=market_df["ds"],
    y=market_df["y"],
    name="Historical Price"
))
fig.update_layout(height=400)
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
5) Recommended action (1 short paragraph)
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
