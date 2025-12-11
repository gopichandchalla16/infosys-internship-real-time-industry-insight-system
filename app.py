import os
import re
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import feedparser
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional Gemini import handled safely
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# -------------------- Page config --------------------
st.set_page_config(page_title="InsightSphere", layout="wide", initial_sidebar_state="expanded")

# -------------------- CSS / Premium UI (Apple-style glass) --------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .glass {
        background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 30px rgba(2,6,23,0.5);
    }
    .metric {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
    }
    .title {
        font-size: 2.6rem;
        font-weight: 700;
        margin-bottom: 2px;
        letter-spacing: -0.02em;
    }
    .subtitle {
        color: rgba(255,255,255,0.72);
        margin-bottom: 12px;
    }
    /* Light theme overrides */
    .light .glass { background: linear-gradient(135deg, rgba(255,255,255,0.85), rgba(255,255,255,0.95)); border: 1px solid rgba(15,23,42,0.04); box-shadow: 0 6px 18px rgba(2,6,23,0.06); }
    .light .metric { background: rgba(0,0,0,0.03); }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Secrets (do not commit) --------------------
# Use Streamlit > Settings > Secrets for keys, or set environment variables
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else os.getenv("GEMINI_API_KEY", "")
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "") if hasattr(st, "secrets") else os.getenv("SLACK_WEBHOOK_URL", "")

# Configure Gemini if available
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        gen_model = None
else:
    gen_model = None

# -------------------- Sidebar controls --------------------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="Examples: AAPL, TSLA, MSFT, RELIANCE.NS").upper().strip()
    company_override = st.text_input("Company name (optional)", value="").strip()
    forecast_days = st.slider("Forecast horizon (days)", min_value=3, max_value=21, value=7)
    theme = st.selectbox("Theme", ["Dark (default)", "Light"])
    st.markdown("---")
    st.markdown("Integrations (optional)")
    st.write(f"- Gemini: {'Enabled' if gen_model is not None else 'Not configured'}")
    st.write(f"- Slack: {'Configured' if SLACK_WEBHOOK_URL else 'Not configured'}")
    if st.button("Run analysis", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Toggle light CSS class if Light chosen
if theme.startswith("Light"):
    st.markdown("<div class='light'>", unsafe_allow_html=True)
else:
    st.markdown("<div>", unsafe_allow_html=True)

# -------------------- Utility functions --------------------
INVALID_COMPANY_KEYWORDS = {"abc", "xyz", "test", "testing", "demo", "sample", "fake", "dummy", "123", "qwerty", "asdf"}

def is_invalid_company_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return True
    s = name.strip().lower()
    if s.isnumeric() or len(s) < 2:
        return True
    for bad in INVALID_COMPANY_KEYWORDS:
        if bad in s:
            return True
    return False

# Robust yfinance fetch that uses a requests session and falls back gracefully
@st.cache_data(ttl=600, show_spinner=False)
def fetch_historical(ticker: str, period: str = "1y"):
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    try:
        # Primary method
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True, session=session)
        if df is None or df.empty:
            raise ValueError("No data")
        df = df.reset_index()
        # normalize date column
        if "Date" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"Date": "date"})
        # Extract market info
        t = yf.Ticker(ticker, session=session)
        try:
            info = t.fast_info or {}
        except Exception:
            info = {}
        price = info.get("last_price", float(df["Close"].iloc[-1]))
        market_cap = info.get("market_cap", None)
        name = info.get("long_name") or info.get("short_name") or ticker
        return df, price, market_cap, name
    except Exception:
        # fallback without session
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period=period, auto_adjust=True)
            if hist is None or hist.empty:
                return None, None, None, None
            hist = hist.reset_index()
            if "Date" in hist.columns and "date" not in hist.columns:
                hist = hist.rename(columns={"Date": "date"})
            info = {}
            try:
                info = t.fast_info or {}
            except Exception:
                info = {}
            price = info.get("last_price", float(hist["Close"].iloc[-1]))
            market_cap = info.get("market_cap", None)
            name = info.get("long_name") or ticker
            return hist, price, market_cap, name
        except Exception:
            return None, None, None, None

# -------------------- Fetch & validate --------------------
if not ticker:
    st.error("Please provide a ticker symbol.")
    st.stop()

hist_df, current_price, market_cap, detected_name = fetch_historical(ticker, period="1y")
if hist_df is None:
    st.error(f"Could not fetch historical data for ticker '{ticker}'. Try a different ticker (AAPL, TSLA, MSFT, RELIANCE.NS, GOOGL).")
    st.stop()

company_name = company_override if company_override else (detected_name or ticker)

# -------------------- Sentiment: Gemini (optional) + local fallback --------------------
POS_WORDS = ["growth", "strong", "bullish", "positive", "optimistic", "profit", "surge", "beats", "outperform", "record"]
NEG_WORDS = ["weak", "bearish", "loss", "regulatory", "lawsuit", "slowing", "concern", "fraud", "volatility", "drop"]

def local_sentiment(text: str) -> float:
    if not text:
        return 0.0
    t = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in t:
            score += 10
    for w in NEG_WORDS:
        if w in t:
            score -= 10
    return float(max(-100, min(100, score)))

@st.cache_data(ttl=900, show_spinner=False)
def compute_sentiment(company: str, ticker: str) -> float:
    # build safe Google News RSS query; ensure no control characters
    q = f"{company} {ticker} stock".strip()
    q_safe = re.sub(r'[^A-Za-z0-9 \-_.]', '', q)
    url = f"https://news.google.com/rss/search?q={q_safe.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"

    try:
        feed = feedparser.parse(url)
    except Exception:
        return 0.0

    texts = []
    for entry in feed.entries[:8]:
        title = getattr(entry, "title", "") or ""
        summary = getattr(entry, "summary", "") or ""
        texts.append(f"{title}. {summary}")

    if not texts:
        return 0.0

    scores = []
    for txt in texts:
        if gen_model is not None:
            try:
                prompt = f"Rate the sentiment of the following news text from -100 (very negative) to 100 (very positive). Return a single integer.\n\nText:\n{txt[:1600]}"
                resp = gen_model.generate_content(prompt)
                raw = getattr(resp, "text", "") or str(resp)
                m = re.search(r"-?\d+", raw)
                if m:
                    scores.append(int(m.group()))
                    continue
            except Exception:
                pass
        # fallback
        scores.append(int(local_sentiment(txt)))
    avg = float(np.mean(scores)) if scores else 0.0
    return float(max(-100.0, min(100.0, avg)))

with st.spinner("Analyzing news sentiment..."):
    agg_sentiment = compute_sentiment(company_name, ticker)

# -------------------- Forecasting (Holt-Winters) --------------------
def arima_like_forecast(df: pd.DataFrame, periods: int = 7):
    # We use Exponential Smoothing (Holt-Winters) — fast and stable for short horizons
    s = df.copy()
    if "date" not in s.columns and "Date" in s.columns:
        s = s.rename(columns={"Date": "date"})
    s["date"] = pd.to_datetime(s["date"])
    series = s["Close"].astype(float)
    # small safety: if too small, return None
    if len(series) < 10:
        return None
    model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
    fit = model.fit(optimized=True)
    pred = fit.forecast(periods)
    dates = pd.date_range(start=s["date"].iloc[-1] + timedelta(days=1), periods=periods, freq="D")
    out = pd.DataFrame({"date": dates, "yhat": pred})
    out["yhat_lower"] = out["yhat"] * 0.92
    out["yhat_upper"] = out["yhat"] * 1.08
    return out

forecast_df = arima_like_forecast(hist_df, forecast_days)
if forecast_df is None:
    st.error("Not enough historical data to produce a forecast.")
    st.stop()

proj_price = float(forecast_df["yhat"].mean())
proj_pct = ((proj_price - float(current_price)) / float(current_price)) * 100.0 if current_price else 0.0

# -------------------- Signal engine --------------------
def compute_signal(pct: float, sentiment_score: float):
    if pct > 3 and sentiment_score > 20:
        return {"signal": "STRONG BUY", "color": "#059669", "reason": f"Projected upside {pct:.2f}% and positive sentiment {sentiment_score:.1f}"}
    if pct > 1 and sentiment_score > 10:
        return {"signal": "BUY", "color": "#059669", "reason": f"Moderate upside {pct:.2f}% and supportive sentiment {sentiment_score:.1f}"}
    if pct < -3 and sentiment_score < -20:
        return {"signal": "STRONG SELL", "color": "#DC2626", "reason": f"Projected downside {pct:.2f}% and negative sentiment {sentiment_score:.1f}"}
    if pct < -1 and sentiment_score < -10:
        return {"signal": "SELL", "color": "#DC2626", "reason": f"Moderate downside {pct:.2f}% and negative sentiment {sentiment_score:.1f}"}
    return {"signal": "HOLD", "color": "#D97706", "reason": f"Mixed signals ({pct:.2f}%) and sentiment {sentiment_score:.1f}"}

signal_info = compute_signal(proj_pct, agg_sentiment)

# -------------------- UI: Header & KPIs --------------------
st.markdown(f"<div class='glass'><div class='title'>{company_name} — InsightSphere</div><div class='subtitle'>Real-Time Strategic Intelligence — Executive Dashboard</div></div>", unsafe_allow_html=True)
st.markdown("")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(f"<div class='metric'><div style='font-size:14px'>Current Price</div><div style='font-size:20px;font-weight:700'>${float(current_price):,.2f}</div></div>", unsafe_allow_html=True)
k2.markdown(f"<div class='metric'><div style='font-size:14px'>7-Day Forecast (avg)</div><div style='font-size:20px;font-weight:700'>${proj_price:,.2f}</div></div>", unsafe_allow_html=True)
k3.markdown(f"<div class='metric'><div style='font-size:14px'>Projected Move</div><div style='font-size:20px;font-weight:700'>{proj_pct:+.2f}%</div></div>", unsafe_allow_html=True)
k4.markdown(f"<div class='metric'><div style='font-size:14px'>Signal</div><div style='font-size:18px;font-weight:700;color:{signal_info['color']}'>{signal_info['signal']}</div></div>", unsafe_allow_html=True)

st.markdown(f"**Rationale:** {signal_info['reason']}  \n**Aggregate sentiment (news):** {agg_sentiment:+.1f}")

# -------------------- Plotly composite chart --------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08, subplot_titles=("Historical vs Forecast", "7-day Forecast"))
fig.add_trace(go.Scatter(x=hist_df["date"], y=hist_df["Close"], mode="lines", name="Historical"), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["yhat"], mode="lines+markers", name="Forecast", line=dict(dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]), y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]), fill="toself", fillcolor="rgba(0,102,204,0.12)", line=dict(color="rgba(0,0,0,0)"), showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=forecast_df["date"].dt.strftime("%Y-%m-%d"), y=forecast_df["yhat"], name="Forecast Price"), row=2, col=1)
fig.update_layout(height=780, template=("plotly_dark" if theme.startswith("Dark") else "plotly_white"))
st.plotly_chart(fig, use_container_width=True)

# -------------------- Gemini Insight Generator --------------------
def generate_gemini_insights(company: str, ticker: str, price: float, forecast_price: float, pct_change: float, sentiment: float) -> str:
    if gen_model is None:
        return "Gemini model not configured. Add GEMINI_API_KEY to Streamlit Secrets to enable AI insights."
    prompt = (
        "You are a senior strategic intelligence analyst for enterprise leadership.\n"
        "Produce a concise, executive-grade analysis in the following sections:\n"
        "1) Executive summary (2 sentences)\n"
        "2) Top 3 market drivers (bullet list)\n"
        "3) Top 3 risks/threats (bullet list)\n"
        "4) Top 3 strategic opportunities (bullet list)\n"
        "5) Recommended actions for C-suite (1 short paragraph)\n\n"
        f"Company: {company}\nTicker: {ticker}\nCurrent Price: {price}\n7-day avg forecast: {forecast_price}\nProjected % change: {pct_change:+.2f}%\nAggregate sentiment: {sentiment:+.1f}\n\n"
        "Be direct, concise, and use enterprise language. Use short bullets and numbered items where appropriate."
    )
    try:
        resp = gen_model.generate_content(prompt)
        text = getattr(resp, "text", None) or str(resp)
        return text.strip()
    except Exception as e:
        return f"Insight generation failed: {e}"

st.subheader("AI-Generated Strategic Insights")
with st.spinner("Generating strategic insights (Gemini)..."):
    insights_text = generate_gemini_insights(company_name, ticker, float(current_price), proj_price, proj_pct, agg_sentiment)
st.markdown(f"<div class='glass' style='padding:16px;white-space:pre-wrap'>{insights_text}</div>", unsafe_allow_html=True)

# -------------------- Slack alert --------------------
if SLACK_WEBHOOK_URL:
    if st.button("Send Alert to Slack", use_container_width=True):
        payload = {
            "text": (
                f"*InsightSphere Alert*\n*{company_name} ({ticker})*\n"
                f"Signal: {signal_info['signal']}\n"
                f"Price: ${float(current_price):.2f}  Forecast(avg): ${proj_price:.2f} ({proj_pct:+.2f}%)\n"
                f"Sentiment: {agg_sentiment:+.1f}\n\n"
                f"{insights_text[:800]}..."
            )
        }
        try:
            r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
            if r.status_code == 200:
                st.success("Alert sent to Slack.")
            else:
                st.error(f"Slack returned {r.status_code}: {r.text}")
        except Exception as e:
            st.error(f"Failed to send Slack alert: {e}")

st.caption("© 2025 Infosys Springboard Internship — Real-Time Strategic Intelligence Dashboard")
# close theme wrapper
st.markdown("</div>", unsafe_allow_html=True)
