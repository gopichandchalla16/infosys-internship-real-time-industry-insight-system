import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# ---------------- Prophet (Primary) ----------------
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
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric {background:#0f172a; padding:16px; border-radius:14px; text-align:center;}
.metric-title {color:#94a3b8; font-size:0.9rem;}
.metric-value {color:#e5e7eb; font-size:1.5rem; font-weight:700;}
.badge {padding:8px 16px; border-radius:12px; font-weight:700; font-size:1.1rem;}
.section {background:#020617; padding:20px; border-radius:18px; margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SLACK_WEBHOOK = st.secrets.get("SLACK_WEBHOOK_URL", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is mandatory.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🔎 Asset Selection")
    user_input = st.text_input(
        "Company / Crypto / Ticker",
        value="Tesla",
        help="Examples: Tesla, AAPL, TSLA, BTC, ETH"
    )
    horizon = st.slider("Forecast Horizon (Days)", 3, 14, 7)
    run = st.button("🚀 Run Analysis", use_container_width=True)

if not run:
    st.info("Enter an asset and click **Run Analysis**.")
    st.stop()


# ============================================================
# SYMBOL RESOLUTION
# ============================================================
def resolve_symbol(query: str):
    crypto = {
        "btc": "BTC-USD", "bitcoin": "BTC-USD",
        "eth": "ETH-USD", "ethereum": "ETH-USD"
    }
    q = query.lower().strip()
    if q in crypto:
        return crypto[q], "Crypto"

    # Try direct ticker
    try:
        df = yf.download(query.upper(), period="5d", progress=False)
        if not df.empty:
            return query.upper(), "Equity"
    except:
        pass

    # Yahoo search
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.replace(' ', '+')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code == 200:
            for item in r.json().get("quotes", []):
                if item.get("quoteType") in ["EQUITY", "CRYPTOCURRENCY"]:
                    return item["symbol"], item["quoteType"].title()
    except:
        pass

    return None, None


symbol, asset_type = resolve_symbol(user_input)
if not symbol:
    st.error("❌ Could not resolve asset.")
    st.stop()

st.success(f"Resolved: **{symbol}** ({asset_type})")


# ============================================================
# MARKET DATA (ROBUST + SAFE FALLBACK)
# ============================================================
@st.cache_data(ttl=600)
def fetch_market(symbol):
    try:
        df = yf.download(symbol, period="1y", auto_adjust=True, progress=False)
        if not df.empty and len(df) > 60:
            df = df.reset_index()[["Date", "Close"]]
            df.columns = ["ds", "y"]
            return df, "Yahoo Finance"
    except:
        pass
    return None, "Unavailable"


market_df, source = fetch_market(symbol)

# -------- FINAL SAFETY NET (NO CRASH EVER) --------
if market_df is None:
    st.warning(
        "⚠️ Live market data temporarily unavailable (provider rate limits). "
        "Showing demo data for analytical demonstration."
    )

    dates = pd.date_range(end=datetime.today(), periods=180)
    prices = np.linspace(420, 460, len(dates)) + np.random.normal(0, 2, len(dates))

    market_df = pd.DataFrame({
        "ds": dates.date,
        "y": prices
    })
    source = "Demo Fallback"


current_price = float(market_df["y"].iloc[-1])
st.caption(f"Data source: {source} • {len(market_df)} days")


# ============================================================
# COMPANY PROFILE
# ============================================================
company_info = {"name": user_input}

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


# ============================================================
# NEWS SENTIMENT (GEMINI)
# ============================================================
@st.cache_data(ttl=1800)
def gemini_sentiment(query):
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
    try:
        feed = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).text
        import feedparser
        feed = feedparser.parse(feed)

        scores = []
        for e in feed.entries[:6]:
            text = f"{e.title} {getattr(e,'summary','')}"
            prompt = (
                "Rate sentiment from -100 to +100 as an integer only:\n\n"
                f"{text[:1000]}"
            )
            try:
                r = GEMINI_MODEL.generate_content(prompt)
                m = re.search(r"-?\d+", r.text)
                scores.append(int(m.group()) if m else 0)
            except:
                scores.append(0)

        return float(np.mean(scores)) if scores else 0.0
    except:
        return 0.0


sentiment = gemini_sentiment(user_input)


# ============================================================
# FORECASTING (PROPHET → ARIMA)
# ============================================================
try:
    if PROPHET_AVAILABLE:
        m = Prophet(daily_seasonality=False, weekly_seasonality=True)
        m.fit(market_df.rename(columns={"ds": "ds", "y": "y"}))
        future = m.make_future_dataframe(periods=horizon)
        fc = m.predict(future)
        forecast_price = fc["yhat"].tail(horizon).mean()
        model_used = "Prophet"
    else:
        raise Exception
except:
    series = market_df["y"]
    try:
        model = ARIMA(series, order=(5, 1, 0))
        fit = model.fit()
        forecast_price = fit.forecast(horizon).mean()
        model_used = "ARIMA"
    except:
        forecast_price = current_price
        model_used = "Static"


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

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${current_price:.2f}")
c2.metric("Forecast Avg", f"${forecast_price:.2f}")
c3.metric("Projected Move", f"{pct_change:+.2f}%")
c4.metric("Sentiment", f"{sentiment:+.1f}")
c5.metric("Model", model_used)

st.markdown(
    f"<div class='badge' style='background:{color}; color:white;'>{signal}</div>",
    unsafe_allow_html=True
)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=market_df["ds"],
    y=market_df["y"],
    name="Price",
    line=dict(color="#3b82f6")
))
fig.update_layout(height=450, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)


# ============================================================
# COMPANY OVERVIEW
# ============================================================
if asset_type == "Equity":
    st.subheader("🏢 Company Overview")
    st.write(
        f"**Sector:** {company_info.get('sector','N/A')} | "
        f"**Industry:** {company_info.get('industry','N/A')} | "
        f"**Country:** {company_info.get('country','N/A')}"
    )
    if company_info.get("website"):
        st.write(f"**Website:** {company_info['website']}")

    if WIKI_AVAILABLE:
        try:
            st.info(wikipedia.summary(company_info["name"], sentences=4))
        except:
            pass


# ============================================================
# EXECUTIVE SUMMARY (GEMINI)
# ============================================================
st.subheader("📝 Executive Strategic Summary")
try:
    prompt = f"""
    Provide an executive market intelligence summary for:
    {company_info['name']} ({symbol})

    Current Price: {current_price}
    Forecast ({horizon}d): {forecast_price:.2f} ({pct_change:+.2f}%)
    Sentiment: {sentiment:+.1f}
    Signal: {signal}

    Include key drivers, risks, opportunities, and recommendation.
    """
    summary = GEMINI_MODEL.generate_content(prompt).text
    st.write(summary)
except:
    st.warning("AI summary temporarily unavailable.")


# ============================================================
# SLACK ALERT
# ============================================================
if SLACK_WEBHOOK and st.button("📤 Send Slack Alert"):
    payload = {
        "text": (
            f"*InsightSphere Alert*\n"
            f"{company_info['name']} ({symbol})\n"
            f"Signal: {signal}\n"
            f"Price: ${current_price:.2f} → ${forecast_price:.2f}\n"
            f"Sentiment: {sentiment:+.1f}"
        )
    }
    try:
        requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
        st.success("Slack alert sent.")
    except:
        st.error("Slack delivery failed.")

st.caption("© 2025 Infosys Springboard Internship — Real-Time Strategic Intelligence System")
