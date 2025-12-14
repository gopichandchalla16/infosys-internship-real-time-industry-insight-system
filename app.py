import re
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

# ---------- Optional Prophet ----------
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ---------- Gemini ----------
import google.generativeai as genai

# ---------- Wikipedia ----------
try:
    import wikipedia
    WIKI_AVAILABLE = True
except Exception:
    WIKI_AVAILABLE = False


# ============================================================
# PAGE CONFIG + UI
# ============================================================
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.metric {background:#0f172a;padding:16px;border-radius:14px;text-align:center;}
.metric h3{color:#94a3b8;font-size:0.9rem;margin-bottom:4px;}
.metric h2{color:#e5e7eb;font-size:1.4rem;}
.badge{padding:8px 18px;border-radius:14px;font-weight:700;color:white;}
.card{background:#020617;padding:22px;border-radius:18px;margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SECRETS
# ============================================================
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY is required in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")  # Most stable


# ============================================================
# SIDEBAR (USER FRIENDLY)
# ============================================================
with st.sidebar:
    st.header("🔎 Asset Selection")
    user_query = st.text_input(
        "Company name or Ticker",
        value="Tesla",
        help="Examples: Tesla, Apple, Reliance, TSLA, AAPL, BTC"
    )
    horizon = st.slider("Forecast Horizon (Days)", 3, 14, 7)
    run = st.button("🚀 Run Analysis", use_container_width=True)

if not run:
    st.info("Enter a company name or ticker and click **Run Analysis**.")
    st.stop()


# ============================================================
# TICKER RESOLUTION (NO FAKE COMPANIES)
# ============================================================
def resolve_symbol(query: str):
    crypto = {"btc": "BTC-USD", "bitcoin": "BTC-USD", "eth": "ETH-USD"}
    q = query.lower().strip()

    if q in crypto:
        return crypto[q], "Crypto"

    # Direct ticker check
    try:
        df = yf.download(query.upper(), period="5d", progress=False)
        if not df.empty:
            return query.upper(), "Equity"
    except:
        pass

    # Yahoo search
    try:
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query.replace(' ', '+')}"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        data = r.json()
        for item in data.get("quotes", []):
            if item.get("quoteType") in ["EQUITY", "CRYPTOCURRENCY"]:
                return item.get("symbol"), item.get("quoteType").title()
    except:
        pass

    return None, None


symbol, asset_type = resolve_symbol(user_query)
if not symbol:
    st.error("❌ Could not identify a real company or asset. Please try a valid name.")
    st.stop()

st.success(f"Resolved asset: **{symbol}** ({asset_type})")


# ============================================================
# MARKET DATA (ROBUST + DEMO FALLBACK)
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

    # Demo fallback (guaranteed)
    dates = pd.date_range(end=datetime.today(), periods=180)
    prices = np.cumsum(np.random.normal(0, 1, 180)) + 100
    demo = pd.DataFrame({"ds": dates, "y": prices})
    return demo, "Demo Fallback"


market_df, source = fetch_market(symbol)
current_price = float(market_df["y"].iloc[-1])

if source == "Demo Fallback":
    st.warning(
        "⚠️ Live market data temporarily unavailable (provider rate limits). "
        "Showing **demo data** for analytical demonstration."
    )

st.caption(f"Data source: **{source}** • {len(market_df)} days")


# ============================================================
# COMPANY PROFILE (NEVER EMPTY)
# ============================================================
company = {
    "name": user_query,
    "sector": "N/A",
    "industry": "N/A",
    "country": "N/A",
    "website": "N/A"
}

if asset_type == "Equity":
    try:
        info = yf.Ticker(symbol).info
        company.update({
            "name": info.get("longName", user_query),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "country": info.get("country", "N/A"),
            "website": info.get("website", "N/A")
        })
    except:
        pass

    if WIKI_AVAILABLE:
        try:
            wiki_page = wikipedia.page(company["name"])
            company["website"] = company["website"] if company["website"] != "N/A" else wiki_page.url
            company["country"] = company["country"] if company["country"] != "N/A" else "Public Company"
            wiki_summary = wikipedia.summary(company["name"], sentences=4)
        except:
            wiki_summary = "Wikipedia summary unavailable."
else:
    wiki_summary = ""


# ============================================================
# NEWS SENTIMENT (RATE SAFE)
# ============================================================
@st.cache_data(ttl=1800)
def gemini_sentiment(query):
    try:
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
        feed = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
        import feedparser
        feed = feedparser.parse(feed)

        scores = []
        for e in feed.entries[:5]:
            prompt = f"Return sentiment score -100 to +100 as integer:\n{e.title}"
            try:
                r = GEMINI_MODEL.generate_content(prompt)
                m = re.search(r"-?\d+", r.text)
                scores.append(int(m.group()) if m else 0)
            except:
                scores.append(0)
        return float(np.mean(scores)) if scores else 0.0
    except:
        return 0.0


sentiment = gemini_sentiment(user_query)


# ============================================================
# FORECASTING (PROPHET → ARIMA)
# ============================================================
try:
    if PROPHET_AVAILABLE and source != "Demo Fallback":
        m = Prophet()
        dfp = market_df.rename(columns={"ds": "ds", "y": "y"})
        m.fit(dfp)
        future = m.make_future_dataframe(periods=horizon)
        forecast = m.predict(future)
        forecast_price = forecast["yhat"].tail(horizon).mean()
        model_used = "Prophet"
    else:
        raise Exception
except:
    model = ARIMA(market_df["y"], order=(5, 1, 0))
    fit = model.fit()
    pred = fit.forecast(horizon)
    forecast_price = float(pred.mean())
    model_used = "ARIMA"

pct_change = ((forecast_price - current_price) / current_price) * 100


# ============================================================
# SIGNAL
# ============================================================
if pct_change > 3 and sentiment > 15:
    signal, color = "STRONG BUY", "#16a34a"
elif pct_change > 1:
    signal, color = "BUY", "#22c55e"
elif pct_change < -3 and sentiment < -15:
    signal, color = "STRONG SELL", "#dc2626"
elif pct_change < -1:
    signal, color = "SELL", "#ef4444"
else:
    signal, color = "HOLD", "#eab308"


# ============================================================
# DASHBOARD
# ============================================================
st.title("📊 Infosys InsightSphere")
st.subheader(f"{company['name']} ({symbol}) • {asset_type}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current Price", f"${current_price:.2f}")
c2.metric("Forecast Avg", f"${forecast_price:.2f}")
c3.metric("Projected Move", f"{pct_change:+.2f}%")
c4.metric("Sentiment", f"{sentiment:+.1f}")
c5.metric("Model", model_used)

st.markdown(f"<div class='badge' style='background:{color}'>{signal}</div>", unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=market_df["ds"], y=market_df["y"], name="Price"))
fig.update_layout(height=420, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

if asset_type == "Equity":
    st.subheader("🏢 Company Overview")
    st.write(
        f"**Sector:** {company['sector']} | "
        f"**Industry:** {company['industry']} | "
        f"**Country:** {company['country']}"
    )
    if company["website"] != "N/A":
        st.write(f"**Website:** {company['website']}")
    st.info(wiki_summary)


# ============================================================
# EXECUTIVE SUMMARY (SAFE)
# ============================================================
fallback = f"""
**Executive Overview**

{company['name']} is currently trading at ${current_price:.2f}.
The {horizon}-day outlook suggests a {pct_change:+.2f}% move.
Overall signal: **{signal}** based on technical and sentiment indicators.
"""

try:
    prompt = f"""
Provide executive summary for {company['name']} ({symbol})
Price: {current_price}
Forecast: {forecast_price}
Sentiment: {sentiment}
Signal: {signal}
"""
    summary = GEMINI_MODEL.generate_content(prompt).text
except:
    summary = fallback

st.subheader("📝 Executive Strategic Summary")
st.write(summary)

# ============================================================
# SLACK ALERT
# ============================================================
if SLACK_WEBHOOK_URL and st.button("📤 Send Slack Alert"):
    payload = {"text": f"{company['name']} ({symbol})\nSignal: {signal}\nPrice: ${current_price:.2f}"}
    requests.post(SLACK_WEBHOOK_URL, json=payload)

st.caption("© 2025 Infosys Springboard Internship — Real-Time Strategic Intelligence System")
