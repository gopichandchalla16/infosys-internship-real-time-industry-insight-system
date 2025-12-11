import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import google.generativeai as genai
import requests

# ========================= CONFIG =========================
st.set_page_config(page_title="InsightSphere", layout="wide", page_icon="rocket")

# ========================= SECRETS =========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SLACK_WEBHOOK = st.secrets["SLACK_WEBHOOK_URL"]
except:
    st.error("Add GEMINI_API_KEY and SLACK_WEBHOOK_URL in Streamlit Secrets")
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

# ========================= UI =========================
st.markdown("""
<style>
    .title {font-size: 3.6rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.4rem; margin-bottom: 2rem;}
    .signal {text-align: center; padding: 2rem; border-radius: 25px; font-size: 2.8rem; font-weight: bold; color: white;}
    .card {background: #1e293b; padding: 1.8rem; border-radius: 18px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.5);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Infosys InsightSphere</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Strategic Intelligence • Executive Dashboard</div>', unsafe_allow_html=True)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", value="AAPL", help="TSLA, AAPL, MSFT, RELIANCE.NS").upper().strip()
    company = st.text_input("Company Name (optional)", value="Apple")
    days = st.slider("Forecast Horizon (Days)", 3, 21, 7)
    use_prophet = st.checkbox("Use Prophet Model", value=True)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ========================= BULLETPROOF YFINANCE (THIS FIXES EVERYTHING) =========================
@st.cache_data(ttl=600, show_spinner="Fetching real-time data...")
def get_stock_data(ticker):
    # Critical: Proper headers to bypass Yahoo blocking
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    try:
        # Method 1: Direct download with session
        df = yf.download(ticker, period="1y", auto_adjust=True, progress=False, session=session)
        if df.empty or len(df) < 30:
            raise ValueError("Empty data")
        
        t = yf.Ticker(ticker, session=session)
        info = t.info
        price = info.get("currentPrice") or info.get("regularMarketPrice") or df["Close"].iloc[-1]
        cap = info.get("marketCap")
        name = info.get("longName") or info.get("shortName") or company
        
        df = df.reset_index()
        return df, price, cap, name
    
    except Exception as e:
        st.error(f"Data fetch failed. Trying fallback...")
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="1y", auto_adjust=True)
            if hist.empty:
                return None, None, None, None
            info = t.info
            price = info.get("currentPrice") or hist["Close"].iloc[-1]
            return hist.reset_index(), price, info.get("marketCap"), info.get("longName", company)
        except:
            return None, None, None, None

# Run fetch
result = get_stock_data(ticker)
if result[0] is None:
    st.error(f"Invalid ticker: {ticker}\n\nTry: AAPL, TSLA, MSFT, RELIANCE.NS, GOOGL")
    st.stop()

hist_df, price, market_cap, company_name = result

# ========================= SENTIMENT =========================
@st.cache_data(ttl=1800)
def get_sentiment(comp, tkr):
    url = f"https://news.google.com/rss/search?q={comp}+{tkr}+stock&hl=en"
    feed = feedparser.parse(url)
    texts = [f"{e.title}. {e.get('summary','')}" for e in feed.entries[:10]]
    
    scores = []
    for text in texts:
        try:
            r = model.generate_content(f"Rate sentiment -100 to +100. Return ONLY the number.\n\n{text[:1300]}")
            import re
            num = re.search(r'-?\d+', r.text)
            scores.append(int(num.group()) if num else 0)
        except:
            scores.append(0)
    return round(np.mean(scores), 1) if scores else 0

with st.spinner("Analyzing news with Gemini..."):
    sentiment = get_sentiment(company_name, ticker)

# ========================= FORECAST =========================
def forecast(df, days):
    df2 = df[["Date", "Close"]].copy()
    df2.columns = ["ds", "y"]
    df2["ds"] = pd.to_datetime(df2["ds"])

    if use_prophet:
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m.fit(df2)
            future = m.make_future_dataframe(periods=days)
            fc = m.predict(future)
            return fc.tail(days), "Prophet"
        except:
            pass

    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(df2["y"], trend="add")
    fit = model.fit()
    pred = fit.forecast(days)
    dates = pd.date_range(df2["ds"].iloc[-1] + timedelta(1), periods=days)
    res = pd.DataFrame({"ds": dates, "yhat": pred})
    res["yhat_lower"] = res["yhat"] * 0.92
    res["yhat_upper"] = res["yhat"] * 1.08
    return res, "Light Model"

fc_df, model_used = forecast(hist_df, days)
forecast_price = round(fc_df["yhat"].mean(), 2)
pct_change = round((forecast_price - price) / price * 100, 2)

# ========================= SIGNAL =========================
if pct_change >= 3 and sentiment >= 20:
    signal, color = "STRONG BUY", "#00ff88"
elif pct_change >= 1.5 and sentiment >= 5:
    signal, color = "BUY", "#00cc66"
elif pct_change <= -3 and sentiment <= -20:
    signal, color = "STRONG SELL", "#ff0033"
elif pct_change <= -1.5 and sentiment <= -5:
    signal, color = "SELL", "#ff4444"
else:
    signal, color = "HOLD", "#ffaa00"

# ========================= DASHBOARD =========================
c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 2])
with c1:
    st.markdown(f"<div class='card'><h3>{company_name}</h3><h1>${price:.2f}</h1></div>", unsafe_allow_html=True)
with c2:
    cap_str = f"${market_cap/1e12:.2f}T" if market_cap and market_cap >= 1e12 else f"${market_cap/1e9:.2f}B" if market_cap else "N/A"
    st.markdown(f"<div class='card'><h4>Market Cap</h4><h2>{cap_str}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='card'><h4>Sentiment</h4><h2>{sentiment:+.0f}</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='signal' style='background:{color}'>SIGNAL: {signal}</div>", unsafe_allow_html=True)

# Plotly Dashboard
fig = make_subplots(rows=2, cols=2,
    subplot_titles=("Price & Forecast", "Sentiment Gauge", f"{days}-Day Forecast", "Summary"),
    specs=[[{"type": "scatter"}, {"type": "indicator"}], [{"type": "bar"}, {"type": "table"}]]
)

fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Close"], name="Price", line=dict(color="#60a5fa", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast", line=dict(color="#8b5cf6", width=4, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(
    x=pd.concat([fc_df["ds"], fc_df["ds"][::-1]]),
    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(139,92,246,0.2)", line=dict(color="transparent"), showlegend=False
), row=1, col=1)

fig.add_trace(go.Indicator(mode="gauge+number", value=sentiment,
    gauge={'axis': {'range': [-100,100]}, 'bar': {'color': "purple"}},
    title={'text': "Sentiment"}), row=1, col=2)

fig.add_trace(go.Bar(x=fc_df["ds"].dt.strftime("%b %d"), y=fc_df["yhat"], marker_color="#8b5cf6"), row=2, col=1)

fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"], fill_color="#1e293b", font=dict(color="white")),
    cells=dict(values=[
        ["Company", "Ticker", "Price", "Forecast", "Change", "Sentiment", "Signal"],
        [company_name, ticker, f"${price:.2f}", f"${forecast_price:.2f}", f"{pct_change:+.2f}%", f"{sentiment:+.1f}", signal]
    ])
), row=2, col=2)

fig.update_layout(height=1000, showlegend=False, template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# ========================= SLACK =========================
if st.button("Send Alert to Slack", type="primary", use_container_width=True):
    msg = f"*InsightSphere Alert*\n*{company_name} ({ticker})*\nSignal: {signal}\nPrice: ${price:.2f} to ${forecast_price:.2f} ({pct_change:+.2f}%)\nSentiment: {sentiment:+.1f}"
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg})
        st.success("Alert sent!")
    except:
        st.error("Slack failed")

st.caption("© 2025 Infosys Springboard • Team: Gopichand, Anshika, Janmejay, Vaishnavi")
