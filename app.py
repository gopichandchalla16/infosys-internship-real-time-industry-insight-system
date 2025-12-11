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

# ========================= PAGE CONFIG =========================
st.set_page_config(page_title="InsightSphere", layout="wide", page_icon="chart_with_upwards_trend")

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
    .title {font-size: 3.5rem; font-weight: bold; text-align: center; background: linear-gradient(90deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.4rem; margin-bottom: 2rem;}
    .signal {text-align: center; padding: 1.8rem; border-radius: 25px; font-size: 2.5rem; font-weight: bold; color: white; margin: 2rem 0;}
    .card {background: #1e293b; padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 8px 25px rgba(0,0,0,0.5);}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Infosys InsightSphere</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Strategic Intelligence Dashboard</div>', unsafe_allow_html=True)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.markdown("### Controls")
    ticker = st.text_input("Stock Ticker", value="TSLA", help="e.g. TSLA, AAPL, RELIANCE.NS").upper().strip()
    company = st.text_input("Company Name (optional)", value="Tesla")
    days = st.slider("Forecast Horizon (Days)", 3, 21, 7)
    use_prophet = st.checkbox("Use Prophet Model (Recommended)", value=True)
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        # Clear cache for fresh data
        st.cache_data.clear()
        st.success("Refreshing all data...")
        st.rerun()

# ========================= DATA FETCH (FIXED!) =========================
@st.cache_data(ttl=600, show_spinner="Fetching stock data...")
def fetch_stock(tkr):
    try:
        ticker_obj = yf.Ticker(tkr)
        hist = ticker_obj.history(period="1y", auto_adjust=True)
        if hist.empty or len(hist) < 50:
            return None
        info = ticker_obj.info
        current_price = info.get("currentPrice") or info.get("regularMarketPrice") or hist["Close"].iloc[-1]
        market_cap = info.get("marketCap")
        name = info.get("longName") or info.get("shortName") or company
        return hist.reset_index(), current_price, market_cap, name
    except Exception as e:
        st.error(f"yfinance error: {e}")
        return None

# Run fetch
result = fetch_stock(ticker)
if result is None:
    st.error(f"Invalid or blocked ticker: {ticker}")
    st.stop()

hist_df, price, market_cap, company_name = result

# ========================= NEWS & SENTIMENT =========================
@st.cache_data(ttl=1800)
def get_news_and_sentiment(comp, tkr):
    url = f"https://news.google.com/rss/search?q={comp}+{tkr}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    texts = [f"{e.title}. {e.get('summary','')}".strip() for e in feed.entries[:12]]
    
    def score(text):
        try:
            r = model.generate_content(f"Rate sentiment from -100 (very negative) to +100 (very positive). Return ONLY the number.\n\n{text[:1400]}")
            import re
            num = re.search(r'-?\d+', r.text)
            return int(num.group()) if num else 0
        except:
            return 0
    
    scores = [score(t) for t in texts]
    return np.mean(scores) if scores else 0

with st.spinner("Gemini analyzing latest news sentiment..."):
    avg_sentiment = get_news_and_sentiment(company_name, ticker)

# ========================= FORECASTING =========================
def forecast(hist, days, prophet=True):
    df = hist[["Date", "Close"]].copy()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"])

    if prophet:
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(df)
            future = m.make_future_dataframe(periods=days)
            fc = m.predict(future)
            return fc.tail(days), "Prophet"
        except:
            pass

    # Lightweight fallback
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(df["y"], trend="add", damped_trend=True)
    fit = model.fit()
    pred = fit.forecast(days)
    dates = pd.date_range(start=df["ds"].iloc[-1] + timedelta(days=1), periods=days)
    res = pd.DataFrame({"ds": dates, "yhat": pred.values})
    res["yhat_lower"] = res["yhat"] * 0.92
    res["yhat_upper"] = res["yhat"] * 1.08
    return res, "Exponential Smoothing"

fc_df, model_used = forecast(hist_df, days, use_prophet)
forecast_price = round(fc_df["yhat"].mean(), 2)
pct_change = round((forecast_price - price) / price * 100, 2)

# ========================= SIGNAL =========================
if pct_change >= 3 and avg_sentiment >= 20:
    signal, color = "STRONG BUY", "#00ff88"
elif pct_change >= 1.5 and avg_sentiment >= 5:
    signal, color = "BUY", "#00cc66"
elif pct_change <= -3 and avg_sentiment <= -20:
    signal, color = "STRONG SELL", "#ff0033"
elif pct_change <= -1.5 and avg_sentiment <= -5:
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
    st.markdown(f"<div class='card'><h4>Sentiment</h4><h2>{avg_sentiment:+.0f}</h2></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='signal' style='background:{color}'>SIGNAL: {signal}</div>", unsafe_allow_html=True)

# Plotly Dashboard
fig = make_subplots(rows=2, cols=2,
    subplot_titles=("Price & Forecast", "Sentiment Gauge", f"{days}-Day Forecast", "Executive Summary"),
    specs=[[{"type": "scatter"}, {"type": "indicator"}], [{"type": "bar"}, {"type": "table"}]]
)

fig.add_trace(go.Scatter(x=hist_df["Date"], y=hist_df["Close"], name="Historical", line=dict(color="#60a5fa", width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast", line=dict(color="#8b5cf6", width=4, dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(
    x=pd.concat([fc_df["ds"], fc_df["ds"][::-1]]),
    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(139,92,246,0.2)", line=dict(color="transparent"), showlegend=False
), row=1, col=1)

fig.add_trace(go.Indicator(
    mode="gauge+number", value=avg_sentiment,
    gauge={'axis': {'range': [-100,100]}, 'bar': {'color': "purple"}, 'steps': [
        {'range': [-100,-25], 'color': "red"},
        {'range': [-25,25], 'color': "gray"},
        {'range': [25,100], 'color': "green"}]},
    title={'text': "Sentiment Score"}
), row=1, col=2)

fig.add_trace(go.Bar(x=fc_df["ds"].dt.strftime("%b %d"), y=fc_df["yhat"], marker_color="#8b5cf6"), row=2, col=1)

fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"], fill_color="#1e293b", font=dict(color="white")),
    cells=dict(values=[
        ["Company", "Ticker", "Current Price", f"{days}D Forecast", "Change %", "Sentiment", "Signal", "Model"],
        [company_name, ticker, f"${price:.2f}", f"${forecast_price:.2f}", f"{pct_change:+.2f}%", f"{avg_sentiment:+.1f}", signal, model_used]
    ])
), row=2, col=2)

fig.update_layout(height=1000, showlegend=False, template="plotly_dark", title_text="Strategic Intelligence Dashboard")
st.plotly_chart(fig, use_container_width=True)

# ========================= SLACK ALERT =========================
if st.button("Send Alert to Slack", type="primary", use_container_width=True):
    msg = f"""
*InsightSphere Alert*  
*{company_name} ({ticker})*  
*Signal:* {signal}  
*Price:* ${price:.2f} to ${forecast_price:.2f} ({pct_change:+.2f}%)  
*Sentiment:* {avg_sentiment:+.1f}/100  
*Model:* {model_used}
    """.strip()
    try:
        requests.post(SLACK_WEBHOOK, json={"text": msg})
        st.success("Alert sent to Slack!")
    except:
        st.error("Slack failed")

st.markdown("---")
st.caption("© 2025 Infosys Springboard Internship • Team: Gopichand, Anshika, Janmejay, Vaishnavi")
