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

# ========================= SECRETS (FIXED!) =========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SLACK_URL = st.secrets["SLACK_WEBHOOK_URL"]
except Exception as e:
    st.error("Please add GEMINI_API_KEY and SLACK_WEBHOOK_URL in Streamlit Secrets")
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

# ========================= UI STYLE =========================
st.markdown("""
<style>
    .big-title {font-size: 3.2rem !important; font-weight: bold; text-align: center; margin: 0;}
    .subtitle {text-align: center; color: #94a3b8; font-size: 1.3rem; margin-bottom: 2rem;}
    .signal-box {text-align: center; padding: 1.5rem; border-radius: 20px; font-size: 2.2rem; font-weight: bold; color: white; margin: 2rem 0;}
    .metric-card {background: linear-gradient(135deg, #1e293b, #334155); padding: 1.5rem; border-radius: 16px; text-align: center; box-shadow: 0 8px 20px rgba(0,0,0,0.4);}
    .stButton>button {width: 100%; height: 3.5rem; font-size: 1.2rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Infosys InsightSphere</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Strategic Intelligence • Executive Dashboard</div>', unsafe_allow_html=True)

# ========================= SIDEBAR =========================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.title("Controls")
    ticker = st.text_input("Stock Ticker", "TSLA").upper().strip()
    company = st.text_input("Company Name", "Tesla").strip()
    days = st.slider("Forecast Horizon (Days)", 3, 21, 7)
    use_prophet = st.checkbox("Use Prophet (Best Accuracy)", value=True)
    
    st.markdown("---")
    if st.button("Run Full Analysis", type="primary", use_container_width=True):
        st.success("Analysis Started!")
        st.rerun()

# ========================= DATA FETCH =========================
@st.cache_data(ttl=300, show_spinner="Fetching real-time data...")
def get_stock_data(tkr):
    try:
        stock = yf.Ticker(tkr)
        hist = stock.history(period="1y", auto_adjust=True)
        if hist.empty:
            return None
        info = stock.info
        return {
            "hist": hist.reset_index(),
            "price": info.get("currentPrice") or hist["Close"].iloc[-1],
            "cap": info.get("marketCap"),
            "name": info.get("longName", company)
        }
    except:
        return None

@st.cache_data(ttl=600)
def get_news(comp, tkr):
    url = f"https://news.google.com/rss/search?q={comp}+{tkr}+stock&hl=en"
    feed = feedparser.parse(url)
    return [f"{e.title}. {e.get('summary','')}" for e in feed.entries[:15]]

# ========================= RUN ANALYSIS =========================
data = get_stock_data(ticker)
if not data:
    st.error(f"Invalid ticker: {ticker}")
    st.stop()

df = data["hist"]
price = data["price"]
cap = data["cap"]
name = data["name"] or company

news = get_news(name, ticker)

# ========================= SENTIMENT =========================
def get_sentiment(text):
    try:
        r = model.generate_content(
            f"Rate this financial sentiment from -100 (very bearish) to +100 (very bullish). Return ONLY the number.\n\nText: {text[:1400]}"
        )
        import re
        num = re.search(r'-?\d+', r.text)
        return int(num.group()) if num else 0
    except:
        return 0

with st.spinner("Gemini is analyzing market sentiment..."):
    sentiments = [get_sentiment(txt) for txt in news]
avg_sentiment = round(np.mean(sentiments), 1) if sentiments else 0

# ========================= FORECAST =========================
def make_forecast(df, days, use_prophet=True):
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

    # Lightweight fallback
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(df2["y"], trend="add")
    fit = model.fit()
    pred = fit.forecast(days)
    dates = pd.date_range(df2["ds"].iloc[-1] + timedelta(1), periods=days)
    res = pd.DataFrame({"ds": dates, "yhat": pred})
    res["yhat_lower"] = res["yhat"] * 0.93
    res["yhat_upper"] = res["yhat"] * 1.07
    return res, "Light Model"

fc_df, model_used = make_forecast(df, days, use_prophet)
proj_price = round(fc_df["yhat"].mean(), 2)
change_pct = round((proj_price - price) / price * 100, 2)

# ========================= SIGNAL =========================
if change_pct >= 3 and avg_sentiment >= 20:
    signal, color = "STRONG BUY", "#00ff88"
elif change_pct >= 1.5 and avg_sentiment >= 5:
    signal, color = "BUY", "#00cc66"
elif change_pct <= -3 and avg_sentiment <= -20:
    signal, color = "STRONG SELL", "#ff0033"
elif change_pct <= -1.5 and avg_sentiment <= -5:
    signal, color = "SELL", "#ff4444"
else:
    signal, color = "HOLD", "#ffaa00"

# ========================= DASHBOARD =========================
c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.5])
with c1:
    st.markdown(f"<div class='metric-card'><h3>{name}</h3><h1>${price:.2f}</h1><small>Current Price</small></div>", unsafe_allow_html=True)
with c2:
    cap_str = f"${cap/1e12:.2f}T" if cap and cap >= 1e12 else f"${cap/1e9:.2f}B" if cap else "N/A"
    st.markdown(f"<div class='metric-card'><h4>Market Cap</h4><h2>{cap_str}</h2></div>", unsafe_allow_html=True)
with c3:
    st.markdown(f"<div class='metric-card'><h4>Sentiment</h4><h2>{avg_sentiment:+.0f}</h2><small>/100</small></div>", unsafe_allow_html=True)
with c4:
    st.markdown(f"<div class='signal-box' style='background:{color}'>{signal}</div>", unsafe_allow_html=True)

# Full Dashboard Chart
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Price Trend & Forecast", "Market Sentiment", f"{days}-Day Price Forecast", "Executive Summary"),
    specs=[[{"type": "scatter"}, {"type": "indicator"}],
           [{"type": "bar"}, {"type": "table"}]],
    vertical_spacing=0.12
)

# Price Chart
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical Price", line=dict(width=3, color="#60a5fa")), row=1, col=1)
fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast", line=dict(width=4, color="#3b82f6", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(
    x=pd.concat([fc_df["ds"], fc_df["ds"][::-1]]),
    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(59,130,246,0.25)", line=dict(color="rgba(0,0,0,0)"), showlegend=False
), row=1, col=1)

# Sentiment Gauge
fig.add_trace(go.Indicator(
    mode = "gauge+number",
    value = avg_sentiment,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "Sentiment Score"},
    gauge = {
        'axis': {'range': [-100, 100]},
        'bar': {'color': "cyan"},
        'steps': [
            {'range': [-100, -25], 'color': "red"},
            {'range': [-25, 25], 'color': "gray"},
            {'range': [25, 100], 'color': "green"}],
        'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 0}}
), row=1, col=2)

# Forecast Bars
fig.add_trace(go.Bar(x=fc_df["ds"].dt.strftime("%b %d"), y=fc_df["yhat"], marker_color="#3b82f6", name="Forecast"), row=2, col=1)

# Summary Table
fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"], fill_color="#1e293b", font=dict(color="white", size=14)),
    cells=dict(values=[
        ["Company", "Ticker", "Price Now", f"{days}D Forecast", "Change %", "Sentiment", "Signal", "Model Used"],
        [name, ticker, f"${price:.2f}", f"${proj_price:.2f}", f"{change_pct:+.2f}%", f"{avg_sentiment:+.1f}", signal, model_used]
    ], fill_color="#334155", font=dict(color="white"))
), row=2, col=2)

fig.update_layout(height=950, showlegend=False, template="plotly_dark", margin=dict(t=80))
st.plotly_chart(fig, use_container_width=True)

# ========================= SLACK ALERT =========================
if st.button("Send Alert to Slack Channel", type="primary", use_container_width=True):
    alert = f"""
*InsightSphere Alert*  
*{name} ({ticker})*  
*Signal:* {signal}  
*Price:* ${price:.2f} to ${proj_price:.2f} ({change_pct:+.2f}%)  
*Sentiment:* {avg_sentiment:+.1f}/100  
*Model:* {model_used}  
*Time:* {datetime.now().strftime('%b %d, %Y %H:%M')}
    """.strip()
    try:
        requests.post(SLACK_URL, json={"text": alert})
        st.success("Alert sent to Slack!")
    except:
        st.error("Failed to send Slack alert")

# ========================= FOOTER =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#64748b; font-size:1rem;'>"
    "© 2025 Infosys Springboard Internship • Team: Gopichand, Anshika, Janmejay, Vaishnavi"
    "</p>",
    unsafe_allow_html=True
)
