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

# Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    SLACK_URL = st.secrets["SLACK_WEBHOOK_URL"]
except:
    st.error("Set GEMINI_API_KEY and SLACK_WEBHOOK_URL in Streamlit Secrets)
    st.stop()

model = genai.GenerativeModel("gemini-1.5-flash")

# ========================= UI =========================
st.markdown("""
<style>
    .big-title {font-size: 3rem !important; font-weight: bold; text-align: center; margin-bottom: 0;}
    .subtitle {text-align: center; color: #94a3b8; margin-bottom: 2rem;}
    .signal-box {text-align: center; padding: 1.5rem; border-radius: 20px; font-size: 2rem; font-weight: bold; color: white; margin: 1rem 0;}
    .metric-card {background: #1e293b; padding: 1rem; border-radius: 12px; text-align: center;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">Infosys InsightSphere</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Strategic Intelligence Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", " , "TSLA").upper()
    company = st.text_input("Company Name", "Tesla").strip()
    days = st.slider("Forecast Days", 3, 21, 7)
    use_prophet = st.checkbox("Use Prophet Model", True)
    if st.button("Run Analysis", type="primary", use_container_width=True):
        st.rerun()

# ========================= DATA =========================
@st.cache_data(ttl=300)
def get_data(tkr):
    stock = yf.Ticker(tkr)
    hist = stock.history(period="1y")
    if hist.empty: return None
    info = stock.info
    return {
        "hist": hist.reset_index(),
        "price": info.get("currentPrice") or hist["Close"].iloc[-1],
        "cap": info.get("marketCap"),
        "name": info.get("longName", company)
    }

@st.cache_data(ttl=600)
def get_news(comp, tkr):
    url = f"https://news.google.com/rss/search?q={comp}+{tkr}+stock&hl=en-US"
    feed = feedparser.parse(url)
    return [f"{e.title}. {e.get('summary','')}" for e in feed.entries[:12]]

data = get_data(ticker)
if not data:
    st.error("Invalid ticker or no data")
    st.stop()

df = data["hist"]
price = data["price"]
cap = data["cap"]
name = data["name"]

news_texts = get_news(company, ticker)

# ========================= SENTIMENT
def score(text):
    try:
        r = model.generate_content(f"Rate sentiment -100 (very bearish) to +100 (very bullish). Return ONLY the number.\n\n{text[:1400]}")
        return int(''.join(filter(str.isdigit, r.text)) or "0") * (1 if any(x in r.text for x in ["+", "positive"]) else -1)
    except:
        return 0

with st.spinner("Analyzing news sentiment with Gemini..."):
    sentiments = [score(txt) for txt in news_texts]
avg_sentiment = np.mean(sentiments) if sentiments else 0

# FORECASTING
def forecast(df, days, prophet=True):
    df2 = df.copy()
    df2["ds"] = pd.to_datetime(df2["Date"])
    df2["y"] = df2["Close"]

    if prophet:
        try:
            from prophet import Prophet
            m = Prophet(daily_seasonality=True, weekly_seasonality=True)
            m.fit(df2)
            future = m.make_future_dataframe(periods=days)
            fc = m.predict(future)
            return fc.tail(days), "Prophet"
        except:
            pass

    # ARIMA fallback
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    model = ExponentialSmoothing(df2["y"], trend="add")
    fc = model.fit().forecast(days)
    dates = pd.date_range(df2["ds"].iloc[-1] + timedelta(1), periods=days)
    res = pd.DataFrame({"ds": dates, "yhat": fc})
    res["yhat_lower"] = res["yhat"] * 0.93
    res["yhat_upper"] = res["yhat"] * 1.07
    return res, "Exponential Smoothing"

fc_df, model_name = forecast(df, days, use_prophet)
proj_price = fc_df["yhat"].mean()
change_pct = (proj_price - price) / price * 100

# SIGNAL
if change_pct >= 3 and avg_sentiment >= 20:
    signal, color = "STRONG BUY", "#00ff9d"
elif change_pct >= 1.5 and avg_sentiment >= 5:
    signal, color = "BUY", "#00ff44"
elif change_pct <= -3 and avg_sentiment <= -20:
    signal, color = "STRONG SELL", "#ff1744"
elif change_pct <= -1.5 and avg_sentiment <= -5:
    signal, color = "SELL", "#ff5722"
else:
    signal, color = "HOLD", "#ffa726"

# ========================= DASHBOARD =========================
col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1.2])
with col1:
    st.markdown(f"<div class='metric-card'><h3>{name}</h3><h2>${price:.2f}</h2></div>", unsafe_allow_html=True)
with col2:
    cap_str = f"${cap/1e12:.2f}T" if cap and cap >= 1e12 else f"${cap/1e9:.2f}B"
    st.markdown(f"<div class='metric-card'><h4>Market Cap</h4><h2>{cap_str}</h2></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><h4>Sentiment</h4><h2>{avg_sentiment:+.0f}</h2></div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div class='signal-box' style='background:{color}'>{signal}</div>", unsafe_allow_html=True)

# Main Chart
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Price & Forecast", "Sentiment Gauge", f"{days}-Day Forecast", "Summary"),
    specs=[[{"type": "scatter"}, {"type": "indicator"}],
           [{"type": "bar"}, {"type": "table"}]]
)

# Price + Forecast
fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical", line=dict(color="#60a5fa")), row=1, col=1)
fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"], name="Forecast", line=dict(color="#3b82f6", dash="dot")), row=1, col=1)
fig.add_trace(go.Scatter(
    x=pd.concat([fc_df["ds"], fc_df["ds"][::-1]]),
    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
    fill="toself", fillcolor="rgba(59,130,246,0.2)", line=dict(color="rgba(0,0,0,0)"), showlegend=False
), row=1, col=1)

# Sentiment Gauge
fig.add_trace(go.Indicator(
    mode="gauge+number+delta",
    value=avg_sentiment,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Market Sentiment"},
    delta={'reference': 0},
    gauge={'axis': {'range': [-100, 100]},
           'bar': {'color': "cyan"},
           'steps': [{'range': [-100, -20], 'color': "red"},
                     {'range': [-20, 20], 'color': "gray"},
                     {'range': [20, 100], 'color': "green"}]}
), row=1, col=2)

# Forecast Bars
fig.add_trace(go.Bar(x=fc_df["ds"].dt.strftime("%b %d"), y=fc_df["yhat"], marker_color="#3b82f6"), row=2, col=1)

# Summary Table
fig.add_trace(go.Table(
    header=dict(values=["Metric", "Value"], fill_color="#1e293b", font=dict(color="white")),
    cells=dict(values=[
        ["Company", "Ticker", "Current Price", f"{days}D Forecast", "Expected Move", "Sentiment", "Signal", "Model"],
        [name, ticker, f"${price:.2f}", f"${proj_price:.2f}", f"{change_pct:+.2f}%", f"{avg_sentiment:+.1f}", signal, model_name]
    ])
), row=2, col=2)

fig.update_layout(height=900, showlegend=False, title_text="Executive Dashboard", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# Slack Alert
if st.button("Send Alert to Slack", type="primary", use_container_width=True):
    msg = f"""
*InsightSphere Alert*
*{name} ({ticker})*
*Signal:* {signal}
*Price:* ${price:.2f} to ${proj_price:.2f} ({change_pct:+.2f}%)
*Sentiment:* {avg_sentiment:+.1f}/100
*Model:* {model_name}
    """.strip()
    requests.post(SLACK_URL, json={"text": msg})
    st.success("Alert sent!")

st.caption("© 2025 Infosys Springboard Internship | Team: Gopichand, Anshika, Janmejay, Vaishnavi")
