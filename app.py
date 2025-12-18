# app.py
import sys
import os
from datetime import datetime, timedelta

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai  # Added for Gemini Integration

# Import custom modules (Ensure these files exist in your core/ folder)
from core.strategy import compute_competitive_index, classify_strategic_signal
from core.llm_strategy import generate_strategic_explanation
from core.market_data import fetch_market_data
from core.news_fetcher import fetch_news
from core.sentiment import analyze_sentiment
from core.forecast import run_prophet
from core.alerts import build_alert, send_slack
from core.utils import get_ticker, ALLOWED_COMPANIES

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Real-Time Market Intelligence", layout="wide", page_icon="📈")

# -----------------------------
# GLOBAL PREMIUM UI (GLASS + DARK)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg0:#05070c; --bg1:#090d18; --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08); --stroke: rgba(255,255,255,0.10);
  --txt:#e8e8e8; --muted:#a7b0c0; --aqua:#00e5ff; --amber:#ffb347;
  --red:#ff4d6d; --green:#22c55e;
}
.stApp{ background: radial-gradient(1200px 600px at 10% 0%, #0b1c2a 0%, var(--bg0) 55%) , linear-gradient(180deg, var(--bg1), var(--bg0)); color: var(--txt); }
.header-wrap{ background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(255,179,71,0.07)); border: 1px solid var(--stroke); border-radius: 18px; padding: 18px; backdrop-filter: blur(10px); margin-bottom: 14px; margin-top:44px; }
.title{ font-size: 2.0rem; font-weight: 800; background: linear-gradient(90deg, var(--aqua), var(--amber)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.glass{ background: var(--card); border: 1px solid var(--stroke); border-radius: 16px; padding: 14px; backdrop-filter: blur(10px); }
.kpi-value{ font-size: 1.55rem; font-weight: 800; }
.badge{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; border: 1px solid var(--stroke); background: rgba(255,255,255,0.05); }
.news-card{ padding: 14px; margin-bottom: 10px; border-radius: 12px; background: rgba(255,255,255,0.03); }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# CACHED DATA WRAPPERS (Fixes Delays)
# -----------------------------
@st.cache_data(ttl=600)
def get_cached_market_data(comp): return fetch_market_data(comp)

@st.cache_data(ttl=600)
def get_cached_news(comp): return fetch_news(comp)

@st.cache_data(ttl=600)
def get_cached_sentiment(news_list): return analyze_sentiment(news_list)

@st.cache_data(ttl=3600)
def get_cached_forecast(df): return run_prophet(df)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2621/2621841.png", width=80)
    st.markdown("## ⚙️ Configuration")
    
    # 1. API KEYS (GEMINI & SLACK)
    with st.expander("🔑 API & Webhooks", expanded=True):
        gemini_key = st.text_input("Gemini API Key", type="password", help="Enter Google AI Studio Key")
        slack_webhook = st.text_input("Slack Webhook URL", type="password", help="Enter Incoming Webhook URL")
    
    if gemini_key:
        genai.configure(api_key=gemini_key)

    st.markdown("---")
    company = st.selectbox("Select Target Company", sorted(ALLOWED_COMPANIES))
    ticker = get_ticker(company)
    range_label = st.selectbox("Time Horizon", ["1M", "3M", "6M", "1Y", "MAX"], index=1)

    st.markdown("---")
    auto_refresh = st.toggle("Live Auto-Refresh", value=False)
    if auto_refresh:
        refresh_secs = st.slider("Interval (s)", 15, 120, 30)
        st.autorefresh(interval=refresh_secs * 1000, key="auto_refresh")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(f'<div class="header-wrap"><div class="title">Real-Time Industry Insight</div>'
            f'<div style="color:var(--muted)">Analyzing <b>{company}</b> ({ticker}) — Generated on {datetime.now().strftime("%H:%M")}</div></div>', unsafe_allow_html=True)

# -----------------------------
# MAIN DATA PROCESSING
# -----------------------------
with st.spinner("Processing Intelligence..."):
    market_df = get_cached_market_data(company)
    if market_df is None or market_df.empty:
        st.error("Could not fetch market data.")
        st.stop()

    # News & Sentiment
    news = get_cached_news(company) or []
    sentiment_details, sentiment_counts = get_cached_sentiment(news)
    
    # Forecast & Strategy
    forecast_df = get_cached_forecast(market_df) if len(market_df) > 60 else None
    comp_idx = compute_competitive_index(market_df, sentiment_counts, forecast_df)
    strat_signal = classify_strategic_signal(market_df, sentiment_counts, forecast_df)

# -----------------------------
# TABS LOGIC
# -----------------------------
t_strat, t_mkt, t_fore, t_news, t_alert = st.tabs(["🧠 Strategy", "📈 Market", "🔮 Forecast", "📰 News", "🔔 Alerts"])

# =========================================================
# TAB: STRATEGY (GEMINI INTEGRATION)
# =========================================================
with t_strat:
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"""<div class="glass"><h3>Index: {comp_idx}/100</h3><p>Signal: <b>{strat_signal['signal']}</b></p></div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🤖 Gemini Executive Brief")
        if not gemini_key:
            st.warning("Please provide a Gemini API Key in the sidebar to generate AI insights.")
        else:
            # Using Cache for LLM to avoid repeated API calls
            @st.cache_data(ttl=3600)
            def get_ai_brief(c, ci, ss, sc):
                return generate_strategic_explanation(c, ci, ss, sc)
            
            with st.spinner("Gemini is analyzing..."):
                brief = get_ai_brief(company, comp_idx, strat_signal, sentiment_counts)
                st.markdown(f'<div class="glass" style="line-height:1.6">{brief}</div>', unsafe_allow_html=True)

# =========================================================
# TAB: MARKET
# =========================================================
with t_mkt:
    fig = go.Figure(data=[go.Candlestick(x=market_df.index, open=market_df['Open'], high=market_df['High'], low=market_df['Low'], close=market_df['Close'])])
    fig.update_layout(template="plotly_dark", height=450, margin=dict(t=0, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# TAB: FORECAST
# =========================================================
with t_fore:
    if forecast_df is not None:
        fig_f = px.line(forecast_df, x='ds', y='yhat', title="7-Day Price Projection")
        fig_f.update_layout(template="plotly_dark")
        st.plotly_chart(fig_f, use_container_width=True)
    else:
        st.info("Insufficient data for Prophet forecasting.")

# =========================================================
# TAB: ALERTS (SLACK INTEGRATION)
# =========================================================
with t_alert:
    st.subheader("🚨 Real-Time Notification Center")
    alert_payload = build_alert(company, ticker, sentiment_counts, {"competitive_index": comp_idx, "strategic_signal": strat_signal})
    
    st.json(alert_payload)
    
    if st.button("🚀 Trigger Slack Webhook"):
        if not slack_webhook:
            st.error("Error: No Slack Webhook URL found in the sidebar!")
        else:
            with st.spinner("Sending to Slack..."):
                # Call send_slack with the dynamic webhook URL
                success = send_slack(alert_payload, webhook_url=slack_webhook)
                if success:
                    st.success("Successfully pushed to Slack channel!")
                else:
                    st.error("Failed to send. Check your Webhook URL.")

# Footer
st.markdown("---")
st.caption(f"Real-Time Industry Insight System | Version 2.1 (Optimized)")
