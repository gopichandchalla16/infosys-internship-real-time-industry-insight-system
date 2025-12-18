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
from core.strategy import compute_competitive_index, classify_strategic_signal
from core.llm_strategy import generate_strategic_explanation
from core.market_data import fetch_market_data
from core.news_fetcher import fetch_news
from core.sentiment import analyze_sentiment
from core.forecast import run_prophet
from core.alerts import build_alert, send_slack
from core.utils import get_ticker
from core.utils import ALLOWED_COMPANIES

# --- NEW: GEMINI INTEGRATION ---
import google.generativeai as genai

# Setup Gemini API (Use Streamlit Secrets for security)
GEMINI_API_KEY = st.sidebar.text_input("Gemini API Key", type="password", help="Enter your Google AI API Key to enable Deep Strategic Analysis.")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
# -----------------------------

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
  --bg0:#05070c;
  --bg1:#090d18;
  --card: rgba(255,255,255,0.06);
  --card2: rgba(255,255,255,0.08);
  --stroke: rgba(255,255,255,0.10);
  --txt:#e8e8e8;
  --muted:#a7b0c0;
  --aqua:#00e5ff;
  --amber:#ffb347;
  --red:#ff4d6d;
  --green:#22c55e;
}

.stApp{
  background: radial-gradient(1200px 600px at 10% 0%, #0b1c2a 0%, var(--bg0) 55%) , linear-gradient(180deg, var(--bg1), var(--bg0));
  color: var(--txt);
}

.block-container{ padding-top: 1.4rem; padding-bottom: 3rem; }

h1, h2, h3, h4 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

.header-wrap{
  background: linear-gradient(135deg, rgba(0,229,255,0.10), rgba(255,179,71,0.07));
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  backdrop-filter: blur(10px);
  box-shadow: 0 14px 50px rgba(0,0,0,0.35);
  margin-bottom: 14px;
  margin-top:44px;
}

.title{
  font-size: 2.0rem;
  font-weight: 800;
  letter-spacing: -0.4px;
  margin: 0;
  background: linear-gradient(90deg, var(--aqua), var(--amber));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.subtitle{
  margin-top: 6px;
  color: var(--muted);
  font-size: 0.98rem;
}

.glass{
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  backdrop-filter: blur(10px);
  box-shadow: 0 10px 34px rgba(0,0,0,0.35);
}

.glass-hover{
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.glass-hover:hover{
  transform: translateY(-3px);
  border-color: rgba(0,229,255,0.30);
  box-shadow: 0 14px 44px rgba(0,229,255,0.10), 0 18px 60px rgba(0,0,0,0.45);
}

.kpi-label{ color: var(--muted); font-size: 0.85rem; margin-bottom: 6px; }
.kpi-value{ font-size: 1.55rem; font-weight: 800; letter-spacing: -0.2px; }
.kpi-sub{ color: var(--muted); font-size: 0.82rem; margin-top: 4px; }

.badge{
  display:inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.78rem;
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.05);
  color: var(--txt);
}

.badge-pos{ border-color: rgba(34,197,94,0.35); background: rgba(34,197,94,0.12); }
.badge-neg{ border-color: rgba(255,77,109,0.35); background: rgba(255,77,109,0.12); }
.badge-neu{ border-color: rgba(167,176,192,0.35); background: rgba(167,176,192,0.10); }

.news-card{
  padding: 14px;
  margin-bottom: 10px;
}
.news-title{
  font-size: 1.03rem;
  font-weight: 750;
  text-decoration: none;
  color: var(--aqua);
}
.news-meta{ color: var(--muted); font-size: 0.82rem; margin-top: 3px; }
.news-desc{ color: #d7dbe6; font-size: 0.92rem; margin-top: 8px; line-height: 1.35rem; }

hr{
  border: none;
  height: 1px;
  background: rgba(255,255,255,0.07);
  margin: 12px 0;
}

[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.03);
  border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.03);
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.06);
}

.stButton>button{
  background: linear-gradient(90deg, var(--aqua), var(--amber));
  border: none;
  color: #0a0a0a;
  font-weight: 800;
  border-radius: 999px;
  padding: 0.55rem 1rem;
  transition: transform 0.16s ease;
}
.stButton>button:hover{ transform: scale(1.03); }

.small-note{ color: var(--muted); font-size: 0.85rem; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# HELPERS
# -----------------------------
def _safe_pct(a: float, b: float) -> float:
    if b == 0 or pd.isna(b) or pd.isna(a):
        return 0.0
    return float((a - b) / b * 100.0)

def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)


def _sentiment_badge(label: str) -> str:
    lbl = (label or "").lower()
    if lbl == "positive":
        return '<span class="badge badge-pos">Positive</span>'
    if lbl == "negative":
        return '<span class="badge badge-neg">Negative</span>'
    return '<span class="badge badge-neu">Neutral</span>'

@st.cache_data(show_spinner=False, ttl=300)
def load_market_data(company: str) -> pd.DataFrame:
    return fetch_market_data(company)

@st.cache_data(show_spinner=False, ttl=300)
def load_news(company: str):
    return fetch_news(company)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
with st.sidebar:
    st.markdown("## Controls")
    company = st.selectbox(
        "Company",
        sorted(ALLOWED_COMPANIES),
)

    ticker = get_ticker(company)

    range_label = st.selectbox(
        "Time Range",
        ["1M", "3M", "6M", "1Y", "MAX"],
        index=1,
        help="Controls how much historical data is shown in charts and KPIs.",
    )

    show_candles = st.toggle("Candlestick View", value=True)
    show_volume = st.toggle("Show Volume", value=True)
    show_indicators = st.toggle("Show Indicators (MA + RSI)", value=True)

    st.markdown("---")
    news_limit = st.slider("News Articles", 5, 30, 12)
    auto_refresh = st.toggle("Auto Refresh (Market + News)", value=False)
    refresh_secs = st.slider("Refresh Interval (sec)", 15, 120, 30, disabled=not auto_refresh)

    st.markdown("---")
    slack_enabled = st.toggle("Enable Slack Button", value=True)
    st.markdown(
        "<div class='small-note'>Tip: Use this for your demo. KPIs + Candlestick + Sentiment + News Cards impress quickly.</div>",
        unsafe_allow_html=True,
    )

# Auto refresh
if auto_refresh:
    st.caption(f"Auto-refresh enabled every {refresh_secs}s")
    st.autorefresh(interval=refresh_secs * 1000, key="auto_refresh")

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    f"""
<div class="header-wrap">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; flex-wrap:wrap;">
    <div>
      <div class="title">Real-Time Market Intelligence</div>
      <div class="subtitle">
        Company: <b>{company}</b> ({ticker}) · Strategic Intelligence · Market Trends · AI Sentiment · Forecast · Alerts
      </div>
    </div>
    <div style="text-align:right;">
      <div class="badge">Internship Project</div>
      <div class="news-meta">{datetime.now().strftime("%d %b %Y · %I:%M %p")}</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# LOAD DATA
# -----------------------------
with st.spinner("Loading market data and news..."):
    market_df = load_market_data(company)

if market_df is None or market_df.empty:
    st.error("Market data unavailable. Please check your data source or ticker mapping.")
    st.stop()

# Ensure datetime index
if not isinstance(market_df.index, pd.DatetimeIndex):
    # try common patterns: Date column or reset index
    if "Date" in market_df.columns:
        market_df["Date"] = pd.to_datetime(market_df["Date"])
        market_df = market_df.set_index("Date")
    else:
        market_df.index = pd.to_datetime(market_df.index)

market_df = market_df.sort_index()

# Filter by range
end_dt = market_df.index.max()
if range_label == "1M":
    start_dt = end_dt - pd.Timedelta(days=31)
elif range_label == "3M":
    start_dt = end_dt - pd.Timedelta(days=93)
elif range_label == "6M":
    start_dt = end_dt - pd.Timedelta(days=186)
elif range_label == "1Y":
    start_dt = end_dt - pd.Timedelta(days=366)
else:
    start_dt = market_df.index.min()

df = market_df.loc[market_df.index >= start_dt].copy()
if df.empty:
    df = market_df.tail(120).copy()

# Indicators
df["Return"] = df["Close"].pct_change() * 100
df["MA7"] = df["Close"].rolling(7).mean()
df["MA21"] = df["Close"].rolling(21).mean()
df["RSI14"] = _compute_rsi(df["Close"], 14)

# News + Sentiment
with st.spinner("Fetching headlines and running sentiment..."):
    news = load_news(company) or []
    sentiment_details, sentiment_counts = analyze_sentiment(news)

# Forecast
with st.spinner("Generating forecast..."):
    forecast_df = run_prophet(market_df) if len(market_df) >= 60 else None

# -----------------------------
# STRATEGIC INTELLIGENCE LOGIC
# -----------------------------
competitive_index = compute_competitive_index(
    market_df=market_df,
    sentiment_counts=sentiment_counts or {},
    forecast_df=forecast_df,
)

strategic_signal = classify_strategic_signal(
    market_df=market_df,
    sentiment_counts=sentiment_counts or {},
    forecast_df=forecast_df,
)

# Alert payload
alert = build_alert(
    company,
    ticker,
    sentiment_counts or {},
    strategic={
        "competitive_index": competitive_index,
        "strategic_signal": strategic_signal,
    },
)
# -----------------------------
# LLM STRATEGIC EXPLANATION
# -----------------------------
llm_explanation = generate_strategic_explanation(
    company=company,
    competitive_index=competitive_index,
    strategic_signal=strategic_signal,
    sentiment_counts=sentiment_counts or {},
)

# --- NEW: GEMINI API OVERRIDE ---
if GEMINI_API_KEY:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze the current strategic position for {company} ({ticker}).
        Competitive Index: {competitive_index}/100
        Signal: {strategic_signal['signal']}
        Sentiment Breakdown: {sentiment_counts}
        
        Write a professional executive strategy brief (max 150 words). 
        Identify the single biggest opportunity or threat and provide a specific strategic recommendation.
        """
        response = model.generate_content(prompt)
        llm_explanation = response.text
    except Exception as e:
        st.sidebar.warning(f"Gemini Analysis Failed: {e}")
# -----------------------------


# -----------------------------
# KPI ROW
# -----------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
chg_1d = _safe_pct(last_close, prev_close)

first_close = float(df["Close"].iloc[0])
chg_range = _safe_pct(last_close, first_close)

volatility = float(df["Return"].std(skipna=True)) if df["Return"].notna().any() else 0.0
avg_volume = float(df["Volume"].tail(20).mean()) if "Volume" in df.columns else np.nan

pos = int((sentiment_counts or {}).get("positive", 0))
neg = int((sentiment_counts or {}).get("negative", 0))
neu = int((sentiment_counts or {}).get("neutral", 0))
total_news = max(pos + neg + neu, len(news))

sent_score = 0.0
if total_news > 0:
    sent_score = ((pos - neg) / total_news) * 100.0

k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, label, value, sub):
    col.markdown(
        f"""
<div class="glass glass-hover">
  <div class="kpi-label">{label}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-sub">{sub}</div>
</div>
""",
        unsafe_allow_html=True,
    )

kpi_card(k1, "Last Close", f"{last_close:,.2f}", f"1D: {chg_1d:+.2f}%")
kpi_card(k2, f"Change ({range_label})", f"{chg_range:+.2f}%", "Price performance")
kpi_card(k3, "Volatility", f"{volatility:.2f}%", "Std dev of daily returns")
kpi_card(k4, "Sentiment Score", f"{sent_score:+.1f}", f"Pos {pos} · Neu {neu} · Neg {neg}")
kpi_card(
    k5,
    "Alert",
    f"{(alert or {}).get('alert_type','N/A')}",
    (alert or {}).get("strategic_action", "—")[:34] + "…",
)


st.markdown("<hr/>", unsafe_allow_html=True)

# -----------------------------
# MAIN CONTENT TABS
# -----------------------------
tab_strategy, tab_overview, tab_forecast, tab_sentiment, tab_news, tab_alerts = st.tabs(
    ["🧠 Strategy", "📈 Market", "🧠 Forecast", "💬 Sentiment", "📰 News", "🔔 Alerts"]
)


# =========================================================
# TAB: MARKET
# =========================================================
with tab_overview:
    left, right = st.columns([1.75, 1.0], gap="large")

    with left:
        st.markdown("### Price Trend")

        fig = go.Figure()

        if show_candles and all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="OHLC",
                    increasing_line_color="rgba(34,197,94,0.85)",
                    decreasing_line_color="rgba(255,77,109,0.85)",
                )
            )
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="Close", line=dict(width=2.8)))

        if show_indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["MA7"], name="MA 7", line=dict(width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA21"], name="MA 21", line=dict(width=2)))

        fig.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=12, r=12, t=40, b=12),
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        if show_volume and "Volume" in df.columns:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))
            fig2.update_layout(
                template="plotly_dark",
                height=180,
                margin=dict(l=12, r=12, t=20, b=12),
                hovermode="x unified",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Returns Snapshot")
        r1, r2 = st.columns(2)

        with r1:
            fig_ret = px.histogram(df.dropna(subset=["Return"]), x="Return", nbins=40, template="plotly_dark")
            fig_ret.update_layout(height=260, margin=dict(l=12, r=12, t=30, b=12))
            st.plotly_chart(fig_ret, use_container_width=True)

        with r2:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI(14)", line=dict(width=2.4)))
            fig_rsi.add_hline(y=70, line_width=1, line_dash="dash", line_color="rgba(255,179,71,0.85)")
            fig_rsi.add_hline(y=30, line_width=1, line_dash="dash", line_color="rgba(0,229,255,0.85)")
            fig_rsi.update_layout(template="plotly_dark", height=260, margin=dict(l=12, r=12, t=30, b=12))
            st.plotly_chart(fig_rsi, use_container_width=True)

    with right:
        st.markdown("### Quick Summary")

        trend_txt = "Uptrend" if df["MA7"].iloc[-1] >= df["MA21"].iloc[-1] else "Downtrend"
        rsi_now = float(df["RSI14"].iloc[-1])
        rsi_txt = "Overbought" if rsi_now >= 70 else "Oversold" if rsi_now <= 30 else "Neutral"

        st.markdown(
            f"""
<div class="glass">
  <div class="kpi-label">Trend</div>
  <div class="kpi-value">{trend_txt}</div>
  <div class="kpi-sub">Based on MA7 vs MA21 crossover</div>
</div>
<br/>
<div class="glass">
  <div class="kpi-label">RSI(14)</div>
  <div class="kpi-value">{rsi_now:.1f}</div>
  <div class="kpi-sub">{rsi_txt}</div>
</div>
<br/>
<div class="glass">
  <div class="kpi-label">Average Volume (20d)</div>
  <div class="kpi-value">{("" if np.isnan(avg_volume) else f"{avg_volume:,.0f}")}</div>
  <div class="kpi-sub">Liquidity proxy</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### Data Export")
        export_df = df.reset_index().rename(columns={"index": "Date"})
        st.download_button(
            "Download Market Data (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{company}_{range_label}_market.csv",
            mime="text/csv",
        )
# =========================================================
# TAB: STRATEGIC INTELLIGENCE
# =========================================================
with tab_strategy:
    st.markdown("## 🧠 Strategic Intelligence Summary")

    c1, c2 = st.columns([1, 1.2], gap="large")

    # -----------------------------
    # COMPETITIVE POSITIONING INDEX
    # -----------------------------
    with c1:
        strength = (
            "Dominant" if competitive_index >= 80 else
            "Strong" if competitive_index >= 65 else
            "Neutral" if competitive_index >= 45 else
            "Weak"
        )

        st.markdown(
            f"""
<div class="glass glass-hover">
  <div class="kpi-label">Competitive Positioning Index</div>
  <div class="kpi-value">{competitive_index} / 100</div>
  <div class="kpi-sub">Strategic Strength: <b>{strength}</b></div>
  <br/>
  <div class="small-note">
    Computed using price momentum, sentiment polarity,
    forecast direction, and news intensity.
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("## 🤖 AI-Generated Strategic Explanation")

        st.markdown(
            f"""
<div class="glass glass-hover">
  <div class="kpi-label">Executive Strategy Brief</div>
  <pre style="
    white-space: pre-wrap;
    font-size: 0.92rem;
    line-height: 1.45rem;
    color: #e8e8e8;
    margin-top: 10px;
  ">
{llm_explanation}
  </pre>
</div>
""",
            unsafe_allow_html=True,
        )
        

    # -----------------------------
    # OPPORTUNITY / THREAT SIGNAL
    # -----------------------------
    with c2:
        signal_color = (
            "#22c55e" if strategic_signal["signal"] == "OPPORTUNITY"
            else "#ff4d6d" if strategic_signal["signal"] == "THREAT"
            else "#ffb347"
        )

        st.markdown(
            f"""
<div class="glass glass-hover">
  <div class="kpi-label">Strategic Signal</div>
  <div class="kpi-value" style="color:{signal_color}">
    {strategic_signal['signal']}
  </div>
  <div class="kpi-sub">Confidence Level: {strategic_signal['confidence']}</div>
  <hr/>
  <ul style="margin-left:18px;">
    {''.join([f"<li>{r}</li>" for r in strategic_signal["reason"]])}
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # -----------------------------
    # STRATEGIC INTERPRETATION
    # -----------------------------
    st.markdown("### 📌 Strategic Interpretation")

    interpretation = (
        "Current indicators suggest a favorable strategic position with upside potential."
        if strategic_signal["signal"] == "OPPORTUNITY"
        else
        "Warning signals detected. Risk mitigation and close monitoring are recommended."
        if strategic_signal["signal"] == "THREAT"
        else
        "Signals are mixed. Continued observation is advised before taking strategic action."
    )

    st.markdown(
        f"""
<div class="glass">
  <div class="small-note">{interpretation}</div>
</div>
""",
        unsafe_allow_html=True,
    )

# =========================================================
# TAB: FORECAST
# =========================================================
with tab_forecast:
    st.markdown("### 7-Day Forecast (Prophet)")

    if forecast_df is None or getattr(forecast_df, "empty", False):
        st.info("Forecast unavailable. Showing recent close trend instead.")
        st.line_chart(market_df["Close"].tail(60))
    else:
        hist = market_df.reset_index()
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist[date_col], y=hist["Close"], name="Historical", line=dict(width=2.4)))
        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat"], name="Forecast", line=dict(width=2.4)))

        fig.add_trace(go.Scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], showlegend=False, line=dict(width=0)))
        fig.add_trace(
            go.Scatter(
                x=forecast_df["ds"],
                y=forecast_df["yhat_lower"],
                fill="tonexty",
                showlegend=False,
                line=dict(width=0),
                fillcolor="rgba(255,179,71,0.14)",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=520,
            hovermode="x unified",
            margin=dict(l=12, r=12, t=40, b=12),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Forecast Table")
        fshow = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10).copy()
        fshow.columns = ["Date", "Forecast", "Lower", "Upper"]
        st.dataframe(fshow, use_container_width=True)

# =========================================================
# TAB: SENTIMENT
# =========================================================
with tab_sentiment:
    st.markdown("### Sentiment Overview (FinBERT)")

    cA, cB = st.columns([1.1, 1.2], gap="large")
    with cA:
        fig_pie = px.pie(
            names=["Positive", "Neutral", "Negative"],
            values=[pos, neu, neg],
            template="plotly_dark",
            hole=0.55,
        )
        fig_pie.update_layout(height=360, margin=dict(l=12, r=12, t=40, b=12))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cB:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=sent_score,
                number={"suffix": " "},
                delta={"reference": 0},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {"color": "rgba(0,229,255,0.85)"},
                    "steps": [
                        {"range": [-100, -20], "color": "rgba(255,77,109,0.20)"},
                        {"range": [-20, 20], "color": "rgba(167,176,192,0.16)"},
                        {"range": [20, 100], "color": "rgba(34,197,94,0.20)"},
                    ],
                },
                title={"text": "Net Sentiment Score"},
            )
        )
        gauge.update_layout(template="plotly_dark", height=360, margin=dict(l=12, r=12, t=40, b=12))
        st.plotly_chart(gauge, use_container_width=True)

    st.markdown("### Article-Level Sentiment")
    df_sent = pd.DataFrame(sentiment_details or [])
    if df_sent.empty:
        st.info("No sentiment details available.")
    else:
        st.dataframe(df_sent, use_container_width=True)

# =========================================================
# TAB: NEWS
# =========================================================
with tab_news:
    st.markdown("### Latest Headlines")

    if not news:
        st.info("No headlines found right now.")
    else:
        # Try to merge sentiment labels back to news if sentiment_details contains them
        # Expected fields can vary; we handle best-effort.
        sent_map = {}
        if isinstance(sentiment_details, list) and sentiment_details:
            for item in sentiment_details:
                t = (item.get("title") or "").strip()
                if t:
                    sent_map[t] = (item.get("sentiment") or item.get("label") or item.get("prediction") or "neutral")

        shown = 0
        for n in news:
            if shown >= news_limit:
                break

            title = (n.get("title") or "Untitled").strip()
            link = n.get("link") or "#"
            desc = (n.get("description") or n.get("summary") or "").strip()
            source = (n.get("source") or n.get("publisher") or "").strip()
            published = (n.get("published") or n.get("pubDate") or n.get("date") or "").strip()

            label = sent_map.get(title, n.get("sentiment") or "neutral")
            badge = _sentiment_badge(str(label))

            st.markdown(
                f"""
<div class="glass glass-hover news-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:10px; flex-wrap:wrap;">
    <div style="flex:1; min-width:260px;">
      <a class="news-title" href="{link}" target="_blank">{title}</a>
      <div class="news-meta">{source} {'· ' + published if published else ''}</div>
    </div>
    <div>{badge}</div>
  </div>
  {"<div class='news-desc'>" + desc + "</div>" if desc else ""}
</div>
""",
                unsafe_allow_html=True,
            )
            shown += 1

# =========================================================
# TAB: ALERTS
# =========================================================
with tab_alerts:
    st.markdown("### Alert Payload")

    st.markdown(
        f"""
<div class="glass">
  <div class="kpi-label">Alert Type</div>
  <div class="kpi-value">{(alert or {}).get("alert_type","N/A")}</div>
  <div class="kpi-sub">{(alert or {}).get("message","")}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### JSON Preview")
    st.json(alert)

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    with b2:
        # --- EDITED SLACK ALERTS PART ---
        if slack_enabled and st.button("🚨 Send Slack Alert"):
            # Enrich the alert with Gemini explanation if available
            slack_alert = alert.copy()
            slack_alert["gemini_analysis"] = llm_explanation
            
            # Create a more visual Slack block if your send_slack supports it, 
            # otherwise send_slack(alert) remains untouched as requested.
            ok = send_slack(slack_alert) 
            
            if ok:
                st.success("High-Priority Slack alert sent with AI Analysis.")
            else:
                st.warning("Slack webhook not configured in core/alerts.py.")
        # -----------------------------

    with b3:
        st.download_button(
            "Download Alert JSON",
            data=pd.Series(alert).to_json().encode("utf-8"),
            file_name=f"{company}_alert.json",
            mime="application/json",
        )
