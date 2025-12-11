import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import streamlit as st
import pandas as pd
import numpy as np
import requests
import feedparser
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional Gemini import (safe)
try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

# ------------------- Page config & CSS -------------------
st.set_page_config(page_title="InsightSphere", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
    .glass { background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)); backdrop-filter: blur(14px); border-radius: 16px; padding: 18px; border:1px solid rgba(255,255,255,0.06); }
    .metric { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 12px; text-align:center; }
    .title { font-size:1.9rem; font-weight:700; margin-bottom:6px; }
    .subtitle { color: rgba(255,255,255,0.78); margin-bottom:10px; }
    .light .glass { background: rgba(255,255,255,0.92); color:#0b1220; border:1px solid rgba(0,0,0,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------- Secrets -------------------
ALPHA_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "") if hasattr(st, "secrets") else os.getenv("ALPHA_VANTAGE_API_KEY", "")
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else os.getenv("GEMINI_API_KEY", "")
SLACK_WEBHOOK = st.secrets.get("SLACK_WEBHOOK_URL", "") if hasattr(st, "secrets") else os.getenv("SLACK_WEBHOOK_URL", "")

if genai is not None and GEMINI_KEY:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        gen_model = None
else:
    gen_model = None

# ------------------- Sidebar / Inputs -------------------
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Stock Ticker", value="TSLA").upper().strip()
    company_override = st.text_input("Company name (optional)", value="Tesla").strip()
    days = st.slider("Forecast horizon (days)", 3, 21, 7)
    theme = st.radio("Theme", ["Dark", "Light"], index=0)
    st.markdown("---")
    st.markdown("Integrations")
    st.write(f"- Alpha Vantage: {'Configured' if ALPHA_KEY else 'Not configured'}")
    st.write(f"- Gemini: {'Configured' if gen_model is not None else 'Not configured'}")
    st.write(f"- Slack: {'Configured' if SLACK_WEBHOOK else 'Not configured'}")
    if st.button("Run analysis", use_container_width=True):
        st.cache_data.clear()
        st.experimental_rerun()

# Apply light wrapper if needed
if theme == "Light":
    st.markdown("<div class='light'>", unsafe_allow_html=True)
else:
    st.markdown("<div>", unsafe_allow_html=True)

# ------------------- Utilities -------------------
def alpha_get_json(params: dict, retries: int = 2, pause: float = 1.1) -> Optional[dict]:
    """Call Alpha Vantage and return JSON, handle rate-limit 'Note' responses gracefully."""
    if not ALPHA_KEY:
        return None
    base = "https://www.alphavantage.co/query"
    params = params.copy()
    params["apikey"] = ALPHA_KEY
    for attempt in range(retries):
        try:
            r = requests.get(base, params=params, timeout=12)
            if r.status_code != 200:
                time.sleep(pause)
                continue
            data = r.json()
            # API sends {"Note": "..."} when rate limited; handle by pausing
            if isinstance(data, dict) and data.get("Note"):
                time.sleep(pause * (attempt + 1))
                continue
            return data
        except Exception:
            time.sleep(pause)
            continue
    return None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_alpha_daily(symbol: str) -> Optional[pd.DataFrame]:
    """Fetch daily adjusted series from Alpha Vantage and return dataframe with date & Close."""
    data = alpha_get_json({"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "compact"})
    if not data:
        return None
    key = next((k for k in data.keys() if "Time Series" in k), None)
    if not key:
        return None
    ts = data[key]
    rows = []
    for date_str, vals in ts.items():
        rows.append({"date": pd.to_datetime(date_str), "Close": float(vals.get("5. adjusted close", vals.get("4. close")) )})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

@st.cache_data(ttl=300, show_spinner=False)
def fetch_alpha_intraday_latest(symbol: str, interval: str = "5min") -> Optional[float]:
    """Get latest intraday price (close of last interval) from Alpha Vantage."""
    data = alpha_get_json({"function": "TIME_SERIES_INTRADAY", "symbol": symbol, "interval": interval, "outputsize": "compact"})
    if not data:
        return None
    key = next((k for k in data.keys() if "Time Series" in k), None)
    if not key:
        return None
    ts = data[key]
    last_ts = sorted(ts.keys())[-1]
    last_close = float(ts[last_ts].get("4. close"))
    return last_close

@st.cache_data(ttl=900, show_spinner=False)
def fetch_alpha_news_sentiment(symbol: str, limit: int = 8) -> pd.DataFrame:
    """Use Alpha Vantage NEWS_SENTIMENT if available; returns dataframe similar to Google News."""
    data = alpha_get_json({"function": "NEWS_SENTIMENT", "tickers": symbol, "limit": limit})
    if not data:
        return pd.DataFrame()
    feed = data.get("feed", [])
    rows = []
    for item in feed[:limit]:
        title = item.get("title", "")
        summary = item.get("summary", "")
        url = item.get("url", "")
        time_published = item.get("time_published")
        try:
            time_published = pd.to_datetime(time_published)
        except Exception:
            time_published = None
        rows.append({
            "source": "alpha_vantage",
            "title": title,
            "summary": summary,
            "text": f"{title}. {summary}",
            "link": url,
            "published_at": time_published,
            "av_sentiment": item.get("overall_sentiment_score")
        })
    return pd.DataFrame(rows)

def fetch_google_news_rss(company: str, symbol: str, max_items: int = 8) -> pd.DataFrame:
    """Fallback: Google News RSS scraping via feedparser (safe-escaped query)."""
    q = f"{company} {symbol} stock".strip()
    q_safe = re.sub(r'[^A-Za-z0-9 \-_.]', '', q)
    url = f"https://news.google.com/rss/search?q={q_safe.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        rows = []
        for entry in feed.entries[:max_items]:
            rows.append({
                "source": "google_news",
                "title": getattr(entry, "title", ""),
                "summary": getattr(entry, "summary", ""),
                "text": f"{getattr(entry,'title','')}. {getattr(entry,'summary','')}",
                "link": getattr(entry, "link", ""),
                "published_at": None
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ------------------- Sentiment engine (Gemini optional + local heuristic) -------------------
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
def build_corpus_and_sentiment(symbol: str, company: str) -> (pd.DataFrame, float):
    """Build news corpus using Alpha Vantage news (preferred) or Google News fallback; apply sentiment."""
    av_news = fetch_alpha_news_sentiment(symbol) if ALPHA_KEY else pd.DataFrame()
    gnews = fetch_google_news_rss(company, symbol)
    parts = []
    if not av_news.empty:
        parts.append(av_news)
    if not gnews.empty:
        parts.append(gnews)
    if not parts:
        corpus = pd.DataFrame()
    else:
        corpus = pd.concat(parts, ignore_index=True)
    # compute sentiment per item
    scores = []
    for _, row in corpus.iterrows():
        if row.get("av_sentiment") is not None:
            try:
                scores.append(float(row.get("av_sentiment")))
                continue
            except Exception:
                pass
        text = row.get("text", "")
        # Try Gemini for a small number of items if configured
        if gen_model is not None and len(scores) < 6:
            try:
                prompt = f"Rate sentiment -100 to 100. Reply WITH a single integer only.\n\n{text[:1500]}"
                resp = gen_model.generate_content(prompt)
                raw = getattr(resp, "text", "") or str(resp)
                m = re.search(r"-?\d+", raw)
                if m:
                    scores.append(int(m.group()))
                    continue
            except Exception:
                pass
        scores.append(local_sentiment(text))
    corpus["sentiment"] = scores if scores else []
    agg = float(np.mean(scores)) if scores else 0.0
    return corpus, agg

# ------------------- Fetch price & historical -------------------
if not ALPHA_KEY:
    st.warning("Alpha Vantage API key not configured — set ALPHA_VANTAGE_API_KEY in Streamlit Secrets. The app will not function reliably without it.")
    # continue but warn user; calls will fail later

with st.spinner("Fetching historical data from Alpha Vantage..."):
    hist = fetch_alpha_daily(ticker)
    latest_intraday = fetch_alpha_intraday_latest(ticker) if ALPHA_KEY else None

if hist is None or hist.empty:
    st.error(f"Could not fetch historical data for '{ticker}'. Confirm the Symbol and that your Alpha Vantage key is valid. Typical examples: AAPL, TSLA, MSFT, RELIANCE.NS.")
    st.stop()

current_price = float(latest_intraday) if latest_intraday else float(hist["Close"].iloc[-1])
market_cap_display = "N/A"  # Alpha Vantage free-tier does not provide market cap reliably

# ------------------- Forecasting (Holt-Winters) -------------------
def forecast_holtwinters(df: pd.DataFrame, periods: int = 7):
    s = df.copy().reset_index(drop=True)
    s["date"] = pd.to_datetime(s["date"])
    series = s["Close"].astype(float)
    if len(series) < 10:
        return None
    model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
    fit = model.fit()
    pred = fit.forecast(periods)
    dates = pd.date_range(start=s["date"].iloc[-1] + timedelta(days=1), periods=periods, freq="D")
    out = pd.DataFrame({"date": dates, "yhat": pred})
    out["yhat_lower"] = out["yhat"] * 0.92
    out["yhat_upper"] = out["yhat"] * 1.08
    return out

with st.spinner("Building forecast..."):
    forecast_df = forecast_holtwinters(hist, days)
if forecast_df is None:
    st.error("Not enough historical points to forecast.")
    st.stop()

proj_price = float(forecast_df["yhat"].mean())
proj_pct = ((proj_price - current_price) / current_price) * 100 if current_price else 0.0

# ------------------- Sentiment & corpus -------------------
with st.spinner("Gathering news & computing sentiment..."):
    corpus_df, agg_sentiment = build_corpus_and_sentiment(ticker, company_override or ticker)

# ------------------- Compute signal -------------------
def compute_signal(pct: float, sentiment: float):
    if pct > 3 and sentiment > 20:
        return {"signal": "STRONG BUY", "color": "#059669", "reason": "Strong upside with positive sentiment"}
    if pct > 1 and sentiment > 10:
        return {"signal": "BUY", "color": "#059669", "reason": "Moderate upside with supportive sentiment"}
    if pct < -3 and sentiment < -20:
        return {"signal": "STRONG SELL", "color": "#DC2626", "reason": "Strong downside with negative sentiment"}
    if pct < -1 and sentiment < -10:
        return {"signal": "SELL", "color": "#DC2626", "reason": "Moderate downside with negative sentiment"}
    return {"signal": "HOLD", "color": "#D97706", "reason": "Mixed signals"}

signal_info = compute_signal(proj_pct, agg_sentiment)

# ------------------- UI: Header & KPIs -------------------
st.markdown(f"<div class='glass'><div class='title'>{company_override or ticker} — InsightSphere</div><div class='subtitle'>Real-Time Strategic Intelligence</div></div>", unsafe_allow_html=True)
st.write("")

c1, c2, c3, c4 = st.columns(4)
c1.markdown(f"<div class='metric'><div>Price</div><div style='font-size:20px; font-weight:700'>${current_price:,.2f}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='metric'><div>7-day forecast (avg)</div><div style='font-size:20px; font-weight:700'>${proj_price:,.2f}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='metric'><div>Projected move</div><div style='font-size:20px; font-weight:700'>{proj_pct:+.2f}%</div></div>", unsafe_allow_html=True)
c4.markdown(f"<div class='metric'><div>Signal</div><div style='font-size:18px; font-weight:700; color:{signal_info['color']}'>{signal_info['signal']}</div></div>", unsafe_allow_html=True)

st.markdown(f"**Rationale:** {signal_info['reason']}  \n**Aggregate sentiment:** {agg_sentiment:+.1f}")

# ------------------- Plotly composite chart -------------------
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65,0.35], subplot_titles=("Historical vs Forecast", "7-day Forecast"))
fig.add_trace(go.Scatter(x=hist["date"], y=hist["Close"], name="Historical"), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast_df["date"], y=forecast_df["yhat"], name="Forecast", line=dict(dash="dash")), row=1, col=1)
fig.add_trace(go.Scatter(x=list(forecast_df["date"]) + list(forecast_df["date"][::-1]), y=list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1]), fill="toself", fillcolor="rgba(0,102,204,0.12)", line=dict(color="transparent"), showlegend=False), row=1, col=1)
fig.add_trace(go.Bar(x=forecast_df["date"].dt.strftime("%Y-%m-%d"), y=forecast_df["yhat"], name="Forecast Price"), row=2, col=1)
fig.update_layout(height=780, template=("plotly_dark" if theme=="Dark" else "plotly_white"))
st.plotly_chart(fig, use_container_width=True)

# ------------------- AI-generated strategic insights (Gemini) -------------------
def generate_gemini_insights(company: str, ticker: str, price: float, forecast_price: float, pct_change: float, sentiment: float) -> str:
    if gen_model is None:
        return "Gemini not configured. Add GEMINI_API_KEY in Streamlit Secrets to enable AI insights."
    prompt = (
        "You are an enterprise strategic intelligence analyst. Produce a concise executive-grade analysis:\n"
        "1) Executive summary (2 sentences)\n2) Key market drivers (3 bullets)\n3) Risks (3 bullets)\n4) Opportunities (3 bullets)\n5) Recommended actions (one short paragraph)\n\n"
        f"Company: {company}\nTicker: {ticker}\nPrice: {price}\n7-day avg forecast: {forecast_price}\nProjected % change: {pct_change:+.2f}%\nAggregate sentiment: {sentiment:+.1f}\n\nUse clear bullets and short paragraphs."
    )
    try:
        resp = gen_model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"Insight generation failed: {e}"

st.subheader("AI-Generated Strategic Insights")
with st.spinner("Generating insights..."):
    insights = generate_gemini_insights(company_override or ticker, ticker, current_price, proj_price, proj_pct, agg_sentiment)
st.markdown(f"<div class='glass' style='white-space:pre-wrap'>{insights}</div>", unsafe_allow_html=True)

# ------------------- Corpus sample (top 6) -------------------
if corpus_df is not None and not corpus_df.empty:
    st.subheader("News Corpus (sample)")
    sample = corpus_df.head(6).copy()
    sample["text"] = sample["text"].str.slice(0, 300)
    st.dataframe(sample[["source", "published_at", "text"]], use_container_width=True)

# ------------------- Slack alert -------------------
if SLACK_WEBHOOK and st.button("Send Alert to Slack"):
    snippet = insights[:800] if insights else ""
    payload = {"text": f"*InsightSphere Alert*\\n*{company_override or ticker} ({ticker})*\\nSignal: {signal_info['signal']}\\nPrice: ${current_price:.2f} → ${proj_price:.2f} ({proj_pct:+.2f}%)\\nSentiment: {agg_sentiment:+.1f}\\n\\n{snippet}"}
    try:
        r = requests.post(SLACK_WEBHOOK, json=payload, timeout=10)
        if r.status_code == 200:
            st.success("Alert sent to Slack.")
        else:
            st.error(f"Slack returned {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Failed to send Slack alert: {e}")

st.caption("© 2025 Infosys Springboard Internship — Real-Time Strategic Intelligence Dashboard")
st.markdown("</div>", unsafe_allow_html=True)
