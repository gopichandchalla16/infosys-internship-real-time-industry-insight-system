import os
import json
import random
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# ---------------------------
# Page / Theme configuration
# ---------------------------
st.set_page_config(
    page_title="InsightSphere — Strategic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimal Apple-style CSS
st.markdown(
    """
    <style>
    body { background: #FFFFFF; color: #111827; }
    .stApp { background-color: #FFFFFF; }
    .metric-card { background: #FAFAFA; border-radius: 12px; padding: 14px; box-shadow: 0 1px 4px rgba(17,24,39,0.06); }
    .signal-badge { padding: 8px 14px; border-radius: 12px; color: white; font-weight: 700; }
    h1 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; color: #111827; }
    .small-muted { color: #6B7280; font-size: 0.95rem; }
    .stButton>button { background-color: #111827; color: white; border-radius: 8px; padding: 6px 12px; }
    .stButton>button:hover { opacity: 0.92; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Secrets & configuration
# ---------------------------
# Streamlit secrets (set via .streamlit/secrets.toml or Streamlit Cloud UI)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", "")).strip()

# Toggle safe flags
GEMINI_ENABLED = bool(GEMINI_API_KEY)
ALPHA_VANTAGE_ENABLED = bool(ALPHA_VANTAGE_API_KEY)
SLACK_ENABLED = bool(SLACK_WEBHOOK_URL)

# Random seed
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------
# Utility functions
# ---------------------------
INVALID_COMPANY_KEYWORDS = {
    "abc", "xyz", "test", "testing", "demo", "sample",
    "qwerty", "asdf", "fake", "dummy", "123", "456"
}


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


@st.cache_data(show_spinner=False)
def search_ticker_by_company(company_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a company name to a ticker using Yahoo Finance search API.
    Returns (symbol, official_name) or (None, None) if not found.
    """
    try:
        q = company_name.strip().replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        r = requests.get(url, headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json()
        for item in data.get("quotes", []):
            if item.get("quoteType") == "EQUITY":
                sym = item.get("symbol")
                longname = item.get("longname") or item.get("shortname") or sym
                return sym.upper(), longname
    except Exception:
        # silent fallback
        return None, None
    return None, None

# ---------------------------
# Market data functions
# ---------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_historical_data_yf(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    Returns historical OHLC with 'date' and 'Close' columns.
    """
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df = df.reset_index()
    # Normalize date column
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" not in df.columns and "index" in df.columns:
        df = df.rename(columns={"index": "date"})
    df = df.rename(columns={c: c for c in df.columns})
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    # Keep only date and Close
    if "Close" not in df.columns:
        raise ValueError("Downloaded data has no 'Close' column")
    df = df[["date", "Close"]].dropna().reset_index(drop=True)
    return df


@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_metrics(ticker: str) -> Dict[str, Any]:
    t = yf.Ticker(ticker)
    out = {"current_price": None, "market_cap": None, "sector": None, "industry": None}
    try:
        hist = t.history(period="5d")
        out["current_price"] = float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        out["current_price"] = None
    try:
        # t.info can be slow or deprecated; guard it
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        out["market_cap"] = info.get("marketCap")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
    except Exception:
        pass
    return out

# ---------------------------
# News & corpus
# ---------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fetch_google_news(query: str, max_items: int = 10) -> pd.DataFrame:
    """
    Fetch Google News RSS search results and return a DataFrame.
    """
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link = entry.get("link", "").strip()
        try:
            published_at = datetime(*entry.published_parsed[:6])
        except Exception:
            published_at = None
        text = f"{title}. {summary}".replace("\n", " ").strip()
        rows.append({
            "source": "google_news",
            "title": title,
            "summary": summary,
            "text": text,
            "link": link,
            "published_at": published_at
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def fetch_alpha_vantage_news_sentiment(symbol: str, limit: int = 10) -> pd.DataFrame:
    """
    Use Alpha Vantage NEWS_SENTIMENT endpoint if the key is present.
    Returns DataFrame similar to Google News results.
    """
    if not ALPHA_VANTAGE_ENABLED:
        return pd.DataFrame()
    try:
        base = "https://www.alphavantage.co/query"
        params = {"function": "NEWS_SENTIMENT", "tickers": symbol, "sort": "LATEST", "limit": limit, "apikey": ALPHA_VANTAGE_API_KEY}
        r = requests.get(base, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        feed = data.get("feed", [])
        rows = []
        for item in feed[:limit]:
            title = item.get("title", "")
            summary = item.get("summary", "")
            url = item.get("url", "")
            published_at = item.get("time_published")
            try:
                published_at = pd.to_datetime(published_at)
            except Exception:
                published_at = None
            sentiment_score = None
            try:
                sentiment_score = float(item.get("overall_sentiment_score", 0))
            except Exception:
                sentiment_score = None
            rows.append({
                "source": "av_news",
                "title": title,
                "summary": summary,
                "text": f"{title}. {summary}",
                "link": url,
                "published_at": published_at,
                "av_sentiment": sentiment_score
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def build_corpus(av_news: pd.DataFrame, google_news: pd.DataFrame, tweets_df: pd.DataFrame = None) -> pd.DataFrame:
    parts = []
    if isinstance(av_news, pd.DataFrame) and not av_news.empty:
        parts.append(av_news)
    if isinstance(google_news, pd.DataFrame) and not google_news.empty:
        parts.append(google_news)
    if tweets_df is not None and isinstance(tweets_df, pd.DataFrame) and not tweets_df.empty:
        parts.append(tweets_df)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["text"]).sort_values("published_at", ascending=False).reset_index(drop=True)
    return df

# ---------------------------
# Sentiment engine
# ---------------------------
POS_WORDS = ["growth", "strong", "bullish", "positive", "optimistic", "profit", "surge", "beats", "outperform", "record"]
NEG_WORDS = ["weak", "bearish", "loss", "regulatory", "lawsuit", "slowing", "concern", "fraud", "volatility", "drop"]


def local_sentiment(text: str) -> float:
    if not text:
        return 0.0
    t = (text or "").lower()
    score = 0
    for w in POS_WORDS:
        if w in t:
            score += 10
    for w in NEG_WORDS:
        if w in t:
            score -= 10
    return float(max(-100, min(100, score)))


def gemini_sentiment_safe(text: str) -> float:
    """
    Call Gemini LLM for sentiment if configured. Must return a float in [-100, 100].
    Uses a minimal prompt and conservative parsing. Falls back to local_sentiment.
    """
    if not GEMINI_ENABLED:
        return local_sentiment(text)
    try:
        # local import to avoid importing google.generativeai when not needed
        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a safe minimal request; the SDK may vary — guard against failures.
        prompt = (
            "You are a financial sentiment model. "
            "Read the following text and respond with a single integer between -100 (very negative) and 100 (very positive).\n\n"
            f"Text:\n{text[:1800]}"
        )
        # The SDK has different surface across versions; try a conservative call pattern
        try:
            resp = genai.generate(model="models/text-bison-001", input=prompt)
            raw = getattr(resp, "output", None) or getattr(resp, "text", None) or str(resp)
            raw = str(raw)
        except Exception:
            # fallback to older interface
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", "")
        raw = raw.strip()
        m = re.search(r"-?\d+", raw)
        if m:
            val = int(m.group(0))
            return float(max(-100, min(100, val)))
    except Exception:
        return local_sentiment(text)
    return local_sentiment(text)


def apply_sentiment(corpus: pd.DataFrame, llm_budget: int = 6) -> pd.DataFrame:
    """
    Apply sentiment scoring:
    - Use Alpha Vantage's sentiment if available in av_news rows
    - Use Gemini for a limited number of news items (llm_budget)
    - Use local_sentiment otherwise (tweets and remaining news)
    """
    if corpus is None or corpus.empty:
        return pd.DataFrame()
    out = corpus.copy()
    scores = []
    llm_used = 0
    for _, r in out.iterrows():
        text = r.get("text", "")
        src = r.get("source", "")
        # If Alpha Vantage provided sentiment, prefer it
        if src == "av_news" and pd.notna(r.get("av_sentiment")):
            try:
                scores.append(float(r.get("av_sentiment")))
                continue
            except Exception:
                pass
        # Use Gemini sparingly on news
        if src in ("google_news", "av_news") and GEMINI_ENABLED and llm_used < llm_budget:
            val = gemini_sentiment_safe(text)
            llm_used += 1
            scores.append(val)
            continue
        # Fallback to local heuristic
        scores.append(local_sentiment(text))
    out["sentiment"] = scores
    return out


def aggregate_sentiment(corpus: pd.DataFrame) -> float:
    if corpus is None or corpus.empty or "sentiment" not in corpus.columns:
        return 0.0
    return float(max(-100.0, min(100.0, corpus["sentiment"].mean())))

# ---------------------------
# Wikipedia
# ---------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_wikipedia_summary(company_name: str, sentences: int = 3) -> str:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        results = wikipedia.search(company_name)
        if not results:
            return "No Wikipedia page found."
        page = results[0]
        try:
            return wikipedia.summary(page, sentences=sentences)
        except Exception:
            # Try first disambiguation option
            try:
                return wikipedia.summary(results[0], sentences=sentences)
            except Exception:
                return "Summary not available."
    except Exception:
        return "Wikipedia not available."
    return "Summary not available."

# ---------------------------
# Forecasting (ARIMA only)
# ---------------------------

def arima_forecast(market_df: pd.DataFrame, periods: int = 7) -> Optional[pd.DataFrame]:
    """
    Fit an ARIMA model and produce a simple forecast. Returns DataFrame with date,yhat,yhat_lower,yhat_upper.
    """
    try:
        df = market_df.copy().set_index("date")
        series = df["Close"].asfreq("D").ffill()
        # choose small order to keep stable
        model = ARIMA(series, order=(1, 1, 1))
        res = model.fit()
        fc = res.get_forecast(steps=periods)
        mean = fc.predicted_mean
        ci = fc.conf_int(alpha=0.05)
        dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods, freq="D")
        out = pd.DataFrame({
            "date": dates,
            "yhat": mean.values,
            "yhat_lower": ci.iloc[:, 0].values,
            "yhat_upper": ci.iloc[:, 1].values
        })
        return out
    except Exception:
        return None


def build_forecast(market_df: pd.DataFrame, periods: int = 7) -> Tuple[pd.DataFrame, str]:
    """
    Currently ARIMA is the production model. Returns (forecast_df, model_name)
    """
    fc = arima_forecast(market_df, periods=periods)
    if fc is not None:
        return fc, "ARIMA"
    raise RuntimeError("Forecasting failed (ARIMA).")


def compute_projected_move(market_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict[str, float]:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(forecast_df["yhat"].mean())
    pct = ((proj_price - last_price) / last_price) * 100.0 if last_price != 0 else 0.0
    return {"last_price": last_price, "proj_price": proj_price, "pct_change": pct}


def compute_signal(projected: Dict[str, float], sentiment_score: float) -> Dict[str, Any]:
    pct = projected["pct_change"]
    s = sentiment_score
    if pct > 3 and s > 20:
        return {"signal": "STRONG BUY", "color": "#059669", "reason": f"Strong upside {pct:.2f}% with positive sentiment {s:.1f}"}
    if pct > 1 and s > 10:
        return {"signal": "BUY", "color": "#059669", "reason": f"Moderate upside {pct:.2f}% with supportive sentiment {s:.1f}"}
    if pct < -3 and s < -20:
        return {"signal": "STRONG SELL", "color": "#DC2626", "reason": f"Strong downside {pct:.2f}% with negative sentiment {s:.1f}"}
    if pct < -1 and s < -10:
        return {"signal": "SELL", "color": "#DC2626", "reason": f"Moderate downside {pct:.2f}% with negative sentiment {s:.1f}"}
    return {"signal": "HOLD", "color": "#D97706", "reason": f"Mixed signals ({pct:.2f}%) and sentiment {s:.1f}"}

# ---------------------------
# Slack Alerts
# ---------------------------

def send_slack_alert(company: str, ticker: str, signal_info: Dict[str, Any], projected: Dict[str, float], sentiment: float, model: str, corpus: pd.DataFrame):
    """
    Send alert to Slack if SLACK_WEBHOOK_URL configured, otherwise show preview in UI.
    """
    title = f"{company} ({ticker}) - {signal_info['signal']}"
    text = (
        f"*{company}* ({ticker})\n"
        f"*Signal:* {signal_info['signal']}\n"
        f"*7D Forecast Move:* {projected['pct_change']:+.2f}% → ${projected['proj_price']:.2f}\n"
        f"*Sentiment:* {sentiment:.1f}\n"
        f"*Model:* {model}\n"
        f"*Reason:* {signal_info['reason']}\n"
    )
    # Compose a brief top-3 insights
    top_insights = []
    try:
        for _, row in corpus.head(3).iterrows():
            top_insights.append(f"- {str(row.get('title') or row.get('text'))[:160]}")
    except Exception:
        top_insights = []
    text += "\n*Top insights:*\n" + ("\n".join(top_insights) if top_insights else "- none -")
    if not SLACK_ENABLED:
        st.info("Slack webhook not configured — preview below")
        st.markdown(f"**Alert preview**\n\n{text}")
        return
    payload = {"text": text}
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if r.status_code == 200:
            st.success("Slack alert sent.")
        else:
            st.error(f"Slack returned {r.status_code}: {r.text}")
    except Exception as e:
        st.error(f"Failed to send Slack alert: {e}")

# ---------------------------
# UI - Sidebar and Pages
# ---------------------------

def sidebar_inputs(default_company: str = "Tesla, Inc.", default_ticker: str = "TSLA"):
    st.sidebar.header("Company")
    company_name = st.sidebar.text_input("Company name", value=default_company)
    ticker_in = st.sidebar.text_input("Ticker (optional)", value=default_ticker)
    auto_alert = st.sidebar.checkbox("Auto-send strong signals to Slack", value=False)
    st.sidebar.markdown("### Integrations")
    st.sidebar.markdown(f"- Alpha Vantage: {'Enabled' if ALPHA_VANTAGE_ENABLED else 'Not configured'}")
    st.sidebar.markdown(f"- Gemini LLM: {'Enabled' if GEMINI_ENABLED else 'Not configured'}")
    st.sidebar.markdown(f"- Slack: {'Configured' if SLACK_ENABLED else 'Not configured (preview only)'}")
    return company_name.strip(), ticker_in.strip().upper(), auto_alert


def page_company_overview(state: dict):
    st.header("Company Overview")
    st.markdown(f"**{state['company_name']} ({state['ticker']})**")
    # Fetch metrics
    with st.spinner("Fetching company metrics..."):
        metrics = fetch_market_metrics(state["ticker"])
        wiki = fetch_wikipedia_summary(state["company_name"])
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"<div class='metric-card'><strong>Current Price</strong><div style='font-size:20px'>${metrics.get('current_price') or 'N/A'}</div></div>", unsafe_allow_html=True)
        mc = metrics.get("market_cap")
        mc_str = "N/A"
        try:
            if mc and mc > 0:
                if mc >= 1e12:
                    mc_str = f"${mc/1e12:.2f} T"
                elif mc >= 1e9:
                    mc_str = f"${mc/1e9:.2f} B"
                else:
                    mc_str = f"${mc:,.0f}"
        except Exception:
            mc_str = "N/A"
        st.markdown(f"<div class='metric-card'><strong>Market Cap</strong><div style='font-size:16px'>{mc_str}</div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-card'><strong>Sector</strong><div style='font-size:14px'>{metrics.get('sector') or 'N/A'}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("**Wikipedia Summary**")
        st.info(wiki)
    st.markdown("---")
    st.write("Progress: Sprint 1 completed — data sourcing & cleaning.")
    st.caption("Use the Market & Forecast page to fetch historical data and generate forecasts.")


def page_market_and_forecast(state: dict):
    st.header("Market & Forecast")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Fetch 1-year historical data"):
            try:
                df = fetch_historical_data_yf(state["ticker"], period="1y")
                state["market_df"] = df
                st.success(f"Loaded {len(df)} rows ({df['date'].min().date()} → {df['date'].max().date()})")
            except Exception as e:
                st.error(f"Failed to fetch historical data: {e}")
    with col2:
        if "market_df" in state and st.button("Generate 7-day forecast"):
            try:
                fc, model = build_forecast(state["market_df"], periods=7)
                state["forecast_df"] = fc
                state["forecast_model"] = model
                st.success(f"Forecast ready ({model})")
            except Exception as e:
                st.error(f"Forecast failed: {e}")
    if "market_df" in state:
        st.markdown("### Historical (last 90 rows)")
        st.dataframe(state["market_df"].tail(90), use_container_width=True)
    if "forecast_df" in state:
        st.markdown("### Forecast (7 days)")
        st.dataframe(state["forecast_df"], use_container_width=True)


def page_sentiment(state: dict):
    st.header("Market & Sentiment Corpus")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Build corpus (AlphaVantage + Google News)"):
            try:
                av_news = fetch_alpha_vantage_news_sentiment(state["ticker"], limit=10) if ALPHA_VANTAGE_ENABLED else pd.DataFrame()
                gnews = fetch_google_news(f"{state['company_name']} {state['ticker']} stock", max_items=12)
                corpus = build_corpus(av_news, gnews, tweets_df=None)
                state["corpus_raw"] = corpus
                st.success(f"Corpus built: {len(corpus)} items")
            except Exception as e:
                st.error(f"Failed to build corpus: {e}")
    with c2:
        if "corpus_raw" in state and st.button("Analyze sentiment (Gemini + local)"):
            try:
                scored = apply_sentiment(state["corpus_raw"], llm_budget=6)
                state["corpus"] = scored
                state["agg_sentiment"] = aggregate_sentiment(scored)
                st.success(f"Aggregate sentiment: {state['agg_sentiment']:.2f}")
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
    if "corpus_raw" in state:
        st.markdown("### Corpus sample")
        df = state["corpus_raw"].copy()
        if not df.empty:
            df = df.head(10)
            df["text"] = df["text"].str.slice(0, 200)
            st.dataframe(df[["source", "published_at", "text"]], use_container_width=True)
    if "corpus" in state:
        st.metric("Aggregate Sentiment", f"{state['agg_sentiment']:.2f}")
        st.markdown("### Sentiment distribution")
        hist = state["corpus"]["sentiment"].dropna()
        if not hist.empty:
            st.bar_chart(pd.DataFrame(hist).rename(columns={"sentiment": "score"}))


def page_dashboard(state: dict):
    st.header("Executive Dashboard")
    # Required pieces
    required = ["market_df", "forecast_df", "corpus", "agg_sentiment"]
    if not all(k in state for k in required):
        st.warning("Please complete Market & Forecast and Sentiment pages before the dashboard.")
        return
    market_df = state["market_df"]
    forecast_df = state["forecast_df"]
    corpus = state["corpus"]
    agg_sent = state["agg_sentiment"]
    projected = compute_projected_move(market_df, forecast_df)
    signal_info = compute_signal(projected, agg_sent)
    model_type = state.get("forecast_model", "ARIMA")
    # KPI row
    k1, k2, k3, k4 = st.columns([1, 1, 1, 1])
    with k1:
        st.markdown(f"<div class='metric-card'><strong>Current Price</strong><div style='font-size:20px'>${projected['last_price']:.2f}</div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='metric-card'><strong>7D Avg Forecast</strong><div style='font-size:20px'>${projected['proj_price']:.2f}</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='metric-card'><strong>Projected Move</strong><div style='font-size:18px'>{projected['pct_change']:+.2f}%</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='metric-card'><strong>Signal</strong><div style='margin-top:8px'><span class='signal-badge' style='background:{signal_info['color']}'>{signal_info['signal']}</span></div></div>", unsafe_allow_html=True)
    st.markdown(f"**Rationale:** {signal_info['reason']}")
    # Plotly composite
    hist_dates = pd.to_datetime(market_df["date"].tail(180))
    hist_close = market_df["Close"].tail(180)
    fc_dates = pd.to_datetime(forecast_df["date"])
    fc_mean = forecast_df["yhat"]
    fc_low = forecast_df["yhat_lower"]
    fc_high = forecast_df["yhat_upper"]
    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"colspan": 2}, None],
                               [{"type": "bar"}, {"type": "pie"}]],
                        subplot_titles=("Historical vs Forecast", "7-Day Forecast", "Sentiment Breakdown"))
    # Price + CI
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, mode="lines", name="Historical", line=dict(color="#111827")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, mode="lines+markers", name="Forecast", line=dict(color="#0066CC", dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(fc_dates) + list(fc_dates[::-1]),
                             y=list(fc_high) + list(fc_low[::-1]),
                             fill="toself", fillcolor="rgba(0,102,204,0.12)", line=dict(color="rgba(0,0,0,0)"),
                             showlegend=False, name="CI"), row=1, col=1)
    # Forecast bars
    fig.add_trace(go.Bar(x=fc_dates.dt.strftime("%Y-%m-%d"), y=fc_mean, name="Forecast Price", marker_color="#0066CC"), row=2, col=1)
    # Sentiment pie
    def bucket(v):
        if v >= 20: return "Bullish"
        if v <= -20: return "Bearish"
        return "Neutral"
    buckets = corpus["sentiment"].apply(bucket).value_counts() if "sentiment" in corpus.columns else pd.Series()
    labels = buckets.index.tolist()
    values = buckets.values.tolist()
    colors_map = {"Bullish": "#059669", "Neutral": "#D97706", "Bearish": "#DC2626"}
    fig.add_trace(go.Pie(labels=labels, values=values, marker=dict(colors=[colors_map.get(l, "#9CA3AF") for l in labels]), hole=0.4), row=2, col=2)
    fig.update_layout(height=800, showlegend=True, title_text=f"{state['company_name']} ({state['ticker']}) — Executive Dashboard")
    st.plotly_chart(fig, use_container_width=True)
    # Actions
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Send Slack Alert"):
            send_slack_alert(state["company_name"], state["ticker"], signal_info, projected, agg_sent, model_type, corpus)
    with c2:
        csv = corpus.to_csv(index=False)
        st.download_button("Download Corpus CSV", csv, file_name=f"{state['ticker']}_corpus_{datetime.utcnow().strftime('%Y%m%d')}.csv", mime="text/csv")
    # Auto-alert if enabled and strong signal
    if state.get("auto_alert", False) and signal_info["signal"] in ("STRONG BUY", "STRONG SELL"):
        send_slack_alert(state["company_name"], state["ticker"], signal_info, projected, agg_sent, model_type, corpus)
        st.success("Auto-alert sent for strong signal.")

# ---------------------------
# Main
# ---------------------------

def main():
    # Header
    st.title("InsightSphere")
    st.markdown("<div class='small-muted'>Real-Time Strategic Intelligence — Executive Insights</div>", unsafe_allow_html=True)
    st.markdown("---")
    # Sidebar inputs
    company_name_in, ticker_in, auto_alert = sidebar_inputs()
    # Validate and resolve
    try:
        if ticker_in:
            # verify ticker quickly
            hist = fetch_historical_data_yf(ticker_in, period="1mo")
            tinfo = {}
            try:
                tinfo = yf.Ticker(ticker_in).info or {}
            except Exception:
                tinfo = {}
            company_name = tinfo.get("longName") or company_name_in or ticker_in
            ticker = ticker_in.upper()
        else:
            if is_invalid_company_name(company_name_in):
                raise ValueError("Invalid company name (too short or test-like). Please supply a real company name.")
            sym, longname = search_ticker_by_company(company_name_in)
            if not sym:
                raise ValueError(f"Could not resolve a ticker for '{company_name_in}'. Please enter a valid ticker.")
            # verify data exists
            hist = fetch_historical_data_yf(sym, period="1mo")
            company_name = longname or company_name_in
            ticker = sym
    except Exception as e:
        st.error(f"Company validation error: {e}")
        st.stop()

    # Session state initialization
    if "ticker" not in st.session_state or st.session_state.get("ticker") != ticker:
        # reset except for auto_alert flag
        preserved = {"auto_alert": auto_alert}
        st.session_state.clear()
        st.session_state.update(preserved)
        st.session_state["company_name"] = company_name
        st.session_state["ticker"] = ticker
        st.session_state["auto_alert"] = auto_alert
    else:
        st.session_state["auto_alert"] = auto_alert

    state = st.session_state
    # Navigation
    page = st.sidebar.radio("Navigation", ["Company Overview", "Market & Forecast", "Sentiment", "Dashboard"], index=3)

    # Render page
    try:
        if page == "Company Overview":
            page_company_overview(state)
        elif page == "Market & Forecast":
            page_market_and_forecast(state)
        elif page == "Sentiment":
            page_sentiment(state)
        elif page == "Dashboard":
            page_dashboard(state)
        else:
            st.info("Select a page from the sidebar.")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

    st.markdown("---")
    st.markdown("<div class='small-muted' style='text-align:center;'>Infosys InsightSphere — Enterprise Edition</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
