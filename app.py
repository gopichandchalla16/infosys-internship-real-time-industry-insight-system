import os
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import tweepy
import wikipedia
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Try Prophet (optional fallback to ARIMA)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Gemini 2.0 Flash Setup
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_ENABLED = False
GEMINI_MODEL = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        GEMINI_ENABLED = True
    except Exception:
        st.warning("Gemini API key invalid; using local sentiment fallback.")

# Alpha Vantage Setup
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", ""))
AV_BASE_URL = "https://www.alphavantage.co/query"

# Twitter (X) Setup for Real Posts
TWITTER_BEARER_TOKEN = st.secrets.get("TWITTER_BEARER_TOKEN", os.getenv("TWITTER_BEARER_TOKEN", ""))
TWITTER_API_KEY = st.secrets.get("TWITTER_API_KEY", os.getenv("TWITTER_API_KEY", ""))
TWITTER_API_SECRET = st.secrets.get("TWITTER_API_SECRET", os.getenv("TWITTER_API_SECRET", ""))
TWITTER_CLIENT = None
if TWITTER_BEARER_TOKEN and TWITTER_API_KEY and TWITTER_API_SECRET:
    try:
        TWITTER_CLIENT = tweepy.Client(bearer_token=TWITTER_BEARER_TOKEN, consumer_key=TWITTER_API_KEY, consumer_secret=TWITTER_API_SECRET)
    except Exception:
        st.warning("Twitter API setup failed; news corpus will be news-only.")

# Slack Setup
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", ""))

# UI Colors (Infosys-style)
PRIMARY_BLUE = "#007CC3"
DARK_BLUE = "#003B5C"
TEAL = "#00A1A1"
LIGHT_GREY = "#E5E5E5"
BG_GREY = "#F5F7FA"
WARNING_ORANGE = "#FF8C42"
SUCCESS_GREEN = "#00A676"
DANGER_RED = "#D7263D"

st.set_page_config(page_title="Infosys InsightSphere", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS
st.markdown(
    f"""
    <style>
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 5px solid {PRIMARY_BLUE};
        margin: 10px 0;
        text-align: center;
    }}
    .signal-badge {{
        padding: 10px 16px;
        border-radius: 20px;
        color: white;
        font-weight: 700;
        font-size: 16px;
    }}
    .small-muted {{ color:#666; font-size:0.9rem; }}
    .stApp {{ background-color: {BG_GREY}; }}
    h1 {{ color: {PRIMARY_BLUE}; font-family: 'Arial Black', sans-serif; }}
    .stButton > button {{ background-color: {PRIMARY_BLUE}; color: white; border-radius: 8px; padding: 8px 16px; border: none; }}
    .stButton > button:hover {{ background-color: {DARK_BLUE}; }}
    .progress-bar {{ background-color: {LIGHT_GREY}; border-radius: 10px; padding: 3px; }}
    .slack-alert-preview {{ background-color: #f9f9f9; padding: 15px; border-radius: 8px; border-left: 4px solid #36a64f; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Random seeds for reproducibility in any stochastic parts
random.seed(42)
np.random.seed(42)

# -------------------------------------------------------------------------
# Alpha Vantage Functions
# -------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_av_data(function: str, symbol: str = None, **params) -> Dict:
    """Fetch data from Alpha Vantage API."""
    if not ALPHA_VANTAGE_API_KEY:
        return {}
    params.update({"function": function, "apikey": ALPHA_VANTAGE_API_KEY})
    if symbol:
        params["symbol"] = symbol
    try:
        url = f"{AV_BASE_URL}?{'&'.join(f'{k}={v}' for k, v in params.items())}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Alpha Vantage API error: {e}")
        return {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_av_overview(symbol: str) -> Dict:
    """Fetch company overview from Alpha Vantage."""
    data = fetch_av_data("OVERVIEW", symbol)
    return data if data and "Symbol" in data else {}

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_av_news_sentiment(symbol: str, limit: int = 10) -> pd.DataFrame:
    """Fetch real news and sentiment from Alpha Vantage."""
    data = fetch_av_data("NEWS_SENTIMENT", tickers=symbol, limit=limit)
    if not data or "feed" not in data:
        return pd.DataFrame()
    rows = []
    for item in data["feed"][:limit]:
        rows.append({
            "title": item.get("title", ""),
            "summary": item.get("summary", ""),
            "url": item.get("url", ""),
            "time_published": item.get("time_published", ""),
            "sentiment_score": float(item.get("overall_sentiment_score", 0)),
            "sentiment_label": item.get("overall_sentiment_label", "")
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["text"] = df["title"] + ". " + df["summary"]
        df["source"] = "av_news"
        df["published_at"] = pd.to_datetime(df["time_published"], errors="coerce")
    return df

# -------------------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------------------
INVALID_COMPANY_KEYWORDS = {"abc", "xyz", "test", "testing", "demo", "sample", "qwerty", "asdf", "fake", "dummy", "123", "456"}

def is_invalid_company_name(name: str) -> bool:
    if not name or not isinstance(name, str):
        return True
    s = name.strip().lower()
    if s.isnumeric() or len(s) < 2:
        return True
    return any(bad in s for bad in INVALID_COMPANY_KEYWORDS)

@st.cache_data(show_spinner=False)
def search_ticker_by_company(company_name: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        q = company_name.strip().replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=6)
        data = r.json()
        for item in data.get("quotes", []):
            if item.get("quoteType") == "EQUITY":
                sym = item.get("symbol")
                longname = item.get("longname") or item.get("shortname") or sym
                return sym.upper(), longname
    except Exception:
        pass
    return None, None

@st.cache_data(show_spinner=False)
def fetch_historical_data_yf(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError("No data returned for ticker")
    df = df[["Close"]].dropna().reset_index().rename(columns={"Date": "date"})
    return df

@st.cache_data(show_spinner=False)
def fetch_market_metrics(ticker: str) -> Dict:
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="5d")
    last_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
    return {
        "current_price": last_price,
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }

@st.cache_data(show_spinner=False)
def fetch_google_news(query: str, max_items: int = 10) -> pd.DataFrame:
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link = entry.get("link", "").strip()
        try:
            pub_date = datetime(*entry.published_parsed[:6])
        except:
            pub_date = datetime.now()
        text = f"{title}. {summary}".replace("\n", " ").strip()
        rows.append({
            "source": "google_news",
            "title": title,
            "summary": summary,
            "text": text,
            "link": link,
            "published_at": pub_date
        })
    return pd.DataFrame(rows)

@st.cache_data(ttl=300, show_spinner=False)  # Cache tweets for 5 min
def fetch_real_tweets(client: tweepy.Client, query: str, max_results: int = 20) -> pd.DataFrame:
    """Fetch real recent tweets using Tweepy v2."""
    if not client:
        return pd.DataFrame()
    try:
        tweets = client.search_recent_tweets(
            query=query,
            max_results=max_results,
            tweet_fields=["created_at", "author_id", "public_metrics"]
        )
        rows = []
        if tweets.data:
            for tweet in tweets.data:
                text = tweet.text.strip()
                created_at = pd.to_datetime(tweet.created_at)
                rows.append({
                    "source": "twitter",
                    "title": "",
                    "summary": "",
                    "text": text,
                    "link": f"https://twitter.com/i/status/{tweet.id}",
                    "published_at": created_at
                })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("published_at", ascending=False).head(20)
        return df
    except Exception as e:
        st.warning(f"Twitter fetch error: {e}. Using news-only corpus.")
        return pd.DataFrame()

def build_corpus(av_news: pd.DataFrame, google_news: pd.DataFrame, tweets: pd.DataFrame) -> pd.DataFrame:
    """Build unified real corpus from all sources."""
    dfs = [df for df in [av_news, google_news, tweets] if not df.empty]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["text"]).sort_values("published_at", ascending=False).reset_index(drop=True)
    return df

# Local Sentiment Fallback (Lexicon-based, as in notebook)
POS_WORDS = ["growth", "strong", "bullish", "positive", "optimistic", "profit", "surge", "beats", "outperform"]
NEG_WORDS = ["weak", "bearish", "loss", "regulatory", "lawsuit", "slowing", "concern", "fraud", "volatility"]

def local_sentiment(text: str) -> float:
    t = (text or "").lower()
    score = sum(10 for w in POS_WORDS if w in t) - sum(10 for w in NEG_WORDS if w in t)
    return max(-100.0, min(100.0, score))

def gemini_sentiment(text: str) -> float:
    """Real Gemini 2.0 Flash sentiment scoring."""
    if not GEMINI_ENABLED or not GEMINI_MODEL:
        return local_sentiment(text)
    try:
        prompt = f"Analyze the financial sentiment in this text. Output ONLY one integer from -100 (very negative) to +100 (very positive): {text[:2000]}"
        response = GEMINI_MODEL.generate_content(prompt)
        return float(response.text.strip())
    except Exception:
        return local_sentiment(text)

def apply_sentiment(corpus: pd.DataFrame, llm_budget: int = 6) -> pd.DataFrame:
    """Apply sentiment: Gemini for news (limited), local for tweets."""
    out = corpus.copy()
    scores = []
    llm_used = 0
    for _, row in out.iterrows():
        if row["source"] in ["google_news", "av_news"] and llm_used < llm_budget:
            val = gemini_sentiment(row["text"])
            llm_used += 1
        else:
            val = local_sentiment(row["text"])
        scores.append(val)
    out["sentiment"] = scores
    return out

def aggregate_sentiment(corpus: pd.DataFrame) -> float:
    if "sentiment" not in corpus.columns or corpus.empty:
        return 0.0
    return float(corpus["sentiment"].mean().clip(-100, 100))

# Wikipedia Summary
@st.cache_data(show_spinner=False)
def fetch_wikipedia_summary(company_name: str, sentences: int = 4) -> str:
    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(company_name)
        if not results:
            return "No Wikipedia page found."
        page_title = results[0]
        try:
            return wikipedia.summary(page_title, sentences=sentences)
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                return wikipedia.summary(e.options[0], sentences=sentences)
            except:
                return "Summary unavailable (disambiguation)."
        except:
            return "Summary not available."
    except Exception:
        return "Wikipedia summary not available."

# Forecasting
def prophet_forecast(market_df: pd.DataFrame, periods: int = 7) -> Optional[pd.DataFrame]:
    if not PROPHET_AVAILABLE:
        return None
    try:
        df = market_df.rename(columns={"date": "ds", "Close": "y"})
        m = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        fc = m.predict(future)
        return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods).rename(columns={"ds": "date"})
    except Exception:
        return None

def arima_forecast(market_df: pd.DataFrame, periods: int = 7) -> Optional[pd.DataFrame]:
    try:
        series = market_df.set_index("date")["Close"]
        model = ARIMA(series, order=(5,1,0))
        res = model.fit()
        fc_steps = res.forecast(steps=periods)
        dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods, freq="D")
        low = fc_steps * 0.95
        high = fc_steps * 1.05
        return pd.DataFrame({"date": dates, "yhat": fc_steps, "yhat_lower": low, "yhat_upper": high})
    except Exception:
        return None

def build_forecast(market_df: pd.DataFrame, periods: int = 7) -> Tuple[pd.DataFrame, str]:
    fc = prophet_forecast(market_df, periods)
    if fc is not None:
        return fc, "Prophet"
    fc = arima_forecast(market_df, periods)
    if fc is not None:
        return fc, "ARIMA"
    raise RuntimeError("Forecasting failed.")

def compute_projected_move(market_df: pd.DataFrame, forecast_df: pd.DataFrame) -> Dict:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(forecast_df["yhat"].mean())
    pct = ((proj_price - last_price) / last_price) * 100
    return {"last_price": last_price, "proj_price": proj_price, "pct_change": pct}

def compute_signal(projected_move: Dict, sentiment_score: float) -> Dict:
    pct = projected_move["pct_change"]
    s = sentiment_score
    if pct > 3 and s > 20:
        return {"signal": "STRONG BUY", "color": SUCCESS_GREEN, "reason": f"Strong upside ({pct:.2f}%) + positive sentiment ({s:.1f})."}
    elif pct > 1 and s > 10:
        return {"signal": "BUY", "color": SUCCESS_GREEN, "reason": f"Moderate upside ({pct:.2f}%) + supportive sentiment ({s:.1f})."}
    elif pct < -3 and s < -20:
        return {"signal": "STRONG SELL", "color": DANGER_RED, "reason": f"Strong downside ({pct:.2f}%) + negative sentiment ({s:.1f})."}
    elif pct < -1 and s < -10:
        return {"signal": "SELL", "color": DANGER_RED, "reason": f"Moderate downside ({pct:.2f}%) + negative sentiment ({s:.1f})."}
    else:
        return {"signal": "HOLD", "color": WARNING_ORANGE, "reason": f"Mixed signals: {pct:.2f}% move, sentiment {s:.1f}."}

# Slack Alerts
def send_slack_alert(company: str, ticker: str, signal_info: Dict, projected: Dict, sentiment: float, model: str, corpus: pd.DataFrame):
    if not SLACK_WEBHOOK_URL:
        st.info("**Slack Preview** (Webhook not configured):")
        st.markdown(f"""
        <div class='slack-alert-preview'>
            <strong>🚨 Strategic Intelligence Alert</strong><br>
            *Company:* {company} ({ticker})<br>
            *Signal:* {signal_info['signal']}<br>
            *7-Day Projected Move:* {projected['pct_change']:+.2f}% (to ${projected['proj_price']:.2f})<br>
            *Sentiment Score:* {sentiment:.1f}<br>
            *Forecast Model:* {model}<br>
            *Rationale:* {signal_info['reason']}<br><br>
            *Top Insights:*<br>
            {chr(10).join([f"• {t[:100]}..." for t in corpus['text'].head(3).tolist()])}
        </div>
        """, unsafe_allow_html=True)
        return
    blocks = [
        {"type": "header", "text": {"type": "plain_text", "text": f"🚨 {company} ({ticker}) - {signal_info['signal']}"}},
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Current Price:*\n${projected['last_price']:.2f}"},
                {"type": "mrkdwn", "text": f"*7D Forecast:*\n${projected['proj_price']:.2f} ({projected['pct_change']:+.2f}%)"},
                {"type": "mrkdwn", "text": f"*Sentiment:*\n{sentiment:.1f} ({'Positive' if sentiment > 0 else 'Negative'})"},
                {"type": "mrkdwn", "text": f"*Model:*\n{model}"}
            ]
        },
        {"type": "section", "text": {"type": "mrkdwn", "text": f"*Rationale:*\n{signal_info['reason']}"}},
        {"type": "divider"},
        {"type": "section", "text": {"type": "mrkdwn", "text": "*Recent Insights*"}}
    ]
    for _, row in corpus.head(3).iterrows():
        blocks.append({
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"• *{row.get('title', row.get('text', ''))[:100]}...*\n_{row.get('summary', '')[:200]}..._"}
        })
    payload = {
        "username": "InsightSphere Bot",
        "icon_emoji": ":robot_face:",
        "blocks": blocks
    }
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if r.status_code == 200:
            st.success("✅ Slack alert sent!")
        else:
            st.error(f"Slack error: {r.status_code}")
    except Exception as e:
        st.error(f"Slack send failed: {e}")

# UI Functions
def header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.markdown(f"<h1 style='color:{PRIMARY_BLUE}; margin:0;'>📊 InsightSphere</h1>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='small-muted' style='font-size:1.1rem;'>Real-Time Industry Insight & Strategic Intelligence System</div>", unsafe_allow_html=True)
    st.markdown("---")

def sidebar_inputs():
    st.sidebar.header("🎯 Company Configuration")
    company_name = st.sidebar.text_input("Company Name", value="Tesla, Inc.", help="Enter full name for auto-ticker detection")
    ticker_in = st.sidebar.text_input("Ticker (optional)", value="TSLA", help="Overrides auto-detection")
    av_status = "✅ Configured" if ALPHA_VANTAGE_API_KEY else "⚠️ Missing"
    twitter_status = "✅ Configured" if TWITTER_CLIENT else "⚠️ Missing"
    gemini_status = "✅ Enabled (2.0 Flash)" if GEMINI_ENABLED else "⚠️ Key Missing"
    slack_status = "✅ Configured" if SLACK_WEBHOOK_URL else "⚠️ Preview Only"
    st.sidebar.info(f"**Alpha Vantage:** {av_status}\n**Twitter API:** {twitter_status}\n**Gemini LLM:** {gemini_status}\n**Slack:** {slack_status}")
    auto_alert = st.sidebar.checkbox("🚨 Auto-Send Strong Signals to Slack", value=False)
    return company_name.strip(), ticker_in.strip().upper(), auto_alert

def validate_and_resolve(company_name: str, ticker_in: str) -> Tuple[str, str]:
    if ticker_in:
        # Verify ticker
        hist = fetch_historical_data_yf(ticker_in, "1mo")
        t = yf.Ticker(ticker_in)
        longname = t.info.get("longName", company_name)
        return longname, ticker_in
    if is_invalid_company_name(company_name):
        raise ValueError("Invalid company name (too short or test-like).")
    sym, longname = search_ticker_by_company(company_name)
    if not sym:
        raise ValueError(f"No ticker found for '{company_name}'.")
    # Verify real data
    fetch_historical_data_yf(sym, "1mo")
    return longname or company_name, sym

def progress_bar(completed: int, total: int):
    progress = completed / total
    st.progress(progress)
    st.markdown(f"**Progress:** {completed}/{total} steps complete")

# Secrets Validation Function (Newly Added)
def validate_secrets():
    """Non-blocking validation of core secrets with sidebar feedback."""
    warnings = []
    if not GEMINI_API_KEY:
        warnings.append("⚠️ Add GEMINI_API_KEY for LLM sentiment.")
    if not ALPHA_VANTAGE_API_KEY:
        warnings.append("⚠️ Add ALPHA_VANTAGE_API_KEY for enriched news/fundamentals.")
    if not SLACK_WEBHOOK_URL:
        warnings.append("⚠️ Add SLACK_WEBHOOK_URL for alerts (previews work).")
    if warnings:
        st.sidebar.warning("**Secrets Check:** " + " | ".join(warnings))
    else:
        st.sidebar.success("✅ All secrets configured!")

# Pages
def page_company_overview(state):
    st.subheader("🏢 Company Profile")
    profile_yf = fetch_market_metrics(state["ticker"])
    profile_av = fetch_av_overview(state["ticker"])
    wiki = fetch_wikipedia_summary(state["company_name"])
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='margin:0; color:{PRIMARY_BLUE};'>{state['company_name']}</h3>
            <div class='small-muted'>{state['ticker']}</div>
        </div>
        """, unsafe_allow_html=True)
        sector = profile_yf.get("sector", profile_av.get("Sector", "N/A"))
        industry = profile_yf.get("industry", profile_av.get("Industry", "N/A"))
        mc = profile_yf.get("market_cap", profile_av.get("MarketCapitalization", 0))
        mc_str = f"${mc/1e12:.2f}T" if mc >= 1e12 else f"${mc/1e9:.2f}B" if mc >= 1e9 else "N/A"
        st.markdown(f"""
        <div class='metric-card'><strong>Sector:</strong> {sector}</div>
        <div class='metric-card'><strong>Industry:</strong> {industry}</div>
        <div class='metric-card'><strong>Market Cap:</strong> {mc_str}</div>
        <div class='metric-card'><strong>Current Price:</strong> ${profile_yf.get('current_price', 'N/A'):.2f}</div>
        """, unsafe_allow_html=True)
    with col2:
        pe = profile_av.get("PERatio", "N/A")
        eps = profile_av.get("EPS", "N/A")
        beta = profile_av.get("Beta", "N/A")
        dividend = profile_av.get("DividendPerShare", "N/A")
        st.markdown(f"""
        <div class='metric-card'><strong>P/E Ratio:</strong> {pe}</div>
        <div class='metric-card'><strong>EPS:</strong> {eps}</div>
        <div class='metric-card'><strong>Beta:</strong> {beta}</div>
        <div class='metric-card'><strong>Dividend/Share:</strong> {dividend}</div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("**📖 Wikipedia Summary**")
        st.info(wiki)
    progress_bar(1, 4)

def page_market_and_forecast(state):
    st.subheader("📈 Real-Time Market Data & Forecasting")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Fetch 1-Year Historical Data", key="fetch_data"):
            with st.spinner("Fetching real yfinance data..."):
                state["market_df"] = fetch_historical_data_yf(state["ticker"], "1y")
                st.success(f"✅ Loaded {len(state['market_df'])} rows ({state['market_df']['date'].min().date()} to {state['market_df']['date'].max().date()})")
    with col2:
        if "market_df" in state and st.button("🔮 Generate 7-Day Forecast", key="generate_forecast"):
            with st.spinner("Training model..."):
                fc_df, model_type = build_forecast(state["market_df"])
                state["forecast_df"] = fc_df
                state["forecast_model"] = model_type
                st.success(f"✅ {model_type} forecast ready!")
    if "market_df" in state:
        st.markdown("**Historical Close Prices**")
        st.dataframe(state["market_df"].tail(90)[["date", "Close"]], use_container_width=True)
    if "forecast_df" in state:
        st.markdown("**7-Day Forecast with Confidence Intervals**")
        st.dataframe(state["forecast_df"], use_container_width=True)
    progress_bar(2 if "forecast_df" in state else 1, 4)

def page_sentiment(state):
    st.subheader("📰 Real-Time News & Social Sentiment")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📰 Fetch News + Social Posts", key="fetch_news"):
            with st.spinner("Fetching real sources..."):
                # Real AV news if available
                av_news = fetch_av_news_sentiment(state["ticker"], 10)
                # Real Google News
                google_news = fetch_google_news(f"{state['company_name']} {state['ticker']} stock", 10)
                # Real Tweets
                tweet_query = f"{state['ticker']} stock OR {state['company_name']} stock lang:en -is:retweet"
                tweets = fetch_real_tweets(TWITTER_CLIENT, tweet_query, 20)
                state["corpus_df_raw"] = build_corpus(av_news, google_news, tweets)
                st.success(f"✅ Corpus built: {len(state['corpus_df_raw'])} items ({state['corpus_df_raw']['source'].value_counts().to_dict()})")
    with col2:
        if "corpus_df_raw" in state and st.button("🤖 Score Sentiment (Gemini + Local)", key="analyze_sentiment"):
            with st.spinner("Analyzing with Gemini 2.0 Flash..."):
                state["corpus_df"] = apply_sentiment(state["corpus_df_raw"])
                state["agg_sentiment"] = aggregate_sentiment(state["corpus_df"])
                st.success(f"✅ Aggregate Score: {state['agg_sentiment']:.1f} (Gemini used for {sum(state['corpus_df']['source'].isin(['google_news', 'av_news']))} items)")
    if "corpus_df_raw" in state:
        st.markdown("**Unified Corpus Sample (Real Sources)**")
        display_df = state["corpus_df_raw"][["source", "text", "published_at"]].head(10).copy()
        display_df["text"] = display_df["text"].str[:150] + "..."
        st.dataframe(display_df, use_container_width=True)
    if "corpus_df" in state:
        st.metric("Overall Sentiment", f"{state['agg_sentiment']:.1f}", delta=f"{'🔥 Bullish' if state['agg_sentiment'] > 20 else '❄️ Bearish' if state['agg_sentiment'] < -20 else '⚖️ Neutral'}")
        # Breakdown
        def bucket_sent(v):
            if v >= 20: return "Bullish"
            elif v <= -20: return "Bearish"
            return "Neutral"
        buckets = state["corpus_df"]["sentiment"].apply(bucket_sent).value_counts()
        st.bar_chart(buckets)
    progress_bar(3 if "corpus_df" in state else 2, 4)

def page_dashboard(state):
    st.subheader("📊 Executive Dashboard")
    required = ["market_df", "forecast_df", "corpus_df", "agg_sentiment"]
    if not all(k in state for k in required):
        st.warning("⚠️ Complete prior steps for full dashboard.")
        st.stop()
    projected = compute_projected_move(state["market_df"], state["forecast_df"])
    signal = compute_signal(projected, state["agg_sentiment"])
    model = state.get("forecast_model", "N/A")
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>💰 Current Price</h4>
            <h2>${projected['last_price']:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🔮 7D Avg Forecast</h4>
            <h2>${projected['proj_price']:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        delta = projected['pct_change']
        icon = "📈" if delta > 0 else "📉"
        st.markdown(f"""
        <div class='metric-card'>
            <h4>{icon} Projected Change</h4>
            <h2>{delta:+.2f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <h4>🚨 Signal</h4>
            <span class='signal-badge' style='background:{signal["color"]}'>{signal["signal"]}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown(f"**Rationale:** {signal['reason']}")
    # Plotly Dashboard
    hist_dates = pd.to_datetime(state["market_df"]["date"].tail(100))
    hist_close = state["market_df"]["Close"].tail(100)
    fc_dates = pd.to_datetime(state["forecast_df"]["date"])
    fc_mean = state["forecast_df"]["yhat"]
    fc_low = state["forecast_df"]["yhat_lower"]
    fc_high = state["forecast_df"]["yhat_upper"]
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("📉 Price History & Forecast", "📊 Sentiment Gauge", "📊 7-Day Forecast Bars", "🗳️ Sentiment Breakdown"),
        specs=[[{"secondary_y": False}, {"type": "indicator"}], [{"type": "bar"}, {"type": "pie"}]],
        vertical_spacing=0.08, horizontal_spacing=0.1
    )
    # Price Chart
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, name="Historical", line=dict(color=DARK_BLUE, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, name="Forecast", line=dict(color=TEAL, dash="dash", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(fc_dates) + list(fc_dates[::-1]), y=list(fc_high) + list(fc_low[::-1]), fill="toself",
                  fillcolor="rgba(0,161,161,0.2)", line=dict(color="rgba(0,0,0,0)"), name="CI", showlegend=False), row=1, col=1)
    # Gauge
    gauge_color = SUCCESS_GREEN if state["agg_sentiment"] > 0 else DANGER_RED
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=state["agg_sentiment"],
        title={"text": "Sentiment (-100 to +100)"},
        gauge={
            "axis": {"range": [-100, 100]},
            "bar": {"color": gauge_color},
            "steps": [{"range": [-100, -20], "color": DANGER_RED}, {"range": [-20, 20], "color": WARNING_ORANGE}, {"range": [20, 100], "color": SUCCESS_GREEN}],
            "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.75, "value": state["agg_sentiment"]}
        }
    ), row=1, col=2)
    # Bars
    fig.add_trace(go.Bar(x=fc_dates.dt.strftime('%m-%d'), y=fc_mean, marker_color=TEAL, name="Forecast Price"), row=2, col=1)
    # Pie
    def bucket_sent(v):
        if v >= 20: return "Bullish"
        elif v <= -20: return "Bearish"
        return "Neutral"
    buckets = state["corpus_df"]["sentiment"].apply(bucket_sent).value_counts()
    colors = [SUCCESS_GREEN, WARNING_ORANGE, DANGER_RED]
    fig.add_trace(go.Pie(labels=buckets.index, values=buckets.values, hole=0.4, marker_colors=colors), row=2, col=2)
    fig.update_layout(height=800, showlegend=True, title=f"🎯 {state['company_name']} ({state['ticker']}) | {datetime.now().strftime('%B %d, %Y')}")
    st.plotly_chart(fig, use_container_width=True)
    # Actions
    st.subheader("🚨 Alerts & Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📱 Send Slack Alert"):
            send_slack_alert(state["company_name"], state["ticker"], signal, projected, state["agg_sentiment"], model, state["corpus_df"])
    with col2:
        csv = state["corpus_df"].to_csv(index=False)
        st.download_button("📥 Download Insights CSV", csv, f"{state['ticker']}_insights_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
    # Auto-Alert
    if st.session_state.get('auto_alert', False) and signal['signal'] in ['STRONG BUY', 'STRONG SELL']:
        send_slack_alert(state["company_name"], state["ticker"], signal, projected, state["agg_sentiment"], model, state["corpus_df"])
        st.success("🚨 Auto-alert sent!")
        st.balloons()
    progress_bar(4, 4)

# Main App
def main():
    header()
    validate_secrets()  # NEW: Secrets validation on load
    company_name_input, ticker_input, auto_alert = sidebar_inputs()
    st.session_state['auto_alert'] = auto_alert
    try:
        company_name, ticker = validate_and_resolve(company_name_input, ticker_input)
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
    st.sidebar.success(f"**Loaded:** {company_name} ({ticker})")
    page = st.sidebar.selectbox("📋 Step-by-Step Navigation", ["Company Overview", "Market & Forecast", "Sentiment Analysis", "Executive Dashboard"], index=3)
    # Reset state if company changes
    if "ticker" not in st.session_state or st.session_state["ticker"] != ticker:
        for k in list(st.session_state.keys()):
            if k not in ["auto_alert"]:
                del st.session_state[k]
        st.session_state["company_name"] = company_name
        st.session_state["ticker"] = ticker
    state = st.session_state
    if page == "Company Overview":
        page_company_overview(state)
    elif page == "Market & Forecast":
        page_market_and_forecast(state)
    elif page == "Sentiment Analysis":
        page_sentiment(state)
    elif page == "Executive Dashboard":
        page_dashboard(state)
    # Footer
    st.markdown("---")
    st.markdown(f"<div class='small-muted' style='text-align:center;'>Powered by Infosys Springboard | © {datetime.now().year} | Real-Time Data Only</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
