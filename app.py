import os
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import wikipedia
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Try Prophet (optional)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Try Gemini (optional)
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
GEMINI_ENABLED = False
GEMINI_MODEL = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")
        GEMINI_ENABLED = True
    except Exception:
        pass

# Slack webhook
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

# CSS
st.markdown(
    f"""
    <style>
    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border-left: 4px solid {PRIMARY_BLUE};
    }}
    .signal-badge {{
        padding: 8px 14px;
        border-radius: 999px;
        color: white;
        font-weight: 700;
    }}
    .small-muted {{ color:#666; font-size:0.9rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------------------------------------------------------
# Utilities
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
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError("No data")
    df = df[["Close"]].reset_index().rename(columns={"Date": "date"})
    return df

@st.cache_data(show_spinner=False)
def fetch_market_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    current_price = t.history(period="1d")["Close"].iloc[-1] if not t.history(period="1d").empty else None
    return {
        "current_price": current_price,
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
        title = entry.get("title", "")
        summary = entry.get("summary", "")
        link = entry.get("link", "")
        pub = datetime(*entry.published_parsed[:6]) if "published_parsed" in entry else None
        rows.append({"source": "news", "title": title, "summary": summary, "text": f"{title}. {summary}", "link": link, "published_at": pub})
    return pd.DataFrame(rows)

def generate_mock_tweets(company: str, days: int = 5, posts_per_day: int = 6) -> pd.DataFrame:
    POS = ["strong earnings", "bullish momentum", "positive guidance", "analysts optimistic", "record demand"]
    NEG = ["weak outlook", "bearish signals", "regulatory issues", "slowing demand", "market concerns"]
    NEU = ["no major change", "watching closely", "stable performance", "in line with expectations"]
    now = datetime.utcnow()
    rows = []
    for d in range(days):
        dt = now - timedelta(days=d)
        for _ in range(posts_per_day):
            r = random.random()
            if r < 0.35: phrase = random.choice(POS)
            elif r < 0.7: phrase = random.choice(NEG)
            else: phrase = random.choice(NEU)
            label = "positive" if phrase in POS else ("negative" if phrase in NEG else "neutral")
            rows.append({"source":"twitter","title":"","summary":"","text":f"{company} shows {phrase} today. ({label})","link":"","published_at":dt - timedelta(minutes=random.randint(0,600))})
    return pd.DataFrame(rows)

def build_corpus(news_df: pd.DataFrame, tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([news_df, tweets_df], ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["text"]).sort_values("published_at", ascending=False).reset_index(drop=True)
    return df

# Sentiment
POS_WORDS = ["growth","strong","bullish","positive","optimistic","profit","surge","beats","outperform"]
NEG_WORDS = ["weak","bearish","loss","regulatory","lawsuit","slowing","concern","fraud","volatility"]

def local_sentiment(text: str) -> int:
    t = (text or "").lower()
    score = sum(10 for w in POS_WORDS if w in t) - sum(10 for w in NEG_WORDS if w in t)
    return max(-100, min(100, score))

def gemini_sentiment_safe(text: str) -> int:
    if not GEMINI_ENABLED or not GEMINI_MODEL:
        return local_sentiment(text)
    try:
        prompt = f"Output one integer between -100 and 100 for the sentiment of this financial text:\n\n{text[:2000]}"
        res = GEMINI_MODEL.generate_content(prompt)
        return int(res.text.strip())
    except Exception:
        return local_sentiment(text)

def apply_sentiment_rate_safe(corpus: pd.DataFrame) -> pd.DataFrame:
    out = corpus.copy()
    scores = []
    llm_used = 0
    llm_budget = 6
    for _, row in out.iterrows():
        if row["source"] == "news" and llm_used < llm_budget:
            val = gemini_sentiment_safe(row["text"])
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

# Wikipedia summary (fixed)
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
        fc = res.forecast(steps=periods)
        dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods)
        return pd.DataFrame({"date": dates, "yhat": fc.values, "yhat_lower": fc.values * 0.95, "yhat_upper": fc.values * 1.05})
    except Exception:
        return None

def build_forecast(market_df: pd.DataFrame, periods: int = 7) -> Tuple[pd.DataFrame, str]:
    fc = prophet_forecast(market_df, periods)
    if fc is not None:
        return fc, "Prophet"
    fc = arima_forecast(market_df, periods)
    if fc is not None:
        return fc, "ARIMA"
    raise RuntimeError("Both forecasting methods failed.")

def compute_projected_move(market_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(forecast_df["yhat"].iloc[-1])
    pct = (proj_price / last_price - 1) * 100
    return {"last_price": last_price, "proj_price": proj_price, "pct_change": pct}

def compute_signal(projected_move: dict, sentiment_score: float) -> dict:
    pct = projected_move["pct_change"]
    s = sentiment_score
    if pct > 3 and s > 20:
        return {"signal": "STRONG BUY", "color": SUCCESS_GREEN, "reason": f"Strong upside ({pct:.2f}%) + positive sentiment ({s:.1f})."}
    if pct > 1 and s > 10:
        return {"signal": "BUY", "color": SUCCESS_GREEN, "reason": f"Moderate upside ({pct:.2f}%) + supportive sentiment ({s:.1f})."}
    if pct < -3 and s < -20:
        return {"signal": "STRONG SELL", "color": DANGER_RED, "reason": f"Strong downside ({pct:.2f}%) + negative sentiment ({s:.1f})."}
    if pct < -1 and s < -10:
        return {"signal": "SELL", "color": DANGER_RED, "reason": f"Moderate downside ({pct:.2f}%) + negative sentiment ({s:.1f})."}
    return {"signal": "HOLD", "color": WARNING_ORANGE, "reason": f"Mixed signals: {pct:.2f}% move, sentiment {s:.1f}."}

def send_slack_alert(company: str, ticker: str, signal_info: dict, projected_move: dict, sentiment_score: float, model_type: str):
    payload = {"text": f"*{company} ({ticker})*\nSignal: *{signal_info['signal']}*\n7D Move: {projected_move['pct_change']:.2f}%\nSentiment: {sentiment_score:.1f}\nModel: {model_type}\n{signal_info['reason']}"}
    if not SLACK_WEBHOOK_URL:
        st.info("Slack preview:")
        st.json(payload)
        return
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload)
        st.success("Slack alert sent!" if r.status_code == 200 else f"Slack error {r.status_code}")
    except Exception as e:
        st.warning(f"Slack failed: {e}")

# UI Functions
def header():
    st.markdown(f"<h1 style='color:{PRIMARY_BLUE};'>Infosys InsightSphere</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Real-Time Industry Insight & Strategic Intelligence Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

def sidebar_inputs():
    st.sidebar.header("Company Input")
    company_name = st.sidebar.text_input("Company Name", value="Tesla, Inc.")
    ticker_in = st.sidebar.text_input("Ticker (optional)", value="TSLA")
    return company_name.strip(), ticker_in.strip().upper()

def validate_and_resolve(company_name: str, ticker_in: str) -> Tuple[str, str]:
    if ticker_in:
        df = fetch_historical_data_yf(ticker_in, "1mo")
        t = yf.Ticker(ticker_in)
        longname = t.info.get("longName", company_name)
        return longname, ticker_in
    if is_invalid_company_name(company_name):
        raise ValueError("Invalid company name.")
    sym, longname = search_ticker_by_company(company_name)
    if not sym:
        raise ValueError("No ticker found.")
    fetch_historical_data_yf(sym, "1mo")
    return longname or company_name, sym

# Pages
def page_company_overview(state):
    st.subheader("Company Overview")
    profile = fetch_market_metrics(state["ticker"])
    wiki = fetch_wikipedia_summary(state["company_name"])
    col1, col2 = st.columns([1.3, 2])
    with col1:
        st.markdown(f"<div class='metric-card'><strong>{state['company_name']}</strong><div class='small-muted'>{state['ticker']}</div></div>", unsafe_allow_html=True)
        st.write("**Sector:**", profile.get("sector", "N/A"))
        st.write("**Industry:**", profile.get("industry", "N/A"))
        mc = profile.get("market_cap")
        mcstr = f"${mc/1e12:.2f}T" if mc and mc >= 1e12 else f"${mc/1e9:.2f}B" if mc and mc >= 1e9 else "N/A"
        st.write("**Market Cap:**", mcstr)
        st.write("**Current Price:**", f"${profile.get('current_price'):.2f}" if profile.get("current_price") else "N/A")
    with col2:
        st.markdown("**Wikipedia Summary**")
        st.write(wiki)

def page_market_and_forecast(state):
    st.subheader("Market Data & Forecast")
    if st.button("Fetch Historical Data"):
        state["market_df"] = fetch_historical_data_yf(state["ticker"], "1y")
        st.success("Data fetched.")
    if "market_df" in state and st.button("Compute 7-Day Forecast"):
        fc, model = build_forecast(state["market_df"])
        state["forecast_df"] = fc
        state["forecast_model"] = model
        st.success(f"Forecast ready ({model}).")
    if "market_df" in state:
        st.dataframe(state["market_df"].tail(90))
    if "forecast_df" in state:
        st.dataframe(state["forecast_df"])

def page_sentiment(state):
    st.subheader("News & Sentiment")
    if st.button("Fetch News & Build Corpus"):
        news = fetch_google_news(f"{state['company_name']} {state['ticker']}")
        tweets = generate_mock_tweets(state["company_name"])
        state["corpus_df_raw"] = build_corpus(news, tweets)
        st.success("Corpus built.")
    if "corpus_df_raw" in state and st.button("Analyze Sentiment"):
        state["corpus_df"] = apply_sentiment_rate_safe(state["corpus_df_raw"])
        state["agg_sentiment"] = aggregate_sentiment(state["corpus_df"])
        st.success("Sentiment analyzed.")
    if "corpus_df_raw" in state:
        st.dataframe(state["corpus_df_raw"].head(10))
    if "corpus_df" in state:
        st.metric("Aggregate Sentiment (-100 to +100)", f"{state['agg_sentiment']:.2f}")

def page_dashboard(state):
    st.subheader("Executive Dashboard")
    required = ["market_df", "forecast_df", "corpus_df", "agg_sentiment"]
    if not all(k in state for k in required):
        st.warning("Complete previous steps first.")
        return
    projected = compute_projected_move(state["market_df"], state["forecast_df"])
    signal = compute_signal(projected, state["agg_sentiment"])
    model = state.get("forecast_model", "N/A")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card'>Current Price<br>${projected['last_price']:.2f}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'>7D Forecast<br>${projected['proj_price']:.2f}</div>", unsafe_allow_html=True)
    with c3:
        arrow = "↑" if projected['pct_change'] > 0 else "↓"
        st.markdown(f"<div class='metric-card'>Projected Move<br>{projected['pct_change']:.2f}% {arrow}</div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card'>Signal<br><span class='signal-badge' style='background:{signal['color']}'>{signal['signal']}</span></div>", unsafe_allow_html=True)
    st.write("**Rationale:**", signal['reason'])

    # Plot
    hist_dates = pd.to_datetime(state["market_df"]["date"])
    hist_close = state["market_df"]["Close"]
    fc_dates = pd.to_datetime(state["forecast_df"]["date"])
    fc_mean = state["forecast_df"]["yhat"]
    fc_low = state["forecast_df"]["yhat_lower"]
    fc_high = state["forecast_df"]["yhat_upper"]

    fig = make_subplots(rows=3, cols=2,
                        subplot_titles=("Price & 7-Day Forecast", "Sentiment Gauge", "Daily Forecast Bars", "Sentiment Buckets"),
                        specs=[[{"colspan":2}, None], [{"type":"indicator"}, {"type":"table"}], [{"type":"bar"}, {"type":"table"}]])

    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, name="Historical", line=dict(color=DARK_BLUE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, name="Forecast", line=dict(color=TEAL, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates.tolist() + fc_dates.tolist()[::-1],
                             y=fc_high.tolist() + fc_low.tolist()[::-1],
                             fill="toself", fillcolor="rgba(0,172,193,0.2)", line=dict(color="rgba(0,0,0,0)"), name="CI"), row=1, col=1)

    fig.add_trace(go.Indicator(mode="gauge+number", value=state["agg_sentiment"],
                               gauge={"axis": {"range": [-100,100]}, "bar": {"color": DARK_BLUE}}, title={"text": "Sentiment"}), row=2, col=1)

    table_metrics = [["Company", state["company_name"]], ["Ticker", state["ticker"]], ["Current Price", f"${projected['last_price']:.2f}"],
                     ["7D Forecast", f"${projected['proj_price']:.2f}"], ["Projected Change", f"{projected['pct_change']:.2f}%"], ["Signal", signal['signal']]]
    fig.add_trace(go.Table(header=dict(values=["Metric","Value"], fill_color=LIGHT_GREY),
                           cells=dict(values=list(zip(*table_metrics)))), row=2, col=2)

    fig.add_trace(go.Bar(x=fc_dates, y=fc_mean, name="Forecast Price"), row=3, col=1)

    buckets = pd.cut(state["corpus_df"]["sentiment"], bins=[-100,-20,20,100], labels=["Bearish","Neutral","Bullish"]).value_counts()
    bucket_table = [["Bullish", buckets.get("Bullish",0)], ["Neutral", buckets.get("Neutral",0)], ["Bearish", buckets.get("Bearish",0)]]
    fig.add_trace(go.Table(header=dict(values=["Bucket","Count"], fill_color=LIGHT_GREY),
                           cells=dict(values=list(zip(*bucket_table)))), row=3, col=2)

    fig.update_layout(height=1100, showlegend=False, title_text=f"Strategic Intelligence Dashboard — {state['company_name']} ({state['ticker']})")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Send Slack Alert"):
        send_slack_alert(state["company_name"], state["ticker"], signal, projected, state["agg_sentiment"], model)

# Main
def main():
    header()
    company_name_input, ticker_input = sidebar_inputs()
    try:
        company_name, ticker = validate_and_resolve(company_name_input, ticker_input)
    except Exception as e:
        st.sidebar.error(f"Validation error: {e}\nUsing default Tesla.")
        company_name, ticker = "Tesla, Inc.", "TSLA"

    st.sidebar.success(f"**Selected:** {company_name} ({ticker})")

    page = st.sidebar.radio("Navigation", ["Company Overview", "Market & Forecast", "Sentiment", "Dashboard"], index=3)

    if "ticker" not in st.session_state or st.session_state["ticker"] != ticker:
        st.session_state.clear()
        st.session_state["company_name"] = company_name
        st.session_state["ticker"] = ticker

    state = st.session_state

    if page == "Company Overview":
        page_company_overview(state)
    elif page == "Market & Forecast":
        page_market_and_forecast(state)
    elif page == "Sentiment":
        page_sentiment(state)
    elif page == "Dashboard":
        page_dashboard(state)

if __name__ == "__main__":
    main()
