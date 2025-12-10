import os
import math
import random
from datetime import datetime, timedelta

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

# =============================================================================
# APP CONFIG & THEME (Infosys-style: blue/teal, clean corporate)
# =============================================================================

st.set_page_config(
    page_title="Infosys InsightSphere - Strategic Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY_BLUE = "#007CC3"
DARK_BLUE = "#003B5C"
TEAL = "#00A1A1"
LIGHT_GREY = "#E5E5E5"
BG_GREY = "#F5F7FA"
WARNING_ORANGE = "#FF8C42"
SUCCESS_GREEN = "#00A676"
DANGER_RED = "#D7263D"

CUSTOM_CSS = f"""
<style>
    .stApp {{
        background-color: {BG_GREY};
        font-family: "Roboto", "Segoe UI", sans-serif;
    }}
    .infobox-card {{
        background: white;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-left: 4px solid {PRIMARY_BLUE};
        margin-bottom: 10px;
    }}
    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border-top: 3px solid {PRIMARY_BLUE};
    }}
    .metric-title {{
        font-size: 0.8rem;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    .metric-value {{
        font-size: 1.4rem;
        font-weight: 600;
        color: {DARK_BLUE};
    }}
    .metric-sub {{
        font-size: 0.8rem;
        color: #888;
    }}
    .signal-badge {{
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 600;
        color: white;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =============================================================================
# SECRETS & GLOBAL FLAGS
# =============================================================================

def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)

GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
SLACK_WEBHOOK_URL = get_secret("SLACK_WEBHOOK_URL")
ALPHA_VANTAGE_API_KEY = get_secret("ALPHA_VANTAGE_API_KEY")

# Prophet availability
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# Gemini availability
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        GEMINI_ENABLED = True
    else:
        GEMINI_ENABLED = False
except Exception:
    GEMINI_ENABLED = False

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

INVALID_COMPANY_KEYWORDS = {
    "abc", "xyz", "test", "testing", "demo", "sample",
    "qwerty", "asdf", "fake", "dummy", "123", "456"
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_invalid_company_name(name: str) -> bool:
    name_clean = (name or "").strip().lower()
    if name_clean.isnumeric() or len(name_clean) < 2:
        return True
    for bad in INVALID_COMPANY_KEYWORDS:
        if bad in name_clean:
            return True
    return False


@st.cache_data(show_spinner=False)
def search_ticker_by_company(company_name: str):
    """Use Yahoo Finance search API to find a ticker for the given company name."""
    try:
        q = company_name.replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
        }
        r = requests.get(url, headers=headers, timeout=5)
        data = r.json()
        for item in data.get("quotes", []):
            if item.get("quoteType") == "EQUITY":
                symbol = item.get("symbol")
                longname = item.get("longname") or item.get("shortname") or symbol
                return symbol.upper(), longname
        return None, None
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def fetch_historical_data(ticker: str, period="1y", interval="1d") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"No market data for {ticker}")
    df = df.dropna().reset_index()
    df = df.rename(columns={"Date": "date"})
    return df


@st.cache_data(show_spinner=False)
def fetch_market_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    last_price = None
    market_cap = None

    try:
        hist = t.history(period="5d")
        if not hist.empty:
            last_price = float(hist["Close"].iloc[-1])
    except Exception:
        pass

    try:
        info = t.info
        market_cap = info.get("marketCap")
    except Exception:
        pass

    return {"current_price": last_price, "market_cap": market_cap}


@st.cache_data(show_spinner=False)
def fetch_google_news(query: str, max_items: int = 15) -> pd.DataFrame:
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        link = entry.get("link", "").strip()
        try:
            pub = datetime(*entry.published_parsed[:6])
        except Exception:
            pub = None
        text = f"{title}. {summary}".replace("\n", " ").strip()
        rows.append(
            {
                "source": "news",
                "title": title,
                "summary": summary,
                "text": text,
                "link": link,
                "published_at": pub,
            }
        )
    return pd.DataFrame(rows)


def generate_mock_tweets(company: str, days=5, posts_per_day=6) -> pd.DataFrame:
    POS = [
        "strong earnings", "bullish momentum", "positive guidance",
        "analysts optimistic", "record demand", "growth accelerating"
    ]
    NEG = [
        "weak outlook", "bearish signals", "regulatory issues",
        "market concerns", "slowing demand", "negative pressure"
    ]
    NEU = [
        "sideways movement", "no major change", "watching closely",
        "stable performance", "in line with expectations"
    ]
    now = datetime.utcnow()
    rows = []
    for d in range(days):
        day = now - timedelta(days=d)
        for _ in range(posts_per_day):
            r = random.random()
            if r < 0.33:
                phrase = random.choice(POS); label = "positive"
            elif r < 0.66:
                phrase = random.choice(NEG); label = "negative"
            else:
                phrase = random.choice(NEU); label = "neutral"
            text = f"{company} shows {phrase} today. ({label})"
            rows.append(
                {
                    "source": "twitter",
                    "title": "",
                    "summary": "",
                    "text": text,
                    "link": "",
                    "published_at": day - timedelta(minutes=random.randint(0, 600)),
                }
            )
    return pd.DataFrame(rows)


def build_corpus(news_df: pd.DataFrame, tweets_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([news_df, tweets_df], ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["text"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


# =============================================================================
# SENTIMENT ENGINE
# =============================================================================

POS_WORDS = [
    "growth", "strong", "bullish", "positive", "optimistic",
    "uptrend", "profit", "beats", "high demand", "surge", "record", "outperform"
]
NEG_WORDS = [
    "weak", "bearish", "negative", "downtrend", "loss", "regulatory",
    "lawsuit", "slowing", "fraud", "concern", "volatility", "pressure"
]


def local_sentiment(text: str) -> int:
    text_l = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in text_l:
            score += 10
    for w in NEG_WORDS:
        if w in text_l:
            score -= 10
    return max(-100, min(100, score))


def gemini_sentiment(text: str) -> int:
    if not GEMINI_ENABLED:
        return local_sentiment(text)
    try:
        prompt = f"""
        You are a financial sentiment model.
        Score the sentiment of the following text.
        Respond with only a single integer between -100 (very negative)
        and 100 (very positive).

        Text:
        {text}
        """
        resp = GEMINI_MODEL.generate_content(prompt)
        val = int(resp.text.strip())
        return max(-100, min(100, val))
    except Exception:
        return local_sentiment(text)


def apply_sentiment(corpus: pd.DataFrame) -> pd.DataFrame:
    scores = []
    for _, row in corpus.iterrows():
        txt = row["text"]
        if row["source"] == "news" and GEMINI_ENABLED:
            scores.append(gemini_sentiment(txt))
        else:
            scores.append(local_sentiment(txt))
    out = corpus.copy()
    out["sentiment"] = scores
    return out


def aggregate_sentiment(corpus: pd.DataFrame) -> float:
    if corpus.empty or "sentiment" not in corpus.columns:
        return 0.0
    return float(max(-100, min(100, corpus["sentiment"].mean())))


# =============================================================================
# FORECASTING ENGINE (PROPHET → ARIMA)
# =============================================================================

def prophet_forecast(market_df: pd.DataFrame, periods: int = 7):
    if not PROPHET_AVAILABLE:
        return None
    try:
        df = market_df[["date", "Close"]].rename(columns={"date": "ds", "Close": "y"})
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods, freq="D")
        fc = m.predict(future)
        fc = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
        return fc.rename(columns={"ds": "date"})
    except Exception:
        return None


def arima_forecast(market_df: pd.DataFrame, periods: int = 7):
    try:
        series = market_df.set_index("date")["Close"].asfreq("D").ffill()
        model = ARIMA(series, order=(1, 1, 1))
        fit = model.fit()
        res = fit.get_forecast(steps=periods)
        mean = res.predicted_mean
        conf = res.conf_int()

        out = pd.DataFrame(
            {
                "date": mean.index.to_pydatetime(),
                "yhat": mean.values,
                "yhat_lower": conf.iloc[:, 0].values,
                "yhat_upper": conf.iloc[:, 1].values,
            }
        )
        return out
    except Exception:
        return None


def build_forecast(market_df: pd.DataFrame, periods: int = 7):
    fc = prophet_forecast(market_df, periods)
    if fc is not None:
        return fc, "Prophet"
    fc = arima_forecast(market_df, periods)
    if fc is not None:
        return fc, "ARIMA"
    raise RuntimeError("Both Prophet and ARIMA forecasting failed.")


# =============================================================================
# SIGNAL ENGINE
# =============================================================================

def compute_projected_move(market_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(forecast_df["yhat"].iloc[-1])
    abs_change = proj_price - last_price
    pct_change = (abs_change / last_price) * 100.0 if last_price != 0 else 0.0
    return {
        "last_price": last_price,
        "proj_price": proj_price,
        "abs_change": abs_change,
        "pct_change": pct_change,
    }


def compute_signal(projected_move: dict, sentiment_score: float) -> dict:
    pct = projected_move["pct_change"]
    sentiment = sentiment_score

    if pct > 3 and sentiment > 20:
        sig = "BUY"
        reason = f"Strong upside of {pct:.1f}% with supportive sentiment ({sentiment:.1f})."
    elif pct < -3 and sentiment < -20:
        sig = "SELL"
        reason = f"Strong downside of {pct:.1f}% with negative sentiment ({sentiment:.1f})."
    elif pct > 1 and sentiment > 10:
        sig = "BUY"
        reason = f"Moderate upside of {pct:.1f}% with positive sentiment ({sentiment:.1f})."
    elif pct < -1 and sentiment < -10:
        sig = "SELL"
        reason = f"Moderate downside of {pct:.1f}% with negative sentiment ({sentiment:.1f})."
    else:
        sig = "HOLD"
        reason = f"Mixed or weak signals: {pct:.1f}% move with sentiment {sentiment:.1f}."

    return {"signal": sig, "reason": reason, "pct_change": pct, "sentiment": sentiment}


# =============================================================================
# SLACK ALERTS
# =============================================================================

def send_slack_alert(
    webhook_url: str,
    company: str,
    ticker: str,
    signal_info: dict,
    projected_move: dict,
    sentiment_score: float,
    model_type: str,
):
    lines = [
        f"*Strategic Intelligence Alert* for *{company}* ({ticker})",
        f"- Signal: *{signal_info['signal']}*",
        f"- Projected 7-Day Move: {projected_move['pct_change']:.2f}%",
        f"- Sentiment Score: {sentiment_score:.2f} (-100 to 100)",
        f"- Forecast Model: {model_type}",
        f"- Rationale: {signal_info['reason']}",
    ]
    payload = {"text": "\n".join(lines)}

    if not webhook_url:
        st.info("Slack webhook not configured. Alert payload preview:")
        st.code(payload, language="json")
        return

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        if resp.status_code != 200:
            st.warning(f"Slack error {resp.status_code}. Printing payload:")
            st.code(payload, language="json")
    except Exception as e:
        st.warning(f"Slack alert error: {e}. Printing payload instead:")
        st.code(payload, language="json")


# =============================================================================
# COMPANY PROFILE
# =============================================================================

@st.cache_data(show_spinner=False)
def fetch_company_profile(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info
    except Exception:
        pass
    return {
        "longName": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "country": info.get("country"),
        "website": info.get("website"),
        "employees": info.get("fullTimeEmployees"),
        "summary": info.get("longBusinessSummary"),
    }


def fetch_wikipedia_summary(name: str) -> str:
    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(name)
        if not results:
            return "Wikipedia summary unavailable."
        page = wikipedia.page(results[0], auto_suggest=False)
        return page.summary
    except Exception:
        return "Wikipedia summary unavailable."


# =============================================================================
# UI HELPERS
# =============================================================================

def metric_card(title: str, value: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def signal_badge(signal: str) -> str:
    sig = (signal or "").upper()
    if sig == "BUY":
        color = SUCCESS_GREEN
    elif sig == "SELL":
        color = DANGER_RED
    else:
        color = WARNING_ORANGE
    return f'<span class="signal-badge" style="background:{color};"> {sig} </span>'


# =============================================================================
# PAGES
# =============================================================================

def page_company_overview(state):
    st.markdown("### 🧭 Company Overview")

    col1, col2 = st.columns([1.4, 2.0])

    with col1:
        st.markdown(
            f"""
            <div class="infobox-card">
                <h3 style="margin-bottom:4px;">{state['company_name']} ({state['ticker']})</h3>
                <div style="font-size:0.9rem;color:#666;">Real-Time Strategic Intelligence Profile</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        profile = fetch_company_profile(state["ticker"])
        wiki_summary = fetch_wikipedia_summary(state["company_name"])

        st.markdown("#### Key Metadata")
        meta_cols = st.columns(2)
        with meta_cols[0]:
            st.markdown(f"**Sector:** {profile.get('sector', 'N/A')}")
            st.markdown(f"**Industry:** {profile.get('industry', 'N/A')}")
        with meta_cols[1]:
            st.markdown(f"**Country:** {profile.get('country', 'N/A')}")
            st.markdown(f"**Employees:** {profile.get('employees', 'N/A')}")

        website = profile.get("website")
        if website:
            st.markdown(f"**Website:** [{website}]({website})")

    with col2:
        st.markdown("#### Wikipedia Summary")
        st.write(wiki_summary)

        st.markdown("#### Business Overview (Yahoo Finance)")
        st.write(profile.get("summary", "Not available."))


def page_market_forecast(state):
    st.markdown("### 📊 Market Data & 7-Day Forecast")

    with st.spinner("Fetching market data and generating forecast..."):
        market_df = fetch_historical_data(state["ticker"])
        metrics = fetch_market_metrics(state["ticker"])
        fc_df, model_type = build_forecast(market_df, periods=7)

    state["market_df"] = market_df
    state["forecast_df"] = fc_df
    state["forecast_model"] = model_type
    state["market_metrics"] = metrics

    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(fc_df["yhat"].iloc[-1])
    pct_change = (proj_price - last_price) / last_price * 100.0 if last_price != 0 else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Current Price", f"${last_price:.2f}", "Latest close")
    with c2:
        metric_card("Forecasted Price (7D)", f"${proj_price:.2f}", f"Model: {model_type}")
    with c3:
        move_arrow = "▲" if pct_change > 0 else ("▼" if pct_change < 0 else "➜")
        metric_card("Projected 7D Move", f"{pct_change:.2f}%", move_arrow)

    st.markdown("#### Historical Price vs Forecast")

    hist_dates = market_df["date"]
    hist_close = market_df["Close"]

    last_hist_date = hist_dates.max()
    future_fc = fc_df[fc_df["date"] > last_hist_date].copy()
    if future_fc.empty:
        future_fc = fc_df.copy()

    fc_dates = future_fc["date"]
    fc_mean = future_fc["yhat"]
    fc_low = future_fc["yhat_lower"]
    fc_high = future_fc["yhat_upper"]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode="lines",
            name="Historical Close",
            line=dict(color=DARK_BLUE),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_mean,
            mode="lines+markers",
            name="Forecast (7D)",
            line=dict(color=TEAL, width=2),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_high) + list(fc_low[::-1]),
            fill="toself",
            fillcolor="rgba(0, 124, 195, 0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Forecast CI",
        )
    )

    fig.update_layout(
        height=420,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor="white",
        paper_bgcolor=BG_GREY,
    )
    fig.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Raw Forecast Data")
    st.dataframe(fc_df.reset_index(drop=True))


def page_sentiment_analysis(state):
    st.markdown("### 🧠 News & Sentiment Analysis")

    company = state["company_name"]
    ticker = state["ticker"]

    with st.spinner("Fetching news and synthesizing sentiment..."):
        news_df = fetch_google_news(f"{company} {ticker} stock", max_items=10)
        tweets_df = generate_mock_tweets(company, days=5, posts_per_day=6)
        corpus = build_corpus(news_df, tweets_df)
        corpus_sent = apply_sentiment(corpus)
        agg_sent = aggregate_sentiment(corpus_sent)

    state["news_df"] = news_df
    state["tweets_df"] = tweets_df
    state["corpus_df"] = corpus_sent
    state["agg_sentiment"] = agg_sent

    c1, c2 = st.columns([1, 2])

    with c1:
        metric_card("Aggregate Sentiment", f"{agg_sent:.2f}", "Range: -100 to 100")

    with c2:
        st.markdown("#### Sentiment Gauge")
        gauge_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=agg_sent,
                title={"text": "Market Sentiment"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {"color": DARK_BLUE},
                    "steps": [
                        {"range": [-100, -25], "color": "rgba(215, 38, 61, 0.35)"},
                        {"range": [-25, 25], "color": "rgba(200, 200, 200, 0.5)"},
                        {"range": [25, 100], "color": "rgba(0, 166, 118, 0.35)"},
                    ],
                },
            )
        )
        gauge_fig.update_layout(height=260, margin=dict(l=40, r=40, t=40, b=20))
        st.plotly_chart(gauge_fig, use_container_width=True)

    st.markdown("#### Recent News (Top 5)")
    if not news_df.empty:
        st.dataframe(news_df[["title", "published_at", "link"]].head(5))
    else:
        st.info("No news articles found.")

    st.markdown("#### Sentiment Corpus (Sample)")
    st.dataframe(corpus_sent.head(12)[["source", "published_at", "text", "sentiment"]])


def page_competitor_insights(state):
    st.markdown("### 🏁 Competitor Insights")

    base_ticker = state["ticker"]
    st.markdown(
        f"Use this view to benchmark **{base_ticker}** against peers on returns and sentiment."
    )

    competitor_input = st.text_input(
        "Enter competitor tickers (comma-separated, e.g., NIO, F, GM):",
        value="",
        help="Optional: Add peer tickers to compare sentiment and 30D returns.",
    )

    tickers = [t.strip().upper() for t in competitor_input.split(",") if t.strip()]
    if not tickers:
        st.info("Enter at least one competitor ticker above to see insights.")
        return

    rows = []
    with st.spinner("Fetching competitor data..."):
        for tkr in tickers:
            try:
                df = fetch_historical_data(tkr, period="3mo")
                if len(df) < 10:
                    continue
                if len(df) > 21:
                    ret_30 = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100.0
                else:
                    ret_30 = np.nan

                news_df = fetch_google_news(f"{tkr} stock", max_items=5)
                tweets_df = generate_mock_tweets(tkr, days=3, posts_per_day=4)
                corpus = build_corpus(news_df, tweets_df)
                corpus_sent = apply_sentiment(corpus)
                agg_sent = aggregate_sentiment(corpus_sent)

                rows.append(
                    {
                        "Ticker": tkr,
                        "30D Return (%)": ret_30,
                        "Sentiment Score": agg_sent,
                        "News Count": int((corpus_sent["source"] == "news").sum()),
                    }
                )
            except Exception:
                continue

    if not rows:
        st.warning("No valid competitor data could be fetched.")
        return

    comp_df = pd.DataFrame(rows)
    st.markdown("#### Competitor Summary Table")
    st.dataframe(comp_df)

    st.markdown("#### Positioning Map: Sentiment vs 30-Day Return")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comp_df["Sentiment Score"],
            y=comp_df["30D Return (%)"],
            mode="markers+text",
            text=comp_df["Ticker"],
            textposition="top center",
            marker=dict(size=14, color=PRIMARY_BLUE),
        )
    )
    fig.update_layout(
        xaxis_title="Sentiment Score (−100 to 100)",
        yaxis_title="30-Day Return (%)",
        height=420,
        plot_bgcolor="white",
        paper_bgcolor=BG_GREY,
    )
    fig.update_xaxes(showgrid=True, gridcolor=LIGHT_GREY)
    fig.update_yaxes(showgrid=True, gridcolor=LIGHT_GREY)
    st.plotly_chart(fig, use_container_width=True)


def page_dashboard_and_alerts(state):
    st.markdown("### 📈 Executive Dashboard & Alerts")

    required_keys = [
        "market_df",
        "forecast_df",
        "corpus_df",
        "agg_sentiment",
        "market_metrics",
        "forecast_model",
    ]
    missing = [k for k in required_keys if k not in state]
    if missing:
        st.warning(
            "Some analytical components are missing. "
            "Please visit **Market & Forecast** and **Sentiment Analysis** pages first."
        )
        return

    market_df = state["market_df"]
    forecast_df = state["forecast_df"]
    corpus_df = state["corpus_df"]
    agg_sentiment = state["agg_sentiment"]
    market_metrics = state["market_metrics"]
    model_type = state["forecast_model"]

    projected_move = compute_projected_move(market_df, forecast_df)
    signal_info = compute_signal(projected_move, agg_sentiment)
    state["signal_info"] = signal_info

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current Price", f"${projected_move['last_price']:.2f}")
    with c2:
        metric_card("Forecasted Price (7D)", f"${projected_move['proj_price']:.2f}", model_type)
    with c3:
        move_arrow = "▲" if projected_move["pct_change"] > 0 else ("▼" if projected_move["pct_change"] < 0 else "➜")
        metric_card("Projected 7D Move", f"{projected_move['pct_change']:.2f}%", move_arrow)
    with c4:
        st.markdown(
            signal_badge(signal_info["signal"]),
            unsafe_allow_html=True,
        )

    st.markdown(f"**Rationale:** {signal_info['reason']}")

    st.markdown("---")
    st.markdown("#### Strategic Intelligence Dashboard")

    hist_dates = market_df["date"]
    hist_close = market_df["Close"]

    last_hist_date = hist_dates.max()
    future_fc = forecast_df[forecast_df["date"] > last_hist_date].copy()
    if future_fc.empty:
        future_fc = forecast_df.copy()

    fc_dates = future_fc["date"]
    fc_mean = future_fc["yhat"]
    fc_low = future_fc["yhat_lower"]
    fc_high = future_fc["yhat_upper"]

    mc_raw = market_metrics.get("market_cap")
    if mc_raw:
        try:
            mc_raw = float(mc_raw)
            if mc_raw >= 1e12:
                market_cap_str = f"${mc_raw/1e12:.2f} Trillion"
            elif mc_raw >= 1e9:
                market_cap_str = f"${mc_raw/1e9:.2f} Billion"
            elif mc_raw >= 1e6:
                market_cap_str = f"${mc_raw/1e6:.2f} Million"
            else:
                market_cap_str = f"${mc_raw:,.0f}"
        except Exception:
            market_cap_str = "N/A"
    else:
        market_cap_str = "N/A"

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        subplot_titles=(
            "Historical Price vs 7-Day Forecast",
            "Aggregate Sentiment Gauge",
            "Executive Metrics Summary",
            "Risk Profile (Synthetic Indices)",
            "Sentiment Breakdown",
        ),
    )

    # Price chart
    fig.add_trace(
        go.Scatter(
            x=hist_dates,
            y=hist_close,
            mode="lines",
            name="Historical Close",
            line=dict(color=DARK_BLUE),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_mean,
            mode="lines+markers",
            name="Forecast (7D)",
            line=dict(color=TEAL),
            marker=dict(size=6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_high) + list(fc_low[::-1]),
            fill="toself",
            fillcolor="rgba(0, 124, 195, 0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="Forecast CI",
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)

    # Sentiment gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=agg_sentiment,
            title={"text": "Sentiment (−100 to 100)"},
            gauge={
                "axis": {"range": [-100, 100]},
                "bar": {"color": DARK_BLUE},
                "steps": [
                    {"range": [-100, -25], "color": "rgba(215, 38, 61, 0.35)"},
                    {"range": [-25, 25], "color": "rgba(200,200,200,0.45)"},
                    {"range": [25, 100], "color": "rgba(0,166,118,0.35)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.7,
                    "value": agg_sentiment,
                },
            },
        ),
        row=2,
        col=1,
    )

    # Executive table
    metrics_header = ["Metric", "Value"]
    metrics_rows = [
        ["Company", state["company_name"]],
        ["Ticker", state["ticker"]],
        ["Market Cap", market_cap_str],
        ["Current Price", f"${projected_move['last_price']:.2f}"],
        ["Forecasted Price (7D)", f"${projected_move['proj_price']:.2f}"],
        ["7-Day Change", f"{projected_move['pct_change']:.2f}%"],
        ["Aggregate Sentiment", f"{agg_sentiment:.2f}"],
        ["Signal", signal_info["signal"]],
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=metrics_header, fill_color=LIGHT_GREY, align="left"),
            cells=dict(values=list(zip(*metrics_rows)), align="left"),
        ),
        row=2,
        col=2,
    )

    # Risk indices
    window_df = market_df.tail(60)
    if len(window_df) > 1:
        returns = window_df["Close"].pct_change().dropna()
        price_vol = float(returns.std() * np.sqrt(252) * 100.0)
    else:
        price_vol = 0.0
    ci_width = float((fc_high - fc_low).mean()) if len(fc_high) else 0.0
    forecast_uncertainty = (
        (ci_width / projected_move["proj_price"] * 100.0)
        if projected_move["proj_price"] != 0
        else 0.0
    )
    sent_vol = float(corpus_df["sentiment"].std()) if len(corpus_df) > 1 else 0.0
    news_count = int((corpus_df["source"] == "news").sum()) if "source" in corpus_df.columns else 0

    risk_labels = ["Price Volatility", "Forecast Uncertainty", "Sentiment Volatility", "News Flow Intensity"]
    risk_values = [
        max(0.0, min(100.0, price_vol / 2.0)),
        max(0.0, min(100.0, forecast_uncertainty)),
        max(0.0, min(100.0, abs(sent_vol))),
        max(0.0, min(100.0, news_count * 5.0)),
    ]
    fig.add_trace(
        go.Bar(x=risk_labels, y=risk_values, name="Risk Indices", marker_color=PRIMARY_BLUE),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Relative Risk (0–100)", row=3, col=1)

    # Sentiment breakdown
    def bucket_sent(v):
        if v >= 20:
            return "Bullish"
        elif v <= -20:
            return "Bearish"
        return "Neutral"

    sent_buckets = corpus_df["sentiment"].apply(bucket_sent)
    counts = sent_buckets.value_counts().to_dict()
    bull_cnt = counts.get("Bullish", 0)
    bear_cnt = counts.get("Bearish", 0)
    neut_cnt = counts.get("Neutral", 0)

    sent_table_header = ["Bucket", "Count"]
    sent_table_rows = [
        ["Bullish", bull_cnt],
        ["Neutral", neut_cnt],
        ["Bearish", bear_cnt],
    ]
    fig.add_trace(
        go.Table(
            header=dict(values=sent_table_header, fill_color=LIGHT_GREY, align="left"),
            cells=dict(values=list(zip(*sent_table_rows)), align="left"),
        ),
        row=3,
        col=2,
    )

    fig.update_layout(
        height=1000,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",
        paper_bgcolor=BG_GREY,
        title=f"Strategic Intelligence Dashboard — {state['company_name']} ({state['ticker']})",
    )

    sig_color = WARNING_ORANGE
    if signal_info["signal"] == "BUY":
        sig_color = SUCCESS_GREEN
    elif signal_info["signal"] == "SELL":
        sig_color = DANGER_RED

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=1.12,
        text=f"Signal: {signal_info['signal']}",
        showarrow=False,
        font=dict(size=14, color="white"),
        align="left",
        bordercolor=sig_color,
        borderwidth=2,
        borderpad=4,
        bgcolor=sig_color,
        opacity=0.95,
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Slack Alert")

    st.write(
        "Send a real-time signal notification to your configured Slack channel. "
        "If no webhook is configured, the payload will be printed instead."
    )
    if st.button("Send Slack Alert Now", type="primary"):
        send_slack_alert(
            SLACK_WEBHOOK_URL,
            state["company_name"],
            state["ticker"],
            signal_info,
            projected_move,
            agg_sentiment,
            model_type,
        )
        st.success("Slack alert processed (or printed as fallback).")


# =============================================================================
# MAIN APP LAYOUT
# =============================================================================

def main():
    st.sidebar.markdown(
        f"""
        <h2 style="color:{PRIMARY_BLUE};margin-bottom:0;">Infosys InsightSphere</h2>
        <div style="font-size:0.8rem;color:#666;margin-bottom:1rem;">
        Real-Time Industry Insight & Strategic Intelligence System
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_company = "Tesla, Inc."
    default_ticker = "TSLA"

    company_input = st.sidebar.text_input("Company Name", value=default_company)
    ticker_input = st.sidebar.text_input("Ticker (optional)", value="")

    if ticker_input.strip():
        ticker = ticker_input.strip().upper()
        company_name = company_input.strip() or default_company
    else:
        if is_invalid_company_name(company_input):
            st.sidebar.error("Invalid or test-like company name. Using default: Tesla, Inc. (TSLA)")
            company_name = default_company
            ticker = default_ticker
        else:
            sym, longname = search_ticker_by_company(company_input)
            if not sym:
                st.sidebar.warning("Could not resolve ticker. Falling back to TSLA (Tesla, Inc.).")
                company_name = default_company
                ticker = default_ticker
            else:
                ticker = sym
                company_name = longname or company_input

    st.sidebar.markdown(f"**Using:** {company_name} (`{ticker}`)")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Company Overview",
            "Market & Forecast",
            "Sentiment Analysis",
            "Competitor Insights",
            "Dashboard & Alerts",
        ],
    )

    state = st.session_state
    state["company_name"] = company_name
    state["ticker"] = ticker

    if page == "Company Overview":
        page_company_overview(state)
    elif page == "Market & Forecast":
        page_market_forecast(state)
    elif page == "Sentiment Analysis":
        page_sentiment_analysis(state)
    elif page == "Competitor Insights":
        page_competitor_insights(state)
    elif page == "Dashboard & Alerts":
        page_dashboard_and_alerts(state)


if __name__ == "__main__":
    main()
