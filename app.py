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
GEMINI_API_KEY = None
GEMINI_ENABLED = False
try:
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))
    if GEMINI_API_KEY:
        import google.generativeai as genai  # may fail if not installed
        genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash")
        GEMINI_ENABLED = True
except Exception:
    GEMINI_ENABLED = False

# Slack webhook
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", ""))

# Alpha Vantage key (not required; optional)
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", ""))

# UI Colors (Infosys-style)
PRIMARY_BLUE = "#007CC3"
DARK_BLUE = "#003B5C"
TEAL = "#00A1A1"
LIGHT_GREY = "#E5E5E5"
BG_GREY = "#F5F7FA"
WARNING_ORANGE = "#FF8C42"
SUCCESS_GREEN = "#00A676"
DANGER_RED = "#D7263D"

st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Lightweight CSS for nicer metric cards and buttons
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
    """Query Yahoo search API to find a plausible ticker and official company name."""
    try:
        q = company_name.strip().replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
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
def fetch_historical_data_yf(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLC using yfinance and normalize to DataFrame with 'date' and 'Close'."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df = df.dropna().reset_index().rename(columns={"Date": "date"})
    # Keep only Date & Close to keep things predictable
    df = df[["date", "Close"]]
    return df

@st.cache_data(show_spinner=False)
def fetch_market_metrics(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    out = {}
    try:
        hist = t.history(period="5d")
        out["current_price"] = float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        out["current_price"] = None
    try:
        info = t.info
        out["market_cap"] = info.get("marketCap")
        out["sector"] = info.get("sector")
        out["industry"] = info.get("industry")
    except Exception:
        out.setdefault("market_cap", None)
    return out

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
        try:
            pub = datetime(*entry.published_parsed[:6])
        except Exception:
            pub = None
        rows.append({"source": "news", "title": title, "summary": summary, "text": f"{title}. {summary}", "link": link, "published_at": pub})
    return pd.DataFrame(rows)

def generate_mock_tweets(company: str, days:int=5, posts_per_day:int=6) -> pd.DataFrame:
    POS = ["strong earnings","bullish momentum","positive guidance","analysts optimistic","record demand"]
    NEG = ["weak outlook","bearish signals","regulatory issues","slowing demand","market concerns"]
    NEU = ["no major change","watching closely","stable performance","in line with expectations"]
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

# ---------------------
# Sentiment engine
# ---------------------
POS_WORDS = ["growth","strong","bullish","positive","optimistic","profit","surge","beats","outperform"]
NEG_WORDS = ["weak","bearish","loss","regulatory","lawsuit","slowing","concern","fraud","volatility"]

def local_sentiment(text:str) -> int:
    t = (text or "").lower()
    score = 0
    for w in POS_WORDS:
        if w in t: score += 10
    for w in NEG_WORDS:
        if w in t: score -= 10
    return max(-100, min(100, score))

def gemini_sentiment_safe(text:str) -> int:
    """Call Gemini safely, fallback to local_sentiment on error or rate-limit."""
    if not GEMINI_ENABLED:
        return local_sentiment(text)
    try:
        prompt = f"You are a financial sentiment model. Output one integer between -100 and 100 for the sentiment of this text:\n\n{text}"
        res = GEMINI_MODEL.generate_content(prompt)
        return int(res.text.strip())
    except Exception:
        return local_sentiment(text)

def apply_sentiment_rate_safe(corpus:pd.DataFrame) -> pd.DataFrame:
    """Use Gemini only on a small set of news items; tweets use local scoring."""
    out = corpus.copy()
    scores = []
    llm_budget = 6  # limit LLM calls to small number
    llm_used = 0
    for _, row in out.iterrows():
        if row.get("source") == "news" and GEMINI_ENABLED and llm_used < llm_budget:
            try:
                val = gemini_sentiment_safe(row["text"])
                llm_used += 1
                scores.append(val)
            except Exception:
                scores.append(local_sentiment(row["text"]))
        else:
            scores.append(local_sentiment(row["text"]))
    out["sentiment"] = scores
    return out

def aggregate_sentiment(corpus:pd.DataFrame) -> float:
    if corpus is None or corpus.empty or "sentiment" not in corpus.columns:
        return 0.0
    return float(max(-100, min(100, corpus["sentiment"].mean())))

# ---------------------
# Forecasting: Prophet primary, ARIMA fallback
# ---------------------
def prophet_forecast(market_df:pd.DataFrame, periods:int=7) -> Optional[pd.DataFrame]:
    if not PROPHET_AVAILABLE:
        return None
    try:
        df = market_df[["date","Close"]].rename(columns={"date":"ds","Close":"y"})
        m = Prophet(daily_seasonality=True)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        fc = m.predict(future)
        fc = fc[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods).rename(columns={"ds":"date"})
        return fc
    except Exception:
        return None

def arima_forecast(market_df:pd.DataFrame, periods:int=7) -> Optional[pd.DataFrame]:
    try:
        series = market_df.set_index("date")["Close"].asfreq("D").ffill()
        model = ARIMA(series, order=(1,1,1))
        res = model.fit()
        fc = res.get_forecast(steps=periods)
        mean = fc.predicted_mean
        ci = fc.conf_int()
        out = pd.DataFrame({"date":mean.index.to_pydatetime(),"yhat":mean.values,"yhat_lower":ci.iloc[:,0].values,"yhat_upper":ci.iloc[:,1].values})
        return out
    except Exception:
        return None

def build_forecast(market_df:pd.DataFrame, periods:int=7) -> Tuple[pd.DataFrame,str]:
    fc = prophet_forecast(market_df,periods)
    if fc is not None:
        return fc,"Prophet"
    fc = arima_forecast(market_df,periods)
    if fc is not None:
        return fc,"ARIMA"
    raise RuntimeError("Forecasting failed with both Prophet and ARIMA")

def compute_projected_move(market_df:pd.DataFrame, forecast_df:pd.DataFrame) -> dict:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(forecast_df["yhat"].iloc[-1])
    abs_change = proj_price - last_price
    pct = (abs_change / last_price)*100.0 if last_price!=0 else 0.0
    return {"last_price":last_price,"proj_price":proj_price,"abs_change":abs_change,"pct_change":pct}

def compute_signal(projected_move:dict, sentiment_score:float) -> dict:
    pct = projected_move["pct_change"]
    s = sentiment_score
    if pct > 3 and s > 20:
        return {"signal":"BUY","reason":f"Strong upside ({pct:.2f}%) + positive sentiment ({s:.1f}).","pct":pct,"sent":s}
    if pct < -3 and s < -20:
        return {"signal":"SELL","reason":f"Strong downside ({pct:.2f}%) + negative sentiment ({s:.1f}).","pct":pct,"sent":s}
    if pct > 1 and s > 10:
        return {"signal":"BUY","reason":f"Moderate upside ({pct:.2f}%) + supportive sentiment ({s:.1f}).","pct":pct,"sent":s}
    if pct < -1 and s < -10:
        return {"signal":"SELL","reason":f"Moderate downside ({pct:.2f}%) + negative sentiment ({s:.1f}).","pct":pct,"sent":s}
    return {"signal":"HOLD","reason":f"Mixed/weak signals: {pct:.2f}% with sentiment {s:.1f}.","pct":pct,"sent":s}

# ---------------------
# Slack alert
# ---------------------
def send_slack_alert(webhook_url:str, company:str, ticker:str, signal_info:dict, projected_move:dict, sentiment_score:float, model_type:str):
    payload = {
        "text": (
            f"*{company}* ({ticker})\n"
            f"Signal: *{signal_info['signal']}*\n"
            f"Projected move (7D): {projected_move['pct_change']:.2f}%\n"
            f"Proj price: ${projected_move['proj_price']:.2f}\n"
            f"Sentiment: {sentiment_score:.2f}\n"
            f"Model: {model_type}\n"
            f"Reason: {signal_info['reason']}"
        )
    }
    if not webhook_url:
        st.info("Slack webhook not configured — previewing payload:")
        st.json(payload)
        return
    try:
        r = requests.post(webhook_url, json=payload, timeout=5)
        if r.status_code != 200:
            st.warning(f"Slack returned {r.status_code}; printing payload.")
            st.json(payload)
        else:
            st.success("Slack alert sent.")
    except Exception as e:
        st.warning(f"Slack send failed: {e}; printing payload.")
        st.json(payload)

# -------------------------------------------------------------------------
# Streamlit app UI
# -------------------------------------------------------------------------

def header():
    st.markdown(f"<h1 style='color:{PRIMARY_BLUE}; margin-bottom:6px;'>Infosys InsightSphere</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Real-Time Industry Insight & Strategic Intelligence Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

def sidebar_inputs(default_company="Tesla, Inc.", default_ticker="TSLA"):
    st.sidebar.header("Company Input")
    company_name = st.sidebar.text_input("Company Name", value=default_company)
    ticker_in = st.sidebar.text_input("Ticker (optional)", value=default_ticker)
    return company_name.strip(), ticker_in.strip().upper()

def validate_and_resolve(company_name:str, ticker_in:str) -> Tuple[str,str]:
    """Return (company_name, ticker) — validated. Raises ValueError if invalid."""
    # If ticker provided, verify it returns history
    if ticker_in:
        try:
            df = fetch_historical_data_yf(ticker_in, period="1mo")
            if df is None or df.empty:
                raise ValueError("Ticker returned no data.")
            # fill company name using Yahoo metadata
            t = yf.Ticker(ticker_in)
            info = {}
            try:
                info = t.info
                longname = info.get("longName")
                if longname:
                    return longname, ticker_in
            except Exception:
                pass
            return company_name or ticker_in, ticker_in
        except Exception as e:
            raise ValueError(f"Ticker validation failed: {e}")

    # No ticker given: search by company name
    if is_invalid_company_name(company_name):
        raise ValueError("Invalid or test-like company name provided.")
    sym, longname = search_ticker_by_company(company_name)
    if not sym:
        raise ValueError(f"Could not find a valid ticker for '{company_name}'. Please enter the ticker directly.")
    # Confirm symbol returns market data
    try:
        df = fetch_historical_data_yf(sym, period="1mo")
        if df is None or df.empty:
            raise ValueError("Resolved ticker returned no market data.")
    except Exception as e:
        raise ValueError(f"Ticker resolved but validation failed: {e}")
    return longname or company_name, sym

def page_company_overview(state):
    st.subheader("Company Overview")
    try:
        profile = fetch_market_metrics(state["ticker"])
        wiki = fetch_wikipedia_summary(state["company_name"])
        col1, col2 = st.columns([1.3,2])
        with col1:
            st.markdown(f"<div class='metric-card'><strong>{state['company_name']}</strong><div class='small-muted'>{state['ticker']}</div></div>", unsafe_allow_html=True)
            st.write("")
            st.markdown("**Sector:** " + str(profile.get("sector","N/A")))
            st.markdown("**Industry:** " + str(profile.get("industry","N/A")))
            mc = profile.get("market_cap")
            if mc:
                if mc>=1e12: mcstr=f"${mc/1e12:.2f} T"
                elif mc>=1e9: mcstr=f"${mc/1e9:.2f} B"
                else: mcstr=f"${mc:,.0f}"
            else:
                mcstr="N/A"
            st.markdown("**Market Cap:** " + mcstr)
            st.markdown("**Current Price:** " + (f"${profile.get('current_price'):.2f}" if profile.get("current_price") else "N/A"))
        with col2:
            st.markdown("**Wikipedia Summary**")
            st.write(wiki)
    except Exception as e:
        st.error(f"Company overview failed: {e}")

def page_market_and_forecast(state):
    st.subheader("Market Data & Forecast")
    col_fetch, col_actions = st.columns([1,1])
    with col_fetch:
        if st.button("Fetch Historical Data"):
            try:
                state["market_df"] = fetch_historical_data_yf(state["ticker"], period="1y")
                st.success("Historical data fetched.")
            except Exception as e:
                st.error(f"Fetch failed: {e}")
    with col_actions:
        if "market_df" in state:
            if st.button("Compute 7-day Forecast"):
                try:
                    fc, model = build_forecast(state["market_df"], periods=7)
                    state["forecast_df"] = fc
                    state["forecast_model"] = model
                    st.success(f"Forecast generated using {model}.")
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
        else:
            st.info("Fetch historical data first.")

    if "market_df" in state:
        df = state["market_df"]
        st.markdown("**Historical (last 90 rows shown)**")
        st.dataframe(df.tail(90))
    if "forecast_df" in state:
        st.markdown("**Forecast (7 days)**")
        st.dataframe(state["forecast_df"])

def page_sentiment(state):
    st.subheader("News & Sentiment")
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("Fetch News & Generate Corpus"):
            try:
                news = fetch_google_news(f"{state['company_name']} {state['ticker']} stock", max_items=10)
                tweets = generate_mock_tweets(state['company_name'], days=5, posts_per_day=6)
                corpus = build_corpus(news, tweets)
                state["news_df"] = news
                state["tweets_df"] = tweets
                state["corpus_df_raw"] = corpus
                st.success("Corpus ready.")
            except Exception as e:
                st.error(f"News fetch failed: {e}")
    with col2:
        if "corpus_df_raw" in state and st.button("Compute Sentiment"):
            state["corpus_df"] = apply_sentiment_rate_safe(state["corpus_df_raw"])
            state["agg_sentiment"] = aggregate_sentiment(state["corpus_df"])
            st.success("Sentiment computed.")

    if "news_df" in state:
        st.markdown("**Top News (sample)**")
        st.dataframe(state["news_df"][["title","published_at","link"]].head(6))
    if "corpus_df" in state:
        st.markdown("**Sentiment sample**")
        st.dataframe(state["corpus_df"].head(10)[["source","published_at","text","sentiment"]])
        st.metric("Aggregate Sentiment", f"{state.get('agg_sentiment',0.0):.2f}")

def page_competitors(state):
    st.subheader("Competitor Insights")
    comp_input = st.text_input("Peer tickers (comma separated):", value="")
    if st.button("Run Competitor Analysis"):
        tickers = [t.strip().upper() for t in comp_input.split(",") if t.strip()]
        if not tickers:
            st.info("Please enter at least one ticker.")
        else:
            rows = []
            for tkr in tickers:
                try:
                    df = fetch_historical_data_yf(tkr, period="3mo")
                    if df.empty: continue
                    ret_30 = (df["Close"].iloc[-1]/df["Close"].iloc[-21]-1)*100 if len(df)>21 else np.nan
                    news = fetch_google_news(f"{tkr} stock", max_items=5)
                    tweets = generate_mock_tweets(tkr, days=3, posts_per_day=4)
                    corpus = build_corpus(news,tweets)
                    corpus_sent = apply_sentiment_rate_safe(corpus)
                    agg = aggregate_sentiment(corpus_sent)
                    rows.append({"Ticker":tkr,"30D Return %":ret_30,"Sentiment":agg,"News Count":(corpus_sent['source']=='news').sum()})
                except Exception:
                    continue
            if rows:
                comp_df = pd.DataFrame(rows)
                st.dataframe(comp_df)
                st.markdown("### Positioning Map")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=comp_df["Sentiment"], y=comp_df["30D Return %"], mode="markers+text", text=comp_df["Ticker"], textposition="top center"))
                fig.update_layout(xaxis_title="Sentiment", yaxis_title="30D Return (%)", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No competitor data available or tickers invalid.")

def page_dashboard(state):
    st.subheader("Executive Dashboard")
    if not all(k in state for k in ("market_df","forecast_df","corpus_df","agg_sentiment")):
        st.warning("Please run Market & Forecast and Sentiment pages first.")
        return
    market_df = state["market_df"]
    forecast_df = state["forecast_df"]
    corpus_df = state["corpus_df"]
    agg_sent = state["agg_sentiment"]
    model_type = state.get("forecast_model","N/A")
    projected_move = compute_projected_move(market_df, forecast_df)
    signal_info = compute_signal(projected_move, agg_sent)
    # Top KPI cards
    c1,c2,c3,c4 = st.columns([1,1,1,1])
    with c1:
        st.markdown(f"<div class='metric-card'><div style='font-weight:700'>Current Price</div><div>${projected_move['last_price']:.2f}</div></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card'><div style='font-weight:700'>Forecast (7D)</div><div>${projected_move['proj_price']:.2f}</div></div>", unsafe_allow_html=True)
    with c3:
        move = projected_move['pct_change']
        arrow = "▲" if move>0 else ("▼" if move<0 else "➜")
        st.markdown(f"<div class='metric-card'><div style='font-weight:700'>Projected Move</div><div>{move:.2f}% {arrow}</div></div>", unsafe_allow_html=True)
    with c4:
        badge_color = SUCCESS_GREEN if signal_info['signal']=="BUY" else (DANGER_RED if signal_info['signal']=="SELL" else WARNING_ORANGE)
        st.markdown(f"<div class='metric-card'><div style='font-weight:700'>Signal</div><div><span class='signal-badge' style='background:{badge_color};'>{signal_info['signal']}</span></div></div>", unsafe_allow_html=True)

    st.markdown(f"**Rationale:** {signal_info['reason']}")
    # Compose Plotly dashboard (price + forecast + CI + forecast bars + sentiment gauge + metrics)
    hist_dates = market_df['date']; hist_close = market_df['Close']
    last_hist = hist_dates.max()
    future_fc = forecast_df[forecast_df['date']>last_hist].copy()
    if future_fc.empty: future_fc = forecast_df.copy()
    fc_dates = future_fc['date']; fc_mean = future_fc['yhat']; fc_low = future_fc['yhat_lower']; fc_high = future_fc['yhat_upper']

    fig = make_subplots(rows=3, cols=2, specs=[[{"colspan":2}, None],[{"type":"indicator"},{"type":"table"}],[{"type":"bar"},{"type":"table"}]])
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, name="Historical", line=dict(color=DARK_BLUE)), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, name="Forecast", line=dict(color=TEAL)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(fc_dates)+list(fc_dates[::-1]), y=list(fc_high)+list(fc_low[::-1]), fill="toself", fillcolor="rgba(0,124,195,0.12)", line=dict(color="rgba(0,0,0,0)"), name="CI"), row=1, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1); fig.update_yaxes(title_text="Price", row=1, col=1)

    # sentiment gauge
    fig.add_trace(go.Indicator(mode="gauge+number", value=agg_sent, title={"text":"Sentiment"}, gauge={"axis":{"range":[-100,100]}, "bar":{"color":DARK_BLUE}}), row=2, col=1)

    # metrics table
    mc = fetch_market_metrics(state["ticker"]).get("market_cap")
    if mc:
        if mc>=1e12: mcstr=f"${mc/1e12:.2f} T"
        elif mc>=1e9: mcstr=f"${mc/1e9:.2f} B"
        else: mcstr=f"${mc:,.0f}"
    else:
        mcstr="N/A"
    table_data = [["Company",state["company_name"]],["Ticker",state["ticker"]],["Market Cap",mcstr],["Current Price",f"${projected_move['last_price']:.2f}"],["Forecast (7D)",f"${projected_move['proj_price']:.2f}"],["7D Change",f"{projected_move['pct_change']:.2f}%"],["Sentiment",f"{agg_sent:.2f}"],["Signal",signal_info['signal']]]
    fig.add_trace(go.Table(header=dict(values=["Metric","Value"],fill_color=LIGHT_GREY), cells=dict(values=list(zip(*table_data)))), row=2, col=2)

    # forecast bars & sentiment breakdown
    fig.add_trace(go.Bar(x=fc_dates, y=fc_mean, name="Daily Forecast"), row=3, col=1)
    # sentiment breakdown table
    def bucket(v):
        if v>=20: return "Bullish"
        if v<=-20: return "Bearish"
        return "Neutral"
    sb = state['corpus_df']['sentiment'].apply(bucket).value_counts().to_dict() if 'corpus_df' in state else {}
    sent_rows = [["Bullish", sb.get("Bullish",0)],["Neutral",sb.get("Neutral",0)],["Bearish",sb.get("Bearish",0)]]
    fig.add_trace(go.Table(header=dict(values=["Bucket","Count"],fill_color=LIGHT_GREY), cells=dict(values=list(zip(*sent_rows)))), row=3, col=2)

    fig.update_layout(height=1000, showlegend=False, title_text=f"Executive Dashboard - {state['company_name']} ({state['ticker']})")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Send Slack Alert"):
        send_slack_alert(SLACK_WEBHOOK_URL, state["company_name"], state["ticker"], signal_info, projected_move, agg_sent, model_type)

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    header()
    company_name_input, ticker_input = sidebar_inputs()
    # validate
    try:
        company_name, ticker = validate_and_resolve(company_name_input, ticker_input)
    except Exception as e:
        st.sidebar.error(str(e))
        # fallback to default
        company_name = "Tesla, Inc."
        ticker = "TSLA"

    st.sidebar.markdown(f"**Using:** {company_name} (`{ticker}`)")
    page = st.sidebar.radio("Navigation", ["Company Overview","Market & Forecast","Sentiment","Competitors","Dashboard"], index=4)

    # state container
    if "company_name" not in st.session_state or st.session_state["company_name"]!=company_name:
        # reset analysis state when switching company
        st.session_state.clear()
        st.session_state["company_name"]=company_name
        st.session_state["ticker"]=ticker

    state = st.session_state

    if page=="Company Overview":
        page_company_overview(state)
    elif page=="Market & Forecast":
        page_market_and_forecast(state)
    elif page=="Sentiment":
        page_sentiment(state)
    elif page=="Competitors":
        page_competitors(state)
    elif page=="Dashboard":
        page_dashboard(state)

if __name__=="__main__":
    main()
