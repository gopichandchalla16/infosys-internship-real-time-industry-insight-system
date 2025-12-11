import os
import time
import threading
import re
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any, Optional

import streamlit as st
import yfinance as yf
import feedparser
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional: Gemini (Google) LLM — if you don't have it, sentiment falls back to local heuristic
GEMINI_ENABLED = False
try:
    import google.generativeai as genai  # type: ignore
    GEMINI_ENABLED = True
except Exception:
    GEMINI_ENABLED = False

# ---------------------------
# Page & runtime
# ---------------------------
st.set_page_config(page_title="InsightSphere (Extended)", layout="wide")
st.title("Infosys InsightSphere — Extended")

# ---------------------------
# Secrets (do NOT commit)
# ---------------------------
GEMINI_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")) if "secrets" in dir(st) else os.getenv("GEMINI_API_KEY", "")
SLACK_WEBHOOK = st.secrets.get("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", "")) if "secrets" in dir(st) else os.getenv("SLACK_WEBHOOK_URL", "")
WEBSOCKET_URL = st.secrets.get("WEBSOCKET_URL", os.getenv("WEBSOCKET_URL", "")) if "secrets" in dir(st) else os.getenv("WEBSOCKET_URL", "")

if GEMINI_KEY and GEMINI_ENABLED:
    try:
        genai.configure(api_key=GEMINI_KEY)
        gen_model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        gen_model = None
else:
    gen_model = None

# ---------------------------
# UI - Sidebar Controls
# ---------------------------
st.sidebar.header("Controls")
# Theme toggle
theme_choice = st.sidebar.radio("Theme", ["Dark", "Light"], index=0)

# Stocks input: primary ticker, plus multi and competitors
ticker = st.sidebar.text_input("Primary Ticker", value="AAPL").upper().strip()
company_name = st.sidebar.text_input("Company name (optional)", value="")
multi_tickers = st.sidebar.text_input("Multi Tick ers (comma-separated)", value="AAPL, MSFT").upper()
competitors = st.sidebar.text_input("Competitors (comma-separated)", value="GOOGL, AMZN").upper()

# Forecast days
days = st.sidebar.slider("Forecast horizon (days)", 3, 21, 7)

# Live updates controls
enable_live = st.sidebar.checkbox("Enable live updates (polling or websocket)", value=True)
poll_interval = st.sidebar.slider("Polling interval (seconds)", 5, 60, 15)

# Action
if st.sidebar.button("Run / Refresh", use_container_width=True):
    st.cache_data.clear()
    st.experimental_rerun()

# Apply theme CSS (simple)
if theme_choice == "Dark":
    plotly_template = "plotly_dark"
    st.markdown(
        """<style>
        .stApp { background-color: #0f172a; color: #e6eef8; }
        .card { background: #0b1220; padding: 12px; border-radius: 12px; }
        </style>""", unsafe_allow_html=True
    )
else:
    plotly_template = "plotly_white"
    st.markdown(
        """<style>
        .stApp { background-color: #ffffff; color: #111827; }
        .card { background: #f8fafc; padding: 12px; border-radius: 12px; }
        </style>""", unsafe_allow_html=True
    )

# ---------------------------
# Utilities
# ---------------------------
INVALID_COMPANY_KEYWORDS = {"abc","xyz","test","demo","sample","fake","dummy","123"}
POS_WORDS = ["growth","strong","bullish","positive","optimistic","profit","surge","beats","outperform","record"]
NEG_WORDS = ["weak","bearish","loss","regulatory","lawsuit","slowing","concern","fraud","volatility","drop"]

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

@st.cache_data(ttl=600)
def search_ticker_by_company(company_name: str) -> Optional[str]:
    # lightweight resolution via Yahoo search
    try:
        q = company_name.strip().replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={q}"
        headers = {"User-Agent":"Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=6)
        r.raise_for_status()
        data = r.json()
        for itm in data.get("quotes", []):
            if itm.get("quoteType") == "EQUITY":
                return itm.get("symbol")
    except Exception:
        return None
    return None

# ---------------------------
# Robust yfinance fetch (session + fallback)
# ---------------------------
@st.cache_data(ttl=600)
def fetch_stock_data(tk: str, period: str = "1y", interval: str = "1d") -> Tuple[Optional[pd.DataFrame], Optional[float], Optional[float], Optional[str], Optional[str]]:
    session = requests.Session()
    session.headers.update({"User-Agent":"Mozilla/5.0"})
    try:
        df = yf.download(tk, period=period, interval=interval, auto_adjust=True, progress=False, session=session)
        if df is None or df.empty or len(df) < 2:
            raise ValueError("No data")
        df = df.reset_index()
        # normalize columns
        if "Date" in df.columns and "date" not in df.columns:
            df = df.rename(columns={"Date":"date"})
        # try to extract info
        t = yf.Ticker(tk, session=session)
        try:
            info = t.fast_info or {}
        except Exception:
            info = {}
        price = info.get("last_price") or float(df["Close"].iloc[-1])
        market_cap = info.get("market_cap")
        longname = info.get("long_name") or info.get("short_name") or tk
        sector = None
        try:
            info2 = t.info or {}
            sector = info2.get("sector")
        except Exception:
            sector = None
        return df, price, market_cap, longname, sector
    except Exception:
        # fallback non-session
        try:
            t = yf.Ticker(tk)
            hist = t.history(period=period, interval=interval, auto_adjust=True)
            if hist is None or hist.empty:
                return None, None, None, None, None
            hist = hist.reset_index().rename(columns={"Date":"date"}) if "Date" in hist.reset_index().columns else hist.reset_index()
            info = {}
            try:
                info = t.fast_info or {}
            except Exception:
                info = {}
            price = info.get("last_price") or float(hist["Close"].iloc[-1])
            market_cap = info.get("market_cap")
            longname = info.get("long_name") or tk
            sector = None
            try:
                info2 = t.info or {}
                sector = info2.get("sector")
            except Exception:
                sector = None
            return hist, price, market_cap, longname, sector
        except Exception:
            return None, None, None, None, None

# ---------------------------
# Sentiment (Gemini or local heuristic)
# ---------------------------
def local_sentiment(text: str) -> float:
    if not text: return 0.0
    t = text.lower()
    score = 0
    for w in POS_WORDS:
        if w in t: score += 10
    for w in NEG_WORDS:
        if w in t: score -= 10
    return float(max(-100, min(100, score)))

@st.cache_data(ttl=1800)
def analyze_sentiment(company: str, tk: str, use_gemini: bool = True) -> float:
    # gather some headlines
    q = f"{company} {tk} stock" if company else f"{tk} stock"
    url = f"https://news.google.com/rss/search?q={q}"
    feed = feedparser.parse(url)
    texts = []
    for e in feed.entries[:8]:
        title = getattr(e, "title", "")
        summary = getattr(e, "summary", "")
        texts.append(f"{title}. {summary}")
    if not texts:
        return 0.0

    scores = []
    for txt in texts:
        if use_gemini and gen_model is not None:
            try:
                # trimmed prompt
                prompt = f"Provide a single integer from -100 (very negative) to 100 (very positive) that represents sentiment for the text.\n\nText: {txt[:1600]}"
                resp = gen_model.generate_content(prompt)
                raw = getattr(resp, "text", str(resp))
                m = re.search(r"-?\d+", str(raw))
                if m:
                    scores.append(int(m.group()))
                    continue
            except Exception:
                pass
        # fallback
        scores.append(int(local_sentiment(txt)))
    avg = float(np.mean(scores)) if scores else 0.0
    return float(round(max(-100, min(100, avg)), 1))

# ---------------------------
# Forecast (Holt-Winters simple)
# ---------------------------
def forecast_holtwinters(df: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    df2 = df.copy()
    # locate close column and date
    if "date" not in df2.columns:
        if "Date" in df2.columns:
            df2 = df2.rename(columns={"Date":"date"})
    df2["date"] = pd.to_datetime(df2["date"])
    series = df2["Close"].astype(float)
    model = ExponentialSmoothing(series, trend="add", initialization_method="estimated")
    fit = model.fit(optimized=True)
    pred = fit.forecast(days)
    dates = pd.date_range(df2["date"].iloc[-1] + timedelta(1), periods=days, freq="D")
    out = pd.DataFrame({"ds": dates, "yhat": pred})
    out["yhat_lower"] = out["yhat"] * 0.94
    out["yhat_upper"] = out["yhat"] * 1.06
    return out

# ---------------------------
# Live updates (websocket or polling)
# ---------------------------
def start_polling(tickers: List[str], interval_seconds: int = 15):
    """
    Poll yfinance for the latest price every `interval_seconds` seconds.
    Stores results in st.session_state['live_prices'].
    """
    def _poll():
        while True:
            try:
                for tk in tickers:
                    df, price, cap, name, sector = fetch_stock_data(tk, period="5d", interval="1m")
                    if price is not None:
                        st.session_state["live_prices"][tk] = {"price": price, "ts": datetime.utcnow().isoformat()}
            except Exception:
                pass
            time.sleep(interval_seconds)
    if "live_thread" not in st.session_state:
        st.session_state["live_prices"] = {tk: {"price": None, "ts": None} for tk in tickers}
        th = threading.Thread(target=_poll, daemon=True)
        st.session_state["live_thread"] = th
        th.start()

# Websocket support (optional) - uses websocket-client library (added to requirements)
def start_websocket_listener(ws_url: str, tickers: List[str]):
    """
    If you have a websocket URL that pushes JSON with { 'symbol': 'AAPL', 'price': 123.4 },
    this thread will consume it and update st.session_state['live_prices'].
    """
    try:
        import websocket  # websocket-client
    except Exception:
        return

    def _run():
        def on_message(ws, message):
            try:
                data = None
                try:
                    data = json.loads(message)
                except Exception:
                    # try parse simple "SYMBOL:PRICE"
                    parts = str(message).split(":")
                    if len(parts) >= 2:
                        sym = parts[0].strip().upper()
                        val = float(parts[1])
                        data = {"symbol": sym, "price": val}
                if data:
                    sym = data.get("symbol") or data.get("s")
                    pr = data.get("price") or data.get("p")
                    if sym and pr:
                        if sym in st.session_state["live_prices"]:
                            st.session_state["live_prices"][sym] = {"price": float(pr), "ts": datetime.utcnow().isoformat()}
            except Exception:
                pass

        def on_error(ws, err):
            pass

        def on_close(ws, code, reason):
            pass

        def on_open(ws):
            # optionally subscribe
            pass

        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close, on_open=on_open)
        ws.run_forever()

    if "ws_thread" not in st.session_state and ws_url:
        st.session_state["live_prices"] = {tk: {"price": None, "ts": None} for tk in tickers}
        th = threading.Thread(target=_run, daemon=True)
        st.session_state["ws_thread"] = th
        th.start()

# ---------------------------
# Gather tickers list
# ---------------------------
primary = ticker
multi_list = [t.strip().upper() for t in multi_tickers.split(",") if t.strip()]
unique_tickers = list(dict.fromkeys([primary] + multi_list))
competitor_list = [t.strip().upper() for t in competitors.split(",") if t.strip()]

all_tickers = list(dict.fromkeys(unique_tickers + competitor_list))

# Start live updates (websocket preferred)
if enable_live:
    if WEBSOCKET_URL:
        start_websocket_listener(WEBSOCKET_URL, all_tickers)
    else:
        start_polling(all_tickers, poll_interval)

# ---------------------------
# Fetch data for all tickers (historical + meta)
# ---------------------------
stock_data: Dict[str, Dict[str, Any]] = {}
for tk in unique_tickers:
    df, price, cap, longname, sector = fetch_stock_data(tk, period="1y", interval="1d")
    if df is None:
        st.warning(f"No historical data for {tk}. Skipping.")
        continue
    stock_data[tk] = {"df": df, "price": price, "market_cap": cap, "name": longname, "sector": sector}
    # compute forecast
    try:
        fc = forecast_holtwinters(df, days)
        stock_data[tk]["forecast"] = fc
        stock_data[tk]["forecast_price"] = float(round(fc["yhat"].mean(), 4))
        stock_data[tk]["pct_change"] = float(round((stock_data[tk]["forecast_price"] - (price or fc["yhat"].iloc[0])) / (price or fc["yhat"].iloc[0]) * 100.0, 2))
    except Exception:
        stock_data[tk]["forecast"] = None
        stock_data[tk]["forecast_price"] = None
        stock_data[tk]["pct_change"] = None

# Competitors meta fetch (only current price)
competitor_meta = {}
for tk in competitor_list:
    df, price, cap, longname, sector = fetch_stock_data(tk, period="5d", interval="1d")
    competitor_meta[tk] = {"price": price, "market_cap": cap, "name": longname, "sector": sector}

# ---------------------------
# Sentiment for primary
# ---------------------------
if unique_tickers:
    primary_name = stock_data.get(primary, {}).get("name", company_name or primary)
    sentiment_score = analyze_sentiment(primary_name, primary, use_gemini=(gen_model is not None))
else:
    sentiment_score = 0.0

# ---------------------------
# UI Layout — Multi-stock KPIs & comparison
# ---------------------------
st.subheader("Overview & Multi-Stock KPIs")
k_cols = st.columns(max(1, len(unique_tickers)))
for i, tk in enumerate(unique_tickers):
    col = k_cols[i % len(k_cols)]
    info = stock_data.get(tk)
    if not info:
        col.write(f"**{tk}** - no data")
        continue
    price = info["price"] or 0.0
    fc_price = info.get("forecast_price")
    pct = info.get("pct_change")
    name = info.get("name", tk)
    col.markdown(f"<div class='card'><strong>{name} ({tk})</strong><br><div style='font-size:22px'>${price:.2f}</div><div style='font-size:12px'>7d forecast: ${fc_price if fc_price else 'N/A'}</div><div style='font-size:12px'>proj: {pct if pct is not None else 'N/A'}%</div></div>", unsafe_allow_html=True)

# Competitor comparison table
if competitor_meta:
    st.subheader("Competitor Comparison")
    comp_rows = []
    for tk, meta in competitor_meta.items():
        comp_rows.append({"Ticker": tk, "Name": meta.get("name") or tk, "Price": meta.get("price") or "N/A", "Market Cap": meta.get("market_cap") or "N/A", "Sector": meta.get("sector") or "N/A"})
    comp_df = pd.DataFrame(comp_rows)
    st.dataframe(comp_df)

# ---------------------------
# Industry-wide heatmap (percent moves)
# ---------------------------
st.subheader("Industry Heatmap (Percent Changes)")
heat_tickers = all_tickers
heat_rows = []
heat_sectors = []
for tk in heat_tickers:
    df_meta = None
    try:
        df_meta = fetch_stock_data(tk, period="7d", interval="1d")
    except Exception:
        df_meta = None
    if df_meta and df_meta[0] is not None:
        hist = df_meta[0]
        pct = (float(hist["Close"].iloc[-1]) - float(hist["Close"].iloc[0])) / float(hist["Close"].iloc[0]) * 100.0 if len(hist) > 1 else 0.0
        heat_rows.append(pct)
        heat_sectors.append(df_meta[4] or "Unknown")
    else:
        heat_rows.append(0.0)
        heat_sectors.append("Unknown")

if heat_rows:
    # build heatmap matrix (simple single-row heatmap)
    heat_df = pd.DataFrame({"ticker": heat_tickers, "pct_change": heat_rows, "sector": heat_sectors})
    # pivot to a matrix by sector
    pivot = heat_df.pivot_table(index="sector", columns="ticker", values="pct_change", fill_value=0.0)
    fig_heat = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn"))
    fig_heat.update_layout(height=300, template=plotly_template)
    st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------
# Multi-stock charts (small multiples)
# ---------------------------
st.subheader("Price Charts & Forecasts (Multi-stock)")
rows = []
for tk, info in stock_data.items():
    df = info["df"]
    fc = info.get("forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["Close"], name=f"{tk} Close"))
    if fc is not None:
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
        fig.add_trace(go.Scatter(x=pd.concat([fc["ds"], fc["ds"][::-1]]),
                                 y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                                 fill="toself", fillcolor="rgba(0,200,150,0.08)", line=dict(color="transparent"),
                                 showlegend=False, name="CI"))
    fig.update_layout(title=f"{tk}", template=plotly_template, height=300)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Live price panel
# ---------------------------
st.subheader("Live Prices")
live_cols = st.columns(3)
for idx, tk in enumerate(all_tickers):
    c = live_cols[idx % 3]
    lp = st.session_state.get("live_prices", {}).get(tk, {"price": None})
    p = lp.get("price")
    ts = lp.get("ts")
    txt = f"${p:.2f}" if p else "N/A"
    c.metric(label=tk, value=txt, delta="live")

# ---------------------------
# Executive summary & signal for primary ticker
# ---------------------------
st.subheader("Executive Summary")
primary_info = stock_data.get(primary)
if primary_info:
    st.markdown(f"**{primary_info.get('name')} ({primary})**")
    st.write(f"- Current price: ${primary_info.get('price'):.2f}")
    st.write(f"- Forecast avg: ${primary_info.get('forecast_price')}")
    st.write(f"- Projected change: {primary_info.get('pct_change'):+.2f}%")
    st.write(f"- Sentiment (news): {sentiment_score:+.1f}")
    # Signal logic (example)
    s = primary_info.get("pct_change") or 0.0
    sent = sentiment_score
    if s > 3 and sent > 20:
        sig = "STRONG BUY"
    elif s > 1 and sent > 5:
        sig = "BUY"
    elif s < -3 and sent < -20:
        sig = "STRONG SELL"
    elif s < -1 and sent < -5:
        sig = "SELL"
    else:
        sig = "HOLD"
    st.markdown(f"### Signal: {sig}")

# ---------------------------
# Slack alert
# ---------------------------
if SLACK_WEBHOOK and st.button("Send summary to Slack"):
    summary = f"InsightSphere — {primary} summary\nSignal: {sig}\nPrice: ${primary_info.get('price'):.2f}\nProj change: {primary_info.get('pct_change'):+.2f}%\nSentiment: {sentiment_score:+.1f}"
    try:
        requests.post(SLACK_WEBHOOK, json={"text": summary})
        st.success("Sent")
    except Exception as e:
        st.error(f"Slack failed: {e}")
st.caption("© 2025 Infosys Springboard • Team: Gopichand, Anshika, Janmejay, Vaishnavi")
