import os
import re
import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import feedparser
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from PIL import Image

# Try importing Prophet (selector will handle gracefully if unavailable)
PROPHET_AVAILABLE = True
try:
    from prophet import Prophet
except Exception:
    PROPHET_AVAILABLE = False

# ---------------------------
# Page / Theme configuration
# ---------------------------
st.set_page_config(
    page_title="Infosys InsightSphere",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------
# Dark/Light mode toggle
# ---------------------------
def apply_theme(dark_mode: bool):
    # Minimal CSS theme using CSS variables; avoids Streamlit theme reload delays
    base_bg = "#0B1220" if dark_mode else "#F9FAFB"
    card_bg = "#111827" if dark_mode else "#FFFFFF"
    text_col = "#E5E7EB" if dark_mode else "#111827"
    subtext = "#9CA3AF" if dark_mode else "#6B7280"
    accent = "#2563EB"
    badge_buy = "#059669"
    badge_sell = "#DC2626"
    badge_hold = "#D97706"

    st.markdown(
        f"""
        <style>
        :root {{
          --bg: {base_bg};
          --card: {card_bg};
          --text: {text_col};
          --subtext: {subtext};
          --accent: {accent};
          --buy: {badge_buy};
          --sell: {badge_sell};
          --hold: {badge_hold};
        }}
        body, .stApp {{ background-color: var(--bg) !important; color: var(--text) !important; }}
        .metric-card {{
          background: var(--card); border-radius: 14px; padding: 14px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.15); margin-bottom: 12px;
        }}
        .signal-badge {{ padding: 8px 14px; border-radius: 12px; color: white; font-weight: 700; }}
        h1, h2, h3, h4 {{ color: var(--text) !important; }}
        .subtext {{ color: var(--subtext); font-size: 0.95rem; }}
        .stButton>button {{
          background-color: var(--accent); color: white; border-radius: 8px; padding: 6px 12px;
          border: none;
        }}
        .stButton>button:hover {{ opacity: 0.92; }}
        .stTextInput > div > div > input,
        .stTextInput > label,
        .stSelectbox > div > div,
        .stCheckbox > label {{ color: var(--text) !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
apply_theme(st.session_state.dark_mode)

# ---------------------------
# Secrets & mandatory integrations
# ---------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", "")).strip()
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", os.getenv("ALPHA_VANTAGE_API_KEY", "")).strip()
SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", os.getenv("SLACK_WEBHOOK_URL", "")).strip()

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key is required. Set GEMINI_API_KEY in .streamlit/secrets.toml.")
    st.stop()
if not SLACK_WEBHOOK_URL:
    st.error("❌ Slack webhook URL is required. Set SLACK_WEBHOOK_URL in .streamlit/secrets.toml.")
    st.stop()
if not ALPHA_VANTAGE_API_KEY:
    st.error("❌ Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY in .streamlit/secrets.toml.")
    st.stop()

# ---------------------------
# Utility functions (fast, guarded)
# ---------------------------
def fetch_historical_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df = df.reset_index()
    df = df.rename(columns={"Date": "date"})
    df = df[["date", "Close"]].dropna().reset_index(drop=True)
    return df

def fetch_market_metrics(ticker: str) -> dict:
    # Use fast_info (avoid slow .info)
    t = yf.Ticker(ticker)
    fi = getattr(t, "fast_info", {}) or {}
    current_price = float(fi.get("last_price") or fi.get("last_close") or np.nan)
    market_cap = fi.get("market_cap")
    return {"current_price": current_price, "market_cap": market_cap}

def fetch_google_news(query: str, max_items: int = 8) -> pd.DataFrame:
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
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
        rows.append({"title": title, "summary": summary, "link": link, "published_at": published_at, "text": text})
    return pd.DataFrame(rows)

def fetch_alpha_vantage_news_sentiment(symbol: str, limit: int = 10) -> pd.DataFrame:
    base = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "limit": limit,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    r = requests.get(base, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    feed = data.get("feed", []) or []
    rows = []
    for item in feed[:limit]:
        title = item.get("title", "")
        summary = item.get("summary", "")
        url = item.get("url", "")
        published_at = pd.to_datetime(item.get("time_published"), errors="coerce")
        score = item.get("overall_sentiment_score")
        try:
            score = float(score) if score is not None else None
        except Exception:
            score = None
        rows.append({
            "title": title,
            "summary": summary,
            "link": url,
            "published_at": published_at,
            "text": f"{title}. {summary}",
            "av_sentiment": score,
        })
    return pd.DataFrame(rows)

def gemini_sentiment(text: str) -> float:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    prompt = (
        "You are a financial sentiment scorer. "
        "Return a single integer between -100 (very negative) and 100 (very positive) for the text below.\n\n"
        f"Text:\n{text[:1600]}"
    )
    model = genai.GenerativeModel("gemini-2.0-flash")
    resp = model.generate_content(prompt)
    raw = getattr(resp, "text", "") or ""
    m = re.search(r"-?\d+", raw)
    return float(m.group(0)) if m else 0.0

def build_corpus(company: str, ticker: str, limit: int = 12) -> pd.DataFrame:
    av_news = fetch_alpha_vantage_news_sentiment(ticker, limit=min(limit, 10))
    gnews = fetch_google_news(f"{company} {ticker} stock", max_items=limit)
    parts = []
    if not av_news.empty:
        parts.append(av_news)
    if not gnews.empty:
        parts.append(gnews)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["text"]).sort_values("published_at", ascending=False).reset_index(drop=True)
    return df

def score_corpus_with_gemini(corpus: pd.DataFrame, llm_budget: int = 8) -> pd.DataFrame:
    if corpus is None or corpus.empty:
        return pd.DataFrame()
    out = corpus.copy()
    scores = []
    used = 0
    for _, row in out.iterrows():
        if pd.notna(row.get("av_sentiment")):
            scores.append(float(row["av_sentiment"]))
            continue
        text = row.get("text", "")
        if used < llm_budget and text:
            try:
                val = gemini_sentiment(text)
                used += 1
                scores.append(float(max(-100, min(100, val))))
                continue
            except Exception:
                pass
        scores.append(0.0)
    out["sentiment"] = scores
    return out

def make_wordcloud(texts: list[str], dark_mode: bool) -> Image.Image:
    bg = "#0B1220" if dark_mode else "#FFFFFF"
    wc = WordCloud(width=800, height=400, background_color=bg, colormap="viridis")
    joined = " ".join(t for t in texts if isinstance(t, str))
    img = wc.generate(joined).to_image()
    return img

# ---------------------------
# Forecasting models
# ---------------------------
def forecast_arima(market_df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
    df = market_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["Close"].asfreq("D").ffill()
    model = ARIMA(series, order=(1, 1, 1))
    res = model.fit()
    fc = res.get_forecast(steps=periods)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    dates = pd.date_range(start=series.index[-1] + timedelta(days=1), periods=periods, freq="D")
    return pd.DataFrame({
        "date": dates,
        "yhat": mean.values,
        "yhat_lower": ci.iloc[:, 0].values,
        "yhat_upper": ci.iloc[:, 1].values
    })

def forecast_prophet(market_df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not available. Please install the 'prophet' package.")
    df = market_df.copy()
    df = df.rename(columns={"date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df)
    future = m.make_future_dataframe(periods=periods, freq="D")
    fc = m.predict(future)
    tail = fc.tail(periods)
    return pd.DataFrame({
        "date": tail["ds"].values,
        "yhat": tail["yhat"].values,
        "yhat_lower": tail["yhat_lower"].values,
        "yhat_upper": tail["yhat_upper"].values
    })

def build_forecast(market_df: pd.DataFrame, model_name: str, periods: int = 7) -> tuple[pd.DataFrame, str]:
    if model_name == "Prophet":
        return forecast_prophet(market_df, periods), "Prophet"
    return forecast_arima(market_df, periods), "ARIMA"

# ---------------------------
# Signal logic and Slack
# ---------------------------
def compute_projected_move(market_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict:
    last_price = float(market_df["Close"].iloc[-1])
    proj_price = float(np.mean(forecast_df["yhat"]))
    pct = ((proj_price - last_price) / last_price) * 100.0 if last_price else 0.0
    return {"last_price": last_price, "proj_price": proj_price, "pct_change": pct}

def compute_signal(projected: dict, sentiment_score: float, thresholds: dict) -> dict:
    up_thr = thresholds.get("buy_threshold_pct", 3.0)
    down_thr = thresholds.get("sell_threshold_pct", -3.0)
    pos_sent = thresholds.get("positive_sentiment", 15.0)
    neg_sent = thresholds.get("negative_sentiment", -15.0)

    pct = projected["pct_change"]
    s = sentiment_score

    if pct >= up_thr and s >= pos_sent:
        return {"signal": "STRONG BUY", "color": "var(--buy)", "reason": f"Upside {pct:.2f}% with positive sentiment {s:.1f}"}
    if pct >= up_thr / 2 and s >= pos_sent / 2:
        return {"signal": "BUY", "color": "var(--buy)", "reason": f"Moderate upside {pct:.2f}% with supportive sentiment {s:.1f}"}
    if pct <= down_thr and s <= neg_sent:
        return {"signal": "STRONG SELL", "color": "var(--sell)", "reason": f"Downside {pct:.2f}% with negative sentiment {s:.1f}"}
    if pct <= down_thr / 2 and s <= neg_sent / 2:
        return {"signal": "SELL", "color": "var(--sell)", "reason": f"Moderate downside {pct:.2f}% with negative sentiment {s:.1f}"}
    return {"signal": "HOLD", "color": "var(--hold)", "reason": f"Mixed signals ({pct:.2f}%) and sentiment {s:.1f}"}

def send_slack_alert(company: str, ticker: str, signal_info: dict, projected: dict, sentiment: float, model: str, top_items: list[str]):
    title = f"{company} ({ticker}) - {signal_info['signal']}"
    text = (
        f"*{company}* ({ticker})\n"
        f"*Signal:* {signal_info['signal']}\n"
        f"*7D Forecast Move:* {projected['pct_change']:+.2f}% → ${projected['proj_price']:.2f}\n"
        f"*Sentiment:* {sentiment:.1f}\n"
        f"*Model:* {model}\n"
        f"*Reason:* {signal_info['reason']}\n\n"
        f"*Top insights:*\n" + ("\n".join([f"- {x}" for x in top_items]) if top_items else "- none -")
    )
    payload = {"text": text}
    try:
        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=10)
        if r.status_code == 200:
            st.success("Slack alert sent.")
        else:
            st.error(f"Slack error: {r.status_code} — {r.text}")
    except Exception as e:
        st.error(f"Failed to send Slack alert: {e}")

# ---------------------------
# UI components
# ---------------------------
def metric_card(title: str, value: str):
    st.markdown(
        f"<div class='metric-card'><div style='font-weight:600; margin-bottom:6px;'>{title}</div>"
        f"<div style='font-size:20px'>{value}</div></div>",
        unsafe_allow_html=True,
    )

def overview_tab(company: str, ticker: str):
    st.subheader("Overview")
    try:
        metrics = fetch_market_metrics(ticker)
        df = fetch_historical_data(ticker, period="6mo")
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        price = metrics.get("current_price")
        price_str = f"${price:.2f}" if price and np.isfinite(price) else "N/A"
        metric_card("Current price", price_str)
    with c2:
        mc = metrics.get("market_cap")
        if mc and mc > 0:
            if mc >= 1e12:
                mc_str = f"${mc/1e12:.2f} T"
            elif mc >= 1e9:
                mc_str = f"${mc/1e9:.2f} B"
            else:
                mc_str = f"${mc:,.0f}"
        else:
            mc_str = "N/A"
        metric_card("Market cap", mc_str)
    with c3:
        metric_card("Data range", f"{pd.to_datetime(df['date']).min().date()} → {pd.to_datetime(df['date']).max().date()}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pd.to_datetime(df["date"]), y=df["Close"], mode="lines", name="Close", line=dict(color="#60A5FA")))
    fig.update_layout(title=f"{company} ({ticker}) — 6-month price", height=400)
    st.plotly_chart(fig, use_container_width=True)

def forecast_tab(company: str, ticker: str, model_name: str, periods: int):
    st.subheader("Forecast")
    try:
        market_df = fetch_historical_data(ticker, period="1y")
        fc_df, used_model = build_forecast(market_df, model_name, periods=periods)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        return

    projected = compute_projected_move(market_df, fc_df)

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Last close", f"${projected['last_price']:.2f}")
    with c2:
        metric_card(f"{periods}D avg forecast", f"${projected['proj_price']:.2f}")
    with c3:
        metric_card("Projected move", f"{projected['pct_change']:+.2f}%")

    hist_dates = pd.to_datetime(market_df["date"].tail(180))
    hist_close = market_df["Close"].tail(180)
    fc_dates = pd.to_datetime(fc_df["date"])
    fc_mean = fc_df["yhat"]
    fc_low = fc_df["yhat_lower"]
    fc_high = fc_df["yhat_upper"]

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "xy"}]], subplot_titles=(f"Historical vs Forecast — {used_model}",))
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, mode="lines", name="Historical", line=dict(color="#93C5FD")))
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, mode="lines+markers", name="Forecast", line=dict(color="#2563EB", dash="dash")))
    fig.add_trace(
        go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_high) + list(fc_low[::-1]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            name="Confidence interval"
        )
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Model used: {used_model}")

def sentiment_tab(company: str, ticker: str, llm_budget: int, show_wordcloud: bool):
    st.subheader("Sentiment")
    try:
        corpus = build_corpus(company, ticker, limit=12)
        if corpus.empty:
            st.warning("No corpus items found (news may be temporarily unavailable).")
            return
        scored = score_corpus_with_gemini(corpus, llm_budget=llm_budget)
        agg = float(scored["sentiment"].mean())
    except Exception as e:
        st.error(f"Sentiment analysis failed: {e}")
        return

    st.metric("Aggregate sentiment", f"{agg:.2f}")

    st.markdown("### Recent items")
    preview = scored.copy()
    preview["text"] = preview["text"].str.slice(0, 200)
    st.dataframe(preview[["published_at", "title", "sentiment", "link"]], use_container_width=True)

    st.markdown("### Sentiment distribution")
    st.bar_chart(scored["sentiment"])

    if show_wordcloud:
        wc_img = make_wordcloud(list(scored["text"].values), st.session_state.dark_mode)
        buf = io.BytesIO()
        wc_img.save(buf, format="PNG")
        st.image(buf.getvalue(), caption="Sentiment word cloud", use_column_width=True)

def alerts_tab(company: str, ticker: str, thresholds: dict, model_name: str, periods: int):
    st.subheader("Alerts")
    try:
        market_df = fetch_historical_data(ticker, period="1y")
        fc_df, used_model = build_forecast(market_df, model_name, periods=periods)
        corpus = build_corpus(company, ticker, limit=12)
        scored = score_corpus_with_gemini(corpus, llm_budget=8) if not corpus.empty else pd.DataFrame()
        agg_sent = float(scored["sentiment"].mean()) if not scored.empty else 0.0
    except Exception as e:
        st.error(f"Preparation failed: {e}")
        return

    projected = compute_projected_move(market_df, fc_df)
    signal_info = compute_signal(projected, agg_sent, thresholds)

    st.markdown(
        f"<div class='metric-card'><strong>Signal</strong>"
        f"<div style='margin-top:8px'><span class='signal-badge' style='background:{signal_info['color']}'>{signal_info['signal']}</span></div>"
        f"<div class='subtext' style='margin-top:6px'>{signal_info['reason']}</div></div>",
        unsafe_allow_html=True
    )

    st.markdown("### Alert preview")
    top_items = []
    try:
        for _, row in (scored.head(3) if not scored.empty else pd.DataFrame()).iterrows():
            top_items.append(str(row.get("title") or row.get("text"))[:160])
    except Exception:
        pass
    preview_msg = (
        f"{company} ({ticker})\n"
        f"Signal: {signal_info['signal']}\n"
        f"Projected: {projected['pct_change']:+.2f}% → ${projected['proj_price']:.2f}\n"
        f"Sentiment: {agg_sent:.1f}\n"
        f"Model: {used_model}\n"
        f"Reason: {signal_info['reason']}\n"
        f"Top insights:\n" + ("\n".join([f"- {x}" for x in top_items]) if top_items else "- none -")
    )
    st.code(preview_msg)

    if st.button("Send Slack alert"):
        send_slack_alert(company, ticker, signal_info, projected, agg_sent, used_model, top_items)

# ---------------------------
# Main app
# ---------------------------
def main():
    st.title("Infosys InsightSphere")
    st.markdown("<div class='subtext'>Real-Time Strategic Intelligence — Executive Insights</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("Configuration")
    company = st.sidebar.text_input("Company name", "Tesla, Inc.").strip()
    ticker = st.sidebar.text_input("Ticker", "TSLA").strip().upper()

    st.sidebar.header("Forecast settings")
    model_name = st.sidebar.selectbox("Forecast model", options=["ARIMA", "Prophet"], index=0, help="Prophet requires the 'prophet' package.")
    if model_name == "Prophet" and not PROPHET_AVAILABLE:
        st.sidebar.warning("Prophet is not installed; ARIMA will be used.")
        model_name = "ARIMA"
    periods = st.sidebar.slider("Forecast horizon (days)", min_value=3, max_value=21, value=7, step=1)

    st.sidebar.header("Sentiment settings")
    llm_budget = st.sidebar.slider("Gemini scoring budget (items)", min_value=1, max_value=12, value=8)
    show_wordcloud = st.sidebar.checkbox("Show word cloud", value=True)

    st.sidebar.header("Alert thresholds")
    buy_thr = st.sidebar.number_input("Buy threshold (%)", value=3.0, step=0.5, format="%.1f")
    sell_thr = st.sidebar.number_input("Sell threshold (%)", value=-3.0, step=0.5, format="%.1f")
    pos_sent = st.sidebar.number_input("Positive sentiment threshold", value=15.0, step=1.0, format="%.0f")
    neg_sent = st.sidebar.number_input("Negative sentiment threshold", value=-15.0, step=1.0, format="%.0f")
    thresholds = {
        "buy_threshold_pct": buy_thr,
        "sell_threshold_pct": sell_thr,
        "positive_sentiment": pos_sent,
        "negative_sentiment": neg_sent,
    }

    st.sidebar.header("Appearance")
    toggle = st.sidebar.checkbox("Dark mode", value=st.session_state.dark_mode)
    if toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = toggle
        apply_theme(st.session_state.dark_mode)

    # Tabs
    tabs = st.tabs(["Overview", "Forecast", "Sentiment", "Alerts"])
    with tabs[0]:
        overview_tab(company, ticker)
    with tabs[1]:
        forecast_tab(company, ticker, model_name, periods)
    with tabs[2]:
        sentiment_tab(company, ticker, llm_budget, show_wordcloud)
    with tabs[3]:
        alerts_tab(company, ticker, thresholds, model_name, periods)

    st.markdown("---")
    st.markdown("<div class='subtext' style='text-align:center'>Infosys InsightSphere — Enterprise Edition</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
