import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import feedparser
import wikipedia
import os
import random
import sys
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION AND CONDITIONAL IMPORTS ---

# Global settings
random.seed(42)
np.random.seed(42)
pd.set_option("display.max_columns", 50)
FORECAST_DAYS = 7
DEFAULT_COMPANY_NAME = "Infosys Ltd."
DEFAULT_TICKER = "INFY"
INVALID_COMPANY_KEYWORDS = {"abc", "xyz", "test", "testing", "demo", "sample", "qwerty", "asdf", "fake", "dummy", "123", "456"}

# LLM & Forecasting Setup
# Priority: Streamlit secrets > Environment variable
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    pass # Prophet (and pystan) is a heavy dependency and might fail in some environments

GEMINI_ENABLED = False
GEMINI_MODEL = None
try:
    import google.generativeai as genai
    from statsmodels.tsa.arima.model import ARIMA
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a model that can handle system instructions
        GEMINI_MODEL = genai.GenerativeModel("gemini-2.5-flash") 
        GEMINI_ENABLED = True
except Exception:
    pass # Fails if API key or library is missing/broken
    
# SYSTEM INSTRUCTION for LLM Scoring
SYSTEM_INSTRUCTION = (
    "You are an expert financial sentiment analyst. "
    "Analyze the following text regarding a publicly traded company. "
    "Respond ONLY with a single integer score between -100 (Extremely Negative) and +100 (Extremely Positive). "
    "Do not include any other text, explanation, or punctuation."
)

# =======================================================
# 2. STREAMLIT PAGE CONFIGURATION (MUST BE HERE)
# =======================================================
st.set_page_config(layout="wide", page_title="Strategic Intelligence System")

# --- 3. DATA FETCHING (with Streamlit Caching) ---

@st.cache_data(ttl=3600, show_spinner="Fetching Company Profile...")
def fetch_company_info(company_name: str, ticker: str):
    """Fetches Wikipedia and Yahoo Finance summaries."""
    # Wikipedia Summary
    wiki_summary = "Wikipedia summary not available."
    try:
        wikipedia.set_lang("en")
        results = wikipedia.search(company_name)
        wiki_summary = wikipedia.summary(results[0], sentences=4) if results else "No Wikipedia page found."
    except Exception:
        pass

    # Yahoo Finance Info
    yf_info = yf.Ticker(ticker).info
    return {
        "wiki_summary": wiki_summary,
        "sector": yf_info.get("sector", "N/A"),
        "industry": yf_info.get("industry", "N/A"),
        "country": yf_info.get("country", "N/A"),
        "website": yf_info.get("website", "N/A"),
        "businessSummary": yf_info.get("longBusinessSummary", "Summary not available."),
        "longName": yf_info.get("longName", company_name)
    }

@st.cache_data(ttl=300, show_spinner="Fetching Market Data...")
def fetch_historical_data(ticker: str):
    """Fetch historical OHLCV data and current metrics."""
    # Fetch 1 year of daily data
    df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=False).dropna().reset_index()
    df = df.rename(columns={"Date": "date"})

    t = yf.Ticker(ticker)
    # Get current price from a recent history check
    hist_5d = t.history(period="5d")
    current_price = float(hist_5d["Close"].iloc[-1]) if not hist_5d.empty else None
    market_cap = t.info.get("marketCap")

    return df, current_price, market_cap

@st.cache_data(ttl=300, show_spinner="Fetching News & Generating Mock Social Data...")
def fetch_sentiment_data(company_name: str):
    """Fetch Google News and generate mock social data."""
    
    # 1. Google News Fetcher
    query = company_name.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    news_rows = []
    for entry in feed.entries[:10]:
        title = entry.get("title", "").strip()
        summary = entry.get("summary", "").strip()
        text = f"{title}. {summary}".replace("\n", " ").strip()
        try: pub_date = datetime(*entry.published_parsed[:6])
        except: pub_date = datetime.now()
        news_rows.append({"source": "news", "text": text, "link": entry.get("link", ""), "published_at": pub_date})
    news_df = pd.DataFrame(news_rows)

    # 2. Mock Social Data Generator
    templates = {
        "positive": [f"{company_name} smashed earnings targets! Bullish momentum is here. (positive)", f"Future looks bright for {company_name}. Major new product line announced. (positive)"],
        "negative": [f"{company_name} shows weak outlook today. Investors are nervous. (negative)", f"Analyst downgrade for {company_name}. Sell signal is flashing. (negative)"],
        "neutral": [f"Watching {company_name} closely ahead of next week's meeting. (neutral)", f"Market consolidation around {company_name}. Sideways trading likely. (neutral)"]
    }
    twitter_rows = []
    end_time = datetime.now()
    for i in range(15):
        sentiment_key = random.choice(list(templates.keys()))
        text = random.choice(templates[sentiment_key])
        time_offset = random.random() * 24
        post_time = end_time - timedelta(hours=time_offset)
        twitter_rows.append({"source": "twitter", "text": text, "link": "", "published_at": post_time})
    twitter_df = pd.DataFrame(twitter_rows)

    # 3. Unified Corpus
    cols = ["source", "text", "link", "published_at"]
    corpus_df = pd.concat([news_df[cols], twitter_df[cols]], ignore_index=True)
    return corpus_df.sort_values("published_at", ascending=False).reset_index(drop=True)

# --- 4. SENTIMENT ANALYSIS ---

def score_sentiment_llm(text: str) -> int:
    """Scores sentiment using the Gemini LLM."""
    try:
        response = GEMINI_MODEL.generate_content(
            text, config={"system_instruction": SYSTEM_INSTRUCTION}
        )
        score = int(float(response.text.strip().replace(",", "")))
        return max(-100, min(100, score))
    except Exception:
        return 0

def score_sentiment_rule_based_fallback(text: str) -> int:
    """A simple rule-based fallback sentiment scorer."""
    text_lower = text.lower()
    score = 0
    positive_words = ["bullish", "smashed", "strong", "growth", "potential", "buy", "surge", "gain"]
    negative_words = ["weak", "recall", "tumbling", "sell", "downgrade", "headwinds", "plague", "dip", "loss"]
    for word in positive_words: score += 10 * text_lower.count(word)
    for word in negative_words: score -= 15 * text_lower.count(word)
    return max(-100, min(100, score))

@st.cache_data(ttl=300, show_spinner="Running Sentiment Scoring...")
def apply_sentiment_scoring(corpus_df: pd.DataFrame):
    """Applies the LLM or fallback scorer to the entire corpus."""
    corpus_df["sentiment"] = 0
    
    if GEMINI_ENABLED and not corpus_df.empty:
        # Use a placeholder in the UI for the scoring process
        # The spinner in @st.cache_data handles the time, no extra st.info needed here.
        for index, row in corpus_df.iterrows():
            try:
                score = score_sentiment_llm(row["text"])
                corpus_df.loc[index, "sentiment"] = score
            except:
                corpus_df.loc[index, "sentiment"] = score_sentiment_rule_based_fallback(row["text"])
    else:
        corpus_df["sentiment"] = corpus_df["text"].apply(score_sentiment_rule_based_fallback)
    
    sent_norm = corpus_df["sentiment"].mean() if not corpus_df.empty else 0.0
    return corpus_df, sent_norm

# --- 5. TIME-SERIES FORECASTING ---

@st.cache_data(ttl=3600, show_spinner="Running Time-Series Forecasting...")
def run_time_series_forecasting(df: pd.DataFrame):
    """Runs Prophet or ARIMA for forecasting."""
    if df.empty:
        return pd.DataFrame(), "N/A", pd.DataFrame()

    model_used = "ARIMA" # Default to ARIMA
    
    if PROPHET_AVAILABLE:
        try:
            model_used = "Prophet"
            df_prophet = df.rename(columns={"date": "ds", "Close": "y"})
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, interval_width=0.90)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=FORECAST_DAYS)
            forecast_df = model.predict(future)
            forecast_df = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})
            
        except Exception:
            model_used = "ARIMA" # Fallback if Prophet runs but fails during fit/predict

    if model_used == "ARIMA":
        # ARIMA Fallback
        try:
            train_data = df["Close"].values[-60:]
            model = ARIMA(train_data, order=(1, 1, 0))
            model_fit = model.fit()
            forecast_results = model_fit.get_forecast(steps=FORECAST_DAYS)
            yhat = forecast_results.predicted_mean
            conf_int = forecast_results.conf_int(alpha=0.10) 
            
            last_date = df["date"].iloc[-1]
            forecast_dates = [last_date + timedelta(days=i) for i in range(1, FORECAST_DAYS + 1)]
            
            forecast_df_future = pd.DataFrame({
                "date": forecast_dates,
                "yhat": yhat,
                "yhat_lower": conf_int.iloc[:, 0].values,
                "yhat_upper": conf_int.iloc[:, 1].values
            })
            
            # Combine historical and forecast for plotting consistency
            forecast_df = df.rename(columns={"Close": "yhat"})[["date", "yhat"]].copy()
            forecast_df["yhat_lower"] = forecast_df["yhat"]
            forecast_df["yhat_upper"] = forecast_df["yhat"]
            forecast_df = pd.concat([forecast_df, forecast_df_future], ignore_index=True)

        except Exception:
            st.error("ARIMA forecasting also failed.")
            return pd.DataFrame(), "ARIMA (Failed)", pd.DataFrame()


    # Filter for future forecast data for metrics
    last_hist_date = df["date"].max()
    future_forecast_df = forecast_df[forecast_df["date"] > last_hist_date].copy()

    return forecast_df, model_used, future_forecast_df

# --- 6. SIGNAL COMPUTATION & HELPERS ---

def determine_forecast_trend(future_forecast_df: pd.DataFrame, current_price: float):
    """Calculates the 7-day predicted price change and trend direction."""
    if future_forecast_df.empty or not current_price:
        return 0.0, "NEUTRAL"
    final_predicted_price = future_forecast_df["yhat"].iloc[-1]
    price_change_pct = ((final_predicted_price - current_price) / current_price) * 100
    if price_change_pct > 2: trend = "UP"
    elif price_change_pct < -2: trend = "DOWN"
    else: trend = "FLAT"
    return price_change_pct, trend

def compute_trading_signal(forecast_trend: str, aggregate_sentiment: float) -> str:
    """Generates the final trading signal (BUY/SELL/HOLD)."""
    if forecast_trend == "UP" and aggregate_sentiment >= 10:
        return "BUY"
    elif forecast_trend == "DOWN" and aggregate_sentiment <= -10:
        return "SELL"
    else:
        return "HOLD"

def format_market_cap(mc_raw):
    """Formats raw market cap number into Trillion/Billion/Million string."""
    if mc_raw:
        if mc_raw >= 1e12: return f"${mc_raw/1e12:.2f} Trillion"
        if mc_raw >= 1e9: return f"${mc_raw/1e9:.2f} Billion"
        if mc_raw >= 1e6: return f"${mc_raw/1e6:.2f} Million"
        return f"${mc_raw:,}"
    return "N/A"

def compute_risk_profile(market_df: pd.DataFrame, corpus_df: pd.DataFrame):
    """Generates synthetic risk indices (0-100)."""
    risk = {}
    
    # 1. Volatility Risk
    market_df["daily_return"] = market_df["Close"].pct_change()
    volatility = market_df["daily_return"].std() * np.sqrt(252) * 100
    risk["Volatility"] = min(100, max(0, volatility * 2))

    # 2. Sentiment Dispersion Risk
    if not corpus_df.empty:
        sentiment_variance = corpus_df["sentiment"].std()
        risk["Sentiment Dispersion"] = min(100, max(0, sentiment_variance * 2.5))
    else:
        risk["Sentiment Dispersion"] = 50

    # 3. Liquidity Risk
    avg_volume = market_df["Volume"].mean()
    risk["Liquidity"] = min(100, max(0, 100 - (avg_volume / 1e8))) 
    
    return risk

# --- 7. PLOTLY DASHBOARD COMPONENT ---

def plot_price_and_forecast(market_df, forecast_df, future_forecast_df, ticker):
    """Generates the Plotly chart for historical data and forecast."""
    fig = go.Figure()

    # Get Historical Data (Filter forecast_df to only show historical points)
    hist_data = forecast_df[forecast_df["date"] <= market_df["date"].max()]
    
    # Historical Price
    fig.add_trace(go.Scatter(
        x=hist_data["date"], y=hist_data["yhat"], mode="lines", 
        name="Historical Close", line=dict(color='blue')
    ))

    # Forecast Line
    future_dates = future_forecast_df["date"]
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_forecast_df["yhat"], mode="lines+markers", 
        name="Forecast (7D)", line=dict(color='red', dash='dash')
    ))

    # Confidence Interval
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates)[::-1],
        y=list(future_forecast_df["yhat_upper"]) + list(future_forecast_df["yhat_lower"])[::-1],
        fill="toself", fillcolor="rgba(255, 0, 0, 0.1)", line=dict(color="rgba(0,0,0,0)"),
        name="Forecast CI", hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"Price History and {FORECAST_DAYS}-Day Forecast for {ticker}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=500
    )
    return fig

# --- 8. STREAMLIT APPLICATION LAYOUT ---

def main_app():

    # --- Sidebar for Input ---
    with st.sidebar:
        st.title("Settings")
        user_company_input = st.text_input("Enter Company Name/Ticker", value=DEFAULT_COMPANY_NAME)
        st.caption("E.g., Infosys Ltd., MSFT, GOOGL")
        
        st.markdown("---")
        
        if not GEMINI_API_KEY:
            st.warning("LLM API Key not found! Sentiment will use a rule-based fallback.")
        elif GEMINI_ENABLED:
            st.success("Gemini LLM is configured and active.")
        else:
            st.warning("Gemini LLM is configured, but failed to initialize.")
            
        if PROPHET_AVAILABLE:
            st.info("Prophet forecasting available.")
        else:
            st.warning("Prophet not available. Using ARIMA fallback.")
            
        st.markdown("---")
        st.markdown("Developed using Streamlit, Plotly, yfinance, and Google Gemini.")
        
    # --- Main Application Logic ---
    st.title("Real-Time Strategic Intelligence System")
    st.markdown("---")
    
    # 1. Company Validation and Ticker Lookup
    if not user_company_input or user_company_input.strip().lower() in INVALID_COMPANY_KEYWORDS:
        st.error("Please enter a valid company name or ticker.")
        st.stop()
        
    try:
        yf_ticker = yf.Ticker(user_company_input)
        info = yf_ticker.info
        
        # Determine the best display name
        long_name = info.get("longName")
        company_name = long_name if long_name and len(long_name) > 3 else user_company_input
        ticker = info.get("symbol", user_company_input).upper()
        
        # Simple check to see if the ticker returned valid info
        if not info.get("marketCap") and not info.get("currentPrice"):
            st.error(f"Could not retrieve sufficient market data for '{user_company_input}'. Please verify the ticker or company name.")
            st.stop()
            
    except Exception:
        st.error(f"Could not find a valid ticker for '{user_company_input}'. Check the spelling.")
        st.stop()

    st.header(f"Analysis for: {company_name} ({ticker})")

    # 2. RUN ANALYSIS
    try:
        # Step A: Data Acquisition
        market_df, current_price, market_cap = fetch_historical_data(ticker)
        corpus_df = fetch_sentiment_data(company_name)
        
        # Step B: Sentiment Scoring
        corpus_df, sent_norm = apply_sentiment_scoring(corpus_df.copy())
        
        # Step C: Forecasting
        forecast_df, model_used, future_forecast_df = run_time_series_forecasting(market_df.copy())
        
        # Step D: Signal Generation
        forecast_pct, forecast_trend = determine_forecast_trend(future_forecast_df, current_price)
        trading_signal = compute_trading_signal(forecast_trend, sent_norm)
        risk_dict = compute_risk_profile(market_df, corpus_df)
        
        # Get Info
        info_data = fetch_company_info(company_name, ticker)

    except Exception as e:
        st.error(f"An unexpected error occurred during analysis: {e}")
        st.stop()

    # --- 3. Dashboard Metrics and Signal ---

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Trading Signal", trading_signal, help="BUY/SELL/HOLD based on combined Forecast and Sentiment.")
    col2.metric("Current Price", f"${current_price:,.2f}", 
                f"{forecast_pct:+.2f}% (7-Day Forecast)", 
                help="Current market price and 7-Day forecasted change.")
    col3.metric("Aggregate Sentiment", f"{sent_norm:+.1f}", 
                help="Normalized average sentiment score from -100 (Bearish) to +100 (Bullish).")
    col4.metric("Market Cap", format_market_cap(market_cap), help="Total valuation of the company's shares.")

    st.markdown("---")

    # --- 4. Main Charts and Tabs ---
    
    st.plotly_chart(plot_price_and_forecast(market_df, forecast_df, future_forecast_df, ticker), use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Company Profile", "Sentiment Analysis & Corpus", "Risk & Metrics"])

    with tab1:
        st.subheader("Business Profile")
        st.markdown(f"**Sector:** {info_data['sector']} | **Industry:** {info_data['industry']} | **Country:** {info_data['country']}")
        
        st.markdown("#### Wikipedia Summary")
        st.info(info_data["wiki_summary"])
        
        st.markdown("#### Business Summary (Yahoo Finance)")
        st.write(info_data["businessSummary"])

    with tab2:
        st.subheader("Sentiment Analysis Results")
        
        sent_col1, sent_col2 = st.columns([1, 2])
        
        with sent_col1:
            st.markdown("#### Score Breakdown")
            
            # Sentiment buckets for table/chart
            def bucket_sentiment(score):
                if score >= 10: return "Bullish"
                if score <= -10: return "Bearish"
                return "Neutral"
            sent_buckets = corpus_df["sentiment"].apply(bucket_sentiment)
            sent_counts = sent_buckets.value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]

            st.dataframe(sent_counts, hide_index=True, use_container_width=True)
            st.info(f"Scoring Model: {'Gemini LLM' if GEMINI_ENABLED else 'Rule-Based Fallback'}")

        with sent_col2:
            st.markdown("#### Unified Text Corpus")
            corpus_display = corpus_df.rename(columns={
                "published_at": "Timestamp", 
                "text": "Source Text", 
                "sentiment": "Score"
            }).drop(columns=['link'])
            st.dataframe(corpus_display, use_container_width=True, height=400)

    with tab3:
        st.subheader("Risk Profile and Forecasting Details")
        
        risk_col, metric_col = st.columns(2)
        
        with risk_col:
            st.markdown("#### Synthetic Risk Indices (0-100)")
            risk_df = pd.DataFrame(list(risk_dict.items()), columns=["Risk Index", "Score"]).set_index("Risk Index")
            st.dataframe(
                risk_df.style.bar(subset=["Score"], color=['#ff4b4b', '#f5e416', '#00BA7C'], align='mid'), 
                use_container_width=True
            )
            st.caption("Scores are synthetic, based on historical volatility, sentiment dispersion, and liquidity.")

        with metric_col:
            st.markdown("#### Forecasting Details")
            st.metric("Model Used", model_used)
            if not future_forecast_df.empty:
                st.metric("Forecasted End Price", f"${future_forecast_df['yhat'].iloc[-1]:,.2f}")
                st.metric("90% Confidence Interval Range", 
                          f"${future_forecast_df['yhat_lower'].iloc[-1]:,.2f} - ${future_forecast_df['yhat_upper'].iloc[-1]:,.2f}")
                st.caption(f"Forecast period: {future_forecast_df['date'].iloc[0].strftime('%Y-%m-%d')} to {future_forecast_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
            else:
                st.info("Forecast data is not available.")


if __name__ == "__main__":
    # Custom styles for a cleaner look
    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    main_app()
