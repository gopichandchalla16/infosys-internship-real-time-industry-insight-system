import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import requests
import random
import time
from datetime import datetime, timedelta
from typing import Union, Dict

# --- Conditional Imports ---
try:
    import wikipedia
except ImportError:
    wikipedia = None
    
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_ENABLED = False 
except ImportError:
    GEMINI_ENABLED = False

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Global Settings ---
random.seed(42)
np.random.seed(42)

# --- 0. Initialization & Secrets ---

# --- THEME MANAGEMENT LOGIC START ---
THEME_KEY = "current_theme"
DEFAULT_THEME = "dark" # Setting dark as the initial default
if THEME_KEY not in st.session_state:
    st.session_state[THEME_KEY] = DEFAULT_THEME

def toggle_theme():
    """Toggles the 'current_theme' session state variable and forces a rerun."""
    if st.session_state[THEME_KEY] == "light":
        st.session_state[THEME_KEY] = "dark"
    else:
        st.session_state[THEME_KEY] = "light"
    # st.experimental_rerun() must be called in the main function/logic, 
    # not within this function, to ensure proper state transition.

# Determine current theme for config
current_theme = st.session_state[THEME_KEY]
# --- THEME MANAGEMENT LOGIC END ---

# --- SET PAGE CONFIGURATION (MUST BE FIRST STREAMLIT CALL) ---

# Streamlit config uses the base setting to control light/dark mode
if current_theme == "dark":
    st.set_page_config(
        layout="wide", 
        page_title="Strategic Intelligence System",
        initial_sidebar_state="expanded",
    )
else:
    st.set_page_config(
        layout="wide", 
        page_title="Strategic Intelligence System",
        initial_sidebar_state="expanded",
    )
    
# --- END SET PAGE CONFIGURATION ---

def initialize_llm_and_keys():
    """Initializes LLM and checks for API keys from Streamlit secrets."""
    global GEMINI_ENABLED, GEMINI_MODEL, ALPHA_VANTAGE_API_KEY, SLACK_WEBHOOK_URL
    
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            # Using gemini-2.0-flash is faster and cost-effective for sentiment
            GEMINI_MODEL = genai.GenerativeModel("gemini-2.0-flash") 
            GEMINI_ENABLED = True
        except Exception:
            GEMINI_ENABLED = False

    ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    SLACK_WEBHOOK_URL = st.secrets.get("SLACK_WEBHOOK_URL", "")

    return GEMINI_ENABLED, ALPHA_VANTAGE_API_KEY, SLACK_WEBHOOK_URL

# --- WIKIPEDIA/YFINANCE SUMMARY FUNCTIONS ---

@st.cache_data(ttl=24*3600)
def get_wikipedia_summary(company_name: str, sentences: int = 4) -> str:
    """
    Fetch a short Wikipedia summary for the company, handling errors gracefully.
    """
    if wikipedia is None:
        return "Wikipedia library not installed."
        
    try:
        wikipedia.set_lang("en")
        try:
            return wikipedia.summary(company_name, sentences=sentences, auto_suggest=True)
        except wikipedia.exceptions.PageError:
            search_results = wikipedia.search(company_name, results=5)
            if search_results:
                page_title = search_results[0]
                return wikipedia.summary(page_title, sentences=sentences)
            return "No Wikipedia page found for this company."
        except wikipedia.exceptions.DisambiguationError as e:
            for option in e.options:
                if "corporation" in option.lower() or "company" in option.lower():
                    try: return wikipedia.summary(option, sentences=sentences)
                    except: continue
            if e.options:
                try: return wikipedia.summary(e.options[0], sentences=sentences)
                except: pass
            return "Wikipedia page is ambiguous or unavailable."
        except Exception:
            return "Wikipedia summary not available."

    except Exception:
        return "Wikipedia summary not available due to general error."

@st.cache_data(ttl=24*3600)
def get_yfinance_company_summary(ticker: str) -> dict:
    """
    Get company business summary, sector, industry, and key metadata from Yahoo Finance.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info

        return {
            "longName": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "country": info.get("country"),
            "website": info.get("website"),
            "businessSummary": info.get("longBusinessSummary"),
        }
    except Exception:
        return {
            "longName": None, "sector": None, "industry": None, 
            "country": None, "website": None, "businessSummary": "Not available."
        }


# --- CORE HELPER FUNCTIONS ---

def sentiment_label(score: float) -> str:
    """Classifies numerical sentiment score into a descriptive label."""
    if score >= 20:
        return "Bullish"
    elif score <= -20:
        return "Bearish"
    else:
        return "Neutral"

def trend_label(pct: float) -> str:
    """Classifies percentage price change into a descriptive trend."""
    if pct > 2:
        return "↑ Strong Uptrend"
    elif pct < -2:
        return "↓ Strong Downtrend"
    else:
        return "→ Sideways"

def format_market_cap(mc_raw: Union[float, int, None]) -> str:
    """Formats market capitalization into Trillions, Billions, or Millions."""
    if not mc_raw:
        return "N/A"
    
    mc_raw = float(mc_raw)
    if mc_raw >= 1e12: 
        return f"${mc_raw/1e12:.2f} Trillion"
    elif mc_raw >= 1e9: 
        return f"${mc_raw/1e9:.2f} Billion"
    elif mc_raw >= 1e6: 
        return f"${mc_raw/1e6:.2f} Million"
    else: 
        return f"${mc_raw:,.0f}"

# --- 1. Core Data Sourcing & Utility Functions ---

@st.cache_data(ttl=24*3600)
def search_ticker_by_company(company_name: str):
    """Searches Yahoo Finance for the stock ticker and full company name."""
    try:
        query = company_name.replace(" ", "+")
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
        response = requests.get(url, headers=headers, timeout=5)
        data = response.json()
        
        for item in data.get("quotes", []):
            if item.get("quoteType") == "EQUITY":
                symbol = item.get("symbol")
                longname = item.get("longname") or item.get("shortname") or symbol
                return symbol, longname
        return None, None
    except Exception:
        return None, None

@st.cache_data(ttl=60*60)
def fetch_historical_data(ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Fetches historical price data from Yahoo Finance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna().reset_index()
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(ttl=60*60)
def fetch_market_metrics(ticker: str) -> Dict[str, Union[str, float, None]]:
    """Fetches current price and fundamental metrics."""
    t = yf.Ticker(ticker)
    info = {}
    try:
        info = t.info
    except Exception:
        pass
        
    hist = t.history(period="5d")
    last_price = float(hist["Close"].iloc[-1]) if not hist.empty and not hist["Close"].empty else 0.0

    return {
        "current_price": last_price,
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "country": info.get("country", "N/A"),
    }
    
def get_market_data_robust(ticker: str, alpha_vantage_key: str):
    """Fetches historical data, currently defaulting to Yahoo Finance."""
    try:
        market_df = fetch_historical_data(ticker)
        st.sidebar.success("Market Data Source: Yahoo Finance")
        return market_df
    except Exception as e:
        st.sidebar.error(f"Data fetch failed: {e}")
        return pd.DataFrame()

# --- 2. Sentiment Analysis & Corpus ---

@st.cache_data(ttl=60*60)
def fetch_google_news(query: str, max_items: int = 10) -> pd.DataFrame:
    """Fetches recent news items from Google RSS."""
    q = query.replace(" ", "+")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    rows = []
    
    for entry in feed.entries[:max_items]:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        text = title.replace("\n", " ").strip()

        rows.append({
            "source": "news", "title": title, "text": text, 
            "link": link, "published_at": datetime.now() 
        })
    return pd.DataFrame(rows)

def generate_mock_social_sentiment(company: str, posts_per_day: int = 15) -> pd.DataFrame:
    """Generates mock social posts for a quick sentiment base."""
    POS_PHRASES = ["strong earnings", "bullish momentum", "record demand"]
    NEG_PHRASES = ["weak outlook", "bearish signals", "slowing demand"]
    NEUTRAL_PHRASES = ["sideways movement", "watching closely"]
    
    rows = []
    
    for _ in range(posts_per_day):
        roll = random.random()
        if roll < 0.33:
            phrase = random.choice(POS_PHRASES); sentiment_score=random.randint(25, 60)
        elif roll < 0.66:
            phrase = random.choice(NEG_PHRASES); sentiment_score=random.randint(-60, -25)
        else:
            phrase = random.choice(NEUTRAL_PHRASES); sentiment_score=random.randint(-15, 15)

        text = f"{company} shows {phrase} today."

        rows.append({
            "source": "twitter", "title": "Social Post", "text": text,
            "sentiment": sentiment_score
        })
    return pd.DataFrame(rows)

# Lexicon for local sentiment (used as fallback or for social media)
LEXICON = {
    "buy": 50, "gain": 40, "growth": 30, "strong": 20, "positive": 20, "upgrade": 30,
    "sell": -50, "loss": -40, "drop": -30, "weak": -20, "negative": -20, "downgrade": -30,
}

def local_sentiment(text: str) -> int:
    """Simple lexicon-based sentiment scoring."""
    score = 0
    words = text.lower().split()
    for word, value in LEXICON.items():
        if word in words:
            score += value
    return max(-100, min(100, score))

def gemini_sentiment(text: str) -> int:
    """Uses Gemini API for more nuanced sentiment analysis (for news)."""
    global GEMINI_ENABLED, GEMINI_MODEL

    if not GEMINI_ENABLED:
        return local_sentiment(text)
    
    try:
        prompt = f"""
Analyze the sentiment of the following market-related text for the primary company.
Output ONLY an integer between -100 (extremely negative) and +100 (extremely positive).

Text:
{text}
"""
        time.sleep(0.5)
        result = GEMINI_MODEL.generate_content(prompt)
        # Robustly parse the integer output
        val_str = result.text.strip().replace('+', '').replace('-', 'N').replace('.', '').replace('*', '')
        val = int(''.join(filter(str.isdigit, val_str)).replace('N', '-'))
        return max(-100, min(100, val))

    except Exception:
        return local_sentiment(text)

@st.cache_data(ttl=60*60)
def compute_sentiment_for_corpus(df: pd.DataFrame, gemini_enabled: bool) -> pd.DataFrame:
    """Applies sentiment scoring to the corpus."""
    if df.empty:
        return df

    news_df = df[df["source"] == "news"].copy()
    social_df = df[df["source"] != "news"].copy()

    if not news_df.empty:
        scores = []
        progress_bar = st.progress(0, text="Analyzing news sentiment...")
        total_rows = len(news_df)
        
        for i, row in news_df.iterrows():
            scores.append(gemini_sentiment(row["text"]) if gemini_enabled else local_sentiment(row["text"]))
            progress_bar.progress((i + 1) / total_rows, text=f"Analyzing news sentiment... {i+1}/{total_rows}")
        
        news_df["sentiment"] = scores
        progress_bar.empty()
    
    # Apply local sentiment to social posts if not already scored (mock data might pre-score)
    if not social_df.empty and "sentiment" not in social_df.columns:
        social_df["sentiment"] = social_df["text"].apply(local_sentiment)
    elif "sentiment" not in social_df.columns: # Handle empty sentiment column case if present
        social_df["sentiment"] = 0

    return pd.concat([news_df, social_df], ignore_index=True)

# --- 3. Forecasting Functions ---

@st.cache_data(ttl=60*60)
def run_forecasting(df: pd.DataFrame, days: int = 7) -> tuple[pd.DataFrame, str]:
    """Runs a time series forecast using Prophet, falling back to ARIMA."""
    if df.empty or len(df) < 50:
        return pd.DataFrame(), "N/A (Insufficient Data)"
    
    prophet_df = df.rename(columns={"date": "ds", "Close": "y"})[["ds", "y"]]
    last_date = prophet_df['ds'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    
    forecast_df = pd.DataFrame()
    model_name = "ARIMA"
    
    # 1. Prophet Attempt
    if PROPHET_AVAILABLE:
        try:
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=days)
            forecast = model.predict(future)
            # Filter only the future dates
            forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][forecast['ds'].isin(future_dates)]
            model_name = "Prophet"
            return forecast_df, model_name
        except Exception:
            pass

    # 2. ARIMA Fallback
    if ARIMA_AVAILABLE:
        try:
            arima_data = prophet_df['y'].values
            # Using standard ARIMA order
            arima_model = ARIMA(arima_data, order=(5,1,0)) 
            arima_model_fit = arima_model.fit()

            forecast_result = arima_model_fit.get_forecast(steps=days)
            arima_predictions = forecast_result.predicted_mean
            conf_ints = forecast_result.conf_int()
            
            forecast_df = pd.DataFrame({
                'ds': future_dates,
                'yhat': arima_predictions,
                'yhat_lower': conf_ints[:, 0],
                'yhat_upper': conf_ints[:, 1]
            })
            model_name = "ARIMA"
            return forecast_df, model_name
        except Exception:
            pass
            
    return pd.DataFrame(), "N/A (Model Error)"

def calculate_signal(forecast_df: pd.DataFrame, market_df: pd.DataFrame, sentiment_score: float) -> tuple[str, float, float]:
    """Determines the final trading signal."""
    if forecast_df.empty or market_df.empty:
        return "HOLD", 0.0, 0.0

    try:
        last_historical_price = market_df['Close'].iloc[-1]
        if isinstance(last_historical_price, pd.Series):
             last_historical_price = last_historical_price.item()
        else:
             last_historical_price = float(last_historical_price)
    except Exception:
        return "HOLD", 0.0, 0.0
        
    forecasted_price_7_day = forecast_df['yhat'].mean() 
    
    if last_historical_price == 0.0:
        return "HOLD", 0.0, 0.0

    price_change_7_day = ((forecasted_price_7_day - last_historical_price) / last_historical_price) * 100

    # Thresholds for Signal Generation
    SIGNAL_BUY_THRESHOLD_PRICE_CHANGE = 1.5
    SIGNAL_SELL_THRESHOLD_PRICE_CHANGE = -1.5
    SIGNAL_BUY_THRESHOLD_SENTIMENT = 30.0
    SIGNAL_SELL_THRESHOLD_SENTIMENT = -30.0

    trading_signal = "HOLD"

    if (price_change_7_day > SIGNAL_BUY_THRESHOLD_PRICE_CHANGE) and (sentiment_score > SIGNAL_BUY_THRESHOLD_SENTIMENT):
        trading_signal = "BUY"
    elif (price_change_7_day < SIGNAL_SELL_THRESHOLD_PRICE_CHANGE) and (sentiment_score < SIGNAL_SELL_THRESHOLD_SENTIMENT):
        trading_signal = "SELL"
    
    return trading_signal, price_change_7_day, forecasted_price_7_day

# --- 4. Advanced Plotly Dashboard Visualization (FIXED AND OPTIMIZED) ---

def create_advanced_dashboard_figure(market_df, future_fc, corpus_df, market_metrics, COMPANY_NAME, TICKER, last_price, final_fc_price, forecast_pct, sent_norm, trading_signal, model_name):
    """
    Generates a comprehensive Plotly dashboard figure.
    FIX: Now correctly uses the 'date' column for forecast dates (renamed upstream).
    """
    
    # Helper Data
    hist_dates = market_df["date"]
    hist_close = market_df["Close"]
    
    # FIX: Use 'date' instead of 'ds' as the column was renamed in the main app logic
    fc_dates = future_fc["date"]
    fc_mean = future_fc["yhat"]
    fc_low = future_fc["yhat_lower"]
    fc_high = future_fc["yhat_upper"]
    
    sent_text = sentiment_label(sent_norm)
    trend_text = trend_label(forecast_pct)

    # Define colors based on signal for consistent visuals
    signal_color = "green" if trading_signal == "BUY" else ("red" if trading_signal == "SELL" else "orange")
    market_cap_str = format_market_cap(market_metrics.get("market_cap"))
    
    # --- Risk Metrics Calculation ---
    window_df = market_df.tail(60)
    
    price_change_series = window_df["Close"].pct_change().dropna()
    price_vol = 0.0
    if len(price_change_series) > 0:
        price_vol_std = price_change_series.std()
        try:
            price_vol = price_vol_std.item() * np.sqrt(252) * 100.0
        except:
            price_vol = price_vol_std * np.sqrt(252) * 100.0
    
    ci_width = (fc_high - fc_low).mean() if len(fc_high) > 0 else 0.0
    forecast_uncertainty = (ci_width / final_fc_price * 100.0) if final_fc_price != 0 else 0.0
    
    sent_vol = corpus_df["sentiment"].std()
    sent_vol = sent_vol.item() if isinstance(sent_vol, pd.Series) and not sent_vol.empty else (sent_vol if not pd.isna(sent_vol) else 0.0)
    
    news_count = int((corpus_df["source"] == "news").sum()) if "source" in corpus_df.columns else 0

    risk_labels = ["Price Volatility (Ann. %)", "Forecast Uncertainty (CI %)", "Sentiment Volatility (Std Dev)", "News Flow Intensity (Items)"]
    risk_values = [
        min(100.0, price_vol * 1.5), 
        min(100.0, forecast_uncertainty * 2.0), 
        min(100.0, abs(sent_vol)), 
        min(100.0, news_count * 5.0)
    ]
    
    # --- Sentiment Breakdown ---
    def bucket_sentiment(v):
        return "Bullish" if v >= 20 else ("Bearish" if v <= -20 else "Neutral")
    sent_buckets = corpus_df["sentiment"].apply(bucket_sentiment)
    sent_counts = sent_buckets.value_counts().reindex(["Bullish", "Neutral", "Bearish"], fill_value=0).to_dict()

    bull_cnt = sent_counts.get("Bullish", 0)
    bear_cnt = sent_counts.get("Bearish", 0)
    neut_cnt = sent_counts.get("Neutral", 0)

    # Plotly Subplots Setup
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"colspan": 2}, None],
            [{"type": "indicator"}, {"type": "table"}],
            [{"type": "bar"}, {"type": "table"}],
        ],
        subplot_titles=(
            "Historical Price vs 7-Day Forecast (Last 1 Year)",
            "Aggregate Sentiment Gauge",
            "Executive Metrics Summary",
            "Risk Profile (Synthetic Indices)",
            "Sentiment Breakdown"
        )
    )

    # 1. Price Chart (Row 1, Col 1) - Visual Fix: Added Range Slider
    fig.add_trace(go.Scatter(x=hist_dates, y=hist_close, mode="lines", name="Historical Close", line=dict(color='#2ECC71')), row=1, col=1)
    fig.add_trace(go.Scatter(x=fc_dates, y=fc_mean, mode="lines+markers", name=f"{model_name} Forecast (7D)", line=dict(color='#3498DB', dash='dot')), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(fc_high) + list(fc_low[::-1]),
            fill="toself", fillcolor="rgba(52, 152, 219, 0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Forecast CI", hoverinfo="skip"
        ), row=1, col=1
    )
    
    fig.update_xaxes(
        title_text="Date", 
        row=1, col=1,
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)

    # 2. Sentiment Gauge (Row 2, Col 1) - Visual Fix: Clearer labels
    gauge_color = "red" if sent_norm <= -20 else ("green" if sent_norm >= 20 else "orange")

    fig.add_trace(
        go.Indicator(
            mode="gauge+number", value=sent_norm,
            title={"text": f"Aggregate Sentiment: <span style='color:{gauge_color};'>{sent_text} ({sent_norm:.1f})</span>"},
            gauge={
                "axis": {"range": [-100, 100], "tickvals": [-100, -50, 0, 50, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [-100, -20], "color": "rgba(231, 76, 60, 0.4)"}, 
                    {"range": [-20, 20], "color": "rgba(243, 156, 18, 0.4)"},  
                    {"range": [20, 100], "color": "rgba(46, 204, 113, 0.4)"}  
                ],
            }
        ), row=2, col=1
    )

    # 3. Executive Metrics Table (Row 2, Col 2) - Visual Fix: High contrast headers/cells
    metrics_header = ["Metric", "Value"]
    metrics_rows = [
        ["Company", COMPANY_NAME],
        ["Ticker", TICKER],
        ["Market Cap", market_cap_str],
        ["Sector / Industry", f"{market_metrics.get('sector', 'N/A')} / {market_metrics.get('industry', 'N/A')}"],
        ["Current Price", f"${last_price:.2f}"],
        ["Forecasted Price (7D)", f"${final_fc_price:.2f}"],
        ["7-Day Change", f"{forecast_pct:.2f}% ({trend_text})"],
        ["Forecasting Model", model_name],
        ["**FINAL SIGNAL**", f"**{trading_signal}**"],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=metrics_header, fill_color="#2980B9", font=dict(color='white', size=14), align="left", line_color='white', line_width=1),
            cells=dict(values=list(zip(*metrics_rows)), align="left", fill_color='#F4F4F4', line_color='lightgrey', line_width=1),
        ), row=2, col=2
    )

    # 4. Risk Profile Bar Chart (Row 3, Col 1)
    fig.add_trace(
        go.Bar(
            x=[l.split('(')[0].strip() for l in risk_labels], 
            y=risk_values, name="Risk Indices",
            marker_color=['#F39C12', '#E67E22', '#D35400', '#C0392B'],
            text=[f"{v:.2f}" for v in risk_values],
            textposition='outside'
        ), row=3, col=1
    )
    fig.update_yaxes(title_text="Scaled Risk Index (0–100)", row=3, col=1)

    # 5. Sentiment Breakdown Table (Row 3, Col 2) - Visual Fix: High contrast headers/cells
    sent_table_header = ["Sentiment Bucket", "Count"]
    sent_table_rows = [
        ["Bullish", bull_cnt],
        ["Neutral", neut_cnt],
        ["Bearish", bear_cnt],
        ["Total Items", bull_cnt + bear_cnt + neut_cnt],
    ]

    fig.add_trace(
        go.Table(
            header=dict(values=sent_table_header, fill_color="#2980B9", font=dict(color='white', size=14), align="left", line_color='white', line_width=1),
            cells=dict(values=list(zip(*sent_table_rows)), align="left", fill_color='#F4F4F4', line_color='lightgrey', line_width=1),
        ), row=3, col=2
    )

    # Global Layout & Theming
    title_text = f"Strategic Intelligence Dashboard — {COMPANY_NAME} ({TICKER})"

    # Use the current_theme session state to select Plotly template
    plotly_template = 'plotly_dark' if st.session_state[THEME_KEY] == "dark" else 'plotly_white'

    fig.update_layout(
        height=1000, showlegend=False,
        title={
            'text': title_text, 'y':0.98, 'x':0.5, 'xanchor': 'center', 'yanchor': 'top',
            'font': dict(size=24)
        },
        margin=dict(l=40, r=40, t=100, b=40),
        # Use the dynamically selected template
        template=plotly_template
    )

    # Signal Badge Annotation
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=1.05,
        text=f"***FINAL SIGNAL: {trading_signal}***", showarrow=False,
        font=dict(size=24, color="white", family="Arial Black"), align="left",
        bgcolor=signal_color, opacity=1.0, borderpad=15
    )

    return fig

# --- 5. Notification Function ---

def send_slack_notification(message: str, webhook_url: str):
    """Sends a detailed message to Slack."""
    if not webhook_url:
        st.sidebar.warning("Slack webhook URL not configured in Streamlit secrets. Notification skipped.")
        return
        
    try:
        response = requests.post(webhook_url, json={"text": message}, timeout=10)
        response.raise_for_status() 
        st.sidebar.success("Slack notification sent successfully!")
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Failed to send Slack notification: {e}")

# --- 6. Main App Layout ---

# Custom CSS for theme button and general aesthetics
current_theme_color = "#FFFFFF" if current_theme == "dark" else "#000000"
button_bg_color = "#333333" if current_theme == "dark" else "#F0F2F6"
button_text_color = "#F0F2F6" if current_theme == "dark" else "#333333"

st.markdown(f"""
<style>
/* Adjust Streamlit elements for better aesthetics */
.st-emotion-cache-1jmpsc3, .st-emotion-cache-1629p8f {{
    border-radius: 8px;
    padding: 10px;
}}
/* Apply custom styling to the theme button in the sidebar */
.stButton>button {{
    color: {button_text_color};
    background-color: {button_bg_color};
    border-radius: 0.5rem;
    border: 1px solid {current_theme_color}22; /* slight border */
    transition: background-color 0.3s;
}}
.stButton>button:hover {{
    background-color: {button_bg_color}99; /* slight hover effect */
}}
/* Hide Plotly specific elements if needed */
.modebar, .plotly-notifier, .plotly-fade {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)


GEMINI_ENABLED, ALPHA_VANTAGE_API_KEY, SLACK_WEBHOOK_URL = initialize_llm_and_keys()

# --- Sidebar Input & Theme Button ---
st.sidebar.header("System Configuration")

# THEME TOGGLE BUTTON
theme_button_label = "Switch to Light Theme ☀️" if current_theme == "dark" else "Switch to Dark Theme 🌙"
if st.sidebar.button(theme_button_label, use_container_width=True):
    toggle_theme()
    st.rerun() # Forces the app to restart and apply the new st.set_page_config

st.sidebar.info(f"Theme: **{current_theme.capitalize()}**")
st.sidebar.info(f"Gemini LLM: {'✅ Enabled' if GEMINI_ENABLED else '❌ Disabled (Lexicon Fallback)'}")
st.sidebar.info(f"Slack Alerts: {'✅ Enabled' if SLACK_WEBHOOK_URL else '❌ Disabled (No Webhook Key)'}")

st.sidebar.markdown("---")
st.sidebar.header("Company Input")
company_name_input = st.sidebar.text_input("Enter Company Name or Ticker:", "NVIDIA Corporation")

if not company_name_input:
    st.info("Please enter a company name or stock ticker in the sidebar to begin the analysis.")
    st.stop()

# --- Main App Logic ---
st.title("💡 Strategic Intelligence System")
st.caption("Real-Time Forecasting, Industry Insights, and Risk Assessment.")

# 1. Get Ticker and Company Name
with st.spinner(f"Searching for ticker for '{company_name_input}'..."):
    TICKER, COMPANY_NAME = search_ticker_by_company(company_name_input)

if not TICKER:
    st.error(f"Could not find a valid stock ticker for '{company_name_input}'. Please try a different name or use the exact ticker (e.g., AAPL).")
    st.stop()
else:
    st.sidebar.success(f"Selected: **{COMPANY_NAME} ({TICKER})**")

    # --- Company Profile Fetching ---
    with st.spinner("Fetching company profile and metadata..."):
        wiki_summary = get_wikipedia_summary(COMPANY_NAME)
        yf_summary = get_yfinance_company_summary(TICKER)
        
    # --- Company Profile Display ---
    st.header(f"🏛️ Company Profile: {COMPANY_NAME}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Key Metadata")
        st.markdown(f"**Ticker:** `{TICKER}`")
        st.markdown(f"**Sector:** {yf_summary.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {yf_summary.get('industry', 'N/A')}")
        st.markdown(f"**Country:** {yf_summary.get('country', 'N/A')}")
        website = yf_summary.get('website')
        if website:
             st.markdown(f"**Website:** [Link]({website})")
        else:
             st.markdown("**Website:** N/A")

    with col2:
        st.subheader("Business Overview")
        business_summary = yf_summary.get("businessSummary", wiki_summary) 
        st.markdown(business_summary)

    st.markdown("---")
    st.header("💡 Strategic Intelligence Dashboard")
    
    # 2. Fetch Data
    with st.spinner(f"Fetching market data, news, and core metrics for {COMPANY_NAME}..."):
        market_df = get_market_data_robust(TICKER, ALPHA_VANTAGE_API_KEY)
        market_metrics = fetch_market_metrics(TICKER) 
        
        news_df = fetch_google_news(f"{COMPANY_NAME} stock")
        tweets_df = generate_mock_social_sentiment(COMPANY_NAME, posts_per_day=30)
        corpus_df = pd.concat([news_df, tweets_df], ignore_index=True)
        
    if market_df.empty:
        st.error(f"No historical price data available for {TICKER}. Cannot generate dashboard.")
        st.stop()
        
    # 3. Run Analysis
    with st.spinner("Running Advanced Sentiment Analysis and Time Series Forecasting..."):
        
        corpus_df_with_sentiment = compute_sentiment_for_corpus(corpus_df.copy(), GEMINI_ENABLED)
        aggregate_sentiment_score = corpus_df_with_sentiment["sentiment"].mean()
        
        forecast_df, model_name = run_forecasting(market_df.copy())
        
        trading_signal, price_change_7_day, forecasted_price_7_day = calculate_signal(
            forecast_df, market_df, aggregate_sentiment_score
        )
        
        last_price = market_metrics.get("current_price", 0.0)

    # 4. Generate Dashboard
    if not forecast_df.empty:
        # FIX APPLIED: Rename 'ds' to 'date' here, ensuring the column exists before the function call
        forecast_df_renamed = forecast_df.rename(columns={"ds": "date"})

        dashboard_fig = create_advanced_dashboard_figure(
            market_df, forecast_df_renamed, corpus_df_with_sentiment, 
            market_metrics, COMPANY_NAME, TICKER, last_price, forecasted_price_7_day, 
            price_change_7_day, aggregate_sentiment_score, trading_signal, model_name
        )
        
        st.plotly_chart(dashboard_fig, use_container_width=True) 
        
    else:
        st.warning("Forecasting failed. Displaying core metrics only.")
        # Fallback KPI Display
        st.metric(label="Current Price", value=f"${last_price:.2f}")
        st.metric(label="Aggregate Sentiment Score", value=f"{aggregate_sentiment_score:.2f} ({sentiment_label(aggregate_sentiment_score)})")
        st.metric(label="Final Trading Signal", value=trading_signal)
        
    # 5. Send Detailed Slack Alert
    slack_message = f"""
    *🚨 STRATEGIC INTELLIGENCE ALERT - {COMPANY_NAME} ({TICKER}) 🚨*
    
    >>>*FINAL TRADING SIGNAL: {trading_signal}*
    
    *--- Executive Summary ---*
    • Current Price: *${last_price:.2f}*
    • Forecasted Price (7-Day Avg): *${forecasted_price_7_day:.2f}*
    • Price Change (7-Day): *{price_change_7_day:.2f}%*
    • Aggregate Sentiment: *{aggregate_sentiment_score:.2f}* ({sentiment_label(aggregate_sentiment_score)})
    • Market Cap: *{format_market_cap(market_metrics.get('market_cap'))}*
    • Forecasting Model: `{model_name}`
    
    *--- Strategy Insight ---*
    The system recommends a *{trading_signal}* position driven by a *{sentiment_label(aggregate_sentiment_score)}* sentiment and a *{trend_label(price_change_7_day)}* price trend.

    *View the full dashboard for detailed metrics and risk profile.*
    """
    
    # st.experimental_rerun() prevents us from reaching this step if the theme button was pressed.
    if st.sidebar.checkbox("Send Slack Notification (Check to Send)"):
        send_slack_notification(slack_message, SLACK_WEBHOOK_URL)
    
    st.markdown("---")
    st.subheader("Source Data & Forecasting Details")
    col_news, col_forecast_data = st.columns(2)
    with col_news:
        st.caption("Top News & Social Sentiment Scores (Max 10)")
        st.dataframe(
            corpus_df_with_sentiment.sort_values(by="published_at", ascending=False)[["source", "title", "sentiment"]].head(10),
            use_container_width=True, hide_index=True
        )
    with col_forecast_data:
        st.caption(f"7-Day Forecast Data ({model_name})")
        # Ensure that if forecast_df was empty, we don't try to use it here.
        if 'forecast_df_renamed' in locals() and not forecast_df_renamed.empty:
            st.dataframe(
                forecast_df_renamed.rename(columns={'date': 'Date', 'yhat': 'Prediction', 'yhat_lower': 'Lower CI', 'yhat_upper': 'Upper CI'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Forecast data is unavailable.")
        
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Infosys Springboard Internship © 2025**")
# Added Footer line for the sidebar
st.sidebar.markdown("---") 
st.sidebar.markdown("Real Time Industry Intelligence System")