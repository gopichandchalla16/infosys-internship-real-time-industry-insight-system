# 📊 Financial Sentiment Analysis & Market Intelligence Pipeline

**Infosys Springboard Internship — Sprint 1 to Sprint 3**

A robust, end-to-end data science and AI project that analyzes **financial market sentiment**, **price trends**, and **future forecasts** for stocks and cryptocurrencies using real-world data sources and finance-specific NLP models.

---

## 📌 Project Overview

This project demonstrates a complete **financial analytics pipeline** combining:

* Market price data
* News-based sentiment analysis
* Trend detection using technical indicators
* Time-series forecasting
* Automated alerting

The notebook is designed to be **fault-tolerant**, **reproducible**, and **extensible**, making it suitable for real-world analytical workflows.

---

## 🚀 Key Features

### 🔹 Controlled User Input

* Users can only select companies/assets from a **predefined whitelist**
* Prevents invalid inputs and ensures data consistency

### 🔹 Company Context (Wikipedia)

* Automatically fetches a clean company overview
* Handles disambiguation, missing pages, and network errors gracefully

### 🔹 Market Data Collection

* **Yahoo Finance (yfinance)** for short-term price trends
* **Alpha Vantage API** for daily OHLC data
* Supports both **stocks and cryptocurrencies**

### 🔹 News Aggregation

* Latest headlines fetched using **Google News RSS**
* Duplicate removal and text cleaning applied

### 🔹 Sentiment Analysis

* **FinBERT (Finance-specific Transformer Model)** for accurate sentiment detection
* Confidence-weighted sentiment aggregation
* Safe fallback mechanisms if models fail

### 🔹 Visual Analytics

* Sentiment distribution (bar charts)
* Sentiment polarity trend (line plots)
* Closing price trend analysis
* Candlestick charts using Plotly

### 🔹 Trend Detection (Technical Analysis)

* 20-day and 50-day Simple Moving Average (SMA)
* Market classification:

  * ✅ Bullish
  * ❌ Bearish
  * ↔ Sideways

### 🔹 Forecasting

* **Facebook Prophet** for time-series forecasting
* 7-day price forecast with confidence intervals
* Cross-validation (when sufficient data is available)

### 🔹 Automated Slack Alerts

* Sends alerts based on dominant sentiment signals
* Structured Slack messages with actionable insights

---

## 🧠 Technology Stack

| Category             | Tools / Libraries                   |
| -------------------- | ----------------------------------- |
| Programming Language | Python                              |
| Data Handling        | Pandas, NumPy                       |
| Market Data          | yfinance, Alpha Vantage             |
| NLP & ML             | FinBERT, TextBlob, Prophet          |
| Visualization        | Matplotlib, Plotly                  |
| Web & APIs           | Requests, Feedparser, Wikipedia API |
| Automation           | Slack Webhooks                      |
| Environment          | Google Colab                        |

---

## 📂 Project Structure

```text
Project_Sprint1.ipynb
│
├── User Input & Validation
├── Wikipedia Company Overview
├── Market Data Fetch (Yahoo Finance & Alpha Vantage)
├── News Collection (Google News RSS)
├── Sentiment Analysis (FinBERT)
├── Visualization & Trend Analysis
├── Slack Alerts
└── Prophet Forecasting (Sprint 3)
```

---

## ⚙️ Setup & Requirements

### 1️⃣ Environment

* Google Colab (recommended)
* Python 3.9+

### 2️⃣ Required API Keys

Set the following as **environment variables** or **Colab secrets**:

```text
ALPHA_VANTAGE_API   = Your Alpha Vantage API Key
SLACK_WEBHOOK_URL  = (Optional) Slack Webhook URL
```

### 3️⃣ Install Dependencies

All dependencies are installed automatically inside the notebook using safe installation checks.

---

## ▶️ How to Run

1. Open the notebook in **Google Colab**
2. Configure required API keys
3. Run all cells sequentially (`Runtime → Run all`)
4. Select a company from the allowed list
5. View insights, visualizations, and alerts

---

## 📈 Sample Outputs

* Company overview summary
* Market trend (Bullish / Bearish / Sideways)
* Sentiment distribution & polarity charts
* Closing price trends & candlestick charts
* 7-day forecast table
* Slack alert notifications

---

## ⚠️ Notes & Limitations

* FinBERT model downloads can be **slow on CPU** — GPU runtime is recommended
* Forecast accuracy depends on market volatility and historical data quality
* News headlines may not always reflect full article sentiment

---


## 🎯 Internship Context

This project was developed as part of the **Infosys Springboard Internship** and represents:

* Sprint 1: Data collection & sentiment analysis
* Sprint 2: Visualization & alerting
* Sprint 3: Forecasting & predictive insights

---

## 📌 Future Enhancements

* Real-time dashboard (Streamlit / Power BI)
* Model evaluation & benchmarking
* Multilingual sentiment analysis
* Portfolio-level sentiment aggregation

---

## ⭐ Acknowledgements

* Infosys Springboard
* Hugging Face (FinBERT)
* Yahoo Finance
* Alpha Vantage
