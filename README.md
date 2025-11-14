# 🌐 Real-Time Industry Insight & Strategic Intelligence System

### 📊 AI-Powered Market Intelligence • LLM Summaries • Financial Sentiment • Forecasting • Alerts

---

## 🚀 Overview

The **Real-Time Industry Insight & Strategic Intelligence System** is an AI-driven analytics platform designed to continuously ingest real-world financial + news + social media data and transform it into strategic insights. It integrates **LLMs (Gemini API)**, **Finance-tuned Transformers (FinBERT)**, predictive modeling, and real-time alerting to assist decision-makers with actionable intelligence.

This system is built as part of the Infosys project for real-time industry intelligence.

---

## ✨ Key Features

* **📥 Real-Time Data Ingestion** (market data, business news, tweets)
* **🧹 Data Cleaning & Preprocessing** — pipelines for consistent and reproducible analytics
* **🧠 Dual Sentiment Engine:**

  * **Gemini API (LLM-based)** → contextual sentiment + business summaries
  * **FinBERT (Hugging Face: ProsusAI/finbert)** → finance-specific sentiment classification
* **📈 Predictive Modeling** using ARIMA / Prophet / LSTM (extensible)
* **📊 Interactive Visual Dashboard** using Plotly
* **🔔 Slack Alerts** triggered by sentiment shifts or market anomalies
* **♻️ Modular Architecture** designed for scaling to multiple companies or industries

---

## 🧠 Sentiment Analysis Engine

This platform uses **two complementary models** to achieve robust sentiment evaluation:

### **1️⃣ FinBERT — ProsusAI/finbert (Hugging Face)**

FinBERT is a transformer model trained specifically on **financial text**, providing domain-accurate polarity predictions.

**Advantages:**

* Optimized for financial reports, market news, earnings calls
* Much higher precision in finance context than generic LLMs
* Fast inference; can run locally or on cloud

**Use-case Examples:**

* Market-moving news classification
* Earnings-call transcript evaluation

### **2️⃣ Gemini API — LLM-based Sentiment + Summaries**

Gemini is used to:

* Generate **structured sentiment outputs**
* Provide **human-like summaries** of news & trends
* Extract **themes, risk signals, opportunities**
* Reduce noise and add interpretability

**Advantages:**

* Handles long-text + reasoning
* Captures nuance missed by classifiers
* Provides contextual insights and narratives


---

## 🏗️ System Architecture

```
        ┌────────────┐
        │  Sources   │
        │  News API  │
        │  Twitter   │
        │ Price Data │
        └──────┬─────┘
               │
        ┌──────▼──────┐
        │  Ingestion  │
        │  Pipeline   │
        └──────┬──────┘
               │
    ┌──────────▼──────────┐
    │ Preprocessing Layer │
    └──────────┬──────────┘
               │
       ┌───────▼────────────┐
       │ Dual Sentiment AI  │
       │ FinBERT + Gemini   │
       └───────┬────────────┘
               │
   ┌───────────▼─────────────┐
   │   Forecasting Engine    │
   └───────────┬─────────────┘
               │
   ┌───────────▼──────────────┐
   │   Dashboard & Visuals    │
   └───────────┬──────────────┘
               │
        ┌──────▼─────────┐
        │ Alerts (Slack) │
        └────────────────┘
```


---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

```
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Setup Environment Variables

Create a `.env` file:

```
GEMINI_API_KEY=your_key_here
SLACK_WEBHOOK_URL=your_webhook
```

---

## 🔮 Forecasting Engine

Planned & supported models:

* **ARIMA** — statistical baseline
* **LSTM** — non-linear time-series modeling

---

## 📈 Dashboard & Visualization

The system supports interactive charts:

* Price trends
* Sentiment over time (FinBERT + Gemini)
* Volume & volatility
* Theme extraction

Plotly-based dashboard (can be migrated to Streamlit).

---

## 🔔 Real-Time Alerts

Alerts trigger when:

* Sentiment divergence exceeds threshold
* Price deviates from forecast
* Market-moving news is detected

Delivered via Slack Webhooks.

---

## 🤝 Contributors

* **Anshika Gupta**
* **Gopichand**
* **Janmejay**
* **Vaishnavi**


