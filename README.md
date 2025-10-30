# 🚀 Real-Time Industry Insight & Strategic Intelligence System  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)  
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-orange?logo=jupyter)](https://jupyter.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Alpha Vantage](https://img.shields.io/badge/Data-Alpha%20Vantage-blue)](https://www.alphavantage.co/)  
[![News API](https://img.shields.io/badge/NewsAPI-Live%20Headlines-orange)](https://newsapi.org/)  
[![Infosys](https://img.shields.io/badge/Infosys-Springboard%20Internship-lightgrey)](https://infyspringboard.onwingspan.com/)  

---

## 🧭 Overview

The **Real-Time Strategic Intelligence System** is an **AI-powered data analysis pipeline** designed to collect, process, and visualize **multi-source market and sentiment data** in real time.  

Developed as part of the **Infosys Springboard Internship**, this system delivers:
- 📈 Market trend analysis from financial APIs  
- 📰 Live news and sentiment insights  
- 🧠 Company summaries powered by NLP  

This forms the **data foundation** for advanced trend forecasting and business intelligence.

---

## 🎯 Sprint 1 — *Data Foundation Phase*

The first sprint established a **fully automated interactive notebook**  
(`Project_Sprint1.ipynb`) capable of collecting, cleaning, and visualizing real-time data for any company or asset.

### 🔍 Features Implemented

| # | Module | Description | Library / API |
|---|---------|-------------|----------------|
| 1️⃣ | **Dynamic Input** | User can analyze any company or asset (e.g., *Infosys*, *TCS*, *Netflix*) | `input()` |
| 2️⃣ | **Company Summary** | Fetches short summaries from Wikipedia | `wikipedia` |
| 3️⃣ | **Market Data** | Collects price trends for financial analysis | `yfinance` / `Alpha Vantage` |
| 4️⃣ | **News & Headlines** | Retrieves current events & market context | `feedparser` / `newsapi-python` |
| 5️⃣ | **Sentiment Analysis** | Analyzes tone (Positive/Negative/Neutral) | `TextBlob` |
| 6️⃣ | **Data Structuring** | Converts all data into clean DataFrames | `pandas` |

---

## 📊 Key Visualizations

| Visualization | Description |
|----------------|-------------|
| 🟩 **Sentiment Distribution** | Bar chart showing positive/negative/neutral ratio |
| 📈 **Sentiment Trend** | Line chart tracking sentiment polarity over time |
| 💹 **Market Price Trend** | Closing price chart from financial API |

All charts dynamically update based on the selected company.

---

## ⚙️ Setup & Installation

### 🧩 Requirements
```bash
pip install pandas numpy matplotlib wikipedia yfinance feedparser textblob jupyter alpha_vantage newsapi-python
```
| Service           | Usage                        | Required? |
| ----------------- | ---------------------------- | --------- |
| **Alpha Vantage** | Fetches price & stock data   | ✅ Yes     |
| **News API**      | Retrieves recent global news | ✅ Yes     |
```
┌────────────────────────────────────────────┐
│ User Input → Company / Asset Name          │
├────────────────────────────────────────────┤
│ Wikipedia Summary   → Contextual Overview  │
│ Market Data (API)   → Financial Trendline  │
│ News & Headlines    → Market Context       │
│ Synthetic Tweets    → Social Sentiment     │
├────────────────────────────────────────────┤
│ Text Preprocessing  → Cleaning & Tokenizing│
│ Sentiment Analysis   → NLP via TextBlob    │
│ Data Structuring     → Pandas DataFrames   │
├────────────────────────────────────────────┤
│ Visualization → Matplotlib / Plotly Charts │
└────────────────────────────────────────────┘
```


👩‍💻 Team Members

| Name          | Role                                |
| ------------- | ----------------------------------- |
| **Gopichand** | Data Pipeline & API Integration     |
| **Anshika**   | Sentiment Analysis & Visualization  |
| **Janmejay**  | System Architecture & Documentation |

