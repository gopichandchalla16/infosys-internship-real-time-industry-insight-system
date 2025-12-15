# 📊 Market & News Sentiment Intelligence System (Infosys Internship)

## 📌 Project Overview

This project is developed as part of the **Infosys Springboard Internship**. The notebook builds a **Market & News Sentiment Intelligence System** that fetches real-time stock market data and financial news, performs sentiment analysis using **FinBERT with a safe fallback mechanism**, and visualizes the relationship between **market price trends and public sentiment**.

This project demonstrates the complete **data pipeline from data acquisition → NLP-based sentiment analysis → visualization**.

---

## 🎯 Objectives

* Fetch live stock market data using Yahoo Finance
* Fetch financial news related to selected companies
* Perform safe and reliable sentiment analysis using **FinBERT**
* Provide fallback sentiment analysis using **TextBlob**
* Compare **sentiment polarity vs stock price trends**
* Display results using powerful visualizations

---

## 🚀 Features

* ✅ Real-time stock data fetching
* ✅ Automated financial news extraction
* ✅ FinBERT-based financial sentiment analysis
* ✅ Safe fallback sentiment using TextBlob
* ✅ Sentiment aggregation across multiple news articles
* ✅ Market vs sentiment trend visualization
* ✅ Fake data fallback when real news is unavailable

---

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries Used:**
  - requests, BeautifulSoup – Web scraping
  - pandas – Data processing
  - matplotlib, seaborn – Data visualization
  - yfinance – Stock market data
  - wikipedia – Company information
  - transformers – FinBERT sentiment model
  - textblob – Fallback sentiment analysis
  - faker – Dummy text generation
  - prophet – Time-series stock price forecasting

---

## ⚙️ How It Works
1. User selects a company
2. System fetches:
   - Live stock prices
   - Financial news headlines
3. News is passed through **FinBERT** for sentiment analysis
4. If FinBERT fails, **TextBlob** is used as backup
5. Sentiment polarity and confidence scores are calculated
6. **Stock prices are forecasted using Facebook Prophet**
7. Stock price vs sentiment trends and predictions are plotted


---

## 📊 Output

* Sentiment polarity scores
* Confidence score visualization
* Market price vs sentiment trend graphs

---

## ▶️ How To Run

1. Open this notebook in **Google Colab**
2. Enable **GPU runtime** (recommended for FinBERT)
3. Run all cells sequentially
4. Select a valid company name from the allowed list

---

## 🔐 Security Practices

* No hardcoded API keys
* Safe exception handling during NLP inference
* Automatic fallback when model loading fails

---

## 🚧 Limitations

* FinBERT requires high memory
* CPU inference is slow
* News scraping depends on website availability

---

## 🔮 Future Scope

* Real-time dashboard using Streamlit
* Integration with live trading APIs
* Multi-language financial sentiment analysis
* Deep learning-based price prediction

---

## 👩‍💻 Team Members

* Anshika Gupta
* Gopichand
* Janmejay Singh
* Vaishnavi

---

✅ *This project is part of Infosys Springboard Internship Program.*
