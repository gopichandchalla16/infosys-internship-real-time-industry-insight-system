Real-Time Industry Insight & Strategic Intelligence System

🚀 Project Overview

This project implements a Strategic Intelligence System designed to collect, process, analyze, and visualize multi-source data streams for any specified company or market asset in real-time.

Developed during the Infosys Springboard Internship, the system provides a comprehensive, rapid snapshot of market mood (Sentiment Analysis) and fundamental price trends, laying the groundwork for advanced financial forecasting.

🎯 Sprint 1 Accomplishments: Data Foundation

The initial phase focused on building a robust, interactive data pipeline within the Project_Sprint1.ipynb notebook.

Data Sources & Analysis Features
                                                                                                                      
 1.Dynamic Sourcing -        Allows the user to enter any asset name (e.g., Infosys, TCS, Netflix) for analysis.                                      Python input()

2.Company Summary  -                Fetches a concise, structured summary of the asset or company.                                                    wikipedia

3.Market Data      -           Collects historical and latest closing prices for trend analysis.                                                    yfinance / Financial API

4.News & Headlines  -                Gathers recent news items to capture current market events.                                                    feedparser (Google News RSS)

5.Sentiment Analysis -  Processes news and synthetic social media data (50 generated tweets) to determine market mood (Positive/Negative/Neutral).   TextBlob

6.Data Structuring    -         All raw data is cleaned, labeled, and converted into analytical pandas DataFrames.                                      pandas  

Key Visualizations Produced:

Sentiment Distribution: Bar chart showing the overall mood based on the analyzed text data.

Sentiment Trend: Line chart visualizing polarity changes over the dataset time series.

Market Price Trend: Line chart tracking the asset's closing price over time.

💻 Setup and Installation

Requirements

pandas

numpy

matplotlib

wikipedia

yfinance

feedparser

textblob

jupyter

Alpha Vantage API Key

How to Run

Launch Jupyter:

Start your Jupyter server and open the main analysis file: Project_Sprint1.ipynb.

Execute:  Run all cells sequentially. When prompted, enter the company name or symbol you wish to analyze (e.g., Netflix).

➡️ Next Steps (Future Development)

The system is now ready for expansion in future sprints:

Real-Time Dashboard: Develop an interactive front-end (e.g., using Streamlit, Dash, or Angular) to display all charts and data live.

AI-Driven Forecasting: Implement Time Series forecasting models (like LSTM or Prophet) to predict future price movements.

Advanced NLP: Utilize specialized models for more granular, industry-specific text analysis.

Team Members:

Gopichand

Anshika

Janmejay
