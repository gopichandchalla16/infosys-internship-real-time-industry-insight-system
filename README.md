Real-Time Industry Insight & Strategic Intelligence System

Project Overview

This project, developed as part of the Infosys Springboard Internship, is a Real-Time Industry Insight and Strategic Intelligence System. Its core objective is to dynamically gather, analyze, and visualize multi-source data streams to provide a comprehensive, real-time snapshot of any specified company or market asset.

The system uses Python-based data scraping, cleaning (via pandas), and fundamental analysis techniques to extract actionable insights, such as market sentiment and price trends.

print 1: Data Sourcing & Handling (Completed)

The first sprint established the foundational data pipeline and analysis capabilities within a flexible, interactive Jupyter Notebook (Project_Sprint1.ipynb).

Key Features & Data Sources

Feature

Description

Libraries Used

Dynamic Input

Allows the user to enter any company or asset name (e.g., "Tesla," "Bitcoin") to dynamically source data.

input()

Industry Summary

Collects a concise, real-time summary of the company/asset.

wikipedia

Market Data

Fetches the latest stock/crypto market prices and historical data.

yfinance (or similar financial API)

News Headlines

Gathers the most recent news headlines relevant to the asset.

feedparser (for Google News RSS)

Sentiment Analysis

Applies basic Natural Language Processing (NLP) to news headlines and synthetic social media data (50 generated tweets) to determine market mood.

TextBlob

Data Structure

Cleans and organizes all raw data into structured, easy-to-use pandas DataFrames.

pandas

Visualizations

The Sprint 1 notebook successfully generated key visualizations to communicate initial findings:

Sentiment Distribution: A bar chart showing the breakdown of positive, negative, and neutral sentiments.

Sentiment Trend: A line chart visualizing how sentiment polarity changes over a time series of posts/headlines.

Market Price Trend: A line chart tracking the asset's closing price over time.

Future Scope (Sprint 2 Onward)

Sprint 2 will build upon this foundation by integrating more sophisticated features:

AI-Driven Analytics: Implementing advanced machine learning models (e.g., LSTM, Prophet) for price forecasting.

Real-Time Dashboard: Creating an interactive web dashboard (using tools like Streamlit or Dash) to display all insights and charts live.

Deeper NLP: Utilizing state-of-the-art LLMs or pre-trained models (e.g., BERT) for more nuanced sentiment and entity recognition.

Setup and Requirements

To run the analysis notebook, you will need the following Python packages:

pip install pandas numpy matplotlib wikipedia yfinance feedparser textblob


How to Run the Notebook

Clone the Repository:

git clone [https://github.com/gopichandchalla16/infosys-internship-real-time-industry-insight-system.git](https://github.com/gopichandchalla16/infosys-internship-real-time-industry-insight-system.git)
cd infosys-internship-real-time-industry-insight-system


Open the Notebook:
Launch Jupyter Lab or VS Code and open Project_Sprint1.ipynb.

Execute Cells:
Run the cells sequentially. When prompted, enter the name of the company or asset you wish to analyze (e.g., Netflix, TCS, Bitcoin).

Team Members:

Gopichand

Anshika

Janmejay
