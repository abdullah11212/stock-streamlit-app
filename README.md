# Stock Price Prediction using GRU & Heston Volatility

This project is a Streamlit-based web application for stock price forecasting.
It combines statistical analysis, jump detection, the Heston stochastic volatility model,
and a GRU (Gated Recurrent Unit) deep learning model.

## Features
- Historical stock data from Yahoo Finance
- Statistical analysis & normality tests
- Jump detection using the 3σ rule
- Heston stochastic volatility modeling
- GRU deep learning model
- Future price forecasting
- Interactive Streamlit web interface

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
