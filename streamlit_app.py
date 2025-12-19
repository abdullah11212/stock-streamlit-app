# =========================================================
# STREAMLIT APP : GRU + HESTON VOLATILITY STOCK FORECAST
# =========================================================

# --------- INSTALL NOTE ----------
# pip install streamlit yfinance arch tensorflow scikit-learn statsmodels scipy matplotlib

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import random
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ---------------- SEED ----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Stock Prediction (GRU + Heston)", layout="wide")
st.title("📈 Stock Price Prediction using GRU + Heston Volatility")

st.sidebar.header("User Inputs")

ticker = st.sidebar.text_input("Stock Ticker", value="GBCO.CA")
future_days = st.sidebar.number_input(
    "Future Prediction Days",
    min_value=1,
    max_value=365,
    value=30,
    step=1
)

run_btn = st.sidebar.button("Run Prediction")

# =========================================================
# FUNCTIONS
# =========================================================

@st.cache_data(show_spinner=False)
def load_data(ticker):
    data = yf.download(ticker, period="max")
    if "Adj Close" in data.columns:
        data["Price"] = data["Adj Close"]
    else:
        data["Price"] = data["Close"]
    return data.dropna()

def statistical_analysis(data):
    log_returns = np.log(data["Price"] / data["Price"].shift(1)).dropna()

    mean_r = log_returns.mean()
    vol = log_returns.std(ddof=1)
    var = log_returns.var(ddof=1)
    skew = log_returns.skew()
    excess_kurt = log_returns.kurt()
    pearson_kurt = excess_kurt + 3

    ks_D, ks_p = stats.kstest(log_returns, "norm", args=(mean_r, vol))
    sh_stat, sh_p = stats.shapiro(log_returns)
    ad = stats.anderson(log_returns, dist="norm")

    threshold = 3 * vol
    data = data.copy()
    data["Log_Return"] = log_returns
    data.dropna(inplace=True)
    data["Jump"] = ((data["Log_Return"] - mean_r).abs() > threshold).astype(int)

    return {
        "log_returns": log_returns,
        "mean": mean_r,
        "vol": vol,
        "var": var,
        "skew": skew,
        "excess_kurt": excess_kurt,
        "pearson_kurt": pearson_kurt,
        "ks": (ks_D, ks_p),
        "shapiro": (sh_stat, sh_p),
        "anderson": ad,
        "threshold": threshold,
        "jump_count": int(data["Jump"].sum()),
        "jump_percent": 100 * data["Jump"].sum() / len(data),
        "data": data
    }

def heston_loglike(params, returns):
    kappa, theta, sigma_v, rho, v0 = params
    dt = 1 / 252
    vt = v0
    loglike = 0

    for r in returns:
        dv = kappa * (theta - vt) * dt + sigma_v * np.sqrt(max(vt, 1e-8)) * np.random.normal()
        vt = max(vt + dv, 1e-8)
        var_ret = vt * dt
        loglike += -0.5 * (np.log(2 * np.pi * var_ret) + (r ** 2) / var_ret)

    return -loglike

def heston_volatility(data):
    returns = 100 * data["Price"].pct_change().dropna()
    params0 = [1.0, 0.02, 0.2, -0.3, 0.02]

    res = minimize(heston_loglike, params0, args=(returns.values,), method="Nelder-Mead")
    kappa, theta, sigma_v, rho, v0 = res.x

    dt = 1 / 252
    v_path = []
    v = v0

    for _ in range(len(data)):
        dv = kappa * (theta - v) * dt + sigma_v * np.sqrt(max(v, 1e-8)) * np.random.normal()
        v = max(v + dv, 1e-8)
        v_path.append(np.sqrt(v))

    data["Volatility"] = v_path
    return data

def prepare_sequences(data, seq_len=60):
    features = data[["Price", "Volatility"]].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])

    return np.array(X), np.array(y), scaler, scaled

def build_gru(input_shape):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def future_forecast(model, last_seq, scaler, future_days, n_features):
    preds = []
    current_seq = last_seq.copy()

    for _ in range(future_days):
        pred = model.predict(current_seq.reshape(1, *current_seq.shape), verbose=0)[0, 0]
        preds.append(pred)

        new_row = np.zeros((n_features,))
        new_row[0] = pred
        current_seq = np.vstack([current_seq[1:], new_row])

    full = np.zeros((len(preds), n_features))
    full[:, 0] = preds
    return scaler.inverse_transform(full)[:, 0]

# =========================================================
# MAIN EXECUTION
# =========================================================

if run_btn:
    st.subheader("📥 Data Loading")
    data = load_data(ticker)
    st.write(f"Data starts from: **{data.index.min().date()}**")

    st.subheader("📊 Statistical Analysis")
    stats_res = statistical_analysis(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean", f"{stats_res['mean']:.6f}")
    col2.metric("Volatility", f"{stats_res['vol']:.6f}")
    col3.metric("Skewness", f"{stats_res['skew']:.4f}")
    col4.metric("Pearson Kurtosis", f"{stats_res['pearson_kurt']:.4f}")

    st.subheader("⚡ Jump Detection")
    st.write(f"Jump Count: **{stats_res['jump_count']}**")
    st.write(f"Jump Percentage: **{stats_res['jump_percent']:.2f}%**")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(stats_res["data"].index, stats_res["data"]["Price"], label="Price")
    jumps = stats_res["data"]["Jump"] == 1
    ax.scatter(
        stats_res["data"].index[jumps],
        stats_res["data"]["Price"][jumps],
        color="red",
        marker="^",
        label="Jumps"
    )
    ax.legend()
    ax.set_title("Price with Jump Detection")
    st.pyplot(fig)

    st.subheader("📉 Heston Volatility")
    data = heston_volatility(stats_res["data"])

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(data.index, data["Volatility"], color="purple")
    ax2.set_title("Heston Conditional Volatility")
    st.pyplot(fig2)

    st.subheader("🤖 GRU Model Training")
    X, y, scaler, scaled = prepare_sequences(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = build_gru((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=25, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    pred_test = model.predict(X_test)
    test_full = np.zeros((len(pred_test), scaled.shape[1]))
    test_full[:, 0] = pred_test[:, 0]
    pred_prices = scaler.inverse_transform(test_full)[:, 0]

    actual_full = np.zeros((len(y_test), scaled.shape[1]))
    actual_full[:, 0] = y_test
    actual_prices = scaler.inverse_transform(actual_full)[:, 0]

    st.subheader("📈 Performance Metrics")
    st.write({
        "MAE": mean_absolute_error(actual_prices, pred_prices),
        "RMSE": np.sqrt(mean_squared_error(actual_prices, pred_prices)),
        "MAPE (%)": np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100,
        "R²": r2_score(actual_prices, pred_prices)
    })

    st.subheader("🔮 Future Forecast")
    last_seq = X[-1]
    future_preds = future_forecast(
        model, last_seq, scaler, future_days, scaled.shape[1]
    )

    future_index = pd.date_range(
        start=data.index[-1], periods=future_days + 1, freq="B"
    )[1:]

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(data.index, data["Price"], label="Historical Price")
    ax3.plot(future_index, future_preds, label="Future Prediction", linestyle="--")
    ax3.legend()
    ax3.set_title("Future Price Forecast")
    st.pyplot(fig3)
