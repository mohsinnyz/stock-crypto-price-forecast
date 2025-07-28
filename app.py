# app.py

import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download
import plotly.graph_objs as go
import warnings

# 🚫 Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# 🔧 App Config
st.set_page_config(page_title="AI Price Forecast", page_icon="📊", layout="wide")

# 🎯 Hugging Face Repo Info
HF_REPO = "mohsinnyz/stock-crypto-price-forecast"
MODEL_FILE = "BTC-USD_best_model.pth"
MODEL_PATH = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)

# 🧠 LSTM Model Definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# 🚀 Load Model
device = "cpu"
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 🎨 Sidebar Controls
with st.sidebar:
    st.title("⚙️ Settings")
    popular_tickers = ["BTC-USD", "ETH-USD", "AAPL", "GOOGL", "TSLA"]
    option = st.selectbox("Choose a Ticker", popular_tickers + ["Custom"])

    if option == "Custom":
        ticker = st.text_input("Enter Custom Ticker:", "BTC-USD")
    else:
        ticker = option

    seq_len = st.slider("Sequence Length (Days)", 30, 100, 60)
    st.caption("🔍 Data from Yahoo Finance")

# 🧠 App Title
st.title("📈 AI-Powered Stock & Crypto Forecast")

# 🔍 Fetch and Predict
data = yf.download(ticker, period="2y")

if data.empty or "Close" not in data:
    st.error("❌ Failed to retrieve data. Try another ticker.")
else:
    close_prices = data["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X_input = scaled[-seq_len:]
    X_input = torch.tensor(X_input[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_input).numpy()
    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    last_price = close_prices[-1][0]
    diff = pred_price - last_price
    percent_change = (diff / last_price) * 100

    daily_returns = data["Close"].pct_change().dropna()
    volatility = float(daily_returns.std()) * 100
    conf_range = 0.02 * pred_price

    # 📊 Layout with Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Forecast", "🗃️ Raw Data", "🧠 Model Info"])

    with tab1:
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📉 Last Price", f"${last_price:.2f}")
        col2.metric("📌 Predicted", f"${pred_price:.2f}", f"{percent_change:+.2f}%")
        col3.metric("⚡ Volatility", f"{volatility:.2f}%")
        col4.metric("🎯 Confidence", f"±${conf_range:.2f}")

        # Chart
        last_days = close_prices[-seq_len:].flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=last_days, mode='lines', name="Last Prices"))
        fig.add_trace(go.Scatter(x=[seq_len], y=[pred_price],
                                 mode='markers+text',
                                 text=["Prediction"],
                                 textposition="top center",
                                 marker=dict(color='red', size=10),
                                 name="Predicted"))
        fig.update_layout(title=f"{ticker} - Forecast", xaxis_title="Days", yaxis_title="Price (USD)")
        st.plotly_chart(fig, use_container_width=True)

        # Commentary
        st.markdown("### 💡 Model Insight")
        if diff > 0:
            st.success("🔼 Model suggests an **upward trend**.")
        elif diff < 0:
            st.warning("🔽 Model suggests a **downward movement**.")
        else:
            st.info("➖ Model predicts **stability** in price.")

    with tab2:
        st.markdown(f"### 🗃️ Recent {ticker} Data")
        st.dataframe(data.tail(100))

    with tab3:
        st.markdown("### 🧠 LSTM Model Architecture")
        st.code(model.__str__(), language="python")
        st.markdown("**Source:** [🤗 Hugging Face](https://huggingface.co/mohsinnyz/stock-crypto-price-forecast)")

# 📎 Footer
st.markdown("---")
st.caption("🔗 Built with Streamlit | Model by @mohsinnyz | Data: Yahoo Finance")
