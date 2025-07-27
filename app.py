import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt

# ✅ Hugging Face Repo
HF_REPO = "mohsinnyz/stock-crypto-price-forecast"
MODEL_FILE = "BTC-USD_best_model.pth"

# ✅ Download Model
MODEL_PATH = hf_hub_download(repo_id=HF_REPO, filename=MODEL_FILE)

# ✅ LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ✅ Load Model
device = "cpu"
model = LSTMModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ Streamlit UI
st.title("📈 Stock & Crypto Price Forecast")
ticker = st.text_input("Enter Ticker (e.g. BTC-USD, ETH-USD, AAPL):", "BTC-USD")

if st.button("Predict"):
    data = yf.download(ticker, period="2y")
    close_prices = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    seq_len = 60
    X_input = scaled[-seq_len:]
    X_input = torch.tensor(X_input[np.newaxis, :, :], dtype=torch.float32)

    with torch.no_grad():
        pred_scaled = model(X_input).numpy()

    pred_price = scaler.inverse_transform(pred_scaled)[0][0]
    st.success(f"📌 Predicted Next Price: **{pred_price:.2f} USD**")

    # ✅ Plot
    last_days = close_prices[-60:]
    plt.figure(figsize=(8, 4))
    plt.plot(range(60), last_days, label="Last 60 Days")
    plt.scatter(60, pred_price, color="red", label="Predicted Next Price")
    plt.legend()
    st.pyplot(plt)
