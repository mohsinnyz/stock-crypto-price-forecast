
```
# 📈 Stock & Crypto Price Forecast (LSTM)

A deep learning project that predicts stock and cryptocurrency prices using an LSTM model.  
The model is trained on historical data from Yahoo Finance and deployed via Hugging Face Hub.

---

## 🚀 Features
- Predicts stock & crypto prices (e.g., BTC-USD, ETH-USD, AAPL, TSLA)
- Uses an LSTM model trained on T4 GPU
- Fetches live data from Yahoo Finance
- Frontend built with Streamlit
- Model hosted on Hugging Face Hub

---

## 📂 Project Structure
```

stock-crypto-price-forecast/
│── app.py            # Streamlit frontend
│── train.py          # LSTM training + Hugging Face upload
│── requirements.txt  # Dependencies
│── README.md         # Project documentation
│── model/            # Saved models (auto-uploaded to HF)

````

---

## ⚙️ Installation
```bash
git clone https://github.com/yourusername/stock-crypto-price-forecast.git
cd stock-crypto-price-forecast
pip install -r requirements.txt
````

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

*(or)*

```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🌐 Model on Hugging Face

🔗 [Stock & Crypto Price Forecast Model](https://huggingface.co/mohsinnyz/stock-crypto-price-forecast)

---

## 📊 Example Tickers

* `BTC-USD`
* `ETH-USD`
* `AAPL`
* `TSLA`
* `MSFT`


## 👨‍💻 Author

**Mohsin**
AI/ML Engineer • [https://linkedin.com/in/mohsinnyz](#) • [https://github.com/mohsinnyz](#)

```
