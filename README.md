# Deriv ML-Optimized Trading Bot

A professional trading bot for Deriv Rise/Fall options on the 5-minute timeframe. This bot combines the **UT Bot Alerts** indicator with an advanced **XGBoost + VIP Similarity** ML filter to achieve high win rates.

## 🚀 Key Features

- **Dashboard:** Professional web UI with real-time balance tracking, trade logs, and performance metrics.
- **Dark Mode:** Responsive UI that looks great on mobile and desktop.
- **10 Optimized Strategies:** Pre-configured strategies for 5 synthetic indices (R_100, R_75, R_50, R_25, R_10).
- **IMBA Trend Filter:** Integrated [IMBA] Algo Trend Line to ensure the bot only buys in uptrends and sells in downtrends.
- **VIP Similarity ML:** Advanced filter that learns from historical winners (VIPs) and losers to block signals that resemble past losses.
- **Backtesting Module:** Run historical simulations for any symbol and duration with automated data caching.
- **Dockerized:** Ready for deployment on platforms like Railway.com or Heroku.

## 📈 The Strategy Research Process

The strategies were developed through a multi-step optimization process:

1.  **Indicator Porting:** The original PineScript *UT Bot Alerts* was ported to Python, maintaining its core trailing stop logic.
2.  **3-Candle Exit Rule:** Specifically designed for Rise/Fall options. A signal is confirmed at candle close, entry happens at the next open, and the trade expires exactly 3 candles later (15 minutes).
3.  **ML Optimization:** An XGBoost Classifier combined with a VIP Similarity Engine analyzes raw UT Bot signals. By studying historical winners (VIPs) and losers against technical indicators (including the IMBA Trend Line), the model learns to recognize market conditions that lead to losses.
4.  **Symbol-Specific Tuning:** Backtests were conducted across 5 major synthetic indices to generate high-accuracy strategy configurations documented in the `Profitable strategy/` folder.

## 🛠 Installation

### Local Setup

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  Open `http://localhost:5000` in your browser.

### Docker (Railway/Cloud)

The project includes a `Dockerfile`. Simply connect your repository to **Railway.com** or run locally with:
```bash
docker build -t deriv-bot .
docker run -p 5000:5000 deriv-bot
```

## ⚙️ Configuration

1.  **API Token:** Get your API token from Deriv Settings -> API Token (Scopes: Read, Trade).
2.  **Symbol & Strategy:** Select the index and one of the 10 optimized strategies.
3.  **Stake:** Set the trade amount as a percentage of your account balance.
4.  **Mode:** Use the toggle to switch between **Demo** (Practice) and **Live** trading.

## 📂 Project Structure

- `app.py`: Flask web server and backtesting backend.
- `trading_bot.py`: Live trading engine and Deriv API integration.
- `ml_filter.py`: Random Forest model for trade signal filtering.
- `strategy_utils.py`: UT Bot logic and backtesting engine.
- `indicators.py`: Technical indicator calculations using the `ta` library.
- `Profitable strategy/`: Detailed performance reports for each symbol.
- `data/`: Cached historical data for backtesting.

---
*Disclaimer: Trading involves risk. Perform thorough backtesting on demo accounts before trading live.*
