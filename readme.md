# Stock Prediction & Buy/Sell Signal Engine — AAPL

An end-to-end machine learning project that started as a simple LSTM price predictor and evolved into a full buy/sell signal system with backtesting.

> **Personal note:** This project started because I wanted to understand if ML could tell me when to buy and sell stocks. Turns out it's a lot harder than it sounds — but I learned a ton along the way.

---

## Project Status

| Phase | Description | Status |
|---|---|---|
| Phase 1 | LSTM price prediction (AAPL) | ✅ Done |
| Phase 2 | Buy/sell signal engine + backtesting | ✅ Done |
| Phase 3 | Data update 2025–2026 + risk management | 🔜 Planned |
| Phase 4 | Web app + deployment | 🔜 Planned |

---

## Phase 1 — LSTM Stock Price Prediction

### What it does
Predicts the next day closing price of AAPL using a stacked LSTM model trained on historical data (2018–2024).

### Questions answered
1. What was the change in price of the stock over time?
2. What was the daily return of the stock on average?
3. What was the moving average of various stocks?
4. What was the correlation between different stocks?
5. How much value do we put at risk by investing in a particular stock?
6. Can we predict future stock behavior using LSTM?
7. What will the closing price be for the next 2 days?

### Notebooks
| Notebook | Description |
|---|---|
| `phase1a_upgrade_revised.ipynb` | Feature engineering (RSI, MACD, Bollinger Bands) + LSTM model |
| `phase1b_tuning.ipynb` | Hyperparameter tuning — layers, dropout, sequence length |
| `phase1c_multistock.ipynb` | Extend model to multiple stocks |

### Tech stack
`Python` `TensorFlow / Keras` `yfinance` `pandas` `numpy` `scikit-learn` `ta` `matplotlib`

### Key findings
- Adding RSI, MACD, and Bollinger Bands as features significantly improved RMSE compared to closing price alone
- Best model: 2-layer stacked LSTM, sequence length 60, dropout 0.2
- The model predicts price direction reasonably well but struggles with sharp reversals

---

## Phase 2 — Buy/Sell Signal Engine

This is the part I actually wanted to build from day one. Instead of predicting *price*, we predict *action*: should I buy, sell, or hold today?

### How it works

```
Historical data
      ↓
Feature engineering (RSI, MACD, BB, SMA + lag features)
      ↓
Labeling: future return over 5 days → BUY / SELL / HOLD
      ↓
Classification model (XGBoost)
      ↓
Backtesting: simulate trading with $10,000 initial capital
      ↓
Signal visualization on price chart
```

### Notebooks
| Notebook | Description |
|---|---|
| `phase2_step2_labeling.ipynb` | Label BUY/SELL/HOLD using 5-day future return, threshold 2% |
| `phase2_step3_classification.ipynb` | Baseline Random Forest + XGBoost classifier |
| `phase2_step3b_improved.ipynb` | + Lag features, momentum features, hyperparameter tuning |
| `phase2_step4_backtesting.ipynb` | Simulate 4 strategies, equity curve, trade log |
| `phase2_step5_visualization.ipynb` | Signal chart, zoom view, Phase 2 summary |

### Features used (30 total)
**Base (10):** `Close`, `Volume`, `RSI`, `MACD`, `MACD_Signal`, `BB_Width`, `SMA_20`, `SMA_50`, `Volume_Ratio`, `Daily_Return`

**Lag features (15):** RSI, MACD, MACD_Signal, Close, BB_Width — each at 1, 3, and 5 days ago

**Momentum features (5):** `Return_3d`, `Return_5d`, `RSI_Slope`, `MACD_Cross`, `Volume_Momentum`

### Backtesting results (Aug 2023 – Dec 2024, $10,000 initial capital)

| Strategy | Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|
| Buy & Hold (benchmark) | +44.9% | 1.38 | -16.6% | — |
| Signal — all | +7.5% | 0.54 | -11.4% | 8 |
| **Signal — conf ≥ 40%** | **+19.4%** | **1.33** | **-6.5%** | **5** |
| Signal — conf ≥ 50% | 0.0% | 0.00 | 0.0% | 0 |

### Key findings

**The model didn't beat buy & hold — and that's expected.**
AAPL went from ~$178 to ~$250 (+40%) almost non-stop during the test period. Any strategy that occasionally moves to cash will underperform in a strong bull market. The model spent 65% of the test period recommending SELL, sitting on the sidelines while AAPL climbed.

**But the confidence-filtered strategy is interesting.**
Signal (conf ≥ 40%) achieved a Sharpe ratio of 1.33 — nearly identical to buy & hold's 1.38 — but with a max drawdown of only -6.5% vs -16.6%. This means comparable risk-adjusted returns with significantly less volatility. For someone who can't stomach a 16% portfolio drop, this is actually useful.

**Trade log (Signal all — 8 round trips):**
- 5 wins / 3 losses (62.5% win rate)
- Average win: +$371 | Average loss: -$368
- Risk/Reward ratio: 1.01x (needs improvement)

### Honest assessment
The model's classification accuracy is ~29% — below random guess for 3 classes. This sounds bad, but accuracy is the wrong metric for trading signals. What matters is whether the model is right *when it counts* (high volatility, big moves). The Sharpe ratio comparison suggests it has some signal, but the model needs more work before being used with real money.

---

## Project Structure

```
stock-prediction-project/
├── notebooks/
│   ├── phase 1/
│   │   ├── phase1a_upgrade_revised.ipynb
│   │   ├── phase1b_tuning.ipynb
│   │   └── phase1c_multistock.ipynb
│   └── phase 2/
│       ├── phase2_step2_labeling.ipynb
│       ├── phase2_step3_classification.ipynb
│       ├── phase2_step3b_improved.ipynb
│       ├── phase2_step4_backtesting.ipynb
│       └── phase2_step5_visualization.ipynb
├── data/
│   └── processed/
│       └── aapl_labeled.pkl
├── models/
│   ├── xgb_signal_model_v2.pkl
│   ├── signal_scaler_v2.pkl
│   ├── signal_model_meta_v2.json
│   ├── phase2_summary.png
│   └── phase2_step5_main_chart.png
└── README.md
```

---

## Setup

```bash
git clone https://github.com/AngeloDusu/stock-prediction-project.git
cd stock-prediction-project
pip install -r requirements.txt
```

**Required packages:**
```
tensorflow
xgboost
scikit-learn
yfinance
ta
pandas
numpy
matplotlib
joblib
```

Run notebooks in order — Phase 1 notebooks first, then Phase 2 Step 2 → 5.

---

## What I learned

Phase 1 taught me that predicting stock *prices* with LSTM is doable but limited — the model learns the general shape of price movement but struggles with sharp reversals.

Phase 2 was more interesting and more humbling. Building a buy/sell classifier forced me to think about what information actually predicts future price direction, how to properly evaluate a trading strategy (Sharpe ratio matters more than accuracy), and why most retail trading algorithms fail to beat a simple buy and hold strategy — especially in bull markets.

The biggest lesson: a model that performs okay at classification can still be useful as a *risk management tool* even if it doesn't maximize returns.

---

## Limitations

- Only uses technical indicators — no news sentiment, earnings data, or macro factors
- Model trained on data up to 2022, not retrained with 2025–2026 data yet
- No risk management (stop-loss, position sizing) implemented
- Backtesting uses closing prices — doesn't account for slippage or bid-ask spread
- Single stock (AAPL) — may not generalize

---

## Roadmap

**Phase 3 (next):**
- [ ] Update data to include 2025–2026
- [ ] Retrain all models on updated data
- [ ] Add risk management: stop-loss, position sizing
- [ ] Backtest again with updated model + risk management
- [ ] Optional: add Fear & Greed Index as sentiment feature

**Phase 4:**
- [ ] FastAPI backend — input ticker, output signal + prediction
- [ ] Streamlit dashboard — price chart, signals, portfolio performance
- [ ] Deploy to Render / Railway / HuggingFace Spaces

---

## Reference

Phase 1 was inspired by this Kaggle notebook (highly recommend):
[Stock Market Analysis & Prediction using LSTM](https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm/notebook)
