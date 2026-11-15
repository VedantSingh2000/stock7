# 🔬 What Was Tried, What Failed, and Lessons Learned (Historical Retrospective)

> **Document Purpose:** An exhaustive, candid technical breakdown of all historical stock prediction experiments developed on the Windows workstation (`D:\stock`) between July 2025 and March 2026. This document exists to ensure that the Antigravity agent on the Mac and human developers do **not** repeat past mistakes, understand why certain approaches failed, and extract the real gems that can make the modern system stronger.

---

## 🧭 Executive Summary of the Evolution

| Phase / Repo | Core Idea | Key Models / Tools | Outcome / Why it Failed | What We Learned |
|---|---|---|---|---|
| **1. `stock-prediction`** | Single model to predict next Open for any ticker | RandomForest / Tree Regressor | ❌ **Failed to generalize:** One stock model cannot predict different volatility regimes. | Different stocks need individual models and localized calibration. |
| **2. `stock5-prediction`** | 5 stocks × 4 targets (Open, Close, High, Low) | 20 `RandomForestRegressor` models | ⚠️ **Limited utility:** No walk-forward testing; fixed MAE buffers; manual open input. | Introduced Open-relative % targets and Min-Margin ranking. |
| **3. `stock6`** | Multi-model tournament & GridSearchCV | XGBoost, LightGBM, ExtraTrees, RF | ⚠️ **Overfitting:** Intraday High/Low targets suffered severe noise; Open was most predictable. | Open price is predictable; intraday extreme ranges require volatility features (ATR). |
| **4. `stock7`** | Automated pipeline & 60-day profit simulation | 20 tuned models + individual scalers | ❌ **Fatal Data Leakage & Unit Bug:** ~90% directional accuracy was an illusion caused by feature lookahead; error units subtracted as ₹ instead of %. | Never trust backtest accuracy > 60% without leakage checks; enforce strict % unit math. |
| **5. `dixon-experiments`** | Hourly predictions + Gemini News Sentiment | Gemini 1.5 Flash, GNews, CatBoost, Hourly pivots | ❌ **Timestamp Latency & Noise:** News published hours after price moves; hourly noise swallowed by spreads. | Raw PR news is lagging; sentiment works best as an asynchronous bias filter, not an intraday trigger. |
| **6. `infosys90d`** | 90-day hourly tournament | CatBoost, LightGBM, XGBoost, pandas-ta | ⚠️ **Spread / Friction erosion:** Micro-edges vanished once brokerage and slippage were factored in. | Daily candle boundaries with multi-timeframe confirmation beat micro hourly predictions. |
| **7. `stock_anti`** | Dual-Phase ("Open-based" vs "Close-based") + Macro | Dual RF models, Nifty 50 (`^NSEI`), S&P 500 (`^GSPC`), RSI, ATR | ✅ **Best architectural idea:** Separated overnight gap prediction from post-open range expansion. | Splitting prediction into Pre-Market (overnight) and Post-Open (actual open known) is essential. |

---

## 🔍 Detailed Post-Mortem of Every Experiment

### 1. `stock-prediction` (First Prototype, July 2025)
* **What was tried:**
  * Trained a single ensemble regressor on Tata Motors daily OHLC data.
  * Attempted to use the exact same scaler and weights to predict next-day open prices for *any* arbitrary ticker input in a Streamlit text box.
  * Features used: `SMA_10`, `SMA_30`, `Price_Change`, `High_Low_Diff`, `Volume_Change`, `Close_Shift_1`.
* **Why it was unsuccessful:**
  * **Volatility Mismatch:** A model trained on a mid-beta auto stock (Tata Motors) completely failed on defensive stocks (Colgate) or high-beta IT stocks.
  * **Brittle Data Pipeline:** Didn't handle MultiIndex columns returned by modern `yfinance` versions, crashing on fresh data.
  * **No Risk Envelope:** Predicted a single price point with zero uncertainty boundary or stop-loss concept.

---

### 2. `stock5-prediction` (5 Stocks × 4 Targets)
* **What was tried:**
  * Isolated 5 liquid NSE stocks: `TATAMOTORS.NS`, `UNITDSPR.NS`, `PETRONET.NS`, `COLPAL.NS`, `BEL.NS`.
  * Trained 4 separate `RandomForestRegressor` models per stock (Total: 20 models) predicting `Target_Open`, `Target_Close`, `Target_High`, `Target_Low`.
  * Normalized targets as percentage changes relative to today's Open price:
    $$\text{Target\_}X = \frac{X_{t+1} - \text{Open}_t}{\text{Open}_t} \times 100$$
  * Introduced the **Safety Buffer Formula**:
    $$\text{Target} = \text{PredHigh} - \text{MAE}_{\text{high}}$$
    $$\text{StopLoss} = \text{PredLow} + \text{MAE}_{\text{low}}$$
    $$\text{Min Margin} = \text{Target} - \text{StopLoss}$$
* **What worked:**
  * Open-relative percentage normalization made predictions scale-independent (e.g. BEL at ₹200 vs Tata Motors at ₹900).
  * The Min-Margin concept prioritized margin of safety rather than chasing maximum upside.
* **Why it fell short:**
  * MAE was calculated once over the entire training set and kept static. During high-volatility regimes, the fixed buffer was easily broken.
  * Depended on manual user entry for "Today's Open" in the UI.

---

### 3. `stock6` (Ensemble Shootout & Tuning)
* **What was tried:**
  * Evaluated `LinearRegression`, `RandomForestRegressor`, `ExtraTreesRegressor`, `XGBRegressor`, and `LGBMRegressor` via `GridSearchCV`.
  * Systematic holdout evaluation per target across all stocks.
* **What we discovered:**
  * `Target_Open` consistently yielded the highest $R^2$ scores across all models.
  * `Target_High` and `Target_Low` suffered significantly higher variance; tree models tended to overfit to extreme historical spikes unless heavily constrained by `max_depth` (3 to 5).
  * `ExtraTrees` and `RandomForest` produced smoother predictions than gradient-boosted trees, which were prone to sharp boundary artifacts on small tabular samples.

---

### 4. `stock7` (The 60-Day Profit Simulator & Fatal Flaws)
* **What was tried:**
  * Full automated training pipeline generating `multi_stock_models.pkl`, `multi_stock_scalers.pkl`, `multi_stock_errors.pkl`, and `multi_stock_backtest.pkl`.
  * Added a 60-day interactive simulation page showing color-coded daily profit/loss tables and monthly aggregation.
* **The Fatal Flaws (Critical Learnings):**
  1. 🚨 **Data Leakage in Backtest:**
     - The backtest reported directional accuracy of ~90% across all 5 stocks.
     - In real-world next-day equity forecasting, genuine directional accuracy is rarely above 53–55%.
     - **Cause:** Lookahead leakage during feature generation or target shifting (e.g., using information from day $t+1$ or using $Close_t$ in formulas that assumed open-of-day execution). The model was essentially predicting the past.
  2. 🚨 **The Unit Conversion Bug:**
     - Model errors (MAE) were computed in **percentage terms** (e.g., $0.65\%$).
     - In `stock7`'s `app.py`, the code subtracted the raw float value directly from the rupee price:
       ```python
       # BUGGY CODE in stock7:
       target = pred_high - mae_high  # e.g., ₹900 - 0.65 = ₹899.35 (instead of 0.65% = ₹5.85!)
       ```
     - As a result, the "safety buffer" was basically ₹0.50 instead of an actual percentage cushion, leaving paper trades completely unbuffered.

---

### 5. `dixon-experiments` (Hourly Predictions + Gemini News Sentiment)
* **What was tried:**
  * 500-day and 700-day hourly datasets on `DIXON.NS` paired with Nifty 50 and Nifty IT indices.
  * Fetched news articles via GNews and NewsAPI on specific queries (`"Dixon manufacturing deal"`, `"Dixon quarterly results"`, `"Electronics sector India"`).
  * Passed news headlines and snippets to **Gemini 1.5 Flash** with custom prompts to rate sentiment on a structured scale.
  * Merged sentiment scores as lag features into hourly CatBoost and XGBoost models.
* **Why it was unsuccessful:**
  1. **Timestamp Lookahead & Reporting Lag:**
     - Mainstream news articles are almost always published *after* institutional money has already reacted in the market.
     - Article timestamps were often in UTC or reflected syndication repost times, causing the model to use sentiment from news that was published hours after market close to "predict" daytime movement.
  2. **Low Signal-to-Noise in PR:**
     - General company PR announcements showed nearly zero correlation with immediate intraday price volatility.
  3. **High Frequency Execution Realities:**
     - Hourly predictions required frequent trading. Transaction costs (STT, exchange turnover fees, GST, broker brokerage) rapidly consumed any statistical edge.

---

### 6. `infosys90d` (90-Day Hourly Multi-Model Tournament)
* **What was tried:**
  * Tested CatBoost, ExtraTrees, GradientBoosting, LightGBM, RandomForest, and XGBoost on 90 days of hourly Infosys data with `pandas-ta` indicators (RSI, Bollinger Bands, MACD).
* **Findings:**
  * Complex boosting models (CatBoost/XGBoost) had high validation scores on recent 15-day chunks, but degraded rapidly under unseen volatility spikes.
  * Pure technical indicators on hourly intervals generate substantial false breakouts without a higher-timeframe daily regime anchor.

---

### 7. `stock_anti` (The Dual-Phase Architecture — Best Structural Breakthrough)
* **What was tried:**
  * Implemented in `train.py` across Reliance, SBI, HDFC Bank, and TCS.
  * **Dual-Model Paradigm:**
    1. **"Close-based" Model:** Run night-before using yesterday's Close to predict expected gap and range before market open.
    2. **"Open-based" Model:** Run at 09:15:05 AM immediately once the actual NSE Opening price is established. Predicts % change from actual Open to today's High and Low.
  * **Macro Correlation:** Fed benchmark index features from `^NSEI` (Nifty 50) and `^GSPC` (S&P 500 overnight action) into the stock feature vector.
  * **True Volatility Scaling:** Used `ATR_14` (Average True Range) and `RSI_14`.
* **Why this is the most valuable architecture:**
  * It acknowledges that before 09:15 AM, the open price is uncertain. Once the open is printed, conditional prediction of High and Low becomes dramatically more accurate.
  * Directly inspired the modern `intraday-predictor`'s 09:15 AM entry and 10:15 AM first-hour breakout logic!

---

## 🚫 Hard Rules: What NEVER to Repeat

1. **NEVER use standard random `train_test_split` on market data:**
   Must always use strict temporal walk-forward validation (`TimeSeriesSplit` or rolling window without lookahead).
2. **NEVER trust directional accuracy exceeding 60%:**
   If a model claims 70%, 80%, or 90% directional accuracy on next-day equity data, **assume data leakage immediately** and run leakage unit tests.
3. **NEVER mix percentage and currency units:**
   Always maintain explicit naming: `_pct` for percentages, `_inr` for rupee amounts. Never subtract percentage variables directly from rupee variables without multiplying by price.
4. **NEVER use unverified news timestamps for intraday triggers:**
   News sentiment must only be used as a macro pre-market filter (e.g. at 08:45 AM before market open), never as a real-time intra-hour trigger unless using a millisecond-precision financial news wire.
5. **NEVER calculate backtests without full transaction friction:**
   Every backtest must calculate Zerodha/Kotak brokerage, STT (0.025% on sell), Exchange turnover fees, SEBI charges, Stamp duty, and GST. Gross profits are illusory.

---

## 💎 Valuable Concepts to Retain & Enhance in `intraday-predictor`

1. **The Dual-Phase Paradigm:**
   - Pre-market overnight scan (08:45 AM) using overnight S&P 500 and GIFT Nifty.
   - Live Open recalibration (09:15 AM) injecting actual NSE Open to compute tight target and stop-loss levels.
2. **Min Margin Ranking for Stock Selection:**
   - Filter down a candidate pool by picking the stock with the largest buffered spread ($\text{Target} - \text{StopLoss}$), rather than the one with the highest speculative return.
3. **Macro Benchmark Features:**
   - Continuing to feed Nifty 50 and sectoral index relative momentum into individual stock predictions.
4. **Volatility-Scaled Buffers:**
   - Instead of static historical MAE, dynamically scale target and stop buffers using current `ATR(14)` or rolling volatility.
