# 🚀 Stock7 (Archived / Legacy v3)

> ⚠️ **ARCHIVE NOTICE & CRITICAL LEARNINGS:**  
> This repository contains historical research from **July 2025**. It represents the most feature-complete prototype of our legacy series, featuring an automated pipeline and a 60-day profit simulation dashboard.
>
> 🚨 **Technical Caveats:**
> 1. **Data Leakage in Backtest:** The saved 60-day backtest reported ~90% directional accuracy across all 5 stocks. Subsequent leakage audits proved this was caused by future lookahead during feature and target shifting. Real next-day equity direction predictability is ~50–55%.
> 2. **Unit Conversion Bug:** Model MAE errors were calculated in **percentage terms (%)**, but subtracted directly from price in rupees (**₹**) inside `app.py` (e.g. ₹900 − 0.65 = ₹899.35 instead of 0.65% = ₹5.85), effectively eliminating the planned safety buffer.
> 
> Active production development, leak-free walk-forward validation, and automated cloud pipelines are now at **[VedantSingh2000/intraday-predictor](https://github.com/VedantSingh2000/intraday-predictor)**.

---

## 📌 Project Overview
* **Stocks (5):** `TATAMOTORS.NS`, `UNITDSPR.NS`, `PETRONET.NS`, `COLPAL.NS`, `BEL.NS`.
* **Features (7):** `['Open', 'High', 'Low', 'Close', 'SMA_10', 'SMA_30', 'Price_Change']`.
* **Scalers:** Individual `StandardScaler` instances per stock per target.
* **Modes in `app.py`:**
  1. Live prediction page with dynamic error boundaries.
  2. 60-day interactive backtest simulation with color-coded errors.
  3. Monthly Min-Margin aggregation table.

---

## 📓 Notebooks & Code
* **`combined.ipynb`**: Complete end-to-end training and evaluation notebook used to train all 20 models and produce `multi_stock_scalers.pkl`, `multi_stock_errors.pkl`, and `multi_stock_backtest.pkl`.
* **`extracted_code.py`**: Standalone Python script extraction of the pipeline.
* **`RETROSPECTIVE_AND_LEARNINGS.md`**: Complete post-mortem detailing all experiments, failure modes, what NOT to repeat, and takeaways.
