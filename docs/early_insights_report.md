# Week 2 Early Insights Report

## Scope
This report summarizes preliminary descriptive findings from Week 2 data cleaning,
feature engineering, and exploratory data analysis for XLF, XLK, and XLE. It does
not make investment recommendations or train a predictive model.

## 1. Data Cleaning and Alignment
The three ETF series were aligned on common trading dates. Macroeconomic variables
were aligned only after approximate publication lags were applied. CPI year-over-year
inflation was derived from the CPI index.

## 2. Sector Performance and Risk
Across the available sample, **XLK** had the highest cumulative return
at **872.11%**. **XLE** had the highest
annualized volatility at **29.14%**.

## 3. Sector Correlation
The strongest absolute daily-return correlation was between **XLF** and **XLE**,
with a correlation of **0.658**.

## 4. Technical Feature Snapshot
Latest feature date: **2026-07-17**

- XLF: RSI(14) = 65.58 (neutral-range), 20-day momentum = 4.45%.
- XLK: RSI(14) = 42.39 (neutral-range), 20-day momentum = -5.38%.
- XLE: RSI(14) = 61.84 (neutral-range), 20-day momentum = 6.27%.

RSI, MACD, momentum, rolling returns, and rolling volatility are candidate model
features only; no single indicator is treated as a trading rule.

## 5. Optional Macroeconomic Regime Tag
Latest heuristic macro regime:

**Expansion | Inflation:High | Rates:Stable**

These regime labels are research tags, not official economic-cycle classifications.

## 6. Overall Early Insight
The Week 2 pipeline establishes a common daily timeline, captures trend/momentum/risk
features, and quantifies cross-sector correlation. The next stage should define a
prediction target and use time-aware train/validation/test splits to avoid look-ahead bias.

## Methodology Limitation
The macro availability lags used here are approximate. Production-grade backtesting
should use exact point-in-time release/vintage data.
