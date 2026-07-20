from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ETF_TICKERS = ["XLF","XLK","XLE"]

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's original recursive smoothing method."""
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= period:
        return rsi

    avg_gain = gains.iloc[1:period + 1].mean()
    avg_loss = losses.iloc[1:period + 1].mean()

    def to_rsi(gain_value: float, loss_value: float) -> float:
        if pd.isna(gain_value) or pd.isna(loss_value):
            return np.nan
        if loss_value == 0:
            return 100.0 if gain_value > 0 else 50.0
        rs = gain_value / loss_value
        return 100.0 - (100.0 / (1.0 + rs))

    rsi.iloc[period] = to_rsi(avg_gain, avg_loss)

    for i in range(period + 1, len(close)):
        gain_i = gains.iloc[i]
        loss_i = losses.iloc[i]
        if pd.isna(gain_i) or pd.isna(loss_i):
            continue
        avg_gain = ((avg_gain * (period - 1)) + gain_i) / period
        avg_loss = ((avg_loss * (period - 1)) + loss_i) / period
        rsi.iloc[i] = to_rsi(avg_gain, avg_loss)

    return rsi

def add_features(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    close = pd.to_numeric(df[f"{ticker}_Adj_Close"], errors="coerce")
    ret = close.pct_change(fill_method=None)
    df[f"{ticker}_Daily_Return"] = ret
    df[f"{ticker}_Rolling_Return_5D"] = close.pct_change(5, fill_method=None)
    df[f"{ticker}_Rolling_Return_20D"] = close.pct_change(20, fill_method=None)
    df[f"{ticker}_Rolling_Return_60D"] = close.pct_change(60, fill_method=None)
    df[f"{ticker}_Momentum_20D"] = close/close.shift(20)-1
    df[f"{ticker}_Rolling_Volatility_20D"] = ret.rolling(20, min_periods=20).std()*np.sqrt(252)
    df[f"{ticker}_Rolling_Volatility_60D"] = ret.rolling(60, min_periods=60).std()*np.sqrt(252)
    df[f"{ticker}_RSI_14"] = compute_rsi(close, 14)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12-ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    df[f"{ticker}_MACD"] = macd
    df[f"{ticker}_MACD_Signal"] = signal
    df[f"{ticker}_MACD_Hist"] = macd-signal
    return df

def feature_dictionary() -> pd.DataFrame:
    rows = [
        ["Daily_Return","1-day adjusted-close percentage return"],
        ["Rolling_Return_5D","5-trading-day adjusted-close percentage return"],
        ["Rolling_Return_20D","20-trading-day adjusted-close percentage return"],
        ["Rolling_Return_60D","60-trading-day adjusted-close percentage return"],
        ["Momentum_20D","20-trading-day price momentum"],
        ["Rolling_Volatility_20D","20-day return volatility annualized by sqrt(252)"],
        ["Rolling_Volatility_60D","60-day return volatility annualized by sqrt(252)"],
        ["RSI_14","14-period RSI using Wilder original recursive smoothing"],
        ["MACD","12-period EMA minus 26-period EMA"],
        ["MACD_Signal","9-period EMA of MACD"],
        ["MACD_Hist","MACD minus MACD signal"],
        ["Inflation_YoY","Year-over-year percentage change derived from CPI"],
        ["Macro_Regime","Optional heuristic growth/inflation/rate regime tag"],
    ]
    return pd.DataFrame(rows, columns=["feature","description"])

def main() -> int:
    try:
        prices_path = PROCESSED_DIR/"sector_prices.csv"
        macro_path = PROCESSED_DIR/"macro_aligned_daily.csv"
        if not prices_path.exists() or not macro_path.exists():
            raise FileNotFoundError("Run src/data_cleaning/clean_data.py first.")
        features = pd.read_csv(prices_path, parse_dates=["Date"]).sort_values("Date")
        macro = pd.read_csv(macro_path, parse_dates=["Date"]).sort_values("Date")
        for ticker in ETF_TICKERS:
            features = add_features(features, ticker)
        features = features.merge(macro, on="Date", how="left", validate="one_to_one")
        features.to_csv(PROCESSED_DIR/"sector_features.csv", index=False, date_format="%Y-%m-%d")
        feature_dictionary().to_csv(PROCESSED_DIR/"feature_dictionary.csv", index=False)
        print("[PASS] Saved data/processed/sector_features.csv")
        print("[PASS] Saved data/processed/feature_dictionary.csv")
        print("[DONE] Week 2 feature engineering completed.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 2 feature engineering failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
