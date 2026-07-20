from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ETF_TICKERS = ["XLF", "XLK", "XLE"]

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}. Run Week 1 first.")

def load_etf(ticker: str) -> pd.DataFrame:
    path = RAW_DIR / f"{ticker}.csv"
    require_file(path)
    df = pd.read_csv(path)
    required = {"Date","Open","High","Low","Close","Adj Close","Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: missing columns {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    numeric = ["Open","High","Low","Close","Adj Close","Volume"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Adj Close"])
    rename = {
        "Open":f"{ticker}_Open","High":f"{ticker}_High","Low":f"{ticker}_Low",
        "Close":f"{ticker}_Close","Adj Close":f"{ticker}_Adj_Close","Volume":f"{ticker}_Volume"
    }
    return df[["Date"] + numeric].rename(columns=rename)

def build_sector_prices() -> pd.DataFrame:
    merged = None
    for ticker in ETF_TICKERS:
        current = load_etf(ticker)
        merged = current if merged is None else merged.merge(current, on="Date", how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise RuntimeError("No aligned ETF observations were produced.")
    merged = merged.sort_values("Date").reset_index(drop=True)
    if merged["Date"].duplicated().any():
        raise ValueError("Aligned ETF data contain duplicate dates.")
    return merged

def load_fred(filename: str, value_name: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    require_file(path)
    df = pd.read_csv(path)
    required = {"Date","Value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{filename}: missing columns {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
    return df.rename(columns={"Value":value_name})[["Date", value_name]]

def align_macro_to_daily(trading_dates: pd.Series) -> pd.DataFrame:
    daily = pd.DataFrame({"Date":pd.to_datetime(trading_dates)}).sort_values("Date")
    gdp = load_fred("GDP_GROWTH.csv","GDP_Growth").dropna(subset=["GDP_Growth"]).copy()
    cpi = load_fred("CPI.csv","CPI_Index").copy()
    fed = load_fred("FEDFUNDS.csv","FedFunds_Rate").dropna(subset=["FedFunds_Rate"]).copy()

    cpi["Inflation_YoY"] = cpi["CPI_Index"].pct_change(12, fill_method=None) * 100.0
    cpi = cpi.dropna(subset=["CPI_Index"]).copy()

    # Approximate availability lags to reduce obvious look-ahead bias.
    gdp["AvailableDate"] = gdp["Date"] + pd.DateOffset(months=3)
    cpi["AvailableDate"] = cpi["Date"] + pd.DateOffset(months=1)
    fed["AvailableDate"] = fed["Date"] + pd.DateOffset(months=1)

    def merge_asof(left, right, cols):
        right = right[["AvailableDate"] + cols].sort_values("AvailableDate")
        out = pd.merge_asof(
            left.sort_values("Date"), right,
            left_on="Date", right_on="AvailableDate", direction="backward"
        )
        return out.drop(columns=["AvailableDate"])

    aligned = merge_asof(daily, gdp, ["GDP_Growth"])
    aligned = merge_asof(aligned, cpi, ["CPI_Index","Inflation_YoY"])
    aligned = merge_asof(aligned, fed, ["FedFunds_Rate"])

    aligned["Growth_Regime"] = np.select(
        [aligned["GDP_Growth"].isna(), aligned["GDP_Growth"] > 0],
        ["Unknown","Expansion"], default="Contraction"
    )
    aligned["Inflation_Regime"] = np.select(
        [aligned["Inflation_YoY"].isna(), aligned["Inflation_YoY"] < 2, aligned["Inflation_YoY"] < 3],
        ["Unknown","Low","Moderate"], default="High"
    )
    rate_change = aligned["FedFunds_Rate"] - aligned["FedFunds_Rate"].shift(63)
    aligned["Rate_Regime"] = np.select(
        [aligned["FedFunds_Rate"].isna(), rate_change > 0.25, rate_change < -0.25],
        ["Unknown","Rising","Falling"], default="Stable"
    )
    aligned["Macro_Regime"] = (
        aligned["Growth_Regime"] + " | Inflation:" + aligned["Inflation_Regime"]
        + " | Rates:" + aligned["Rate_Regime"]
    )
    return aligned

def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        prices = build_sector_prices()
        macro = align_macro_to_daily(prices["Date"])
        summary = pd.DataFrame([
            {
                "dataset":"sector_prices","rows":len(prices),
                "start_date":prices["Date"].min().date(),"end_date":prices["Date"].max().date(),
                "duplicate_dates":int(prices["Date"].duplicated().sum()),
                "missing_values":int(prices.isna().sum().sum())
            },
            {
                "dataset":"macro_aligned_daily","rows":len(macro),
                "start_date":macro["Date"].min().date(),"end_date":macro["Date"].max().date(),
                "duplicate_dates":int(macro["Date"].duplicated().sum()),
                "missing_values":int(macro.isna().sum().sum())
            }
        ])
        prices.to_csv(PROCESSED_DIR/"sector_prices.csv", index=False, date_format="%Y-%m-%d")
        macro.to_csv(PROCESSED_DIR/"macro_aligned_daily.csv", index=False, date_format="%Y-%m-%d")
        summary.to_csv(PROCESSED_DIR/"cleaning_summary.csv", index=False)
        print(f"[PASS] Aligned ETF data: {len(prices):,} rows")
        print("[PASS] Saved data/processed/sector_prices.csv")
        print("[PASS] Saved data/processed/macro_aligned_daily.csv")
        print("[PASS] Saved data/processed/cleaning_summary.csv")
        print("[DONE] Week 2 data cleaning and time-series alignment completed.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 2 cleaning failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
