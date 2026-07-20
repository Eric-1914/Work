"""
Week 1 Data Infrastructure & Acquisition
Project: Machine Learning-Based Sector Rotation Strategy

Baseline scope:
- Sector ETFs: XLF, XLK, XLE
- FRED macro data:
    * Real GDP growth: A191RL1Q225SBEA
    * CPI: CPIAUCSL
    * Federal Funds Effective Rate: FEDFUNDS

Outputs:
- data/raw/XLF.csv
- data/raw/XLK.csv
- data/raw/XLE.csv
- data/raw/GDP_GROWTH.csv
- data/raw/CPI.csv
- data/raw/FEDFUNDS.csv
- data/raw/data_inventory.csv

Notes:
- CPIAUCSL is a CPI index. YoY inflation should be derived from CPI during
  Week 2 feature engineering rather than treated as if the index itself were
  an inflation rate.
- The script stores raw observations only; time-series alignment is deferred
  to Week 2.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

START_DATE = "2015-01-01"
# yfinance treats end as exclusive; tomorrow ensures today's latest available
# trading observation can be included when available.
END_DATE = (date.today() + timedelta(days=1)).isoformat()

ETF_TICKERS = ["XLF", "XLK", "XLE"]

FRED_SERIES = {
    "GDP_GROWTH": "A191RL1Q225SBEA",
    "CPI": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
}


def ensure_output_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_etf(ticker: str) -> Path:
    """Download one ETF's daily OHLCV history and save it as CSV."""
    print(f"[ETF] Downloading {ticker}: {START_DATE} -> latest available")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
        timeout=30,
    )

    if df.empty:
        raise RuntimeError(f"No ETF data returned for {ticker}.")

    # yfinance may return MultiIndex columns depending on version.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    expected = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker} is missing expected columns: {missing}")

    df = df[expected].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if df.empty:
        raise RuntimeError(f"{ticker} contains no usable observations.")

    output = RAW_DIR / f"{ticker}.csv"
    df.to_csv(output, date_format="%Y-%m-%d")

    print(
        f"      Saved {len(df):,} rows to {output.relative_to(PROJECT_ROOT)} "
        f"({df.index.min().date()} -> {df.index.max().date()})"
    )
    return output


def download_fred(name: str, series_id: str) -> Path:
    """Download a FRED series through FRED's public CSV graph endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    print(f"[FRED] Downloading {name} ({series_id})")

    df = pd.read_csv(url)

    if df.empty or len(df.columns) < 2:
        raise RuntimeError(f"No usable FRED data returned for {series_id}.")

    date_col = df.columns[0]
    value_col = df.columns[1]

    df = df.rename(columns={date_col: "Date", value_col: "Value"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date")
    df = df[df["Date"] >= pd.Timestamp(START_DATE)]
    df = df.drop_duplicates(subset=["Date"], keep="last")

    if df.empty:
        raise RuntimeError(f"{series_id} has no observations after {START_DATE}.")

    output = RAW_DIR / f"{name}.csv"
    df.to_csv(output, index=False, date_format="%Y-%m-%d")

    usable = df["Value"].notna().sum()
    print(
        f"       Saved {len(df):,} rows ({usable:,} non-null) to "
        f"{output.relative_to(PROJECT_ROOT)} "
        f"({df['Date'].min().date()} -> {df['Date'].max().date()})"
    )
    return output


def build_inventory() -> Path:
    """Create a compact inventory of the Week 1 datasets."""
    records = [
        {
            "dataset": "XLF",
            "category": "ETF market data",
            "frequency": "Daily (trading days)",
            "source": "Yahoo Finance via yfinance",
            "series_or_ticker": "XLF",
            "file": "data/raw/XLF.csv",
        },
        {
            "dataset": "XLK",
            "category": "ETF market data",
            "frequency": "Daily (trading days)",
            "source": "Yahoo Finance via yfinance",
            "series_or_ticker": "XLK",
            "file": "data/raw/XLK.csv",
        },
        {
            "dataset": "XLE",
            "category": "ETF market data",
            "frequency": "Daily (trading days)",
            "source": "Yahoo Finance via yfinance",
            "series_or_ticker": "XLE",
            "file": "data/raw/XLE.csv",
        },
        {
            "dataset": "GDP_GROWTH",
            "category": "Macroeconomic data",
            "frequency": "Quarterly",
            "source": "FRED",
            "series_or_ticker": "A191RL1Q225SBEA",
            "file": "data/raw/GDP_GROWTH.csv",
        },
        {
            "dataset": "CPI",
            "category": "Macroeconomic data",
            "frequency": "Monthly",
            "source": "FRED",
            "series_or_ticker": "CPIAUCSL",
            "file": "data/raw/CPI.csv",
        },
        {
            "dataset": "FEDFUNDS",
            "category": "Macroeconomic data",
            "frequency": "Monthly",
            "source": "FRED",
            "series_or_ticker": "FEDFUNDS",
            "file": "data/raw/FEDFUNDS.csv",
        },
    ]

    output = RAW_DIR / "data_inventory.csv"
    pd.DataFrame(records).to_csv(output, index=False)
    return output


def main() -> int:
    ensure_output_dir()

    try:
        for ticker in ETF_TICKERS:
            download_etf(ticker)

        for name, series_id in FRED_SERIES.items():
            download_fred(name, series_id)

        inventory = build_inventory()
        print(f"[DONE] Inventory saved to {inventory.relative_to(PROJECT_ROOT)}")
        print("[DONE] Week 1 data collection completed successfully.")
        return 0

    except Exception as exc:
        print(f"\n[ERROR] Week 1 data collection failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
