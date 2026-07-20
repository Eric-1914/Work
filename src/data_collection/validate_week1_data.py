"""
Week 1 raw-data validation.

Checks:
- Required files exist.
- Dates are valid, unique, and sorted.
- ETF OHLCV columns exist.
- ETF prices and volume are not negative.
- FRED series contain usable numeric observations.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

ETF_FILES = ["XLF.csv", "XLK.csv", "XLE.csv"]
FRED_FILES = ["GDP_GROWTH.csv", "CPI.csv", "FEDFUNDS.csv"]


def validate_etf(path: Path) -> None:
    df = pd.read_csv(path, parse_dates=["Date"])

    expected = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{path.name}: empty dataset")
    if df["Date"].isna().any():
        raise ValueError(f"{path.name}: invalid dates")
    if df["Date"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate dates")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError(f"{path.name}: dates are not sorted")

    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    numeric_cols = price_cols + ["Volume"]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[price_cols].isna().all(axis=1).any():
        raise ValueError(f"{path.name}: fully empty price row detected")
    if (df[price_cols] < 0).any().any():
        raise ValueError(f"{path.name}: negative price detected")
    if (df["Volume"] < 0).any():
        raise ValueError(f"{path.name}: negative volume detected")

    print(
        f"[PASS] {path.name}: {len(df):,} rows, "
        f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    )


def validate_fred(path: Path) -> None:
    df = pd.read_csv(path, parse_dates=["Date"])

    expected = {"Date", "Value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    if df.empty:
        raise ValueError(f"{path.name}: empty dataset")
    if df["Date"].isna().any():
        raise ValueError(f"{path.name}: invalid dates")
    if df["Date"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate dates")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError(f"{path.name}: dates are not sorted")

    usable = pd.to_numeric(df["Value"], errors="coerce").notna().sum()
    if usable == 0:
        raise ValueError(f"{path.name}: no usable numeric observations")

    print(
        f"[PASS] {path.name}: {len(df):,} rows, {usable:,} usable values, "
        f"{df['Date'].min().date()} -> {df['Date'].max().date()}"
    )


def main() -> int:
    try:
        for filename in ETF_FILES + FRED_FILES:
            path = RAW_DIR / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {path}")

        for filename in ETF_FILES:
            validate_etf(RAW_DIR / filename)

        for filename in FRED_FILES:
            validate_fred(RAW_DIR / filename)

        print("[DONE] All Week 1 raw datasets passed validation.")
        return 0

    except Exception as exc:
        print(f"[ERROR] Validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
