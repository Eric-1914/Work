from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
DOCS_DIR = PROJECT_ROOT / "docs"

TICKERS = ["XLF", "XLK", "XLE"]

REQUIRED_FILES = [
    PROCESSED_DIR / "sector_prices.csv",
    PROCESSED_DIR / "macro_aligned_daily.csv",
    PROCESSED_DIR / "cleaning_summary.csv",
    PROCESSED_DIR / "sector_features.csv",
    PROCESSED_DIR / "feature_dictionary.csv",
    TABLE_DIR / "sector_return_summary.csv",
    TABLE_DIR / "sector_correlation.csv",
    TABLE_DIR / "latest_feature_snapshot.csv",
    TABLE_DIR / "macro_regime_summary.csv",
    DOCS_DIR / "early_insights_report.md",
]

REQUIRED_FIGURES = [
    FIGURE_DIR / "normalized_sector_prices.png",
    FIGURE_DIR / "sector_return_correlation.png",
    FIGURE_DIR / "rolling_return_20d.png",
    FIGURE_DIR / "rolling_volatility_20d.png",
    FIGURE_DIR / "momentum_20d.png",
    FIGURE_DIR / "rsi_14.png",
    FIGURE_DIR / "macd_histogram.png",
]


def validate_files() -> None:
    for path in REQUIRED_FILES + REQUIRED_FIGURES:
        if not path.exists():
            raise FileNotFoundError(f"Missing required Week 2 output: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Week 2 output is empty: {path}")


def validate_prices() -> None:
    path = PROCESSED_DIR / "sector_prices.csv"
    df = pd.read_csv(path, parse_dates=["Date"])

    if df.empty:
        raise ValueError("sector_prices.csv is empty.")
    if df["Date"].isna().any():
        raise ValueError("sector_prices.csv contains invalid dates.")
    if df["Date"].duplicated().any():
        raise ValueError("sector_prices.csv contains duplicate dates.")
    if not df["Date"].is_monotonic_increasing:
        raise ValueError("sector_prices.csv is not sorted by date.")

    for ticker in TICKERS:
        col = f"{ticker}_Adj_Close"
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        values = pd.to_numeric(df[col], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{col} contains missing values after alignment.")
        if (values <= 0).any():
            raise ValueError(f"{col} contains non-positive prices.")


def validate_features() -> None:
    path = PROCESSED_DIR / "sector_features.csv"
    df = pd.read_csv(path, parse_dates=["Date"])

    if df.empty:
        raise ValueError("sector_features.csv is empty.")
    if df["Date"].duplicated().any():
        raise ValueError("sector_features.csv contains duplicate dates.")

    for ticker in TICKERS:
        required = [
            f"{ticker}_Daily_Return",
            f"{ticker}_Rolling_Return_5D",
            f"{ticker}_Rolling_Return_20D",
            f"{ticker}_Rolling_Return_60D",
            f"{ticker}_Momentum_20D",
            f"{ticker}_Rolling_Volatility_20D",
            f"{ticker}_Rolling_Volatility_60D",
            f"{ticker}_RSI_14",
            f"{ticker}_MACD",
            f"{ticker}_MACD_Signal",
            f"{ticker}_MACD_Hist",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{ticker}: missing feature columns {missing}")

        rsi = pd.to_numeric(df[f"{ticker}_RSI_14"], errors="coerce").dropna()
        if not rsi.between(0, 100).all():
            raise ValueError(f"{ticker}: RSI values fall outside [0, 100].")

        vol20 = pd.to_numeric(df[f"{ticker}_Rolling_Volatility_20D"], errors="coerce").dropna()
        vol60 = pd.to_numeric(df[f"{ticker}_Rolling_Volatility_60D"], errors="coerce").dropna()
        if (vol20 < 0).any() or (vol60 < 0).any():
            raise ValueError(f"{ticker}: rolling volatility contains negative values.")

    # Warm-up NaNs are expected for rolling indicators, but each core feature
    # must eventually contain usable observations.
    feature_suffixes = [
        "RSI_14", "MACD", "Momentum_20D",
        "Rolling_Return_20D", "Rolling_Volatility_20D"
    ]
    for ticker in TICKERS:
        for suffix in feature_suffixes:
            col = f"{ticker}_{suffix}"
            if pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
                raise ValueError(f"{col} has no usable observations.")


def validate_correlation() -> None:
    corr = pd.read_csv(TABLE_DIR / "sector_correlation.csv", index_col=0)
    if corr.shape != (3, 3):
        raise ValueError("sector_correlation.csv must be a 3x3 matrix.")
    values = corr.apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any():
        raise ValueError("sector correlation matrix contains non-numeric values.")
    if ((values < -1) | (values > 1)).any().any():
        raise ValueError("Correlation values fall outside [-1, 1].")


def main() -> int:
    try:
        validate_files()
        validate_prices()
        validate_features()
        validate_correlation()
        print("[PASS] Required Week 2 output files exist and are non-empty.")
        print("[PASS] Cleaned sector price data passed validation.")
        print("[PASS] Technical and rolling features passed validation.")
        print("[PASS] Sector correlation matrix passed validation.")
        print("[DONE] All Week 2 outputs passed validation.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 2 validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
