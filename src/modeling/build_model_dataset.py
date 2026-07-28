"""
Week 3 - Build the monthly machine-learning dataset.

Prediction target
-----------------
For each sector ETF and each monthly signal date:

    next_21d_sector_return
        = adjusted close 21 trading days later / current adjusted close - 1

    next_21d_equal_weight_return
        = mean of XLF, XLK, and XLE next-21-trading-day returns

    relative_return_21d
        = sector return - equal-weight benchmark return

    target_outperform
        = 1 when relative_return_21d > 0, otherwise 0

The monthly signal uses the final available trading observation in each calendar
month. The target therefore represents approximately next-month relative return.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "sector_features.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "modeling"

TICKERS = ["XLF", "XLK", "XLE"]
HORIZON_DAYS = 21


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"sector_features.csv is missing columns: {missing}")


def build_daily_panel(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    macro_numeric = ["GDP_Growth", "Inflation_YoY", "FedFunds_Rate"]
    macro_categorical = ["Growth_Regime", "Inflation_Regime", "Rate_Regime"]

    for ticker in TICKERS:
        required = [
            "Date",
            f"{ticker}_Adj_Close",
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
        ] + macro_numeric + macro_categorical
        require_columns(df, required)

        close = pd.to_numeric(df[f"{ticker}_Adj_Close"], errors="coerce")

        current = pd.DataFrame(
            {
                "Date": df["Date"],
                "Sector": ticker,
                "Adj_Close": close,
                "Daily_Return": pd.to_numeric(
                    df[f"{ticker}_Daily_Return"], errors="coerce"
                ),
                "Rolling_Return_5D": pd.to_numeric(
                    df[f"{ticker}_Rolling_Return_5D"], errors="coerce"
                ),
                "Rolling_Return_20D": pd.to_numeric(
                    df[f"{ticker}_Rolling_Return_20D"], errors="coerce"
                ),
                "Rolling_Return_60D": pd.to_numeric(
                    df[f"{ticker}_Rolling_Return_60D"], errors="coerce"
                ),
                "Momentum_20D": pd.to_numeric(
                    df[f"{ticker}_Momentum_20D"], errors="coerce"
                ),
                "Rolling_Volatility_20D": pd.to_numeric(
                    df[f"{ticker}_Rolling_Volatility_20D"], errors="coerce"
                ),
                "Rolling_Volatility_60D": pd.to_numeric(
                    df[f"{ticker}_Rolling_Volatility_60D"], errors="coerce"
                ),
                "RSI_14": pd.to_numeric(
                    df[f"{ticker}_RSI_14"], errors="coerce"
                ),
                "MACD_Pct": (
                    pd.to_numeric(df[f"{ticker}_MACD"], errors="coerce") / close
                ),
                "MACD_Signal_Pct": (
                    pd.to_numeric(
                        df[f"{ticker}_MACD_Signal"], errors="coerce"
                    )
                    / close
                ),
                "MACD_Hist_Pct": (
                    pd.to_numeric(
                        df[f"{ticker}_MACD_Hist"], errors="coerce"
                    )
                    / close
                ),
                "GDP_Growth": pd.to_numeric(df["GDP_Growth"], errors="coerce"),
                "Inflation_YoY": pd.to_numeric(
                    df["Inflation_YoY"], errors="coerce"
                ),
                "FedFunds_Rate": pd.to_numeric(
                    df["FedFunds_Rate"], errors="coerce"
                ),
                "Growth_Regime": df["Growth_Regime"].astype("string"),
                "Inflation_Regime": df["Inflation_Regime"].astype("string"),
                "Rate_Regime": df["Rate_Regime"].astype("string"),
                "Next_21D_Sector_Return": (
                    close.shift(-HORIZON_DAYS) / close - 1.0
                ),
            }
        )
        rows.append(current)

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["Date", "Sector"]).reset_index(drop=True)

    benchmark = (
        panel.groupby("Date", as_index=False)["Next_21D_Sector_Return"]
        .mean()
        .rename(
            columns={
                "Next_21D_Sector_Return": "Next_21D_EqualWeight_Return"
            }
        )
    )
    panel = panel.merge(benchmark, on="Date", how="left", validate="many_to_one")
    panel["Relative_Return_21D"] = (
        panel["Next_21D_Sector_Return"]
        - panel["Next_21D_EqualWeight_Return"]
    )
    panel["Target_Outperform"] = np.where(
        panel["Relative_Return_21D"].notna(),
        (panel["Relative_Return_21D"] > 0).astype(int),
        np.nan,
    )

    relative_source = [
        "Rolling_Return_20D",
        "Rolling_Return_60D",
        "Momentum_20D",
        "Rolling_Volatility_20D",
        "RSI_14",
        "MACD_Hist_Pct",
    ]
    for feature in relative_source:
        date_mean = panel.groupby("Date")[feature].transform("mean")
        panel[f"Market_Mean_{feature}"] = date_mean
        panel[f"Relative_{feature}"] = panel[feature] - date_mean
        panel[f"Rank_{feature}"] = panel.groupby("Date")[feature].rank(
            method="average", pct=True
        )

    return panel


def sample_month_end(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["Month"] = panel["Date"].dt.to_period("M")

    monthly = (
        panel.sort_values(["Sector", "Date"])
        .groupby(["Sector", "Month"], as_index=False)
        .tail(1)
        .drop(columns=["Month"])
        .sort_values(["Date", "Sector"])
        .reset_index(drop=True)
    )

    feature_columns = [
        "Daily_Return",
        "Rolling_Return_5D",
        "Rolling_Return_20D",
        "Rolling_Return_60D",
        "Momentum_20D",
        "Rolling_Volatility_20D",
        "Rolling_Volatility_60D",
        "RSI_14",
        "MACD_Pct",
        "MACD_Signal_Pct",
        "MACD_Hist_Pct",
        "Market_Mean_Rolling_Return_20D",
        "Relative_Rolling_Return_20D",
        "Rank_Rolling_Return_20D",
        "Market_Mean_Rolling_Return_60D",
        "Relative_Rolling_Return_60D",
        "Rank_Rolling_Return_60D",
        "Market_Mean_Momentum_20D",
        "Relative_Momentum_20D",
        "Rank_Momentum_20D",
        "Market_Mean_Rolling_Volatility_20D",
        "Relative_Rolling_Volatility_20D",
        "Rank_Rolling_Volatility_20D",
        "Market_Mean_RSI_14",
        "Relative_RSI_14",
        "Rank_RSI_14",
        "Market_Mean_MACD_Hist_Pct",
        "Relative_MACD_Hist_Pct",
        "Rank_MACD_Hist_Pct",
        "GDP_Growth",
        "Inflation_YoY",
        "FedFunds_Rate",
    ]

    categorical_columns = [
        "Sector",
        "Growth_Regime",
        "Inflation_Regime",
        "Rate_Regime",
    ]

    required = (
        ["Date", "Target_Outperform", "Relative_Return_21D"]
        + feature_columns
        + categorical_columns
    )
    require_columns(monthly, required)

    monthly = monthly.dropna(
        subset=feature_columns
        + categorical_columns
        + ["Target_Outperform", "Relative_Return_21D"]
    ).copy()
    monthly["Target_Outperform"] = monthly["Target_Outperform"].astype(int)

    if monthly.empty:
        raise RuntimeError("No usable monthly modeling observations were produced.")

    counts = monthly.groupby("Date")["Sector"].nunique()
    incomplete_dates = counts[counts != len(TICKERS)]
    if not incomplete_dates.empty:
        monthly = monthly[
            ~monthly["Date"].isin(incomplete_dates.index)
        ].copy()

    return monthly.sort_values(["Date", "Sector"]).reset_index(drop=True)


def build_target_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    return (
        monthly.groupby("Sector")["Target_Outperform"]
        .agg(["count", "mean", "sum"])
        .reset_index()
        .rename(
            columns={
                "mean": "outperformance_rate",
                "sum": "outperformance_count",
            }
        )
    )


def main() -> int:
    try:
        if not INPUT_PATH.exists():
            raise FileNotFoundError(
                "data/processed/sector_features.csv is missing. "
                "Complete Week 2 first."
            )

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        source = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
        source = source.sort_values("Date").drop_duplicates("Date", keep="last")

        panel = build_daily_panel(source)
        monthly = sample_month_end(panel)

        dataset_path = OUTPUT_DIR / "model_dataset.csv"
        summary_path = OUTPUT_DIR / "target_summary.csv"
        config_path = OUTPUT_DIR / "modeling_config.json"

        monthly.to_csv(dataset_path, index=False, date_format="%Y-%m-%d")
        build_target_summary(monthly).to_csv(summary_path, index=False)

        config = {
            "target": "Target_Outperform",
            "continuous_target": "Relative_Return_21D",
            "forecast_horizon_trading_days": HORIZON_DAYS,
            "benchmark": "Equal-weight mean of XLF, XLK, and XLE",
            "sampling_frequency": "Calendar month-end signal observations",
            "sectors": TICKERS,
            "rows": int(len(monthly)),
            "unique_signal_dates": int(monthly["Date"].nunique()),
            "start_date": str(monthly["Date"].min().date()),
            "end_date": str(monthly["Date"].max().date()),
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        print(
            f"[PASS] Built monthly model dataset: {len(monthly):,} rows, "
            f"{monthly['Date'].nunique()} signal dates"
        )
        print("[PASS] Defined next-21-trading-day relative-return target.")
        print("[PASS] Saved data/modeling/model_dataset.csv")
        print("[PASS] Saved data/modeling/target_summary.csv")
        print("[PASS] Saved data/modeling/modeling_config.json")
        print("[DONE] Week 3 modeling dataset completed.")
        return 0

    except Exception as exc:
        print(f"[ERROR] Week 3 dataset construction failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
