"""Validate Week 7 backtest outputs by recomputing key accounting and metrics."""

from __future__ import annotations

from pathlib import Path
import math
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
CONFIG_PATH = PROJECT_ROOT / "data" / "backtesting" / "week7_backtest_config.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "week7_backtest_report.md"
WEEK6_PATH = TABLE_DIR / "week6_optimized_portfolio.csv"
SECTOR_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "sector_features.csv"

SECTORS = ("XLF", "XLK", "XLE")
ASSETS = (*SECTORS, "CASH")
ANNUALIZATION = 252
TOL = 1e-8
STRATEGY = "Week7_Strategy_Net"
SPY = "SPY_Benchmark"
EW = "Equal_Weighted_Sectors"

REQUIRED_FILES = [
    TABLE_DIR / "week7_daily_backtest.csv",
    TABLE_DIR / "week7_rebalance_log.csv",
    TABLE_DIR / "week7_execution_schedule.csv",
    TABLE_DIR / "week7_benchmark_daily.csv",
    TABLE_DIR / "week7_performance_summary.csv",
    TABLE_DIR / "week7_rolling_validation.csv",
    TABLE_DIR / "week7_cost_sensitivity.csv",
    FIGURE_DIR / "week7_equity_curve.png",
    FIGURE_DIR / "week7_drawdown_comparison.png",
    FIGURE_DIR / "week7_rolling_sharpe.png",
    FIGURE_DIR / "week7_cost_sensitivity.png",
    CONFIG_PATH,
    REPORT_PATH,
]


def _fail(message: str) -> None:
    raise AssertionError(message)


def _close(a: float, b: float, tol: float = 1e-8) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    return abs(float(a) - float(b)) <= tol


def _metric_block(returns: pd.Series) -> dict[str, float]:
    r = returns.astype(float).reset_index(drop=True)
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax().clip(lower=1.0) - 1.0
    n = len(r)
    cagr = float(equity.iloc[-1] ** (ANNUALIZATION / n) - 1.0)
    std = float(r.std(ddof=1)) if n > 1 else 0.0
    sharpe = float(r.mean() / std * math.sqrt(ANNUALIZATION)) if std > 1e-12 else np.nan
    downside = np.minimum(r.to_numpy(dtype=float), 0.0)
    downside_daily = float(np.sqrt(np.mean(np.square(downside))))
    sortino = float(r.mean() / downside_daily * math.sqrt(ANNUALIZATION)) if downside_daily > 1e-12 else np.nan
    return {
        "Total_Return": float(equity.iloc[-1] - 1.0),
        "CAGR": cagr,
        "Sharpe_0RF": sharpe,
        "Sortino_0MAR": sortino,
        "Max_Drawdown": float(dd.min()),
        "Daily_Win_Rate": float((r > 0).mean()),
    }


def validate_files() -> None:
    missing = [str(p.relative_to(PROJECT_ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        _fail(f"Missing Week 7 outputs: {missing}")
    if not WEEK6_PATH.exists() or not SECTOR_FEATURES_PATH.exists():
        _fail("Week 6 targets or sector daily returns are missing.")


def validate_execution_and_accounting() -> None:
    daily = pd.read_csv(TABLE_DIR / "week7_daily_backtest.csv", parse_dates=["Date", "Signal_Date"])
    rebal = pd.read_csv(TABLE_DIR / "week7_rebalance_log.csv", parse_dates=["Signal_Date", "Execution_Date"])
    schedule = pd.read_csv(TABLE_DIR / "week7_execution_schedule.csv", parse_dates=["Signal_Date", "Execution_Date", "Effective_From_Date"])
    returns = pd.read_csv(SECTOR_FEATURES_PATH, parse_dates=["Date"])
    targets = pd.read_csv(WEEK6_PATH, parse_dates=["Signal_Date"])

    if daily.empty or rebal.empty or schedule.empty:
        _fail("Week 7 daily, rebalance, and execution tables must be non-empty.")
    if not daily["Date"].is_monotonic_increasing or daily["Date"].duplicated().any():
        _fail("Week 7 daily backtest dates must be unique and sorted.")
    if (daily["Equity"] <= 0).any():
        _fail("Week 7 equity must stay positive.")
    if not (schedule["Execution_Date"] > schedule["Signal_Date"]).all():
        _fail("Every execution date must be strictly after its signal date.")

    available = pd.DatetimeIndex(returns["Date"].dropna().sort_values().unique())
    for row in schedule.itertuples(index=False):
        later = available[available > pd.Timestamp(row.Signal_Date)]
        if len(later) < 2 or pd.Timestamp(row.Execution_Date) != pd.Timestamp(later[0]):
            _fail("Execution schedule does not use a valid first post-signal trading date with a later effective observation.")
        if pd.isna(row.Effective_From_Date) or pd.Timestamp(row.Effective_From_Date) != pd.Timestamp(later[1]):
            _fail("Effective_From_Date must be the next available trading observation after execution.")

    if set(targets["Signal_Date"].unique()) != set(schedule["Signal_Date"].unique()):
        _fail("Execution schedule does not cover every Week 6 signal date exactly once.")

    calc_equity = (1.0 + daily["Net_Return"].astype(float)).cumprod()
    if not np.allclose(calc_equity, daily["Equity"].astype(float), atol=2e-8, rtol=2e-8):
        _fail("Saved equity does not equal compounded net daily returns.")
    calc_dd = calc_equity / calc_equity.cummax().clip(lower=1.0) - 1.0
    if not np.allclose(calc_dd, daily["Drawdown"].astype(float), atol=2e-8, rtol=2e-8):
        _fail("Saved drawdown does not match recomputed equity drawdown.")

    # Recompute turnover from the saved drifted pre-trade and new target weights.
    expected_ow = 0.5 * sum(
        (rebal[f"Target_{asset}"].astype(float) - rebal[f"PreTrade_{asset}"].astype(float)).abs()
        for asset in ASSETS
    )
    if not np.allclose(expected_ow, rebal["One_Way_Turnover"].astype(float), atol=1e-10, rtol=1e-10):
        _fail("One-way turnover does not match drifted pre-trade versus target weights.")
    expected_risky = sum(
        (rebal[f"Target_{sector}"].astype(float) - rebal[f"PreTrade_{sector}"].astype(float)).abs()
        for sector in SECTORS
    )
    if not np.allclose(expected_risky, rebal["Risky_Trade_Notional"].astype(float), atol=1e-10, rtol=1e-10):
        _fail("Risky trade notional does not match saved pre-trade versus target sector weights.")

    # Friction accounting: total cost fraction equals risky trade notional times total bps.
    total_bps = rebal["Transaction_Cost_Bps"].astype(float) + rebal["Slippage_Bps"].astype(float)
    expected_cost = rebal["Risky_Trade_Notional"].astype(float) * total_bps / 10000.0
    if not np.allclose(expected_cost, rebal["Total_Trading_Cost_Fraction"].astype(float), atol=1e-10, rtol=1e-10):
        _fail("Trading-cost fractions do not match risky trade notional and bps assumptions.")


def validate_benchmarks_and_metrics() -> None:
    daily = pd.read_csv(TABLE_DIR / "week7_daily_backtest.csv", parse_dates=["Date"])
    bench = pd.read_csv(TABLE_DIR / "week7_benchmark_daily.csv", parse_dates=["Date"])
    summary = pd.read_csv(TABLE_DIR / "week7_performance_summary.csv")
    aligned = daily[["Date", "Net_Return"]].merge(bench, on="Date", how="inner").sort_values("Date")
    if len(aligned) < 20:
        _fail("Insufficient aligned benchmark observations.")

    if not np.allclose(
        aligned["Equal_Weighted_Sectors_Equity"].astype(float),
        (1.0 + aligned["Equal_Weighted_Sector_Return"].astype(float)).cumprod(),
        atol=2e-8, rtol=2e-8,
    ):
        _fail("Equal-weight benchmark equity does not compound its saved daily return.")
    if not np.allclose(
        aligned["SPY_Equity"].astype(float),
        (1.0 + aligned["SPY"].astype(float)).cumprod(),
        atol=2e-8, rtol=2e-8,
    ):
        _fail("SPY benchmark equity does not compound its saved daily return.")
    if abs(float(aligned["SPY"].iloc[0])) > 1e-12:
        _fail("SPY benchmark incorrectly earns the execution-date return before the position can be held.")
    if abs(float(aligned["Equal_Weighted_Sector_Return"].iloc[0])) > 1e-12:
        _fail("Equal-weight benchmark incorrectly earns the execution-date return before the position can be held.")

    lookup = summary.set_index("Strategy")
    for required in (STRATEGY, SPY, EW):
        if required not in lookup.index:
            _fail(f"Missing performance summary row: {required}")

    cases = {
        STRATEGY: aligned["Net_Return"],
        SPY: aligned["SPY"],
        EW: aligned["Equal_Weighted_Sector_Return"],
    }
    for name, returns in cases.items():
        m = _metric_block(returns)
        row = lookup.loc[name]
        for col, expected in m.items():
            if not _close(row[col], expected, 2e-8):
                _fail(f"Metric mismatch for {name} / {col}: saved={row[col]}, recomputed={expected}")


def validate_rolling_and_sensitivity() -> None:
    rolling = pd.read_csv(TABLE_DIR / "week7_rolling_validation.csv")
    sensitivity = pd.read_csv(TABLE_DIR / "week7_cost_sensitivity.csv")
    if sensitivity.empty:
        _fail("Cost sensitivity output must not be empty.")
    bps = sensitivity["Total_Friction_Bps_Per_Risky_Dollar_Traded"].astype(float)
    if not bps.is_monotonic_increasing:
        _fail("Cost sensitivity scenarios must be ordered from low to high friction.")
    # With identical targets and returns, increasing positive costs cannot improve terminal return.
    total_return = sensitivity["Total_Return"].astype(float).to_numpy()
    if np.any(np.diff(total_return) > 2e-10):
        _fail("Higher trading friction unexpectedly increased total strategy return.")

    if not rolling.empty:
        for col in ("Observations", "CAGR", "Sharpe_0RF", "Sortino_0MAR", "Max_Drawdown", "Daily_Win_Rate"):
            if col not in rolling.columns:
                _fail(f"Rolling validation is missing {col}.")
        if (rolling["Observations"] < 63).any():
            _fail("Rolling validation contains a window shorter than the configured minimum.")


def main() -> int:
    try:
        validate_files()
        validate_execution_and_accounting()
        validate_benchmarks_and_metrics()
        validate_rolling_and_sensitivity()
        print("[PASS] Required Week 7 outputs and Week 6 handoff files exist.")
        print("[PASS] Execution and effective dates follow the conservative post-signal timing rule.")
        print("[PASS] Equity, drawdown, turnover-related cost accounting, and saved daily returns passed recomputation.")
        print("[PASS] SPY and equal-weight benchmark series and performance metrics passed independent recomputation.")
        print("[PASS] Rolling validation and trading-cost sensitivity checks passed.")
        print("[DONE] All Week 7 outputs passed validation.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 7 validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
