"""Isolated end-to-end smoke and regression test for Week 7."""

from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SECTORS = ("XLF", "XLK", "XLE")
ASSETS = (*SECTORS, "CASH")


def _copy_code(root: Path) -> None:
    for rel in [
        "run_week7.py",
        "src/backtesting/backtest_week7.py",
        "src/validation/validate_week7.py",
        "docs/week7_methodology.md",
    ]:
        src = SOURCE_ROOT / rel
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _make_data(root: Path) -> None:
    rng = np.random.default_rng(20260821)
    dates = pd.bdate_range("2024-01-02", periods=420)
    market = rng.normal(0.00035, 0.008, len(dates))
    xlf = market + rng.normal(0.00005, 0.0045, len(dates))
    xlk = 1.08 * market + rng.normal(0.00015, 0.0055, len(dates))
    xle = 0.75 * market + rng.normal(0.00002, 0.0065, len(dates))
    spy = 0.92 * market + rng.normal(0.00008, 0.0025, len(dates))

    features = pd.DataFrame({
        "Date": dates,
        "XLF_Daily_Return": xlf,
        "XLK_Daily_Return": xlk,
        "XLE_Daily_Return": xle,
        "SPY_Daily_Return": spy,
    })
    path = root / "data/processed/sector_features.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)

    signal_dates = pd.date_range(dates[60], dates[-25], freq=pd.offsets.MonthEnd())
    # Map calendar month-end signals to dates that can be weekends; execution
    # logic must still find the first later available trading observation.
    rows = []
    patterns = [
        {"XLF": 0.45, "XLK": 0.35, "XLE": 0.00, "CASH": 0.20},
        {"XLF": 0.20, "XLK": 0.50, "XLE": 0.20, "CASH": 0.10},
        {"XLF": 0.00, "XLK": 0.40, "XLE": 0.40, "CASH": 0.20},
        {"XLF": 0.40, "XLK": 0.00, "XLE": 0.30, "CASH": 0.30},
    ]
    for i, signal in enumerate(signal_dates):
        weights = patterns[i % len(patterns)]
        for asset in ASSETS:
            rows.append({
                "Signal_Date": signal,
                "Strategy": "Week6_Optimized",
                "Asset": asset,
                "Target_Weight": weights[asset],
                "Selected": bool(asset != "CASH" and weights[asset] > 0),
            })
    out = root / "results/tables/week6_optimized_portfolio.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)


def _run(root: Path, rel: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / rel)], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )


def _assert_regressions(root: Path) -> None:
    daily = pd.read_csv(root / "results/tables/week7_daily_backtest.csv", parse_dates=["Date"])
    rebal = pd.read_csv(root / "results/tables/week7_rebalance_log.csv", parse_dates=["Execution_Date", "Signal_Date"])
    schedule = pd.read_csv(root / "results/tables/week7_execution_schedule.csv", parse_dates=["Execution_Date", "Signal_Date", "Effective_From_Date"])
    sensitivity = pd.read_csv(root / "results/tables/week7_cost_sensitivity.csv")
    benchmark = pd.read_csv(root / "results/tables/week7_benchmark_daily.csv")
    summary = pd.read_csv(root / "results/tables/week7_performance_summary.csv").set_index("Strategy")

    assert not daily.empty and not rebal.empty
    assert (schedule["Execution_Date"] > schedule["Signal_Date"]).all()
    assert schedule["Effective_From_Date"].notna().all()
    assert (schedule["Effective_From_Date"] > schedule["Execution_Date"]).all()

    # The first day is the initial execution date. The portfolio begins in cash,
    # so there is no pre-trade market exposure; only trading friction can move equity.
    first = daily.iloc[0]
    assert abs(float(first["Gross_Return"])) < 1e-12
    assert bool(first["Rebalanced"])
    assert float(first["Net_Return"]) <= 0.0

    # Costs must be charged on risky ETF trades, not on the cash leg itself.
    first_trade = rebal.iloc[0]
    expected = float(first_trade["Risky_Trade_Notional"]) * (
        float(first_trade["Transaction_Cost_Bps"]) + float(first_trade["Slippage_Bps"])
    ) / 10000.0
    assert math.isclose(float(first_trade["Total_Trading_Cost_Fraction"]), expected, abs_tol=1e-12)


    # Benchmarks must not earn the first execution day's close-to-close return.
    assert abs(float(benchmark.iloc[0]["SPY"])) < 1e-12
    assert abs(float(benchmark.iloc[0]["Equal_Weighted_Sector_Return"])) < 1e-12
    assert bool(benchmark.iloc[0]["Equal_Weighted_Rebalanced"])
    # Cost sensitivity should be monotone in terminal total return.
    ordered = sensitivity.sort_values("Total_Friction_Bps_Per_Risky_Dollar_Traded")
    diffs = np.diff(ordered["Total_Return"].to_numpy(dtype=float))
    assert (diffs <= 1e-10).all()

    # All three required comparisons should exist and include Week 7 metrics.
    for name in ("Week7_Strategy_Net", "SPY_Benchmark", "Equal_Weighted_Sectors"):
        assert name in summary.index
        for col in ("CAGR", "Sharpe_0RF", "Sortino_0MAR", "Max_Drawdown", "Daily_Win_Rate"):
            assert col in summary.columns

    # Equity must equal compounded net return exactly up to floating-point tolerance.
    recomputed = (1.0 + daily["Net_Return"].astype(float)).cumprod()
    assert np.allclose(recomputed, daily["Equity"].astype(float), atol=2e-8, rtol=2e-8)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="week7_smoke_") as tmp:
        root = Path(tmp)
        _copy_code(root)
        _make_data(root)
        result = _run(root, "run_week7.py")
        print(result.stdout, end="")
        if result.returncode != 0:
            raise RuntimeError("Week 7 isolated pipeline failed.")
        _assert_regressions(root)
        print("[PASS] Conservative execution/effective timing and initial-deployment accounting passed.")
        print("[PASS] Risky-trade cost accounting and cost-sensitivity monotonicity passed.")
        print("[PASS] Benchmark timing fairness, metrics, and compounded-equity regression checks passed.")
        print("[PASS] Week 7 smoke test completed in an isolated temporary repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
