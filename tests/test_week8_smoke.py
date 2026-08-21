"""Smoke test for the Week 8 reporting and reproducibility pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _copy_week8_code(destination: Path) -> None:
    files = [
        "run_week8.py",
        "src/reporting/week8_utils.py",
        "src/reporting/build_week8_report.py",
        "src/validation/validate_week8.py",
        "tests/test_week8_smoke.py",
        "docs/week8_methodology.md",
    ]
    for relative in files:
        src = PROJECT_ROOT / relative
        dst = destination / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # Minimal prior-stage files make the module inventory representative.
    (destination / "run_week7.py").write_text("print('week7 placeholder')\n", encoding="utf-8")
    (destination / "docs" / "week7_methodology.md").write_text("# Week 7\n", encoding="utf-8")


def _write_inputs(root: Path) -> None:
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (root / "data" / "backtesting").mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(
        [
            ["Week7_Strategy_Net", 252, 0.12, 0.12, 0.10, 1.20, 1.80, -0.08, 0.52, 12, 3.0, 0.25, 3.0, 0.004],
            ["SPY_Benchmark", 252, 0.09, 0.09, 0.13, 0.75, 1.05, -0.14, 0.51, 0, 0.0, 0.0, 0.0, 0.0],
            ["Equal_Weighted_Sectors", 252, 0.11, 0.11, 0.12, 0.95, 1.35, -0.11, 0.51, 0, 0.0, 0.0, 0.0, 0.0],
        ],
        columns=[
            "Strategy", "Daily_Observations", "Total_Return", "CAGR", "Annualized_Volatility", "Sharpe_0RF",
            "Sortino_0MAR", "Max_Drawdown", "Daily_Win_Rate", "Rebalance_Count", "Total_One_Way_Turnover",
            "Average_One_Way_Turnover", "Annualized_One_Way_Turnover", "Total_Trading_Cost_Fraction",
        ],
    )
    summary.to_csv(tables / "week7_performance_summary.csv", index=False)

    dates = pd.bdate_range("2025-01-02", periods=80)
    returns = np.linspace(-0.002, 0.003, len(dates))
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.r_[1.0, equity])[:-1]
    drawdown = equity / np.maximum(1.0, np.maximum.accumulate(equity)) - 1.0
    daily = pd.DataFrame({"Date": dates, "Net_Return": returns, "Equity": equity, "Drawdown": drawdown})
    daily.to_csv(tables / "week7_daily_backtest.csv", index=False)

    spy = np.linspace(-0.0015, 0.0025, len(dates))
    eq = np.linspace(-0.001, 0.0028, len(dates))
    bench = pd.DataFrame(
        {
            "Date": dates,
            "SPY": spy,
            "SPY_Equity": np.cumprod(1.0 + spy),
            "SPY_Drawdown": 0.0,
            "Equal_Weighted_Sector_Return": eq,
            "Equal_Weighted_Sectors_Equity": np.cumprod(1.0 + eq),
            "Equal_Weighted_Sectors_Drawdown": 0.0,
            "Equal_Weight_Rebalanced": False,
        }
    )
    bench.to_csv(tables / "week7_benchmark_daily.csv", index=False)

    rolling = pd.DataFrame(
        {
            "Window_Start": ["2025-01-02", "2025-02-03", "2025-03-03"],
            "Window_End": ["2025-03-31", "2025-04-30", "2025-05-30"],
            "Observations": [60, 60, 60],
            "Total_Return": [0.04, 0.03, -0.01],
            "CAGR": [0.18, 0.13, -0.04],
            "Sharpe_0RF": [1.4, 1.0, -0.3],
            "Sortino_0MAR": [2.0, 1.5, -0.4],
            "Max_Drawdown": [-0.04, -0.05, -0.09],
            "Daily_Win_Rate": [0.53, 0.52, 0.48],
        }
    )
    rolling.to_csv(tables / "week7_rolling_validation.csv", index=False)

    cost = pd.DataFrame(
        {
            "Total_Friction_Bps_Per_Risky_Dollar_Traded": [0.0, 10.0, 40.0],
            "Transaction_Cost_Bps": [0.0, 5.0, 20.0],
            "Slippage_Bps": [0.0, 5.0, 20.0],
            "Total_Return": [0.13, 0.12, 0.095],
            "CAGR": [0.13, 0.12, 0.095],
            "Sharpe_0RF": [1.28, 1.20, 0.96],
            "Sortino_0MAR": [1.9, 1.8, 1.4],
            "Max_Drawdown": [-0.075, -0.08, -0.09],
            "Total_Trading_Cost_Fraction": [0.0, 0.004, 0.016],
        }
    )
    cost.to_csv(tables / "week7_cost_sensitivity.csv", index=False)

    rebalance = pd.DataFrame(
        {
            "Signal_Date": ["2025-01-31", "2025-02-28", "2025-03-31"],
            "Execution_Date": ["2025-02-03", "2025-03-03", "2025-04-01"],
            "One_Way_Turnover": [0.8, 0.3, 0.4],
            "Total_Trading_Cost_Fraction": [0.0008, 0.0003, 0.0004],
        }
    )
    rebalance.to_csv(tables / "week7_rebalance_log.csv", index=False)

    week6 = pd.DataFrame(
        {
            "Signal_Date": ["2025-03-31"] * 4,
            "Strategy": ["Week6_Optimized"] * 4,
            "Asset": ["XLF", "XLK", "XLE", "CASH"],
            "Target_Weight": [0.30, 0.40, 0.10, 0.20],
            "Selected": [True, True, True, False],
        }
    )
    week6.to_csv(tables / "week6_optimized_portfolio.csv", index=False)

    config = {"transaction_cost_bps": 5.0, "slippage_bps": 5.0, "annualization_days": 252}
    (root / "data" / "backtesting" / "week7_backtest_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="week8_smoke_") as tmp:
        root = Path(tmp)
        _copy_week8_code(root)
        _write_inputs(root)

        result = subprocess.run([sys.executable, "run_week8.py"], cwd=root, text=True, capture_output=True)
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise AssertionError("Week 8 end-to-end smoke pipeline failed.")

        scorecard = pd.read_csv(root / "results" / "tables" / "week8_final_scorecard.csv")
        cagr = scorecard.loc[scorecard["Metric"] == "CAGR"].iloc[0]
        assert np.isclose(cagr["Strategy_minus_SPY"], 0.03)
        assert np.isclose(cagr["Strategy_minus_EqualWeight"], 0.01)

        allocation = pd.read_csv(root / "results" / "tables" / "week8_latest_allocation.csv")
        assert np.isclose(allocation["Target_Weight"].sum(), 1.0)

        robust = pd.read_csv(root / "results" / "tables" / "week8_robustness_summary.csv")
        values = dict(zip(robust["Measure"], robust["Value"]))
        assert np.isclose(values["Positive_Rolling_Sharpe_Share"], 2.0 / 3.0)
        assert np.isclose(values["Return_Drag_Zero_to_Highest_Friction"], 0.035)

        dashboard = (root / "results" / "dashboard" / "week8_dashboard.html").read_text(encoding="utf-8")
        assert "Week 8 — Final Analysis Dashboard" in dashboard
        assert "week8_latest_allocation.png" in dashboard

        manifest = json.loads((root / "data" / "reporting" / "week8_reproducibility_manifest.json").read_text(encoding="utf-8"))
        assert manifest["reproduction_command"] == "python3 run_week8.py"
        assert manifest["input_sha256"]

    print("[PASS] Week 8 scorecard and benchmark deltas passed known-value checks.")
    print("[PASS] Robustness, allocation, dashboard, and reproducibility outputs passed smoke checks.")
    print("[PASS] Week 8 smoke test completed in an isolated temporary repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
