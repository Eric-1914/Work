"""Independent validation for Week 8 reporting and reproducibility outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.week8_utils import (  # noqa: E402
    SCORECARD_METRICS,
    Week8Paths,
    read_csv,
    sha256_file,
)

PATHS = Week8Paths(PROJECT_ROOT)

REQUIRED_OUTPUTS = [
    PATHS.tables_dir / "week8_final_scorecard.csv",
    PATHS.tables_dir / "week8_robustness_summary.csv",
    PATHS.tables_dir / "week8_latest_allocation.csv",
    PATHS.tables_dir / "week8_data_quality_audit.csv",
    PATHS.tables_dir / "week8_module_inventory.csv",
    PATHS.figures_dir / "week8_cagr_comparison.png",
    PATHS.figures_dir / "week8_risk_adjusted_comparison.png",
    PATHS.figures_dir / "week8_rolling_sharpe.png",
    PATHS.figures_dir / "week8_cost_robustness.png",
    PATHS.figures_dir / "week8_latest_allocation.png",
    PATHS.dashboard_dir / "week8_dashboard.html",
    PATHS.reporting_dir / "week8_reporting_config.json",
    PATHS.reporting_dir / "week8_reproducibility_manifest.json",
    PATHS.docs_dir / "week8_final_report.md",
    PATHS.docs_dir / "week8_project_documentation.md",
]


def _assert_required_outputs() -> None:
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_OUTPUTS if not path.exists()]
    if missing:
        raise AssertionError(f"Missing Week 8 outputs: {missing}")
    empty = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_OUTPUTS if path.stat().st_size <= 0]
    if empty:
        raise AssertionError(f"Empty Week 8 outputs: {empty}")


def _validate_scorecard() -> None:
    source = read_csv(PATHS.week7_summary, "Week 7 performance summary").set_index("Strategy")
    saved = read_csv(PATHS.tables_dir / "week8_final_scorecard.csv", "Week 8 scorecard")
    if saved["Metric"].tolist() != SCORECARD_METRICS:
        raise AssertionError("Week 8 scorecard metric order/schema changed unexpectedly.")
    for _, row in saved.iterrows():
        metric = row["Metric"]
        for strategy in ["Week7_Strategy_Net", "SPY_Benchmark", "Equal_Weighted_Sectors"]:
            expected = float(source.loc[strategy, metric])
            actual = float(row[strategy])
            if not np.isclose(expected, actual, atol=1e-12, rtol=1e-10):
                raise AssertionError(f"Scorecard mismatch for {strategy} / {metric}: {actual} vs {expected}")
        if not np.isclose(
            float(row["Strategy_minus_SPY"]),
            float(row["Week7_Strategy_Net"]) - float(row["SPY_Benchmark"]),
            atol=1e-12,
        ):
            raise AssertionError(f"Strategy-minus-SPY mismatch for {metric}")
        if not np.isclose(
            float(row["Strategy_minus_EqualWeight"]),
            float(row["Week7_Strategy_Net"]) - float(row["Equal_Weighted_Sectors"]),
            atol=1e-12,
        ):
            raise AssertionError(f"Strategy-minus-equal-weight mismatch for {metric}")


def _validate_allocation() -> None:
    source = read_csv(PATHS.week6_portfolio, "Week 6 optimized portfolio")
    source["Signal_Date"] = pd.to_datetime(source["Signal_Date"])
    latest_date = source["Signal_Date"].max()
    expected = source.loc[source["Signal_Date"] == latest_date, ["Asset", "Target_Weight"]].copy()
    expected = expected.sort_values("Asset").reset_index(drop=True)

    saved = read_csv(PATHS.tables_dir / "week8_latest_allocation.csv", "Week 8 latest allocation")
    actual = saved[["Asset", "Target_Weight"]].sort_values("Asset").reset_index(drop=True)
    if expected["Asset"].tolist() != actual["Asset"].tolist():
        raise AssertionError("Week 8 latest allocation assets do not match Week 6.")
    if not np.allclose(expected["Target_Weight"], actual["Target_Weight"], atol=1e-12):
        raise AssertionError("Week 8 latest allocation weights do not match Week 6.")
    if not np.isclose(actual["Target_Weight"].sum(), 1.0, atol=1e-10):
        raise AssertionError("Week 8 latest allocation does not sum to one.")


def _validate_robustness() -> None:
    rolling = read_csv(PATHS.week7_rolling, "Week 7 rolling validation")
    cost = read_csv(PATHS.week7_cost, "Week 7 cost sensitivity").sort_values("Total_Friction_Bps_Per_Risky_Dollar_Traded")
    rebalance = read_csv(PATHS.week7_rebalance, "Week 7 rebalance log")
    saved = read_csv(PATHS.tables_dir / "week8_robustness_summary.csv", "Week 8 robustness summary")
    values = dict(zip(saved["Measure"], saved["Value"]))

    sharpe = pd.to_numeric(rolling["Sharpe_0RF"], errors="raise")
    cagr = pd.to_numeric(rolling["CAGR"], errors="raise")
    drawdown = pd.to_numeric(rolling["Max_Drawdown"], errors="raise")
    zero = cost.loc[cost["Total_Friction_Bps_Per_Risky_Dollar_Traded"].idxmin()]
    high = cost.loc[cost["Total_Friction_Bps_Per_Risky_Dollar_Traded"].idxmax()]
    expected = {
        "Rolling_Window_Count": float(len(rolling)),
        "Rolling_Sharpe_Median": float(sharpe.median()),
        "Rolling_Sharpe_Min": float(sharpe.min()),
        "Positive_Rolling_Sharpe_Share": float((sharpe > 0).mean()),
        "Positive_Rolling_CAGR_Share": float((cagr > 0).mean()),
        "Worst_Rolling_Max_Drawdown": float(drawdown.min()),
        "Zero_Friction_Total_Return": float(zero["Total_Return"]),
        "Highest_Friction_Total_Return": float(high["Total_Return"]),
        "Return_Drag_Zero_to_Highest_Friction": float(zero["Total_Return"] - high["Total_Return"]),
        "Total_One_Way_Turnover": float(pd.to_numeric(rebalance["One_Way_Turnover"], errors="raise").sum()),
        "Total_Trading_Cost_Fraction": float(pd.to_numeric(rebalance["Total_Trading_Cost_Fraction"], errors="raise").sum()),
    }
    for key, expected_value in expected.items():
        if key not in values:
            raise AssertionError(f"Missing robustness measure: {key}")
        if not np.isclose(float(values[key]), expected_value, atol=1e-12, rtol=1e-10):
            raise AssertionError(f"Robustness mismatch for {key}: {values[key]} vs {expected_value}")


def _validate_quality_and_dashboard() -> None:
    audit = read_csv(PATHS.tables_dir / "week8_data_quality_audit.csv", "Week 8 quality audit")
    passed = audit["Passed"].astype(str).str.lower().map({"true": True, "false": False})
    if passed.isna().any() or not passed.all():
        raise AssertionError("Week 8 data-quality audit contains failing checks.")

    dashboard = (PATHS.dashboard_dir / "week8_dashboard.html").read_text(encoding="utf-8")
    required_tokens = [
        "Week 8 — Final Analysis Dashboard",
        "week8_cagr_comparison.png",
        "week8_risk_adjusted_comparison.png",
        "week8_rolling_sharpe.png",
        "week8_cost_robustness.png",
        "week8_latest_allocation.png",
    ]
    missing = [token for token in required_tokens if token not in dashboard]
    if missing:
        raise AssertionError(f"Week 8 dashboard is missing expected content: {missing}")


def _validate_manifest() -> None:
    manifest_path = PATHS.reporting_dir / "week8_reproducibility_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("reproduction_command") != "python3 run_week8.py":
        raise AssertionError("Unexpected Week 8 reproduction command.")
    hashes = manifest.get("input_sha256", {})
    if not hashes:
        raise AssertionError("Week 8 reproducibility manifest has no input hashes.")
    for relative, saved_hash in hashes.items():
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise AssertionError(f"Manifest input no longer exists: {relative}")
        if sha256_file(path) != saved_hash:
            raise AssertionError(f"Manifest hash mismatch for input: {relative}")

    modules = read_csv(PATHS.tables_dir / "week8_module_inventory.csv", "Week 8 module inventory")
    required_module_paths = {"run_week8.py", "src/reporting/build_week8_report.py", "src/reporting/week8_utils.py", "src/validation/validate_week8.py"}
    found = set(modules["Path"].astype(str))
    if not required_module_paths.issubset(found):
        raise AssertionError(f"Module inventory is missing Week 8 modules: {sorted(required_module_paths - found)}")
    for _, row in modules.iterrows():
        path = PROJECT_ROOT / str(row["Path"])
        if not path.exists() or sha256_file(path) != str(row["SHA256"]):
            raise AssertionError(f"Module hash mismatch for {row['Path']}")


def main() -> int:
    try:
        _assert_required_outputs()
        _validate_scorecard()
        _validate_allocation()
        _validate_robustness()
        _validate_quality_and_dashboard()
        _validate_manifest()
        print("[PASS] Required Week 8 tables, figures, dashboard, reports, config, and manifest exist.")
        print("[PASS] Final scorecard values match the validated Week 7 performance summary exactly.")
        print("[PASS] Latest allocation and robustness statistics passed independent recomputation.")
        print("[PASS] Data-quality audit, dashboard references, and reproducibility hashes passed validation.")
        print("[PASS] Week 8 module inventory contains the reporting and validation code used for this run.")
        print("[DONE] All Week 8 outputs passed validation.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Week 8 validation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
