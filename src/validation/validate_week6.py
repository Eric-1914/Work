"""Validation checks for Week 6 risk-management and optimization outputs."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization.optimize_week6 import (  # noqa: E402
    ASSETS,
    CASH,
    DAILY_FEATURES_PATH,
    DEFENSIVE_GROSS_EXPOSURE,
    DRAWDOWN_TRIGGER,
    MAX_ONE_WAY_TURNOVER,
    MAX_SECTOR_WEIGHT,
    MIN_LOOKBACK_DAYS,
    OPT_STRATEGY,
    SECTORS,
    TABLE_DIR,
    TARGET_VOLATILITY,
    WEEK5_PATH,
    build_constraint_audit,
    build_daily_performance,
    build_rebalance_trades,
    build_risk_metrics,
    build_week6_weights,
    load_daily_returns,
    load_week5_portfolio,
)


REQUIRED_FILES = [
    TABLE_DIR / "week6_mvo_weights.csv",
    TABLE_DIR / "week6_inverse_vol_weights.csv",
    TABLE_DIR / "week6_optimized_portfolio.csv",
    TABLE_DIR / "week6_optimization_diagnostics.csv",
    TABLE_DIR / "week6_rebalance_trades.csv",
    TABLE_DIR / "week6_constraint_audit.csv",
    TABLE_DIR / "week6_daily_performance.csv",
    TABLE_DIR / "week6_risk_metrics.csv",
    PROJECT_ROOT / "results" / "figures" / "week6_equity_curve.png",
    PROJECT_ROOT / "results" / "figures" / "week6_drawdown_comparison.png",
    PROJECT_ROOT / "results" / "figures" / "week6_optimized_weights.png",
    PROJECT_ROOT / "results" / "figures" / "week6_turnover_control.png",
    PROJECT_ROOT / "data" / "strategy" / "week6_risk_config.json",
    PROJECT_ROOT / "docs" / "week6_risk_report.md",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_files() -> None:
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Week 6 outputs: {missing}")
    if not WEEK5_PATH.exists():
        raise FileNotFoundError("Week 5 handoff is missing.")
    if not DAILY_FEATURES_PATH.exists():
        raise FileNotFoundError("Week 2 daily return source is missing.")


def _read_weights(filename: str) -> pd.DataFrame:
    data = pd.read_csv(TABLE_DIR / filename, parse_dates=["Signal_Date"])
    required = {
        "Signal_Date",
        "Strategy",
        "Asset",
        "Target_Weight",
        "Selected",
        "Outperformance_Probability",
        "Sector_Rank",
        "Macro_Regime",
        "model",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")
    return data


def check_weight_integrity(data: pd.DataFrame, expected_strategy: str) -> None:
    _assert(set(data["Strategy"].unique()) == {expected_strategy}, f"Unexpected strategy in {expected_strategy} weights.")
    weights = pd.to_numeric(data["Target_Weight"], errors="coerce")
    _assert(weights.notna().all(), f"{expected_strategy} has nonnumeric weights.")
    _assert(np.isfinite(weights.to_numpy(dtype=float)).all(), f"{expected_strategy} has nonfinite weights.")
    _assert((weights >= -1e-12).all(), f"{expected_strategy} contains a short position.")

    for date, group in data.groupby("Signal_Date", sort=True):
        _assert(set(group["Asset"]) == set(ASSETS), f"{expected_strategy} missing an asset on {date.date()}.")
        _assert(len(group) == len(ASSETS), f"{expected_strategy} has duplicate assets on {date.date()}.")
        _assert(abs(float(group["Target_Weight"].sum()) - 1.0) <= 1e-8, f"{expected_strategy} weights do not sum to 1 on {date.date()}.")
        sector = group[group["Asset"].isin(SECTORS)]
        _assert(float(sector["Target_Weight"].max()) <= MAX_SECTOR_WEIGHT + 1e-9, f"{expected_strategy} breaks sector cap on {date.date()}.")


def check_signal_eligibility(weights: pd.DataFrame, week5: pd.DataFrame) -> None:
    for date, month in weights.groupby("Signal_Date", sort=True):
        source = week5[week5["Signal_Date"] == date]
        eligible = set(source.loc[(source["Asset"].isin(SECTORS)) & (source["Target_Weight"] > 1e-10), "Asset"])
        held = set(month.loc[(month["Asset"].isin(SECTORS)) & (month["Target_Weight"] > 1e-10), "Asset"])
        _assert(held.issubset(eligible), f"Week 6 added a sector not selected by Week 5 on {date.date()}.")


def check_constraints(final: pd.DataFrame, diagnostics: pd.DataFrame, audit: pd.DataFrame) -> None:
    _assert(bool(audit["All_Constraints_OK"].all()), "Saved Week 6 constraint audit contains a failure.")
    diag = diagnostics.set_index("Signal_Date")

    previous = {sector: 0.0 for sector in SECTORS}
    previous[CASH] = 1.0
    for date, month in final.groupby("Signal_Date", sort=True):
        current = month.set_index("Asset")["Target_Weight"].astype(float).to_dict()
        turnover = 0.5 * sum(abs(current[a] - previous[a]) for a in ASSETS)
        saved_turnover = float(diag.loc[date, "Final_One_Way_Turnover"])
        _assert(abs(turnover - saved_turnover) <= 1e-9, f"Turnover mismatch on {date.date()}.")

        exception = bool(diag.loc[date, "Turnover_Limit_Exception"])
        limit_applied = bool(diag.loc[date, "Turnover_Limit_Applied"])
        _assert((not limit_applied) or turnover <= MAX_ONE_WAY_TURNOVER + 1e-9 or exception, f"Turnover limit violated without risk exception on {date.date()}.")

        final_vol = float(diag.loc[date, "Final_Forecast_Vol"])
        _assert(final_vol <= TARGET_VOLATILITY + 1e-9, f"Volatility target failed on {date.date()}.")

        drawdown = float(diag.loc[date, "Pre_Rebalance_Drawdown"])
        gross = sum(current[s] for s in SECTORS)
        if drawdown <= DRAWDOWN_TRIGGER + 1e-9:
            _assert(gross <= DEFENSIVE_GROSS_EXPOSURE + 1e-9, f"Defensive exposure cap failed on {date.date()}.")

        lookback_end = pd.Timestamp(diag.loc[date, "Lookback_End"])
        _assert(lookback_end <= date, f"Look-ahead detected in risk window on {date.date()}.")
        _assert(int(diag.loc[date, "Lookback_Observations"]) >= MIN_LOOKBACK_DAYS, f"Insufficient risk observations on {date.date()}.")
        previous = current


def check_recomputed_outputs(
    week5: pd.DataFrame,
    daily_returns: pd.DataFrame,
    saved_mvo: pd.DataFrame,
    saved_rp: pd.DataFrame,
    saved_final: pd.DataFrame,
    saved_diag: pd.DataFrame,
) -> None:
    mvo, rp, final, diagnostics = build_week6_weights(week5, daily_returns)

    for label, expected, actual in [
        ("MVO", mvo, saved_mvo),
        ("inverse volatility", rp, saved_rp),
        ("optimized", final, saved_final),
    ]:
        e = expected.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)
        a = actual.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)
        _assert(list(e["Signal_Date"]) == list(a["Signal_Date"]), f"{label} signal dates differ from recomputation.")
        _assert(list(e["Asset"]) == list(a["Asset"]), f"{label} assets differ from recomputation.")
        _assert(np.allclose(e["Target_Weight"], a["Target_Weight"], atol=1e-10), f"{label} weights differ from recomputation.")

    e = diagnostics.sort_values("Signal_Date").reset_index(drop=True)
    a = saved_diag.sort_values("Signal_Date").reset_index(drop=True)
    numeric_cols = [
        "MVO_Objective",
        "MVO_Forecast_Vol",
        "Inverse_Vol_PreTarget_Vol",
        "Inverse_Vol_Scale",
        "Blended_PreTarget_Vol",
        "Blended_Vol_Scale",
        "PostTurnover_PreRisk_Vol",
        "PostTurnover_Vol_Scale",
        "Pre_Rebalance_Drawdown",
        "Raw_One_Way_Turnover",
        "Turnover_Scale",
        "Final_One_Way_Turnover",
        "Final_Gross_Exposure",
        "Final_Forecast_Vol",
    ]
    for col in numeric_cols:
        _assert(np.allclose(e[col], a[col], atol=1e-10, equal_nan=True), f"Diagnostic column {col} differs from recomputation.")


def check_tables_match_recomputation(
    week5: pd.DataFrame,
    daily_returns: pd.DataFrame,
    mvo: pd.DataFrame,
    rp: pd.DataFrame,
    final: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    expected_trades = build_rebalance_trades(final, diagnostics)
    saved_trades = pd.read_csv(TABLE_DIR / "week6_rebalance_trades.csv", parse_dates=["Signal_Date"])
    e = expected_trades.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)
    a = saved_trades.sort_values(["Signal_Date", "Asset"]).reset_index(drop=True)
    _assert(np.allclose(e["Trade_Weight"], a["Trade_Weight"], atol=1e-10), "Saved rebalance trades differ from recomputation.")
    _assert(np.allclose(e["Turnover_Contribution"], a["Turnover_Contribution"], atol=1e-10), "Saved turnover contributions differ from recomputation.")

    expected_audit = build_constraint_audit(final, diagnostics)
    saved_audit = pd.read_csv(TABLE_DIR / "week6_constraint_audit.csv", parse_dates=["Signal_Date"])
    _assert(list(expected_audit["All_Constraints_OK"]) == list(saved_audit["All_Constraints_OK"]), "Constraint audit differs from recomputation.")

    expected_perf = build_daily_performance(week5, mvo, rp, final, daily_returns)
    saved_perf = pd.read_csv(TABLE_DIR / "week6_daily_performance.csv", parse_dates=["Date"])
    ep = expected_perf.sort_values(["Date", "Strategy"]).reset_index(drop=True)
    ap = saved_perf.sort_values(["Date", "Strategy"]).reset_index(drop=True)
    _assert(list(ep["Date"]) == list(ap["Date"]), "Daily performance dates differ from recomputation.")
    _assert(list(ep["Strategy"]) == list(ap["Strategy"]), "Daily performance strategy labels differ from recomputation.")
    _assert(np.allclose(ep["Daily_Return"], ap["Daily_Return"], atol=1e-12), "Daily performance returns differ from recomputation.")

    expected_metrics = build_risk_metrics(expected_perf, week5, mvo, rp, final)
    saved_metrics = pd.read_csv(TABLE_DIR / "week6_risk_metrics.csv")
    em = expected_metrics.sort_values("Strategy").reset_index(drop=True)
    am = saved_metrics.sort_values("Strategy").reset_index(drop=True)
    _assert(list(em["Strategy"]) == list(am["Strategy"]), "Risk metric strategies differ from recomputation.")
    for col in ["Total_Return", "Annualized_Return", "Annualized_Volatility", "Sharpe_0RF", "Max_Drawdown", "Average_Gross_Exposure", "Average_One_Way_Turnover"]:
        _assert(np.allclose(em[col], am[col], atol=1e-10, equal_nan=True), f"Risk metric {col} differs from recomputation.")


def check_config() -> None:
    config_path = PROJECT_ROOT / "data" / "strategy" / "week6_risk_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _assert(config.get("week") == 6, "Week 6 config has wrong week number.")
    constraints = config.get("constraints", {})
    _assert(abs(float(constraints.get("max_sector_weight")) - MAX_SECTOR_WEIGHT) <= 1e-12, "Config sector cap mismatch.")
    _assert(abs(float(constraints.get("max_one_way_turnover_per_rebalance")) - MAX_ONE_WAY_TURNOVER) <= 1e-12, "Config turnover limit mismatch.")
    _assert(abs(float(constraints.get("drawdown_trigger")) - DRAWDOWN_TRIGGER) <= 1e-12, "Config drawdown trigger mismatch.")


def main() -> int:
    check_files()
    week5 = load_week5_portfolio()
    daily_returns = load_daily_returns()
    mvo = _read_weights("week6_mvo_weights.csv")
    rp = _read_weights("week6_inverse_vol_weights.csv")
    final = _read_weights("week6_optimized_portfolio.csv")
    diagnostics = pd.read_csv(TABLE_DIR / "week6_optimization_diagnostics.csv", parse_dates=["Signal_Date", "Lookback_Start", "Lookback_End"])
    audit = pd.read_csv(TABLE_DIR / "week6_constraint_audit.csv", parse_dates=["Signal_Date"])

    check_weight_integrity(mvo, "MVO_Constrained")
    check_weight_integrity(rp, "Inverse_Volatility_Targeted")
    check_weight_integrity(final, OPT_STRATEGY)
    check_signal_eligibility(mvo, week5)
    check_signal_eligibility(rp, week5)
    check_signal_eligibility(final, week5)
    check_constraints(final, diagnostics, audit)
    check_recomputed_outputs(week5, daily_returns, mvo, rp, final, diagnostics)
    check_tables_match_recomputation(week5, daily_returns, mvo, rp, final, diagnostics)
    check_config()

    print("[PASS] Required Week 6 files and Week 2/Week 5 interfaces are valid.")
    print("[PASS] MVO, inverse-volatility, and optimized weights were independently recomputed.")
    print("[PASS] Long-only, sector-cap, turnover, and drawdown constraints passed.")
    print("[PASS] Every risk-estimation window ends on or before its signal date.")
    print("[PASS] Rebalance trades, daily performance, and risk metrics match recomputation.")
    print("[DONE] All Week 6 outputs passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
