"""End-to-end Week 6 smoke test in an isolated temporary repository."""

from __future__ import annotations

from pathlib import Path
import shutil
import importlib.util
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SECTORS = ["XLF", "XLK", "XLE"]


def make_daily_features() -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    dates = pd.bdate_range("2024-01-02", periods=430)
    n = len(dates)

    # Correlated, realistic-scale daily returns plus an intentional stress episode.
    common = rng.normal(0.00025, 0.0070, n)
    returns = {}
    for i, sector in enumerate(SECTORS):
        idio = rng.normal(0.00005 * (i - 1), 0.0045 + i * 0.0005, n)
        returns[sector] = common + idio

    # Force a material drawdown while XLF / XLK are selected so the defensive
    # branch is exercised by the smoke test.
    stress_start, stress_end = 250, 268
    returns["XLF"][stress_start:stress_end] -= 0.028
    returns["XLK"][stress_start:stress_end] -= 0.025
    returns["XLE"][stress_start:stress_end] -= 0.010

    data = pd.DataFrame({"Date": dates})
    for sector in SECTORS:
        data[f"{sector}_Daily_Return"] = returns[sector]
    return data


def make_week5_portfolio(daily: pd.DataFrame) -> pd.DataFrame:
    # Start after enough history exists and use actual business dates.
    candidate_dates = daily.loc[80:, "Date"].dt.to_period("M")
    month_end_indices = (
        daily.loc[80:].assign(Month=candidate_dates)
        .groupby("Month", as_index=False)
        .tail(1)
        .index
    )
    signal_dates = daily.loc[month_end_indices, "Date"].iloc[:12].tolist()

    rows = []
    patterns = [
        ("XLF", "XLK"),
        ("XLK", "XLE"),
        ("XLF",),
        ("XLF", "XLE"),
        (),
        ("XLF", "XLK"),
    ]
    for idx, date in enumerate(signal_dates):
        selected = patterns[idx % len(patterns)]
        probs = {"XLF": 0.58, "XLK": 0.64, "XLE": 0.54}
        if selected == ("XLK", "XLE"):
            probs = {"XLF": 0.42, "XLK": 0.68, "XLE": 0.59}
        elif selected == ("XLF", "XLE"):
            probs = {"XLF": 0.66, "XLK": 0.45, "XLE": 0.57}

        ordered = sorted(SECTORS, key=lambda s: (-probs[s], s))
        ranks = {sector: rank for rank, sector in enumerate(ordered, start=1)}
        raw = {sector: (probs[sector] if sector in selected else 0.0) for sector in SECTORS}
        total = sum(raw.values())
        if len(selected) == 1:
            only = selected[0]
            weights = {sector: (0.60 if sector == only else 0.0) for sector in SECTORS}
        elif total > 0:
            weights = {sector: raw[sector] / total for sector in SECTORS}
        else:
            weights = {sector: 0.0 for sector in SECTORS}
        cash = max(0.0, 1.0 - sum(weights.values()))

        for asset in [*SECTORS, "CASH"]:
            if asset == "CASH":
                probability = np.nan
                rank = np.nan
                rank_score = np.nan
                selected_flag = cash > 1e-10
                weight = cash
            else:
                probability = probs[asset]
                rank = ranks[asset]
                rank_score = 1.0 - (rank - 1) / 2.0
                selected_flag = asset in selected
                weight = weights[asset] if selected_flag else 0.0
            rows.append(
                {
                    "Signal_Date": date,
                    "Strategy": "Combined_Default",
                    "Asset": asset,
                    "Target_Weight": weight,
                    "Selected": selected_flag,
                    "Outperformance_Probability": probability,
                    "Sector_Rank": rank,
                    "Rank_Score_0_1": rank_score,
                    "Macro_Regime": "Growth=Stable|Inflation=Moderate|Rate=Stable",
                    "model": "Regime_Switching_Ensemble",
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="week6_smoke_") as temp_dir:
        root = Path(temp_dir)
        for relative in [
            "run_week6.py",
            "src/optimization/optimize_week6.py",
            "src/validation/validate_week6.py",
            "docs/week6_methodology.md",
        ]:
            source = SOURCE_ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        daily = make_daily_features()
        feature_path = root / "data" / "processed" / "sector_features.csv"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(feature_path, index=False)

        week5 = make_week5_portfolio(daily)
        week5_path = root / "results" / "tables" / "week5_default_portfolio.csv"
        week5_path.parent.mkdir(parents=True, exist_ok=True)
        week5.to_csv(week5_path, index=False)

        result = subprocess.run(
            [sys.executable, str(root / "run_week6.py")],
            cwd=root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode

        optimized = pd.read_csv(root / "results" / "tables" / "week6_optimized_portfolio.csv")
        mvo = pd.read_csv(root / "results" / "tables" / "week6_mvo_weights.csv")
        inv = pd.read_csv(root / "results" / "tables" / "week6_inverse_vol_weights.csv")
        performance = pd.read_csv(root / "results" / "tables" / "week6_daily_performance.csv")
        diagnostics = pd.read_csv(root / "results" / "tables" / "week6_optimization_diagnostics.csv")
        audit = pd.read_csv(root / "results" / "tables" / "week6_constraint_audit.csv")
        metrics = pd.read_csv(root / "results" / "tables" / "week6_risk_metrics.csv")

        if not audit["All_Constraints_OK"].astype(bool).all():
            raise AssertionError("Smoke test found a failed Week 6 constraint audit row.")

        sector = optimized[optimized["Asset"].isin(SECTORS)]
        if float(sector["Target_Weight"].max()) > 0.600000001:
            raise AssertionError("Sector cap was violated in smoke test.")

        sums = optimized.groupby("Signal_Date")["Target_Weight"].sum()
        if not np.allclose(sums.to_numpy(dtype=float), 1.0, atol=1e-9):
            raise AssertionError("Optimized weights do not sum to one.")

        if not diagnostics["Defensive_Overlay_Active"].astype(bool).any():
            raise AssertionError("Stress scenario did not exercise the drawdown defensive overlay.")

        normal = diagnostics[
            diagnostics["Turnover_Limit_Applied"].astype(bool)
            & ~diagnostics["Turnover_Limit_Exception"].astype(bool)
        ]
        if (normal["Final_One_Way_Turnover"] > 0.500000001).any():
            raise AssertionError("Normal turnover limit was violated.")

        first_date = diagnostics.sort_values("Signal_Date").iloc[0]
        if bool(first_date["Turnover_Limit_Applied"]):
            raise AssertionError("Initial deployment should not be treated as a rebalance.")

        # MVO must preserve the Week 5 risky budget before later risk scaling;
        # inverse-volatility may only preserve or reduce it through volatility targeting.
        week5_gross = (
            week5[week5["Asset"].isin(SECTORS)]
            .groupby("Signal_Date")["Target_Weight"]
            .sum()
        )
        for table, exact in [(mvo, True), (inv, False)]:
            gross = table[table["Asset"].isin(SECTORS)].groupby("Signal_Date")["Target_Weight"].sum()
            source = week5_gross.reindex(pd.to_datetime(gross.index)).to_numpy(dtype=float)
            values = gross.to_numpy(dtype=float)
            if exact and not np.allclose(values, source, atol=1e-9):
                raise AssertionError("MVO changed the Week 5 risky budget before risk scaling.")
            if (values > source + 1e-9).any():
                raise AssertionError("Week 6 created gross exposure above the Week 5 risky budget.")

        # Independently verify the saved Sharpe ratio definition.
        opt_returns = performance.loc[performance["Strategy"] == "Week6_Optimized", "Daily_Return"].astype(float)
        std = float(opt_returns.std(ddof=1))
        expected_sharpe = float(opt_returns.mean()) / std * np.sqrt(252) if std > 1e-12 else np.nan
        saved_sharpe = float(metrics.loc[metrics["Strategy"] == "Week6_Optimized", "Sharpe_0RF"].iloc[0])
        if not np.isclose(expected_sharpe, saved_sharpe, atol=1e-10, equal_nan=True):
            raise AssertionError("Saved Sharpe ratio does not match the standard daily-return formula.")

        # Regression test for the original turnover bug: a mandatory sale can be
        # reallocated directly without unnecessarily parking the proceeds in cash.
        spec = importlib.util.spec_from_file_location("week6_opt_test", root / "src" / "optimization" / "optimize_week6.py")
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        previous = {"XLF": 0.60, "XLK": 0.40, "XLE": 0.0, "CASH": 0.0}
        proposed = {"XLF": 0.0, "XLK": 0.40, "XLE": 0.60, "CASH": 0.0}
        limited, _, _, exception, _ = module.apply_turnover_limit(previous, proposed, ["XLK", "XLE"])
        if not exception or not np.isclose(limited["XLE"], 0.60, atol=1e-9) or limited["CASH"] > 1e-9:
            raise AssertionError("Mandatory-exit turnover handling stranded reusable proceeds in cash.")

        if set(metrics["Strategy"]) != {
            "Week5_Baseline",
            "MVO_Constrained",
            "Inverse_Volatility_Targeted",
            "Week6_Optimized",
        }:
            raise AssertionError("Risk metrics are missing a strategy comparison row.")

        print(result.stdout.strip())
        print("[PASS] Stress scenario triggered the drawdown defensive overlay.")
        print("[PASS] Sector caps, weight sums, initial deployment, and normal turnover limits passed edge checks.")
        print("[PASS] Week 5 risk budget, Sharpe calculation, and mandatory-exit reallocation passed regression checks.")
        print("[PASS] Week 6 smoke test completed in an isolated temporary repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
