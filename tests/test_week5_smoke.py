"""End-to-end Week 5 smoke test using an isolated temporary repository."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pandas as pd


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def make_signals() -> pd.DataFrame:
    dates = pd.date_range("2025-01-31", periods=8, freq=pd.offsets.MonthEnd())
    probability_rows = [
        (0.75, 0.55, 0.35),
        (0.64, 0.61, 0.40),
        (0.49, 0.47, 0.44),
        (0.82, 0.46, 0.45),
        (0.58, 0.57, 0.56),
        (0.52, 0.51, 0.20),
        (0.70, 0.30, 0.60),
        (0.51, 0.50, 0.49),
    ]
    sectors = ["XLF", "XLK", "XLE"]
    rows = []
    for date, probs in zip(dates, probability_rows):
        pairs = sorted(zip(sectors, probs), key=lambda item: (-item[1], item[0]))
        rank_map = {sector: rank for rank, (sector, _) in enumerate(pairs, start=1)}
        for sector, probability in zip(sectors, probs):
            rank = rank_map[sector]
            rows.append(
                {
                    "Date": date,
                    "Sector": sector,
                    "Macro_Regime": "Growth=High_Growth|Inflation=Low_Inflation|Rate=Stable_Rates",
                    "model": "Random_Forest",
                    "Outperformance_Probability": probability,
                    "Predicted_Class": int(probability >= 0.5),
                    "Sector_Rank": rank,
                    "Rank_Score_0_1": 1.0 - (rank - 1) / 2.0,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="week5_smoke_") as temp_dir:
        root = Path(temp_dir)
        for relative in [
            "run_week5.py",
            "src/strategy/build_strategy_week5.py",
            "src/validation/validate_week5.py",
            "docs/week5_methodology.md",
        ]:
            source = SOURCE_ROOT / relative
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        input_path = root / "results" / "tables" / "week4_selected_model_signals.csv"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        make_signals().to_csv(input_path, index=False)

        result = subprocess.run(
            [sys.executable, str(root / "run_week5.py")],
            cwd=root,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr, file=sys.stderr)
            return result.returncode

        # Independent checks for representative edge cases.
        default = pd.read_csv(root / "results" / "tables" / "week5_default_portfolio.csv")
        threshold = pd.read_csv(root / "results" / "tables" / "week5_threshold_weights.csv")

        third_date = str(pd.Timestamp("2025-03-31").date())
        third = default[default["Signal_Date"].astype(str).str[:10] == third_date]
        third_cash = float(third.loc[third["Asset"] == "CASH", "Target_Weight"].iloc[0])
        if abs(third_cash - 1.0) > 1e-12:
            raise AssertionError("Combined strategy should be 100% cash when no sector passes 0.50.")

        fourth_date = str(pd.Timestamp("2025-04-30").date())
        fourth = default[default["Signal_Date"].astype(str).str[:10] == fourth_date]
        xlf_weight = float(fourth.loc[fourth["Asset"] == "XLF", "Target_Weight"].iloc[0])
        cash_weight = float(fourth.loc[fourth["Asset"] == "CASH", "Target_Weight"].iloc[0])
        if abs(xlf_weight - 0.60) > 1e-12 or abs(cash_weight - 0.40) > 1e-12:
            raise AssertionError("Single eligible sector should be capped at 60% with 40% cash.")

        fifth_date = str(pd.Timestamp("2025-05-31").date())
        fifth = threshold[threshold["Signal_Date"].astype(str).str[:10] == fifth_date]
        sector_weights = fifth[fifth["Asset"].isin(["XLF", "XLK", "XLE"])]["Target_Weight"]
        if not ((sector_weights - (1.0 / 3.0)).abs() < 1e-12).all():
            raise AssertionError("Three threshold-qualified sectors should be equal-weighted.")

        print(result.stdout.strip())
        print("[PASS] Edge cases for cash, position caps, and equal weighting passed.")
        print("[PASS] Week 5 smoke test completed in an isolated temporary repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
